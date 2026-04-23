"""LLM semantic evaluation of Leo root cause analysis via OpenRouter."""

from __future__ import annotations

import json
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class LLMEvalResult:
    """Result from LLM semantic evaluation of one kernel/vendor pair."""

    kernel: str
    vendor: str
    match_quality: int = 0  # 0-100
    confidence: str = ""  # Low/Medium/High
    reasoning: str = ""
    error: str = ""


def load_env():
    """Load environment variables from .env file."""
    env_file = Path(__file__).resolve().parent.parent.parent / ".env"
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ[key.strip()] = value.strip()


def build_eval_prompt(r) -> Optional[str]:
    """Build LLM evaluation prompt for one kernel/vendor pair.

    Returns None if the result has no Leo data or no optimization diff.
    """
    if not r.leo_stall_text or not r.diff_text:
        return None

    # Build optional sections
    source_section = ""
    if r.original_source:
        source_section = f"""
## Original Source Code ({r.diff_file}):
```
{r.original_source}
```
"""

    dep_chains_section = ""
    if r.leo_dep_chains:
        dep_chains_section = f"""
## Leo Dependency Chains (multi-hop back-slicing paths):
{r.leo_dep_chains}
"""

    return f"""You are an expert GPU performance analyst evaluating a root cause analysis tool called "Leo".

Leo uses PC sampling + back-slicing dataflow analysis to find root causes of GPU stalls.
Its output has two key columns:
- "Stall Location / Stall Opcode": the SYMPTOM — where the stall is observed (e.g., s_waitcnt, DFMA waiting)
- "Root Cause Location / Root Opcode" (after "<--"): the ROOT CAUSE — the upstream instruction that actually causes the stall, found via backward dataflow slicing

The "Dependency Chains" show multi-hop back-slicing paths: stall ← intermediate ← root cause, revealing the full dataflow that leads to the stall.

Your task: evaluate whether Leo's ROOT CAUSE (not the symptom) correctly explains the fundamental bottleneck that the optimization addresses.

Scoring rubric:
- 90-100: Root cause precisely identifies the bottleneck the optimization fixes (e.g., root cause = scalar global loads, optimization = vectorize loads)
- 70-89: Root cause is related but imprecise (e.g., root cause = memory stall, optimization = specific cache optimization)
- 40-69: Root cause only identifies the symptom, not the underlying cause (e.g., root cause = waitcnt, optimization = restructure data layout)
- 0-39: Root cause is unrelated to what the optimization actually fixes
{source_section}
## Leo Root Cause Analysis (Stall Table):
{r.leo_stall_text}
{dep_chains_section}
## Optimization Diff (original → optimized):
{r.diff_text}

## Task:
Does Leo's identified ROOT CAUSE (the instruction/location after "<--") correctly explain WHY the bottleneck exists, matching the optimization's intent? Consider the dependency chains to understand the full back-slicing path.

Respond in JSON: {{"root_cause_score": <int 0-100>, "confidence": "<Low|Medium|High>", "reasoning": "<explain whether root cause matches the optimization intent, not just the symptom>"}}"""


def _eval_single(
    r, client, model: str, log_file=None, log_lock=None,
    max_retries: int = 5,
) -> LLMEvalResult:
    """Evaluate a single kernel/vendor pair via LLM API."""
    prompt = build_eval_prompt(r)
    if prompt is None:
        return LLMEvalResult(
            kernel=r.kernel, vendor=r.vendor,
            error="skipped (no Leo data or no diff)",
        )

    last_text = ""
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=1500,
            )
            text = (response.choices[0].message.content or "").strip()
        except Exception as e:
            if attempt < max_retries:
                import time
                time.sleep(1)
                continue
            return LLMEvalResult(
                kernel=r.kernel, vendor=r.vendor, error=str(e),
            )

        # Log raw response for debugging
        if log_file is not None and log_lock is not None:
            with log_lock:
                retry_tag = f" (retry {attempt})" if attempt > 1 else ""
                log_file.write(f"\n{'='*80}\n")
                log_file.write(f"KERNEL: {r.kernel}  VENDOR: {r.vendor}{retry_tag}\n")
                log_file.write(f"{'='*80}\n")
                log_file.write(f"RAW RESPONSE:\n{text}\n")
                log_file.flush()

        # Retry on empty or truncated response
        if not text or (text.startswith("{") and not text.endswith("}")):
            last_text = text
            if attempt < max_retries:
                import time
                time.sleep(1)
                continue

        last_text = text
        break

    text = last_text

    try:
        # Extract JSON from response — try multiple strategies
        data = None

        # Strategy 1: direct parse
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            pass

        # Strategy 2: extract from markdown code block
        if data is None and "```" in text:
            for block in text.split("```"):
                block = block.strip()
                if block.startswith("json"):
                    block = block[4:].strip()
                if block.startswith("{"):
                    try:
                        data = json.loads(block)
                        break
                    except json.JSONDecodeError:
                        continue

        # Strategy 3: find first {...} with regex
        if data is None:
            m = re.search(r"\{[^{}]*\}", text)
            if m:
                try:
                    data = json.loads(m.group())
                except json.JSONDecodeError:
                    pass

        # Strategy 4: find outermost {...} allowing nested braces
        if data is None:
            m = re.search(r"\{.*\}", text, re.DOTALL)
            if m:
                try:
                    data = json.loads(m.group())
                except json.JSONDecodeError:
                    pass

        # Strategy 5: extract fields from truncated JSON via regex
        if data is None:
            mq = re.search(r'"root_cause_score"\s*:\s*(\d+)', text)
            cf = re.search(r'"confidence"\s*:\s*"?(High|Medium|Low)', text)
            rs = re.search(r'"reasoning"\s*:\s*"?(.+)', text, re.DOTALL)
            if mq:
                reasoning = ""
                if rs:
                    reasoning = rs.group(1).rstrip('"}').strip()
                    if len(reasoning) > 200:
                        reasoning = reasoning[:197] + "..."
                return LLMEvalResult(
                    kernel=r.kernel, vendor=r.vendor,
                    match_quality=int(mq.group(1)),
                    confidence=cf.group(1) if cf else "",
                    reasoning=reasoning,
                )

        if data is None:
            return LLMEvalResult(
                kernel=r.kernel, vendor=r.vendor,
                reasoning=text[:200] if text else "",
                error="JSON parse failed",
            )

        # Accept both "root_cause_score" and legacy "match_quality"
        score = data.get("root_cause_score", data.get("match_quality", 0))
        return LLMEvalResult(
            kernel=r.kernel,
            vendor=r.vendor,
            match_quality=int(score),
            confidence=str(data.get("confidence", "")),
            reasoning=str(data.get("reasoning", "")),
        )
    except Exception as e:
        return LLMEvalResult(
            kernel=r.kernel, vendor=r.vendor,
            error=str(e),
        )


def run_llm_eval(
    results: list, model: str, concurrency: int,
    log_path: Optional[Path] = None,
) -> list[LLMEvalResult]:
    """Run LLM semantic evaluation for all evaluable results."""
    from openai import OpenAI

    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    # Probe the model slug once up-front; OpenRouter preview slugs (e.g.
    # google/gemini-3.1-pro-preview) can be retired or renamed. Failing
    # early with a pointer to the model list saves a full run of 404s.
    try:
        client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=1,
        )
    except Exception as e:
        msg = str(e).lower()
        if "not a valid model" in msg or "model_not_found" in msg or "404" in msg:
            print(
                f"ERROR: OpenRouter model '{model}' is not available.\n"
                f"  List current models: curl https://openrouter.ai/api/v1/models | jq '.data[].id'\n"
                f"  Pass a current slug via --llm-model (see AE Known Deviations).",
                file=sys.stderr,
            )
            sys.exit(2)
        # Any other error (network, auth) — let the per-kernel loop surface it.

    # Filter to evaluable results (have both Leo data and diff)
    evaluable = [r for r in results if r.leo_stall_text and r.diff_text]
    total = len(evaluable)
    print(f"\nLLM eval: {total} evaluable pairs (model: {model})", file=sys.stderr)
    if log_path:
        print(f"  Log file: {log_path}", file=sys.stderr)

    eval_results: list[LLMEvalResult] = []
    log_file = open(log_path, "w") if log_path else None
    log_lock = threading.Lock()

    try:
        if log_file:
            log_file.write(f"LLM Eval Log — model: {model}, total: {total}\n")

        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            future_to_result = {
                executor.submit(_eval_single, r, client, model, log_file, log_lock): r
                for r in evaluable
            }

            for i, future in enumerate(as_completed(future_to_result), 1):
                vr = future_to_result[future]
                er = future.result()
                eval_results.append(er)

                if er.error and "skipped" not in er.error:
                    print(f"  [{i}/{total}] {vr.kernel}/{vr.vendor}: ERROR {er.error}",
                          file=sys.stderr)
                else:
                    print(f"  [{i}/{total}] {vr.kernel}/{vr.vendor}: {er.match_quality}/100 "
                          f"({er.confidence})", file=sys.stderr)
    finally:
        if log_file:
            log_file.close()

    # Sort by kernel/vendor for consistent output
    eval_results.sort(key=lambda x: (x.kernel, x.vendor))
    return eval_results


def format_llm_report(eval_results: list[LLMEvalResult], model: str) -> str:
    """Format human-readable LLM evaluation report."""
    w = 120
    lines = []
    lines.append("")
    lines.append("=" * w)
    lines.append(f"LLM SEMANTIC EVALUATION ({model})".center(w))
    lines.append("=" * w)
    lines.append("")

    # Table header
    lines.append(f"{'Kernel':<25} {'Vendor':<8} {'Score':>5} {'Confidence':<12} {'Reasoning'}")
    lines.append("-" * w)

    scored = [er for er in eval_results if not er.error]
    errors = [er for er in eval_results if er.error]

    for er in scored:
        reasoning = er.reasoning[:65] + "..." if len(er.reasoning) > 68 else er.reasoning
        lines.append(
            f"{er.kernel:<25} {er.vendor:<8} {er.match_quality:>5} {er.confidence:<12} {reasoning}"
        )

    if errors:
        lines.append("")
        lines.append("Errors/Skipped:")
        for er in errors:
            lines.append(f"  {er.kernel:<25} {er.vendor:<8} {er.error}")

    lines.append("-" * w)

    # Summary statistics
    if scored:
        avg_score = sum(er.match_quality for er in scored) / len(scored)
        high_conf = sum(1 for er in scored if er.confidence == "High")
        lines.append(f"Average score: {avg_score:.1f}/100  |  "
                      f"High confidence: {high_conf}/{len(scored)} ({100*high_conf/len(scored):.0f}%)  |  "
                      f"Evaluated: {len(scored)}/{len(eval_results)}")
    else:
        lines.append("No results scored.")

    lines.append("")
    return "\n".join(lines)
