#!/usr/bin/env bash
# Quick check of load averages across all lab servers.
# Usage: ./scripts/check_loads.sh

set -uo pipefail

HOSTS=(headroom illyad gilgamesh odyssey voltar athena hopper1 hopper2)

printf "%-12s  %-20s  %6s  %10s  %10s  %10s  %-24s  %-6s  %s\n" \
       "HOST" "UPTIME" "USERS" "LOAD(1m)" "LOAD(5m)" "LOAD(15m)" "GPU" "HEALTH" "GPU MEM"
printf "%s\n" "----------------------------------------------------------------------------------------------------------------------"

for host in "${HOSTS[@]}"; do
    result=$(ssh -o ConnectTimeout=5 -o BatchMode=yes "$host" \
        'cat /proc/loadavg; uptime' 2>/dev/null)

    if [ $? -ne 0 ]; then
        printf "%-12s  %-20s\n" "$host" "UNREACHABLE"
        continue
    fi

    loadavg=$(echo "$result" | head -1)
    uptime_line=$(echo "$result" | tail -1)

    load1=$(echo "$loadavg" | awk '{print $1}')
    load5=$(echo "$loadavg" | awk '{print $2}')
    load15=$(echo "$loadavg" | awk '{print $3}')

    updays=$(echo "$uptime_line" | grep -oP '\d+ days?' || echo "< 1 day")
    users=$(echo "$uptime_line" | grep -oP '\d+ users?' || echo "0 users")

    # Check GPU model, health, and memory usage
    gpu_info=$(ssh -o ConnectTimeout=5 -o BatchMode=yes "$host" \
        'if command -v nvidia-smi &>/dev/null; then
            # Get all GPU names, clean up, group by model with counts
            raw=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null \
                | sed "s/NVIDIA //" | sed "s/ PCIe//" | sed "s/-SXM4-40GB//" \
                | sed "s/ 480GB//" | sed "s/ Server Edition//" | sed "s/ Blackwell//" \
                | sed "s/RTX PRO /RTX-PRO-/" | sed "s/Tesla //" | sed "s/ 80GB//" \
                | sed "s/-PCIE-16GB//" | sed "s/-PCIE//")
            model=$(echo "$raw" | sort | uniq -c | sort -rn \
                | awk "{if (\$1>1) print \$1\"x \"substr(\$0, index(\$0,\$2)); else print substr(\$0, index(\$0,\$2))}" \
                | paste -sd "+" -)
            if nvidia-smi -q 2>&1 | grep -q "not in ready state"; then
                health="RESET"
            else
                health="OK"
            fi
            # GPU memory: sum used and total across all GPUs
            mem=$(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null \
                | awk -F", " "{u+=\$1; t+=\$2} END{printf \"%dG/%dG (%d%%)\", u/1024, t/1024, (t>0?u*100/t:0)}")
            echo "${model}|${health}|${mem}"
        elif command -v rocm-smi &>/dev/null; then
            model=$(rocm-smi --showproductname 2>/dev/null | grep -oP "MI\d+\w*" | head -1)
            count=$(rocm-smi --showproductname 2>/dev/null | grep -c "MI[0-9]")
            [ -z "$model" ] && model="AMD GPU"
            if [ "$count" -gt 1 ]; then
                model="${count}x ${model}"
            fi
            # AMD GPU memory: parse "VRAM Total Memory (B): NNN" and "VRAM Total Used Memory (B): NNN"
            mem=$(rocm-smi --showmeminfo vram 2>/dev/null \
                | awk "/VRAM Total Memory \(B\)/{t+=\$NF} /VRAM Total Used Memory \(B\)/{u+=\$NF} END{if(t>0) printf \"%dG/%dG (%d%%)\", u/1073741824, t/1073741824, u*100/t; else print \"?\"}")
            [ -z "$mem" ] && mem="?"
            echo "${model}|OK|${mem}"
        elif command -v xpu-smi &>/dev/null; then
            # Intel GPU: get friendly name from lspci or xpu-smi, with PCI ID fallback mapping
            imodel=$(lspci 2>/dev/null | grep -i "accelerator\|display" | grep -v ASPEED | grep -ioP "Max \d+|Arc \w+|Flex \d+|Ponte Vecchio|Data Center GPU \w+" | head -1)
            if [ -z "$imodel" ]; then
                raw=$(xpu-smi discovery 2>/dev/null | grep -oP "Device Name: \K.*" | head -1 | sed "s/Intel Corporation //" | sed "s/ (rev.*)//" | xargs)
                # Map known PCI device IDs to friendly names
                case "$raw" in
                    *0bda*|*0bd5*|*0bd6*) imodel="Max 1100 (PVC)" ;;
                    *0bd0*|*0bd4*)        imodel="Max 1550 (PVC)" ;;
                    *) imodel="${raw:-Intel GPU}" ;;
                esac
            fi
            [ -z "$imodel" ] && imodel="Intel GPU"
            # Intel GPU memory: used from stats, total estimated from used + util%
            used_mib=$(xpu-smi stats -d 0 2>/dev/null | grep "GPU Memory Used" | grep -oP "\d+" | tail -1)
            util_pct=$(xpu-smi stats -d 0 2>/dev/null | grep "GPU Memory Util" | grep -oP "\d+" | tail -1)
            if [ -n "$used_mib" ] && [ -n "$util_pct" ] && [ "$util_pct" -gt 0 ] 2>/dev/null; then
                total_mib=$((used_mib * 100 / util_pct))
                mem=$(awk "BEGIN{printf \"%dG/%dG (%d%%)\", ${used_mib}/1024, ${total_mib}/1024, ${util_pct}}")
            elif [ -n "$used_mib" ]; then
                mem="${used_mib}MiB used"
            else
                mem="?"
            fi
            echo "${imodel}|OK|${mem}"
        elif test -d /dev/dri/by-path; then
            imodel=$(lspci 2>/dev/null | grep -i "accelerator\|display\|vga" | grep -ioP "Max \d+|Arc \w+|Flex \d+|Ponte Vecchio" | head -1)
            [ -z "$imodel" ] && imodel="Intel GPU"
            echo "${imodel}|OK|?"
        else
            echo "-|-|-"
        fi' 2>/dev/null || echo "?|?|?")

    gpu_model=$(echo "$gpu_info" | cut -d'|' -f1)
    gpu_status=$(echo "$gpu_info" | cut -d'|' -f2)
    gpu_mem=$(echo "$gpu_info" | cut -d'|' -f3)

    printf "%-12s  %-20s  %6s  %10s  %10s  %10s  %-24s  %-6s  %s\n" \
           "$host" "$updays" "$users" "$load1" "$load5" "$load15" "$gpu_model" "$gpu_status" "$gpu_mem"
done
