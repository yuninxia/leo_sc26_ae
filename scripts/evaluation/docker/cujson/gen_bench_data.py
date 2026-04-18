#!/usr/bin/env python3
"""Generate a large synthetic JSON dataset for cuJSON profiling."""
import json, random, string, sys

random.seed(42)
n = int(sys.argv[1]) if len(sys.argv) > 1 else 50000
out = sys.argv[2] if len(sys.argv) > 2 else "bench_large.json"

records = []
for i in range(n):
    records.append({
        "id": i,
        "name": "".join(random.choices(string.ascii_letters, k=20)),
        "email": "".join(random.choices(string.ascii_lowercase, k=10)) + "@example.com",
        "age": random.randint(18, 80),
        "score": round(random.uniform(0, 100), 4),
        "active": random.choice([True, False]),
        "tags": ["".join(random.choices(string.ascii_lowercase, k=8))
                 for _ in range(random.randint(1, 5))],
        "address": {
            "street": "".join(random.choices(string.ascii_letters + " ", k=30)),
            "city": "".join(random.choices(string.ascii_uppercase, k=10)),
            "zip": "".join(random.choices(string.digits, k=5))
        },
        "bio": "".join(random.choices(string.ascii_letters + " ",
                                       k=random.randint(100, 500)))
    })

with open(out, "w") as f:
    json.dump(records, f)
print(f"Generated {n} records -> {out}")
