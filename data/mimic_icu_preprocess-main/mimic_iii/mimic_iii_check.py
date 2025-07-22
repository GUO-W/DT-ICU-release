from pathlib import Path

root = Path(r"./1.4/processed_icu")          # ← adjust if needed
expected = {"dynamic.csv", "diagnoses.csv", "demo.csv"}

empty = [
    p for p in root.iterdir() if p.is_dir()
    and (set(f.name for f in p.iterdir()) & expected) != expected
]

print(f"Total ICU-stay folders : {sum(1 for _ in root.iterdir() if _.is_dir())}")
print(f"Empty / incomplete     : {len(empty)}")
print("Examples:", [e.name for e in empty[:10]])