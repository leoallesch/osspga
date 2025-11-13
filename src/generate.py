import random
from pathlib import Path

def generate(n, m, min_p=1, max_p=99, seed=None, name=None):
    """Generate one OSSP instance"""
    if seed is not None:
        random.seed(seed)

    p = []
    for j in range(n):
        row = []
        for k in range(m):
            row.append(random.randint(min_p, max_p))
        p.append(row)

    if name is None:
        seed_part = f"_s{seed}" if seed is not None else ""
        name = f"ossp_{n}x{m}{seed_part}.txt"

    path = Path("data") / name
    path.parent.mkdir(exist_ok=True)

    with open(path, "w") as f:
        f.write(f"{n} {m}\n")
        for row in p:
            f.write(" ".join(map(str, row)) + "\n")

    lb = max(
        max(sum(row) for row in p),
        max(sum(p[j][k] for j in range(n)) for k in range(m))
    )
    print(f"Generated → {path.name}")
    print(f"   {n} jobs × {m} machines | LB = {lb}")
    print()


# ====================== PRESET INSTANCES ======================
def make_all():
    print("OSSP Data Generator")
    print("=" * 55)
    print("Generating 8 high-quality test instances...\n")

    generate(3,  3,  max_p=30, seed=5)
    generate(5,  5,  max_p=30, seed=42)
    generate(7,  7,  max_p=30, seed=7)
    generate(10, 8,  max_p=30, seed=123)
    generate(12, 6,  max_p=30, seed=42)
    generate(6,  12, max_p=30, seed=42)
    generate(15, 15, max_p=30, seed=1)
    generate(20, 10, max_p=30, seed=999)


# ====================== RUN ======================
if __name__ == "__main__":
    make_all()  # Always generate the full set