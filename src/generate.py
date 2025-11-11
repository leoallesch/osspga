import random
from pathlib import Path

def generate(n, m, min_p=1, max_p=99, density=1.0, seed=None, name=None):
    """Generate one OSSP instance"""
    if seed is not None:
        random.seed(seed)

    p = []
    for j in range(n):
        row = []
        for k in range(m):
            if random.random() < density:
                row.append(random.randint(min_p, max_p))
            else:
                row.append(0)
        p.append(row)

    if name is None:
        dens = "dense" if density >= 0.95 else f"{int(density*100)}pct"
        seed_part = f"_s{seed}" if seed is not None else ""
        name = f"ossp_{n}x{m}_{dens}{seed_part}.txt"

    path = Path("ossp_instances") / name
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
    print(f"   {n} jobs × {m} machines | LB = {lb} | density {density:.2f}")
    print()


# ====================== PRESET INSTANCES ======================
def make_all():
    print("OSSP Instance Generator")
    print("=" * 55)
    print("Generating 8 high-quality test instances...\n")

    generate(3,  3,  max_p=30,  density=1.0, seed=5,   name="tiny_3x3.txt")
    generate(5,  5,  max_p=50,  density=1.0, seed=42,  name="small_5x5.txt")
    generate(7,  7,  max_p=60,  density=1.0, seed=7,   name="square_7x7.txt")
    generate(10, 8,  max_p=99,  density=1.0, seed=123, name="medium_10x8.txt")
    generate(12, 6,  max_p=95,  density=0.85, seed=42, name="wide_12x6.txt")
    generate(6,  12, max_p=95,  density=0.85, seed=42, name="tall_6x12.txt")
    generate(15, 15, max_p=80,  density=0.6,  seed=1,   name="sparse_15x15.txt")
    generate(20, 10, max_p=99,  density=0.9,  seed=999, name="large_20x10.txt")

    print("All done!")
    print("Open your GA GUI → Load any file from: ossp_instances/")
    print("Ready to test!")


# ====================== RUN ======================
if __name__ == "__main__":
    make_all()  # Always generate the full set