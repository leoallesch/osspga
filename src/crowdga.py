# crowdga_robust.py
# CrowdGA v3.1 — Robust Anti–Local-Minima Edition
# Adds adaptive mutation, delayed adjacency seeding, diverse elites, and restart strategy.

import random
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional


# ====================== DATA MODEL ======================
@dataclass
class OSSPInstance:
    n: int
    m: int
    p: List[List[int]]
    name: str = "unknown"

    def __post_init__(self):
        """Precompute simple lower bound for reference."""
        job_sums = [sum(row) for row in self.p]
        mach_sums = [sum(self.p[i][j] for i in range(self.n)) for j in range(self.m)]
        max_op = max(max(row) for row in self.p) if self.p else 0
        self.LB = max(max(job_sums, default=0), max(mach_sums, default=0), max_op)


# ====================== EVALUATION ======================
def evaluate_makespan(schedule: List[List[int]], inst: OSSPInstance) -> int:
    """Standard active schedule decoder for open shop scheduling."""
    job_t = [0] * inst.n
    mach_t = [0] * inst.m
    for j in range(inst.n):
        for m in schedule[j]:
            if inst.p[j][m] == 0:
                continue
            start = max(job_t[j], mach_t[m])
            end = start + inst.p[j][m]
            job_t[j] = end
            mach_t[m] = end
    return max(job_t + mach_t)


# ====================== GENETIC OPERATORS ======================
def create_random_individual(inst: OSSPInstance) -> List[List[int]]:
    """Create a random valid individual schedule."""
    ind = []
    for j in range(inst.n):
        machines = [m for m in range(inst.m) if inst.p[j][m] > 0]
        random.shuffle(machines)
        ind.append(machines)
    return ind


def crossover(p1: List[List[int]], p2: List[List[int]]) -> Tuple[List[List[int]], List[List[int]]]:
    """Partial order crossover per job."""
    c1, c2 = [row[:] for row in p1], [row[:] for row in p2]
    for j in range(len(p1)):
        if random.random() < 0.9 and len(p1[j]) > 1:
            a, b = sorted(random.sample(range(len(p1[j])), 2))
            seg1, seg2 = p1[j][a:b], p2[j][a:b]
            rest1 = [x for x in p2[j] if x not in seg1]
            rest2 = [x for x in p1[j] if x not in seg2]
            c1[j] = rest1[:a] + seg1 + rest1[a:]
            c2[j] = rest2[:a] + seg2 + rest2[a:]
    return c1, c2


def mutate(ind: List[List[int]], rate: float = 0.4):
    """Swap-mutation within job sequence."""
    for row in ind:
        if random.random() < rate and len(row) > 1:
            i, j = random.sample(range(len(row)), 2)
            row[i], row[j] = row[j], row[i]


# ====================== WISDOM OF CROWDS ======================
def build_adjacency_table(elites: List[List[List[int]]], inst: OSSPInstance) -> List[List[List[int]]]:
    """Adjacency frequency: adj[j][a][b] = how often machine b follows a for job j."""
    adj = [[[0 for _ in range(inst.m)] for _ in range(inst.m)] for _ in range(inst.n)]
    for elite in elites:
        for j in range(inst.n):
            seq = elite[j]
            for k in range(len(seq) - 1):
                a, b = seq[k], seq[k + 1]
                adj[j][a][b] += 1
    return adj


def create_individual_from_adjacency(inst: OSSPInstance, adj: List[List[List[int]]]) -> List[List[int]]:
    """Generate individual guided by adjacency table."""
    individual = []
    for j in range(inst.n):
        machines = [m for m in range(inst.m) if inst.p[j][m] > 0]
        if len(machines) <= 1:
            individual.append(machines)
            continue

        seq = []
        remaining = set(machines)
        current = random.choice(machines)
        seq.append(current)
        remaining.remove(current)

        while remaining:
            next_candidates = [(m, adj[j][current][m]) for m in remaining]
            total_weight = sum(w for _, w in next_candidates)
            if total_weight == 0:
                current = random.choice(list(remaining))
            else:
                r = random.uniform(0, total_weight)
                cum = 0
                for m, w in next_candidates:
                    cum += w
                    if r <= cum:
                        current = m
                        break
            seq.append(current)
            remaining.remove(current)
        individual.append(seq)
    return individual


# ====================== SELECTION ======================
def tournament_select(pop, fitnesses, k=5, prob_best=0.8):
    """Soft tournament selection — occasionally picks a non-best to preserve diversity."""
    group = random.sample(list(zip(pop, fitnesses)), k)
    group.sort(key=lambda x: x[1])
    if random.random() < prob_best:
        return [row[:] for row in group[0][0]]
    else:
        return [row[:] for row in random.choice(group[1:])[0]]


# ====================== MAIN CROWDGA LOOP ======================
def crowdga(
    inst: OSSPInstance,
    pop_size: int = 200,
    max_seconds: float = 30.0,
    target_gap: float = 0.0,
    seed: Optional[int] = None,
) -> Tuple[int, List[List[int]], float]:
    """
    CrowdGA v3.1 - Robust Anti–Local-Minima Version
    Adaptive mutation, delayed wisdom-of-crowds, elite diversity, and restarts.
    """
    if seed is not None:
        random.seed(seed)

    population = [create_random_individual(inst) for _ in range(pop_size)]
    best_sched, best_ms = None, float("inf")
    elites: List[List[List[int]]] = []

    start_time = time.time()
    generation = 0

    while time.time() - start_time < max_seconds:
        generation += 1

        # Evaluate population
        makespans = [evaluate_makespan(ind, inst) for ind in population]
        min_ms = min(makespans)

        # Update best
        if min_ms < best_ms:
            best_ms = min_ms
            best_sched = [row[:] for row in population[makespans.index(min_ms)]]
            elites.append(best_sched)
            gap = (best_ms - inst.LB) / inst.LB * 100
            print(f"Gen {generation:4d} | Best: {best_ms:4d} | LB: {inst.LB:3d} | Gap: {gap:5.2f}%")
            if gap <= target_gap:
                print(f"TARGET GAP {target_gap}% REACHED!")
                break

        # Maintain top 30% elites
        elite_count = max(10, pop_size // 3)
        elites = sorted(elites, key=lambda e: evaluate_makespan(e, inst))[:elite_count]

        # Delayed wisdom seeding
        use_wisdom = generation >= 5
        adj_table = build_adjacency_table(elites, inst) if use_wisdom else None

        # Adaptive mutation rate
        mut_rate = 0.7 if generation < 10 else 0.4

        # Tournament selection
        survivors = [tournament_select(population, makespans) for _ in range(pop_size)]

        # === Generate New Population ===
        new_pop = []

        # 1. Elitism
        new_pop.append([row[:] for row in best_sched])

        # 2. Wisdom-guided individuals
        if adj_table and use_wisdom:
            for _ in range(pop_size // 5):
                new_pop.append(create_individual_from_adjacency(inst, adj_table))

        # 3. Crossover + Mutation
        while len(new_pop) < pop_size:
            p1, p2 = random.sample(survivors, 2)
            c1, c2 = crossover(p1, p2)
            mutate(c1, rate=mut_rate)
            mutate(c2, rate=mut_rate)
            new_pop.extend([c1, c2])

        # 4. Periodic random restarts
        if generation % 10 == 0:
            for _ in range(pop_size // 10):
                new_pop[random.randint(0, pop_size - 1)] = create_random_individual(inst)

        # Ensure correct population size
        population = new_pop[:pop_size]

    elapsed = time.time() - start_time
    return best_ms, best_sched, elapsed
