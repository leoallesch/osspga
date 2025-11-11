# file: ga_simple.py
import random
from typing import List, Tuple
from dataclasses import dataclass
import time

@dataclass
class OSSPInstance:
    n: int
    m: int
    p: List[List[int]]
    name: str = "unknown"
    
    def __post_init__(self):
        self.per_job = [[] for _ in range(self.n)]
        for j in range(self.m):
            for i in range(self.n):
                if self.p[i][j] > 0:
                    self.per_job[i].append(j)
        
        self.job_total = [sum(row) for row in self.p]
        self.mach_total = [sum(self.p[i][j] for i in range(self.n)) for j in range(self.m)]
        self.LB = max(
            max(self.job_total),
            max(self.mach_total),
            max(max(row) for row in self.p)
        )

# ===============================================================
# SINGLE ELITE DECODER: Active Schedule (Best in Practice)
# ===============================================================

def decode_active(chrom: List[List[int]], inst: OSSPInstance) -> int:
    """Fast active schedule — push every operation as early as possible"""
    job_time = [0] * inst.n
    mach_time = [0] * inst.m
    
    for job in range(inst.n):
        for mach in chrom[job]:
            if inst.p[job][mach] == 0:
                continue
            dur = inst.p[job][mach]
            start = max(job_time[job], mach_time[mach])
            job_time[job] = start + dur
            mach_time[mach] = start + dur
    
    return max(max(job_time), max(mach_time))

# ===============================================================
# GENETIC ALGORITHM (Clean & Fast)
# ===============================================================

def create_individual(inst: OSSPInstance) -> List[List[int]]:
    chrom = []
    for i in range(inst.n):
        machines = [j for j in range(inst.m) if inst.p[i][j] > 0]
        random.shuffle(machines)
        chrom.append(machines)
    return chrom

def swap_mutation(chrom: List[List[int]], pm: float = 0.4):
    for row in chrom:
        if random.random() < pm and len(row) > 1:
            i, j = random.sample(range(len(row)), 2)
            row[i], row[j] = row[j], row[i]

def order_crossover(p1: List[int], p2: List[int]) -> List[int]:
    if len(p1) < 2:
        return p1[:]
    a, b = sorted(random.sample(range(len(p1)), 2))
    child = p1[a:b]
    remaining = [x for x in p2 if x not in child]
    return remaining[:a] + child + remaining[a:]

def crossover(parent1: List[List[int]], parent2: List[List[int]]) -> Tuple[List[List[int]], List[List[int]]]:
    c1, c2 = [], []
    for i in range(len(parent1)):
        if random.random() < 0.9:
            cc1 = order_crossover(parent1[i], parent2[i])
            cc2 = order_crossover(parent2[i], parent1[i])
        else:
            cc1, cc2 = parent1[i][:], parent2[i][:]
        c1.append(cc1)
        c2.append(cc2)
    return c1, c2

# ===============================================================
# MAIN GA (No WoC — Just Power)
# ===============================================================

def ga_ossp(
    inst: OSSPInstance,
    pop_size: int = 300,
    max_gen: int = 2000,
    timeout: float = 30.0,
    seed: int = 42
) -> Tuple[int, List[List[int]]]:
    random.seed(seed)
    population = [create_individual(inst) for _ in range(pop_size)]
    best_makespan = float('inf')
    best_chrom = None
    start_time = time.time()

    for gen in range(max_gen):
        if time.time() - start_time > timeout:
            print(f"Timeout at gen {gen}")
            break

        # Evaluate
        fitness = [decode_active(chrom, inst) for chrom in population]
        curr_best = min(fitness)
        if curr_best < best_makespan:
            best_makespan = curr_best
            best_chrom = population[fitness.index(curr_best)].copy()

        if gen % 100 == 0 or gen == max_gen - 1:
            gap = (best_makespan - inst.LB) / inst.LB * 100
            print(f"Gen {gen:4d} → Best: {best_makespan:4d} | LB: {inst.LB} | Gap: {gap:5.2f}%")

        # Tournament selection
        new_pop = []
        for _ in range(pop_size):
            tournament = random.sample(list(zip(population, fitness)), 5)
            winner = min(tournament, key=lambda x: x[1])[0]
            new_pop.append([row[:] for row in winner])

        # Breed
        offspring = []
        for i in range(0, pop_size, 2):
            p1 = new_pop[i]
            p2 = new_pop[(i + 1) % pop_size]
            c1, c2 = crossover(p1, p2)
            swap_mutation(c1, pm=0.5)
            swap_mutation(c2, pm=0.5)
            offspring.extend([c1, c2])

        population = offspring[:pop_size]

    return best_makespan, best_chrom