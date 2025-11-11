import random
import time
from dataclasses import dataclass
from typing import List
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
from datetime import datetime


# ====================== INSTANCE ======================
@dataclass
class OSSPInstance:
    n: int
    m: int
    p: List[List[int]]
    name: str = "unknown"

    def __post_init__(self):
        job_sums = [sum(row) for row in self.p]
        mach_sums = [sum(self.p[j][i] for j in range(self.n)) for i in range(self.m)]
        self.LB = max(max(job_sums, default=0), max(mach_sums, default=0))


# ====================== NON-DELAY DECODER ======================
def evaluate_makespan(schedule: List[List[int]], inst: OSSPInstance) -> int:
    job_t = [0] * inst.n
    mach_t = [0] * inst.m
    for j in range(inst.n):
        for m in schedule[j]:
            if inst.p[j][m] == 0:
                continue
            start = max(job_t[j], mach_t[m])
            end = start + inst.p[j][m]
            job_t[j] = mach_t[m] = end
    return max(job_t + mach_t)


# ====================== LOCAL SEARCH ======================
def local_search(chrom, inst, max_iters=2000):
    best_ms = evaluate_makespan(chrom, inst)
    improved = True
    iters = 0
    while improved and iters < max_iters:
        improved = False
        for j in range(inst.n):
            seq = chrom[j]
            if len(seq) < 2:
                continue
            for i in range(len(seq)):
                for k in range(i + 1, len(seq)):
                    if iters >= max_iters:
                        return best_ms
                    seq[i], seq[k] = seq[k], seq[i]
                    new_ms = evaluate_makespan(chrom, inst)
                    if new_ms < best_ms:
                        best_ms = new_ms
                        improved = True
                    else:
                        seq[i], seq[k] = seq[k], seq[i]
                    iters += 1
            for i in range(len(seq)):
                for k in range(len(seq)):
                    if i == k or abs(i - k) <= 1:
                        continue
                    if iters >= max_iters:
                        return best_ms
                    op = seq.pop(i)
                    seq.insert(k if k < i else k - 1, op)
                    new_ms = evaluate_makespan(chrom, inst)
                    if new_ms < best_ms:
                        best_ms = new_ms
                        improved = True
                    else:
                        op = seq.pop(k if k < i else k - 1)
                        seq.insert(i, op)
                    iters += 1
    return best_ms


# ====================== GENETIC ALGORITHM ======================
def genetic_algorithm(inst,
                      pop_size=200,
                      max_stagnant_gens=200,
                      max_stagnant_secs=15.0,
                      hard_timeout=120.0):
    population = [create_individual(inst) for _ in range(pop_size)]
    best_chrom = create_individual(inst)
    best_ms = evaluate_makespan(best_chrom, inst)
    history = [best_ms]
    adjacency = [[0] * inst.m for _ in range(inst.m)]
    start_time = time.time()
    last_improve_time = start_time
    last_improve_gen = 0
    gen = 0

    while True:
        gen += 1
        now = time.time()
        elapsed = now - start_time
        stagnant_gens = gen - last_improve_gen
        stagnant_secs = now - last_improve_time

        if best_ms <= inst.LB:
            print(f"OPTIMAL FOUND! Gen {gen}, {elapsed:.1f}s")
            break
        if stagnant_gens >= max_stagnant_gens:
            print(f"Stopped: no improvement in {stagnant_gens} generations")
            break
        if stagnant_secs >= max_stagnant_secs:
            print(f"Stopped: no improvement for {stagnant_secs:.1f}s")
            break
        if elapsed >= hard_timeout:
            print(f"Hard timeout reached ({hard_timeout}s)")
            break

        makespans = [evaluate_makespan(ch, inst) for ch in population]
        min_ms = min(makespans)

        improved = False
        if min_ms < best_ms:
            best_ms = min_ms
            best_idx = makespans.index(min_ms)
            best_chrom = [row[:] for row in population[best_idx]]
            adjacency = build_adjacency(best_chrom, inst)
            improved_ms = local_search(best_chrom, inst, max_iters=2000)
            if improved_ms < best_ms:
                best_ms = improved_ms
            history.append(best_ms)
            last_improve_time = now
            last_improve_gen = gen
            improved = True
        else:
            history.append(best_ms)

        if stagnant_gens > 25 and random.random() < 0.6:
            population = [create_individual(inst) for _ in range(int(pop_size * 0.75))] + [best_chrom] * (pop_size // 4)
            random.shuffle(population)

        survivors = [
            min(random.sample(list(zip(population, makespans)), 5), key=lambda x: x[1])[0]
            for _ in range(pop_size)
        ]
        offspring = [best_chrom]
        for _ in range(pop_size // 2):
            p1, p2 = random.sample(survivors, 2)
            c1, c2 = crossover([r[:] for r in p1], [r[:] for r in p2])
            mutate(c1); mutate(c2)
            offspring.extend([c1, c2])

        for chrom in offspring[1:]:
            if random.random() < 0.35:
                for seq in chrom:
                    if len(seq) > 2 and random.random() < 0.6:
                        i = random.randrange(len(seq) - 1)
                        if adjacency[seq[i]][seq[i + 1]] < 2:
                            seq[i], seq[i + 1] = seq[i + 1], seq[i]

        population = offspring[:pop_size]

    print("Final local search...")
    final_ms = local_search(best_chrom, inst, max_iters=10000)
    if final_ms < best_ms:
        best_ms = final_ms
        print(f"Final polish → {best_ms}")

    return best_ms, best_chrom, history


# ====================== SUPPORT FUNCTIONS ======================
def create_individual(inst):
    ind = []
    for j in range(inst.n):
        ops = [m for m in range(inst.m) if inst.p[j][m] > 0]
        random.shuffle(ops)
        ind.append(ops)
    return ind


def crossover(p1, p2):
    c1, c2 = [r[:] for r in p1], [r[:] for r in p2]
    for j in range(len(p1)):
        if random.random() < 0.9 and len(p1[j]) > 1:
            a, b = sorted(random.sample(range(len(p1[j])), 2))
            seg1, seg2 = p1[j][a:b], p2[j][a:b]
            rest1 = [x for x in p2[j] if x not in seg1]
            rest2 = [x for x in p1[j] if x not in seg2]
            c1[j] = rest1[:a] + seg1 + rest1[a:]
            c2[j] = rest2[:a] + seg2 + rest2[a:]
    return c1, c2


def mutate(chrom, rate=0.4):
    for row in chrom:
        if random.random() < rate and len(row) > 1:
            i, j = random.sample(range(len(row)), 2)
            row[i], row[j] = row[j], row[i]


def build_adjacency(best_chrom, inst):
    adj = [[0] * inst.m for _ in range(inst.m)]
    for seq in best_chrom:
        for a, b in zip(seq[:-1], seq[1:]):
            adj[a][b] += 1
    return adj


# ====================== GANTT CHART WITH INSTANCE NAME ======================
def plot_gantt(ax, schedule, inst):
    ax.clear()
    job_time = [0] * inst.n
    mach_time = [0] * inst.m
    colors = plt.cm.tab20.colors

    for j in range(inst.n):
        for m in schedule[j]:
            if inst.p[j][m] == 0:
                continue
            start = max(job_time[j], mach_time[m])
            end = start + inst.p[j][m]
            ax.barh(m, end - start, left=start, height=0.7,
                    color=colors[j % len(colors)], edgecolor='black')
            ax.text(start + (end - start) / 2, m, f'J{j}', va='center', ha='center',
                    fontweight='bold', color='white', fontsize=9)
            job_time[j] = mach_time[m] = end

    makespan = max(job_time + mach_time)
    gap = (makespan - inst.LB) / inst.LB * 100 if inst.LB > 0 else 0
    color = "green" if gap == 0 else "orange" if gap < 5 else "red"

    # INCLUDE INSTANCE NAME IN TITLE
    ax.set_title(f"Instance: {inst.name} | n={inst.n} m={inst.m} | "
                 f"Makespan = {makespan} | LB = {inst.LB} | Gap = {gap:.2f}%",
                 color=color, fontweight="bold", fontsize=11, pad=15)
    
    ax.set_xlabel("Time")
    ax.set_ylabel("Machine")
    ax.set_yticks(range(inst.m))
    ax.set_yticklabels([f"M{m}" for m in range(inst.m)])
    ax.invert_yaxis()
    ax.grid(True, axis='x', alpha=0.3)


# ====================== GUI ======================
class OSSPGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("OSSP Solver – Genetic Algorithm")
        self.geometry("1200x820")
        self.instance = None
        self.best_sched = None
        self.history = None
        self.build_ui()

    def build_ui(self):
        top = ttk.Frame(self, padding=10)
        top.pack(fill='x')

        ttk.Button(top, text="Load Instance", command=self.load_instance).pack(side='left', padx=5)
        self.status = ttk.Label(top, text="No instance loaded", foreground="gray", font=("", 10, "bold"))
        self.status.pack(side='left', padx=20)

        param_frame = ttk.LabelFrame(self, text=" GA Parameters ", padding=10)
        param_frame.pack(fill='x', padx=10, pady=5)

        params = [
            ("Population:", "200", 8),
            ("Stagnant Gens:", "200", 6),
            ("Stagnant Secs:", "15", 6),
            ("Hard Timeout (s):", "120", 6),
        ]

        self.entries = {}
        row = 0
        for label, default, width in params:
            ttk.Label(param_frame, text=label).grid(row=row, column=0, sticky='e', padx=5, pady=2)
            entry = ttk.Entry(param_frame, width=width, justify='center')
            entry.insert(0, default)
            entry.grid(row=row, column=1, padx=5, pady=2)
            self.entries[label] = entry
            row += 1

        btn_frame = ttk.Frame(self)
        btn_frame.pack(fill='x', pady=8)

        self.run_btn = ttk.Button(btn_frame, text="Run GA", command=self.start_ga)
        self.run_btn.pack(side='left', padx=20)

        self.export_btn = ttk.Button(btn_frame, text="Export PNG + TXT", command=self.export_solution, state='disabled')
        self.export_btn.pack(side='left', padx=10)

        self.fig, (self.ax_conv, self.ax_gantt) = plt.subplots(2, 1, figsize=(11, 8), height_ratios=[1, 2])
        self.canvas = FigureCanvasTkAgg(self.fig, self)
        self.canvas.get_tk_widget().pack(fill='both', expand=True, padx=10, pady=10)

        self.ax_gantt.text(0.5, 0.5, "Load instance → Set parameters → Run GA\nStops automatically", 
                           transform=self.ax_gantt.transAxes, ha='center', va='center', fontsize=16, color='darkblue')

    def get_params(self):
        try:
            pop = int(self.entries["Population:"].get())
            if pop < 10: raise ValueError
            max_gens = int(self.entries["Stagnant Gens:"].get())
            if max_gens < 10: raise ValueError
            max_secs = float(self.entries["Stagnant Secs:"].get())
            if max_secs < 1: raise ValueError
            hard_time = float(self.entries["Hard Timeout (s):"].get())
            if hard_time < 10: raise ValueError
            return pop, max_gens, max_secs, hard_time
        except:
            messagebox.showerror("Invalid Input", "Please enter valid numbers:\n"
                                                  "• Population ≥ 10\n"
                                                  "• Stagnant Gens ≥ 10\n"
                                                  "• Stagnant Secs ≥ 1.0\n"
                                                  "• Hard Timeout ≥ 10")
            return None

    def load_instance(self):
        path = filedialog.askopenfilename(initialdir="ossp_instances", filetypes=[("Text files", "*.txt")])
        if not path:
            return
        try:
            with open(path) as f:
                lines = [l.strip() for l in f if l.strip() and not l.startswith('#')]
            n, m = map(int, lines[0].split())
            p = [list(map(int, l.split())) for l in lines[1:1 + n]]
            self.instance = OSSPInstance(n, m, p, os.path.basename(path))
            self.status.config(text=f"Loaded: {self.instance.name} | n={n} m={m} | LB = {self.instance.LB}", foreground="green")
            self.ax_gantt.clear()
            self.ax_gantt.text(0.5, 0.5, f"Ready!\nInstance: {self.instance.name}\nLB = {self.instance.LB}\nClick 'Run GA'", 
                               transform=self.ax_gantt.transAxes, ha='center', va='center', fontsize=14)
            self.canvas.draw()
        except Exception as e:
            messagebox.showerror("Error", f"Invalid file:\n{e}")

    def start_ga(self):
        if not self.instance:
            messagebox.showerror("Error", "Load an instance first!")
            return

        params = self.get_params()
        if not params:
            return
        pop_size, max_stagnant_gens, max_stagnant_secs, hard_timeout = params

        self.run_btn.config(state='disabled')
        self.export_btn.config(state='disabled')
        self.status.config(text="Running GA... (adaptive stopping)", foreground="purple")
        self.ax_conv.clear()
        self.ax_gantt.clear()
        self.ax_gantt.text(0.5, 0.5, "Solving...\nPlease wait", 
                           transform=self.ax_gantt.transAxes, ha='center', va='center', fontsize=16, color='blue')
        self.canvas.draw()

        def run():
            best_ms, best_sched, history = genetic_algorithm(
                self.instance,
                pop_size=pop_size,
                max_stagnant_gens=max_stagnant_gens,
                max_stagnant_secs=max_stagnant_secs,
                hard_timeout=hard_timeout
            )
            self.best_sched = best_sched
            self.history = history
            self.after(0, self.show_results, best_ms)

        threading.Thread(target=run, daemon=True).start()

    def show_results(self, best_ms):
        # CONVERGENCE PLOT WITH INSTANCE NAME
        self.ax_conv.clear()
        self.ax_conv.plot(self.history, 'o-', color='darkgreen', markersize=3, linewidth=2)
        self.ax_conv.set_title(f"Convergence | Instance: {self.instance.name} | "
                              f"Final Makespan: {best_ms} | LB = {self.instance.LB}", 
                              fontweight='bold', fontsize=11)
        self.ax_conv.set_xlabel("Generation")
        self.ax_conv.set_ylabel("Best Makespan")
        self.ax_conv.grid(True, alpha=0.3)

        # GANTT CHART (already includes name via plot_gantt)
        plot_gantt(self.ax_gantt, self.best_sched, self.instance)

        self.fig.tight_layout()
        self.canvas.draw()

        gap = (best_ms - self.instance.LB) / self.instance.LB * 100
        color = "darkgreen" if gap == 0 else "orange"
        self.status.config(text=f"Done! {self.instance.name} | Makespan: {best_ms} | Gap: {gap:.2f}%", foreground=color)
        self.run_btn.config(state='normal')
        self.export_btn.config(state='normal')

        if gap == 0:
            messagebox.showinfo("OPTIMAL!", f"Instance: {self.instance.name}\n"
                                          f"Lower bound achieved!\nMakespan = {best_ms}")

    def export_solution(self):
        if not self.best_sched:
            return
        folder = filedialog.askdirectory(title="Select Export Folder")
        if not folder:
            return

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        timestamp_file = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = os.path.splitext(self.instance.name)[0]
        final_ms = self.history[-1]
        filename = f"{base_name}_ms{final_ms}_{timestamp_file}"

        png_path = os.path.join(folder, f"{filename}.png")
        txt_path = os.path.join(folder, f"{filename}_solution.txt")
        csv_path = os.path.join(folder, f"{filename}_schedule.csv")

        # Save PNG
        self.fig.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')

        # Save full timed schedule (CSV)
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            import csv
            writer = csv.writer(f)
            writer.writerow(["Job", "Machine", "Processing_Time", "Start_Time", "End_Time"])
            job_time = [0] * self.instance.n
            mach_time = [0] * self.instance.m
            for j in range(self.instance.n):
                for m in self.best_sched[j]:
                    if self.instance.p[j][m] == 0:
                        continue
                    p_time = self.instance.p[j][m]
                    start = max(job_time[j], mach_time[m])
                    end = start + p_time
                    writer.writerow([j, m, p_time, start, end])
                    job_time[j] = end
                    mach_time[m] = end

        # Save professional TXT report
        gap = (final_ms - self.instance.LB) / self.instance.LB * 100
        status = "OPTIMAL" if gap == 0 else f"HEURISTIC (+{gap:.3f}%)"

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("# OSSP Solution Report\n")
            f.write(f"Instance: {self.instance.name}\n")
            f.write(f"Jobs: {self.instance.n} | Machines: {self.instance.m}\n")
            f.write(f"Lower Bound: {self.instance.LB}\n")
            f.write(f"Makespan: {final_ms} ({status})\n")
            f.write(f"Gap: {gap:.4f}%\n")
            f.write(f"Solved: {timestamp}\n")
            f.write(f"Algorithm: Genetic Algorithm + Local Search\n\n")

            f.write("=== JOB SEQUENCES (Complete Solution) ===\n")
            for j, seq in enumerate(self.best_sched):
                seq_str = " -> ".join(map(str, seq))
                f.write(f"Job {j:2d}: {seq_str}\n")
            
            f.write("\n=== FILES EXPORTED ===\n")
            f.write(f"- {os.path.basename(png_path)}\n")
            f.write(f"- {os.path.basename(txt_path)}\n")
            f.write(f"- {os.path.basename(csv_path)}  ← Start/end times\n")
            f.write("\n# End of solution\n")

        messagebox.showinfo("Exported!", 
                            f"Professional solution exported!\n\n"
                            f"• Gantt: {os.path.basename(png_path)}\n"
                            f"• Solution: {os.path.basename(txt_path)}\n"
                            f"• Full schedule: {os.path.basename(csv_path)}\n\n")


# ====================== RUN APP ======================
if __name__ == "__main__":
    plt.style.use('default')
    app = OSSPGUI()
    app.mainloop()