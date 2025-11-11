import random, time, os
from dataclasses import dataclass
from typing import List
import tkinter as tk
from tkinter import filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import pandas as pd

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
            if inst.p[j][m] == 0: continue
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
            if len(seq) < 2: continue
            for i in range(len(seq)):
                for k in range(i + 1, len(seq)):
                    if iters >= max_iters: return best_ms
                    seq[i], seq[k] = seq[k], seq[i]
                    new_ms = evaluate_makespan(chrom, inst)
                    if new_ms < best_ms:
                        best_ms = new_ms
                        improved = True
                    else:
                        seq[i], seq[k] = seq[k], seq[i]
                    iters += 1
    return best_ms


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


# ====================== GENETIC ALGORITHM ======================
def genetic_algorithm(inst,
                      pop_size=200,
                      max_stagnant_gens=200,
                      hard_timeout=120.0,
                      wisdom_of_crowds=True,
                      collect_history=False):
    population = [create_individual(inst) for _ in range(pop_size)]
    best_chrom = create_individual(inst)
    best_ms = evaluate_makespan(best_chrom, inst)
    history = [best_ms] if collect_history else None
    adjacency = [[0] * inst.m for _ in range(inst.m)]
    start_time = time.time()
    last_improve_gen = 0
    gen = 0

    while True:
        gen += 1
        now = time.time()
        elapsed = now - start_time
        stagnant_gens = gen - last_improve_gen

        if best_ms <= inst.LB: break
        if stagnant_gens >= max_stagnant_gens: break
        if elapsed >= hard_timeout: break

        makespans = [evaluate_makespan(ch, inst) for ch in population]
        min_ms = min(makespans)

        if min_ms < best_ms:
            best_ms = min_ms
            best_idx = makespans.index(min_ms)
            best_chrom = [row[:] for row in population[best_idx]]
            adjacency = build_adjacency(best_chrom, inst)
            improved_ms = local_search(best_chrom, inst, max_iters=2000)
            if improved_ms < best_ms:
                best_ms = improved_ms
            if collect_history: history.append(best_ms)
            last_improve_gen = gen
        else:
            if collect_history: history.append(best_ms)

        if wisdom_of_crowds and stagnant_gens > 25 and random.random() < 0.6:
            population = [create_individual(inst) for _ in range(int(pop_size * 0.75))] + \
                         [best_chrom] * (pop_size // 4)
            random.shuffle(population)

        survivors = [min(random.sample(list(zip(population, makespans)), 5),
                         key=lambda x: x[1])[0] for _ in range(pop_size)]

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
                        if wisdom_of_crowds and adjacency[seq[i]][seq[i + 1]] < 2:
                            seq[i], seq[i + 1] = seq[i + 1], seq[i]

        population = offspring[:pop_size]

    final_ms = local_search(best_chrom, inst, max_iters=10000)
    if final_ms < best_ms:
        best_ms = final_ms
    if collect_history and (not history or history[-1] != best_ms):
        history.append(best_ms)

    return best_ms, best_chrom, history, gen, time.time() - start_time


# ====================== GANTT CHART ======================
def plot_gantt(ax, schedule, inst, title_suffix=""):
    ax.clear()
    job_time = [0] * inst.n
    mach_time = [0] * inst.m
    colors = plt.cm.tab20.colors
    for j in range(inst.n):
        for m in schedule[j]:
            if inst.p[j][m] == 0: continue
            start = max(job_time[j], mach_time[m])
            end = start + inst.p[j][m]
            ax.barh(m, end - start, left=start, height=0.7,
                    color=colors[j % len(colors)], edgecolor='black')
            ax.text(start + (end - start) / 2, m, f'J{j}', va='center', ha='center',
                    fontweight='bold', color='white', fontsize=9)
            job_time[j] = mach_time[m] = end
    makespan = max(job_time + mach_time)
    gap = (makespan - inst.LB) / inst.LB * 100 if inst.LB > 0 else 0
    col = "green" if gap == 0 else "orange" if gap < 5 else "red"
    ax.set_title(f"{inst.name} | n={inst.n} m={inst.m} | MS={makespan} | LB={inst.LB} | Gap={gap:.2f}% {title_suffix}",
                 color=col, fontweight="bold", fontsize=11, pad=15)
    ax.set_xlabel("Time")
    ax.set_ylabel("Machine")
    ax.set_yticks(range(inst.m))
    ax.set_yticklabels([f"M{m}" for m in range(inst.m)])
    ax.invert_yaxis()
    ax.grid(True, axis='x', alpha=0.3)
    ax.margins(y=0.05)


# ====================== GUI ======================
class OSSPGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Open Shop Scheduling Problem GA Solver")
        self.geometry("1350x900")
        self.instance = None
        self.selected_files = []
        self.experiment_data = None
        self.current_fig = None  # Always holds the figure currently displayed
        self.canvas = None
        self.setup_ui()
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    def setup_ui(self):
        toolbar = tk.Frame(self); toolbar.pack(pady=10, fill=tk.X)

        # ---------- SINGLE FILE ----------
        single_f = tk.LabelFrame(toolbar, text="Single File Input", padx=10, pady=5)
        single_f.pack(fill=tk.X, pady=2)
        tk.Label(single_f, text="OSSP File:").pack(side=tk.LEFT)
        self.file_var = tk.StringVar()
        tk.Entry(single_f, textvariable=self.file_var, width=55).pack(side=tk.LEFT, padx=5)
        tk.Button(single_f, text="Browse", command=self.browse_file).pack(side=tk.LEFT)
        tk.Button(single_f, text="Load", command=self.load_file).pack(side=tk.LEFT, padx=5)

        # ---------- EXPERIMENT FILES ----------
        exp_f = tk.LabelFrame(toolbar, text="Experiment Files", padx=10, pady=5)
        exp_f.pack(fill=tk.X, pady=2)
        tk.Button(exp_f, text="Add Files", command=self.add_files).pack(side=tk.LEFT, padx=5)
        self.file_listbox = tk.Listbox(exp_f, height=3, selectmode=tk.MULTIPLE)
        self.file_listbox.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        tk.Button(exp_f, text="Clear", command=self.clear_files).pack(side=tk.LEFT)

        # ---------- PARAMETERS ----------
        param_f = tk.Frame(toolbar); param_f.pack(fill=tk.X, pady=2)
        tk.Label(param_f, text="Pop Size:").pack(side=tk.LEFT)
        self.pop_var = tk.IntVar(value=200)
        tk.Entry(param_f, textvariable=self.pop_var, width=6).pack(side=tk.LEFT, padx=5)

        tk.Label(param_f, text="Stagnant Gens:").pack(side=tk.LEFT, padx=(20,2))
        self.stag_var = tk.IntVar(value=200)
        tk.Entry(param_f, textvariable=self.stag_var, width=6).pack(side=tk.LEFT, padx=5)

        tk.Label(param_f, text="Timeout (s):").pack(side=tk.LEFT, padx=(20,2))
        self.time_var = tk.IntVar(value=120)
        tk.Entry(param_f, textvariable=self.time_var, width=6).pack(side=tk.LEFT, padx=5)

        self.woc_enabled_var = tk.BooleanVar(value=True)
        tk.Checkbutton(param_f, text="Enable Wisdom-of-Crowds", variable=self.woc_enabled_var).pack(side=tk.LEFT, padx=20)

        tk.Label(param_f, text="Runs per Variant:").pack(side=tk.LEFT, padx=(20,2))
        self.runs_var = tk.IntVar(value=5)
        tk.Entry(param_f, textvariable=self.runs_var, width=6).pack(side=tk.LEFT, padx=5)

        # ---------- CONTROLS ----------
        ctrl_f = tk.Frame(toolbar); ctrl_f.pack(fill=tk.X, pady=2)
        tk.Button(ctrl_f, text="Run Single", command=self.run_single).pack(side=tk.LEFT, padx=5)
        tk.Button(ctrl_f, text="Run Experiment", command=self.run_experiment).pack(side=tk.LEFT, padx=5)
        tk.Button(ctrl_f, text="Export Figure", command=self.export_current_figure).pack(side=tk.LEFT, padx=5)

        # ---------- PLOTS ----------
        results_f = tk.Frame(self); results_f.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.info_lbl = tk.Label(results_f, text="No file loaded", justify=tk.LEFT,
                                 anchor="w", fg="gray", font=("Helvetica", 10))
        self.info_lbl.pack(fill=tk.X, pady=5)

        # Single canvas for all figures
        self.current_fig = Figure(figsize=(13, 8))
        self.canvas = FigureCanvasTkAgg(self.current_fig, results_f)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # ------------------------------------------------------------------
    # File handling
    # ------------------------------------------------------------------
    def browse_file(self):
        f = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")],
                                       initialdir="ossp_instances")
        if f: self.file_var.set(f)

    def load_file(self):
        path = self.file_var.get()
        if not path: return messagebox.showerror("Error", "Select a file first.")
        try:
            with open(path) as f:
                lines = [l.strip() for l in f if l.strip() and not l.startswith('#')]
                n, m = map(int, lines[0].split())
                p = [list(map(int, l.split())) for l in lines[1:1+n]]
            self.instance = OSSPInstance(n, m, p, os.path.basename(path))
            self.info_lbl.config(text=f"Loaded: {self.instance.name} | n={n} m={m} | LB={self.instance.LB}", fg="green")
            self.reset_plot()
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def add_files(self):
        files = filedialog.askopenfilenames(filetypes=[("Text files", "*.txt")],
                                            initialdir="ossp_instances")
        for f in files:
            if f not in self.selected_files:
                self.selected_files.append(f)
                self.file_listbox.insert(tk.END, os.path.basename(f))
        self.update_info()

    def clear_files(self):
        self.selected_files = []
        self.file_listbox.delete(0, tk.END)
        self.update_info()

    def update_info(self):
        txt = f"{len(self.selected_files)} files selected" if self.selected_files else "No experiment files"
        self.info_lbl.config(text=txt, fg="black" if self.selected_files else "gray")

    def reset_plot(self):
        """Clear current figure and prepare for new content"""
        self.current_fig.clear()
        self.canvas.draw()

    # ------------------------------------------------------------------
    # SINGLE RUN → Gantt + Convergence
    # ------------------------------------------------------------------
    def run_single(self):
        if not self.instance: return messagebox.showerror("Error", "Load a file first.")
        try:
            pop = self.pop_var.get()
            stag = self.stag_var.get()
            timeout = self.time_var.get()
            woc = self.woc_enabled_var.get()
            self.info_lbl.config(text="Running single GA... (please wait)", fg="blue")
            self.update_idletasks()

            self.reset_plot()
            ax1 = self.current_fig.add_subplot(121)
            ax2 = self.current_fig.add_subplot(122)

            ms, sched, hist, gens, elapsed = genetic_algorithm(
                self.instance, pop, stag, timeout, woc, collect_history=True)

            # Gantt
            plot_gantt(ax1, sched, self.instance)

            # Convergence
            ax2.plot(range(1, len(hist)+1), hist, 'b-', linewidth=1.5)
            ax2.set_xlabel("Generation"); ax2.set_ylabel("Best Makespan")
            ax2.set_title(f"Convergence ({'GA+WOC' if woc else 'GA Only'})")
            ax2.grid(True, alpha=0.3)

            gap = (ms - self.instance.LB) / self.instance.LB * 100
            self.info_lbl.config(text=f"{self.instance.name} | MS={ms} | Gap={gap:.2f}% | {gens} gens | {elapsed:.2f}s", fg="darkgreen")

            self.current_fig.tight_layout(pad=3)
            self.canvas.draw()
        except Exception as e:
            messagebox.showerror("Error", str(e))

    # ------------------------------------------------------------------
    # EXPERIMENT
    # ------------------------------------------------------------------
    def run_experiment(self):
        if len(self.selected_files) == 0:
            return messagebox.showerror("Error", "Add at least one file for experiment.")
        try:
            runs = max(1, self.runs_var.get())
            pop = self.pop_var.get()
            stag = self.stag_var.get()
            timeout = self.time_var.get()
            total_runs = len(self.selected_files) * runs * 2
            current = 0
            rows = []

            self.info_lbl.config(text=f"Experiment: {len(self.selected_files)} files × {runs} runs each...", fg="blue")
            self.update_idletasks()

            for path in self.selected_files:
                with open(path) as f:
                    lines = [l.strip() for l in f if l.strip() and not l.startswith('#')]
                    n, m = map(int, lines[0].split())
                    p = [list(map(int, l.split())) for l in lines[1:1+n]]
                inst = OSSPInstance(n, m, p, os.path.basename(path))

                # GA Only
                ga_ms, ga_t = [], []
                for i in range(runs):
                    current += 1
                    self.info_lbl.config(text=f"{inst.name} | GA {i+1}/{runs} | {current}/{total_runs}")
                    self.update_idletasks()
                    ms, _, _, _, t = genetic_algorithm(inst, pop, stag, timeout, wisdom_of_crowds=False)
                    ga_ms.append(ms); ga_t.append(t)

                # GA + WOC
                woc_ms, woc_t = [], []
                for i in range(runs):
                    current += 1
                    self.info_lbl.config(text=f"{inst.name} | WOC {i+1}/{runs} | {current}/{total_runs}")
                    self.update_idletasks()
                    ms, _, _, _, t = genetic_algorithm(inst, pop, stag, timeout, wisdom_of_crowds=True)
                    woc_ms.append(ms); woc_t.append(t)

                ga_avg = np.mean(ga_ms); woc_avg = np.mean(woc_ms)
                imp = (ga_avg - woc_avg) / ga_avg * 100 if ga_avg else 0
                rows.append({
                    'size': f"{n}×{m}",
                    'ga_avg': round(ga_avg, 1),
                    'ga_std': round(np.std(ga_ms), 1),
                    'ga_time': round(np.mean(ga_t), 2),
                    'woc_avg': round(woc_avg, 1),
                    'woc_std': round(np.std(woc_ms), 1),
                    'woc_time': round(np.mean(woc_t), 2),
                    'imp_%': round(imp, 2)
                })

            self.experiment_data = pd.DataFrame(rows)

            # Reset and show BIG table
            self.reset_plot()
            ax = self.current_fig.add_subplot(111)
            ax.axis('off')

            df_disp = self.experiment_data.copy()
            df_disp['imp_%'] = df_disp['imp_%'].apply(lambda x: f"{x:+.2f}%")

            col_labels = ['Size', 'GA', '±', 't(s)', 'WOC', '±', 't(s)', 'Imp%']
            table_data = df_disp[['size','ga_avg','ga_std','ga_time','woc_avg','woc_std','woc_time','imp_%']].values

            table = ax.table(cellText=table_data, colLabels=col_labels,
                             cellLoc='center', loc='center', bbox=[0, 0, 1, 1])

            table.auto_set_font_size(False)
            table.set_fontsize(16)
            table.scale(1.4, 3.5)

            for i in range(len(col_labels)):
                table[(0, i)].set_facecolor("#1f4e79")
                table[(0, i)].set_text_props(weight='bold', color='white', fontsize=16)

            for i in range(1, len(df_disp)+1):
                table[(i, 7)].set_facecolor("#d5f4e6")
                table[(i, 7)].set_text_props(weight='bold', color='darkgreen', fontsize=16)

            self.current_fig.tight_layout()
            self.canvas.draw()

            self.info_lbl.config(text=f"Experiment complete! Click 'Export Figure' to save.", fg="darkgreen")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    # ------------------------------------------------------------------
    # EXPORT CURRENT FIGURE
    # ------------------------------------------------------------------
    def export_current_figure(self):
        if not self.current_fig or not self.current_fig.axes:
            return messagebox.showerror("Error", "Nothing to export. Run a task first.")
        f = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG Image", "*.png")],
            initialfile="ossp_result.png"
        )
        if f:
            self.current_fig.savefig(f, bbox_inches='tight', dpi=300)
            messagebox.showinfo("Exported", f"Figure saved to:\n{f}")

    def on_closing(self):
        self.quit(); self.destroy()


# ====================== RUN ======================
if __name__ == "__main__":
    plt.style.use('default')
    OSSPGUI().mainloop()