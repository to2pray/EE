"""
FINAL EXPERIMENT SCRIPT
Runs: QuickSort vs MergeSort
Tracks: Time, Peak Memory, Comparisons
Constraints: 128 / 256 / 512 / 1024 MB
Dataset Types: random, nearly_sorted, reverse, duplicate_heavy
Sizes: 10000, 50000, 100000, 500000
Trials: 5
"""

import os
import csv
import time
from datetime import datetime, timezone  # Fixed import
from tqdm import tqdm
from statistics import mean, stdev

from memconstraint import MemoryMonitor
from mysorts import quicksort_wrapper, mergesort_wrapper
from datagen import (
    generate_random,
    generate_nearly_sorted,
    generate_reverse_sorted,
    generate_duplicate_heavy
)

# -----------------------------
# CONFIGURATION
# -----------------------------
SIZES = [10000, 50000, 100000, 500000]
STRUCTURES = ["random", "nearly_sorted", "reverse", "duplicate_heavy"]
TRIALS = 5
MEMORY_LIMITS = [128, 256, 512, 1024]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "../results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Fixed datetime deprecation
RAW_CSV = os.path.join(
    RESULTS_DIR,
    f"raw_results_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.csv"
)


# --------------------------------
# CSV WRITER
# --------------------------------
def write_raw_header():
    with open(RAW_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "size",
            "structure",
            "trial",
            "memory_limit_MB",
            "QuickSort_time_s",
            "QuickSort_memory_MB",
            "QuickSort_comparisons",
            "QuickSort_success",
            "MergeSort_time_s",
            "MergeSort_memory_MB",
            "MergeSort_comparisons",
            "MergeSort_success",
        ])


def append_raw_row(row):
    with open(RAW_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)


# --------------------------------
# SAVE TABLE 1 PER MEMORY LIMIT
# --------------------------------
def write_summary_table(memory_limit, data_rows):
    """
    data_rows: list of dicts
    """
    outpath = os.path.join(RESULTS_DIR, f"table1_mem{memory_limit}MB.csv")
    with open(outpath, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "size",
            "structure",
            "trials_succeeded",
            "QuickSort_mean_time",
            "QuickSort_std_time",
            "QuickSort_mean_memory",
            "QuickSort_std_memory",
            "QuickSort_mean_comparisons",
            "QuickSort_std_comparisons",
            "MergeSort_mean_time",
            "MergeSort_std_time",
            "MergeSort_mean_memory",
            "MergeSort_std_memory",
            "MergeSort_mean_comparisons",
            "MergeSort_std_comparisons"
        ])
        for row in data_rows:
            w.writerow([
                row["size"],
                row["structure"],
                row["n_trials"],
                row["qs_mean_t"],
                row["qs_std_t"],
                row["qs_mean_m"],
                row["qs_std_m"],
                row["qs_mean_c"],
                row["qs_std_c"],
                row["ms_mean_t"],
                row["ms_std_t"],
                row["ms_mean_m"],
                row["ms_std_m"],
                row["ms_mean_c"],
                row["ms_std_c"],
            ])
    print(f"\nWrote table for mem {memory_limit} MB -> {outpath}")


# --------------------------------
# MAIN EXPERIMENT LOOP
# --------------------------------

def run_experiments():
    print(f"Writing raw results to {RAW_CSV}")
    write_raw_header()

    total_runs = len(SIZES) * len(STRUCTURES) * TRIALS * len(MEMORY_LIMITS)

    pbar = tqdm(total=total_runs, desc="Total runs", unit="run")

    # Loop memory limits separately (so we can compute per-limit table)
    for mem_limit in MEMORY_LIMITS:
        print(f"\n{'='*70}")
        print(f"Running experiments for memory limit = {mem_limit} MB")
        print(f"{'='*70}")

        summary_rows = []
        aggregated = {}  # key = (size, structure), value = list of trial dicts

        for size in SIZES:
            print(f"\nDataset size: {size:,} elements")

            for structure in STRUCTURES:
                print(f"  {structure:20s}: ", end="", flush=True)

                for trial in range(1, TRIALS + 1):
                    
                    # ==========================================
                    # CRITICAL FIX: Generate dataset OUTSIDE memory constraint
                    # ==========================================
                    try:
                        if structure == "random":
                            dataset = generate_random(size)
                        elif structure == "nearly_sorted":
                            dataset = generate_nearly_sorted(size)
                        elif structure == "reverse":
                            dataset = generate_reverse_sorted(size)
                        elif structure == "duplicate_heavy":
                            dataset = generate_duplicate_heavy(size)
                        else:
                            raise ValueError(f"Unknown structure: {structure}")
                    except MemoryError as e:
                        print(f"\n  ERROR: Cannot generate {size} dataset (insufficient memory)")
                        # Skip this configuration entirely
                        pbar.update(1)
                        continue
                    
                    # --------------------
                    # QUICK SORT
                    # --------------------
                    qs_success = False
                    try:
                        with MemoryMonitor(limit_mb=mem_limit) as monitor_qs:
                            start = time.perf_counter()
                            _, qs_comparisons = quicksort_wrapper(dataset)
                            qs_time = time.perf_counter() - start
                            qs_memory = monitor_qs.get_peak_usage()
                        
                        if not monitor_qs.memory_exceeded:
                            qs_success = True
                        else:
                            qs_time = None
                            qs_memory = None
                            qs_comparisons = None
                    
                    except Exception as e:
                        print(f"\n  QS ERROR: {e}")
                        qs_time = None
                        qs_memory = None
                        qs_comparisons = None

                    # --------------------
                    # MERGE SORT
                    # --------------------
                    ms_success = False
                    try:
                        with MemoryMonitor(limit_mb=mem_limit) as monitor_ms:
                            start = time.perf_counter()
                            _, ms_comparisons = mergesort_wrapper(dataset)
                            ms_time = time.perf_counter() - start
                            ms_memory = monitor_ms.get_peak_usage()
                        
                        if not monitor_ms.memory_exceeded:
                            ms_success = True
                        else:
                            ms_time = None
                            ms_memory = None
                            ms_comparisons = None
                    
                    except Exception as e:
                        print(f"\n  MS ERROR: {e}")
                        ms_time = None
                        ms_memory = None
                        ms_comparisons = None

                    # Write to raw CSV
                    append_raw_row([
                        size,
                        structure,
                        trial,
                        mem_limit,
                        qs_time if qs_time is not None else "FAIL",
                        qs_memory if qs_memory is not None else "FAIL",
                        qs_comparisons if qs_comparisons is not None else "FAIL",
                        qs_success,
                        ms_time if ms_time is not None else "FAIL",
                        ms_memory if ms_memory is not None else "FAIL",
                        ms_comparisons if ms_comparisons is not None else "FAIL",
                        ms_success,
                    ])

                    # Store for summary (only successful trials)
                    if qs_success and ms_success:
                        key = (size, structure)
                        aggregated.setdefault(key, [])
                        aggregated[key].append({
                            "qs_t": qs_time,
                            "qs_m": qs_memory,
                            "qs_c": qs_comparisons,
                            "ms_t": ms_time,
                            "ms_m": ms_memory,
                            "ms_c": ms_comparisons,
                        })
                        print("✓", end="", flush=True)
                    else:
                        print("✗", end="", flush=True)

                    pbar.update(1)
                
                print()  # Newline after structure

        # -------------------------
        # BUILD SUMMARY TABLE
        # -------------------------
        print(f"\nComputing summary statistics...")
        for (size, structure), entry_list in sorted(aggregated.items()):
            if not entry_list:
                continue

            qs_times = [e["qs_t"] for e in entry_list]
            qs_mems = [e["qs_m"] for e in entry_list]
            qs_comps = [e["qs_c"] for e in entry_list]

            ms_times = [e["ms_t"] for e in entry_list]
            ms_mems = [e["ms_m"] for e in entry_list]
            ms_comps = [e["ms_c"] for e in entry_list]

            summary_rows.append({
                "size": size,
                "structure": structure,
                "n_trials": len(entry_list),

                "qs_mean_t": mean(qs_times),
                "qs_std_t": stdev(qs_times) if len(qs_times) > 1 else 0,

                "qs_mean_m": mean(qs_mems),
                "qs_std_m": stdev(qs_mems) if len(qs_mems) > 1 else 0,

                "qs_mean_c": mean(qs_comps),
                "qs_std_c": stdev(qs_comps) if len(qs_comps) > 1 else 0,

                "ms_mean_t": mean(ms_times),
                "ms_std_t": stdev(ms_times) if len(ms_times) > 1 else 0,

                "ms_mean_m": mean(ms_mems),
                "ms_std_m": stdev(ms_mems) if len(ms_mems) > 1 else 0,

                "ms_mean_c": mean(ms_comps),
                "ms_std_c": stdev(ms_comps) if len(ms_comps) > 1 else 0,
            })

        write_summary_table(mem_limit, summary_rows)

    pbar.close()
    print("\n" + "="*70)
    print("EXPERIMENTS COMPLETE")
    print("="*70)
    print(f"Raw data: {RAW_CSV}")
    print(f"Summary tables: {RESULTS_DIR}/table1_mem*.csv")
    print("="*70)


# --------------------------------
# ENTRY POINT
# --------------------------------
if __name__ == "__main__":
    try:
        run_experiments()
    except KeyboardInterrupt:
        print("\n\nExperiment interrupted by user (Ctrl+C)")
        print("Partial results saved to CSV files")
    except Exception as e:
        print(f"\n\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()