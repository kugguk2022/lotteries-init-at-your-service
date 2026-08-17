from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


def load_data(path: Path) -> pd.DataFrame:
    """Load draws and ensure the first 7 columns are numeric."""

    try_paths = [
        path,
        Path("data/eurodreams.csv"),
        Path("data/eurodreams_draws_2023_to_2025.csv"),
        Path("data/examples/eurodreams_sample.csv"),
    ]
    df = None
    for p in try_paths:
        if p.exists():
            df = pd.read_csv(p)
            break
    if df is None:
        raise FileNotFoundError(f"Could not find EuroDreams data at any of: {try_paths}")

    # Keep first 7 columns; coerce to numeric and drop invalid rows.
    numeric_cols = df.columns[:7]
    df = df.loc[:, numeric_cols].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=numeric_cols).reset_index(drop=True)
    if df.empty:
        raise ValueError(f"No valid numeric rows after cleaning the input data from {p}")
    return df


def run_edreams(data: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    n_total = data.shape[0]
    for i in range(n_total, 99, -1):
        reduced_data = data.iloc[: i - 1].reset_index(drop=True)
        num_rows = reduced_data.shape[0]

        main_numbers_array = reduced_data.iloc[:, :6].to_numpy()
        max_val = int(main_numbers_array.max())
        d_arr = np.array(list(combinations(range(1, max_val + 1), 2))).T

        mydata = [reduced_data.iloc[:o].reset_index(drop=True) for o in range(1, num_rows + 1)]
        z_list = []
        F_final = None

        for current_data in mydata:
            d1 = current_data.iloc[:, :7].to_numpy()
            last_row = current_data.iloc[-1, :7].to_numpy()
            dfg = np.array(list(combinations(last_row, 2))).T

            r = d1.shape[0]
            present = np.zeros((r, max_val), dtype=bool)
            present[np.arange(r)[:, None], d1.astype(int) - 1] = True

            F_values = np.sum(present[:, d_arr[0] - 1] & present[:, d_arr[1] - 1], axis=0)
            F_final = F_values

            z_entry = []
            for x in range(dfg.shape[1]):
                pair = tuple(sorted((dfg[0, x], dfg[1, x])))
                idx = np.where((d_arr[0] == pair[0]) & (d_arr[1] == pair[1]))[0]
                z_entry.append(F_values[idx[0]] if idx.size > 0 else 0)
            z_list.append(z_entry)

        output = np.array(z_list)
        poi = output.sum(axis=1)
        target = int(round(poi[-1]))

        main_pool_max = max_val
        bonus_pool = range(1, 6)
        main_candidates = list(combinations(range(1, main_pool_max + 1), 6))

        candidate_tickets = []
        for main in main_candidates:
            for bonus in bonus_pool:
                if bonus in main:
                    continue
                candidate_tickets.append((main, bonus))

        candidate_G = []
        for main, bonus in candidate_tickets:
            ticket = list(main) + [bonus]
            pair_sum = 0
            for pair in combinations(ticket, 2):
                sp = tuple(sorted(pair))
                idx = np.where((d_arr[0] == sp[0]) & (d_arr[1] == sp[1]))[0]
                if idx.size > 0:
                    pair_sum += F_final[idx[0]]
            candidate_G.append(pair_sum)
        candidate_G = np.array(candidate_G)

        indices = np.where(candidate_G == target)[0]

        drawn_main = tuple(reduced_data.iloc[-1, :6].to_numpy().astype(int))
        drawn_bonus = int(reduced_data.iloc[-1, 6])
        drawn_ticket = (drawn_main, drawn_bonus)
        if not any(candidate_tickets[idx] == drawn_ticket for idx in indices):
            candidate_tickets.append(drawn_ticket)
            candidate_G = np.append(candidate_G, target)
            indices = np.append(indices, len(candidate_tickets) - 1)

        if indices.size > 0:
            output_rows = []
            for idx in indices:
                main, bonus = candidate_tickets[idx]
                output_rows.append(list(main) + [bonus])
            Edreams_matrix = np.array(output_rows, dtype=int)
        else:
            Edreams_matrix = np.empty((0, 7), dtype=int)

        edreams_filename = out_dir / f"Edreams_{i}.csv"
        pd.DataFrame(Edreams_matrix).to_csv(edreams_filename, index=False, header=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate EuroDreams candidate matrices.")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/eurodreams_draws_2023_to_2025.csv"),
        help="Path to EuroDreams draws CSV (first 7 columns are numbers).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("Edreams_output"),
        help="Output directory for generated matrices.",
    )
    args = parser.parse_args()

    df = load_data(args.data)
    run_edreams(df, args.out_dir)
    print(f"Process completed. Edreams matrices saved in '{args.out_dir}'.")


if __name__ == "__main__":
    main()
