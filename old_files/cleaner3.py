#!/usr/bin/env python3
"""
Concatenate the Reading University Atmospheric Observatory 5-minute day-files
into ONE clean, time-ordered CSV.

Keeps every REAL column and DROPS the blank placeholder columns (the 28 columns
named "-" in the raw header, plus any column that turns out to be entirely empty
across the whole archive).

What it does:
  - stacks all per-day files in chronological order,
  - drops the raw "-" placeholder columns outright,
  - drops the units row that sits under the header in each file
    (saved separately to <output>.units.csv so the info isn't lost),
  - parses TimeStamp as UK-format (day-first) dates and sorts by it,
  - optionally also drops columns that are 100% empty in the assembled data
    (on by default; disable with --keep-empty).

Folder layout expected:  ROOT / <year> / <year>-AVG5-<doy>.csv

Usage:
    python concat_reading_5min.py archive -o reading_5min_clean.csv

    # keep columns that exist in the header but are entirely empty in the data:
    python concat_reading_5min.py archive -o out.csv --keep-empty
"""

import argparse
import glob
import os
import sys

import pandas as pd


def find_files(root):
    pattern = os.path.join(root, "*", "*-AVG5-*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        files = sorted(glob.glob(os.path.join(root, "**", "*-AVG5-*.csv"),
                                 recursive=True))
    return files


def build_column_plan(first_file):
    """
    Read the header + units rows of one file and work out:
      - raw_names: the header exactly as written
      - units_row: the units line beneath it
      - clean_names: names for every column (placeholders get a temporary
        __drop_NN__ name so we can identify and remove them)
      - keep_names: the subset we actually keep (real columns only)
    """
    first = pd.read_csv(first_file, header=None, dtype=str, na_filter=False, nrows=2)
    raw_names = list(first.iloc[0])
    units_row = list(first.iloc[1])

    clean_names = []
    keep_names = []
    seen = {}
    placeholder = 0

    for name in raw_names:
        n = name.strip()
        if n in ("-", ""):
            # Blank placeholder column -> mark for dropping.
            placeholder += 1
            clean_names.append(f"__drop_{placeholder:02d}__")
            continue
        # De-duplicate any repeated real names so nothing collides.
        if n in seen:
            seen[n] += 1
            n = f"{n}_{seen[n]}"
        else:
            seen[n] = 1
        clean_names.append(n)
        keep_names.append(n)

    return raw_names, units_row, clean_names, keep_names


def read_one(path, clean_names, keep_names):
    """Read one day-file: skip units row, apply clean names, keep real columns."""
    try:
        df = pd.read_csv(
            path,
            header=0,           # row 0 is the raw header
            skiprows=[1],       # row 1 is the units row -> skip
            names=clean_names,  # override with our cleaned names
            dtype=str,
            na_filter=False,
        )
    except Exception as e:
        print(f"  [skip] {os.path.basename(path)}: unreadable ({e})", file=sys.stderr)
        return None

    # Drop the placeholder columns entirely.
    df = df[keep_names].copy()

    # Parse timestamp (UK day-first); drop rows with no valid timestamp.
    df["TimeStamp"] = pd.to_datetime(df["TimeStamp"], errors="coerce", dayfirst=True)
    df = df[df["TimeStamp"].notna()]

    # Convert every non-timestamp column to numeric; blanks/junk -> NaN.
    for c in df.columns:
        if c != "TimeStamp":
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", help="Archive root containing per-year subfolders")
    ap.add_argument("-o", "--output", default="reading_5min_clean.csv")
    ap.add_argument("--keep-empty", action="store_true",
                    help="Keep columns that are entirely empty across the archive "
                         "(by default these are dropped too).")
    args = ap.parse_args()

    files = find_files(args.root)
    if not files:
        print(f"No files found under {args.root!r} matching *-AVG5-*.csv",
              file=sys.stderr)
        sys.exit(1)
    print(f"Found {len(files)} day-files.", file=sys.stderr)

    raw_names, units_row, clean_names, keep_names = build_column_plan(files[0])
    n_placeholder = len(clean_names) - len(keep_names)
    print(f"Columns in file: {len(clean_names)}  "
          f"(dropping {n_placeholder} blank placeholders)", file=sys.stderr)

    frames, used = [], 0
    for i, path in enumerate(files, 1):
        d = read_one(path, clean_names, keep_names)
        if d is not None and len(d):
            frames.append(d)
            used += 1
        if i % 500 == 0:
            print(f"  ...{i}/{len(files)} processed", file=sys.stderr)

    if not frames:
        print("No usable data.", file=sys.stderr)
        sys.exit(1)

    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values("TimeStamp").drop_duplicates(subset="TimeStamp", keep="first")

    # Optionally drop columns that are entirely empty across the whole archive
    # (named columns that never actually carried data, e.g. SoilMoisture, CNR4T).
    dropped_empty = []
    if not args.keep_empty:
        for c in df.columns:
            if c != "TimeStamp" and df[c].notna().sum() == 0:
                dropped_empty.append(c)
        if dropped_empty:
            df = df.drop(columns=dropped_empty)

    # Save the units key for the columns we actually kept.
    kept = list(df.columns)
    units_lookup = {clean: unit for clean, unit in zip(clean_names, units_row)}
    raw_lookup = {clean: raw for clean, raw in zip(clean_names, raw_names)}
    units_path = args.output + ".units.csv"
    pd.DataFrame({
        "column": kept,
        "raw_name": [raw_lookup.get(c, c) for c in kept],
        "units": [units_lookup.get(c, "") for c in kept],
    }).to_csv(units_path, index=False)

    df.to_csv(args.output, index=False)

    print("", file=sys.stderr)
    print(f"Files used:        {used}/{len(files)}", file=sys.stderr)
    print(f"Rows:              {len(df):,}", file=sys.stderr)
    print(f"Placeholders cut:  {n_placeholder}", file=sys.stderr)
    if dropped_empty:
        print(f"Empty cols cut:    {len(dropped_empty)} -> {', '.join(dropped_empty)}",
              file=sys.stderr)
    print(f"Columns kept:      {len(df.columns)}", file=sys.stderr)
    print(f"Span:              {df['TimeStamp'].min()}  ->  {df['TimeStamp'].max()}",
          file=sys.stderr)
    print(f"Wrote {args.output}", file=sys.stderr)
    print(f"Wrote {units_path}", file=sys.stderr)


if __name__ == "__main__":
    main()