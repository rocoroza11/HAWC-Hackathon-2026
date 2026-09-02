#!/usr/bin/env python3
"""
Concatenate the Reading University Atmospheric Observatory 5-minute day-files
into ONE clean, time-ordered CSV.

Difference from the assembler: this KEEPS EVERY COLUMN. It does not drop or
select anything. Its only job is to:
  - stack all per-day files in chronological order,
  - give the header clean, unique names (the raw files have 28 blank "-"
    columns; these are renamed unused_01, unused_02, ... so nothing collides),
  - drop the units row that sits under the header in each file
    (saved separately to <output>.units.csv so the info isn't lost),
  - parse the TimeStamp as UK-format dates and sort by it.

Folder layout expected:  ROOT / <year> / <year>-AVG5-<doy>.csv

Usage:
    python concat_reading_5min.py archive -o reading_5min_all_columns.csv
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


def clean_headers(raw_names):
    """
    Return a list of clean, unique column names.
    - Real names are kept as-is.
    - Blank '-' placeholders become unused_01, unused_02, ...
    - Any accidental duplicates get a _2, _3 suffix so nothing collides.
    """
    out = []
    seen = {}
    unused = 0
    for name in raw_names:
        n = name.strip()
        if n == "-" or n == "":
            unused += 1
            n = f"unused_{unused:02d}"
        # de-duplicate any repeated real names
        if n in seen:
            seen[n] += 1
            n = f"{n}_{seen[n]}"
        else:
            seen[n] = 1
        out.append(n)
    return out


def read_one(path, clean_names):
    """Read one day-file: skip the units row, apply clean headers, parse types."""
    try:
        df = pd.read_csv(
            path,
            header=0,          # row 0 is the (raw) header
            skiprows=[1],      # row 1 is the units row -> skip
            names=clean_names, # override with our cleaned names
            dtype=str,
            na_filter=False,
        )
    except Exception as e:
        print(f"  [skip] {os.path.basename(path)}: unreadable ({e})", file=sys.stderr)
        return None

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
    ap.add_argument("-o", "--output", default="reading_5min_all_columns.csv")
    args = ap.parse_args()

    files = find_files(args.root)
    if not files:
        print(f"No files found under {args.root!r} matching *-AVG5-*.csv",
              file=sys.stderr)
        sys.exit(1)
    print(f"Found {len(files)} day-files.", file=sys.stderr)

    # Build the clean header + capture the units row from the first file.
    first = pd.read_csv(files[0], header=None, dtype=str, na_filter=False, nrows=2)
    raw_names = list(first.iloc[0])
    units_row = list(first.iloc[1])
    clean_names = clean_headers(raw_names)

    # Save the header<->units<->raw-name mapping so nothing is lost.
    units_path = args.output + ".units.csv"
    pd.DataFrame({
        "clean_name": clean_names,
        "raw_name": raw_names,
        "units": units_row,
    }).to_csv(units_path, index=False)
    print(f"Saved column/units key -> {units_path}", file=sys.stderr)

    frames, used = [], 0
    for i, path in enumerate(files, 1):
        d = read_one(path, clean_names)
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

    df.to_csv(args.output, index=False)

    print("", file=sys.stderr)
    print(f"Files used:  {used}/{len(files)}", file=sys.stderr)
    print(f"Rows:        {len(df):,}", file=sys.stderr)
    print(f"Columns:     {len(df.columns)} (all kept)", file=sys.stderr)
    print(f"Span:        {df['TimeStamp'].min()}  ->  {df['TimeStamp'].max()}",
          file=sys.stderr)
    print(f"Wrote {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
