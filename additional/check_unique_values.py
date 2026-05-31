#!/usr/bin/env python
# -*- coding: utf-8 -*-
# check_unique_values.py

"""
This script checks the redunant values in each column in tsv file.
It prints the % of identity in each column.
"""


import argparse


def main():
    parser = argparse.ArgumentParser(description="Calculate the percentage of unique values for each column in a data file.")
    parser.add_argument("--input", required=True, help="Input file with labels/values (e.g., test_labels.txt)")
    args = parser.parse_args()

    print(f"Loading file: {args.input}...")
    
    with open(args.input, 'r') as f:
        # Skip empty lines
        lines = [line.strip() for line in f if line.strip()]

    if not lines:
        print("The file is empty.")
        return

    # Check the first row to determine the number of columns
    first_row = lines[0].split()
    num_columns = len(first_row)

    try:
        [float(x) for x in first_row]
        data_lines = lines
        header = [f"Column {i+1}" for i in range(num_columns)]
    except ValueError:
        header = first_row
        data_lines = lines[1:]

    total_rows = len(data_lines)

    if total_rows == 0:
        print("The file contains no data to analyze.")
        return

    # Create a separate set for each column
    unique_values_per_col = [set() for _ in range(num_columns)]

    # Collect data column by column
    for i, line in enumerate(data_lines, start=1):
        values = line.split()
        
        if len(values) != num_columns:
            print(f"[WARNING]: Row {i} has a different number of columns ({len(values)}) than expected ({num_columns}).")
            
        for col_idx in range(min(len(values), num_columns)):
            unique_values_per_col[col_idx].add(values[col_idx])

    # Draw the table (with original column names)
    print(f"\nTotal number of analyzed rows: {total_rows}")
    print("=" * 90)
    print(f"{'Column Name':<45} | {'Unique Count':<15} | {'% Unique'}")
    print("-" * 90)

    for col_idx in range(num_columns):
        col_name = header[col_idx] if col_idx < len(header) else f"Column {col_idx+1}"
        unique_count = len(unique_values_per_col[col_idx])
        percent_unique = (unique_count / total_rows) * 100.0
        
        print(f"{col_name:<45} | {unique_count:<15} | {percent_unique:>6.2f}%")


if __name__ == "__main__":
    main()
