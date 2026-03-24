# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a data science educational notebook for **FP-Growth association rule mining**. It analyzes the BreadBasket_DMS.csv bakery transaction dataset to discover item purchase patterns (e.g., "customers who buy bread also buy coffee").

## Environment & Setup

- **Runtime**: Google Colab (notebook mounts Google Drive for data access)
- **Python dependencies**: `mlxtend` (install/upgrade via `pip install mlxtend --upgrade`), `numpy`, `pandas`
- **To run locally**: Change the `basicpath` variable in the notebook from the Google Drive path to the local directory, and remove the `google.colab` drive mount cell.

## Architecture

Single Jupyter notebook (`2주차-FP_tree.ipynb`) with this flow:

1. **Data import** — Load `BreadBasket_DMS.csv` (columns: Date, Time, Transaction, Item)
2. **Data cleaning** — Lowercase items, remove `None` values
3. **EDA** — Top-10 item frequency analysis
4. **Data transformation** — Convert to vertical (one-hot encoded) format grouped by Transaction; clip values to 0/1
5. **FP-Growth** — Run `fpgrowth()` with `min_support=0.005`, then `association_rules()` to extract rules filtered by confidence (>0.17)
6. **Apriori comparison** — Benchmark FP-Growth vs Apriori using `%timeit`

## Key Libraries

- `mlxtend.frequent_patterns.fpgrowth` — FP-Growth algorithm
- `mlxtend.frequent_patterns.apriori` — Apriori algorithm (for comparison)
- `mlxtend.frequent_patterns.association_rules` — Extract association rules from frequent itemsets

## Notes

- The notebook is a **worksheet with blank cells** for students to fill in. Many code cells contain only variable assignments with no right-hand side (e.g., `data['Item'] =`).
- Language: Korean (교육용 노트북). Markdown cells explain support, confidence, and lift metrics.
- The dataset contains ~21K bakery transactions from 2016-2017.
