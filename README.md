# Adaptive wind power forecasting with selective model retraining

Code and results for RSEIT-2026-000079.

## Contents

| File / folder | What it is |
|---|---|
| `rsdr_paper_code.py` | One Python file that produces every result reported in the paper |
| `check_paper_numbers.py` | Compares all 96 numbers in the paper with the result files |
| `data/` | Location1.csv to Location4.csv: hourly data, 2 Jan 2017 to 31 Dec 2021 |
| `results/` | The results reported in the paper, as produced by this code |
| `requirements.txt` | Library versions used for the reported results |

Data source: M. Rahim, "Wind Power Generation Data - Forecasting", Kaggle,
https://www.kaggle.com/datasets/mubashirrahim/wind-power-generation-data-forecasting

## What the code does

`rsdr_paper_code.py` runs four stages in order. Each writes its own folder.

| Stage | Output folder | Used in the paper |
|---|---|---|
| 03 2020 development | `outputs_skill_trigger_development/` | RSDR threshold selection (Section 3.2) |
| 04 2021 evaluation | `outputs_final_2021/` | Static, persistence, periodic 30/90/180-day and RSDR results (Tables 1-2) |
| 05 post-hoc analysis | `outputs_posthoc_final/` | Block bootstrap intervals, update efficiency, 30-day post-update MAE (Table 3) |
| 06 ADWIN and cost | `outputs_final_cost_adwin/` | ADWIN comparators and update-fitting times (Table 1) |

The stage numbers follow the original pipeline. Stages 00-02 were exploratory
development steps whose results are not reported in the paper, so they are not
included. The separate plotting script used to draw the paper's Figures 2 and 3
is not part of this package; the stages write their own working plots.

## How to run

```
pip install -r requirements.txt
python rsdr_paper_code.py            # all stages, about 15-20 minutes
python rsdr_paper_code.py 04 05      # selected stages (05 needs 04 first)
python check_paper_numbers.py .      # compare the new outputs with the paper
```

Run from this folder, with the four CSV files in `data/`.

## Where each paper number comes from

| Paper | File |
|---|---|
| Section 3.2, threshold selection | `outputs_skill_trigger_development/02_threshold_tradeoff_summary_2020.csv` |
| Table 1, MAE, RMSE, refits | `outputs_final_2021/02_final_2021_aggregate_summary.csv` |
| Table 1, ADWIN rows and update-fitting times | `outputs_final_cost_adwin/02_final_cost_aggregate.csv`, `01_final_cost_policy_summary.csv` |
| Table 2 | `outputs_final_2021/01_final_2021_policy_summary.csv` |
| Table 3 | `outputs_posthoc_final/03_proposed_update_utility.csv` (30-day rows) |
| Bootstrap intervals (Section 4.1) | `outputs_posthoc_final/02_cross_location_block_bootstrap.csv` |
| RSDR update dates | `outputs_final_2021/03_final_2021_update_log.csv` |
| ADWIN alarms | `outputs_final_cost_adwin/04_frozen_adwin_alarm_log.csv` |

`results/paper_number_check.csv` lists all 96 numbers with the value in the
result files. All 96 match.

## Reproducibility

The results in `results/` were produced on Windows 11 with Python 3.13.9,
XGBoost 3.4.1, River 0.26.1, pandas 3.0.1 and numpy 2.4.3.

XGBoost samples rows and columns at random (subsample = colsample_bytree = 0.8,
seed 42). It draws these random numbers with the C++ standard library, which
differs between Windows and Linux or macOS, so on another operating system the
same seed gives slightly different trees. On Linux with the same library
versions the method behaves identically: RSDR makes the same three updates at
Location 3 on 4 July, 3 August and 28 October 2021, and every conclusion holds.
The MAE values differ in the third significant digit, for example a static mean
MAE of 0.00705 instead of 0.00694, so `check_paper_numbers.py` will flag those
small differences there. To reproduce the exact numbers, run on Windows with the
versions in `requirements.txt`.

Fitting times depend on the computer and change between runs.
