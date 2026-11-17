#!/usr/bin/env python3
"""
Adaptive wind power forecasting with selective model retraining
(RSEIT-2026-000079) - complete code for all results reported in the paper.

This single file contains the four pipeline stages whose results appear in the
paper, in the order they were run:

  stage 03  2020 development: RSDR threshold selection (Section 3.2)
  stage 04  untouched 2021 evaluation: static, persistence, periodic and RSDR
            policies, location results and block bootstrap (Tables 1-2)
  stage 05  post-hoc analysis of stage-04 outputs: accuracy/update efficiency,
            cross-location bootstrap and 30-day post-update utility (Table 3)
  stage 06  frozen ADWIN comparators and update-fitting cost (Table 1)

Each stage's code is identical to the original stage script. Only its top-level
names carry a stage prefix (s03_, s04_, ...), so that each stage keeps its own
version of shared helper functions exactly as it was run.

Data
----
Put Location1.csv ... Location4.csv (Kaggle "Wind Power Generation Data -
Forecasting", M. Rahim) in a folder named "data" next to this file.

Run
---
    pip install pandas==3.0.1 numpy==2.4.3 xgboost==3.4.1 river==0.26.1 scikit-learn matplotlib
    python rsdr_paper_code.py              # all stages (03, 04, 05, 06)
    python rsdr_paper_code.py 04 05        # selected stages only (05 needs 04 first)

Outputs are written next to this file, one folder per stage:
    outputs_skill_trigger_development/   (stage 03)
    outputs_final_2021/                  (stage 04)
    outputs_posthoc_final/               (stage 05)
    outputs_final_cost_adwin/            (stage 06)

Reproducibility note
--------------------
The reported results were produced on Windows 11 with Python 3.13.9,
XGBoost 3.4.1, River 0.26.1, pandas 3.0.1 and numpy 2.4.3. XGBoost row/column
subsampling (subsample = colsample_bytree = 0.8) draws its random numbers with
the C++ standard library, which differs between Windows and Linux/macOS, so the
same seed gives slightly different trees on other operating systems. On Linux
the RSDR decisions are identical (3 updates, Location 3, 4 Jul, 3 Aug and
28 Oct 2021) and MAE values differ in the third significant digit.
"""

from pathlib import Path
import json
import sys
import platform
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from xgboost import XGBRegressor
import xgboost
import time


# ============================================================================
# STAGE 03: 2020 development: RSDR threshold selection (paper Section 3.2)
# Original file: 03_skill_trigger_development_2020.py (code unchanged; names prefixed s03_ / S03_)
# ============================================================================
#
# 03_skill_trigger_development_2020.py
# ====================================
#
# Stage 4 of the clean wind-power adaptation study.
#
# PURPOSE
# -------
# Develop and validate a causal, selective retraining trigger using ONLY 2020.
#
# This stage was motivated by two findings from the earlier 2020 experiments:
#
# 1. ADWIN on raw absolute error was too conservative.
# 2. A single ADWIN excess-error alarm was useful for Location3, but missed
#    useful adaptation opportunities at Location2.
#
# The new trigger monitors RELATIVE FORECAST SKILL rather than raw error.
#
# DEFINITION
# ----------
# For each completed day, define:
#
#     SkillRatio = MAE_XGBoost / MAE_Persistence
#
# Lower is better.
#
# At each daily decision point:
#
#     RecentSkill = SkillRatio pooled over the most recent 30 days
#     ReferenceSkill = SkillRatio pooled over the preceding 60 days
#
#     DegradationScore = RecentSkill / ReferenceSkill
#
# If DegradationScore >= threshold, the learned model has lost a substantial
# fraction of its historical advantage relative to persistence.
#
# Candidate thresholds tested:
#     1.4, 1.5, 1.6, 1.7
#
# A 30-day cooldown prevents repeated updates from the same regime.
#
# ADAPTATION
# ----------
# When triggered, XGBoost is retrained using an EXPANDING history:
# all forecast rows whose targets are known at that decision time.
#
# This choice is deliberate. Script 02 showed that short rolling retraining
# windows can be badly contaminated by prolonged zero-generation regimes.
#
# CAUSALITY
# ---------
# A forecast row at time t predicts Power(t+1).
#
# The last forecast of a day, at 23:00, is evaluated when Power at 00:00 of
# the next day becomes available. The daily trigger is then evaluated, and
# any retraining is completed BEFORE forecasting the next 00:00 row
# (for Power at 01:00).
#
# Thus no future target is used.
#
# DATA SPLIT
# ----------
# Training:
#     through 2019-12-31 23:00
#
# Development / trigger selection:
#     2020 ONLY
#
# 2021:
#     NOT USED anywhere in this script.
#
# EXPECTED PROJECT STRUCTURE
# --------------------------
# D:\Wind Drift Paper    00_drift_diagnostic.py
#     01_adwin_signal_development_2020.py
#     02_adaptation_policy_development_2020.py
#     03_skill_trigger_development_2020.py
#     data        Location1.csv
#         Location2.csv
#         Location3.csv
#         Location4.csv
#
# OUTPUTS
# -------
# outputs_skill_trigger_development

warnings.filterwarnings("ignore", category=FutureWarning)


# ============================================================================
# CONFIGURATION
# ============================================================================

S03_BASE_DIR = Path(__file__).resolve().parent
S03_DATA_DIR = S03_BASE_DIR / "data"

S03_OUTPUT_DIR = S03_BASE_DIR / "outputs_skill_trigger_development"
S03_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

S03_LOCATIONS = [
    "Location1",
    "Location2",
    "Location3",
    "Location4",
]

S03_TRAIN_END = pd.Timestamp("2019-12-31 23:00:00")
S03_VALIDATION_START = pd.Timestamp("2020-01-01 00:00:00")
S03_VALIDATION_END = pd.Timestamp("2020-12-31 23:00:00")

S03_RECENT_DAYS = 30
S03_REFERENCE_DAYS = 60
S03_COOLDOWN_DAYS = 30

S03_THRESHOLDS = [
    1.4,
    1.5,
    1.6,
    1.7,
]

S03_RANDOM_STATE = 42

S03_XGB_PARAMS = dict(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    min_child_weight=3,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    eval_metric="rmse",
    random_state=S03_RANDOM_STATE,
    n_jobs=-1,
)

S03_POWER_LAGS = [
    1, 2, 3, 6, 12, 24, 48, 72, 168
]

S03_ROLL_WINDOWS = [
    3, 6, 12, 24, 48, 168
]


# ============================================================================
# UTILITIES
# ============================================================================

def s03_section(title):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def s03_load_raw(location):
    path = S03_DATA_DIR / f"{location}.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"\nMissing dataset:\n{path}\n"
        )

    df = pd.read_csv(path)

    required = [
        "Time",
        "temperature_2m",
        "relativehumidity_2m",
        "dewpoint_2m",
        "windspeed_10m",
        "windspeed_100m",
        "winddirection_10m",
        "winddirection_100m",
        "windgusts_10m",
        "Power",
    ]

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{location}: missing columns: {missing}"
        )

    df["Time"] = pd.to_datetime(
        df["Time"],
        errors="coerce",
    )

    if df["Time"].isna().any():
        raise ValueError(
            f"{location}: invalid timestamps."
        )

    df = (
        df
        .sort_values("Time")
        .reset_index(drop=True)
    )

    return df


def s03_create_features(df):
    """
    Canonical leakage-safe +1-hour forecasting problem.
    """
    d = df.copy()

    # Calendar variables
    d["hour"] = d["Time"].dt.hour
    d["day_of_week"] = d["Time"].dt.dayofweek
    d["month"] = d["Time"].dt.month
    d["day_of_year"] = d["Time"].dt.dayofyear

    d["hour_sin"] = np.sin(
        2 * np.pi * d["hour"] / 24
    )
    d["hour_cos"] = np.cos(
        2 * np.pi * d["hour"] / 24
    )

    d["day_of_week_sin"] = np.sin(
        2 * np.pi * d["day_of_week"] / 7
    )
    d["day_of_week_cos"] = np.cos(
        2 * np.pi * d["day_of_week"] / 7
    )

    d["month_sin"] = np.sin(
        2 * np.pi * d["month"] / 12
    )
    d["month_cos"] = np.cos(
        2 * np.pi * d["month"] / 12
    )

    # Historical power features
    for lag in S03_POWER_LAGS:
        d[f"Power_lag_{lag}"] = (
            d["Power"].shift(lag)
        )

    shifted_power = d["Power"].shift(1)

    for window in S03_ROLL_WINDOWS:
        d[f"Power_roll_mean_{window}"] = (
            shifted_power
            .rolling(window)
            .mean()
        )

        d[f"Power_roll_std_{window}"] = (
            shifted_power
            .rolling(window)
            .std()
        )

    # Strict one-hour-ahead target
    d["Target_Power"] = (
        d["Power"].shift(-1)
    )

    d = d.dropna().reset_index(drop=True)

    return d


def s03_new_model():
    return XGBRegressor(**S03_XGB_PARAMS)


def s03_metrics(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    return {
        "MAE": mean_absolute_error(
            y_true,
            y_pred,
        ),
        "RMSE": (
            mean_squared_error(
                y_true,
                y_pred,
            ) ** 0.5
        ),
        "R2": r2_score(
            y_true,
            y_pred,
        ),
    }


def s03_pooled_skill_ratio(
    xgb_errors,
    persistence_errors,
):
    """
    MAE_XGB / MAE_Persistence for a pooled window.
    """
    xgb_errors = np.asarray(
        xgb_errors,
        dtype=float,
    )

    persistence_errors = np.asarray(
        persistence_errors,
        dtype=float,
    )

    xgb_mae = np.mean(xgb_errors)
    persistence_mae = np.mean(
        persistence_errors
    )

    if persistence_mae <= 1e-12:
        return np.nan

    return xgb_mae / persistence_mae


# ============================================================================
# STATIC BASELINE
# ============================================================================

def s03_run_static(
    featured,
    feature_columns,
):
    train = featured[
        featured["Time"] <= S03_TRAIN_END
    ].copy()

    validation = featured[
        (featured["Time"] >= S03_VALIDATION_START)
        & (featured["Time"] <= S03_VALIDATION_END)
    ].copy().reset_index(drop=True)

    model = s03_new_model()

    model.fit(
        train[feature_columns],
        train["Target_Power"],
    )

    pred = model.predict(
        validation[feature_columns]
    )

    xgb_error = np.abs(
        validation["Target_Power"].to_numpy()
        - pred
    )

    persistence_error = np.abs(
        validation["Target_Power"].to_numpy()
        - validation["Power"].to_numpy()
    )

    return (
        validation,
        pred,
        xgb_error,
        persistence_error,
    )


# ============================================================================
# DYNAMIC SKILL-TRIGGER POLICY
# ============================================================================

def s03_run_skill_trigger_policy(
    featured,
    feature_columns,
    threshold,
):
    """
    Daily prequential evaluation.

    The trigger is recomputed from the CURRENT adaptive model's own
    error stream. Trigger times are therefore not precomputed from
    a static model.
    """

    initial_train = featured[
        featured["Time"] <= S03_TRAIN_END
    ].copy()

    validation = featured[
        (featured["Time"] >= S03_VALIDATION_START)
        & (featured["Time"] <= S03_VALIDATION_END)
    ].copy().reset_index(drop=True)

    # Hard guard against accidental 2021 use.
    validation_years = sorted(
        validation["Time"]
        .dt.year
        .unique()
        .tolist()
    )

    if validation_years != [2020]:
        raise RuntimeError(
            f"Expected 2020 only, got years "
            f"{validation_years}"
        )

    model = s03_new_model()

    model.fit(
        initial_train[feature_columns],
        initial_train["Target_Power"],
    )

    n = len(validation)

    predictions = np.full(
        n,
        np.nan,
        dtype=float,
    )

    xgb_errors = np.full(
        n,
        np.nan,
        dtype=float,
    )

    persistence_errors = np.full(
        n,
        np.nan,
        dtype=float,
    )

    score_at_row = np.full(
        n,
        np.nan,
        dtype=float,
    )

    recent_skill_at_row = np.full(
        n,
        np.nan,
        dtype=float,
    )

    reference_skill_at_row = np.full(
        n,
        np.nan,
        dtype=float,
    )

    updates = []

    last_update_day = None

    # Forecast one calendar day at a time.
    day_groups = list(
        validation.groupby(
            validation["Time"].dt.floor("D")
        ).groups.items()
    )

    for day, index_values in day_groups:

        indices = np.array(
            list(index_values),
            dtype=int,
        )

        # Forecast the full day with the model that was
        # available at the beginning of that day.
        day_pred = model.predict(
            validation.loc[
                indices,
                feature_columns,
            ]
        )

        predictions[indices] = day_pred

        day_true = validation.loc[
            indices,
            "Target_Power",
        ].to_numpy()

        day_persistence = validation.loc[
            indices,
            "Power",
        ].to_numpy()

        xgb_errors[indices] = np.abs(
            day_true - day_pred
        )

        persistence_errors[indices] = np.abs(
            day_true - day_persistence
        )

        last_index = int(indices.max())

        observations_available = (
            last_index + 1
        )

        recent_hours = (
            S03_RECENT_DAYS * 24
        )

        reference_hours = (
            S03_REFERENCE_DAYS * 24
        )

        history_needed = (
            recent_hours
            + reference_hours
        )

        # The trigger is evaluated only after enough
        # fully observed history exists.
        if (
            observations_available
            < history_needed
        ):
            continue

        recent_start = (
            observations_available
            - recent_hours
        )

        reference_start = (
            recent_start
            - reference_hours
        )

        reference_end = recent_start

        recent_xgb = xgb_errors[
            recent_start:
            observations_available
        ]

        recent_persistence = (
            persistence_errors[
                recent_start:
                observations_available
            ]
        )

        reference_xgb = xgb_errors[
            reference_start:
            reference_end
        ]

        reference_persistence = (
            persistence_errors[
                reference_start:
                reference_end
            ]
        )

        recent_skill = s03_pooled_skill_ratio(
            recent_xgb,
            recent_persistence,
        )

        reference_skill = s03_pooled_skill_ratio(
            reference_xgb,
            reference_persistence,
        )

        if (
            np.isnan(recent_skill)
            or np.isnan(reference_skill)
            or reference_skill <= 1e-12
        ):
            continue

        degradation_score = (
            recent_skill
            / reference_skill
        )

        score_at_row[last_index] = (
            degradation_score
        )

        recent_skill_at_row[last_index] = (
            recent_skill
        )

        reference_skill_at_row[last_index] = (
            reference_skill
        )

        cooldown_ok = (
            last_update_day is None
            or (
                day - last_update_day
            ).days >= S03_COOLDOWN_DAYS
        )

        if (
            degradation_score
            >= threshold
            and cooldown_ok
        ):

            # At next midnight, Target_Power for today's
            # 23:00 forecast is observed.
            #
            # We may therefore retrain through today's
            # final forecast row and use the updated model
            # for the next day's 00:00 forecast.
            training_end = validation.loc[
                last_index,
                "Time",
            ]

            retraining_data = featured[
                featured["Time"]
                <= training_end
            ].copy()

            model = s03_new_model()

            model.fit(
                retraining_data[
                    feature_columns
                ],
                retraining_data[
                    "Target_Power"
                ],
            )

            update_effective_time = (
                day
                + pd.Timedelta(days=1)
            )

            updates.append(
                {
                    "Threshold": threshold,
                    "Trigger_Day": day,
                    "Update_Effective_Time":
                        update_effective_time,
                    "Training_End":
                        training_end,
                    "Training_Rows":
                        len(retraining_data),
                    "Degradation_Score":
                        degradation_score,
                    "Recent_Skill_Ratio":
                        recent_skill,
                    "Reference_Skill_Ratio":
                        reference_skill,
                    "Recent_30d_XGB_MAE":
                        np.mean(recent_xgb),
                    "Recent_30d_Persistence_MAE":
                        np.mean(
                            recent_persistence
                        ),
                    "Reference_60d_XGB_MAE":
                        np.mean(reference_xgb),
                    "Reference_60d_Persistence_MAE":
                        np.mean(
                            reference_persistence
                        ),
                }
            )

            last_update_day = day

    stream = validation[
        [
            "Time",
            "Target_Power",
            "Power",
        ]
    ].copy()

    stream["Prediction"] = predictions
    stream["XGB_Absolute_Error"] = (
        xgb_errors
    )
    stream[
        "Persistence_Absolute_Error"
    ] = persistence_errors

    stream[
        "Degradation_Score_Daily"
    ] = score_at_row

    stream[
        "Recent_30d_Skill_Ratio"
    ] = recent_skill_at_row

    stream[
        "Reference_60d_Skill_Ratio"
    ] = reference_skill_at_row

    return (
        stream,
        pd.DataFrame(updates),
    )


# ============================================================================
# SUMMARY
# ============================================================================

def s03_summarize_stream(
    location,
    policy,
    threshold,
    stream,
    update_count,
):
    m = s03_metrics(
        stream["Target_Power"],
        stream["Prediction"],
    )

    persistence_mae = (
        stream[
            "Persistence_Absolute_Error"
        ].mean()
    )

    q4 = (
        stream["Time"]
        >= pd.Timestamp(
            "2020-10-01 00:00:00"
        )
    )

    q4_mae = (
        stream
        .loc[
            q4,
            "XGB_Absolute_Error",
        ]
        .mean()
    )

    return {
        "Location": location,
        "Policy": policy,
        "Threshold": threshold,
        "Updates": update_count,
        "MAE": m["MAE"],
        "RMSE": m["RMSE"],
        "R2": m["R2"],
        "Persistence_MAE":
            persistence_mae,
        "Q4_2020_MAE":
            q4_mae,
        "MAE_Improvement_vs_Persistence_pct":
            100
            * (
                persistence_mae
                - m["MAE"]
            )
            / persistence_mae,
    }


def s03_add_static_comparison(summary):
    summary = summary.copy()

    static = (
        summary[
            summary["Policy"]
            == "Static"
        ][
            [
                "Location",
                "MAE",
            ]
        ]
        .rename(
            columns={
                "MAE":
                    "Static_MAE"
            }
        )
    )

    summary = summary.merge(
        static,
        on="Location",
        how="left",
    )

    summary[
        "MAE_Improvement_vs_Static_pct"
    ] = (
        100
        * (
            summary["Static_MAE"]
            - summary["MAE"]
        )
        / summary["Static_MAE"]
    )

    summary[
        "MAE_Reduction_per_Update"
    ] = np.where(
        summary["Updates"] > 0,
        (
            summary["Static_MAE"]
            - summary["MAE"]
        )
        / summary["Updates"],
        np.nan,
    )

    return summary


def s03_build_threshold_aggregate(summary):
    adaptive = summary[
        summary["Policy"]
        == "SkillTrigger"
    ].copy()

    rows = []

    for threshold, g in adaptive.groupby(
        "Threshold"
    ):
        rows.append(
            {
                "Threshold": threshold,
                "Mean_MAE":
                    g["MAE"].mean(),
                "Median_MAE":
                    g["MAE"].median(),
                "Total_Updates":
                    int(g["Updates"].sum()),
                "Mean_Improvement_vs_Static_pct":
                    g[
                        "MAE_Improvement_vs_Static_pct"
                    ].mean(),
                "Mean_Q4_MAE":
                    g["Q4_2020_MAE"].mean(),
                "Updates_Location1":
                    int(
                        g.loc[
                            g["Location"]
                            == "Location1",
                            "Updates",
                        ].sum()
                    ),
                "Updates_Location2":
                    int(
                        g.loc[
                            g["Location"]
                            == "Location2",
                            "Updates",
                        ].sum()
                    ),
                "Updates_Location3":
                    int(
                        g.loc[
                            g["Location"]
                            == "Location3",
                            "Updates",
                        ].sum()
                    ),
                "Updates_Location4":
                    int(
                        g.loc[
                            g["Location"]
                            == "Location4",
                            "Updates",
                        ].sum()
                    ),
            }
        )

    out = pd.DataFrame(rows)

    # Pareto marker:
    # Configuration A dominates B if A has <= MAE and <= updates,
    # with at least one strict inequality.
    pareto = []

    for i, row in out.iterrows():

        dominated = False

        for j, other in out.iterrows():

            if i == j:
                continue

            no_worse_mae = (
                other["Mean_MAE"]
                <= row["Mean_MAE"]
            )

            no_more_updates = (
                other["Total_Updates"]
                <= row["Total_Updates"]
            )

            strictly_better_one = (
                other["Mean_MAE"]
                < row["Mean_MAE"]
                or other["Total_Updates"]
                < row["Total_Updates"]
            )

            if (
                no_worse_mae
                and no_more_updates
                and strictly_better_one
            ):
                dominated = True
                break

        pareto.append(
            not dominated
        )

    out["Pareto_Efficient"] = pareto

    return out


# ============================================================================
# FIGURES
# ============================================================================

def s03_plot_threshold_tradeoff(
    aggregate,
):
    fig, ax = plt.subplots(
        figsize=(8, 5.5)
    )

    ax.scatter(
        aggregate["Total_Updates"],
        aggregate["Mean_MAE"],
        s=70,
    )

    for _, row in aggregate.iterrows():

        ax.annotate(
            f"T={row['Threshold']:.1f}",
            (
                row["Total_Updates"],
                row["Mean_MAE"],
            ),
            xytext=(5, 5),
            textcoords="offset points",
        )

    ax.set_xlabel(
        "Total model updates across four locations"
    )

    ax.set_ylabel(
        "Mean 2020 validation MAE"
    )

    ax.set_title(
        "Selective adaptation: accuracy-update tradeoff"
    )

    fig.tight_layout()

    fig.savefig(
        S03_OUTPUT_DIR
        / "threshold_accuracy_update_tradeoff.png",
        dpi=220,
    )

    plt.close(fig)


def s03_plot_location_mae(
    summary,
    location,
):
    g = summary[
        summary["Location"]
        == location
    ].copy()

    static_mae = (
        g.loc[
            g["Policy"] == "Static",
            "MAE",
        ].iloc[0]
    )

    adaptive = (
        g[
            g["Policy"]
            == "SkillTrigger"
        ]
        .sort_values("Threshold")
    )

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.plot(
        adaptive["Threshold"],
        adaptive["MAE"],
        marker="o",
        label="Skill-trigger adaptation",
    )

    ax.axhline(
        static_mae,
        linestyle="--",
        label="Static XGBoost",
    )

    ax.set_xlabel(
        "Degradation threshold"
    )

    ax.set_ylabel(
        "2020 validation MAE"
    )

    ax.set_title(
        f"{location}: threshold sensitivity"
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        S03_OUTPUT_DIR
        / f"{location}_threshold_sensitivity.png",
        dpi=220,
    )

    plt.close(fig)


# ============================================================================
# MAIN
# ============================================================================

def s03_main():

    s03_section(
        "SKILL-TRIGGER DEVELOPMENT — 2020 ONLY"
    )

    print(
        f"Recent window: {S03_RECENT_DAYS} days"
    )
    print(
        f"Reference window: {S03_REFERENCE_DAYS} days"
    )
    print(
        f"Cooldown: {S03_COOLDOWN_DAYS} days"
    )
    print(
        f"Thresholds: {S03_THRESHOLDS}"
    )
    print(
        f"XGBoost version: "
        f"{xgboost.__version__}"
    )

    summary_rows = []
    update_parts = []
    stream_parts = []

    for location in S03_LOCATIONS:

        s03_section(location)

        raw = s03_load_raw(location)

        featured = s03_create_features(raw)

        feature_columns = [
            c
            for c in featured.columns
            if c not in {
                "Time",
                "Target_Power",
            }
        ]

        # ------------------------------------------------------------
        # STATIC
        # ------------------------------------------------------------

        (
            static_validation,
            static_pred,
            static_xgb_error,
            static_persistence_error,
        ) = s03_run_static(
            featured,
            feature_columns,
        )

        static_stream = static_validation[
            [
                "Time",
                "Target_Power",
                "Power",
            ]
        ].copy()

        static_stream[
            "Prediction"
        ] = static_pred

        static_stream[
            "XGB_Absolute_Error"
        ] = static_xgb_error

        static_stream[
            "Persistence_Absolute_Error"
        ] = static_persistence_error

        summary_rows.append(
            s03_summarize_stream(
                location=location,
                policy="Static",
                threshold=np.nan,
                stream=static_stream,
                update_count=0,
            )
        )

        # ------------------------------------------------------------
        # THRESHOLD SENSITIVITY
        # ------------------------------------------------------------

        for threshold in S03_THRESHOLDS:

            print(
                f"Running threshold "
                f"{threshold:.1f} ..."
            )

            (
                stream,
                updates,
            ) = s03_run_skill_trigger_policy(
                featured=featured,
                feature_columns=feature_columns,
                threshold=threshold,
            )

            summary_rows.append(
                s03_summarize_stream(
                    location=location,
                    policy="SkillTrigger",
                    threshold=threshold,
                    stream=stream,
                    update_count=len(updates),
                )
            )

            stream["Location"] = location
            stream["Policy"] = "SkillTrigger"
            stream["Threshold"] = threshold

            stream_parts.append(stream)

            if not updates.empty:
                updates["Location"] = location
                update_parts.append(updates)

            print(
                f"  MAE="
                f"{stream['XGB_Absolute_Error'].mean():.6f}"
                f" | updates={len(updates)}"
            )

            if not updates.empty:
                for _, u in updates.iterrows():
                    print(
                        "  update effective "
                        f"{u['Update_Effective_Time']} "
                        "| score="
                        f"{u['Degradation_Score']:.3f}"
                    )

    # ========================================================================
    # RESULTS
    # ========================================================================

    summary = pd.DataFrame(
        summary_rows
    )

    summary = s03_add_static_comparison(
        summary
    )

    aggregate = s03_build_threshold_aggregate(
        summary
    )

    if update_parts:
        updates_all = pd.concat(
            update_parts,
            ignore_index=True,
        )
    else:
        updates_all = pd.DataFrame()

    if stream_parts:
        streams_all = pd.concat(
            stream_parts,
            ignore_index=True,
        )
    else:
        streams_all = pd.DataFrame()

    # ------------------------------------------------------------------------
    # SAVE
    # ------------------------------------------------------------------------

    summary.to_csv(
        S03_OUTPUT_DIR
        / "01_skill_trigger_policy_summary_2020.csv",
        index=False,
    )

    aggregate.to_csv(
        S03_OUTPUT_DIR
        / "02_threshold_tradeoff_summary_2020.csv",
        index=False,
    )

    updates_all.to_csv(
        S03_OUTPUT_DIR
        / "03_skill_trigger_update_log_2020.csv",
        index=False,
    )

    streams_all.to_csv(
        S03_OUTPUT_DIR
        / "04_skill_trigger_error_streams_2020.csv",
        index=False,
    )

    metadata = {
        "stage":
            "skill_trigger_development_2020",
        "uses_2021_for_selection":
            False,
        "recent_days":
            S03_RECENT_DAYS,
        "reference_days":
            S03_REFERENCE_DAYS,
        "cooldown_days":
            S03_COOLDOWN_DAYS,
        "thresholds":
            S03_THRESHOLDS,
        "training_mode_after_trigger":
            "expanding",
        "forecast_horizon_hours":
            1,
        "trigger_definition":
            (
                "(30-day MAE_XGB / 30-day "
                "MAE_persistence) divided by "
                "(preceding 60-day MAE_XGB / "
                "preceding 60-day MAE_persistence)"
            ),
        "xgboost_parameters":
            S03_XGB_PARAMS,
        "python_version":
            sys.version,
        "platform":
            platform.platform(),
        "xgboost_version":
            xgboost.__version__,
        "causality_note":
            (
                "Trigger evaluated after the final "
                "forecast target of each day becomes "
                "observed. Any retraining is effective "
                "for the next day's first forecast."
            ),
    }

    with open(
        S03_OUTPUT_DIR
        / "00_skill_trigger_metadata.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2,
            default=str,
        )

    # Figures
    s03_plot_threshold_tradeoff(
        aggregate
    )

    for location in S03_LOCATIONS:
        s03_plot_location_mae(
            summary,
            location,
        )

    # ------------------------------------------------------------------------
    # PRINT
    # ------------------------------------------------------------------------

    s03_section(
        "THRESHOLD RESULTS"
    )

    print(
        summary[
            [
                "Location",
                "Policy",
                "Threshold",
                "Updates",
                "MAE",
                "RMSE",
                "Q4_2020_MAE",
                "MAE_Improvement_vs_Static_pct",
                "MAE_Reduction_per_Update",
            ]
        ]
        .sort_values(
            [
                "Location",
                "Policy",
                "Threshold",
            ],
            na_position="first",
        )
        .round(6)
        .to_string(index=False)
    )

    s03_section(
        "ACCURACY-UPDATE TRADEOFF"
    )

    print(
        aggregate
        .sort_values(
            [
                "Total_Updates",
                "Mean_MAE",
            ]
        )
        .round(6)
        .to_string(index=False)
    )

    s03_section(
        "DONE"
    )

    print(
        f"Outputs saved to:\n"
        f"{S03_OUTPUT_DIR}"
    )

    print(
        "\nIMPORTANT:\n"
        "Do not choose a threshold from 2021. "
        "The threshold will be frozen from this "
        "2020 development experiment before the "
        "final 2021 test."
    )


# ============================================================================
# STAGE 04: Untouched 2021 evaluation: static, persistence, periodic 30/90/180 d and RSDR (Tables 1-2)
# Original file: 04_final_2021_evaluation.py (code unchanged; names prefixed s04_ / S04_)
# ============================================================================
#
# 04_final_2021_evaluation.py
# ===========================
#
# FINAL untouched 2021 evaluation for the clean wind-power adaptation study.
#
# IMPORTANT
# ---------
# All method choices in this script are FROZEN BEFORE 2021 evaluation.
#
# The proposed selective trigger was developed using 2020 only:
#
#     Recent window:      30 days
#     Reference window:   preceding 60 days
#     Degradation score:
#         (recent MAE_XGB / recent MAE_persistence)
#         ------------------------------------------------
#         (reference MAE_XGB / reference MAE_persistence)
#
#     Frozen threshold:   1.5
#     Cooldown:           30 days
#     Retraining mode:    expanding history
#
# Why threshold 1.5?
# ------------------
# It was selected from the 2020 development experiment because it:
# - updated Location2 and Location3,
# - made no updates in Location1 and Location4,
# - used only 2 updates across all four locations,
# - improved Location2 and Location3 validation MAE,
# - avoided the unnecessary Location4 update produced by threshold 1.4,
# - dominated threshold 1.6 at the same update count,
# - retained Location2 adaptation that threshold 1.7 missed.
#
# NO 2021 result is used to alter this threshold or any policy setting.
#
# FINAL METHODS
# -------------
# 1. Persistence
# 2. Static XGBoost
# 3. Periodic expanding XGBoost every 30 days
# 4. Periodic expanding XGBoost every 90 days
# 5. Periodic expanding XGBoost every 180 days
# 6. Proposed skill-triggered expanding XGBoost (threshold = 1.5)
#
# INITIAL MODEL
# -------------
# All XGBoost methods start the 2021 evaluation from the same model trained on
# all eligible historical forecast rows whose targets are inside 2017-2020.
#
# The final historical training issue time is:
#     2020-12-31 22:00
#
# The 2020-12-31 23:00 forecast row is deliberately excluded from training
# because its target is Power at 2021-01-01 00:00.
#
# TEST
# ----
# Forecast issue times:
#     2021-01-01 00:00 onward
#
# Each issue-time row t forecasts:
#     Power(t+1)
#
# The last raw row has no future target and is therefore naturally excluded.
#
# CAUSALITY
# ---------
# At an issue time t:
# - Power(t) and weather/time features at t are available.
# - The model forecasts Power(t+1).
# - Once Power(t+1) is observed, the forecast error becomes available.
# - Periodic/event retraining uses only forecast rows whose targets have
#   already become observable.
#
# STATISTICS
# ----------
# In addition to MAE/RMSE/R², this script computes:
# - monthly performance,
# - rolling 7-day and 30-day MAE,
# - number and timing of model updates,
# - MAE reduction per update,
# - paired moving-block bootstrap 95% confidence intervals for differences
#   in absolute error between the proposed method and comparison methods.
#
# EXPECTED PROJECT STRUCTURE
# --------------------------
# D:\Wind Drift Paper    04_final_2021_evaluation.py
#     data        Location1.csv
#         Location2.csv
#         Location3.csv
#         Location4.csv
#
# OUTPUT
# ------
# outputs_final_2021

warnings.filterwarnings("ignore", category=FutureWarning)


# ============================================================================
# FROZEN CONFIGURATION
# ============================================================================

S04_BASE_DIR = Path(__file__).resolve().parent
S04_DATA_DIR = S04_BASE_DIR / "data"

S04_OUTPUT_DIR = S04_BASE_DIR / "outputs_final_2021"
S04_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

S04_LOCATIONS = [
    "Location1",
    "Location2",
    "Location3",
    "Location4",
]

# All training targets remain inside 2017-2020.
S04_INITIAL_TRAIN_END = pd.Timestamp(
    "2020-12-31 22:00:00"
)

S04_TEST_START = pd.Timestamp(
    "2021-01-01 00:00:00"
)

S04_TEST_END = pd.Timestamp(
    "2021-12-31 23:00:00"
)

# Frozen proposed trigger, chosen from 2020 only.
S04_RECENT_DAYS = 30
S04_REFERENCE_DAYS = 60
S04_SKILL_THRESHOLD = 1.5
S04_COOLDOWN_DAYS = 30

# Fixed comparison schedules.
S04_PERIODIC_INTERVAL_DAYS = [
    30,
    90,
    180,
]

S04_RANDOM_STATE = 42

S04_XGB_PARAMS = dict(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    min_child_weight=3,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    eval_metric="rmse",
    random_state=S04_RANDOM_STATE,
    n_jobs=-1,
)

S04_POWER_LAGS = [
    1,
    2,
    3,
    6,
    12,
    24,
    48,
    72,
    168,
]

S04_ROLL_WINDOWS = [
    3,
    6,
    12,
    24,
    48,
    168,
]

# Paired moving-block bootstrap.
S04_BOOTSTRAP_REPS = 1000
S04_BOOTSTRAP_BLOCK_HOURS = 168
S04_BOOTSTRAP_SEED = 2026


# ============================================================================
# BASIC UTILITIES
# ============================================================================

def s04_section(title):
    print("\n" + "=" * 104)
    print(title)
    print("=" * 104)


def s04_load_raw(location):
    path = S04_DATA_DIR / f"{location}.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"\nMissing dataset:\n{path}\n"
        )

    df = pd.read_csv(path)

    required = [
        "Time",
        "temperature_2m",
        "relativehumidity_2m",
        "dewpoint_2m",
        "windspeed_10m",
        "windspeed_100m",
        "winddirection_10m",
        "winddirection_100m",
        "windgusts_10m",
        "Power",
    ]

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{location}: missing columns: {missing}"
        )

    df["Time"] = pd.to_datetime(
        df["Time"],
        errors="coerce",
    )

    if df["Time"].isna().any():
        raise ValueError(
            f"{location}: invalid timestamps."
        )

    df = (
        df
        .sort_values("Time")
        .drop_duplicates(
            subset=["Time"],
            keep="first",
        )
        .reset_index(drop=True)
    )

    return df


def s04_create_features(df):
    """
    Canonical leakage-safe one-hour-ahead forecasting problem.

    Forecast issue row t:
        predictors available at time t
        target = Power(t+1)
    """
    d = df.copy()

    # Calendar
    d["hour"] = d["Time"].dt.hour
    d["day_of_week"] = (
        d["Time"].dt.dayofweek
    )
    d["month"] = d["Time"].dt.month
    d["day_of_year"] = (
        d["Time"].dt.dayofyear
    )

    d["hour_sin"] = np.sin(
        2 * np.pi * d["hour"] / 24
    )
    d["hour_cos"] = np.cos(
        2 * np.pi * d["hour"] / 24
    )

    d["day_of_week_sin"] = np.sin(
        2 * np.pi
        * d["day_of_week"]
        / 7
    )
    d["day_of_week_cos"] = np.cos(
        2 * np.pi
        * d["day_of_week"]
        / 7
    )

    d["month_sin"] = np.sin(
        2 * np.pi
        * d["month"]
        / 12
    )
    d["month_cos"] = np.cos(
        2 * np.pi
        * d["month"]
        / 12
    )

    # Historical power
    for lag in S04_POWER_LAGS:
        d[f"Power_lag_{lag}"] = (
            d["Power"].shift(lag)
        )

    shifted_power = (
        d["Power"].shift(1)
    )

    for window in S04_ROLL_WINDOWS:
        d[
            f"Power_roll_mean_{window}"
        ] = (
            shifted_power
            .rolling(window)
            .mean()
        )

        d[
            f"Power_roll_std_{window}"
        ] = (
            shifted_power
            .rolling(window)
            .std()
        )

    # Strict +1 hour target.
    d["Target_Power"] = (
        d["Power"].shift(-1)
    )

    d = (
        d
        .dropna()
        .reset_index(drop=True)
    )

    return d


def s04_feature_columns(featured):
    return [
        c
        for c in featured.columns
        if c not in {
            "Time",
            "Target_Power",
        }
    ]


def s04_new_model():
    return XGBRegressor(
        **S04_XGB_PARAMS
    )


def s04_metrics(y_true, y_pred):
    y_true = np.asarray(
        y_true,
        dtype=float,
    )
    y_pred = np.asarray(
        y_pred,
        dtype=float,
    )

    return {
        "MAE":
            mean_absolute_error(
                y_true,
                y_pred,
            ),
        "RMSE":
            mean_squared_error(
                y_true,
                y_pred,
            ) ** 0.5,
        "R2":
            r2_score(
                y_true,
                y_pred,
            ),
    }


def s04_pooled_skill_ratio(
    xgb_errors,
    persistence_errors,
):
    xgb_errors = np.asarray(
        xgb_errors,
        dtype=float,
    )

    persistence_errors = np.asarray(
        persistence_errors,
        dtype=float,
    )

    xgb_mae = float(
        np.mean(xgb_errors)
    )
    persistence_mae = float(
        np.mean(
            persistence_errors
        )
    )

    if persistence_mae <= 1e-12:
        return np.nan

    return (
        xgb_mae
        / persistence_mae
    )


def s04_prepare_train_test(featured):
    """
    Final guard against temporal leakage.

    Initial training rows end in 2020.
    Test issue times are 2021 only.
    """
    train = featured[
        featured["Time"]
        <= S04_INITIAL_TRAIN_END
    ].copy()

    test = featured[
        (featured["Time"] >= S04_TEST_START)
        & (featured["Time"] <= S04_TEST_END)
    ].copy().reset_index(drop=True)

    if train.empty or test.empty:
        raise RuntimeError(
            "Empty training or test split."
        )

    if train["Time"].max().year != 2020:
        raise RuntimeError(
            "Initial training does not end in 2020."
        )

    test_years = sorted(
        test["Time"]
        .dt.year
        .unique()
        .tolist()
    )

    if test_years != [2021]:
        raise RuntimeError(
            f"Final test must contain only 2021. "
            f"Found years={test_years}"
        )

    # Target-time guard.
    # Since target = t+1, the final training target remains
    # at or before 2020-12-31 23:00.
    max_training_target_time = (
        train["Time"].max()
        + pd.Timedelta(hours=1)
    )

    if (
        max_training_target_time.year
        != 2020
    ):
        raise RuntimeError(
            "Training target crosses into 2021."
        )

    return train, test


# ============================================================================
# POLICY IMPLEMENTATIONS
# ============================================================================

def s04_build_prediction_frame(
    location,
    policy,
    test,
    predictions,
):
    out = test[
        [
            "Time",
            "Target_Power",
            "Power",
        ]
    ].copy()

    out["Location"] = location
    out["Policy"] = policy

    out["Prediction"] = (
        np.asarray(
            predictions,
            dtype=float,
        )
    )

    out["Absolute_Error"] = np.abs(
        out["Target_Power"]
        - out["Prediction"]
    )

    out["Squared_Error"] = (
        out["Target_Power"]
        - out["Prediction"]
    ) ** 2

    return out


def s04_run_persistence(
    location,
    test,
):
    pred = (
        test["Power"]
        .to_numpy(
            dtype=float
        )
    )

    return (
        s04_build_prediction_frame(
            location=location,
            policy="Persistence",
            test=test,
            predictions=pred,
        ),
        [],
    )


def s04_run_static(
    location,
    featured,
    cols,
):
    train, test = (
        s04_prepare_train_test(
            featured
        )
    )

    model = s04_new_model()

    model.fit(
        train[cols],
        train["Target_Power"],
    )

    pred = model.predict(
        test[cols]
    )

    return (
        s04_build_prediction_frame(
            location=location,
            policy="Static",
            test=test,
            predictions=pred,
        ),
        [],
    )


def s04_run_periodic_expanding(
    location,
    featured,
    cols,
    interval_days,
):
    train, test = (
        s04_prepare_train_test(
            featured
        )
    )

    model = s04_new_model()

    model.fit(
        train[cols],
        train["Target_Power"],
    )

    predictions = np.full(
        len(test),
        np.nan,
        dtype=float,
    )

    updates = []

    policy = (
        f"PeriodicExpanding_"
        f"{interval_days}d"
    )

    next_update = (
        S04_TEST_START
        + pd.Timedelta(
            days=interval_days
        )
    )

    start_idx = 0

    while start_idx < len(test):

        current_time = (
            test.loc[
                start_idx,
                "Time",
            ]
        )

        # If an update is due before forecasting this row,
        # all targets through issue time t-1 are known.
        if current_time >= next_update:

            last_observed_issue = (
                current_time
                - pd.Timedelta(
                    hours=1
                )
            )

            retrain = featured[
                featured["Time"]
                <= last_observed_issue
            ].copy()

            model = s04_new_model()

            model.fit(
                retrain[cols],
                retrain[
                    "Target_Power"
                ],
            )

            updates.append(
                {
                    "Location":
                        location,
                    "Policy":
                        policy,
                    "Update_Effective_Time":
                        current_time,
                    "Trigger_Time":
                        pd.NaT,
                    "Training_Start":
                        retrain[
                            "Time"
                        ].min(),
                    "Training_End":
                        retrain[
                            "Time"
                        ].max(),
                    "Training_Rows":
                        len(retrain),
                    "Degradation_Score":
                        np.nan,
                }
            )

            next_update = (
                current_time
                + pd.Timedelta(
                    days=interval_days
                )
            )

        # Predict in one segment until the next update boundary.
        boundary = min(
            next_update,
            S04_TEST_END
            + pd.Timedelta(
                hours=1
            ),
        )

        mask = (
            (test["Time"] >= current_time)
            & (test["Time"] < boundary)
        )

        segment_indices = (
            test.index[mask]
            .to_numpy()
        )

        if len(segment_indices) == 0:
            # Safety fallback.
            segment_indices = np.array(
                [start_idx]
            )

        predictions[
            segment_indices
        ] = model.predict(
            test.loc[
                segment_indices,
                cols,
            ]
        )

        start_idx = (
            int(
                segment_indices.max()
            )
            + 1
        )

    return (
        s04_build_prediction_frame(
            location=location,
            policy=policy,
            test=test,
            predictions=predictions,
        ),
        updates,
    )


def s04_run_skill_trigger(
    location,
    featured,
    cols,
):
    """
    Frozen proposed method.

    The trigger is recomputed from the proposed model's OWN
    observed 2021 error stream, exactly as in the 2020
    development script.
    """
    train, test = (
        s04_prepare_train_test(
            featured
        )
    )

    model = s04_new_model()

    model.fit(
        train[cols],
        train["Target_Power"],
    )

    n = len(test)

    predictions = np.full(
        n,
        np.nan,
        dtype=float,
    )

    xgb_errors = np.full(
        n,
        np.nan,
        dtype=float,
    )

    persistence_errors = np.full(
        n,
        np.nan,
        dtype=float,
    )

    degradation_scores = np.full(
        n,
        np.nan,
        dtype=float,
    )

    recent_skills = np.full(
        n,
        np.nan,
        dtype=float,
    )

    reference_skills = np.full(
        n,
        np.nan,
        dtype=float,
    )

    updates = []
    last_update_day = None

    policy = (
        "SkillTrigger_T1.5_"
        "Expanding"
    )

    day_groups = list(
        test.groupby(
            test["Time"]
            .dt.floor("D")
        ).groups.items()
    )

    recent_hours = (
        S04_RECENT_DAYS * 24
    )

    reference_hours = (
        S04_REFERENCE_DAYS * 24
    )

    required_hours = (
        recent_hours
        + reference_hours
    )

    for day, idx_values in day_groups:

        indices = np.array(
            list(idx_values),
            dtype=int,
        )

        # Forecast full day using model available at day start.
        day_pred = model.predict(
            test.loc[
                indices,
                cols,
            ]
        )

        predictions[
            indices
        ] = day_pred

        day_true = (
            test.loc[
                indices,
                "Target_Power",
            ]
            .to_numpy(
                dtype=float
            )
        )

        day_persistence = (
            test.loc[
                indices,
                "Power",
            ]
            .to_numpy(
                dtype=float
            )
        )

        xgb_errors[
            indices
        ] = np.abs(
            day_true
            - day_pred
        )

        persistence_errors[
            indices
        ] = np.abs(
            day_true
            - day_persistence
        )

        last_idx = int(
            indices.max()
        )

        observations_available = (
            last_idx + 1
        )

        if (
            observations_available
            < required_hours
        ):
            continue

        recent_start = (
            observations_available
            - recent_hours
        )

        reference_start = (
            recent_start
            - reference_hours
        )

        reference_end = (
            recent_start
        )

        recent_xgb = (
            xgb_errors[
                recent_start:
                observations_available
            ]
        )

        recent_persistence = (
            persistence_errors[
                recent_start:
                observations_available
            ]
        )

        reference_xgb = (
            xgb_errors[
                reference_start:
                reference_end
            ]
        )

        reference_persistence = (
            persistence_errors[
                reference_start:
                reference_end
            ]
        )

        recent_skill = (
            s04_pooled_skill_ratio(
                recent_xgb,
                recent_persistence,
            )
        )

        reference_skill = (
            s04_pooled_skill_ratio(
                reference_xgb,
                reference_persistence,
            )
        )

        if (
            np.isnan(recent_skill)
            or np.isnan(
                reference_skill
            )
            or reference_skill
            <= 1e-12
        ):
            continue

        degradation_score = (
            recent_skill
            / reference_skill
        )

        degradation_scores[
            last_idx
        ] = degradation_score

        recent_skills[
            last_idx
        ] = recent_skill

        reference_skills[
            last_idx
        ] = reference_skill

        cooldown_ok = (
            last_update_day is None
            or (
                day
                - last_update_day
            ).days
            >= S04_COOLDOWN_DAYS
        )

        if (
            degradation_score
            >= S04_SKILL_THRESHOLD
            and cooldown_ok
        ):

            # Trigger evaluated after this day's final target
            # becomes known. Update is effective next midnight.
            training_end = (
                test.loc[
                    last_idx,
                    "Time",
                ]
            )

            retrain = featured[
                featured["Time"]
                <= training_end
            ].copy()

            model = s04_new_model()

            model.fit(
                retrain[cols],
                retrain[
                    "Target_Power"
                ],
            )

            update_effective = (
                day
                + pd.Timedelta(
                    days=1
                )
            )

            updates.append(
                {
                    "Location":
                        location,
                    "Policy":
                        policy,
                    "Update_Effective_Time":
                        update_effective,
                    "Trigger_Time":
                        training_end,
                    "Training_Start":
                        retrain[
                            "Time"
                        ].min(),
                    "Training_End":
                        retrain[
                            "Time"
                        ].max(),
                    "Training_Rows":
                        len(retrain),
                    "Degradation_Score":
                        degradation_score,
                    "Recent_30d_Skill_Ratio":
                        recent_skill,
                    "Reference_60d_Skill_Ratio":
                        reference_skill,
                    "Recent_30d_XGB_MAE":
                        float(
                            np.mean(
                                recent_xgb
                            )
                        ),
                    "Recent_30d_Persistence_MAE":
                        float(
                            np.mean(
                                recent_persistence
                            )
                        ),
                    "Reference_60d_XGB_MAE":
                        float(
                            np.mean(
                                reference_xgb
                            )
                        ),
                    "Reference_60d_Persistence_MAE":
                        float(
                            np.mean(
                                reference_persistence
                            )
                        ),
                }
            )

            last_update_day = day

    out = s04_build_prediction_frame(
        location=location,
        policy=policy,
        test=test,
        predictions=predictions,
    )

    out[
        "Persistence_Absolute_Error"
    ] = persistence_errors

    out[
        "Degradation_Score_Daily"
    ] = degradation_scores

    out[
        "Recent_30d_Skill_Ratio"
    ] = recent_skills

    out[
        "Reference_60d_Skill_Ratio"
    ] = reference_skills

    return (
        out,
        updates,
    )


# ============================================================================
# SUMMARIES
# ============================================================================

def s04_policy_summary(
    predictions,
    update_log,
):
    update_counts = {}

    if not update_log.empty:
        update_counts = (
            update_log
            .groupby(
                [
                    "Location",
                    "Policy",
                ]
            )
            .size()
            .to_dict()
        )

    rows = []

    for (
        location,
        policy,
    ), g in predictions.groupby(
        [
            "Location",
            "Policy",
        ]
    ):

        m = s04_metrics(
            g["Target_Power"],
            g["Prediction"],
        )

        rows.append(
            {
                "Location":
                    location,
                "Policy":
                    policy,
                "N":
                    len(g),
                "Updates":
                    int(
                        update_counts.get(
                            (
                                location,
                                policy,
                            ),
                            0,
                        )
                    ),
                "MAE":
                    m["MAE"],
                "RMSE":
                    m["RMSE"],
                "R2":
                    m["R2"],
            }
        )

    summary = pd.DataFrame(
        rows
    )

    static_lookup = (
        summary[
            summary["Policy"]
            == "Static"
        ][
            [
                "Location",
                "MAE",
            ]
        ]
        .rename(
            columns={
                "MAE":
                    "Static_MAE"
            }
        )
    )

    summary = summary.merge(
        static_lookup,
        on="Location",
        how="left",
    )

    summary[
        "MAE_Improvement_vs_Static_pct"
    ] = (
        100
        * (
            summary[
                "Static_MAE"
            ]
            - summary[
                "MAE"
            ]
        )
        / summary[
            "Static_MAE"
        ]
    )

    summary[
        "MAE_Reduction_per_Update"
    ] = np.where(
        summary["Updates"] > 0,
        (
            summary[
                "Static_MAE"
            ]
            - summary[
                "MAE"
            ]
        )
        / summary[
            "Updates"
        ],
        np.nan,
    )

    return summary


def s04_aggregate_summary(
    summary,
):
    rows = []

    for policy, g in summary.groupby(
        "Policy"
    ):

        rows.append(
            {
                "Policy":
                    policy,
                "Mean_MAE":
                    g["MAE"].mean(),
                "Median_MAE":
                    g["MAE"].median(),
                "Mean_RMSE":
                    g["RMSE"].mean(),
                "Total_Updates":
                    int(
                        g["Updates"].sum()
                    ),
                "Mean_Improvement_vs_Static_pct":
                    g[
                        "MAE_Improvement_vs_Static_pct"
                    ].mean(),
                "Locations_Improved_vs_Static":
                    int(
                        (
                            g[
                                "MAE_Improvement_vs_Static_pct"
                            ]
                            > 0
                        ).sum()
                    ),
            }
        )

    out = pd.DataFrame(
        rows
    )

    # Pareto efficiency in error-vs-update space.
    pareto = []

    for i, row in out.iterrows():

        dominated = False

        for j, other in out.iterrows():

            if i == j:
                continue

            if (
                other["Mean_MAE"]
                <= row["Mean_MAE"]
                and other["Total_Updates"]
                <= row[
                    "Total_Updates"
                ]
                and (
                    other["Mean_MAE"]
                    < row["Mean_MAE"]
                    or other[
                        "Total_Updates"
                    ]
                    < row[
                        "Total_Updates"
                    ]
                )
            ):
                dominated = True
                break

        pareto.append(
            not dominated
        )

    out[
        "Pareto_Efficient"
    ] = pareto

    return out


def s04_monthly_summary(
    predictions,
):
    d = predictions.copy()

    d["Month"] = (
        d["Time"]
        .dt.to_period("M")
        .astype(str)
    )

    return (
        d.groupby(
            [
                "Location",
                "Policy",
                "Month",
            ]
        )
        .agg(
            MAE=(
                "Absolute_Error",
                "mean",
            ),
            RMSE=(
                "Squared_Error",
                lambda x:
                    np.sqrt(
                        np.mean(x)
                    ),
            ),
            Mean_Actual_Power=(
                "Target_Power",
                "mean",
            ),
        )
        .reset_index()
    )


def s04_add_rolling_metrics(
    predictions,
):
    parts = []

    for (
        location,
        policy,
    ), g in predictions.groupby(
        [
            "Location",
            "Policy",
        ]
    ):

        g = (
            g
            .sort_values("Time")
            .copy()
        )

        g["MAE_7d"] = (
            g[
                "Absolute_Error"
            ]
            .rolling(
                168,
                min_periods=168,
            )
            .mean()
        )

        g["MAE_30d"] = (
            g[
                "Absolute_Error"
            ]
            .rolling(
                720,
                min_periods=720,
            )
            .mean()
        )

        parts.append(
            g[
                [
                    "Location",
                    "Policy",
                    "Time",
                    "Absolute_Error",
                    "MAE_7d",
                    "MAE_30d",
                ]
            ]
        )

    return pd.concat(
        parts,
        ignore_index=True,
    )


# ============================================================================
# MOVING-BLOCK BOOTSTRAP
# ============================================================================

def s04_moving_block_bootstrap_mean_difference(
    error_a,
    error_b,
    reps=S04_BOOTSTRAP_REPS,
    block_hours=S04_BOOTSTRAP_BLOCK_HOURS,
    seed=S04_BOOTSTRAP_SEED,
):
    """
    Difference = mean(error_a - error_b)

    If comparing Static vs Proposed:
        positive difference means Proposed has lower error.
    """
    a = np.asarray(
        error_a,
        dtype=float,
    )

    b = np.asarray(
        error_b,
        dtype=float,
    )

    if len(a) != len(b):
        raise ValueError(
            "Paired error arrays differ in length."
        )

    d = a - b
    n = len(d)

    observed = float(
        np.mean(d)
    )

    if n < block_hours:
        raise ValueError(
            "Series shorter than bootstrap block."
        )

    rng = np.random.default_rng(
        seed
    )

    max_start = (
        n
        - block_hours
    )

    n_blocks = int(
        np.ceil(
            n
            / block_hours
        )
    )

    boot = np.empty(
        reps,
        dtype=float,
    )

    for r in range(reps):

        starts = rng.integers(
            0,
            max_start + 1,
            size=n_blocks,
        )

        sampled = np.concatenate(
            [
                d[
                    s:
                    s + block_hours
                ]
                for s in starts
            ]
        )[:n]

        boot[r] = np.mean(
            sampled
        )

    ci_low, ci_high = (
        np.quantile(
            boot,
            [
                0.025,
                0.975,
            ],
        )
    )

    # One-sided empirical probability of no improvement
    # when positive means method B is better.
    p_no_improvement = (
        np.mean(
            boot <= 0
        )
    )

    return {
        "Mean_Error_Difference":
            observed,
        "CI95_Low":
            float(ci_low),
        "CI95_High":
            float(ci_high),
        "Bootstrap_P_NoImprovement":
            float(
                p_no_improvement
            ),
    }


def s04_bootstrap_comparisons(
    predictions,
):
    proposed = (
        "SkillTrigger_T1.5_"
        "Expanding"
    )

    comparisons = [
        ("Static", proposed),
        (
            "PeriodicExpanding_30d",
            proposed,
        ),
        (
            "PeriodicExpanding_90d",
            proposed,
        ),
        (
            "PeriodicExpanding_180d",
            proposed,
        ),
    ]

    rows = []

    for location in S04_LOCATIONS:

        loc = predictions[
            predictions["Location"]
            == location
        ]

        for method_a, method_b in comparisons:

            a = (
                loc[
                    loc["Policy"]
                    == method_a
                ]
                .sort_values("Time")
            )

            b = (
                loc[
                    loc["Policy"]
                    == method_b
                ]
                .sort_values("Time")
            )

            if (
                len(a) == 0
                or len(b) == 0
            ):
                continue

            if not np.array_equal(
                a["Time"].to_numpy(),
                b["Time"].to_numpy(),
            ):
                raise RuntimeError(
                    f"Unaligned times for "
                    f"{location}: "
                    f"{method_a} vs {method_b}"
                )

            result = (
                s04_moving_block_bootstrap_mean_difference(
                    error_a=a[
                        "Absolute_Error"
                    ],
                    error_b=b[
                        "Absolute_Error"
                    ],
                    seed=(
                        S04_BOOTSTRAP_SEED
                        + S04_LOCATIONS.index(
                            location
                        )
                    ),
                )
            )

            rows.append(
                {
                    "Location":
                        location,
                    "Method_A":
                        method_a,
                    "Method_B":
                        method_b,
                    "Interpretation":
                        (
                            "Positive difference = "
                            "Method_B lower MAE"
                        ),
                    **result,
                }
            )

    return pd.DataFrame(
        rows
    )


# ============================================================================
# FIGURES
# ============================================================================

def s04_plot_monthly(
    monthly,
    location,
):
    g = monthly[
        monthly["Location"]
        == location
    ].copy()

    methods = [
        "Static",
        "PeriodicExpanding_30d",
        "PeriodicExpanding_90d",
        "PeriodicExpanding_180d",
        "SkillTrigger_T1.5_Expanding",
        "Persistence",
    ]

    fig, ax = plt.subplots(
        figsize=(12, 5.5)
    )

    for method in methods:

        m = g[
            g["Policy"]
            == method
        ].copy()

        if m.empty:
            continue

        m["Month_dt"] = pd.to_datetime(
            m["Month"]
        )

        ax.plot(
            m["Month_dt"],
            m["MAE"],
            marker="o",
            linewidth=1.4,
            label=method,
        )

    ax.set_title(
        f"{location}: final 2021 monthly MAE"
    )

    ax.set_xlabel(
        "Month"
    )

    ax.set_ylabel(
        "MAE"
    )

    ax.legend(
        fontsize=8,
        ncol=2,
    )

    fig.tight_layout()

    fig.savefig(
        S04_OUTPUT_DIR
        / f"{location}_2021_monthly_MAE.png",
        dpi=220,
    )

    plt.close(fig)


def s04_plot_tradeoff(
    aggregate,
):
    g = aggregate[
        aggregate["Policy"]
        != "Persistence"
    ].copy()

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    ax.scatter(
        g["Total_Updates"],
        g["Mean_MAE"],
        s=85,
    )

    for _, row in g.iterrows():

        short = (
            row["Policy"]
            .replace(
                "PeriodicExpanding_",
                "Periodic "
            )
            .replace(
                "SkillTrigger_T1.5_Expanding",
                "Proposed"
            )
        )

        ax.annotate(
            short,
            (
                row["Total_Updates"],
                row["Mean_MAE"],
            ),
            xytext=(6, 5),
            textcoords="offset points",
            fontsize=8,
        )

    ax.set_xlabel(
        "Total 2021 model updates "
        "across four locations"
    )

    ax.set_ylabel(
        "Mean 2021 MAE"
    )

    ax.set_title(
        "Final accuracy-update efficiency tradeoff"
    )

    fig.tight_layout()

    fig.savefig(
        S04_OUTPUT_DIR
        / "2021_accuracy_update_tradeoff.png",
        dpi=220,
    )

    plt.close(fig)


# ============================================================================
# MAIN
# ============================================================================

def s04_main():

    s04_section(
        "FINAL 2021 EVALUATION — FROZEN METHOD"
    )

    print(
        "NO parameter selection is performed "
        "in this script."
    )

    print(
        f"Frozen skill threshold: "
        f"{S04_SKILL_THRESHOLD}"
    )

    print(
        f"Recent/reference windows: "
        f"{S04_RECENT_DAYS}d / "
        f"{S04_REFERENCE_DAYS}d"
    )

    print(
        f"Cooldown: "
        f"{S04_COOLDOWN_DAYS}d"
    )

    print(
        f"Initial training issue-time cutoff: "
        f"{S04_INITIAL_TRAIN_END}"
    )

    print(
        f"XGBoost version: "
        f"{xgboost.__version__}"
    )

    prediction_parts = []
    all_updates = []

    for location in S04_LOCATIONS:

        s04_section(location)

        raw = s04_load_raw(
            location
        )

        featured = (
            s04_create_features(
                raw
            )
        )

        cols = s04_feature_columns(
            featured
        )

        train, test = (
            s04_prepare_train_test(
                featured
            )
        )

        print(
            f"Initial train rows: "
            f"{len(train):,}"
        )

        print(
            f"2021 test rows: "
            f"{len(test):,}"
        )

        print(
            f"Features: "
            f"{len(cols)}"
        )

        # ------------------------------------------------------------
        # Persistence
        # ------------------------------------------------------------

        print(
            "Running Persistence ..."
        )

        pred, updates = (
            s04_run_persistence(
                location,
                test,
            )
        )

        prediction_parts.append(
            pred
        )

        # ------------------------------------------------------------
        # Static
        # ------------------------------------------------------------

        print(
            "Running Static XGBoost ..."
        )

        pred, updates = (
            s04_run_static(
                location,
                featured,
                cols,
            )
        )

        prediction_parts.append(
            pred
        )

        # ------------------------------------------------------------
        # Periodic comparators
        # ------------------------------------------------------------

        for days in S04_PERIODIC_INTERVAL_DAYS:

            print(
                f"Running periodic "
                f"{days}d ..."
            )

            pred, updates = (
                s04_run_periodic_expanding(
                    location=location,
                    featured=featured,
                    cols=cols,
                    interval_days=days,
                )
            )

            prediction_parts.append(
                pred
            )

            all_updates.extend(
                updates
            )

        # ------------------------------------------------------------
        # Frozen proposed method
        # ------------------------------------------------------------

        print(
            "Running proposed frozen "
            "skill trigger ..."
        )

        pred, updates = (
            s04_run_skill_trigger(
                location=location,
                featured=featured,
                cols=cols,
            )
        )

        prediction_parts.append(
            pred
        )

        all_updates.extend(
            updates
        )

        print(
            f"Proposed updates in "
            f"{location}: "
            f"{len(updates)}"
        )

        for u in updates:
            print(
                "  update effective "
                f"{u['Update_Effective_Time']} "
                "| score="
                f"{u['Degradation_Score']:.3f}"
            )

    # ========================================================================
    # COMBINE AND ANALYZE
    # ========================================================================

    predictions = pd.concat(
        prediction_parts,
        ignore_index=True,
    )

    if all_updates:
        update_log = pd.DataFrame(
            all_updates
        )
    else:
        update_log = pd.DataFrame(
            columns=[
                "Location",
                "Policy",
                "Update_Effective_Time",
                "Trigger_Time",
                "Training_Start",
                "Training_End",
                "Training_Rows",
                "Degradation_Score",
            ]
        )

    summary = s04_policy_summary(
        predictions,
        update_log,
    )

    aggregate = s04_aggregate_summary(
        summary
    )

    monthly = s04_monthly_summary(
        predictions
    )

    rolling = s04_add_rolling_metrics(
        predictions
    )

    bootstrap = s04_bootstrap_comparisons(
        predictions
    )

    # ========================================================================
    # SAVE
    # ========================================================================

    summary.to_csv(
        S04_OUTPUT_DIR
        / "01_final_2021_policy_summary.csv",
        index=False,
    )

    aggregate.to_csv(
        S04_OUTPUT_DIR
        / "02_final_2021_aggregate_summary.csv",
        index=False,
    )

    update_log.to_csv(
        S04_OUTPUT_DIR
        / "03_final_2021_update_log.csv",
        index=False,
    )

    monthly.to_csv(
        S04_OUTPUT_DIR
        / "04_final_2021_monthly_metrics.csv",
        index=False,
    )

    rolling.to_csv(
        S04_OUTPUT_DIR
        / "05_final_2021_rolling_metrics.csv",
        index=False,
    )

    bootstrap.to_csv(
        S04_OUTPUT_DIR
        / "06_final_2021_block_bootstrap.csv",
        index=False,
    )

    predictions.to_csv(
        S04_OUTPUT_DIR
        / "07_final_2021_predictions.csv",
        index=False,
    )

    metadata = {
        "stage":
            "final_2021_evaluation",
        "method_frozen_before_2021":
            True,
        "uses_2021_for_parameter_selection":
            False,
        "initial_train_end_issue_time":
            str(
                S04_INITIAL_TRAIN_END
            ),
        "test_start_issue_time":
            str(
                S04_TEST_START
            ),
        "forecast_horizon_hours":
            1,
        "proposed_method": {
            "name":
                "relative_skill_degradation_trigger",
            "threshold":
                S04_SKILL_THRESHOLD,
            "recent_days":
                S04_RECENT_DAYS,
            "reference_days":
                S04_REFERENCE_DAYS,
            "cooldown_days":
                S04_COOLDOWN_DAYS,
            "retraining_mode":
                "expanding",
            "selection_basis":
                (
                    "Frozen from 2020 development. "
                    "Threshold 1.5 made two updates "
                    "(L2/L3), zero updates in L1/L4, "
                    "and provided the preferred "
                    "accuracy-update tradeoff."
                ),
        },
        "periodic_comparators_days":
            S04_PERIODIC_INTERVAL_DAYS,
        "bootstrap": {
            "repetitions":
                S04_BOOTSTRAP_REPS,
            "block_hours":
                S04_BOOTSTRAP_BLOCK_HOURS,
            "seed":
                S04_BOOTSTRAP_SEED,
        },
        "xgboost_parameters":
            S04_XGB_PARAMS,
        "python_version":
            sys.version,
        "platform":
            platform.platform(),
        "xgboost_version":
            xgboost.__version__,
    }

    with open(
        S04_OUTPUT_DIR
        / "00_final_2021_metadata.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2,
            default=str,
        )

    # Figures
    for location in S04_LOCATIONS:
        s04_plot_monthly(
            monthly,
            location,
        )

    s04_plot_tradeoff(
        aggregate
    )

    # ========================================================================
    # PRINT FINAL RESULTS
    # ========================================================================

    s04_section(
        "FINAL 2021 RESULTS BY LOCATION"
    )

    print(
        summary[
            [
                "Location",
                "Policy",
                "Updates",
                "MAE",
                "RMSE",
                "R2",
                "MAE_Improvement_vs_Static_pct",
                "MAE_Reduction_per_Update",
            ]
        ]
        .sort_values(
            [
                "Location",
                "MAE",
            ]
        )
        .round(6)
        .to_string(index=False)
    )

    s04_section(
        "FINAL CROSS-LOCATION "
        "ACCURACY-UPDATE TRADEOFF"
    )

    print(
        aggregate
        .sort_values(
            [
                "Mean_MAE",
                "Total_Updates",
            ]
        )
        .round(6)
        .to_string(index=False)
    )

    s04_section(
        "PAIRED 168-HOUR BLOCK BOOTSTRAP"
    )

    print(
        bootstrap
        .round(6)
        .to_string(index=False)
    )

    s04_section(
        "FINAL TEST COMPLETE"
    )

    print(
        f"All outputs saved to:\n"
        f"{S04_OUTPUT_DIR}"
    )

    print(
        "\nIMPORTANT:\n"
        "Do not modify the frozen threshold "
        "after seeing these 2021 results. "
        "Any additional analysis must be "
        "reported as post-hoc/sensitivity "
        "analysis rather than model selection."
    )


# ============================================================================
# STAGE 05: Post-hoc analysis of stage 04 outputs: update efficiency, cross-location bootstrap, update utility (Table 3)
# Original file: 05_posthoc_final_analysis.py (code unchanged; names prefixed s05_ / S05_)
# ============================================================================
#
# 05_posthoc_final_analysis.py
# ============================
#
# Post-hoc analysis of the FROZEN 2021 final experiment.
#
# This script DOES NOT retrain any model and DOES NOT tune any parameter.
# It only reads outputs_final_2021 generated by 04_final_2021_evaluation.py.
#
# It quantifies:
# 1. Accuracy gain captured per update relative to periodic retraining.
# 2. Percentage reduction in update count.
# 3. Stratified moving-block bootstrap across the four locations.
# 4. Post-update utility over 7-, 30-, and 60-day windows.
# 5. Whether each proposed update was productive relative to the static model.
#
# Expected:
#     outputs_final_2021/
#         01_final_2021_policy_summary.csv
#         02_final_2021_aggregate_summary.csv
#         03_final_2021_update_log.csv
#         07_final_2021_predictions.csv
#
# Output:
#     outputs_posthoc_final/

S05_BASE_DIR = Path(__file__).resolve().parent
S05_INPUT_DIR = S05_BASE_DIR / "outputs_final_2021"
S05_OUTPUT_DIR = S05_BASE_DIR / "outputs_posthoc_final"
S05_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

S05_PROPOSED = "SkillTrigger_T1.5_Expanding"
S05_STATIC = "Static"

S05_COMPARATORS = [
    "PeriodicExpanding_30d",
    "PeriodicExpanding_90d",
    "PeriodicExpanding_180d",
]

S05_BLOCK_HOURS = 168
S05_BOOTSTRAP_REPS = 5000
S05_SEED = 20260831


def s05_moving_block_sample_mean(d, rng, block_hours=S05_BLOCK_HOURS):
    d = np.asarray(d, dtype=float)
    n = len(d)
    n_blocks = int(np.ceil(n / block_hours))
    starts = rng.integers(
        0,
        n - block_hours + 1,
        size=n_blocks,
    )
    sampled = np.concatenate(
        [d[s:s + block_hours] for s in starts]
    )[:n]
    return float(np.mean(sampled))


def s05_load():
    required = [
        "01_final_2021_policy_summary.csv",
        "02_final_2021_aggregate_summary.csv",
        "03_final_2021_update_log.csv",
        "07_final_2021_predictions.csv",
    ]
    for f in required:
        if not (S05_INPUT_DIR / f).exists():
            raise FileNotFoundError(
                f"Missing {S05_INPUT_DIR / f}. "
                "Run 04_final_2021_evaluation.py first."
            )

    summary = pd.read_csv(
        S05_INPUT_DIR / "01_final_2021_policy_summary.csv"
    )
    aggregate = pd.read_csv(
        S05_INPUT_DIR / "02_final_2021_aggregate_summary.csv"
    )
    updates = pd.read_csv(
        S05_INPUT_DIR / "03_final_2021_update_log.csv",
        parse_dates=["Update_Effective_Time", "Trigger_Time"],
    )
    predictions = pd.read_csv(
        S05_INPUT_DIR / "07_final_2021_predictions.csv",
        parse_dates=["Time"],
    )
    return summary, aggregate, updates, predictions


def s05_efficiency_table(aggregate):
    static_mae = float(
        aggregate.loc[
            aggregate["Policy"] == S05_STATIC,
            "Mean_MAE",
        ].iloc[0]
    )
    proposed_row = aggregate.loc[
        aggregate["Policy"] == S05_PROPOSED
    ].iloc[0]
    proposed_mae = float(proposed_row["Mean_MAE"])
    proposed_updates = int(proposed_row["Total_Updates"])

    proposed_gain = static_mae - proposed_mae

    rows = []
    for comparator in S05_COMPARATORS:
        row = aggregate.loc[
            aggregate["Policy"] == comparator
        ].iloc[0]

        comp_mae = float(row["Mean_MAE"])
        comp_updates = int(row["Total_Updates"])
        comp_gain = static_mae - comp_mae

        rows.append(
            {
                "Comparator": comparator,
                "Static_Mean_MAE": static_mae,
                "Proposed_Mean_MAE": proposed_mae,
                "Comparator_Mean_MAE": comp_mae,
                "Proposed_Updates": proposed_updates,
                "Comparator_Updates": comp_updates,
                "Update_Reduction_pct": (
                    100.0
                    * (comp_updates - proposed_updates)
                    / comp_updates
                ),
                "Proposed_Gain_vs_Static": proposed_gain,
                "Comparator_Gain_vs_Static": comp_gain,
                "Fraction_of_Comparator_Gain_Captured": (
                    proposed_gain / comp_gain
                    if comp_gain > 0
                    else np.nan
                ),
                "Proposed_MAE_minus_Comparator_MAE": (
                    proposed_mae - comp_mae
                ),
            }
        )
    return pd.DataFrame(rows)


def s05_stratified_bootstrap(predictions):
    comparisons = [
        S05_STATIC,
        *S05_COMPARATORS,
    ]

    rng = np.random.default_rng(S05_SEED)
    rows = []

    locations = sorted(predictions["Location"].unique())

    for method_a in comparisons:
        loc_diffs = []

        for loc in locations:
            a = predictions[
                (predictions["Location"] == loc)
                & (predictions["Policy"] == method_a)
            ].sort_values("Time")

            b = predictions[
                (predictions["Location"] == loc)
                & (predictions["Policy"] == S05_PROPOSED)
            ].sort_values("Time")

            if not np.array_equal(
                a["Time"].to_numpy(),
                b["Time"].to_numpy(),
            ):
                raise RuntimeError(
                    f"Unaligned prediction times for {loc}: "
                    f"{method_a} vs {S05_PROPOSED}"
                )

            loc_diffs.append(
                a["Absolute_Error"].to_numpy()
                - b["Absolute_Error"].to_numpy()
            )

        observed = float(
            np.mean([d.mean() for d in loc_diffs])
        )

        boot = np.empty(S05_BOOTSTRAP_REPS)

        for r in range(S05_BOOTSTRAP_REPS):
            means = [
                s05_moving_block_sample_mean(d, rng)
                for d in loc_diffs
            ]
            boot[r] = np.mean(means)

        lo, hi = np.quantile(boot, [0.025, 0.975])

        rows.append(
            {
                "Method_A": method_a,
                "Method_B": S05_PROPOSED,
                "Interpretation": (
                    "Positive difference means Proposed has lower MAE"
                ),
                "Observed_Mean_Error_Difference": observed,
                "CI95_Low": float(lo),
                "CI95_High": float(hi),
                "Bootstrap_P_Difference_le_0": float(
                    np.mean(boot <= 0)
                ),
            }
        )

    return pd.DataFrame(rows)


def s05_update_utility(updates, predictions):
    proposed_updates = updates[
        updates["Policy"] == S05_PROPOSED
    ].copy()

    rows = []

    for _, u in proposed_updates.iterrows():
        loc = u["Location"]
        t = u["Update_Effective_Time"]

        static = predictions[
            (predictions["Location"] == loc)
            & (predictions["Policy"] == S05_STATIC)
        ][["Time", "Absolute_Error"]].rename(
            columns={"Absolute_Error": "Static_Error"}
        )

        proposed = predictions[
            (predictions["Location"] == loc)
            & (predictions["Policy"] == S05_PROPOSED)
        ][["Time", "Absolute_Error"]].rename(
            columns={"Absolute_Error": "Proposed_Error"}
        )

        merged = static.merge(proposed, on="Time")

        for days in [7, 30, 60]:
            end = t + pd.Timedelta(days=days)
            g = merged[
                (merged["Time"] >= t)
                & (merged["Time"] < end)
            ]

            static_mae = g["Static_Error"].mean()
            proposed_mae = g["Proposed_Error"].mean()
            improvement = static_mae - proposed_mae

            rows.append(
                {
                    "Location": loc,
                    "Update_Effective_Time": t,
                    "Degradation_Score": u["Degradation_Score"],
                    "Evaluation_Window_Days": days,
                    "Static_MAE": static_mae,
                    "Proposed_MAE": proposed_mae,
                    "MAE_Reduction": improvement,
                    "Improvement_pct": (
                        100.0 * improvement / static_mae
                        if static_mae > 0
                        else np.nan
                    ),
                    "Productive_Update": bool(improvement > 0),
                }
            )

    return pd.DataFrame(rows)


def s05_main():
    summary, aggregate, updates, predictions = s05_load()

    efficiency = s05_efficiency_table(aggregate)
    bootstrap = s05_stratified_bootstrap(predictions)
    utility = s05_update_utility(updates, predictions)

    efficiency.to_csv(
        S05_OUTPUT_DIR / "01_accuracy_update_efficiency.csv",
        index=False,
    )
    bootstrap.to_csv(
        S05_OUTPUT_DIR / "02_cross_location_block_bootstrap.csv",
        index=False,
    )
    utility.to_csv(
        S05_OUTPUT_DIR / "03_proposed_update_utility.csv",
        index=False,
    )

    proposed_updates = updates[
        updates["Policy"] == S05_PROPOSED
    ]

    metadata = {
        "posthoc_only": True,
        "changes_to_frozen_model": False,
        "proposed_policy": S05_PROPOSED,
        "bootstrap_repetitions": S05_BOOTSTRAP_REPS,
        "bootstrap_block_hours": S05_BLOCK_HOURS,
        "bootstrap_seed": S05_SEED,
        "number_of_proposed_updates": int(len(proposed_updates)),
        "important_note": (
            "These analyses quantify the already-frozen 2021 result. "
            "They do not select or alter the threshold, windows, cooldown, "
            "model, or update policy."
        ),
    }

    with open(
        S05_OUTPUT_DIR / "00_posthoc_metadata.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(metadata, f, indent=2)

    print("\nACCURACY / UPDATE EFFICIENCY")
    print(efficiency.round(6).to_string(index=False))

    print("\nCROSS-LOCATION 168-HOUR BLOCK BOOTSTRAP")
    print(bootstrap.round(6).to_string(index=False))

    print("\nPROPOSED UPDATE UTILITY")
    print(utility.round(6).to_string(index=False))

    print(f"\nSaved to: {S05_OUTPUT_DIR}")


# ============================================================================
# STAGE 06: Frozen ADWIN comparators and update-fitting cost in 2021 (Table 1 ADWIN rows and fit times)
# Original file: 06_frozen_adwin_and_cost_2021.py (code unchanged; names prefixed s06_ / S06_)
# ============================================================================
#
# 06_frozen_adwin_and_cost_2021.py
# ================================
#
# LAST model-run experiment for the wind-power selective-adaptation study.
#
# This script does NOT tune the proposed method.
#
# It performs two post-freeze tasks on the untouched/final 2021 period:
#
# A. Frozen ADWIN comparators
#    1. ADWIN on absolute XGBoost error
#    2. ADWIN on excess error relative to persistence
#
#    Frozen ADWIN settings come from the 2020 detector-development stage:
#        delta = 0.002
#        causal direction gate = recent 7-day mean > preceding 7-day mean
#        retraining = expanding history
#
#    When an accepted ADWIN alarm occurs after forecast row t is evaluated,
#    the model is retrained using forecast rows through t and becomes active
#    for the next forecast row.
#
# B. Computational/update cost
#    Re-runs the final policies and records actual model FIT time on this
#    machine:
#        Static
#        Periodic expanding 30d
#        Periodic expanding 90d
#        Periodic expanding 180d
#        Proposed frozen skill trigger
#        ADWIN absolute-error comparator
#        ADWIN excess-error comparator
#
# The proposed method remains frozen:
#     recent window = 30 days
#     reference window = preceding 60 days
#     threshold = 1.5
#     cooldown = 30 days
#     retraining = expanding history
#
# NO 2021 result is used for parameter selection.
#
# Expected project structure
# --------------------------
# D:\Wind Drift Paper    06_frozen_adwin_and_cost_2021.py
#     data        Location1.csv
#         Location2.csv
#         Location3.csv
#         Location4.csv
#
# Output
# ------
# outputs_final_cost_adwin

try:
    import river
    from river.drift import ADWIN
except ImportError as exc:
    raise SystemExit(
        "\nRiver is required.\n"
        "Install with:\n"
        "    py -m pip install river\n"
    ) from exc

warnings.filterwarnings("ignore", category=FutureWarning)


# ============================================================================
# FROZEN CONFIGURATION
# ============================================================================

S06_BASE_DIR = Path(__file__).resolve().parent
S06_DATA_DIR = S06_BASE_DIR / "data"

S06_OUTPUT_DIR = S06_BASE_DIR / "outputs_final_cost_adwin"
S06_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

S06_LOCATIONS = ["Location1", "Location2", "Location3", "Location4"]

S06_INITIAL_TRAIN_END = pd.Timestamp("2020-12-31 22:00:00")
S06_TEST_START = pd.Timestamp("2021-01-01 00:00:00")
S06_TEST_END = pd.Timestamp("2021-12-31 23:00:00")

S06_RANDOM_STATE = 42

S06_XGB_PARAMS = dict(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    min_child_weight=3,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    eval_metric="rmse",
    random_state=S06_RANDOM_STATE,
    n_jobs=-1,
)

S06_POWER_LAGS = [1, 2, 3, 6, 12, 24, 48, 72, 168]
S06_ROLL_WINDOWS = [3, 6, 12, 24, 48, 168]

S06_PERIODIC_DAYS = [30, 90, 180]

# Proposed method: frozen from 2020 only.
S06_RECENT_DAYS = 30
S06_REFERENCE_DAYS = 60
S06_SKILL_THRESHOLD = 1.5
S06_SKILL_COOLDOWN_DAYS = 30

# ADWIN comparator: frozen from 2020 detector development.
S06_ADWIN_DELTA = 0.002
S06_ADWIN_DIRECTION_HOURS = 168

S06_PROPOSED_NAME = "SkillTrigger_T1.5_Expanding"
S06_ADWIN_RAW_NAME = "ADWIN_AbsoluteError_Expanding"
S06_ADWIN_EXCESS_NAME = "ADWIN_ExcessError_Expanding"


# ============================================================================
# DATA / FEATURES
# ============================================================================

def s06_section(title):
    print("\n" + "=" * 104)
    print(title)
    print("=" * 104)


def s06_load_raw(location):
    path = S06_DATA_DIR / f"{location}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}")

    df = pd.read_csv(path)
    df["Time"] = pd.to_datetime(df["Time"], errors="coerce")

    if df["Time"].isna().any():
        raise ValueError(f"{location}: invalid timestamps.")

    return (
        df.sort_values("Time")
        .drop_duplicates(subset=["Time"], keep="first")
        .reset_index(drop=True)
    )


def s06_create_features(df):
    d = df.copy()

    d["hour"] = d["Time"].dt.hour
    d["day_of_week"] = d["Time"].dt.dayofweek
    d["month"] = d["Time"].dt.month
    d["day_of_year"] = d["Time"].dt.dayofyear

    d["hour_sin"] = np.sin(2 * np.pi * d["hour"] / 24)
    d["hour_cos"] = np.cos(2 * np.pi * d["hour"] / 24)
    d["day_of_week_sin"] = np.sin(2 * np.pi * d["day_of_week"] / 7)
    d["day_of_week_cos"] = np.cos(2 * np.pi * d["day_of_week"] / 7)
    d["month_sin"] = np.sin(2 * np.pi * d["month"] / 12)
    d["month_cos"] = np.cos(2 * np.pi * d["month"] / 12)

    for lag in S06_POWER_LAGS:
        d[f"Power_lag_{lag}"] = d["Power"].shift(lag)

    shifted = d["Power"].shift(1)

    for window in S06_ROLL_WINDOWS:
        d[f"Power_roll_mean_{window}"] = shifted.rolling(window).mean()
        d[f"Power_roll_std_{window}"] = shifted.rolling(window).std()

    d["Target_Power"] = d["Power"].shift(-1)

    return d.dropna().reset_index(drop=True)


def s06_get_feature_columns(featured):
    return [
        c for c in featured.columns
        if c not in {"Time", "Target_Power"}
    ]


def s06_prepare(featured):
    train = featured[
        featured["Time"] <= S06_INITIAL_TRAIN_END
    ].copy()

    test = featured[
        (featured["Time"] >= S06_TEST_START)
        & (featured["Time"] <= S06_TEST_END)
    ].copy().reset_index(drop=True)

    if sorted(test["Time"].dt.year.unique().tolist()) != [2021]:
        raise RuntimeError("Test guard failed: expected 2021 only.")

    max_training_target_time = (
        train["Time"].max() + pd.Timedelta(hours=1)
    )

    if max_training_target_time.year != 2020:
        raise RuntimeError("Training target leaks into 2021.")

    return train, test


def s06_make_model():
    return XGBRegressor(**S06_XGB_PARAMS)


def s06_timed_fit(X, y):
    model = s06_make_model()
    start = time.perf_counter()
    model.fit(X, y)
    seconds = time.perf_counter() - start
    return model, seconds


def s06_metric_dict(y, pred):
    return {
        "MAE": mean_absolute_error(y, pred),
        "RMSE": mean_squared_error(y, pred) ** 0.5,
        "R2": r2_score(y, pred),
    }


def s06_skill_ratio(xgb_errors, persistence_errors):
    x = float(np.mean(xgb_errors))
    p = float(np.mean(persistence_errors))
    return np.nan if p <= 1e-12 else x / p


# ============================================================================
# COST LOGGING
# ============================================================================

def s06_fit_log_row(
    location,
    policy,
    fit_type,
    effective_time,
    train_df,
    seconds,
    trigger_time=pd.NaT,
    trigger_value=np.nan,
):
    return {
        "Location": location,
        "Policy": policy,
        "Fit_Type": fit_type,
        "Update_Effective_Time": effective_time,
        "Trigger_Time": trigger_time,
        "Trigger_Value": trigger_value,
        "Training_Start": train_df["Time"].min(),
        "Training_End": train_df["Time"].max(),
        "Training_Rows": len(train_df),
        "Fit_Seconds": seconds,
    }


# ============================================================================
# STATIC
# ============================================================================

def s06_run_static(location, featured, cols):
    train, test = s06_prepare(featured)

    wall_start = time.perf_counter()
    model, initial_fit = s06_timed_fit(
        train[cols], train["Target_Power"]
    )

    pred = model.predict(test[cols])
    wall_seconds = time.perf_counter() - wall_start

    logs = [
        s06_fit_log_row(
            location=location,
            policy="Static",
            fit_type="Initial",
            effective_time=S06_TEST_START,
            train_df=train,
            seconds=initial_fit,
        )
    ]

    return test, pred, logs, wall_seconds


# ============================================================================
# PERIODIC EXPANDING
# ============================================================================

def s06_run_periodic(location, featured, cols, days):
    train, test = s06_prepare(featured)
    policy = f"PeriodicExpanding_{days}d"

    wall_start = time.perf_counter()

    model, initial_fit = s06_timed_fit(
        train[cols], train["Target_Power"]
    )

    logs = [
        s06_fit_log_row(
            location,
            policy,
            "Initial",
            S06_TEST_START,
            train,
            initial_fit,
        )
    ]

    pred = np.full(len(test), np.nan, dtype=float)

    next_update = S06_TEST_START + pd.Timedelta(days=days)
    start_idx = 0

    while start_idx < len(test):
        current_time = test.loc[start_idx, "Time"]

        if current_time >= next_update:
            last_observed_issue = current_time - pd.Timedelta(hours=1)

            retrain = featured[
                featured["Time"] <= last_observed_issue
            ].copy()

            model, fit_seconds = s06_timed_fit(
                retrain[cols],
                retrain["Target_Power"],
            )

            logs.append(
                s06_fit_log_row(
                    location,
                    policy,
                    "Update",
                    current_time,
                    retrain,
                    fit_seconds,
                )
            )

            next_update = current_time + pd.Timedelta(days=days)

        boundary = min(
            next_update,
            S06_TEST_END + pd.Timedelta(hours=1),
        )

        mask = (
            (test["Time"] >= current_time)
            & (test["Time"] < boundary)
        )
        idx = test.index[mask].to_numpy()

        if len(idx) == 0:
            idx = np.array([start_idx])

        pred[idx] = model.predict(test.loc[idx, cols])
        start_idx = int(idx.max()) + 1

    wall_seconds = time.perf_counter() - wall_start

    return test, pred, logs, wall_seconds


# ============================================================================
# FROZEN PROPOSED SKILL TRIGGER
# ============================================================================

def s06_run_skill_trigger(location, featured, cols):
    train, test = s06_prepare(featured)

    wall_start = time.perf_counter()

    model, initial_fit = s06_timed_fit(
        train[cols], train["Target_Power"]
    )

    logs = [
        s06_fit_log_row(
            location,
            S06_PROPOSED_NAME,
            "Initial",
            S06_TEST_START,
            train,
            initial_fit,
        )
    ]

    n = len(test)
    pred = np.full(n, np.nan)
    xgb_err = np.full(n, np.nan)
    pers_err = np.full(n, np.nan)

    last_update_day = None

    recent_hours = S06_RECENT_DAYS * 24
    reference_hours = S06_REFERENCE_DAYS * 24
    required = recent_hours + reference_hours

    day_groups = list(
        test.groupby(test["Time"].dt.floor("D")).groups.items()
    )

    for day, values in day_groups:
        idx = np.array(list(values), dtype=int)

        day_pred = model.predict(test.loc[idx, cols])
        pred[idx] = day_pred

        actual = test.loc[idx, "Target_Power"].to_numpy(dtype=float)
        persistence = test.loc[idx, "Power"].to_numpy(dtype=float)

        xgb_err[idx] = np.abs(actual - day_pred)
        pers_err[idx] = np.abs(actual - persistence)

        last_idx = int(idx.max())
        available = last_idx + 1

        if available < required:
            continue

        recent_start = available - recent_hours
        reference_start = recent_start - reference_hours
        reference_end = recent_start

        recent_skill = s06_skill_ratio(
            xgb_err[recent_start:available],
            pers_err[recent_start:available],
        )
        reference_skill = s06_skill_ratio(
            xgb_err[reference_start:reference_end],
            pers_err[reference_start:reference_end],
        )

        if (
            np.isnan(recent_skill)
            or np.isnan(reference_skill)
            or reference_skill <= 1e-12
        ):
            continue

        score = recent_skill / reference_skill

        cooldown_ok = (
            last_update_day is None
            or (day - last_update_day).days >= S06_SKILL_COOLDOWN_DAYS
        )

        if score >= S06_SKILL_THRESHOLD and cooldown_ok:
            training_end = test.loc[last_idx, "Time"]

            retrain = featured[
                featured["Time"] <= training_end
            ].copy()

            model, fit_seconds = s06_timed_fit(
                retrain[cols],
                retrain["Target_Power"],
            )

            effective_time = day + pd.Timedelta(days=1)

            logs.append(
                s06_fit_log_row(
                    location=location,
                    policy=S06_PROPOSED_NAME,
                    fit_type="Update",
                    effective_time=effective_time,
                    train_df=retrain,
                    seconds=fit_seconds,
                    trigger_time=training_end,
                    trigger_value=score,
                )
            )

            last_update_day = day

    wall_seconds = time.perf_counter() - wall_start

    return test, pred, logs, wall_seconds


# ============================================================================
# FROZEN ADWIN COMPARATOR
# ============================================================================

def s06_direction_check(signal_history):
    """
    At the current alarm, compare the most recent 168 signal values
    with the immediately preceding 168 values.

    Higher signal = worse performance for both comparator signals.
    """
    w = S06_ADWIN_DIRECTION_HOURS

    if len(signal_history) < 2 * w:
        return False, np.nan, np.nan

    previous = np.asarray(signal_history[-2 * w:-w], dtype=float)
    recent = np.asarray(signal_history[-w:], dtype=float)

    previous_mean = float(np.mean(previous))
    recent_mean = float(np.mean(recent))

    return (
        recent_mean > previous_mean,
        previous_mean,
        recent_mean,
    )


def s06_run_adwin(location, featured, cols, signal_type):
    train, test = s06_prepare(featured)

    if signal_type == "absolute_error":
        policy = S06_ADWIN_RAW_NAME
    elif signal_type == "excess_error":
        policy = S06_ADWIN_EXCESS_NAME
    else:
        raise ValueError(signal_type)

    wall_start = time.perf_counter()

    model, initial_fit = s06_timed_fit(
        train[cols], train["Target_Power"]
    )

    logs = [
        s06_fit_log_row(
            location,
            policy,
            "Initial",
            S06_TEST_START,
            train,
            initial_fit,
        )
    ]

    detector = ADWIN(delta=S06_ADWIN_DELTA)

    pred = np.full(len(test), np.nan, dtype=float)
    signal_history = []
    raw_alarm_count = 0
    accepted_alarm_count = 0

    alarm_rows = []

    for i in range(len(test)):
        row = test.iloc[i]

        p = float(
            model.predict(
                test.loc[[i], cols]
            )[0]
        )

        pred[i] = p

        actual = float(row["Target_Power"])
        persistence = float(row["Power"])

        xgb_abs = abs(actual - p)
        pers_abs = abs(actual - persistence)

        if signal_type == "absolute_error":
            signal = xgb_abs
        else:
            signal = xgb_abs - pers_abs

        signal_history.append(signal)

        detector.update(float(signal))

        if detector.drift_detected:
            raw_alarm_count += 1

            accepted, previous_mean, recent_mean = s06_direction_check(
                signal_history
            )

            alarm_rows.append(
                {
                    "Location": location,
                    "Policy": policy,
                    "Issue_Time": row["Time"],
                    "Signal": signal_type,
                    "Raw_ADWIN_Alarm_Number": raw_alarm_count,
                    "Accepted_Degradation_Alarm": bool(accepted),
                    "Previous_7d_Mean": previous_mean,
                    "Recent_7d_Mean": recent_mean,
                    "Signal_At_Alarm": signal,
                }
            )

            if accepted:
                accepted_alarm_count += 1

                # Target for issue time t becomes known at t+1.
                # Update is therefore effective for the next issue row.
                training_end = row["Time"]

                retrain = featured[
                    featured["Time"] <= training_end
                ].copy()

                model, fit_seconds = s06_timed_fit(
                    retrain[cols],
                    retrain["Target_Power"],
                )

                effective_time = (
                    row["Time"] + pd.Timedelta(hours=1)
                )

                logs.append(
                    s06_fit_log_row(
                        location=location,
                        policy=policy,
                        fit_type="Update",
                        effective_time=effective_time,
                        train_df=retrain,
                        seconds=fit_seconds,
                        trigger_time=row["Time"],
                        trigger_value=signal,
                    )
                )

                # Do not manually reset ADWIN. Its internal window has
                # already adapted to the detected change. This keeps the
                # comparator definition closest to the detector-development
                # stage and avoids introducing another untuned parameter.

    wall_seconds = time.perf_counter() - wall_start

    return (
        test,
        pred,
        logs,
        wall_seconds,
        alarm_rows,
    )


# ============================================================================
# SUMMARIES
# ============================================================================

def s06_summarize_run(
    location,
    policy,
    test,
    pred,
    logs,
    wall_seconds,
):
    m = s06_metric_dict(
        test["Target_Power"],
        pred,
    )

    initial_seconds = sum(
        x["Fit_Seconds"]
        for x in logs
        if x["Fit_Type"] == "Initial"
    )

    update_seconds = sum(
        x["Fit_Seconds"]
        for x in logs
        if x["Fit_Type"] == "Update"
    )

    updates = sum(
        x["Fit_Type"] == "Update"
        for x in logs
    )

    return {
        "Location": location,
        "Policy": policy,
        "Updates": updates,
        "MAE": m["MAE"],
        "RMSE": m["RMSE"],
        "R2": m["R2"],
        "Initial_Fit_Seconds": initial_seconds,
        "Update_Fit_Seconds": update_seconds,
        "Total_Fit_Seconds": initial_seconds + update_seconds,
        "Mean_Update_Fit_Seconds": (
            update_seconds / updates
            if updates > 0
            else np.nan
        ),
        "Policy_Wall_Seconds": wall_seconds,
    }


def s06_add_static_comparison(summary):
    static = (
        summary[
            summary["Policy"] == "Static"
        ][["Location", "MAE"]]
        .rename(columns={"MAE": "Static_MAE"})
    )

    out = summary.merge(static, on="Location", how="left")

    out["MAE_Improvement_vs_Static_pct"] = (
        100
        * (out["Static_MAE"] - out["MAE"])
        / out["Static_MAE"]
    )

    return out


def s06_aggregate(summary):
    out = (
        summary.groupby("Policy")
        .agg(
            Mean_MAE=("MAE", "mean"),
            Total_Updates=("Updates", "sum"),
            Total_Update_Fit_Seconds=("Update_Fit_Seconds", "sum"),
            Total_All_Fit_Seconds=("Total_Fit_Seconds", "sum"),
            Mean_Policy_Wall_Seconds=("Policy_Wall_Seconds", "mean"),
            Mean_Improvement_vs_Static_pct=(
                "MAE_Improvement_vs_Static_pct",
                "mean",
            ),
        )
        .reset_index()
    )

    # Compare update burden with monthly periodic retraining.
    monthly = out[
        out["Policy"] == "PeriodicExpanding_30d"
    ]

    if len(monthly) == 1:
        monthly_updates = float(monthly["Total_Updates"].iloc[0])
        monthly_update_seconds = float(
            monthly["Total_Update_Fit_Seconds"].iloc[0]
        )

        out["Update_Count_Reduction_vs_30d_pct"] = (
            100
            * (monthly_updates - out["Total_Updates"])
            / monthly_updates
        )

        if monthly_update_seconds > 0:
            out["Update_Fit_Time_Reduction_vs_30d_pct"] = (
                100
                * (
                    monthly_update_seconds
                    - out["Total_Update_Fit_Seconds"]
                )
                / monthly_update_seconds
            )
        else:
            out["Update_Fit_Time_Reduction_vs_30d_pct"] = np.nan

    return out


# ============================================================================
# MAIN
# ============================================================================

def s06_main():
    s06_section("FROZEN ADWIN COMPARATOR + COMPUTATIONAL COST — FINAL 2021")

    print("No parameter tuning occurs in this script.")
    print(f"Frozen proposed threshold: {S06_SKILL_THRESHOLD}")
    print(f"Frozen ADWIN delta: {S06_ADWIN_DELTA}")
    print(f"River version: {river.__version__}")
    print(f"XGBoost version: {xgboost.__version__}")

    summary_rows = []
    fit_logs = []
    alarm_rows = []

    for location in S06_LOCATIONS:
        s06_section(location)

        raw = s06_load_raw(location)
        featured = s06_create_features(raw)
        cols = s06_get_feature_columns(featured)

        # STATIC
        print("Timing Static ...")
        test, pred, logs, wall = s06_run_static(
            location, featured, cols
        )
        summary_rows.append(
            s06_summarize_run(
                location, "Static", test, pred, logs, wall
            )
        )
        fit_logs.extend(logs)

        # PERIODIC
        for days in S06_PERIODIC_DAYS:
            policy = f"PeriodicExpanding_{days}d"
            print(f"Timing {policy} ...")

            test, pred, logs, wall = s06_run_periodic(
                location, featured, cols, days
            )

            summary_rows.append(
                s06_summarize_run(
                    location, policy, test, pred, logs, wall
                )
            )
            fit_logs.extend(logs)

        # PROPOSED
        print(f"Timing {S06_PROPOSED_NAME} ...")
        test, pred, logs, wall = s06_run_skill_trigger(
            location, featured, cols
        )

        summary_rows.append(
            s06_summarize_run(
                location,
                S06_PROPOSED_NAME,
                test,
                pred,
                logs,
                wall,
            )
        )
        fit_logs.extend(logs)

        # ADWIN RAW
        print(f"Running/timing {S06_ADWIN_RAW_NAME} ...")
        (
            test,
            pred,
            logs,
            wall,
            alarms,
        ) = s06_run_adwin(
            location,
            featured,
            cols,
            signal_type="absolute_error",
        )

        summary_rows.append(
            s06_summarize_run(
                location,
                S06_ADWIN_RAW_NAME,
                test,
                pred,
                logs,
                wall,
            )
        )
        fit_logs.extend(logs)
        alarm_rows.extend(alarms)

        # ADWIN EXCESS
        print(f"Running/timing {S06_ADWIN_EXCESS_NAME} ...")
        (
            test,
            pred,
            logs,
            wall,
            alarms,
        ) = s06_run_adwin(
            location,
            featured,
            cols,
            signal_type="excess_error",
        )

        summary_rows.append(
            s06_summarize_run(
                location,
                S06_ADWIN_EXCESS_NAME,
                test,
                pred,
                logs,
                wall,
            )
        )
        fit_logs.extend(logs)
        alarm_rows.extend(alarms)

    summary = pd.DataFrame(summary_rows)
    summary = s06_add_static_comparison(summary)

    fit_df = pd.DataFrame(fit_logs)
    alarm_df = pd.DataFrame(alarm_rows)

    agg = s06_aggregate(summary)

    summary.to_csv(
        S06_OUTPUT_DIR / "01_final_cost_policy_summary.csv",
        index=False,
    )

    agg.to_csv(
        S06_OUTPUT_DIR / "02_final_cost_aggregate.csv",
        index=False,
    )

    fit_df.to_csv(
        S06_OUTPUT_DIR / "03_fit_time_log.csv",
        index=False,
    )

    alarm_df.to_csv(
        S06_OUTPUT_DIR / "04_frozen_adwin_alarm_log.csv",
        index=False,
    )

    metadata = {
        "post_freeze_analysis": True,
        "changes_to_proposed_method": False,
        "uses_2021_for_tuning": False,
        "proposed": {
            "threshold": S06_SKILL_THRESHOLD,
            "recent_days": S06_RECENT_DAYS,
            "reference_days": S06_REFERENCE_DAYS,
            "cooldown_days": S06_SKILL_COOLDOWN_DAYS,
            "retraining": "expanding",
        },
        "adwin_comparators": {
            "delta": S06_ADWIN_DELTA,
            "direction_gate_hours": S06_ADWIN_DIRECTION_HOURS,
            "signals": ["absolute_error", "excess_error"],
            "retraining": "expanding",
            "manual_detector_reset_after_update": False,
        },
        "timing_definition": (
            "Fit_Seconds measures XGBoost model.fit wall-clock time. "
            "Update_Fit_Seconds excludes the common initial fit. "
            "Policy_Wall_Seconds includes fitting, prediction, and trigger logic."
        ),
        "xgboost_parameters": S06_XGB_PARAMS,
        "python_version": sys.version,
        "platform": platform.platform(),
        "xgboost_version": xgboost.__version__,
        "river_version": river.__version__,
    }

    with open(
        S06_OUTPUT_DIR / "00_final_cost_metadata.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(metadata, f, indent=2, default=str)

    s06_section("FINAL POLICY + COST SUMMARY")
    print(
        summary[
            [
                "Location",
                "Policy",
                "Updates",
                "MAE",
                "RMSE",
                "MAE_Improvement_vs_Static_pct",
                "Update_Fit_Seconds",
                "Total_Fit_Seconds",
                "Policy_Wall_Seconds",
            ]
        ]
        .sort_values(["Location", "MAE"])
        .round(6)
        .to_string(index=False)
    )

    s06_section("CROSS-LOCATION AGGREGATE")
    print(
        agg.sort_values(["Mean_MAE", "Total_Updates"])
        .round(6)
        .to_string(index=False)
    )

    if not alarm_df.empty:
        s06_section("FROZEN ADWIN ALARMS")
        print(
            alarm_df[
                [
                    "Location",
                    "Policy",
                    "Issue_Time",
                    "Accepted_Degradation_Alarm",
                    "Previous_7d_Mean",
                    "Recent_7d_Mean",
                ]
            ].to_string(index=False)
        )
    else:
        s06_section("FROZEN ADWIN ALARMS")
        print("No ADWIN alarms detected.")

    s06_section("DONE")
    print(f"Outputs saved to:\n{S06_OUTPUT_DIR}")
    print(
        "\nThis is the last planned model-run experiment. "
        "Do not change the frozen proposed threshold after these results."
    )


# ============================================================================
# RUNNER
# ============================================================================

STAGE_MAINS = {
    "03": s03_main,   # 2020 development: RSDR threshold selection (paper Section 3.2)
    "04": s04_main,   # Untouched 2021 evaluation: static, persistence, periodic 30/90/180 d and RSDR (Tables 1-2)
    "05": s05_main,   # Post-hoc analysis of stage 04 outputs: update efficiency, cross-location bootstrap, update utility (Table 3)
    "06": s06_main,   # Frozen ADWIN comparators and update-fitting cost in 2021 (Table 1 ADWIN rows and fit times)
}


if __name__ == "__main__":
    selected = sys.argv[1:] or list(STAGE_MAINS)
    unknown = [s for s in selected if s not in STAGE_MAINS]
    if unknown:
        raise SystemExit(f"Unknown stage(s): {unknown}. Choose from {list(STAGE_MAINS)}.")
    for stage in selected:
        print(f"\n########## STAGE {stage} ##########\n", flush=True)
        STAGE_MAINS[stage]()
