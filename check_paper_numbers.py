"""Check every number reported in the paper against the stage output files.

Usage: python check_paper_numbers.py RESULTS_DIR
RESULTS_DIR contains the four stage folders written by rsdr_paper_code.py.
Writes paper_number_check.csv in RESULTS_DIR and prints a summary.
"""
import sys
from pathlib import Path
import pandas as pd

R = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
s03 = R / "outputs_skill_trigger_development"
s04 = R / "outputs_final_2021"
s05 = R / "outputs_posthoc_final"
s06 = R / "outputs_final_cost_adwin"

thr = pd.read_csv(s03 / "02_threshold_tradeoff_summary_2020.csv").set_index("Threshold")
agg = pd.read_csv(s04 / "02_final_2021_aggregate_summary.csv").set_index("Policy")
loc = pd.read_csv(s04 / "01_final_2021_policy_summary.csv")
boot = pd.read_csv(s05 / "02_cross_location_block_bootstrap.csv")
util = pd.read_csv(s05 / "03_proposed_update_utility.csv")
cost = pd.read_csv(s06 / "02_final_cost_aggregate.csv").set_index("Policy")
costloc = pd.read_csv(s06 / "01_final_cost_policy_summary.csv")
alarms = pd.read_csv(s06 / "04_frozen_adwin_alarm_log.csv")
updates = pd.read_csv(s04 / "03_final_2021_update_log.csv")

RSDR, P30, P90, P180 = ("SkillTrigger_T1.5_Expanding", "PeriodicExpanding_30d",
                        "PeriodicExpanding_90d", "PeriodicExpanding_180d")
ABS, EXC = "ADWIN_AbsoluteError_Expanding", "ADWIN_ExcessError_Expanding"
rows = []


def check(where, item, paper, value, digits):
    rows.append(dict(where=where, item=item, paper=paper, from_results=round(value, digits),
                     match=round(value, digits) == round(paper, digits)))


# Section 3.2: threshold selection on 2020
for t, m, u in [(1.4, 0.007132, 3), (1.5, 0.007136, 2), (1.6, 0.007148, 2), (1.7, 0.007232, 1)]:
    check("Sec. 3.2", f"2020 mean MAE, T={t}", m, thr.loc[t, "Mean_MAE"], 6)
    check("Sec. 3.2", f"2020 updates, T={t}", u, thr.loc[t, "Total_Updates"], 0)
check("Sec. 3.2", "T=1.5 MAE above T=1.4 (%)", 0.0615,
      100 * (thr.loc[1.5, "Mean_MAE"] / thr.loc[1.4, "Mean_MAE"] - 1), 4)

# Table 1
static = agg.loc["Static", "Mean_MAE"]
def mean_rmse_adwin(p):
    return costloc.loc[costloc.Policy == p, "RMSE"].mean()
table1 = [("Static", agg.loc["Static", "Mean_MAE"], agg.loc["Static", "Mean_RMSE"], 0, 0.0,
           6.937, 12.105, 0, 0.00),
          ("Persistence", agg.loc["Persistence", "Mean_MAE"], agg.loc["Persistence", "Mean_RMSE"], 0, 0.0,
           21.646, 32.326, 0, 0.00),
          ("Periodic 180 d", agg.loc[P180, "Mean_MAE"], agg.loc[P180, "Mean_RMSE"], agg.loc[P180, "Total_Updates"],
           cost.loc[P180, "Total_Update_Fit_Seconds"], 6.870, 12.062, 8, 65.94),
          ("Periodic 90 d", agg.loc[P90, "Mean_MAE"], agg.loc[P90, "Mean_RMSE"], agg.loc[P90, "Total_Updates"],
           cost.loc[P90, "Total_Update_Fit_Seconds"], 6.853, 12.032, 16, 132.11),
          ("Periodic 30 d", agg.loc[P30, "Mean_MAE"], agg.loc[P30, "Mean_RMSE"], agg.loc[P30, "Total_Updates"],
           cost.loc[P30, "Total_Update_Fit_Seconds"], 6.585, 11.435, 48, 397.42),
          ("ADWIN abs.", cost.loc[ABS, "Mean_MAE"], mean_rmse_adwin(ABS), cost.loc[ABS, "Total_Updates"],
           cost.loc[ABS, "Total_Update_Fit_Seconds"], 6.783, 11.719, 1, 7.91),
          ("ADWIN excess", cost.loc[EXC, "Mean_MAE"], mean_rmse_adwin(EXC), cost.loc[EXC, "Total_Updates"],
           cost.loc[EXC, "Total_Update_Fit_Seconds"], 6.937, 12.105, 0, 0.00),
          ("RSDR (T = 1.5)", agg.loc[RSDR, "Mean_MAE"], agg.loc[RSDR, "Mean_RMSE"], agg.loc[RSDR, "Total_Updates"],
           cost.loc[RSDR, "Total_Update_Fit_Seconds"], 6.771, 11.701, 3, 32.21)]
gains = {"Static": 0.00, "Persistence": -212.05, "Periodic 180 d": 0.96, "Periodic 90 d": 1.21,
         "Periodic 30 d": 5.06, "ADWIN abs.": 2.21, "ADWIN excess": 0.00, "RSDR (T = 1.5)": 2.38}
for name, mae, rmse, n, fit, p_mae, p_rmse, p_n, p_fit in table1:
    check("Table 1", f"{name} MAE x1e-3", p_mae, mae * 1e3, 3)
    check("Table 1", f"{name} RMSE x1e-3", p_rmse, rmse * 1e3, 3)
    check("Table 1", f"{name} gain (%)", gains[name], 100 * (static - mae) / static, 2)
    check("Table 1", f"{name} refits", p_n, n, 0)
    check("Table 1", f"{name} update fit (s)", p_fit, fit, 2)

# Table 2
paper2 = {1: (7.443, 7.443, 0.00, 0.96, 0), 2: (5.696, 5.696, 0.00, 4.11, 0),
          3: (8.596, 7.935, 7.69, 11.84, 3), 4: (6.012, 6.012, 0.00, 1.35, 0)}
for k, (ps, pr, pg, p30g, pu) in paper2.items():
    L = loc[loc.Location == f"Location{k}"].set_index("Policy")
    st, rs, p30 = L.loc["Static", "MAE"], L.loc[RSDR, "MAE"], L.loc[P30, "MAE"]
    check("Table 2", f"Location {k} static MAE x1e-3", ps, st * 1e3, 3)
    check("Table 2", f"Location {k} RSDR MAE x1e-3", pr, rs * 1e3, 3)
    check("Table 2", f"Location {k} RSDR gain (%)", pg, 100 * (st - rs) / st, 2)
    check("Table 2", f"Location {k} 30-day gain (%)", p30g, 100 * (st - p30) / st, 2)
    check("Table 2", f"Location {k} RSDR refits", pu, L.loc[RSDR, "Updates"], 0)
    check("Table 2", f"Location {k} 30-day refits", 12, L.loc[P30, "Updates"], 0)

# Table 3: 30 days after each RSDR update (Location 3)
u30 = util[(util.Evaluation_Window_Days == 30)].set_index("Update_Effective_Time")
for d, (ps, pr, pg) in {"2021-07-04": (2.021, 1.453, 28.11), "2021-08-03": (3.601, 3.277, 9.00),
                        "2021-10-28": (23.867, 18.746, 21.46)}.items():
    check("Table 3", f"{d} static MAE x1e-3", ps, u30.loc[d, "Static_MAE"] * 1e3, 3)
    check("Table 3", f"{d} RSDR MAE x1e-3", pr, u30.loc[d, "Proposed_MAE"] * 1e3, 3)
    check("Table 3", f"{d} gain (%)", pg, u30.loc[d, "Improvement_pct"], 2)

# Text (abstract, Sections 4-6)
b = boot.set_index("Method_A")
check("Sec. 4.1", "static minus RSDR MAE difference", 0.000165, b.loc["Static", "Observed_Mean_Error_Difference"], 6)
check("Sec. 4.1", "95% CI lower", 0.0000714, b.loc["Static", "CI95_Low"], 7)
check("Sec. 4.1", "95% CI upper", 0.000276, b.loc["Static", "CI95_High"], 6)
check("Sec. 4.1", "RSDR minus 30-day MAE gap", 0.000186, -b.loc[P30, "Observed_Mean_Error_Difference"], 6)
check("Sec. 4.1", "RSDR minus 30-day CI lower", 0.000111, -b.loc[P30, "CI95_High"], 6)
check("Sec. 4.1", "RSDR minus 30-day CI upper", 0.000278, -b.loc[P30, "CI95_Low"], 6)
p30m = agg.loc[P30, "Mean_MAE"]; rsm = agg.loc[RSDR, "Mean_MAE"]
check("Abstract/Sec. 4.1", "share of 30-day gain retained (%)", 47.05, 100 * (static - rsm) / (static - p30m), 2)
check("Sec. 4.1", "fewer updates than 30-day (%)", 93.75,
      100 * (1 - agg.loc[RSDR, "Total_Updates"] / agg.loc[P30, "Total_Updates"]), 2)
f30, frs = cost.loc[P30, "Total_Update_Fit_Seconds"], cost.loc[RSDR, "Total_Update_Fit_Seconds"]
check("Abstract/Sec. 4.1", "less update-fitting time (%)", 91.90, 100 * (1 - frs / f30), 2)
check("Sec. 4.1", "additional fitting time of 30-day (s)", 365.21, f30 - frs, 2)
check("Sec. 4.1", "additional refits of 30-day", 45, agg.loc[P30, "Total_Updates"] - agg.loc[RSDR, "Total_Updates"], 0)
rs_dates = sorted(updates.loc[updates.Policy == RSDR, "Update_Effective_Time"].astype(str).str[:10])
check("Sec. 4.2", "RSDR update dates 04 Jul, 03 Aug, 28 Oct", 1,
      float(rs_dates == ["2021-07-04", "2021-08-03", "2021-10-28"]), 0)
acc = alarms[alarms.Accepted_Degradation_Alarm.astype(str) == "True"]
check("Sec. 4.2", "ADWIN abs. accepted update on 26 Oct (Location 3)", 1,
      float(len(acc) == 1 and str(acc.Issue_Time.iloc[0]).startswith("2021-10-26")
            and acc.Location.iloc[0] == "Location3"), 0)
exc = alarms[alarms.Policy == EXC]
check("Sec. 4.2", "excess-error ADWIN raw alarms (all rejected)", 4,
      len(exc) if (exc.Accepted_Degradation_Alarm.astype(str) == "False").all() else -1, 0)

out = pd.DataFrame(rows)
out.to_csv(R / "paper_number_check.csv", index=False)
print(f"{int(out.match.sum())} of {len(out)} paper numbers match the results files.")
bad = out[~out.match]
if len(bad):
    print(bad.to_string(index=False))
