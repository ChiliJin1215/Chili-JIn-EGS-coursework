"""
Case B - grid-scale battery trading on the day-ahead market.
Runs a simple heuristic, an LP arbitrage model, and an LP with
ancillary stacking (the extension). Prints verification checks
and saves figures + a KPI json.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import linprog
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
OUT = HERE / "out"
OUT.mkdir(exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 130, "font.size": 9, "axes.grid": True, "grid.alpha": 0.3,
    "axes.spines.top": False, "axes.spines.right": False,
})


# ---- data ----
df = pd.read_csv(
    HERE / "caseB_grid_battery_market_hourly.csv",
    parse_dates=["timestamp"],
)
T = len(df)
dt = 1.0  # hours

# convert GBP/MWh -> GBP/kWh once up front
price_da  = df["day_ahead_price_gbp_per_mwh"].to_numpy() / 1000.0
price_imb = df["imbalance_price_gbp_per_mwh"].to_numpy()  / 1000.0
anc_pay   = df["ancillary_availability_gbp_per_mw_per_h"].to_numpy() / 1000.0
co2       = df["carbon_intensity_kg_per_kwh_optional"].to_numpy()

# quick unit check so I can point to it in the report
sample_mwh = df["day_ahead_price_gbp_per_mwh"].iloc[0]
assert abs(sample_mwh / 1000.0 - price_da[0]) < 1e-12
print(f"Unit check: {sample_mwh:.2f} GBP/MWh  =  {price_da[0]:.5f} GBP/kWh  ok")


# ---- battery parameters ----
E_CAP   = 2000.0      # kWh
P_CH    = 1000.0      # kW
P_DIS   = 1000.0      # kW
ETA_CH  = 0.938       # one-way, so round-trip = 0.938^2 ~ 0.88
ETA_DIS = 0.938
SOC_MIN = 0.0
SOC_MAX = E_CAP
SOC_0   = 0.5 * E_CAP


# ---- heuristic: charge in the cheapest 30%, discharge in the top 30% ----
def run_heuristic(price, low_q=0.30, high_q=0.70):
    p_lo, p_hi = np.quantile(price, [low_q, high_q])
    soc = np.zeros(T + 1)
    soc[0] = SOC_0
    p_ch  = np.zeros(T)
    p_dis = np.zeros(T)

    for t in range(T):
        if price[t] <= p_lo:
            # charge as much as fits
            room = (SOC_MAX - soc[t]) / (ETA_CH * dt)
            p_ch[t] = min(P_CH, room)
        elif price[t] >= p_hi:
            # discharge whatever the SOC allows
            avail = (soc[t] - SOC_MIN) * ETA_DIS / dt
            p_dis[t] = min(P_DIS, avail)
        soc[t + 1] = soc[t] + ETA_CH * p_ch[t] * dt - p_dis[t] * dt / ETA_DIS

    cashflow = price * (p_dis - p_ch) * dt
    return {
        "p_ch": p_ch, "p_dis": p_dis, "soc": soc,
        "cashflow": cashflow, "profit": cashflow.sum(),
    }


# ---- LP arbitrage with perfect foresight ----
# Decision vector x has length 3*T:
#   x[0 : T]       -> p_ch  (kW)
#   x[T : 2T]      -> p_dis (kW)
#   x[2T : 3T]     -> SOC at end of hour t (kWh)
#
# Minimising  sum_t  price[t] * (p_ch[t] - p_dis[t]) * dt
# is the same as maximising profit.
def solve_lp(price, soc_end=SOC_0, deg_cost=0.0):
    n = T
    nvar = 3 * n

    c = np.zeros(nvar)
    c[0:n]     =  price * dt + deg_cost   # charging costs money
    c[n:2*n]   = -price * dt + deg_cost   # discharging earns money

    bounds = (
        [(0, P_CH)]   * n +
        [(0, P_DIS)]  * n +
        [(SOC_MIN, SOC_MAX)] * n
    )

    # equality constraints: SOC dynamics + terminal SOC
    A_eq = np.zeros((n + 1, nvar))
    b_eq = np.zeros(n + 1)

    # t = 0 is a special case because SOC_{-1} is the fixed SOC_0
    A_eq[0, 2*n + 0] = 1.0              # SOC at t=0
    A_eq[0, 0]       = -ETA_CH * dt     # minus charge term
    A_eq[0, n]       =  dt / ETA_DIS    # plus discharge term
    b_eq[0] = SOC_0

    for t in range(1, n):
        A_eq[t, 2*n + t]     =  1.0
        A_eq[t, 2*n + t - 1] = -1.0
        A_eq[t, t]           = -ETA_CH * dt
        A_eq[t, n + t]       =  dt / ETA_DIS

    # force end-of-horizon SOC
    A_eq[n, 2*n + n - 1] = 1.0
    b_eq[n] = soc_end

    res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    assert res.success, res.message

    x = res.x
    p_ch  = x[0:n]
    p_dis = x[n:2*n]
    soc   = np.concatenate([[SOC_0], x[2*n:3*n]])
    cashflow = price * (p_dis - p_ch) * dt
    return {
        "p_ch": p_ch, "p_dis": p_dis, "soc": soc,
        "cashflow": cashflow, "profit": cashflow.sum(),
    }


# ---- LP with ancillary stacking (extension) ----
# Adds r[t] >= 0 (reserved availability in kW). The battery is paid
# anc_pay[t] per kW per hour for r[t]. Reserved capacity has to be
# physically deliverable, so it competes with arbitrage on both power
# and energy headroom.
def solve_lp_stack(price, anc, soc_end=SOC_0):
    n = T
    nvar = 4 * n   # p_ch, p_dis, soc, r

    c = np.zeros(nvar)
    c[0:n]     =  price * dt
    c[n:2*n]   = -price * dt
    c[3*n:4*n] = -anc * dt           # availability income

    r_ub = min(P_CH, P_DIS)
    bounds = (
        [(0, P_CH)]           * n +
        [(0, P_DIS)]          * n +
        [(SOC_MIN, SOC_MAX)]  * n +
        [(0, r_ub)]           * n
    )

    # same SOC dynamics as the arbitrage LP
    A_eq = np.zeros((n + 1, nvar))
    b_eq = np.zeros(n + 1)

    A_eq[0, 2*n + 0] = 1.0
    A_eq[0, 0]       = -ETA_CH * dt
    A_eq[0, n]       =  dt / ETA_DIS
    b_eq[0] = SOC_0

    for t in range(1, n):
        A_eq[t, 2*n + t]     =  1.0
        A_eq[t, 2*n + t - 1] = -1.0
        A_eq[t, t]           = -ETA_CH * dt
        A_eq[t, n + t]       =  dt / ETA_DIS

    A_eq[n, 2*n + n - 1] = 1.0
    b_eq[n] = soc_end

    # inequality constraints for the headroom
    rows, rhs = [], []
    for t in range(n):
        # inverter power headroom, up and down
        row = np.zeros(nvar); row[t] = 1; row[3*n + t] = 1
        rows.append(row); rhs.append(P_CH)

        row = np.zeros(nvar); row[n + t] = 1; row[3*n + t] = 1
        rows.append(row); rhs.append(P_DIS)

        # energy headroom - must be able to discharge r for 1 hour
        # r * dt - eta_dis * soc_{t-1} <= 0
        row = np.zeros(nvar); row[3*n + t] = dt
        if t == 0:
            rhs.append(SOC_0 * ETA_DIS)
        else:
            row[2*n + t - 1] = -ETA_DIS
            rhs.append(0.0)
        rows.append(row)

        # and must have free space to absorb r for 1 hour
        # r * dt + soc_{t-1} / eta_ch <= E_cap / eta_ch
        row = np.zeros(nvar); row[3*n + t] = dt
        if t == 0:
            rhs.append((E_CAP - SOC_0) / ETA_CH)
        else:
            row[2*n + t - 1] = 1.0 / ETA_CH
            rhs.append(E_CAP / ETA_CH)
        rows.append(row)

    A_ub = np.vstack(rows)
    b_ub = np.array(rhs)

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=bounds, method="highs")
    assert res.success, res.message

    x = res.x
    p_ch  = x[0:n]
    p_dis = x[n:2*n]
    soc   = np.concatenate([[SOC_0], x[2*n:3*n]])
    r     = x[3*n:4*n]
    energy_cf = price * (p_dis - p_ch) * dt
    anc_cf    = anc * r * dt
    return {
        "p_ch": p_ch, "p_dis": p_dis, "soc": soc, "r": r,
        "energy_profit": energy_cf.sum(),
        "anc_profit": anc_cf.sum(),
        "profit": energy_cf.sum() + anc_cf.sum(),
        "cashflow": energy_cf + anc_cf,
    }


# ---- verification ----
def verify(name, sol, has_r=False):
    p_ch  = sol["p_ch"]
    p_dis = sol["p_dis"]
    soc   = sol["soc"]

    lines = []

    soc_ok = (soc >= SOC_MIN - 1e-6).all() and (soc <= SOC_MAX + 1e-6).all()
    lines.append(f"SOC in [0, {E_CAP}]                        : {'PASS' if soc_ok else 'FAIL'}")

    p_ok = ((p_ch >= -1e-6).all() and (p_ch <= P_CH + 1e-6).all()
            and (p_dis >= -1e-6).all() and (p_dis <= P_DIS + 1e-6).all())
    lines.append(f"Power within [0, P_max]                  : {'PASS' if p_ok else 'FAIL'}")

    # how many hours charge and discharge at the same time - should be 0
    simul = ((p_ch > 1e-3) & (p_dis > 1e-3)).sum()
    lines.append(f"No-simultaneity (#violations)             : {simul}")

    # energy balance residual: compare delta SOC against what the equation says
    lhs = np.diff(soc)
    rhs = ETA_CH * p_ch * dt - p_dis * dt / ETA_DIS
    eb_res = np.max(np.abs(lhs - rhs))
    lines.append(f"Energy-balance residual (kWh, max)        : {eb_res:.2e}")

    # empirical round-trip efficiency = energy out / energy in at terminals
    e_in  = (p_ch * dt).sum()
    e_out = (p_dis * dt).sum()
    rte_emp = e_out / e_in if e_in > 0 else float("nan")
    lines.append(f"Empirical RTE (target 0.880)              : {rte_emp:.3f}")

    lines.append(f"SOC_end - SOC_0 (kWh)                     : {soc[-1] - SOC_0:+.2f}")

    if has_r:
        r = sol["r"]
        headroom_ok = ((p_ch + r) <= P_CH + 1e-6).all() and ((p_dis + r) <= P_DIS + 1e-6).all()
        lines.append(f"Headroom (p + r <= P_max)                 : {'PASS' if headroom_ok else 'FAIL'}")

    print(f"\n--- VERIFICATION: {name} ---")
    for line in lines:
        print("  " + line)


# ---- run everything ----
heur  = run_heuristic(price_da)
lp    = solve_lp(price_da)
stack = solve_lp_stack(price_da, anc_pay)

verify("Heuristic",    heur)
verify("LP arbitrage", lp)
verify("LP stacking",  stack, has_r=True)


# ---- KPIs ----
def kpis(sol):
    throughput = sol["p_ch"].sum() * dt + sol["p_dis"].sum() * dt   # kWh
    cycles     = sol["p_dis"].sum() * dt / E_CAP
    return sol["profit"], throughput, cycles

p_h, t_h, c_h = kpis(heur)
p_l, t_l, c_l = kpis(lp)
p_s, t_s, c_s = kpis(stack)

print("\n=========== RESULTS (60 days) ===========")
print(f"{'Strategy':22s} {'Profit GBP':>12s} {'Throughput kWh':>16s} {'Cycles':>9s}")
print(f"{'Heuristic (30/70%)':22s} {p_h:12.2f} {t_h:16.0f} {c_h:9.2f}")
print(f"{'LP arbitrage':22s} {p_l:12.2f} {t_l:16.0f} {c_l:9.2f}")
print(f"{'LP + stacking':22s} {p_s:12.2f} {t_s:16.0f} {c_s:9.2f}")
print(f"  energy part    : {stack['energy_profit']:.2f}")
print(f"  ancillary part : {stack['anc_profit']:.2f}")


# ---- plots ----
hours = np.arange(T)

# Figure 1 - price overview
fig, ax = plt.subplots(figsize=(7.2, 2.4))
ax.plot(hours, price_da * 1000, lw=0.6, label="Day-ahead")
ax.set_xlabel("Hour")
ax.set_ylabel("GBP/MWh")
ax.set_title("Day-ahead price - 60 days, hourly")
ax.legend(loc="upper right")
fig.tight_layout()
fig.savefig(OUT / "fig1_price.png")
plt.close(fig)

# Figure 2 - dispatch for the first week, one row per strategy
W = 24 * 7
fig, axes = plt.subplots(3, 1, figsize=(7.2, 5.4), sharex=True)
for ax, sol, name in zip(
    axes,
    [heur, lp, stack],
    ["Heuristic", "LP arbitrage", "LP + stacking"],
):
    net = sol["p_dis"][:W] - sol["p_ch"][:W]
    colours = np.where(net >= 0, "#2a9d8f", "#e76f51")
    ax.bar(hours[:W], net, width=1.0, color=colours)
    ax2 = ax.twinx()
    ax2.plot(hours[:W], price_da[:W] * 1000, color="k", lw=0.8)
    ax2.set_ylabel("GBP/MWh", color="k")
    ax.set_ylabel("Net power (kW)\n+dis / -ch")
    ax.set_title(f"{name} - week 1")
axes[-1].set_xlabel("Hour")
fig.tight_layout()
fig.savefig(OUT / "fig2_dispatch.png")
plt.close(fig)

# Figure 3 - SOC
fig, ax = plt.subplots(figsize=(7.2, 2.6))
ax.plot(heur["soc"]  / E_CAP * 100, label="Heuristic",     lw=0.8)
ax.plot(lp["soc"]    / E_CAP * 100, label="LP arbitrage",  lw=0.8)
ax.plot(stack["soc"] / E_CAP * 100, label="LP + stacking", lw=0.8)
ax.set_xlabel("Hour")
ax.set_ylabel("SOC (%)")
ax.set_ylim(-2, 102)
ax.legend(ncol=3, loc="lower center")
ax.set_title("State of charge")
fig.tight_layout()
fig.savefig(OUT / "fig3_soc.png")
plt.close(fig)

# Figure 4 - cumulative profit
fig, ax = plt.subplots(figsize=(7.2, 2.6))
ax.plot(np.cumsum(heur["cashflow"]), label="Heuristic")
ax.plot(np.cumsum(lp["cashflow"]),   label="LP arbitrage")
cum_stack = np.cumsum(
    price_da * (stack["p_dis"] - stack["p_ch"]) * dt + anc_pay * stack["r"] * dt
)
ax.plot(cum_stack, label="LP + stacking")
ax.set_xlabel("Hour")
ax.set_ylabel("Cumulative profit (GBP)")
ax.legend()
ax.set_title("Cumulative profit over 60 days")
fig.tight_layout()
fig.savefig(OUT / "fig4_profit.png")
plt.close(fig)

# Figure 5 - sensitivity to RTE
rtes = np.linspace(0.70, 0.98, 8)
sens = []
ETA_CH_save, ETA_DIS_save = ETA_CH, ETA_DIS
for rte in rtes:
    eta = np.sqrt(rte)
    ETA_CH, ETA_DIS = eta, eta
    sens.append(solve_lp(price_da)["profit"])
ETA_CH, ETA_DIS = ETA_CH_save, ETA_DIS_save

fig, ax = plt.subplots(figsize=(7.2, 2.4))
ax.plot(rtes * 100, sens, "o-")
ax.set_xlabel("Round-trip efficiency (%)")
ax.set_ylabel("LP profit (GBP)")
ax.set_title("Profit sensitivity to round-trip efficiency")
fig.tight_layout()
fig.savefig(OUT / "fig5_sens.png")
plt.close(fig)

print("\nFigures written to", OUT)


# ---- dump KPIs for the report ----
json.dump(
    {
        "heur":  {"profit": p_h, "throughput": t_h, "cycles": c_h},
        "lp":    {"profit": p_l, "throughput": t_l, "cycles": c_l},
        "stack": {
            "profit": p_s, "throughput": t_s, "cycles": c_s,
            "energy": stack["energy_profit"], "anc": stack["anc_profit"],
        },
        "price_stats": {
            "min":  float(price_da.min()  * 1000),
            "max":  float(price_da.max()  * 1000),
            "mean": float(price_da.mean() * 1000),
            "std":  float(price_da.std()  * 1000),
        },
        "T": T,
    },
    open(OUT / "kpis.json", "w"),
    indent=2,
)
