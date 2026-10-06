"""Deterministic synthetic data generator for the retail demand forecasting project.

Simulates three years (2023-01-02 to 2025-12-28, 156 ISO weeks) of order-line transactions
for a fictional packaging distributor with 60 SKUs stocked in
4 distribution centres, plus weekly inventory snapshots and purchase orders produced by the
distributor's incumbent "weeks of cover" replenishment rule.

Planted structure that the notebooks are expected to discover:
  * category-specific annual seasonality (Q4 e-commerce peak for boxes and mailers, a summer
    peak for food-service containers, a sharp December peak for gift packaging)
  * SKU-level trends (growth and decline)
  * planned trade promotions with a demand lift
  * weekday ordering pattern and autocorrelated weekly demand shocks
  * an incumbent replenishment rule based on trailing average demand, which builds excess
    stock after seasonal peaks and runs short on the way up

Usage:
    python src/generate_data.py                # default sample (~330k order lines)
    python src/generate_data.py --scale 2.0    # ~2x order volume (larger CSVs)

Writes CSVs to data/raw/.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from inventory import simulate

SEED = 42
START = pd.Timestamp("2023-01-02")  # Monday
N_WEEKS = 156
RAW = Path(__file__).resolve().parents[1] / "data" / "raw"

CATEGORIES = {
    # name: (n_skus, peak_month, amplitude, sharpness, base unit price range)
    "Corrugated Boxes": (12, 11.3, 0.45, 1.0, (0.6, 3.5)),
    "Poly Mailers": (8, 11.5, 0.60, 1.4, (0.15, 0.9)),
    "Packing Tape": (7, 11.0, 0.25, 1.0, (2.5, 6.0)),
    "Stretch Film": (6, 9.0, 0.15, 1.0, (18.0, 45.0)),
    "Void Fill": (7, 11.4, 0.50, 1.2, (12.0, 40.0)),
    "Food Containers": (9, 7.0, 0.35, 1.0, (0.08, 0.6)),
    "Labels": (5, 6.0, 0.08, 1.0, (8.0, 25.0)),
    "Gift Packaging": (6, 12.0, 0.90, 3.0, (0.5, 4.0)),
}

LOCATIONS = pd.DataFrame(
    {
        "location_id": ["DC01", "DC02", "DC03", "DC04"],
        "location_name": ["Columbus DC", "Dallas DC", "Reno DC", "Atlanta DC"],
        "region": ["Midwest", "South Central", "West", "Southeast"],
        "volume_factor": [1.25, 1.0, 0.7, 0.9],
        "lead_time_weeks": [2, 2, 3, 1],
        "summer_shift": [0.0, 0.15, 0.05, 0.12],  # extra summer lift for food containers
    }
)


def seasonal_index(week_frac_month: np.ndarray, peak_month: float, amp: float, sharp: float) -> np.ndarray:
    """Smooth annual seasonality peaking at ``peak_month`` (1-12, fractional)."""
    phase = 2 * np.pi * (week_frac_month - peak_month) / 12.0
    raw = (1 + np.cos(phase)) / 2  # 0..1, 1 at peak
    raw = raw**sharp
    raw = raw - raw.mean()
    return 1 + amp * raw / raw.std() * 0.7


def build_products(rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    i = 1
    for cat, (n, peak, amp, sharp, (plo, phi)) in CATEGORIES.items():
        for k in range(n):
            price = float(np.round(np.exp(rng.uniform(np.log(plo), np.log(phi))), 2))
            margin = rng.uniform(0.22, 0.38)
            pack = int(rng.choice([1, 6, 10, 12, 24, 25, 50, 100, 250, 500],
                                  p=[.05, .1, .1, .1, .15, .15, .15, .1, .05, .05]))
            if price > 10:
                pack = int(rng.choice([1, 4, 6, 12]))
            rows.append(
                {
                    "sku_id": f"SKU{i:04d}",
                    "sku_name": f"{cat} item {k + 1:02d}",
                    "category": cat,
                    "case_pack": pack,
                    "list_price": price,
                    "unit_cost": float(np.round(price * (1 - margin), 3)),
                    "launch_date": START.date().isoformat(),
                }
            )
            i += 1
    return pd.DataFrame(rows)


def build_customers(rng: np.random.Generator, n: int = 420) -> pd.DataFrame:
    segs = rng.choice(["E-commerce", "Retail chain", "Food service", "Independent retail", "3PL"], n,
                      p=[.32, .18, .2, .2, .1])
    home = rng.choice(LOCATIONS.location_id, n, p=[.32, .26, .18, .24])
    tier = rng.choice(["Standard", "Silver", "Gold"], n, p=[.6, .3, .1])
    size = rng.pareto(1.3, n) + 1
    return pd.DataFrame(
        {
            "customer_id": [f"C{i:04d}" for i in range(1, n + 1)],
            "segment": segs,
            "home_location_id": home,
            "price_tier": tier,
            "size_weight": np.round(size / size.sum(), 6),
        }
    )


def build_promotions(rng: np.random.Generator, products: pd.DataFrame) -> pd.DataFrame:
    rows = []
    pid = 1
    for sku in products.sku_id:
        n_promos = rng.integers(5, 10)  # over three years
        weeks = np.sort(rng.choice(np.arange(4, N_WEEKS - 2), n_promos, replace=False))
        last = -10
        for w in weeks:
            if w - last < 6:
                continue
            dur = int(rng.choice([1, 2, 2, 3]))
            start = START + pd.Timedelta(weeks=int(w))
            rows.append(
                {
                    "promo_id": f"P{pid:05d}",
                    "sku_id": sku,
                    "start_date": start.date().isoformat(),
                    "end_date": (start + pd.Timedelta(days=7 * dur - 1)).date().isoformat(),
                    "discount_pct": float(rng.choice([0.08, 0.10, 0.12, 0.15, 0.20])),
                }
            )
            pid += 1
            last = w + dur
    return pd.DataFrame(rows)


def main(scale: float = 1.0) -> None:
    rng = np.random.default_rng(SEED)
    RAW.mkdir(parents=True, exist_ok=True)

    products = build_products(rng)
    customers = build_customers(rng)
    promotions = build_promotions(rng, products)

    days = pd.date_range(START, periods=N_WEEKS * 7, freq="D")
    n_days = len(days)
    week_idx = np.arange(n_days) // 7
    month_frac = days.month.values + (days.day.values - 1) / days.days_in_month.values
    dow_factor = np.array([1.25, 1.2, 1.15, 1.1, 1.0, 0.25, 0.05])[days.dayofweek.values]
    dow_factor = dow_factor / dow_factor.mean()
    t_years = np.arange(n_days) / 365.25

    # promo multiplier per sku per day
    promo_disc = np.zeros((len(products), n_days))
    sku_pos = {s: i for i, s in enumerate(products.sku_id)}
    for p in promotions.itertuples():
        a = (pd.Timestamp(p.start_date) - START).days
        b = (pd.Timestamp(p.end_date) - START).days + 1
        promo_disc[sku_pos[p.sku_id], a:b] = p.discount_pct

    lines = []
    cust_by_loc = {
        loc: customers[customers.home_location_id == loc].reset_index(drop=True)
        for loc in LOCATIONS.location_id
    }
    for si, prod in products.iterrows():
        peak, amp, sharp = CATEGORIES[prod.category][1:4]
        growth = rng.normal(0.06, 0.12)  # annual growth rate
        base_orders_week = np.exp(rng.normal(np.log(8.0), 0.35)) * scale
        # typical order size in units; bulky / expensive items are ordered in smaller quantities
        order_units = np.exp(rng.normal(np.log(150), 0.5)) / np.sqrt(max(prod.list_price, 0.3))
        mean_cases = max(order_units / prod.case_pack, 1.5)
        lift_per_pct = rng.uniform(1.0, 2.5)  # e.g. 15% off -> 1.15x-1.4x
        for li, loc in LOCATIONS.iterrows():
            season = seasonal_index(month_frac, peak, amp * rng.uniform(0.85, 1.15), sharp)
            if prod.category == "Food Containers":
                season = season * (1 + loc.summer_shift * np.cos(2 * np.pi * (month_frac - 7) / 12))
            trend = np.exp(growth * t_years)
            # AR(1) weekly multiplicative shock
            shocks = np.zeros(N_WEEKS)
            eps = rng.normal(0, 0.16, N_WEEKS)
            for w in range(1, N_WEEKS):
                shocks[w] = 0.6 * shocks[w - 1] + eps[w]
            shock = np.exp(shocks[week_idx])
            promo = 1 + lift_per_pct * promo_disc[si]
            rate = base_orders_week / 7 * loc.volume_factor * season * trend * shock * promo * dow_factor
            counts = rng.poisson(rate)
            n_lines = counts.sum()
            if n_lines == 0:
                continue
            day_rep = np.repeat(np.arange(n_days), counts)
            cases = np.maximum(1, np.round(rng.lognormal(np.log(mean_cases), 0.45, n_lines))).astype(int)
            qty = cases * int(prod.case_pack)
            cl = cust_by_loc[loc.location_id]
            cust = rng.choice(len(cl), n_lines, p=cl.size_weight / cl.size_weight.sum())
            tier_disc = cl.price_tier.map({"Standard": 0.0, "Silver": 0.03, "Gold": 0.06}).values[cust]
            price = prod.list_price * (1 - promo_disc[si, day_rep]) * (1 - tier_disc)
            lines.append(
                pd.DataFrame(
                    {
                        "day": day_rep,
                        "customer_id": cl.customer_id.values[cust],
                        "sku_id": prod.sku_id,
                        "location_id": loc.location_id,
                        "quantity": qty,
                        "unit_price": np.round(price, 3),
                    }
                )
            )

    tx = pd.concat(lines, ignore_index=True)
    tx["order_date"] = days[tx.day.values].strftime("%Y-%m-%d")
    tx = tx.sort_values(["day", "location_id", "customer_id", "sku_id"]).reset_index(drop=True)
    order_key = tx.day.astype(str) + "|" + tx.location_id + "|" + tx.customer_id
    tx["order_id"] = "SO" + (pd.factorize(order_key)[0] + 100001).astype(str)
    tx["line_no"] = tx.groupby("order_id").cumcount() + 1
    tx = tx[["order_id", "line_no", "order_date", "customer_id", "location_id", "sku_id", "quantity", "unit_price"]]

    # --- weekly demand and incumbent-policy inventory simulation -------------------------
    tx_week = (pd.to_datetime(tx.order_date) - START).dt.days // 7
    weekly = (
        tx.assign(week=tx_week).groupby(["sku_id", "location_id", "week"]).quantity.sum()
        .unstack("week").reindex(columns=range(N_WEEKS), fill_value=0).fillna(0)
    )
    snaps, pos = [], []
    po_id = 1
    for (sku, loc), row in weekly.iterrows():
        d = row.values.astype(float)
        prod = products.loc[products.sku_id == sku].iloc[0]
        L = int(LOCATIONS.set_index("location_id").loc[loc, "lead_time_weeks"])
        # planners hold extra buffer on top of the rule; varies by SKU-location
        sim = simulate(d, L, int(prod.case_pack), policy="weeks_of_cover",
                       on_hand0=d[:8].mean() * (L + 4))
        week_end = START + pd.to_timedelta(np.arange(N_WEEKS) * 7 + 6, unit="D")
        snaps.append(
            pd.DataFrame(
                {
                    "snapshot_date": week_end.strftime("%Y-%m-%d"),
                    "sku_id": sku,
                    "location_id": loc,
                    "on_hand_units": sim["on_hand"].round().astype(int),
                    "on_order_units": sim["on_order"].round().astype(int),
                    "backorder_units": sim["backorder"].round().astype(int),
                }
            )
        )
        for w in np.nonzero(sim["ordered"])[0]:
            pos.append(
                {
                    "po_id": f"PO{po_id:06d}",
                    "sku_id": sku,
                    "location_id": loc,
                    "order_date": week_end[w].strftime("%Y-%m-%d"),
                    "expected_receipt_date": (week_end[w] + pd.Timedelta(days=7 * L - 6)).strftime("%Y-%m-%d"),
                    "quantity": int(sim["ordered"][w]),
                }
            )
            po_id += 1
    snapshots = pd.concat(snaps, ignore_index=True)
    purchase_orders = pd.DataFrame(pos)

    products.to_csv(RAW / "products.csv", index=False)
    LOCATIONS.drop(columns=["volume_factor", "summer_shift"]).to_csv(RAW / "locations.csv", index=False)
    customers.drop(columns=["size_weight"]).to_csv(RAW / "customers.csv", index=False)
    promotions.to_csv(RAW / "promotions.csv", index=False)
    tx.to_csv(RAW / "transactions.csv", index=False)
    snapshots.to_csv(RAW / "inventory_snapshots.csv", index=False)
    purchase_orders.to_csv(RAW / "purchase_orders.csv", index=False)

    for f in sorted(RAW.glob("*.csv")):
        print(f"{f.name:28s} {sum(1 for _ in open(f)) - 1:>9,d} rows  {f.stat().st_size / 1e6:6.1f} MB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scale", type=float, default=1.0, help="multiplier on order volume (default 1.0)")
    main(ap.parse_args().scale)
