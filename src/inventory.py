"""Weekly periodic-review inventory simulation shared by the data generator and notebooks.

Two replenishment policies are supported:

* ``weeks_of_cover`` (the incumbent planner rule): reorder point and order-up-to level are
  multiples of the trailing 8-week average demand. It ignores seasonality and forecast error.
* ``forecast_min_max``: reorder point and order-up-to level supplied by the caller for every
  week, typically forecast demand over the lead time plus a safety stock derived from measured
  forecast error (reorder point), and the reorder point plus forecast cycle stock (order-up-to).

Sequence inside a week t:
  1. receipts scheduled for week t arrive
  2. demand (plus any open backorders) is served from on-hand stock, the rest is backordered
  3. end-of-week review: compute inventory position and place an order that arrives in week t + L
"""
from __future__ import annotations

import numpy as np

# Incumbent planner rule (documented in README): trailing 8-week average demand.
COVER_LOOKBACK_WEEKS = 8
ROP_EXTRA_WEEKS = 2.0   # reorder when position < avg * (lead_time + 2)
OUT_EXTRA_WEEKS = 5.0   # order up to avg * (lead_time + 5)


def round_up_to_pack(qty: float, pack: int) -> int:
    if qty <= 0:
        return 0
    return int(np.ceil(qty / pack) * pack)


def simulate(
    demand: np.ndarray,
    lead_time: int,
    case_pack: int,
    policy: str = "weeks_of_cover",
    start_week: int = 0,
    on_hand0: float | None = None,
    pipeline0: dict | None = None,
    backorder0: float = 0.0,
    reorder_point: np.ndarray | None = None,
    order_up_to: np.ndarray | None = None,
    rop_extra: float = ROP_EXTRA_WEEKS,
    out_extra: float = OUT_EXTRA_WEEKS,
    history: np.ndarray | None = None,
) -> dict:
    """Simulate one SKU-location from ``start_week`` to the end of ``demand``.

    Parameters
    ----------
    demand : weekly demand for the full horizon (index = week number).
    history : demand used for the trailing average (defaults to ``demand``); weeks before
        ``start_week`` are treated as history.
    reorder_point, order_up_to : for ``forecast_min_max``, arrays aligned with ``demand`` giving
        s_t and S_t (order up to S_t when inventory position < s_t).
    pipeline0 : {arrival_week: qty} open purchase orders at the start.

    Returns arrays (length = len(demand) - start_week) of end-of-week on_hand, on_order,
    backorder, shipped (incl. backorders cleared), filled (this week's demand met
    in-week), receipts and orders placed.
    """
    history = demand if history is None else history
    n = len(demand)
    T = n - start_week
    out = {k: np.zeros(T) for k in ["on_hand", "on_order", "backorder", "shipped", "filled", "receipts", "ordered"]}

    if on_hand0 is None:
        avg0 = demand[: max(start_week, COVER_LOOKBACK_WEEKS)].mean()
        on_hand0 = avg0 * (lead_time + out_extra)
    on_hand = float(on_hand0)
    backorder = float(backorder0)
    pipeline = dict(pipeline0 or {})

    for i, t in enumerate(range(start_week, n)):
        rec = pipeline.pop(t, 0.0)
        on_hand += rec
        filled = min(on_hand, demand[t])  # this week's demand served from stock in the week
        need = demand[t] + backorder
        ship = min(on_hand, need)
        on_hand -= ship
        backorder = need - ship

        on_order = sum(pipeline.values())
        position = on_hand + on_order - backorder
        if policy == "weeks_of_cover":
            lo = max(0, t - COVER_LOOKBACK_WEEKS + 1)
            avg = history[lo : t + 1].mean()
            rop = avg * (lead_time + rop_extra)
            target = avg * (lead_time + out_extra)
            qty = round_up_to_pack(target - position, case_pack) if position < rop else 0
        elif policy == "forecast_min_max":
            rop = reorder_point[t]
            qty = round_up_to_pack(order_up_to[t] - position, case_pack) if position < rop else 0
        else:
            raise ValueError(policy)
        if qty > 0:
            pipeline[t + lead_time] = pipeline.get(t + lead_time, 0.0) + qty
        out["on_hand"][i] = on_hand
        out["on_order"][i] = sum(pipeline.values())
        out["backorder"][i] = backorder
        out["shipped"][i] = ship
        out["filled"][i] = filled
        out["receipts"][i] = rec
        out["ordered"][i] = qty
    out["pipeline_end"] = pipeline
    return out
