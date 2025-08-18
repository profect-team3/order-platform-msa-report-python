import os
from datetime import datetime, timedelta, timezone
from typing import Dict, Tuple

import pandas as pd
from pymongo.collection import Collection

KST = timezone(timedelta(hours=9))
UTC = timezone.utc

STATUSES_MAIN = {"COMPLETED", "REFUNDED"}
STATUSES_ALL = {"COMPLETED", "REFUNDED", "REJECTED", "FAILED"}
STATUSES_CANCEL = {"REFUNDED", "REJECTED"}

AGE_BUCKETS = [(10,19,"10대"), (20,29,"20대"), (30,39,"30대"), (40,49,"40대"), (50,200,"50대+")]

def _to_kst(dt_utc_iso: str) -> datetime:
    s = dt_utc_iso
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    return datetime.fromisoformat(s).astimezone(KST)

def _age_bucket_from_birth(birth: str, ref_kst_dt: datetime) -> str:
    try:
        by = int(birth[:4])
        age = ref_kst_dt.year - by
    except Exception:
        return "50대+"
    for lo, hi, name in AGE_BUCKETS:
        if lo <= age <= hi:
            return name
    return "50대+"

def _bin_reorder_days(days: float) -> str:
    if days <= 7: return "0–7일"
    if days <= 30: return "8–30일"
    if days <= 90: return "31–90일"
    if days <= 180: return "91–180일"
    return "181일+"

def _utc_boundary_for_kst_range(start_kst_date: datetime.date, end_kst_date: datetime.date) -> Tuple[datetime, datetime]:
    start_kst = datetime.combine(start_kst_date, datetime.min.time(), tzinfo=KST)
    end_kst = datetime.combine(end_kst_date, datetime.max.time().replace(microsecond=0), tzinfo=KST)
    return start_kst.astimezone(UTC), end_kst.astimezone(UTC)

def load_frames_30d_unified(col: Collection, store_id: str, start_kst_date, end_kst_date) -> Dict[str, pd.DataFrame]:
    """
    orders_unified에서 6개 지표용 DataFrame들을 생성해 반환함
    """
    start_utc, end_utc = _utc_boundary_for_kst_range(start_kst_date, end_kst_date)

    cursor = col.find({
        "p_orders.store_id": store_id,
        "p_orders.created_at": {"$gte": start_utc.isoformat().replace("+00:00","Z"),
                                "$lte": end_utc.isoformat().replace("+00:00","Z")}
    }, projection={"_id":1, "p_orders":1, "p_order_item":1, "p_user":1,
                   "is_first_order_in_store":1, "prev_order_created_at":1})

    docs = list(cursor)
    if not docs:
        empty = pd.DataFrame()
        return {
            "menu_total": empty, "menu_by_gender": empty, "menu_by_age": empty,
            "hour_menu": empty, "new_vs_return": empty, "cancel_rate": empty,
            "reorder_gap_hist": empty, "reorder_top3": empty
        }

    # ---------- pandas ----------
    orders = pd.json_normalize(docs)
    orders["created_at_kst"] = orders["p_orders.created_at"].apply(lambda s: _to_kst(s))
    orders["hour"] = orders["created_at_kst"].dt.hour
    orders["day"]  = orders["created_at_kst"].dt.strftime("%Y-%m-%d")

    mask_main = orders["p_orders.order_status"].isin(list(STATUSES_MAIN))
    orders_main = orders[mask_main].copy()

    items = orders_main.explode("p_order_item", ignore_index=True)
    items = items[items["p_order_item"].notna()].copy()
    items["menu_name"] = items["p_order_item"].apply(lambda x: x.get("menu_name"))
    items["qty"] = items["p_order_item"].apply(lambda x: x.get("quantity", 0))

    ref_dt = datetime.now(KST)
    items["gender"] = items["p_user.gender"].fillna("M")
    items["age_bucket"] = items["p_user.birthdate"].apply(lambda s: _age_bucket_from_birth(s, ref_dt))

    # ---------- 지표 생성 ----------
    # 1) 메뉴별 주문량
    df_menu_total = (items.groupby("menu_name")["qty"].sum()
                     .reset_index().rename(columns={"qty":"qty_sum"})
                     .sort_values("qty_sum", ascending=False))

    # 2) 메뉴별 성별 주문량
    df_menu_gender = (items.groupby(["menu_name","gender"])["qty"].sum()
                      .reset_index().rename(columns={"qty":"qty_sum"}))

    # 3) 메뉴별 연령대별 주문량
    df_menu_age = (items.groupby(["menu_name","age_bucket"])["qty"].sum()
                   .reset_index().rename(columns={"qty":"qty_sum"}))

    # 4) 시간대별 메뉴 주문량
    df_hour_menu = (items.groupby(["hour","menu_name"])["qty"].sum()
                    .reset_index().rename(columns={"qty":"qty_sum"})
                    .sort_values(["hour","menu_name"]))

    # 5) 신규 vs 재주문
    df_new_return = (orders_main
                     .assign(new=lambda df: df["is_first_order_in_store"].fillna(False))
                     .groupby("new")["_id"].count().reset_index()
                     .rename(columns={"_id":"count"}))
    df_new_return["label"] = df_new_return["new"].map({True:"신규", False:"재주문"})    

    # 6) 취소건수/비율
    orders_mother = orders[orders["p_orders.order_status"].isin(list(STATUSES_ALL))]
    total_mother = len(orders_mother)
    cancel_count = (orders_mother["p_orders.order_status"].isin(list(STATUSES_CANCEL))).sum()
    df_cancel_rate = pd.DataFrame([{
        "total": int(total_mother),
        "cancel": int(cancel_count),
        "cancel_rate": (cancel_count / total_mother) if total_mother > 0 else 0.0
    }])

    # 재주문 간격(일) 히스토그램
    returns = orders_main[orders_main["is_first_order_in_store"] == False].copy()
    returns = returns[returns["prev_order_created_at"].notna()]
    if not returns.empty:
        returns["prev_kst"] = returns["prev_order_created_at"].apply(lambda s: _to_kst(s))
        returns["gap_days"] = (returns["created_at_kst"] - returns["prev_kst"]).dt.total_seconds() / 86400.0
        returns = returns[returns["gap_days"] >= 0]
        returns["gap_bin"] = returns["gap_days"].apply(_bin_reorder_days)
        df_reorder_hist = (returns.groupby("gap_bin")["_id"].count()
                           .reset_index().rename(columns={"_id":"count"})
                           .sort_values("gap_bin"))
    else:
        df_reorder_hist = pd.DataFrame(columns=["gap_bin","count"])

    # 재주문 많은 메뉴 TOP3
    items_returns = items[items["_id"].isin(returns["_id"])]
    df_reorder_top3 = (items_returns.groupby("menu_name")["qty"].sum()
                       .reset_index().rename(columns={"qty":"qty_sum"})
                       .sort_values("qty_sum", ascending=False).head(3))

    return {
        "menu_total": df_menu_total,
        "menu_by_gender": df_menu_gender,
        "menu_by_age": df_menu_age,
        "hour_menu": df_hour_menu,
        "new_vs_return": df_new_return,
        "cancel_rate": df_cancel_rate,
        "reorder_gap_hist": df_reorder_hist,
        "reorder_top3": df_reorder_top3
    }
