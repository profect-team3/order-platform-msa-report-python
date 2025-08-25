from typing import Dict, Any
import pandas as pd
from datetime import datetime
from . import pdf_generator

CANCELED_STATES = {"CANCELED", "CANCELLED"}

STATUSES_MAIN = {"COMPLETED", "REFUNDED"}
STATUSES_MOTHER = {"COMPLETED", "REFUNDED", "REJECTED", "FAILED"}
STATUSES_CANCEL = {"REFUNDED", "REJECTED"}

AGE_BUCKETS = [
    (0, 9, "<10"), (10, 19, "10s"), (20, 29, "20s"), (30, 39, "30s"),
    (40, 49, "40s"), (50, 59, "50s"), (60, 200, "60s+"),
]

def _age_at(at: datetime, birth: datetime) -> int:
    return at.year - birth.year - ((at.month, at.day) < (birth.month, birth.day))

def _age_bucket(age: int) -> str:
    for lo, hi, label in AGE_BUCKETS:
        if lo <= age <= hi:
            return label
    return "UNKNOWN"

def _sex_norm(v) -> str:
    if v is None:
        return "UNKNOWN"
    s = str(v).strip().upper()
    return s if s in {"MALE", "FEMALE"} else "UNKNOWN"

def _bin_reorder_days(days: float) -> str:
    # 0–7 / 8–30 / 31–90 / 91–180 / 181+
    if days <= 7: return "0–7일"
    if days <= 30: return "8–30일"
    if days <= 90: return "31–90일"
    if days <= 180: return "91–180일"
    return "181일+"

def to_orders_df(req) -> pd.DataFrame:
    rows = []
    for o in req.orders:
        rows.append({
            "order_id": str(o.orderId),
            "created_at": o.createdAt,
            "total_price": o.totalPrice,
            "order_channel": o.orderChannel,
            "receipt_method": o.receiptMethod,
            "payment_method": o.paymentMethod,
            "order_status": o.orderStatus,
            "store_name": o.storeName,
            "user_sex": _sex_norm(o.usersex),
            "birthdate": o.birthdate,    # date | None
            "is_first_in_store": bool(o.isFirstOrderInStore),
            "order_seq_in_store": int(o.orderSeqInStore),
            "items": [i.model_dump() for i in o.items],
        })
    return pd.DataFrame(rows)

def explode_items(orders_df: pd.DataFrame) -> pd.DataFrame:
    '''
    주문(order) 단위의 데이터를 상품(item) 단위의 데이터로 변환합니다.
    '''
    if orders_df.empty:
        return pd.DataFrame(columns=[
            "order_id", "created_at", "order_status", "user_sex", "birthdate", "age_band", "hour",
            "order_seq_in_store", "is_first_in_store", "menu_name", "price", "quantity", "sales"
        ])
    df = orders_df.copy()
    df = df.explode("items", ignore_index=True)
    df = df[df["items"].notna()].copy()
    if df.empty:
        return pd.DataFrame(columns=[
            "order_id", "created_at", "order_status", "user_sex", "birthdate", "age_band", "hour",
            "order_seq_in_store", "is_first_in_store", "menu_name", "price", "quantity", "sales"
        ])
    df["menu_name"] = df["items"].map(lambda x: x.get("menuName"))
    df["price"]     = df["items"].map(lambda x: int(x.get("price", 0)))
    df["quantity"]  = df["items"].map(lambda x: int(x.get("quantity", 0)))
    df["sales"]     = df["price"] * df["quantity"]
    df.drop(columns=["items"], inplace=True)
    print(df)
    return df

def build_frames(req) -> Dict[str, pd.DataFrame]:
    """
    orders_unified에서 6개 지표용 DataFrame들을 생성해 반환합니다.
    - 1~5번: order_status ∈ {COMPLETED, REFUNDED}
    - 6번(취소율): 모집합 {COMPLETED, REFUNDED, REJECTED, FAILED}, 취소집합 {REFUNDED, REJECTED}
    - 재주문 간격: prev_order_created_at은 기간 밖(과거)이어도 됨
    """
    orders = to_orders_df(req)

    if not orders.empty:
        orders["created_at"] = pd.to_datetime(orders["created_at"])
        def _ageband(row):
            if pd.isna(row["birthdate"]):
                return "UNKNOWN"
            birth = pd.to_datetime(row["birthdate"]).to_pydatetime()
            age = _age_at(row["created_at"].to_pydatetime(), birth)
            return _age_bucket(age)
        orders["age_band"] = orders.apply(_ageband, axis=1)
        orders["hour"] = orders["created_at"].dt.hour
        orders["is_cancel"] = orders["order_status"].str.upper().isin(CANCELED_STATES)
        orders["is_reorder"] = orders["order_seq_in_store"].fillna(1).astype(int) >= 2
    else:
        orders = pd.DataFrame(columns=[
            "order_id","created_at","total_price","order_channel","receipt_method",
            "payment_method","order_status","store_name","user_sex","birthdate",
            "is_first_in_store","order_seq_in_store","age_band","hour",
            "is_cancel","is_reorder"
        ])

    items = explode_items(orders)
    
    # 1) 메뉴별 주문량 (qty 합)
    df_menu_total = (items.groupby("menu_name")["quantity"].sum()
                     .reset_index().rename(columns={"quantity":"total_qty"})
                     .sort_values("total_qty", ascending=False))

    # 2) 메뉴별 성별 주문량
    df_menu_gender = (items.groupby(["menu_name","user_sex"])["quantity"].sum()
                      .reset_index().rename(columns={"quantity":"qty_sum"}))

    # 3) 메뉴별 연령대별 주문량
    df_menu_age = (items.groupby(["menu_name","age_band"])["quantity"].sum()
                   .reset_index().rename(columns={"quantity":"qty_sum"}))

    # 4) 시간대별 메뉴 주문량 (히트맵/라인용: hour x menu, qty 합)
    df_hour_menu = (items.groupby(["hour","menu_name"])["quantity"].sum()
                    .reset_index().rename(columns={"quantity":"qty_sum"})
                    .sort_values(["hour","menu_name"]))

    # 5) 신규 vs 재주문 (건수 기준 or 주문량 기준? → 요청: 주문량은 quantity 기준이지만
    #    신규/재주문 비율은 "주문 건수"로 보는 게 보통 더 직관적이라 건수로 계산 권장)
    #    만약 quantity 기준 원하면 아래 count→sum(quantity)로 바꿔도 됨.
    df_new_return = (orders
                     .assign(new=lambda df: df["is_first_in_store"].fillna(False))
                     .groupby("new")["order_id"].count().reset_index()
                     .rename(columns={"order_id":"count"}))
    # 라벨링
    df_new_return["label"] = df_new_return["new"].map({True:"신규", False:"재주문"})    

    # 6) 취소건수/비율
    orders_mother = orders[orders["order_status"].isin(list(STATUSES_MOTHER))]
    total_mother = len(orders_mother)
    cancel_count = (orders_mother["order_status"].isin(list(STATUSES_CANCEL))).sum()
    df_cancel_rate = pd.DataFrame([{
        "total": int(total_mother),
        "cancel": int(cancel_count),
        "cancel_rate": (cancel_count / total_mother) if total_mother > 0 else 0.0
    }])

    return {
        "menu_total": df_menu_total,
        "menu_by_gender": df_menu_gender,
        "menu_by_age": df_menu_age,
        "hour_menu": df_hour_menu,
        "new_vs_return": df_new_return,
        "cancel_rate": df_cancel_rate,
        # "reorder_gap_hist": df_cancel_rate,
        # "reorder_top3": df_cancel_rate,
    }
