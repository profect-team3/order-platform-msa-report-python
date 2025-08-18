import os, random, uuid
from datetime import datetime, timedelta, timezone
from pymongo import MongoClient, UpdateOne

MONGO_URI = os.getenv("MONGO_URI")
DB_NAME   = os.getenv("DB_NAME", "report_db")
COLL_NAME = os.getenv("COLL_NAME", "orders_unified")

# 타임존
KST = timezone(timedelta(hours=9))
UTC = timezone.utc

# 샘플 상수
STORE_ID = "517b2aac-f81a-4931-b7a0-eda4377a9218"
MENUS = [
    ("후라이드 치킨", 17000),
    ("양념 치킨", 18000),
    ("피자", 22000),
    ("샌드위치", 8000),
    ("아메리카노", 4500),
    ("라떼", 5500),
]

# enum
PAYMENT_METHODS = ["CREDIT_CARD", "SIMPLE_PAY", "BANK_TRANSFER", "CASH"]
ORDER_CHANNELS  = ["OFFLINE", "ONLINE"]
RECEIPT_METHODS = ["DELIVERY", "TAKE_OUT", "TAKE_IN"]
ORDER_STATUSES  = ["PENDING","ACCEPTED","COOKING","IN_DELIVERY","COMPLETED","REJECTED","REFUNDED","FAILED"]
CANCEL_SET      = {"REJECTED","REFUNDED","FAILED"}

# 유저 풀
USER_POOL = [
    (i, random.choice(["M","F"]),
     f"{random.randint(1965,2006)}-{random.randint(1,12):02d}-{random.randint(1,28):02d}")
    for i in range(1001, 1041)  # 40명
]

# ---- 간격 bin (일) 및 가중치 ----
# 0-7, 8-30, 31-90, 91-180, 181-365
GAP_BINS = [
    (0, 7,   0.45),
    (8, 30,  0.30),
    (31, 90, 0.18),
    (91,180, 0.06),
    (181,365,0.01),  # 6~12개월
]
def sample_gap_days():
    x = random.random()
    acc = 0.0
    for lo, hi, w in GAP_BINS:
        acc += w
        if x <= acc:
            return random.randint(lo, hi)
    lo, hi, _ = GAP_BINS[-1]
    return random.randint(lo, hi)

def pick_status():
    r = random.random()
    if r < 0.86:   # 86% 완료
        return "COMPLETED"
    elif r < 0.90:
        return "REJECTED"
    elif r < 0.94:
        return "REFUNDED"
    else:
        return "FAILED"

def pick_payment():
    r = random.random()
    if r < 0.55: return "CREDIT_CARD"
    if r < 0.80: return "SIMPLE_PAY"
    if r < 0.90: return "BANK_TRANSFER"
    return "CASH"

def pick_channel():
    return "ONLINE" if random.random() < 0.8 else "OFFLINE"

def pick_receipt():
    r = random.random()
    if r < 0.70: return "DELIVERY"
    if r < 0.90: return "TAKE_OUT"
    return "TAKE_IN"

def kst_datetime_with_pattern(day_offset):
    # 최근 30일 범위에서 시간대 패턴 반영
    today_kst = datetime.now(KST).replace(hour=12, minute=0, second=0, microsecond=0)
    base = today_kst - timedelta(days=day_offset)
    r = random.random()
    if r < 0.30:
        hour = random.randint(8, 11)    # 오전: 커피
    elif r < 0.60:
        hour = random.randint(11, 14)   # 점심: 샌드위치
    else:
        hour = random.randint(18, 21)   # 저녁: 치킨/피자
    minute = random.randint(0,59)
    return base.replace(hour=hour, minute=minute)

def choose_menu(hour_kst):
    if 8 <= hour_kst <= 11:
        base = [("아메리카노", 0.45), ("라떼", 0.35), ("샌드위치", 0.20)]
    elif 11 <= hour_kst <= 14:
        base = [("샌드위치", 0.55), ("아메리카노", 0.20), ("라떼", 0.15), ("피자", 0.10)]
    else:
        base = [("후라이드 치킨", 0.40), ("양념 치킨", 0.30), ("피자", 0.25), ("아메리카노", 0.05)]
    x = random.random(); acc = 0
    for name, p in base:
        acc += p
        if x <= acc:
            price = dict(MENUS)[name]
            return name, price
    name = base[-1][0]; price = dict(MENUS)[name]
    return name, price

def generate_docs(n_days=30, avg_orders_per_day=8):
    docs = []
    # 1) 최근 30일 안의 실제 주문 생성
    for d in range(n_days):  # d=0: 오늘, 1: 어제...
        day_orders = max(1, int(random.gauss(avg_orders_per_day, 2)))
        for _ in range(day_orders):
            kst_dt = kst_datetime_with_pattern(d)
            utc_dt = kst_dt.astimezone(UTC)

            user_id, gender, birthdate = random.choice(USER_POOL)
            status = pick_status()
            payment_method = pick_payment()
            order_channel = pick_channel()
            receipt_method = pick_receipt()

            menu_name, price = choose_menu(kst_dt.hour)
            qty = 1 if random.random() < 0.75 else random.randint(2,3)
            items = [{
                "order_item_id": str(uuid.uuid4()),
                "menu_name": menu_name,
                "price": price,
                "quantity": qty
            }]
            if random.random() < 0.25:
                add = random.choice([("아메리카노",4500),("라떼",5500),("콜라",2000)])
                items.append({
                    "order_item_id": str(uuid.uuid4()),
                    "menu_name": add[0],
                    "price": add[1],
                    "quantity": 1
                })

            total_price = sum(it["price"]*it["quantity"] for it in items)
            order_id = str(uuid.uuid4())

            docs.append({
                "_id": order_id,                     # = order_id
                "is_first_order_in_store": None,     # 2차패스에서 채움
                "order_seq_in_store": None,          # 2차패스에서 채움
                "prev_order_created_at": None,       # 2차패스에서 채움

                "p_orders": {
                    "order_id": order_id,
                    "store_id": STORE_ID,
                    "user_id": f"u-{user_id}",
                    "total_price": total_price,
                    "delivery_address": "서울시 강남구 어딘가 123",
                    "order_channel": order_channel,
                    "receipt_method": receipt_method,
                    "payment_method": payment_method,
                    "order_status": status,
                    "is_refundable": status not in {"FAILED"},
                    "order_history": {"steps": ["ACCEPTED","COOKING"] + (["COMPLETED"] if status=="COMPLETED" else [status])},
                    "request_message": random.choice([None,"덜 달게","양념 많이","빨리 부탁"]),
                    "created_at": utc_dt.isoformat().replace("+00:00","Z"),
                    "updated_at": (utc_dt + timedelta(minutes=random.randint(5,45))).isoformat().replace("+00:00","Z")
                },
                "p_order_item": items,
                "p_user": {
                    "user_id": user_id,
                    "gender": gender,
                    "birthdate": birthdate
                }
            })

    # 2) 사용자×가게별 시간순 정렬 후
    #    - seq/first/prev 채우기
    #    - "이번 달 첫 주문"의 prev를 '과거(30일 밖)'로 샘플링해 다양화
    def parse_iso_z(s: str) -> datetime:
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        return datetime.fromisoformat(s).astimezone(UTC)

    buckets = {}  # key: (store_id, user_id)
    for i, d in enumerate(docs):
        k = (d["p_orders"]["store_id"], d["p_orders"]["user_id"])
        dt = parse_iso_z(d["p_orders"]["created_at"])
        buckets.setdefault(k, []).append((i, dt))

    for k, lst in buckets.items():
        lst.sort(key=lambda x: x[1])  # 시간 오름차순
        # 신규 비율 ~30%: prior_count=0 이면 완전 신규
        prior_count = 0 if random.random() < 0.30 else random.randint(1, 8)
        seq = prior_count
        prev_dt_global = None  # '전 우주적' 직전이 아니라, 우리가 아는 윈도 내 직전

        for n, (idx, dt) in enumerate(lst):
            seq += 1
            docs[idx]["order_seq_in_store"] = seq

            if n == 0:
                if prior_count == 0:
                    # 이번 달 첫 주문이 곧 '첫 주문'
                    docs[idx]["is_first_order_in_store"] = True
                    docs[idx]["prev_order_created_at"] = None
                else:
                    # 이번 달 첫 주문이지만, 이전 주문은 과거에 있었음
                    gap_days = sample_gap_days()
                    prev_dt = dt - timedelta(days=gap_days)
                    docs[idx]["is_first_order_in_store"] = False
                    docs[idx]["prev_order_created_at"] = prev_dt.isoformat().replace("+00:00","Z")
                    prev_dt_global = dt  # 현재 dt가 다음 주문의 직전으로 사용됨
            else:
                # 윈도 내 바로 직전 주문을 prev로
                docs[idx]["is_first_order_in_store"] = False
                docs[idx]["prev_order_created_at"] = lst[n-1][1].isoformat().replace("+00:00","Z")

    return docs

def main():
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    col = db[COLL_NAME]

    docs = generate_docs(n_days=30, avg_orders_per_day=8)  # 대략 200~300건
    ops = [UpdateOne({"_id": d["_id"]}, {"$set": d}, upsert=True) for d in docs]
    if ops:
        res = col.bulk_write(ops, ordered=False)
        print("upserted:", res.upserted_count, "modified:", res.modified_count)

    # 인덱스 (기존과 동일)
    col.create_index([("p_orders.store_id",1), ("p_orders.created_at",1)])
    col.create_index([("p_orders.store_id",1), ("p_orders.order_status",1), ("p_orders.created_at",1)])
    col.create_index([("p_orders.store_id",1), ("is_first_order_in_store",1), ("p_orders.created_at",1)])
    col.create_index([("p_orders.store_id",1), ("p_order_item.menu_name",1), ("p_orders.created_at",1)])
    col.create_index([("p_orders.store_id",1), ("p_orders.user_id",1), ("order_seq_in_store",1)])

if __name__ == "__main__":
    main()
