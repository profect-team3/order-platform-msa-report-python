from __future__ import annotations
import os, uuid, re, json
from dataclasses import dataclass
from datetime import datetime, timedelta, date
from typing import List, Dict, Literal
import numpy as np
from dotenv import load_dotenv
import traceback, sys

# === 3rd party ===
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import OPTICS
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import hdbscan
try:
    import kss
    USE_KSS = True
except Exception:
    USE_KSS = False

# ========= DTO =========
@dataclass
class Review:
    reviewId: str
    rating: int
    createdAt: str
    content: str

@dataclass
class Window:
    start: str
    end: str

@dataclass
class Params:
    language: str = "ko"
    topK: int = 3
    minSentenceLen: int = 5
    minClusterSize: int = 3        # HDBSCAN 파라미터

@dataclass
class Payload:
    storeId: str
    window: Window
    reviews: List[Review]
    params: Params

# ========= Sentence Split =========
def split_sentences(text: str) -> List[str]:
    if USE_KSS:
        sents = []
        for s in kss.split_sentences(text):
            s = s.strip(" \t\r\n\"'.,!?")
            if s:
                sents.append(s)
        return sents
    # fallback (단순)
    parts = re.split(r"(?<=[\.!?])\s+|[\n\r]+|(?<=요)\s+", text.strip())
    return [s.strip(" \t\r\n\"'.,!?") for s in parts if s.strip()]

# ========= Sentiment (간단 휴리스틱) =========
POS_WORDS = ["맛있","친절","깔끔","빠르","가성비","추천","만족","좋았","신선","바삭","넉넉"]
NEG_WORDS = ["늦","식었","눅눅","불친절","누락","실망","짜증","최악","별로","차갑","지연"]

def sentiment_of(sentence: str, rating: int) -> Literal["pos","neg","neu"]:
    s = sentence.lower()
    pos_hit = sum(w in s for w in POS_WORDS)
    neg_hit = sum(w in s for w in NEG_WORDS)
    if rating >= 4: pos_hit += 1
    if rating <= 2: neg_hit += 1
    if pos_hit > neg_hit: return "pos"
    if neg_hit > pos_hit: return "neg"
    return "neu"

# ========= OpenAI =========
# 환경변수: OPENAI_API_KEY
load_dotenv()
API_KEY = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key = API_KEY)

def embed_sentences(sents: List[str]) -> np.ndarray:
    # text-embedding-3-small: 1536차원, 비용↓, 품질 충분
    resp = client.embeddings.create(model="text-embedding-3-small", input=sents)
    vecs = np.array([d.embedding for d in resp.data], dtype="float32")
    # cosine 유사도용 정규화 권장
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-9, norms)
    return vecs / norms

def summarize_and_label_with_openai(sentences: List[str], label_hint: str = "") -> Dict[str, str]:
    # 구조화 출력: {label, summary}
    bullets = "\n".join(f"- {s}" for s in sentences[:12])  # 너무 길면 컷
    schema = {
        "type": "json_schema",
        "json_schema": {
            "name": "cluster_out",
            "schema": {
                "type": "object",
                "properties": {
                    "label": {"type":"string"},
                    "summary": {"type":"string"}
                },
                "required": ["label","summary"],
                "additionalProperties": False
            },
            "strict": True
        }
    }
    prompt = (
        "다음 고객 리뷰 문장들을 근거로 결과를 생성하세요.\n"
        "- label: 주제를 2~6자 내의 자연스러운 카테고리로 (예: 포장, 직원 응대, 배달 지연, 온도/식감, 가성비, 누락 등).\n"
        "- summary: 점주용 리포트에 들어갈 분석 요약. 1~2문장, 200자 이내.\n"
        "- 원문을 그대로 복사하지 말고, 의미를 유지하며 재서술할 것.\n"
        f"- 주제 힌트: {label_hint}\n"
        "문장 목록:\n" + bullets
    )
    # r = client.responses.create(
    r = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role":"system","content":"You write concise, evidence-based outputs in Korean."},
            {"role":"user","content": prompt}
        ],
        response_format=schema,
        temperature=0.3, presence_penalty=0.2, frequency_penalty=0.4,
    )
    # out = r.output_parsed
    text = r.choices[0].message.content
    out = json.loads(text)
    return {"label": out["label"].strip(), "summary": out["summary"].strip()}

# ========= Clustering (HDBSCAN; 자동 k) =========
def cluster_embeddings(X: np.ndarray, min_cluster_size: int, min_samples: int) -> List[List[int]]:
    # HDBSCAN: 노이즈(-1)를 제외한 클러스터만 반환
    labels = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                             min_samples=min_samples,
                             metric="euclidean").fit_predict(X)
    groups: Dict[int, List[int]] = {}
    for i, lab in enumerate(labels):
        if lab == -1:  # 노이즈는 일단 버림(옵션: 최근접 클러스터에 흡수)
            continue
        groups.setdefault(lab, []).append(i)
    return list(groups.values())

# --- Fallback (HDBSCAN 설치 불가 시 OPTICS로 대체)
def cluster_embeddings_fallback(X: np.ndarray, min_cluster_size: int) -> List[List[int]]:
    optics = OPTICS(min_samples=min_cluster_size, xi=0.05, min_cluster_size=min_cluster_size)
    labels = optics.fit_predict(X)
    groups: Dict[int, List[int]] = {}
    for i, lab in enumerate(labels):
        if lab == -1:  # noise
            continue
        groups.setdefault(lab, []).append(i)
    return list(groups.values())

# ========= Quote (대표 인용문) =========
def pick_quote(indices: List[int], X: np.ndarray, sents: List[str]) -> str:
    subset = X[indices]
    centroid = subset.mean(axis=0, keepdims=True)
    order = cosine_similarity(subset, centroid).ravel().argsort()[::-1]
    # 너무 짧은 문장은 피하기
    for idx in order:
        if len(sents[indices[idx]]) >= 10:
            return sents[indices[idx]]
    # 그래도 없으면 가장 긴 것으로
    return max((sents[i] for i in indices), key=len)

# ========= 파이프라인 =========

def process_payload(payload: Payload) -> Dict:
    # 1) 문장 분리 + 감성 분리
    pos_sents, neg_sents, valid_sentence_count = [], [], 0
    for rv in payload.reviews:
        sents = split_sentences(rv.content)
        for s in sents:
            if len(s) < payload.params.minSentenceLen:
                continue
            valid_sentence_count += 1
            sent = sentiment_of(s, rv.rating)
            if sent == "pos":
                pos_sents.append(s)
            elif sent == "neg":
                neg_sents.append(s)
            else:
                # 중립 문장은 rating 기반으로 부드럽게 편입
                if rv.rating >= 4:
                    pos_sents.append(s)
                elif rv.rating <= 2:
                    neg_sents.append(s)

    def build_items(sents: List[str], prefix: str, total_sentence_count: int) -> List[Dict]:
        if not sents:
            return []
        # 2) 임베딩
        X = embed_sentences(sents)

        # 3) 클러스터링
        try:
            groups = cluster_embeddings(
                X,
                min_cluster_size=payload.params.minClusterSize,
                min_samples=payload.params.minClusterSize  # min_samples를 min_cluster_size와 같게 설정하여 더 밀도 높은 클러스터를 유도
            )
        except Exception as e:
            print(f"HDBSCAN failed, falling back to OPTICS. Error: {e}", file=sys.stderr)
            groups = cluster_embeddings_fallback(X, min_cluster_size=payload.params.minClusterSize)

        if not groups:
            groups = [list(range(len(sents)))]

        # 4) 의미적으로 유사한 클러스터 병합 (임베딩 기반)
        if len(groups) > 1:
            centroids = np.array([X[idxs].mean(axis=0) for idxs in groups])
            sim_matrix = cosine_similarity(centroids)

            # [DEBUG] 클러스터 간 유사도 확인을 위한 디버그 코드
            # 이 코드를 활성화하여 터미널에서 유사도와 클러스터 내용을 직접 확인하고 임계값을 조절하세요.
            DEBUG_SIMILARITY = False # True로 바꾸면 디버그 로그 출력
            if DEBUG_SIMILARITY:
                print("\n--- [DEBUG] Cluster Similarity Scores ---")
                # 중복 출력을 피하기 위해 상단 삼각형만 순회
                for i in range(len(groups)):
                    for j in range(i + 1, len(groups)):
                        score = sim_matrix[i, j]
                        # 너무 낮은 점수는 제외하고 확인 (e.g., 0.7 이상만)
                        if score > 0.7:
                            print(f"\nSimilarity: {score:.4f} (Cluster {i} vs Cluster {j})")
                            quote_i = pick_quote(groups[i], X, sents)
                            quote_j = pick_quote(groups[j], X, sents)
                            print(f"  - Cluster {i} Quote: '{quote_i}'")
                            print(f"  - Cluster {j} Quote: '{quote_j}'")
                print("--- [DEBUG] End of Similarity Scores ---\n")

            # 임계값을 0.80으로 소폭 완화하여, 의미적으로 매우 유사하지만 벡터가 약간 다른 경우도 병합되도록 조정합니다.
            adjacency_matrix = sim_matrix > 0.8
            n_components, labels = connected_components(
                csgraph=csr_matrix(adjacency_matrix), directed=False, return_labels=True
            )
            merged_groups_dict = {i: [] for i in range(n_components)}
            for i, label in enumerate(labels):
                merged_groups_dict[label].extend(groups[i])
            groups = list(merged_groups_dict.values())

        # 5) 각 (병합된) 클러스터에 대해 요약/라벨링/인용문 생성
        cluster_items = []
        for idxs in groups:
            if not idxs: continue

            cluster_sents = [sents[i] for i in idxs]
            quote = pick_quote(idxs, X, sents)

            try:
                # TF-IDF 힌트 없이, 병합된 문장들만으로 LLM이 생성하도록 함
                gen = summarize_and_label_with_openai(cluster_sents, label_hint="")
                label = gen["label"]
                summary = gen["summary"]
            except Exception as e:
                print("\n[OpenAI ERROR] summarize_and_label failed:", e, file=sys.stderr)
                traceback.print_exc()
                # OpenAI 실패 시 의미 있는 대체값 생성
                label = "주요 의견"
                summary = f"{quote} 등 비슷한 의견이 있었습니다."

            cluster_items.append({
                "count": len(idxs),
                "label": label,
                "quote": quote,
                "summary": summary,
            })

        # 6) 랭킹: 문장 개수(count) 기준 내림차순 정렬 후 상위 K개 선택
        cluster_items.sort(key=lambda x: x["count"], reverse=True)
        top = cluster_items[:payload.params.topK]

        # 7) 출력 형식 맞추기
        items = []
        for rank, it in enumerate(top, start=1):
            items.append({
                "label": it["label"],
                "summary": it["summary"],
                "quote": it["quote"],
                "count": it["count"],
                "share": round(it["count"] / max(1, total_sentence_count), 4),
                "clusterId": f"{prefix}-{rank}"
            })
        return items

    pros = build_items(pos_sents, "pos", valid_sentence_count)
    cons = build_items(neg_sents, "neg", valid_sentence_count)

    covered = sum(it["count"] for it in pros + cons)
    result = {
        "stats": {
            "reviewCount": len(payload.reviews),
            "sentenceCount": valid_sentence_count,
            "coverage": round(covered / max(1, valid_sentence_count), 4)
        },
        "pros": pros,
        "cons": cons,
    }
    return result

# ========= Demo run =========
if __name__ == "__main__":
    now = datetime.utcnow()
    dummy_reviews = [
        # 포장 상태 긍정
        Review("r-1",5,(now-timedelta(days=1)).isoformat()+"Z","포장이 단단히 밀봉되어 국물이 전혀 새지 않았습니다."),
        Review("r-2",5,(now-timedelta(days=2)).isoformat()+"Z","음식이 흔들리지 않게 잘 포장되어 왔어요."),
        Review("r-3",4,(now-timedelta(days=3)).isoformat()+"Z","포장재가 튼튼해서 깔끔합니다!!!"),
        Review("r-4",5,(now-timedelta(days=4)).isoformat()+"Z","국물 요리인데도 포장이 완벽해서 하나도 안 샜어요"),
        Review("r-5",4,(now-timedelta(days=5)).isoformat()+"Z","포장도 꼼꼼하고 위생적으로 왔습니다."),

        # 직원 응대 긍정
        Review("r-6",5,(now-timedelta(days=1)).isoformat()+"Z","사장님이 문의 전화를 친절하게 받아주셨어요."),
        Review("r-7",5,(now-timedelta(days=2)).isoformat()+"Z","요청사항을 꼼꼼히 확인해 주셔서 만족했습니다."),
        Review("r-8",4,(now-timedelta(days=3)).isoformat()+"Z","응대가 따뜻하고 세심해서 기분이 좋았습니다."),
        Review("r-9",5,(now-timedelta(days=4)).isoformat()+"Z","사장님이 직접 전화 주셔서 친절히 안내해 주셨어요."),
        Review("r-10",5,(now-timedelta(days=5)).isoformat()+"Z","응대가 빠르고 정중했습니다."),

        # 가성비 긍정
        Review("r-11",5,(now-timedelta(days=1)).isoformat()+"Z","가격 대비 양이 많아 두 끼나 먹었어요."),
        Review("r-12",4,(now-timedelta(days=2)).isoformat()+"Z","구성이 알차고 양이 넉넉합니다."),
        Review("r-13",5,(now-timedelta(days=3)).isoformat()+"Z","이 가격에 이런 퀄리티는 훌륭합니다."),
        Review("r-14",4,(now-timedelta(days=4)).isoformat()+"Z","양이 충분해서 가성비 최고입니다."),
        Review("r-15",5,(now-timedelta(days=5)).isoformat()+"Z","가격이 저렴한데 맛과 양 모두 만족이에요."),

        # 배달 지연 불만
        Review("r-16",2,(now-timedelta(days=1)).isoformat()+"Z","배달이 약속 시간보다 15분 늦었어요."),
        Review("r-17",1,(now-timedelta(days=2)).isoformat()+"Z","예상 시간보다 20분 이상 지연됐습니다"),
        Review("r-18",2,(now-timedelta(days=3)).isoformat()+"Z","배달 지연으로 음식이 식어버렸습니다."),
        Review("r-19",1,(now-timedelta(days=4)).isoformat()+"Z","도착 예정 시간보다 한참 늦게 왔어요ㅠ"),
        Review("r-20",2,(now-timedelta(days=5)).isoformat()+"Z","배달이 지연되어 불편했습니다"),

        # 온도 유지 불만
        Review("r-21",2,(now-timedelta(days=1)).isoformat()+"Z","치킨이 따뜻하지 않고 식어 있었습니다."),
        Review("r-22",1,(now-timedelta(days=2)).isoformat()+"Z","음식이 차갑게 도착해 실망스러웠어요."),
        Review("r-23",2,(now-timedelta(days=3)).isoformat()+"Z","튀김이 눅눅하고 바삭함이 없었습니다"),
        Review("r-24",1,(now-timedelta(days=4)).isoformat()+"Z","국물이 미지근해서 맛이 덜했어요"),
        Review("r-25",2,(now-timedelta(days=5)).isoformat()+"Z","따뜻해야 할 음식이 차가워져서 별로였어요."),

        # 포장 누락 불만
        Review("r-26",1,(now-timedelta(days=1)).isoformat()+"Z","소스가 누락되어 왔습니다"),
        Review("r-27",2,(now-timedelta(days=2)).isoformat()+"Z","김치와 단무지가 빠져 있었어요."),
        Review("r-28",1,(now-timedelta(days=3)).isoformat()+"Z","반찬이 하나도 오지 않았습니다..."),
        Review("r-29",2,(now-timedelta(days=4)).isoformat()+"Z","음료가 배송되지 않았습니다."),
        Review("r-30",1,(now-timedelta(days=5)).isoformat()+"Z","추가 주문한 사이드가 빠졌어요..."),
        
        # 랜덤
        Review("r-31",5,(now-timedelta(days=2)).isoformat()+"Z","국물이 한 방울도 안 샜고 포장이 깔끔했어요. 요청사항 반영이 안 돼서 전화드렸는데 직원분도 친절했습니다~"),
        Review("r-32",2,(now-timedelta(days=3)).isoformat()+"Z","배달이 20분 늦었고 치킨이 식어서 눅눅했어요."),
        Review("r-33",4,(now-timedelta(days=4)).isoformat()+"Z","가격 대비 양이 많고 구성 알찼습니다. 가성비 최고!"),
        Review("r-34",1,(now-timedelta(days=5)).isoformat()+"Z","소스가 누락되었고 전화 응대도 불친절함"),
        Review("r-35",5,(now-timedelta(days=1)).isoformat()+"Z","포장 뚜껑이 단단히 닫혀 있어 국물이 안 샜어요. 매우 만족"),
        Review("r-36",3,(now-timedelta(days=8)).isoformat()+"Z","맛은 평범했지만 양은 괜찮았어요"),
        Review("r-37",5,(now-timedelta(days=7)).isoformat()+"Z","직원 응대가 친절하고 요청사항을 잘 반영해줬습니다"),
        Review("r-38",2,(now-timedelta(days=6)).isoformat()+"Z","예정 시간보다 늦게 도착했고 음식이 차가워요ㅜㅜ"),
    ]

    payload = Payload(
        storeId=str(uuid.uuid4()),
        window=Window(
            start=(date.today()-timedelta(days=30)).isoformat(),
            end=date.today().isoformat()
        ),
        reviews=dummy_reviews,
        params=Params(topK=3, minSentenceLen=5, minClusterSize=3)
    )

    out = process_payload(payload)
    print(json.dumps(out, ensure_ascii=False, indent=2))
