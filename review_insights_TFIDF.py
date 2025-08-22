from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timedelta, date
from typing import List, Dict, Tuple, Literal
import uuid
import json
import re
import numpy as np

import kss
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from dotenv import load_dotenv
import os
from openai import OpenAI
import hdbscan

# =======================
# Config
# =======================
OPENAI_EMBED_MODEL = "text-embedding-3-small"   # 품질/비용 균형
OPENAI_RESP_MODEL  = "gpt-4.1-mini"             # 구조화 출력 + 요약/라벨 생성
MIN_CLUSTER_SIZE   = 3                           # 싱글톤/듀오 억제 (데이터 작으면 2로)
MAX_SENT_PER_CLUSTER_FOR_LLM = 12               # 너무 길면 잘라서 LLM에 보냄
SUMMARY_MAX_CHARS  = 200

load_dotenv()
API_KEY = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key = API_KEY)

# ---------- DTO-like classes ----------
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

@dataclass
class Payload:
    storeId: str
    window: Window
    reviews: List[Review]
    params: Params

def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in kss.split_sentences(text) if s and len(s.strip()) >= 2]

POS_WORDS = ["맛있", "친절", "깔끔", "빠르", "가성비", "추천", "만족", "좋았", "신선", "바삭"]
NEG_WORDS = ["늦", "식었", "눅눅", "불친절", "누락", "실망", "짜증", "최악", "별로", "차갑", "지연"]

def sentiment_of(sentence: str, rating: int) -> Literal["pos","neg","neu"]:
    s = sentence.lower()
    pos_hit = sum(w in s for w in POS_WORDS)
    neg_hit = sum(w in s for w in NEG_WORDS)
    if rating >= 4: pos_hit += 1
    if rating <= 2: neg_hit += 1
    if pos_hit > neg_hit: return "pos"
    if neg_hit > pos_hit: return "neg"
    return "neu"

# =======================
# OpenAI Embeddings
# =======================
def embed_sentences(sentences: List[str]) -> np.ndarray:
    if not sentences:
        return np.zeros((0, 1536), dtype="float32")
    resp = client.embeddings.create(model=OPENAI_EMBED_MODEL, input=sentences)
    vecs = np.array([d.embedding for d in resp.data], dtype="float32")
    # cosine 안정화를 위해 정규화
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1e-9
    return vecs / norms

# =======================
# HDBSCAN Cluster
# =======================
def cluster_embeddings(X: np.ndarray, min_cluster_size: int = MIN_CLUSTER_SIZE) -> List[List[int]]:
    n = X.shape[0]
    if n == 0:
        return []
    # n이 작을 땐 min_cluster_size를 자동 완화
    mcs = min_cluster_size if n >= min_cluster_size else max(2, n // 2)
    labels = hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=1, metric="euclidean").fit_predict(X)

    groups: Dict[int, List[int]] = {}
    for i, lab in enumerate(labels):
        if lab == -1:
            continue
        groups.setdefault(lab, []).append(i)

    clusters = list(groups.values())

    # 노이즈(-1) 포인트가 있고 유효 클러스터가 있다면 가장 유사한 클러스터에 흡수
    noise_idx = [i for i, lab in enumerate(labels) if lab == -1]
    if noise_idx and clusters:
        centroids = [X[idxs].mean(axis=0, keepdims=True) for idxs in clusters]
        C = np.vstack(centroids)
        sims = cosine_similarity(X[noise_idx], C)  # (noise, clusters)
        best = sims.argmax(axis=1)
        for ni, bi in zip(noise_idx, best):
            clusters[bi].append(ni)

    # 길이 순 정렬
    clusters.sort(key=len, reverse=True)
    return clusters

# =======================
# Representative Quote (근거 문장)
# =======================
def pick_quote(indices: List[int], X: np.ndarray, sentences: List[str]) -> str:
    if not indices:
        return ""
    subset = X[indices]
    centroid = subset.mean(axis=0, keepdims=True)
    order = cosine_similarity(subset, centroid).ravel().argsort()[::-1]
    # 너무 짧은 문장은 제외하고 고르기
    for i in order:
        cand = sentences[indices[i]]
        if len(cand) >= 10:
            return cand
    return sentences[indices[order[0]]]

# =======================
# Labeling (생성 + 규칙 보정)
# =======================
DOMAIN_LABELS = {
    "포장": ["포장","밀봉","용기","새다","누수","깔끔"],
    "직원 응대": ["응대","친절","사장","전화","안내"],
    "가성비": ["가성비","가격 대비","양","구성","넉넉"],
    "배달 지연": ["늦","지연","딜레이","예정 시간","약속 시간"],
    "온도/식감": ["식었","차갑","눅눅","바삭","온도"],
    "누락": ["누락","빠짐","없었","안 왔","누락되"],
}

def label_from_counts(docs: List[str]) -> str:
    # BERTopic 스타일 간소화: 상위 n-gram TF-IDF에서 조사/경어 제거
    vec = CountVectorizer(ngram_range=(1,2), min_df=1, token_pattern=r"(?u)\b\w+\b")
    X = vec.fit_transform(docs)
    tfidf = TfidfTransformer().fit_transform(X)
    vocab = np.array(vec.get_feature_names_out())
    scores = tfidf.sum(axis=0).A1
    order = scores.argsort()[::-1][:50]

    def clean(term: str) -> str:
        t = term
        for suf in ("보다","에서","으로","하고","과","와","은","는","이","가","을","를","의","도","만","께서","세요","습니다","했어요"):
            if t.endswith(suf) and len(t) > len(suf)+1:
                t = t[:-len(suf)]
        return t

    candidates = []
    for i in order:
        t = clean(vocab[i]).strip()
        if len(t) >= 2 and not t.isdigit() and t not in candidates:
            candidates.append(t)

    # 도메인 라벨 매핑 우선
    joined = " ".join(candidates[:10])
    for label, hints in DOMAIN_LABELS.items():
        if any(h in joined for h in hints):
            return label
    return candidates[0] if candidates else "일반 의견"

def gen_label_and_summary(sentences: List[str], label_hint: str = "") -> Tuple[str, str]:
    # OpenAI로 라벨과 요약을 함께 생성(스키마 강제)
    bullets = "\n".join(f"- {s}" for s in sentences[:MAX_SENT_PER_CLUSTER_FOR_LLM])
    schema = {
        "type": "json_schema",
        "json_schema": {
            "name": "cluster_out",
            "schema": {
                "type": "object",
                "properties": {
                    "label":   {"type": "string"},
                    "summary": {"type": "string"},
                },
                "required": ["label","summary"],
                "additionalProperties": False
            },
            "strict": True
        }
    }
    prompt = (
        "다음 고객 리뷰 문장들을 근거로 결과를 작성하세요.\n"
        "- label: 주제를 2~6자로 간결하게(예: 포장, 직원 응대, 배달 지연, 온도/식감, 가성비, 누락 등 카테고리명).\n"
        f"- summary: 1~2문장, {SUMMARY_MAX_CHARS}자 이내. 원문을 그대로 복사하지 말고, 의미를 유지하며 점주가 보기 쉬운 분석 문장으로.\n"
        f"- 주제 힌트(없으면 무시): {label_hint}\n"
        "문장 목록:\n" + bullets
    )
    r = client.responses.create(
        model=OPENAI_RESP_MODEL,
        input=[
            {"role":"system","content":"You are an analyst. Write concise, evidence-based outputs only from the given sentences."},
            {"role":"user","content": prompt}
        ],
        response_format=schema,
        temperature=0.3, presence_penalty=0.2, frequency_penalty=0.4,
    )
    out = r.output_parsed
    # 라벨 사후 보정 (도메인 맵에 더 잘 맞으면 교체)
    rule_label = label_from_counts(sentences)
    final_label = out["label"].strip() if out.get("label") else rule_label
    # 너무 길거나 어색하면 규칙 라벨로 교체
    if len(final_label) > 8 or final_label.count(" ") >= 2:
        final_label = rule_label
    return final_label, out["summary"].strip()

# =======================
# Build items: end-to-end
# =======================
def build_items(sentences: List[str], topK: int, prefix: str) -> List[Dict]:
    if not sentences:
        return []
    X = embed_sentences(sentences)
    clusters = cluster_embeddings(X, min_cluster_size=MIN_CLUSTER_SIZE)

    # 클러스터 메타 구축
    cluster_meta = []
    for idxs in clusters:
        sents = [sentences[i] for i in idxs]
        quote = pick_quote(idxs, X, sentences)
        try:
            label_hint = label_from_counts(sents)
            label, summary = gen_label_and_summary(sents, label_hint=label_hint)
        except Exception as e:
            # OpenAI 실패 시 안전한 대체
            label = label_from_counts(sents)
            # 대표문장 2개를 이어붙여 간단 요약
            two = sents[:2]
            summary = two[0] if len(two) == 1 else f"{two[0]} 또한 {two[1]}"
        cluster_meta.append((len(idxs), label, summary, quote))

    # 규모순 정렬 후 TopK
    cluster_meta.sort(key=lambda x: x[0], reverse=True)
    selected = cluster_meta[:topK]

    total_in_selected = sum(c for c,_,_,_ in selected) or 1
    items = []
    for rank, (cnt, label, summary, quote) in enumerate(selected, start=1):
        items.append({
            "label": label,
            "summary": summary[:SUMMARY_MAX_CHARS].rstrip(),
            "quote": quote,
            "count": cnt,
            "share": round(cnt / total_in_selected, 4),  # 선택된 탑K 내 비중(원하면 전체 문장수로 바꿔도 됨)
            "clusterId": f"{prefix}-{rank}"
        })
    return items

# =======================
# Main pipeline
# =======================
def process_payload(payload: Payload) -> Dict:
    pos_sents, neg_sents, valid_count = [], [], 0

    for rv in payload.reviews:
        sents = split_sentences(rv.content or "")
        for s in sents:
            if len(s) < payload.params.minSentenceLen:
                continue
            valid_count += 1
            sent = sentiment_of(s, rv.rating)
            if sent == "pos":
                pos_sents.append(s)
            elif sent == "neg":
                neg_sents.append(s)
            else:
                # 중립은 평점으로 부드럽게 편입 (데이터 적을 때 군집 안정화)
                if rv.rating >= 4: pos_sents.append(s)
                elif rv.rating <= 2: neg_sents.append(s)

    pros = build_items(pos_sents, payload.params.topK, "pos")
    cons = build_items(neg_sents, payload.params.topK, "neg")

    covered = sum(it["count"] for it in pros + cons)
    coverage = round(covered / max(1, valid_count), 4)

    return {
        "stats": {
            "reviewCount": len(payload.reviews),
            "sentenceCount": valid_count,
            "coverage": coverage
        },
        "pros": pros,
        "cons": cons
    }

# =======================
# Demo run
# =======================
if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("❗ Set OPENAI_API_KEY in your environment first.")

    now = datetime.utcnow()
    dummy_reviews = [
        # 포장 상태 긍정
        Review("r-1",5,(now-timedelta(days=1)).isoformat()+"Z","포장이 단단히 밀봉되어 국물이 전혀 새지 않았습니다."),
        Review("r-2",5,(now-timedelta(days=2)).isoformat()+"Z","음식이 흔들리지 않게 잘 포장되어 왔어요."),
        Review("r-3",4,(now-timedelta(days=3)).isoformat()+"Z","포장재가 튼튼해서 깔끔했습니다."),
        Review("r-4",5,(now-timedelta(days=4)).isoformat()+"Z","국물 요리인데도 포장이 완벽해 흘림이 없었어요."),
        Review("r-5",4,(now-timedelta(days=5)).isoformat()+"Z","포장도 꼼꼼하고 위생적으로 왔습니다."),

        # 직원 응대 긍정
        Review("r-6",5,(now-timedelta(days=1)).isoformat()+"Z","직원분이 문의 전화를 친절하게 받아주셨어요."),
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
        Review("r-17",1,(now-timedelta(days=2)).isoformat()+"Z","예상 시간보다 20분 이상 지연됐습니다."),
        Review("r-18",2,(now-timedelta(days=3)).isoformat()+"Z","배달 지연으로 음식이 식어버렸습니다."),
        Review("r-19",1,(now-timedelta(days=4)).isoformat()+"Z","도착 예정 시간보다 한참 늦게 왔어요."),
        Review("r-20",2,(now-timedelta(days=5)).isoformat()+"Z","배달이 지연되어 불편했습니다."),

        # 온도 유지 불만
        Review("r-21",2,(now-timedelta(days=1)).isoformat()+"Z","치킨이 따뜻하지 않고 식어 있었습니다."),
        Review("r-22",1,(now-timedelta(days=2)).isoformat()+"Z","음식이 차갑게 도착해 실망스러웠어요."),
        Review("r-23",2,(now-timedelta(days=3)).isoformat()+"Z","튀김이 눅눅하고 바삭함이 없었습니다."),
        Review("r-24",1,(now-timedelta(days=4)).isoformat()+"Z","국물이 미지근해서 맛이 덜했어요."),
        Review("r-25",2,(now-timedelta(days=5)).isoformat()+"Z","따뜻해야 할 음식이 차가워져서 별로였어요."),

        # 포장 누락 불만
        Review("r-26",1,(now-timedelta(days=1)).isoformat()+"Z","소스가 누락되어 왔습니다."),
        Review("r-27",2,(now-timedelta(days=2)).isoformat()+"Z","김치와 단무지가 빠져 있었어요."),
        Review("r-28",1,(now-timedelta(days=3)).isoformat()+"Z","반찬이 하나도 오지 않았습니다."),
        Review("r-29",2,(now-timedelta(days=4)).isoformat()+"Z","음료가 배송되지 않았습니다."),
        Review("r-30",1,(now-timedelta(days=5)).isoformat()+"Z","추가 주문한 사이드가 빠졌어요."),
    ]

    payload = Payload(
        storeId=str(uuid.uuid4()),
        window=Window(
            start=(date.today()-timedelta(days=30)).isoformat(),
            end=date.today().isoformat()
        ),
        reviews=dummy_reviews,
        params=Params(language="ko", topK=3, minSentenceLen=5)
    )

    result = process_payload(payload)
    print(json.dumps(result, ensure_ascii=False, indent=2))



# KOR_STOP_SUFFIX = ("보다","에서","으로","하고","과","와","은","는","이","가","을","를","의","도","만")
# def extract_label(sentences: List[str]) -> str:
#     vec = TfidfVectorizer(max_features=1500, ngram_range=(1,2),
#                           token_pattern=r"(?u)\b\w+\b")
#     X = vec.fit_transform(sentences)
#     if X.shape[1] == 0:
#         return "일반 의견"
#     scores = np.asarray(X.sum(axis=0)).ravel()
#     vocab = np.array(vec.get_feature_names_out())
#     for idx in scores.argsort()[::-1]:
#         term = vocab[idx].strip()
#         if len(term) < 2 or term.isdigit(): continue
#         if term.endswith(KOR_STOP_SUFFIX): continue
#         return term[:20]
#     return "일반 의견"


# def representative_sentence(sentences: List[str]) -> str:
#     vec = TfidfVectorizer(max_features=2000, ngram_range=(1,2))
#     X = vec.fit_transform(sentences)
#     if X.shape[0] == 0:
#         return sentences[0] if sentences else ""
#     centroid = np.asarray(X.mean(axis=0)).ravel().reshape(1, -1)
#     sims = cosine_similarity(X, centroid).ravel()
#     return sentences[int(np.argmax(sims))]

# # def cluster_sentences(sentences: List[str], k: int) -> List[List[int]]:
# #     if len(sentences) <= k:
# #         return [[i] for i in range(len(sentences))]
# #     vec = TfidfVectorizer(max_features=3000, ngram_range=(1,2))
# #     X = vec.fit_transform(sentences)
# #     k = max(1, min(k, int(np.sqrt(len(sentences))) + 2))
# #     km = KMeans(n_clusters=k, n_init=10, random_state=42)
# #     labels = km.fit_predict(X)
# #     clusters: Dict[int, List[int]] = {}
# #     for idx, lab in enumerate(labels):
# #         clusters.setdefault(lab, []).append(idx)
# #     return list(clusters.values())

# def cluster_sentences(sentences: List[str], k: int) -> List[List[int]]:
#     n = len(sentences)
#     if n == 0: return []
#     if n <= k: return [[i] for i in range(n)]

#     # ✅ n이 적으면 K를 줄여서 과분할 방지
#     if n <= 12:
#         k = max(1, min(k, n // 3))
#     else:
#         k = max(1, min(k, int(np.sqrt(n)) + 1))

#     vec = TfidfVectorizer(max_features=3000, ngram_range=(1,2))
#     X = vec.fit_transform(sentences)
#     km = KMeans(n_clusters=k, n_init=10, random_state=42)
#     labels = km.fit_predict(X)

#     clusters: Dict[int, List[int]] = {}
#     for idx, lab in enumerate(labels):
#         clusters.setdefault(lab, []).append(idx)

#     groups = list(clusters.values())

#     # ✅ 싱글톤 병합: 1개짜리는 가장 큰 클러스터에 흡수
#     singletons = [g[0] for g in groups if len(g) == 1]
#     bigs = [g for g in groups if len(g) > 1]
#     if singletons and bigs:
#         biggest = max(bigs, key=len)
#         for s in singletons:
#             biggest.append(s)
#         groups = [g for g in bigs if g is not biggest] + [biggest]

#     return groups

# def summarize_with_openai(sentences: list[str], label_hint: str = "", lang: str = "ko") -> str:
#     """
#     클러스터 문장 목록을 전달하면, 원문에 근거한 한 줄 요약을 생성.
#     - 원문에 없는 내용 금지
#     - 1~2문장, 120자 이내
#     """
#     joined = "\n".join(f"- {s}" for s in sentences[:50])  # 과도한 길이는 컷
#     schema = {
#         "type": "json_schema",
#         "json_schema": {
#             "name": "cluster_summary",
#             "schema": {
#                 "type": "object",
#                 "properties": {
#                     "summary": {"type":"string"}
#                 },
#                 "required": ["summary"],
#                 "additionalProperties": False
#             },
#             "strict": True
#         }
#     }
#     prompt = (
#         f"다음 리뷰 문장 목록을 한 주제의 요약으로 1~2문장(최대 120자)으로 한국어로 작성해줘.\n"
#         f"해당 리뷰를 받은 가게의 점주에게 제공할 리포트에 들어가는 문장이야.\n"
#         f"원문에 없는 정보는 넣지 말고, 사실관계만 간결하게.\n"
#         f"원문 문장을 그대로 복사하지 말고 자연스럽게 바꿔 말하세요.\n"
#         f"주제 힌트: {label_hint}\n"
#         f"문장 목록:\n{joined}\n"
#     )

#     resp = client.responses.create(
#         model="gpt-4.1-mini",
#         input=[
#             {"role":"system","content":"You are a careful summarizer. Never invent facts beyond the provided sentences."},
#             {"role":"user","content": prompt}
#         ],
#         response_format=schema,
#     )
#     # Responses API는 structured outputs로 JSON을 보냄
#     return resp.output_parsed["summary"]


# def build_items(sentences: List[str], total_sentences: int, topK: int, prefix: str):
#     if not sentences: return []
#     clusters = cluster_sentences(sentences, k=max(1, min(6, topK*2)))
#     cluster_info: List[Tuple[int,List[str],str,str]] = []
#     # debug: 클러스터 크기 확인
#     for ci, idxs in enumerate(clusters):
#         print(f"[DEBUG] {prefix} cluster#{ci} size={len(idxs)} sample='{sentences[idxs[0]][:40]}'")

#     for idxs in clusters:
#         sents = [sentences[i] for i in idxs]
#         label = extract_label(sents)
#         quote = representative_sentence(sents)
#         try:
#             summary = summarize_with_openai(sents, label_hint=label)
#         except Exception:
#             summary = quote
#         cluster_info.append((len(sents), sents, label, quote, summary))
#     cluster_info.sort(key=lambda x: x[0], reverse=True)
#     selected = cluster_info[:topK]
#     items = []
#     for rank, (cnt, sents, label, quote, summary) in enumerate(selected, start=1):
#         items.append({
#             "label": label,
#             "summary": summary,
#             "quote": quote,
#             "count": cnt,
#             "share": round(cnt/max(1,total_sentences),4),
#             "clusterId": f"{prefix}-{rank}"
#         })
#     return items

# def process_payload(payload: Payload) -> Dict:
#     pos_sents, neg_sents = [], []
#     valid_sent_count = 0
#     for rv in payload.reviews:
#         sents = split_sentences(rv.content)
#         for s in sents:
#             if len(s) < payload.params.minSentenceLen: continue
#             valid_sent_count += 1
#             sent = sentiment_of(s, rv.rating)
#             if sent=="pos": pos_sents.append(s)
#             elif sent=="neg": neg_sents.append(s)

#     pros = build_items(pos_sents, valid_sent_count, payload.params.topK, "pos")
#     cons = build_items(neg_sents, valid_sent_count, payload.params.topK, "neg")
#     covered = sum(it["count"] for it in pros+cons)

#     return {
#         "stats": {
#             "reviewCount": len(payload.reviews),
#             "sentenceCount": valid_sent_count,
#             "coverage": round(covered/max(1,valid_sent_count),4)
#         },
#         "pros": pros,
#         "cons": cons,
#         "diagnostics": {
#             "modelVersions": {
#                 "sentiment":"heuristic-ko-0.1",
#                 "embedding":"tfidf-ngrams-0.1",
#                 "labeling":"tfidf-top-ngram-0.1",
#                 "summarizer":"extractive-closest-0.1"
#             }
#         }
#     }

# # ---------- Dummy run ----------
# if __name__=="__main__":
#     now = datetime.utcnow()
#     # dummy_reviews = [
#     #     Review("r-1",5,(now-timedelta(days=2)).isoformat()+"Z","국물이 한 방울도 안 샜고 포장이 깔끔했어요. 요청사항 반영이 안 돼서 전화드렸는데 직원분도 친절했습니다~"),
#     #     Review("r-2",2,(now-timedelta(days=3)).isoformat()+"Z","배달이 20분 늦었고 치킨이 식어서 눅눅했어요."),
#     #     Review("r-3",4,(now-timedelta(days=4)).isoformat()+"Z","가격 대비 양이 많고 구성 알찼습니다. 가성비 최고!"),
#     #     Review("r-4",1,(now-timedelta(days=5)).isoformat()+"Z","소스가 누락되었고 전화 응대도 불친절함"),
#     #     Review("r-5",5,(now-timedelta(days=1)).isoformat()+"Z","포장 뚜껑이 단단히 닫혀 있어 국물이 안 샜어요. 매우 만족"),
#     #     Review("r-6",3,(now-timedelta(days=8)).isoformat()+"Z","맛은 평범했지만 양은 괜찮았어요"),
#     #     Review("r-7",5,(now-timedelta(days=7)).isoformat()+"Z","직원 응대가 친절하고 요청사항을 잘 반영해줬습니다"),
#     #     Review("r-8",2,(now-timedelta(days=6)).isoformat()+"Z","예정 시간보다 늦게 도착했고 음식이 차가워요"),
#     # ]
    # dummy_reviews = [
    #     # 포장 상태 긍정
    #     Review("r-1",5,(now-timedelta(days=1)).isoformat()+"Z","포장이 단단히 밀봉되어 국물이 전혀 새지 않았습니다."),
    #     Review("r-2",5,(now-timedelta(days=2)).isoformat()+"Z","음식이 흔들리지 않게 잘 포장되어 왔어요."),
    #     Review("r-3",4,(now-timedelta(days=3)).isoformat()+"Z","포장재가 튼튼해서 깔끔했습니다."),
    #     Review("r-4",5,(now-timedelta(days=4)).isoformat()+"Z","국물 요리인데도 포장이 완벽해 흘림이 없었어요."),
    #     Review("r-5",4,(now-timedelta(days=5)).isoformat()+"Z","포장도 꼼꼼하고 위생적으로 왔습니다."),

    #     # 직원 응대 긍정
    #     Review("r-6",5,(now-timedelta(days=1)).isoformat()+"Z","직원분이 문의 전화를 친절하게 받아주셨어요."),
    #     Review("r-7",5,(now-timedelta(days=2)).isoformat()+"Z","요청사항을 꼼꼼히 확인해 주셔서 만족했습니다."),
    #     Review("r-8",4,(now-timedelta(days=3)).isoformat()+"Z","응대가 따뜻하고 세심해서 기분이 좋았습니다."),
    #     Review("r-9",5,(now-timedelta(days=4)).isoformat()+"Z","사장님이 직접 전화 주셔서 친절히 안내해 주셨어요."),
    #     Review("r-10",5,(now-timedelta(days=5)).isoformat()+"Z","응대가 빠르고 정중했습니다."),

    #     # 가성비 긍정
    #     Review("r-11",5,(now-timedelta(days=1)).isoformat()+"Z","가격 대비 양이 많아 두 끼나 먹었어요."),
    #     Review("r-12",4,(now-timedelta(days=2)).isoformat()+"Z","구성이 알차고 양이 넉넉합니다."),
    #     Review("r-13",5,(now-timedelta(days=3)).isoformat()+"Z","이 가격에 이런 퀄리티는 훌륭합니다."),
    #     Review("r-14",4,(now-timedelta(days=4)).isoformat()+"Z","양이 충분해서 가성비 최고입니다."),
    #     Review("r-15",5,(now-timedelta(days=5)).isoformat()+"Z","가격이 저렴한데 맛과 양 모두 만족이에요."),

    #     # 배달 지연 불만
    #     Review("r-16",2,(now-timedelta(days=1)).isoformat()+"Z","배달이 약속 시간보다 15분 늦었어요."),
    #     Review("r-17",1,(now-timedelta(days=2)).isoformat()+"Z","예상 시간보다 20분 이상 지연됐습니다."),
    #     Review("r-18",2,(now-timedelta(days=3)).isoformat()+"Z","배달 지연으로 음식이 식어버렸습니다."),
    #     Review("r-19",1,(now-timedelta(days=4)).isoformat()+"Z","도착 예정 시간보다 한참 늦게 왔어요."),
    #     Review("r-20",2,(now-timedelta(days=5)).isoformat()+"Z","배달이 지연되어 불편했습니다."),

    #     # 온도 유지 불만
    #     Review("r-21",2,(now-timedelta(days=1)).isoformat()+"Z","치킨이 따뜻하지 않고 식어 있었습니다."),
    #     Review("r-22",1,(now-timedelta(days=2)).isoformat()+"Z","음식이 차갑게 도착해 실망스러웠어요."),
    #     Review("r-23",2,(now-timedelta(days=3)).isoformat()+"Z","튀김이 눅눅하고 바삭함이 없었습니다."),
    #     Review("r-24",1,(now-timedelta(days=4)).isoformat()+"Z","국물이 미지근해서 맛이 덜했어요."),
    #     Review("r-25",2,(now-timedelta(days=5)).isoformat()+"Z","따뜻해야 할 음식이 차가워져서 별로였어요."),

    #     # 포장 누락 불만
    #     Review("r-26",1,(now-timedelta(days=1)).isoformat()+"Z","소스가 누락되어 왔습니다."),
    #     Review("r-27",2,(now-timedelta(days=2)).isoformat()+"Z","김치와 단무지가 빠져 있었어요."),
    #     Review("r-28",1,(now-timedelta(days=3)).isoformat()+"Z","반찬이 하나도 오지 않았습니다."),
    #     Review("r-29",2,(now-timedelta(days=4)).isoformat()+"Z","음료가 배송되지 않았습니다."),
    #     Review("r-30",1,(now-timedelta(days=5)).isoformat()+"Z","추가 주문한 사이드가 빠졌어요."),
    # ]


#     payload = Payload(
#         storeId=str(uuid.uuid4()),
#         window=Window(
#             start=(date.today()-timedelta(days=30)).isoformat(),
#             end=date.today().isoformat()
#         ),
#         reviews=dummy_reviews,
#         params=Params()
#     )

#     result = process_payload(payload)
#     print(json.dumps(result,ensure_ascii=False,indent=2))
