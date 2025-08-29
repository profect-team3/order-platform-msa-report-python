from __future__ import annotations
import os, re, json
from typing import List, Dict, Literal
import numpy as np
from dotenv import load_dotenv
import traceback, sys
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import hdbscan

# 공유 클라이언트를 임포트하여 사용
from ..utils.openai_client import client
from ..models import ReviewPayload

try:
    import kss
    USE_KSS = True
except ImportError:
    USE_KSS = False

# === Sentence Split ===
def _split_sentences(text: str) -> List[str]:
    if USE_KSS:
        sents = []
        for s in kss.split_sentences(text):
            s = s.strip(" \t\r\n\"'.,!?")
            if s:
                sents.append(s)
        return sents
    # fallback
    parts = re.split(r"(?<=[\.!?])\s+|[\n\r]+|(?<=요)\s+", text.strip())
    return [s.strip(" \t\r\n\"'.,!?") for s in parts if s.strip()]

# === Sentiment ===
POS_WORDS = ["맛있","친절","깔끔","빠르","가성비","추천","만족","좋았","신선","바삭","넉넉"]
NEG_WORDS = ["늦","식었","눅눅","불친절","누락","실망","짜증","최악","별로","차갑","지연"]

def _sentiment_of(sentence: str, rating: int) -> Literal["pos","neg","neu"]:
    s = sentence.lower()
    pos_hit = sum(w in s for w in POS_WORDS)
    neg_hit = sum(w in s for w in NEG_WORDS)
    if rating >= 4: pos_hit += 1
    if rating <= 2: neg_hit += 1
    if pos_hit > neg_hit: return "pos"
    if neg_hit > pos_hit: return "neg"
    return "neu"

def _embed_sentences(sents: List[str]) -> np.ndarray:
    resp = client.embeddings.create(model="text-embedding-3-small", input=sents)
    vecs = np.array([d.embedding for d in resp.data], dtype="float32")
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-9, norms)
    return vecs / norms

def _summarize_and_label_with_openai(sentences: List[str], label_hint: str = "") -> Dict[str, str]:
    bullets = "\n".join(f"- {s}" for s in sentences[:12])
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
        "- label: 주제를 2~8자 내의 자연스러운 카테고리로 (예: 포장, 직원 응대, 배달 지연, 온도/식감, 가성비, 누락 등).\n"
        "- summary: 점주용 리포트에 들어갈 분석 요약. 1~2문장, 200자 이내.\n"
        "- 원문을 그대로 복사하지 말고, 의미를 유지하며 재서술할 것.\n"
        f"- 주제 힌트: {label_hint}\n"
        "문장 목록:\n" + bullets
    )
    r = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role":"system","content":"You write concise, evidence-based outputs in Korean."},
            {"role":"user","content": prompt}
        ],
        response_format=schema,
        temperature=0.3, presence_penalty=0.2, frequency_penalty=0.4,
    )
    text = r.choices[0].message.content
    out = json.loads(text)
    return {"label": out["label"].strip(), "summary": out["summary"].strip()}

# === Clustering (HDBSCAN) ===
def _cluster_embeddings(X: np.ndarray, min_cluster_size: int, min_samples: int) -> List[List[int]]:
    labels = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                             min_samples=min_samples,
                             metric="euclidean").fit_predict(X)
    groups: Dict[int, List[int]] = {}
    for i, lab in enumerate(labels):
        if lab == -1:
            continue
        groups.setdefault(lab, []).append(i)
    return list(groups.values())

# === Quote ===
def _pick_quote(indices: List[int], X: np.ndarray, sents: List[str]) -> str:
    subset = X[indices]
    centroid = subset.mean(axis=0, keepdims=True)
    order = cosine_similarity(subset, centroid).ravel().argsort()[::-1]
    for idx in order:
        if len(sents[indices[idx]]) >= 10:
            return sents[indices[idx]]
    return max((sents[i] for i in indices), key=len)


# === Pipeline ===

def _build_items(sents: List[str], prefix: str, total_sentence_count: int, top_k: int, min_cluster_size: int) -> List[Dict]:
    if not sents:
        return []
    
    X = _embed_sentences(sents)

    try:
        groups = _cluster_embeddings(
            X,
            min_cluster_size=min_cluster_size,
            min_samples=min_cluster_size
        )
    except Exception as e:
        print(f"HDBSCAN failed, falling back to simple grouping. Error: {e}", file=sys.stderr)
        groups = [list(range(len(sents)))]

    if not groups:
        groups = [list(range(len(sents)))]

    if len(groups) > 1:
        centroids = np.array([X[idxs].mean(axis=0) for idxs in groups])
        sim_matrix = cosine_similarity(centroids)
        adjacency_matrix = sim_matrix > 0.8
        n_components, labels = connected_components(
            csgraph=csr_matrix(adjacency_matrix), directed=False, return_labels=True
        )
        merged_groups_dict = {i: [] for i in range(n_components)}
        for i, label in enumerate(labels):
            merged_groups_dict[label].extend(groups[i])
        groups = list(merged_groups_dict.values())

    cluster_items = []
    for idxs in groups:
        if not idxs: continue

        cluster_sents = [sents[i] for i in idxs]
        quote = _pick_quote(idxs, X, sents)

        try:
            gen = _summarize_and_label_with_openai(cluster_sents, label_hint="")
            label = gen["label"]
            summary = gen["summary"]
        except Exception as e:
            print("\n[OpenAI ERROR] summarize_and_label failed:", e, file=sys.stderr)
            traceback.print_exc()
            label = "주요 의견"
            summary = f"{quote} 등 비슷한 의견이 있었습니다."

        cluster_items.append({
            "count": len(idxs),
            "label": label,
            "quote": quote,
            "summary": summary,
        })

    cluster_items.sort(key=lambda x: x["count"], reverse=True)
    top = cluster_items[:top_k]

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

def analyze_reviews(
    review_payload: ReviewPayload,
    top_k: int = 3,
    min_sentence_len: int = 5,
    min_cluster_size: int = 3
) -> Dict:
    """리뷰 페이로드를 받아 긍/부정에 대한 분석글 생성"""
    pos_sents, neg_sents, valid_sentence_count = [], [], 0
    for rv in review_payload.reviews:
        sents = _split_sentences(rv.content)
        for s in sents:
            if len(s) < min_sentence_len:
                continue
            valid_sentence_count += 1
            sent = _sentiment_of(s, rv.rating)
            if sent == "pos":
                pos_sents.append(s)
            elif sent == "neg":
                neg_sents.append(s)
            else:
                if rv.rating >= 4:
                    pos_sents.append(s)
                elif rv.rating <= 2:
                    neg_sents.append(s)

    pros = _build_items(pos_sents, "pos", valid_sentence_count, top_k, min_cluster_size)
    cons = _build_items(neg_sents, "neg", valid_sentence_count, top_k, min_cluster_size)

    covered = sum(it["count"] for it in pros + cons)
    result = {
        "stats": {
            "reviewCount": len(review_payload.reviews),
            "sentenceCount": valid_sentence_count,
            "coverage": round(covered / max(1, valid_sentence_count), 4) if valid_sentence_count > 0 else 0
        },
        "pros": pros,
        "cons": cons,
    }
    return result