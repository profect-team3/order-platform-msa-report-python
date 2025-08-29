from app.services.review_analyzer import _embed_sentences, analyze_reviews
from app.models import ReviewPayload, ReviewRow
from datetime import datetime, timedelta
import random
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
import kss
import warnings
warnings.filterwarnings('ignore')

# 더미 리뷰 템플릿들 (이전과 동일)
positive_templates = [
    "맛있고 포장이 깔끔했어요. 배달도 빨랐습니다.",
    "양이 많고 가성비가 좋았어요. 추천합니다!",
    "직원분들이 친절하고 요청사항을 잘 반영해줬습니다.",
    "포장이 튼튼하고 국물이 안 샜어요. 매우 만족!",
    "음식이 따뜻하게 잘 도착했고 맛도 좋았어요.",
    "신선한 재료를 사용한 것 같고 맛이 훌륭했습니다.",
    "배달이 예상보다 빨랐고 음식도 뜨거웠어요.",
    "가격 대비 양이 많고 맛도 좋아서 만족스러웠습니다.",
    "직원 응대가 친절하고 음식 포장도 깔끔했어요.",
    "주문한 메뉴가 정확히 왔고 맛도 예상보다 좋았습니다.",
]

negative_templates = [
    "배달이 늦었고 음식이 식어서 왔어요. 실망입니다.",
    "소스가 누락되었고 직원 응대도 불친절했습니다.",
    "예정 시간보다 늦게 도착했고 음식이 차가워요.",
    "포장이 허술해서 국물이 샜고 지저분했어요.",
    "주문한 메뉴가 빠졌는데 확인도 제대로 안 해주네요.",
    "음식이 너무 짜고 양도 적어서 실망스러웠어요.",
    "배달원이 불친절하고 음식도 엉망이었습니다.",
    "1시간이나 늦게 와서 음식이 완전히 식었어요.",
    "주문 내용과 다른 음식이 왔는데 바꿔주지도 않네요.",
    "음식 포장이 엉망이고 맛도 별로였어요.",
]

def generate_dummy_embeddings(texts):
    """더미 임베딩 생성 (1536차원)"""
    np.random.seed(42)  # 재현가능한 결과를 위해
    n_texts = len(texts)
    n_dims = 1536
    
    # 기본 랜덤 임베딩
    embeddings = np.random.normal(0, 1, (n_texts, n_dims))
    
    # 텍스트의 특성에 따라 임베딩을 조정
    for i, text in enumerate(texts):
        # 문자열로 처리
        if isinstance(text, str):
            text_lower = text.lower()
        else:
            text_lower = str(text).lower()
        
        # 긍정적 키워드들에 따른 조정
        if any(word in text_lower for word in ['맛있', '좋았', '만족', '추천', '훌륭', '빨랐', '친절', '깔끔']):
            embeddings[i, :100] += np.random.normal(2, 0.5, 100)  # 긍정 클러스터
        
        # 부정적 키워드들에 따른 조정  
        elif any(word in text_lower for word in ['실망', '늦었', '식어서', '불친절', '엉망', '별로']):
            embeddings[i, 100:200] += np.random.normal(2, 0.5, 100)  # 부정 클러스터
        
        # 배달 관련
        if any(word in text_lower for word in ['배달', '도착', '빨랐', '늦었']):
            embeddings[i, 200:300] += np.random.normal(1.5, 0.5, 100)
            
        # 포장 관련
        if any(word in text_lower for word in ['포장', '국물', '샜', '튼튼']):
            embeddings[i, 300:400] += np.random.normal(1.5, 0.5, 100)
            
        # 직원 관련
        if any(word in text_lower for word in ['직원', '친절', '응대']):
            embeddings[i, 400:500] += np.random.normal(1.5, 0.5, 100)
    
    # 정규화
    embeddings = normalize(embeddings, norm='l2')
    
    return embeddings

def generate_dummy_reviews(count=500):
    """더미 리뷰 데이터 생성 (빠른 실험을 위해 500개로 축소)"""
    reviews = []
    
    for i in range(count):
        if random.random() < 0.7:
            content = random.choice(positive_templates)
            rating = random.choice([4, 5])
        else:
            content = random.choice(negative_templates)
            rating = random.choice([1, 2])
        
        if random.random() < 0.1:
            rating = 3
            content = "맛은 평범했지만 양은 괜찮았어요."
        
        review = ReviewRow(
            reviewId=f"hyper-{i:04d}",
            rating=rating,
            createdAt=(datetime.now() - timedelta(days=random.randint(1, 30))).isoformat() + "Z",
            content=content
        )
        reviews.append(review)
    
    return reviews

def extract_sentences_and_embeddings(reviews):
    """리뷰에서 문장을 추출하고 임베딩을 생성"""
    from app.services.review_analyzer import _split_sentences
    
    # 리뷰 텍스트를 문장으로 분리
    all_sentences = []
    for review in reviews:
        # ReviewRow 객체인 경우 content 속성 사용
        if hasattr(review, 'content'):
            content = review.content
        else:
            content = str(review)
        sentences = kss.split_sentences(content)
        all_sentences.extend(sentences)
    
    print(f"추출된 문장 수: {len(all_sentences)}")
    
    # 임베딩 생성 (더미 데이터 사용)
    embeddings = generate_dummy_embeddings(all_sentences)
    print(f"임베딩 차원: {embeddings.shape}")
    
    return all_sentences, embeddings

def find_optimal_clusters(X, max_k=15, min_k=2):
    """최적 군집 수를 찾기 위한 여러 메트릭 계산"""
    print(f"\n군집 수 {min_k}~{max_k}에 대해 실험 중...")
    
    k_range = range(min_k, min(max_k + 1, len(X) // 10))  # 샘플 수의 1/10까지만
    inertias = []
    silhouette_scores = []
    calinski_scores = []
    
    for k in k_range:
        print(f"K={k} 실험 중...", end=" ")
        
        # K-means 실행
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        
        # 메트릭 계산
        inertia = kmeans.inertia_
        silhouette = silhouette_score(X, labels)
        calinski = calinski_harabasz_score(X, labels)
        
        inertias.append(inertia)
        silhouette_scores.append(silhouette)
        calinski_scores.append(calinski)
        
        print(f"Inertia: {inertia:.2f}, Silhouette: {silhouette:.3f}, Calinski: {calinski:.2f}")
    
    return list(k_range), inertias, silhouette_scores, calinski_scores

def plot_elbow_analysis(k_range, inertias, silhouette_scores, calinski_scores):
    """엘보우 분석 결과를 시각화"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Cluster Number Optimization Analysis', fontsize=16, fontweight='bold')
    
    # 1. Elbow Method (Inertia)
    axes[0, 0].plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
    axes[0, 0].set_title('Elbow Method (Within-cluster Sum of Squares)', fontweight='bold')
    axes[0, 0].set_xlabel('Number of Clusters (K)')
    axes[0, 0].set_ylabel('Inertia')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Silhouette Score
    axes[0, 1].plot(k_range, silhouette_scores, 'ro-', linewidth=2, markersize=8)
    axes[0, 1].set_title('Silhouette Score', fontweight='bold')
    axes[0, 1].set_xlabel('Number of Clusters (K)')
    axes[0, 1].set_ylabel('Silhouette Score')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Calinski-Harabasz Index
    axes[1, 0].plot(k_range, calinski_scores, 'go-', linewidth=2, markersize=8)
    axes[1, 0].set_title('Calinski-Harabasz Index', fontweight='bold')
    axes[1, 0].set_xlabel('Number of Clusters (K)')
    axes[1, 0].set_ylabel('Calinski-Harabasz Score')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 종합 비교 (정규화)
    # 점수들을 0-1로 정규화
    norm_silhouette = np.array(silhouette_scores)
    norm_calinski = (np.array(calinski_scores) - min(calinski_scores)) / (max(calinski_scores) - min(calinski_scores))
    norm_inertia = 1 - (np.array(inertias) - min(inertias)) / (max(inertias) - min(inertias))  # 낮을수록 좋으므로 반전
    
    axes[1, 1].plot(k_range, norm_silhouette, 'ro-', label='Silhouette (higher is better)', linewidth=2, markersize=6)
    axes[1, 1].plot(k_range, norm_calinski, 'go-', label='Calinski-Harabasz (higher is better)', linewidth=2, markersize=6)
    axes[1, 1].plot(k_range, norm_inertia, 'bo-', label='Inertia (lower is better, inverted)', linewidth=2, markersize=6)
    axes[1, 1].set_title('Normalized Metrics Comparison', fontweight='bold')
    axes[1, 1].set_xlabel('Number of Clusters (K)')
    axes[1, 1].set_ylabel('Normalized Score')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def find_elbow_point(k_range, inertias):
    """엘보우 포인트를 더 정확하게 찾는 함수"""
    if len(inertias) < 3:
        return k_range[0]
    
    # 기울기 변화율 계산 방법
    slopes = []
    for i in range(len(inertias) - 1):
        slope = (inertias[i+1] - inertias[i])
        slopes.append(abs(slope))
    
    # 기울기 변화율이 큰 지점들 찾기
    slope_changes = []
    for i in range(len(slopes) - 1):
        change = abs(slopes[i] - slopes[i+1]) / slopes[i] if slopes[i] > 0 else 0
        slope_changes.append(change)
    
    if slope_changes:
        # 변화율이 가장 큰 지점을 엘보우로 선택
        elbow_idx = np.argmax(slope_changes) + 1
        elbow_k = k_range[elbow_idx] if elbow_idx < len(k_range) else k_range[len(k_range)//2]
    else:
        elbow_k = k_range[len(k_range)//3]  # 기본값으로 1/3 지점
    
    return elbow_k

# 메인 실험
print("=== 군집 수 최적화 실험 ===")
print("500개의 더미 리뷰를 생성하고 있습니다...")

# 더미 데이터 생성
dummy_reviews = generate_dummy_reviews(500)
print(f"생성된 리뷰 수: {len(dummy_reviews)}")

# 문장 추출 및 임베딩 생성 (문장 단위)
sentences, embeddings = extract_sentences_and_embeddings(dummy_reviews)

# 차원 축소를 덜 하고 더 많은 정보 보존
if embeddings.shape[1] > 500:
    print("PCA로 차원을 축소합니다...")
    # 500차원으로 축소하여 더 많은 정보 보존
    n_components = min(500, embeddings.shape[0] // 2)  # 샘플 수의 1/2 정도로 제한
    pca = PCA(n_components=n_components, random_state=42)
    embeddings_reduced = pca.fit_transform(embeddings)
    print(f"축소된 임베딩 차원: {embeddings_reduced.shape}")
    print(f"설명된 분산 비율: {pca.explained_variance_ratio_.sum():.3f}")
else:
    embeddings_reduced = embeddings
    print("차원 축소 없이 원본 임베딩 사용")

# 최적 군집 수 찾기
k_range, inertias, silhouette_scores, calinski_scores = find_optimal_clusters(
    embeddings_reduced, 
    max_k=min(15, len(embeddings_reduced) // 10)  # 더 현실적인 범위로 제한
)

# 엘보우 포인트 자동 탐지
elbow_k = find_elbow_point(k_range, inertias)
best_silhouette_k = k_range[np.argmax(silhouette_scores)]
best_calinski_k = k_range[np.argmax(calinski_scores)]

# 추천 군집 수 결정 - 실루엣 스코어가 계속 증가하는 경우를 고려
if max(silhouette_scores) - silhouette_scores[len(silhouette_scores)//2] < 0.1:
    # 실루엣 점수가 크게 개선되지 않으면 엘보우 포인트 사용
    recommended_k = elbow_k
    print(f"실루엣 점수 개선이 미미하여 엘보우 포인트 K={elbow_k} 사용")
else:
    # 실루엣 점수가 중간 이후부터 안정화되는 지점을 찾기
    mid_point = len(silhouette_scores) // 2
    stable_point = mid_point
    for i in range(mid_point, len(silhouette_scores)-1):
        improvement = silhouette_scores[i+1] - silhouette_scores[i]
        if improvement < 0.02:  # 개선이 2% 미만이면 안정화된 것으로 판단
            stable_point = i
            break
    
    recommended_k = k_range[stable_point]
    print(f"실루엣 점수가 안정화되는 지점 K={recommended_k} 사용")

print(f"\n=== 결과 요약 ===")
print(f"엘보우 포인트 (개선된 탐지): K = {elbow_k}")
print(f"최고 실루엣 점수: K = {best_silhouette_k} (점수: {max(silhouette_scores):.3f})")
print(f"최고 칼린스키 점수: K = {best_calinski_k} (점수: {max(calinski_scores):.2f})")
print(f"최종 추천 군집 수: K = {recommended_k}")

# 결과 시각화
fig = plot_elbow_analysis(k_range, inertias, silhouette_scores, calinski_scores)
fig.savefig('cluster_optimization_analysis.png', dpi=150, bbox_inches='tight')
print(f"\n분석 결과가 'cluster_optimization_analysis.png'로 저장되었습니다.")

# 추천 군집 수로 실제 군집화 및 시각화
print(f"\n최종 추천 군집 수 K={recommended_k}로 군집화를 수행합니다...")

kmeans_final = KMeans(n_clusters=recommended_k, random_state=42, n_init=10)
final_labels = kmeans_final.fit_predict(embeddings_reduced)

# 최종 군집화 결과를 2D로 시각화
pca_2d = PCA(n_components=2, random_state=42)
embeddings_2d = pca_2d.fit_transform(embeddings_reduced)

plt.figure(figsize=(14, 10))
# 더 진한 색상 팔레트 사용
colors = plt.cm.tab10(np.linspace(0, 1, recommended_k))
for i in range(recommended_k):
    mask = final_labels == i
    plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
               c=[colors[i]], label=f'Cluster {i+1} ({np.sum(mask)} sentences)', 
               alpha=0.8, s=50)  # alpha를 0.8로 올리고 점 크기도 키움

plt.title(f'Sentence Clustering Result with Optimal K={recommended_k}', fontsize=16, fontweight='bold')
plt.xlabel('PCA Component 1', fontsize=12)
plt.ylabel('PCA Component 2', fontsize=12)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('optimal_clustering_result.png', dpi=150, bbox_inches='tight')
print(f"최종 군집화 결과가 'optimal_clustering_result.png'로 저장되었습니다.")

# 각 군집의 대표 문장 출력
print(f"\n=== Representative Sentences for Each Cluster (K={recommended_k}) ===")
for i in range(recommended_k):
    cluster_mask = final_labels == i
    cluster_sentences = [sentences[j] for j in range(len(sentences)) if cluster_mask[j]]
    
    if cluster_sentences:
        # 클러스터 중심에 가장 가까운 문장을 대표로 선택
        cluster_embeddings = embeddings_reduced[cluster_mask]
        centroid = cluster_embeddings.mean(axis=0)
        distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
        representative_idx = np.argmin(distances)
        representative_sentence = cluster_sentences[representative_idx]
        
        print(f"Cluster {i+1} ({len(cluster_sentences)} sentences): {representative_sentence}")

plt.show()
