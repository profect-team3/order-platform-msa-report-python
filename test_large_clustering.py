from app.services.review_analyzer import analyze_reviews
from app.models import ReviewPayload, ReviewRow
from datetime import datetime, timedelta
import random
import json

# 더미 리뷰 템플릿들
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
    "음식의 온도가 적절했고 포장 상태도 완벽했어요.",
    "재료가 신선하고 조리도 잘 되어 있었습니다.",
    "배달 시간이 정확했고 음식 품질도 우수했어요.",
    "가성비 최고! 양도 많고 맛도 훌륭했습니다.",
    "요청사항을 완벽하게 반영해주셔서 감사했어요.",
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
    "전화 응대가 불친절하고 배달도 너무 늦었습니다.",
    "재료가 신선하지 않은 것 같고 맛이 이상했어요.",
    "가격에 비해 양이 너무 적고 맛도 평범했습니다.",
    "배달비가 비싼데 서비스는 정말 별로네요.",
    "음식이 눅눅하고 온도도 미지근했어요.",
]

def generate_dummy_reviews(count=1000):
    """더미 리뷰 데이터 생성"""
    reviews = []
    
    for i in range(count):
        # 70% 확률로 긍정, 30% 확률로 부정
        if random.random() < 0.7:
            content = random.choice(positive_templates)
            rating = random.choice([4, 5])  # 긍정 리뷰는 4-5점
        else:
            content = random.choice(negative_templates)
            rating = random.choice([1, 2])  # 부정 리뷰는 1-2점
        
        # 가끔 중간 점수도 추가
        if random.random() < 0.1:
            rating = 3
            content = "맛은 평범했지만 양은 괜찮았어요."
        
        # 리뷰 내용에 약간의 변화 추가
        variations = [
            content,
            content + " 다음에도 주문할 것 같아요.",
            content + " 추천드립니다.",
            content + " 감사합니다.",
            content.replace("요.", "습니다."),
            content.replace("어요", "었어요"),
        ]
        
        review = ReviewRow(
            reviewId=f"dummy-{i:04d}",
            rating=rating,
            createdAt=(datetime.now() - timedelta(days=random.randint(1, 30))).isoformat() + "Z",
            content=random.choice(variations)
        )
        reviews.append(review)
    
    return reviews

print("1000개의 더미 리뷰를 생성하고 있습니다...")
dummy_reviews = generate_dummy_reviews(1000)

# ReviewPayload 생성
payload = ReviewPayload(
    storeId="test-store-large",
    reviews=dummy_reviews
)

print(f"총 {len(dummy_reviews)}개의 리뷰가 생성되었습니다.")
print("평점 분포:")
rating_counts = {}
for review in dummy_reviews:
    rating_counts[review.rating] = rating_counts.get(review.rating, 0) + 1

for rating in sorted(rating_counts.keys()):
    print(f"  {rating}점: {rating_counts[rating]}개 ({rating_counts[rating]/len(dummy_reviews)*100:.1f}%)")

print("\n군집화 분석을 시작합니다...")
result = analyze_reviews(payload, visualize=True, min_cluster_size=10)  # 큰 데이터셋에 맞게 조정

# 결과 출력
print("\n=== 분석 결과 ===")
print(f"총 리뷰 수: {result['stats']['reviewCount']}")
print(f"총 문장 수: {result['stats']['sentenceCount']}")
print(f"커버리지: {result['stats']['coverage']:.1%}")

print("\n=== 긍정 의견 ===")
for item in result['pros']:
    print(f"- {item['label']} ({item['count']}건): {item['summary']}")

print("\n=== 부정 의견 ===")
for item in result['cons']:
    print(f"- {item['label']} ({item['count']}건): {item['summary']}")

# 시각화 결과 확인
if 'visualization' in result:
    print("\n=== 군집화 시각화 ===")
    print("시각화 이미지가 생성되었습니다!")
    print(f"이미지 크기: {len(result['visualization'])} 문자")
    
    # 이미지를 파일로 저장
    import base64
    img_data = base64.b64decode(result['visualization'].split(',')[1])
    with open('large_clustering_visualization.png', 'wb') as f:
        f.write(img_data)
    print("시각화 이미지가 'large_clustering_visualization.png'로 저장되었습니다.")
else:
    print("\n시각화가 생성되지 않았습니다.")

# 상세 통계 출력
print(f"\n=== 상세 통계 ===")
print(f"긍정 의견 총 {len(result['pros'])}개 그룹")
print(f"부정 의견 총 {len(result['cons'])}개 그룹")

total_processed = sum(item['count'] for item in result['pros'] + result['cons'])
print(f"처리된 문장 수: {total_processed}/{result['stats']['sentenceCount']}")
