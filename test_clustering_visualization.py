from app.services.review_analyzer import analyze_reviews
from app.models import ReviewPayload, ReviewRow
from datetime import datetime, timedelta
import json

# 샘플 리뷰 데이터
sample_reviews = [
    ReviewRow(
        reviewId="r-1",
        rating=5,
        createdAt=(datetime.now() - timedelta(days=1)).isoformat() + "Z",
        content="맛있고 포장이 깔끔했어요. 배달도 빨랐습니다."
    ),
    ReviewRow(
        reviewId="r-2",
        rating=4,
        createdAt=(datetime.now() - timedelta(days=2)).isoformat() + "Z",
        content="양이 많고 가성비가 좋았어요. 추천합니다!"
    ),
    ReviewRow(
        reviewId="r-3",
        rating=2,
        createdAt=(datetime.now() - timedelta(days=3)).isoformat() + "Z",
        content="배달이 늦었고 음식이 식어서 왔어요. 실망입니다."
    ),
    ReviewRow(
        reviewId="r-4",
        rating=1,
        createdAt=(datetime.now() - timedelta(days=4)).isoformat() + "Z",
        content="소스가 누락되었고 직원 응대도 불친절했습니다."
    ),
    ReviewRow(
        reviewId="r-5",
        rating=5,
        createdAt=(datetime.now() - timedelta(days=5)).isoformat() + "Z",
        content="포장이 튼튼하고 국물이 안 샜어요. 매우 만족!"
    ),
    ReviewRow(
        reviewId="r-6",
        rating=3,
        createdAt=(datetime.now() - timedelta(days=6)).isoformat() + "Z",
        content="맛은 평범했지만 양은 괜찮았어요."
    ),
    ReviewRow(
        reviewId="r-7",
        rating=5,
        createdAt=(datetime.now() - timedelta(days=7)).isoformat() + "Z",
        content="직원분들이 친절하고 요청사항을 잘 반영해줬습니다."
    ),
    ReviewRow(
        reviewId="r-8",
        rating=2,
        createdAt=(datetime.now() - timedelta(days=8)).isoformat() + "Z",
        content="예정 시간보다 늦게 도착했고 음식이 차가워요."
    ),
]

# ReviewPayload 생성
payload = ReviewPayload(
    storeId="test-store-123",
    reviews=sample_reviews
)

# 군집화 시각화 포함하여 분석 실행
print("리뷰 분석을 시작합니다...")
result = analyze_reviews(payload, visualize=True, min_cluster_size=2)  # 군집 크기를 2로 줄임

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
    
    # 이미지를 파일로 저장 (선택사항)
    import base64
    img_data = base64.b64decode(result['visualization'].split(',')[1])
    with open('clustering_visualization.png', 'wb') as f:
        f.write(img_data)
    print("시각화 이미지가 'clustering_visualization.png'로 저장되었습니다.")
else:
    print("\n시각화가 생성되지 않았습니다.")
