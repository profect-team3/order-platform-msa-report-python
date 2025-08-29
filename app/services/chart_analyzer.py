import os
import io
import base64
from typing import Dict
from matplotlib.figure import Figure

# 공유 클라이언트를 임포트하여 사용
from ..utils.openai_client import client

def _figure_to_base64(fig: Figure) -> str:
    """Matplotlib Figure를 base64 인코딩된 PNG 문자열로 변환합니다."""
    if fig is None:
        return None
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def analyze_charts(figures: Dict[str, Figure]) -> str:
    """
    차트 Figure 딕셔너리를 받아 OpenAI Vision API로 분석하고,
    점주를 위한 인사이트와 운영 전략 제안 텍스트를 생성합니다.
    """
    # 유효한 Figure 객체만 필터링하여 base64로 인코딩
    encoded_images = {name: _figure_to_base64(fig) for name, fig in figures.items() if fig and fig.get_axes()}
    
    if not encoded_images:
        return ""

    # 프롬프트 구성
    prompt = (
        "당신은 데이터 분석 전문가이자 컨설턴트입니다. "
        "입력된 차트들은 한 가게의 월간 판매 데이터 리포트입니다. "
        "각 차트의 핵심 인사이트를 도출하고, 이를 바탕으로 점주가 실행할 수 있는 구체적인 운영 전략을 제안하는 글을 작성해주세요.\n\n"
        "글의 형식:\n"
        "1. **종합 분석 및 핵심 요약**: 전체 차트를 아우르는 가장 중요한 트렌드나 문제점을 2~3문장으로 요약합니다.\n"
        "2. **주요 지표별 분석 및 제안**:\n"
        "   - 각 분석 항목마다 소제목을 붙이고, 그 아래에 분석 내용과 액션 아이템 제안을 순서대로 작성합니다.\n"
        "   - 예시:\n"
        "     ### 메뉴 인기 분석\n"
        "     '허니콤보'가 가장 인기 있는 메뉴이며, 특히 20대 여성 고객의 주문이 많습니다.\n"
        "     >> 제안: 20대 여성을 타겟으로 한 '허니콤보 + 사이드메뉴' 세트 할인 프로모션을 고려해보세요.\n"
        "3. **마무리**: 긍정적인 격려와 함께 리포트의 내용을 잘 활용하도록 독려하는 문장으로 마무리합니다.\n\n"
        "출력 규칙:\n"
        "- 전체 리포트의 큰 섹션 제목: `**제목**` 형식 (예: **종합 분석 및 핵심 요약**)\n"
        "- 각 분석 항목의 소제목: `### 소제목` 형식 (예: ### 메뉴 인기 분석)\n"
        "- 구체적인 제안: `>> 제안:` 으로 시작하는 줄에 작성\n"
        "- 문장 내 단어 강조: `**단어**` 형식\n"
        "- 위에 명시된 형식 외 다른 마크다운 기호(예: -, *, > 등)는 사용하지 마세요.\n\n"
        "어조: 점주에게 직접 조언하는 것처럼, 전문적이면서도 이해하기 쉽고 친근한 어조를 사용해주세요."
    )

    # API에 전달할 메시지 생성
    content = [{"type": "text", "text": prompt}]
    for name, b64_img in encoded_images.items():
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{b64_img}"
            }
        })

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": content}],
            max_tokens=1500,
            temperature=0.5,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error calling OpenAI API for chart analysis: {e}")
        return "차트 분석 중 오류가 발생했습니다."
