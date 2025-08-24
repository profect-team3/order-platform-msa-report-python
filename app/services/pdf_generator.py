import io
from datetime import date
from typing import Any, Dict, List

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, KeepTogether, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

from ..utils.fonts import KOR_FONT_PATH

if KOR_FONT_PATH and KOR_FONT_PATH.exists():
    pdfmetrics.registerFont(TTFont('Korean-Font', str(KOR_FONT_PATH)))
    KOR_FONT_NAME = 'Korean-Font'
else:
    KOR_FONT_NAME = 'Helvetica'
    
import io
from reportlab.platypus import Image
from reportlab.lib.units import cm
from reportlab.lib.enums import TA_CENTER

def chart_image_from_fig(fig, max_w_cm=7.8, max_h_cm=8.5, dpi=150):
    """
    - 원본 비율 유지
    - 2열(컬럼 폭 ≈ 8.5cm)에 맞춰 가로·세로 상한 내로 자동 축소
    """
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", transparent=True)  # 여백 최소화
    buf.seek(0)
    img = Image(buf)
    img.hAlign = "CENTER"
    img._restrictSize(max_w_cm*cm, max_h_cm*cm)
    return img

def build_report_pdf(
    menu_total: Any,
    menu_by_gender: Any,
    menu_by_age: Any,
    hour_menu: Any,
    new_vs_return: Any,
    cancel_rate: Any,
    # reorder_gap_hist: Any,
    # reorder_top3: Any,
    store_name: str,
    start_date: date,
    end_date: date,
    review_insights: Dict,
    file_like_object: io.BytesIO
):
    
    doc = SimpleDocTemplate(file_like_object, rightMargin=2*cm, leftMargin=2*cm, topMargin=2*cm, bottomMargin=2*cm)
    
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='Korean-Title', fontName=KOR_FONT_NAME, fontSize=20, spaceAfter=12))
    styles.add(ParagraphStyle(name='Korean-H1', fontName=KOR_FONT_NAME, fontSize=14, spaceBefore=10, spaceAfter=4))
    styles.add(ParagraphStyle(name='Korean-H2', fontName=KOR_FONT_NAME, fontSize=12, spaceBefore=10, spaceAfter=4))
    styles.add(ParagraphStyle(name='Korean-H2-Center', parent=styles['Korean-H2'], alignment=TA_CENTER, spaceBefore=6, spaceAfter=4))
    styles.add(ParagraphStyle(name='Korean-Body', fontName=KOR_FONT_NAME, fontSize=9, leading=14))
    styles.add(ParagraphStyle(name='Korean-Quote', fontName=KOR_FONT_NAME, fontSize=9, leading=12, leftIndent=12, textColor='grey'))
    story = []

    # === 제목 ===
    story.append(Paragraph(f"{store_name} 리포트", styles['Korean-Title']))
    story.append(Paragraph(f"분석 기간: {start_date.isoformat()} ~ {end_date.isoformat()}", styles['Korean-Body']))
    story.append(Spacer(1, 1*cm))

    #  === 주문 분석 차트 ===
    charts = {
        "주문 많은 메뉴": menu_total,
        "시간대별 주문 메뉴": hour_menu,
        "성별 주문 선호 메뉴": menu_by_gender,
        "연령대별 주문 선호 메뉴": menu_by_age,
        "신규/재방문 고객 비율": new_vs_return,
        "주문 취소율": cancel_rate,
        # "재주문 고객 방문 주기": reorder_gap_hist,
        # "재주문 상위 메뉴": reorder_top3,
    }
    
    # === 한 페이지에 차트 모두 넣기 위한 최대 높이 계산 ===
    num_charts = len([fig for _, fig in charts.items() if fig is not None])
    rows = (num_charts + 1) // 2  # 2열 그리드의 행 수

    # 문서 내부 프레임(본문) 높이 (포인트) → cm
    available_h_cm = doc.height / cm

    # 상단 제목/본문/스페이서가 이미 차지한 높이(대략치, 필요시 미세 조정)
    #  - Title 20pt + spaceAfter 12pt ≈ 1.13cm
    #  - "분석 기간" 본문 한 줄(leading=14pt) ≈ 0.5cm
    #  - Spacer(1*cm) = 1.0cm
    header_consumed_cm = 1.13 + 0.5 + 1.0

    # 차트 표의 행 간 여백(=BOTTOMPADDING 효과)을 cm로 가정
    # 현재 TableStyle에서 BOTTOMPADDING=8pt ≈ 0.28cm, 행 사이 합산 반영 위해 약간 버퍼 추가
    inter_row_gap_cm = 0.35

    # 차트가 실제로 쓸 수 있는 총 높이
    charts_total_h_cm = max(0.0, available_h_cm - header_consumed_cm)

    # 행당 배정 높이
    if rows > 0:
        per_row_h_cm = (charts_total_h_cm - inter_row_gap_cm * max(0, rows - 1)) / rows
    else:
        per_row_h_cm = 0.0

    # 셀 안에서 제목/스페이서가 차지하는 높이(대략치)
    cell_title_cm = 0.55   # H2(12pt) + spaceBefore/After 약간
    cell_spacer_cm = 0.20  # Paragraph와 이미지 사이 Spacer(0.2cm)

    # 이미지에 할당 가능한 최대 높이(안전하게 약간 더 줄임)
    max_img_h_cm = max(3.5, per_row_h_cm - (cell_title_cm + cell_spacer_cm + 0.2))  # 0.2cm = 기타 패딩 버퍼
    # === 높이 계산 끝 ===
    
    # 차트들을 2열 그리드로 배치하여 공간 절약
    chart_items = [(title, fig) for title, fig in charts.items() if fig is not None]
    table_data = []
    for i in range(0, len(chart_items), 2):
        row_items = chart_items[i : i+2]
        
        table_row = []
        for title, fig in row_items:
            # 비율 유지 + 자동 축소
            img = chart_image_from_fig(fig, max_w_cm=8, max_h_cm=max_img_h_cm, dpi=150)
            
            cell_story = [
                Paragraph(title, styles['Korean-H2-Center']),
                Spacer(1, 0.2*cm),
                img
            ]
            table_row.append(cell_story)
        
        if len(table_row) == 1:
            table_row.append("")  # 홀수 개의 차트일 경우 빈 셀 추가
        
        table_data.append(table_row)

    if table_data:
        chart_table = Table(table_data, colWidths=[8.5*cm, 8.5*cm], rowHeights=None)
        chart_table.setStyle(TableStyle([
            ('VALIGN', (0,0), (-1,-1), 'TOP'),
            ('ALIGN',  (0,0), (-1,-1), 'CENTER'),
            ('LEFTPADDING', (0,0), (-1,-1), 0),
            ('RIGHTPADDING', (0,0), (-1,-1), 0),
            ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ]))
        story.append(chart_table)

    # === 리뷰 분석글 ===
    if review_insights and (review_insights.get('pros') or review_insights.get('cons')):
        story.append(PageBreak())
        story.append(Paragraph("AI 리뷰 리포트", styles['Korean-H1']))

        def draw_insights(title: str, items: List[Dict]):
            if not items: return
            story.append(Paragraph(title, styles['Korean-H2']))
            for item in items:
                insight_block = [
                    Paragraph(f"<b>{item['label']}</b> ({item['count']}건)", styles['Korean-Body']),
                    Paragraph(item['summary'], styles['Korean-Body']),
                    Paragraph(f"&ldquo;{item['quote']}&rdquo;", styles['Korean-Quote']),
                    Spacer(1, 0.4*cm)
                ]
                story.append(KeepTogether(insight_block))

        draw_insights("👍 이런 점을 좋아해주셨습니다", review_insights.get('pros', []))
        story.append(Spacer(1, 0.5*cm))
        draw_insights("👎 이런 점은 개선이 필요합니다", review_insights.get('cons', []))

    doc.build(story)
    