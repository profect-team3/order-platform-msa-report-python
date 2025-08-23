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


def build_onepage_pdf(
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
    styles.add(ParagraphStyle(name='Korean-Title', fontName=KOR_FONT_NAME, fontSize=24, spaceAfter=12))
    styles.add(ParagraphStyle(name='Korean-H1', fontName=KOR_FONT_NAME, fontSize=16, spaceBefore=12, spaceAfter=6))
    styles.add(ParagraphStyle(name='Korean-H2', fontName=KOR_FONT_NAME, fontSize=14, spaceBefore=10, spaceAfter=4))
    styles.add(ParagraphStyle(name='Korean-Body', fontName=KOR_FONT_NAME, fontSize=10, leading=14))
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

    # 차트들을 2열 그리드로 배치하여 공간 절약
    chart_items = [(title, fig) for title, fig in charts.items() if fig is not None]
    table_data = []
    for i in range(0, len(chart_items), 2):
        row_items = chart_items[i : i+2]
        
        table_row = []
        for title, fig in row_items:
            img_buffer = io.BytesIO()
            # 차트 배경을 투명하게 설정
            fig.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight', transparent=True)
            img_buffer.seek(0)
            
            # 2열 레이아웃에 맞게 이미지 너비 줄임
            img = Image(img_buffer, width=7.8*cm, height=(7.8*cm * fig.get_figheight() / fig.get_figwidth()))
            
            cell_story = [
                Paragraph(title, styles['Korean-H2']), # H1보다 작은 H2 스타일 사용
                Spacer(1, 0.2*cm),
                img
            ]
            table_row.append(cell_story)
        
        if len(table_row) == 1:
            table_row.append("")  # 홀수 개의 차트일 경우 빈 셀 추가
        
        table_data.append(table_row)

    if table_data:
        chart_table = Table(table_data, colWidths=[8*cm, 8*cm], rowHeights=None)
        chart_table.setStyle(TableStyle([
            ('VALIGN', (0,0), (-1,-1), 'TOP'),
            ('LEFTPADDING', (0,0), (-1,-1), 0),
            ('RIGHTPADDING', (0,0), (-1,-1), 0),
            ('BOTTOMPADDING', (0,0), (-1,-1), 12),  # 행 간의 간격
        ]))
        story.append(chart_table)

    # === 리뷰 분석글 ===
    if review_insights and (review_insights.get('pros') or review_insights.get('cons')):
        story.append(Spacer(1, 1.5*cm))  # 페이지 나누기 대신 충분한 간격 줌
        story.append(Paragraph("리뷰 핵심 요약", styles['Korean-H1']))

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
    