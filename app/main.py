import os
from datetime import datetime, timedelta, date
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

from .models import OrderPayload, ReviewPayload, GenerateJsonResponse, ReportGenerationRequest, OrderRow, OrderItemRow, ReviewRow
from .utils.timezone import KST
from .utils import fonts
from .services.aggregator import build_frames  # 주문 데이터 분석
from .services.review_analyzer import analyze_reviews  # 리뷰 데이터 분석
from .services.chart_analyzer import analyze_charts  # 차트 분석
from .services.pdf_generator import build_report_pdf  # PDF 생성
from .services import plotter

import logging
logger = logging.getLogger("app")

load_dotenv()

app = FastAPI(title="Report Server")

@app.get("/health")
def health():
    return {"ok": True, "time": datetime.now(KST).isoformat()}

@app.post("/report/generate-json", response_model=GenerateJsonResponse)
def generate_json(req: ReportGenerationRequest = None):
    
    try:
        fonts.setup_korean_font()

        now = datetime.now(KST)
        end   = now.date()
        start = end - timedelta(days=30)

        order_payload = OrderPayload(storeId=req.storeId, orders=req.orders)
        review_payload = ReviewPayload(storeId=req.storeId, reviews=req.reviews)

        outdir = os.getenv("OUTPUT_DIR", "./out")
        outdir = os.path.abspath(outdir)
        os.makedirs(outdir, exist_ok=True)

        base = f"report_{req.storeId}_{now.strftime('%Y%m%d_%H%M%S')}"
        filename = f"{base}.pdf"
        path = os.path.join(outdir, filename)

        # 주문이 없을 경우를 대비해 storeName 가져옴
        store_name = req.orders[0].storeName if req.orders else req.storeId
        
        # 1-1. 주문 데이터 -> 지표별 데이터프레임 생성
        frames = build_frames(order_payload)

        # 1-2. 데이터프레임 -> 차트(Figure)로 변환
        figures = {
            "menu_total": plotter.plot_menu_total(frames["menu_total"]),
            "menu_by_gender": plotter.plot_menu_gender(frames["menu_by_gender"]),
            "menu_by_age": plotter.plot_menu_age(frames["menu_by_age"]),
            "hour_menu": plotter.plot_hour_menu(frames["hour_menu"]),
            "new_vs_return": plotter.plot_new_vs_return(frames["new_vs_return"]),
            "cancel_rate": plotter.plot_cancel_rate(frames["cancel_rate"]),
        }
        
        # 1-3. 차트 이미지 -> AI 분석글 생성
        chart_insights = analyze_charts(figures)

        # # === DEBUG: 각 차트를 PNG 파일로 저장 ===
        # for name, fig in figures.items():
        #     if fig:
        #         debug_filename = f"debug_chart_{base}_{name}.png"
        #         debug_filepath = os.path.join(outdir, debug_filename)
        #         fig.savefig(debug_filepath, dpi=150, bbox_inches='tight')

        # 2. 리뷰 데이터 분석
        review_insights = analyze_reviews(review_payload, visualize=True)

        # 3. 주문/리뷰 분석 결과들을 모아 PDF 파일로 생성
        with open(path, "wb") as f:
            build_report_pdf(
                figures["menu_total"],
                figures["menu_by_gender"],
                figures["menu_by_age"],
                figures["hour_menu"],
                figures["new_vs_return"],
                figures["cancel_rate"],
                chart_insights,
                # frames.get("reorder_gap_hist"), # NOTE: aggregator.py에서 비활성화
                # frames.get("reorder_top3"),     # NOTE: aggregator.py에서 비활성화
                store_name,
                start,
                end,
                review_insights,
                f
            )

        return JSONResponse({
            "url": f"/static/{filename}",
            "localPath": str(path),
            "createdAt": now.isoformat()
        })
    except Exception as e:
        logger.exception("generate_json FAILED")
        raise HTTPException(status_code=500, detail=str(e))
