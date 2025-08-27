import os
import io
from datetime import datetime, timedelta, date
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from mangum import Mangum

from .models import OrderPayload, ReviewPayload, GenerateJsonResponse, ReportGenerationRequest, OrderRow, OrderItemRow, ReviewRow
from .utils.timezone import KST
from .utils import fonts
from .services.aggregator import build_frames  # 주문 데이터 분석
from .services.review_analyzer import analyze_reviews  # 리뷰 데이터 분석
from .services.chart_analyzer import analyze_charts # 차트 분석
from .services.pdf_generator import build_report_pdf  # PDF 생성
from .services import plotter

import boto3
import logging
logger = logging.getLogger("app")

load_dotenv()

app = FastAPI(title="Report Server")

# Lambda 핸들러
handler = Mangum(app)

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

        # S3 버킷 설정 (Lambda 환경변수에서 가져옴)
        s3_bucket = os.getenv("OUTPUT_S3_BUCKET")
        if not s3_bucket:
            raise ValueError("OUTPUT_S3_BUCKET environment variable is not set.")

        # S3에 저장될 파일 경로와 이름
        base = f"report_{req.storeId}_{now.strftime('%Y%m%d_%H%M%S')}"
        filename = f"{base}.pdf"
        s3_key = f"reports/{filename}"

        # 주문이 없을 경우를 대비해 storeName 가져옴
        store_name = req.orders[0].storeName if req.orders else str(req.storeId)
        
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

        # 2. 리뷰 데이터 분석
        review_insights = analyze_reviews(review_payload)
        
        # 3. 주문/리뷰/차트 분석 결과들을 모아 PDF 파일로 생성
        pdf_buffer = io.BytesIO()
        try:
            build_report_pdf(
                figures["menu_total"],
                figures["menu_by_gender"],
                figures["menu_by_age"],
                figures["hour_menu"],
                figures["new_vs_return"],
                figures["cancel_rate"],
                chart_insights,
                store_name,
                start,
                end,
                review_insights,
                pdf_buffer
            )
        finally:
            # matplotlib Figure 객체 메모리 해제
            for fig in figures.values():
                if fig:
                    plotter.plt.close(fig)

        # 4. 생성된 PDF를 S3에 업로드
        pdf_buffer.seek(0)
        s3_client = boto3.client("s3")
        s3_client.upload_fileobj(pdf_buffer, s3_bucket, s3_key)
        
        region_name = s3_client.meta.region_name
        s3_url = f"https://{s3_bucket}.s3.{region_name}.amazonaws.com/{s3_key}"

        response_data = {
            "url": s3_url,
            "createdAt": now.isoformat()
        }

        return JSONResponse(response_data)
    except Exception as e:
        logger.exception("generate_json FAILED")
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")
