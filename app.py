import os
os.environ.setdefault("MPLBACKEND", "Agg")

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from datetime import datetime, timedelta, timezone
import os
from dotenv import load_dotenv
import logging
logger = logging.getLogger("app")

from data.mongo import get_orders_unified_col
from data.queries_unified import load_frames_30d_unified
from report.build_onepage_layout import build_onepage_pdf
from report.fonts_setup import setup_korean_font

load_dotenv()
KST = timezone(timedelta(hours=9))

app = FastAPI(title="Report Service")

class GenerateReq(BaseModel):
    storeId: str
    period: str = "last30d"

@app.get("/health")
def health():
    return {"ok": True, "time": datetime.now(KST).isoformat()}

@app.post("/report/generate-json")
def generate_json(req: GenerateReq):
    try:
        setup_korean_font()

        timestamp = datetime.now(KST)
        end = timestamp.date()
        start = end - timedelta(days=29)

        col = get_orders_unified_col()
        frames = load_frames_30d_unified(
            col,
            store_id=req.storeId,
            start_kst_date=start,
            end_kst_date=end
        )
        
        # 저장 위치: .env -> 절대경로로 변환
        outdir = os.getenv("OUTPUT_DIR", "./out")
        outdir = os.path.abspath(outdir)
        os.makedirs(outdir, exist_ok=True)

        base = f"report_{req.storeId}_{timestamp.strftime('%Y%m%d_%H%M%S')}"
        filename = f"{base}.pdf"
        path = os.path.join(outdir, filename)
        
        with open(path, "wb") as f:
            build_onepage_pdf(
                frames["menu_total"],
                frames["menu_by_gender"],
                frames["menu_by_age"],
                frames["hour_menu"],
                frames["new_vs_return"],
                frames["cancel_rate"],
                frames["reorder_gap_hist"],
                frames["reorder_top3"],
                req.storeId, start, end, f
            )

        return JSONResponse({
            "jobId": None,
            "filename": filename,
            "url": f"/static/{filename}",
            "localPath": path,
            "storeId": req.storeId,
            "period": {"start": str(start), "end": str(end), "tz": "Asia/Seoul"},
            "createdAt": datetime.now(KST).isoformat()
        })
    except Exception as e:
        logger.exception("generate_json FAILED")
        raise HTTPException(status_code=500, detail=str(e))
