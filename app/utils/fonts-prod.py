import matplotlib as mpl
from matplotlib import font_manager
import os, platform, logging
from pathlib import Path
from typing import Optional

def _find_korean_font_path() -> Optional[Path]:
    """시스템에서 사용 가능한 한국어 폰트 경로를 찾습니다."""
    # Lambda 환경을 위해 프로젝트 루트에 'fonts' 디렉토리를 만들고 폰트 파일을 위치시킵니다.
    # 예: /report_service/fonts/NanumGothic.ttf
    project_root = Path(__file__).resolve().parent.parent.parent
    font_dir = project_root / "fonts"
    
    if not font_dir.exists():
        return None
        
    # TTF 폰트를 우선적으로 찾습니다.
    font_patterns = ["*.ttf", "*.otf", "*.ttc"]
    for pattern in font_patterns:
        found = list(font_dir.glob(pattern))
        if found:
            return found[0]
            
    return None

KOR_FONT_PATH = _find_korean_font_path()

def setup_korean_font() -> None:
    """Matplotlib에 한국어 폰트를 설정합니다."""
    logging.getLogger("fontTools.subset").setLevel(logging.ERROR)

    if KOR_FONT_PATH:
        font_manager.fontManager.addfont(str(KOR_FONT_PATH))
        family = font_manager.FontProperties(fname=str(KOR_FONT_PATH)).get_name()
        mpl.rcParams["font.family"] = family

    mpl.rcParams["axes.unicode_minus"] = False
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"]  = 42
    mpl.rcParams["figure.dpi"]   = 150
