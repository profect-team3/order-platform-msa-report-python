import matplotlib as mpl
from matplotlib import font_manager
import os, platform, logging
from pathlib import Path
from typing import Optional

def _find_korean_font_path() -> Optional[Path]:
    """시스템에서 사용 가능한 한국어 폰트 경로를 찾습니다."""
    system = platform.system()
    candidates = []
    if system == "Darwin":  # macOS
        candidates = [
            # reportlab과의 호환성 위해 TTF 폰트 우선 탐색
            "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
            "/System/Library/Fonts/AppleSDGothicNeo.ttc",
            os.path.expanduser("~/Library/Fonts/NotoSansCJKkr-Regular.otf"),
            "/Library/Fonts/NotoSansCJKkr-Regular.otf",
        ]
    elif system == "Windows":
        candidates = [
            # TTF 폰트 우선 탐색
            r"C:\Windows\Fonts\malgun.ttf",
            r"C:\Windows\Fonts\gulim.ttc",
            r"C:\Windows\Fonts\NotoSansCJKkr-Regular.otf", # OTF는 후순위
        ]
    else:  # Linux
        candidates = [
            # reportlab 호환성을 위해 TTF 폰트 우선 탐색
            # (apt-get install fonts-nanum* 등으로 설치 필요)
            "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
            "/usr/share/fonts/truetype/noto/NotoSansCJKkr-Regular.ttf",
            # OTF/TTC는 실패 가능성 있어 후순위 배치
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        ]

    for p in candidates:
        path = Path(p)
        if path.exists():
            return path
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
