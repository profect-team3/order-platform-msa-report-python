import matplotlib as mpl
from matplotlib import font_manager
import os, platform, logging

def setup_korean_font():
    logging.getLogger("fontTools.subset").setLevel(logging.ERROR)

    system = platform.system()
    candidates = []
    if system == "Darwin":  # macOS
        candidates = [
            os.path.expanduser("~/Library/Fonts/NotoSansCJKkr-Regular.otf"),
            "/Library/Fonts/NotoSansCJKkr-Regular.otf",
            "/System/Library/Fonts/AppleSDGothicNeo.ttc",  # 최후 순위
        ]
    elif system == "Windows":
        candidates = [
            r"C:\Windows\Fonts\NotoSansCJKkr-Regular.otf",
            r"C:\Windows\Fonts\malgun.ttf",
        ]
    else:  # Linux
        candidates = [
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/noto/NotoSansCJKkr-Regular.otf",
        ]

    font_path = next((p for p in candidates if os.path.exists(p)), None)
    if font_path:
        font_manager.fontManager.addfont(font_path)
        family = font_manager.FontProperties(fname=font_path).get_name()
        mpl.rcParams["font.family"] = family

    mpl.rcParams["axes.unicode_minus"] = False
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"]  = 42
    mpl.rcParams["figure.dpi"]   = 150
