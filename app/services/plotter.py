import io
from typing import Dict, Any

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # GUI 백엔드가 없는 환경에서도 실행 가능하도록 설정
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

# --- 사용자 지정 색상 ---

# 단일 데이터를 나타내는 차트 (메뉴별 주문량 등)
PRIMARY_COLOR = "#CAD49D"
# 074F57

# 신규 vs 재주문 파이 차트 색상 (신규, 재주문 순)
NEW_RETURN_COLORS = ["#86c5da", "#7C6354"]

# 주문 취소율 파이 차트 색상 (정상 처리, 취소 순)
CANCEL_RATE_COLORS = ["#86c5da", "#7C6354"]

# 여러 카테고리 비교하는 차트 (성별, 연령대별 등)
CATEGORICAL_PALETTE = [
    "#FAC05E", "#86c5da", "#7C6354","#CAD49D", "#CB9CF2", 
    "#14591D", "#ff9d9a", "#4c72b0", "#B14A70", "#8F8389","#00A6FB", 
    #  "#E1F4CB", 7A7978, 8AC4FF, 6CA6C1
]

# 히트맵 색상 (시간대별 메뉴 주문량)
# Matplotlib에서 지원하는 Colormap 이름 사용 ("PuBu", "PuBuGn", "GnBu")
HEATMAP_CMAP = "GnBu"

def _theme(ax, grid=True):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid:
        ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
    ax.tick_params(labelsize=9)

def _annot_bar(ax, rects, fmt="{:,.0f}", fontsize=8):
    for r in rects:
        h = r.get_height()
        if np.isnan(h) or h == 0:
            continue
        ax.text(r.get_x() + r.get_width() / 2, h, fmt.format(h),
                ha="center", va="bottom", fontsize=fontsize)

def _empty_chart(ax, title):
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.text(0.5, 0.5, "데이터 없음", ha="center", va="center", fontsize=10, alpha=0.7)
    ax.set_axis_off()

# 지표별
def plot_menu_total(df: pd.DataFrame) -> Figure:
    fig, ax = plt.subplots(figsize=(8, 5))
    title = "메뉴별 주문량 (수량 합)"
    if df is None or df.empty:
        _empty_chart(ax, title); return fig

    df = df.nlargest(15, "total_qty")  # 주문량 상위 10개 메뉴
    rects = ax.bar(df["menu_name"], df["total_qty"], color=PRIMARY_COLOR)
    _annot_bar(ax, rects)
    # ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_ylabel("주문 수량")
    ax.tick_params(axis='x', rotation=30)
    _theme(ax)
    return fig

def plot_menu_gender(df: pd.DataFrame) -> Figure:
    fig, ax = plt.subplots(figsize=(8, 5))
    title = "메뉴별 성별 주문량"
    if df is None or df.empty:
        _empty_chart(ax, title); return fig

    pivot = df.pivot_table(index="menu_name", columns="user_sex", values="qty_sum", aggfunc="sum").fillna(0)
    pivot = pivot.nlargest(10, pivot.columns.tolist(), keep='all')  # 주문량 상위 10개 메뉴
    
    pivot.plot(kind='bar', ax=ax, width=0.8, color=CATEGORICAL_PALETTE)
    # ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_ylabel("주문 수량")
    ax.set_xlabel("")
    ax.tick_params(axis='x', rotation=30)
    _theme(ax)
    ax.legend(title="성별")
    return fig

def plot_menu_age(df: pd.DataFrame) -> Figure:
    fig, ax = plt.subplots(figsize=(8, 5))
    title = "메뉴별 연령대 주문량"
    if df is None or df.empty:
        _empty_chart(ax, title); return fig

    pivot = df.pivot_table(index="menu_name", columns="age_band", values="qty_sum", aggfunc="sum").fillna(0)
    
    top_menus = pivot.sum(axis=1).nlargest(10).index  # 주문량 상위 10개 메뉴
    pivot = pivot.loc[top_menus]
    pivot = pivot.sort_index()  #이름(가나다) 순 정렬

    # 연령대 순서 정렬
    age_order = sorted(df['age_band'].unique(), key=lambda x: (x.startswith('<'), x))
    pivot = pivot.reindex(columns=[c for c in age_order if c in pivot.columns])

    pivot.plot(kind='bar', stacked=True, ax=ax, width=0.8, color=CATEGORICAL_PALETTE)
    # ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_ylabel("주문 수량")
    ax.set_xlabel("")
    ax.tick_params(axis='x', rotation=30)
    _theme(ax)
    ax.legend(title="연령대")
    return fig

def plot_hour_menu(df: pd.DataFrame) -> Figure:
    fig, ax = plt.subplots(figsize=(8, 6))
    title = "시간대별 메뉴 주문량"
    if df is None or df.empty:
        _empty_chart(ax, title); return fig

    # 주문량 상위 10개 메뉴
    top_menus = df.groupby('menu_name')['qty_sum'].sum().nlargest(10).index
    df_top = df[df['menu_name'].isin(top_menus)]

    if df_top.empty:
        _empty_chart(ax, title); return fig

    pivot = df_top.pivot_table(index="menu_name", columns="hour", values="qty_sum", aggfunc="sum").fillna(0)
    
    # 모든 시간(0-23)이 열에 포함되도록 보정
    all_hours = pd.RangeIndex(start=0, stop=24, step=1)
    pivot = pivot.reindex(columns=all_hours, fill_value=0)
    
    # 주문량이 없는 메뉴는 히트맵에서 제외
    pivot = pivot.loc[(pivot.sum(axis=1) > 0)]

    if pivot.empty:
        _empty_chart(ax, title); return fig

    im = ax.imshow(pivot, cmap=HEATMAP_CMAP, aspect='auto')
    fig.colorbar(im, ax=ax, shrink=0.8).set_label('주문 수량')
    # ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("시간 (0-23시)")
    ax.set_ylabel("메뉴")
    ax.set_xticks(np.arange(0, 24, 2))  # 2시간 간격으로 눈금 표시
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.grid(which="both", linestyle=":", linewidth=0.5, alpha=0.6)
    fig.tight_layout()
    return fig

def plot_new_vs_return(df: pd.DataFrame) -> Figure:
    fig, ax = plt.subplots(figsize=(6, 4))
    title = "신규 vs 재주문 (주문 건수)"
    if df is None or df.empty or df['count'].sum() == 0:
        _empty_chart(ax, title); return fig

    data = df.set_index('label')['count']
    data.plot(kind='pie', ax=ax, autopct='%1.1f%%', startangle=90, colors=NEW_RETURN_COLORS,
              wedgeprops={'linewidth': 0.5, 'edgecolor': 'white'},
              textprops={'fontsize': 10})
    # ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_ylabel("")
    return fig

def plot_cancel_rate(df: pd.DataFrame) -> Figure:
    fig, ax = plt.subplots(figsize=(6, 4))
    title = "주문 취소율"
    if df is None or df.empty:
        _empty_chart(ax, title); return fig

    total = df.iloc[0]['total']
    cancel = df.iloc[0]['cancel']
    
    if total == 0:
        _empty_chart(ax, title); return fig

    non_cancel = total - cancel
    labels = ['정상 처리', '취소']
    sizes = [non_cancel, cancel]
    
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=CANCEL_RATE_COLORS)
    # ax.set_title(title, fontsize=12, fontweight="bold")
    ax.text(0, 0, f"{cancel}/{total}건", ha='center', va='center', fontsize=12)
    return fig

def plot_reorder_gap_hist(df: pd.DataFrame) -> Figure:
    fig, ax = plt.subplots(figsize=(8, 5))
    title = "재주문 간격 분포"
    if df is None or df.empty:
        _empty_chart(ax, title); return fig

    order = ["0–7일", "8–30일", "31–90일", "91–180일", "181일+"]
    data = df.set_index("gap_bin").reindex(order).fillna(0).reset_index()
    
    rects = ax.bar(data["gap_bin"], data["count"], color=PRIMARY_COLOR)
    _annot_bar(ax, rects)
    # ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_ylabel("재주문 건수")
    return fig
