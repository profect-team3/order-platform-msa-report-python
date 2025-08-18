from datetime import date

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec

# --------- utils ---------
def _annot_bar(ax, rects, fmt="{:,.0f}", fontsize=8):
    for r in rects:
        h = r.get_height()
        if np.isnan(h) or h == 0:
            continue
        ax.text(r.get_x()+r.get_width()/2, h, fmt.format(h),
                ha="center", va="bottom", fontsize=fontsize)

def _theme(ax, grid=True):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid:
        ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
    ax.tick_params(labelsize=9)

def _legend(ax, loc="best", fontsize=8, ncol=1):
    leg = ax.legend(loc=loc, fontsize=fontsize, frameon=False, ncol=ncol)
    if leg:
        for t in leg.get_texts():
            t.set_fontsize(fontsize)

def _empty(ax, title):
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.text(0.5, 0.5, "데이터 없음", ha="center", va="center", fontsize=10, alpha=0.7)
    ax.set_axis_off()

# --------- plots ---------
def plot_menu_total(ax, df_menu_total: pd.DataFrame):
    title = "① 메뉴별 주문량(수량 합)"
    if df_menu_total is None or df_menu_total.empty:
        _empty(ax, title); return

    x = df_menu_total["menu_name"].astype(str)
    y = df_menu_total["qty_sum"].astype(float)
    rects = ax.bar(x, y)
    _annot_bar(ax, rects)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_ylabel("수량")
    ax.set_xticks(range(len(x)), labels=x, rotation=20, ha="right")
    _theme(ax)

def plot_menu_gender(ax, df_menu_gender: pd.DataFrame):
    title = "② 메뉴별 성별 주문량"
    if df_menu_gender is None or df_menu_gender.empty:
        _empty(ax, title); return

    pivot = df_menu_gender.pivot_table(index="menu_name", columns="gender", values="qty_sum", aggfunc="sum").fillna(0)
    cats = pivot.index.tolist()
    genders = list(pivot.columns)
    ind = np.arange(len(cats))
    width = 0.35 if len(genders) <= 2 else 0.7/len(genders)

    for i, g in enumerate(genders):
        rects = ax.bar(ind + i*width, pivot[g].values, width, label=str(g))
        _annot_bar(ax, rects)

    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_ylabel("수량")
    ax.set_xticks(ind + width*(len(genders)-1)/2)
    ax.set_xticklabels(cats, rotation=20, ha="right")
    _theme(ax)
    _legend(ax, loc="upper right", ncol=len(genders))

def plot_menu_age(ax, df_menu_age: pd.DataFrame):
    title = "③ 메뉴별 연령대 주문량"
    if df_menu_age is None or df_menu_age.empty:
        _empty(ax, title); return

    age_order = ["10대","20대","30대","40대","50대+"]
    pivot = df_menu_age.pivot_table(index="menu_name", columns="age_bucket", values="qty_sum", aggfunc="sum").fillna(0)
    pivot = pivot.reindex(columns=[c for c in age_order if c in pivot.columns], fill_value=0)
    bottom = np.zeros(len(pivot))
    ind = np.arange(len(pivot))
    for age in pivot.columns:
        vals = pivot[age].values
        rects = ax.bar(ind, vals, bottom=bottom, label=age)
        bottom += vals
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_ylabel("수량")
    ax.set_xticks(ind)
    ax.set_xticklabels(pivot.index.tolist(), rotation=20, ha="right")
    _theme(ax)
    _legend(ax, loc="upper right", ncol=3)

def plot_hour_menu_heat(ax, df_hour_menu: pd.DataFrame):
    title = "④ 시간대별 메뉴 주문량(수량 합)"
    if df_hour_menu is None or df_hour_menu.empty:
        _empty(ax, title); return

    pivot = df_hour_menu.pivot_table(index="menu_name", columns="hour", values="qty_sum", aggfunc="sum").fillna(0)
    all_hours = list(range(24))
    pivot = pivot.reindex(columns=[h for h in all_hours if h in pivot.columns], fill_value=0)
    im = ax.imshow(pivot.values, aspect="auto")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("시간(KST)")
    ax.set_ylabel("메뉴")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(h) for h in pivot.columns], fontsize=7)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index.tolist(), fontsize=8)

def plot_new_vs_return(ax, df_new_return: pd.DataFrame):
    title = "⑤ 신규 vs 재주문 (건수)"
    if df_new_return is None or df_new_return.empty:
        _empty(ax, title); return

    data = df_new_return.copy()
    if "label" not in data.columns and "new" in data.columns:
        data["label"] = data["new"].map({True:"신규", False:"재주문"})
    
    order = ["신규", "재주문"]
    data = (data.set_index("label")
                  .reindex(order)
                  .fillna({"count": 0})
                  .reset_index())
    
    labels = data["label"].tolist()
    sizes = data["count"].astype(int).tolist()
    if sum(sizes) == 0:
        _empty(ax, title); return
    res = ax.pie(sizes, labels=labels, autopct="%1.0f%%", startangle=90, wedgeprops={"linewidth":0.5, "edgecolor":"white"})
    ax.set_title(title, fontsize=11, fontweight="bold")

def plot_cancel_rate(ax, df_cancel_rate: pd.DataFrame):
    title = "⑥ 취소율 (모집합: 완료/환불/거절/실패)"
    if df_cancel_rate is None or df_cancel_rate.empty:
        _empty(ax, title); return

    total = int(df_cancel_rate.iloc[0]["total"])
    cancel = int(df_cancel_rate.iloc[0]["cancel"])
    rate = float(df_cancel_rate.iloc[0]["cancel_rate"]) if total > 0 else 0.0

    cats = ["취소", "전체"]
    vals = [cancel, total]
    rects = ax.bar(cats, vals)
    _annot_bar(ax, rects, fmt="{:,.0f}")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_ylabel("건수")
    ax.text(0.5, max(vals)*1.05 if max(vals) else 1, f"취소율: {rate*100:.1f}%", ha="center", va="bottom", fontsize=10)
    _theme(ax)

def plot_reorder_hist(ax, df_reorder_hist: pd.DataFrame):
    title = "⑦ 재주문 간격 분포"
    if df_reorder_hist is None or df_reorder_hist.empty:
        _empty(ax, title); return

    order = ["0–7일","8–30일","31–90일","91–180일","181일+"]
    data = df_reorder_hist.set_index("gap_bin").reindex(order).fillna(0).reset_index()
    rects = ax.bar(data["gap_bin"], data["count"])
    _annot_bar(ax, rects)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_ylabel("건수")
    ax.set_xticklabels(data["gap_bin"], rotation=0)
    _theme(ax)

def plot_reorder_top3(ax, df_reorder_top3: pd.DataFrame):
    title = "⑧ 재주문 많은 메뉴 TOP3"
    if df_reorder_top3 is None or df_reorder_top3.empty:
        _empty(ax, title); return

    x = df_reorder_top3["menu_name"].astype(str)
    y = df_reorder_top3["qty_sum"].astype(float)
    rects = ax.bar(x, y)
    _annot_bar(ax, rects)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_ylabel("수량")
    ax.set_xticklabels(x, rotation=10, ha="right")
    _theme(ax)

# --------- 메인: PDF 한 페이지 빌드 ---------
def build_onepage_pdf(
    df_menu_total: pd.DataFrame,
    df_menu_gender: pd.DataFrame,
    df_menu_age: pd.DataFrame,
    df_hour_menu: pd.DataFrame,
    df_new_return: pd.DataFrame,
    df_cancel_rate: pd.DataFrame,
    df_reorder_hist: pd.DataFrame,  # 추가 전
    df_reorder_top3: pd.DataFrame,  # 추가 전
    store_id: str,
    start_date: date,
    end_date: date,
    outfile,
):
    """
    outfile: 바이너리 write 가능한 파일 객체 (예: open(path, 'wb'))
    """
    # 레이아웃
    fig = plt.figure(figsize=(11.69, 8.27))  # A4 가로
    gs = GridSpec(nrows=3, ncols=3, figure=fig, height_ratios=[0.18, 0.41, 0.41], hspace=0.6, wspace=0.35)

    # 타이틀
    ax_title = fig.add_subplot(gs[0, :])
    period_txt = f"{start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')} (KST)"
    ax_title.axis("off")
    ax_title.text(0.01, 0.65, "월간 리포트", fontsize=20, fontweight="bold")
    ax_title.text(0.01, 0.30, f"Store: {store_id}", fontsize=11)
    ax_title.text(0.70, 0.30, f"기간: {period_txt}", fontsize=11)

    ax1 = fig.add_subplot(gs[1, 0])
    plot_menu_total(ax1, df_menu_total)

    ax2 = fig.add_subplot(gs[1, 1])
    plot_menu_gender(ax2, df_menu_gender)

    ax3 = fig.add_subplot(gs[1, 2])
    plot_menu_age(ax3, df_menu_age)

    ax4 = fig.add_subplot(gs[2, 0])
    plot_hour_menu_heat(ax4, df_hour_menu)

    ax5 = fig.add_subplot(gs[2, 1])
    plot_new_vs_return(ax5, df_new_return)

    ax6 = fig.add_subplot(gs[2, 2])
    plot_cancel_rate(ax6, df_cancel_rate)

    pp = PdfPages(outfile)
    pp.savefig(fig, bbox_inches="tight")
    pp.close()
    plt.close(fig)
