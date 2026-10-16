"""Draw a manuscript-ready schematic of the SE-AlexNet human-model alignment workflow."""

from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from matplotlib import font_manager
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.offsetbox import AnnotationBbox, OffsetImage


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
FACE_DIR = ROOT / "0_InputImages"

FONT_PATH = "/System/Library/Fonts/Hiragino Sans GB.ttc"
font_manager.fontManager.addfont(FONT_PATH)
CN_FONT = font_manager.FontProperties(fname=FONT_PATH).get_name()

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": [CN_FONT, "Arial", "DejaVu Sans", "Liberation Sans"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 7,
    "axes.linewidth": 0.8,
})

COL = {
    "ink": "#26313C",
    "muted": "#6C7884",
    "hair": "#C8D0D7",
    "panel": "#F7F9FA",
    "human": "#347EAA",
    "human_pale": "#EAF4FA",
    "model": "#6E5AA7",
    "model_pale": "#F1EEFA",
    "align": "#D26A4E",
    "align_pale": "#FCEDE8",
    "green": "#3E927E",
    "green_pale": "#EAF6F2",
    "bluegray": "#607F9F",
    "bluegray_pale": "#ECF2F7",
    "gold": "#D99B33",
    "gold_pale": "#FCF4E4",
    "heat": "#DD4D3E",
    "yellow": "#F2C94C",
}


def rbox(ax, x, y, w, h, fc="white", ec=None, lw=0.9, radius=0.012, z=2):
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.006,rounding_size={radius}",
        facecolor=fc, edgecolor=ec or COL["hair"], linewidth=lw, zorder=z,
    )
    ax.add_patch(p)
    return p


def arrow(ax, p0, p1, color=None, lw=1.1, ms=8, rad=0.0, z=1):
    a = FancyArrowPatch(
        p0, p1, arrowstyle="-|>", mutation_scale=ms,
        color=color or COL["muted"], linewidth=lw,
        connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0, zorder=z,
    )
    ax.add_patch(a)


def label(ax, x, y, s, size=6.2, color=None, weight="normal", ha="center", va="center", **kwargs):
    kwargs.setdefault("zorder", 10)
    return ax.text(
        x, y, s, fontsize=size, color=color or COL["ink"], fontweight=weight,
        ha=ha, va=va, fontfamily=CN_FONT, linespacing=1.22, **kwargs,
    )


def add_face(ax, path, xy, zoom=0.21, frame=True, edge=None, z=5):
    arr = mpimg.imread(path)
    oi = OffsetImage(arr, zoom=zoom, interpolation="hanning")
    ab = AnnotationBbox(
        oi, xy, frameon=frame,
        bboxprops=dict(boxstyle="round,pad=0.04,rounding_size=0.03",
                       edgecolor=edge or COL["hair"], facecolor="white", linewidth=0.7),
        zorder=z,
    )
    ax.add_artist(ab)
    return ab


def panel_tag(ax, x, y, letter, title):
    label(ax, x, y, letter, size=8.5, weight="bold", ha="left")
    label(ax, x + 0.025, y, title, size=7.4, weight="bold", ha="left")


def person_icon(ax, x, y, color):
    ax.add_patch(Circle((x, y + 0.018), 0.012, facecolor=color, edgecolor="none", zorder=4))
    ax.add_patch(Ellipse((x, y - 0.010), 0.052, 0.042, facecolor=color, edgecolor="none", zorder=3))
    ax.add_patch(Rectangle((x - 0.030, y - 0.033), 0.060, 0.020, facecolor="white", edgecolor="none", zorder=4))


def monitor_icon(ax, x, y, face_path):
    ax.add_patch(FancyBboxPatch((x - 0.048, y - 0.038), 0.096, 0.076,
                                boxstyle="round,pad=0.003,rounding_size=0.006",
                                facecolor="#1F2429", edgecolor=COL["ink"], linewidth=0.8, zorder=3))
    arr = mpimg.imread(face_path)
    ax.imshow(arr, extent=(x - 0.036, x + 0.036, y - 0.027, y + 0.027),
              aspect="auto", zorder=4)
    ax.plot([x, x], [y - 0.041, y - 0.052], color=COL["ink"], lw=1.0, zorder=3)
    ax.plot([x - 0.022, x + 0.022], [y - 0.052, y - 0.052], color=COL["ink"], lw=1.0, zorder=3)


def keycap(ax, x, y, key, answer, fc, ec):
    rbox(ax, x - 0.031, y - 0.026, 0.062, 0.052, fc=fc, ec=ec, lw=0.8, radius=0.008, z=4)
    label(ax, x, y + 0.007, key, size=7.0, weight="bold", color=ec)
    label(ax, x, y - 0.011, answer, size=5.0, color=COL["muted"])


def feature_stack(ax, x, y, w, h, n=4, color="#7E72B5", label_text=None):
    for i in range(n - 1, -1, -1):
        dx, dy = 0.006 * i, 0.005 * i
        rect = Rectangle((x + dx, y + dy), w, h, facecolor=mpl.colors.to_rgba(color, 0.12 + 0.08*i),
                         edgecolor=color, linewidth=0.65, zorder=3+i)
        ax.add_patch(rect)
        for gx, gy in ((0.25, 0.72), (0.62, 0.48), (0.38, 0.25), (0.78, 0.78)):
            ax.add_patch(Circle((x + dx + gx*w, y + dy + gy*h), 0.055*w,
                                facecolor=mpl.colors.to_rgba(COL["gold"], 0.40), edgecolor="none", zorder=4+i))
    if label_text:
        label(ax, x + w/2 + 0.009, y - 0.013, label_text, size=5.0, color=COL["muted"])


def channel_vector(ax, x, y, values, color, height=0.060, width=0.006, gap=0.003):
    maxv = max(values)
    for i, v in enumerate(values):
        h = height * (0.24 + 0.76*v/maxv)
        ax.add_patch(Rectangle((x + i*(width+gap), y), width, h,
                               facecolor=mpl.colors.to_rgba(color, 0.32 + 0.58*v/maxv),
                               edgecolor=color, linewidth=0.35, zorder=5))


def draw_se_module(ax, x, y, w, h):
    rbox(ax, x, y, w, h, fc=COL["model_pale"], ec=COL["model"], lw=1.0, radius=0.010, z=2)
    label(ax, x + 0.018, y + h - 0.018, "SE通道重标定", size=5.6, color=COL["model"], weight="bold", ha="left")
    vals = [0.45, 0.75, 0.35, 0.95, 0.55, 0.25, 0.82, 0.40]
    channel_vector(ax, x + 0.018, y + 0.027, vals, COL["model"], height=0.055, width=0.005, gap=0.003)
    label(ax, x + 0.044, y + 0.014, "C", size=5.0, color=COL["muted"])
    arrow(ax, (x + 0.079, y + 0.052), (x + 0.103, y + 0.052), color=COL["model"], ms=6)
    for i in range(3):
        ax.add_patch(Circle((x + 0.118, y + 0.034 + i*0.018), 0.006,
                            facecolor=COL["gold"], edgecolor="#A77320", linewidth=0.4, zorder=5))
    label(ax, x + 0.118, y + 0.014, "C/R", size=5.0, color=COL["muted"])
    arrow(ax, (x + 0.132, y + 0.052), (x + 0.156, y + 0.052), color=COL["model"], ms=6)
    channel_vector(ax, x + 0.163, y + 0.027, [0.25, 0.62, 0.22, 0.98, 0.35, 0.18, 0.78, 0.28],
                   COL["align"], height=0.055, width=0.005, gap=0.003)
    label(ax, x + 0.189, y + 0.014, "通道权重", size=5.0, color=COL["muted"])
    label(ax, x + w - 0.022, y + 0.052, "×", size=8, color=COL["model"], weight="bold")


def tiny_position_network(ax, x, y, pos, color):
    xs = [x, x + 0.034, x + 0.068, x + 0.102]
    names = ["特征", "FC6", "FC7", "FC8"]
    for i, (xx, name) in enumerate(zip(xs, names)):
        ax.add_patch(Rectangle((xx, y), 0.020, 0.032, facecolor="#EDF1F4",
                               edgecolor=COL["muted"], linewidth=0.45, zorder=3))
        label(ax, xx + 0.010, y - 0.009, name, size=5.0, color=COL["muted"])
        if i < 3:
            arrow(ax, (xx + 0.020, y + 0.016), (xs[i+1], y + 0.016), color=COL["muted"], ms=4.5, lw=0.65)
    se_x = [x + 0.023, x + 0.057, x + 0.091][pos]
    ax.add_patch(Rectangle((se_x, y + 0.006), 0.010, 0.020, facecolor=color,
                           edgecolor="white", linewidth=0.5, zorder=6))


def experiment_card(ax, x, y, w, h, number, title, subtitle, color, icon):
    rbox(ax, x, y, w, h, fc="white", ec=color, lw=0.95, radius=0.014, z=2)
    ax.add_patch(Circle((x + 0.025, y + h - 0.026), 0.015, facecolor=color, edgecolor="none", zorder=4))
    label(ax, x + 0.025, y + h - 0.026, str(number), size=6.2, color="white", weight="bold")
    label(ax, x + 0.048, y + h - 0.020, title, size=6.2, color=color, weight="bold", ha="left")
    label(ax, x + 0.048, y + h - 0.044, subtitle, size=5.0, color=COL["muted"], ha="left")
    icon(ax, x, y, w, h, color)


def main():
    fig = plt.figure(figsize=(7.2, 5.55), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    sad = FACE_DIR / "Full" / "bm_ave_1.jpg"
    neutral = FACE_DIR / "Full" / "bm_ave_11.jpg"
    happy = FACE_DIR / "Full" / "bm_ave_21.jpg"
    eye_mask = FACE_DIR / "E" / "E_11.jpg"
    nose_mask = FACE_DIR / "N" / "N_11.jpg"
    mouth_mask = FACE_DIR / "M" / "M_11.jpg"

    # Figure contract: schematic-led composite; the hero evidence is shared-stimulus alignment.
    label(ax, 0.035, 0.968, "人类行为与SE-AlexNet模型的对齐流程", size=10.2, weight="bold", ha="left")
    label(ax, 0.035, 0.937, "同一组悲伤—开心连续刺激，经由人类判断与模型输出形成可比较的心理物理表征", size=6.1,
          color=COL["muted"], ha="left")

    # a. Two emotion endpoints and morph continuum.
    panel_tag(ax, 0.035, 0.895, "a", "由两种端点情绪构建连续刺激")
    rbox(ax, 0.035, 0.748, 0.930, 0.125, fc=COL["panel"], ec="#D5DDE3", lw=0.75, radius=0.018)
    add_face(ax, sad, (0.125, 0.818), zoom=0.120, edge=COL["human"])
    add_face(ax, happy, (0.875, 0.818), zoom=0.120, edge=COL["align"])
    levels = [0, 20, 30, 40, 50, 60, 70, 80, 100]
    xs = np.linspace(0.225, 0.775, len(levels))
    for i in range(len(xs) - 1):
        t = i / (len(xs) - 2)
        c = tuple((1-t)*np.array(mpl.colors.to_rgb(COL["human"])) +
                  t*np.array(mpl.colors.to_rgb(COL["align"])))
        ax.plot(xs[i:i+2], [0.814, 0.814], color=c, lw=2.0, zorder=1)
    for xx, val in zip(xs, levels):
        ax.add_patch(Circle((xx, 0.814), 0.0042, facecolor="white",
                            edgecolor=COL["muted"], linewidth=0.65, zorder=2))
        ax.plot([xx, xx], [0.789, 0.797], color=COL["muted"], lw=0.6)
        label(ax, xx, 0.780, str(val), size=5.0, color=COL["muted"])
    label(ax, 0.125, 0.765, "悲伤端点\n0%开心", size=5.0, color=COL["human"], weight="bold")
    label(ax, 0.500, 0.852, "由两种情绪端点连续形变 · 选取9个开心信息强度", size=5.4,
          color=COL["muted"], weight="bold")
    label(ax, 0.875, 0.765, "开心端点\n100%开心", size=5.0, color=COL["align"], weight="bold")

    # b. Two parallel lanes.
    panel_tag(ax, 0.035, 0.708, "b", "相同刺激进入两条平行加工路径")
    rbox(ax, 0.035, 0.455, 0.445, 0.225, fc=COL["human_pale"], ec=COL["human"], lw=0.95, radius=0.018)
    rbox(ax, 0.520, 0.455, 0.445, 0.225, fc=COL["model_pale"], ec=COL["model"], lw=0.95, radius=0.018)
    label(ax, 0.058, 0.653, "人类行为通道", size=6.8, color=COL["human"], weight="bold", ha="left")
    label(ax, 0.543, 0.653, "模型计算通道", size=6.8, color=COL["model"], weight="bold", ha="left")
    arrow(ax, (0.310, 0.748), (0.255, 0.680), color=COL["human"], lw=1.25, ms=8)
    arrow(ax, (0.690, 0.748), (0.745, 0.680), color=COL["model"], lw=1.25, ms=8)

    # Human lane: participant -> display -> forced choice -> repeated trials -> human P(happy).
    person_icon(ax, 0.090, 0.585, COL["human"])
    label(ax, 0.090, 0.535, "40名被试", size=5.0, color=COL["human"], weight="bold")
    arrow(ax, (0.125, 0.585), (0.157, 0.585), color=COL["human"], ms=6)
    monitor_icon(ax, 0.212, 0.585, neutral)
    label(ax, 0.212, 0.522, "随机呈现", size=5.0, color=COL["muted"])
    arrow(ax, (0.263, 0.585), (0.288, 0.585), color=COL["human"], ms=6)
    keycap(ax, 0.322, 0.602, "F", "开心", "white", COL["align"])
    keycap(ax, 0.322, 0.541, "J", "悲伤", "white", COL["human"])
    arrow(ax, (0.356, 0.585), (0.382, 0.585), color=COL["human"], ms=6)
    rbox(ax, 0.382, 0.532, 0.076, 0.107, fc="white", ec=COL["human"], lw=0.75, radius=0.010)
    label(ax, 0.420, 0.603, "2AFC", size=6.0, color=COL["human"], weight="bold")
    label(ax, 0.420, 0.573, "9 × 20", size=5.2, weight="bold")
    label(ax, 0.420, 0.547, "180试次", size=5.0, color=COL["muted"])
    label(ax, 0.258, 0.482, "输出：各强度下判断“开心”的比例  P_human(x)", size=5.2,
          color=COL["human"], weight="bold")

    # Model lane: pretraining -> convolutional feature channels -> SE -> two-class output.
    rbox(ax, 0.540, 0.548, 0.086, 0.078, fc="white", ec=COL["model"], lw=0.75, radius=0.010)
    label(ax, 0.583, 0.603, "预训练", size=5.4, color=COL["model"], weight="bold")
    label(ax, 0.583, 0.573, "ImageNet\n或 VGGFace2", size=5.0, color=COL["muted"])
    arrow(ax, (0.626, 0.587), (0.647, 0.587), color=COL["model"], ms=6)
    feature_stack(ax, 0.649, 0.557, 0.046, 0.056, n=4, color=COL["model"], label_text="卷积特征通道")
    arrow(ax, (0.722, 0.587), (0.739, 0.587), color=COL["model"], ms=6)
    draw_se_module(ax, 0.741, 0.523, 0.174, 0.126)
    label(ax, 0.674, 0.630, "AffectNet迁移学习", size=5.0, color=COL["model"], weight="bold")
    arrow(ax, (0.915, 0.587), (0.935, 0.587), color=COL["model"], ms=6)
    ax.add_patch(Circle((0.944, 0.604), 0.009, facecolor=COL["align"], edgecolor="white", lw=0.5, zorder=5))
    ax.add_patch(Circle((0.944, 0.570), 0.009, facecolor=COL["human"], edgecolor="white", lw=0.5, zorder=5))
    label(ax, 0.948, 0.484, "输出：Softmax开心概率  P_model(x)", size=5.2,
          color=COL["model"], weight="bold", ha="right")

    # c. Alignment space, with actual PSE anchors and an illustrative psychometric icon.
    panel_tag(ax, 0.035, 0.417, "c", "在同一心理物理空间中实现人类—模型对齐")
    rbox(ax, 0.035, 0.252, 0.930, 0.140, fc="white", ec=COL["align"], lw=1.0, radius=0.018)
    arrow(ax, (0.258, 0.455), (0.258, 0.392), color=COL["human"], lw=1.15, ms=7)
    arrow(ax, (0.742, 0.455), (0.742, 0.392), color=COL["model"], lw=1.15, ms=7)
    label(ax, 0.070, 0.360, "相同9个刺激强度", size=5.2, color=COL["align"], weight="bold", ha="left")
    label(ax, 0.070, 0.330, "Sigmoid心理物理函数", size=5.2, ha="left")
    label(ax, 0.070, 0.300, "比较：PSE、RMSE、拟合质量", size=5.2, ha="left")

    # Small illustrative psychometric icon (no data values implied).
    xcurve = np.linspace(0.12, 0.88, 160)
    for pse, color, ls, lw in [(0.517, COL["ink"], "-", 1.0), (0.518, COL["model"], "--", 1.15)]:
        ycurve = 0.298 + 0.060*(1/(1+np.exp(-12*(xcurve-pse))))
        ax.plot(0.235 + 0.220*xcurve, ycurve, color=color, ls=ls, lw=lw, zorder=3)
    ax.plot([0.255, 0.435], [0.292, 0.292], color=COL["muted"], lw=0.6)
    ax.plot([0.255, 0.255], [0.292, 0.364], color=COL["muted"], lw=0.6)
    label(ax, 0.345, 0.270, "开心信息强度", size=5.0, color=COL["muted"])
    label(ax, 0.247, 0.330, "P(开心)", size=5.0, color=COL["muted"], rotation=90)
    ax.plot([0.360, 0.392], [0.365, 0.365], color=COL["ink"], lw=1.0)
    label(ax, 0.396, 0.365, "人类", size=5.0, ha="left", color=COL["ink"])
    ax.plot([0.360, 0.392], [0.348, 0.348], color=COL["model"], lw=1.1, ls="--")
    label(ax, 0.396, 0.348, "最优模型", size=5.0, ha="left", color=COL["model"])

    # Actual key PSE positions from the manuscript.
    x0, x1 = 0.515, 0.920
    yaxis = 0.310
    ax.plot([x0, x1], [yaxis, yaxis], color=COL["muted"], lw=0.75)
    for v in [0.5, 0.6, 0.7, 0.8]:
        xx = x0 + (v-0.45)/(0.90-0.45)*(x1-x0)
        ax.plot([xx, xx], [yaxis-0.006, yaxis+0.006], color=COL["muted"], lw=0.55)
        label(ax, xx, yaxis-0.017, f"{v:.1f}", size=5.0, color=COL["muted"])
    def xpos(v):
        return x0 + (v-0.45)/(0.90-0.45)*(x1-x0)
    human_lo, human_hi = xpos(0.517-0.013), xpos(0.517+0.013)
    ax.add_patch(Rectangle((human_lo, yaxis+0.031), human_hi-human_lo, 0.020,
                           facecolor=mpl.colors.to_rgba(COL["human"], 0.18), edgecolor="none", zorder=2))
    for val, yy, color, name in [
        (0.517, yaxis+0.041, COL["human"], "人类 0.517 ± 0.013"),
        (0.518, yaxis+0.015, COL["model"], "面孔预训练·后序R16  0.518"),
        (0.662, yaxis-0.010, COL["gold"], "面孔预训练 AlexNet  0.662"),
        (0.836, yaxis-0.035, COL["muted"], "面孔预训练 VGG16  0.836"),
    ]:
        xx = xpos(val)
        ax.scatter([xx], [yy], s=18, color=color, edgecolor="white", linewidth=0.5, zorder=5)
        label(ax, xx + 0.010, yy, name, size=5.0, color=color, ha="left", weight="bold" if val < 0.52 else "normal")
    label(ax, 0.717, 0.370, "关键对齐结果（PSE）", size=5.5, color=COL["align"], weight="bold")

    # d. Experimental progression with visual icons.
    panel_tag(ax, 0.035, 0.215, "d", "三个实验逐步确定压缩机制并检验特征使用")

    def icon_exp1(ax_, x, y, w, h, color):
        widths = [0.078, 0.060, 0.044, 0.030, 0.019]
        for i, ww in enumerate(widths):
            yy = y + 0.020 + i*0.015
            ax_.add_patch(Rectangle((x + 0.030, yy), ww, 0.008,
                                    facecolor=mpl.colors.to_rgba(color, 0.25+0.13*i),
                                    edgecolor=color, linewidth=0.35, zorder=4))
        label(ax_, x + 0.132, y + 0.057, "R = 2, 4, 8, 16, 32", size=5.0, color=color, ha="left", weight="bold")
        label(ax_, x + 0.132, y + 0.034, "后序位置：FC7—FC8", size=5.0, color=COL["muted"], ha="left")

    def icon_exp2(ax_, x, y, w, h, color):
        for row, pos in enumerate([0, 1, 2]):
            tiny_position_network(ax_, x + 0.026, y + 0.019 + row*0.025, pos, color)
        label(ax_, x + 0.162, y + 0.059, "前序", size=5.0, color=COL["muted"], ha="left")
        label(ax_, x + 0.162, y + 0.044, "中序", size=5.0, color=COL["muted"], ha="left")
        label(ax_, x + 0.162, y + 0.029, "后序", size=5.0, color=color, ha="left", weight="bold")

    def icon_exp3(ax_, x, y, w, h, color):
        add_face(ax_, neutral, (x + 0.070, y + 0.052), zoom=0.105, edge=color, z=5)
        ax_.add_patch(Ellipse((x + 0.070, y + 0.064), 0.055, 0.019,
                              facecolor=mpl.colors.to_rgba(COL["heat"], 0.30), edgecolor="none", zorder=7))
        ax_.add_patch(Ellipse((x + 0.070, y + 0.033), 0.040, 0.014,
                              facecolor=mpl.colors.to_rgba(COL["yellow"], 0.36), edgecolor="none", zorder=7))
        add_face(ax_, eye_mask, (x + 0.135, y + 0.052), zoom=0.078, edge=COL["hair"], z=5)
        add_face(ax_, nose_mask, (x + 0.182, y + 0.052), zoom=0.078, edge=COL["hair"], z=5)
        add_face(ax_, mouth_mask, (x + 0.229, y + 0.052), zoom=0.078, edge=COL["hair"], z=5)
        label(ax_, x + 0.178, y + 0.012, "完整 / 遮挡眼、鼻、口", size=5.0, color=COL["muted"])

    experiment_card(ax, 0.035, 0.034, 0.282, 0.148, 1, "压缩强度", "寻找更接近人类行为的压缩比", COL["green"], icon_exp1)
    experiment_card(ax, 0.359, 0.034, 0.282, 0.148, 2, "插入位置", "比较低层、中层与后期表征压缩", COL["bluegray"], icon_exp2)
    experiment_card(ax, 0.683, 0.034, 0.282, 0.148, 3, "可解释性与有损输入", "Grad-CAM、ROI定量与关键区域遮挡", COL["align"], icon_exp3)
    arrow(ax, (0.317, 0.104), (0.359, 0.104), color=COL["muted"], lw=0.9, ms=7)
    arrow(ax, (0.641, 0.104), (0.683, 0.104), color=COL["muted"], lw=0.9, ms=7)

    # Image artists should not force a square data aspect on the full figure canvas.
    ax.set_aspect("auto")

    base = OUT / "human_model_alignment_workflow_v2"
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.05)
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.05)
    fig.savefig(base.with_suffix(".png"), dpi=600, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight", pad_inches=0.05)


if __name__ == "__main__":
    main()
