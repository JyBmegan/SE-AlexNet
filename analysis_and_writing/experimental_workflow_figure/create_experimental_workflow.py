"""Create an editable overview schematic for the SE-AlexNet study."""

from pathlib import Path

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, PathPatch
from matplotlib.path import Path as MplPath


# Required editable-vector settings.
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
mpl.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42})
plt.rcParams['font.size'] = 7


OUT = Path(__file__).resolve().parent

COL = {
    'ink': '#2B3440',
    'muted': '#65727E',
    'line': '#AAB4BC',
    'panel': '#F7F9FA',
    'human': '#2D74A6',
    'human_light': '#EAF3F9',
    'model': '#6B5AA6',
    'model_light': '#F0EDFA',
    'accent': '#D06A52',
    'accent_light': '#FBEDEA',
    'experiment': '#3B8E7B',
    'experiment_light': '#EAF6F2',
}


def box(ax, x, y, w, h, text, fc='white', ec=None, fontsize=6.5, weight='normal',
        radius=0.012, lw=0.9, color=None):
    ec = ec or COL['line']
    p = FancyBboxPatch((x, y), w, h, boxstyle=f'round,pad=0.006,rounding_size={radius}',
                       facecolor=fc, edgecolor=ec, linewidth=lw, zorder=2)
    ax.add_patch(p)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=fontsize,
            fontweight=weight, color=color or COL['ink'], linespacing=1.25, zorder=3)
    return p


def arrow(ax, start, end, color=None, lw=1.15, ms=9, connectionstyle='arc3'):
    a = FancyArrowPatch(start, end, arrowstyle='-|>', mutation_scale=ms, linewidth=lw,
                        color=color or COL['muted'], connectionstyle=connectionstyle,
                        shrinkA=0, shrinkB=0, zorder=1)
    ax.add_patch(a)


def face(ax, cx, cy, r, mood, label=None):
    """Minimal, non-identifying morphed-face icon."""
    ax.add_patch(Circle((cx, cy), r, facecolor='#F7D9C5', edgecolor='#B78E79', lw=0.7, zorder=3))
    for dx in (-0.34*r, 0.34*r):
        ax.add_patch(Circle((cx + dx, cy + 0.22*r), 0.07*r, color=COL['ink'], zorder=4))
    if mood < 0:
        verts = [(cx-0.38*r, cy-0.30*r), (cx, cy-0.05*r), (cx+0.38*r, cy-0.30*r)]
        codes = [MplPath.MOVETO, MplPath.CURVE3, MplPath.CURVE3]
    elif mood > 0:
        verts = [(cx-0.38*r, cy-0.10*r), (cx, cy-0.36*r), (cx+0.38*r, cy-0.10*r)]
        codes = [MplPath.MOVETO, MplPath.CURVE3, MplPath.CURVE3]
    else:
        verts = [(cx-0.35*r, cy-0.20*r), (cx+0.35*r, cy-0.20*r)]
        codes = [MplPath.MOVETO, MplPath.LINETO]
    ax.add_patch(PathPatch(MplPath(verts, codes), fill=False, edgecolor=COL['ink'], lw=0.9, zorder=4))
    if label:
        ax.text(cx, cy-r-0.020, label, ha='center', va='top', fontsize=5.6, color=COL['muted'])


def experiment_card(ax, x, title, subtitle, content, accent, y=0.075, w=0.273, h=0.205):
    box(ax, x, y, w, h, '', fc='white', ec=accent, lw=1.15, radius=0.018)
    ax.add_patch(FancyBboxPatch((x, y+h-0.045), w, 0.045,
                                boxstyle='round,pad=0.006,rounding_size=0.018',
                                facecolor=accent, edgecolor=accent, linewidth=0, zorder=3))
    ax.text(x+w/2, y+h-0.0225, title, ha='center', va='center', fontsize=7.0,
            fontweight='bold', color='white', zorder=4)
    ax.text(x+w/2, y+h-0.065, subtitle, ha='center', va='top', fontsize=5.8,
            fontweight='bold', color=accent)
    ax.text(x+w/2, y+0.061, content, ha='center', va='center', fontsize=5.6,
            color=COL['ink'], linespacing=1.28)


def main():
    fig = plt.figure(figsize=(7.2, 5.25), facecolor='white')  # double-column width
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # Title and shared stimuli.
    ax.text(0.03, 0.965, 'Overview of the experimental design and analysis workflow',
            fontsize=10, fontweight='bold', ha='left', va='top', color=COL['ink'])
    ax.text(0.03, 0.935, 'The same morphed facial-expression stimuli link human behavioural data to CNN evaluation.',
            fontsize=6.1, ha='left', va='top', color=COL['muted'])

    box(ax, 0.032, 0.786, 0.936, 0.102, '', fc=COL['panel'], ec='#D7DEE3', lw=0.8, radius=0.018)
    ax.text(0.060, 0.849, 'Shared test stimuli', fontsize=7, fontweight='bold', color=COL['ink'], va='center')
    ax.text(0.060, 0.817, '9 average faces spanning sad to happy', fontsize=5.7, color=COL['muted'], va='center')
    face(ax, 0.305, 0.838, 0.026, -1, 'sad')
    face(ax, 0.385, 0.838, 0.026, -0.35, 'low')
    face(ax, 0.465, 0.838, 0.026, 0, 'neutral')
    face(ax, 0.545, 0.838, 0.026, 0.35, 'high')
    face(ax, 0.625, 0.838, 0.026, 1, 'happy')
    ax.text(0.700, 0.843, '0%, 20%, 30%, 40%,\n50%, 60%, 70%, 80%, 100%',
            fontsize=5.8, ha='left', va='center', color=COL['ink'], linespacing=1.25)

    # Human and model branches.
    arrow(ax, (0.32, 0.784), (0.245, 0.704), color=COL['human'])
    arrow(ax, (0.61, 0.784), (0.755, 0.704), color=COL['model'])
    box(ax, 0.060, 0.602, 0.370, 0.098, 'Human behavioural benchmark',
        fc=COL['human_light'], ec=COL['human'], fontsize=7.1, weight='bold', color=COL['human'])
    box(ax, 0.570, 0.602, 0.370, 0.098, 'CNN modelling pipeline',
        fc=COL['model_light'], ec=COL['model'], fontsize=7.1, weight='bold', color=COL['model'])

    # Human branch detail.
    box(ax, 0.072, 0.470, 0.103, 0.080, '40 participants\nnormal/corrected vision', fc='white', ec=COL['human'], fontsize=5.7)
    box(ax, 0.196, 0.470, 0.105, 0.080, '2AFC task\nF: happy   J: sad', fc='white', ec=COL['human'], fontsize=5.7)
    box(ax, 0.322, 0.470, 0.096, 0.080, 'Random order\n9 faces × 20\n= 180 trials', fc='white', ec=COL['human'], fontsize=5.5)
    arrow(ax, (0.175, 0.510), (0.196, 0.510), color=COL['human'], ms=7)
    arrow(ax, (0.301, 0.510), (0.322, 0.510), color=COL['human'], ms=7)
    arrow(ax, (0.245, 0.600), (0.245, 0.550), color=COL['human'], ms=7)
    box(ax, 0.122, 0.365, 0.246, 0.062, 'Psychometric function → PSE and response probability',
        fc=COL['human_light'], ec=COL['human'], fontsize=5.6, weight='bold', color=COL['human'])
    arrow(ax, (0.245, 0.470), (0.245, 0.427), color=COL['human'], ms=7)

    # Model branch detail.
    box(ax, 0.582, 0.470, 0.105, 0.080, 'Pre-training\nImageNet or\nVGGFace2', fc='white', ec=COL['model'], fontsize=5.7)
    box(ax, 0.708, 0.470, 0.105, 0.080, 'Transfer learning\nAffectNet\nemotion task', fc='white', ec=COL['model'], fontsize=5.7)
    box(ax, 0.834, 0.470, 0.094, 0.080, 'Same 9 faces\nSoftmax output\nprobability', fc='white', ec=COL['model'], fontsize=5.5)
    arrow(ax, (0.687, 0.510), (0.708, 0.510), color=COL['model'], ms=7)
    arrow(ax, (0.813, 0.510), (0.834, 0.510), color=COL['model'], ms=7)
    arrow(ax, (0.755, 0.600), (0.755, 0.550), color=COL['model'], ms=7)
    box(ax, 0.632, 0.365, 0.246, 0.062, 'Psychometric function → PSE, RMSE and fit quality',
        fc=COL['model_light'], ec=COL['model'], fontsize=5.6, weight='bold', color=COL['model'])
    arrow(ax, (0.755, 0.470), (0.755, 0.427), color=COL['model'], ms=7)

    # Convergence.
    arrow(ax, (0.368, 0.396), (0.485, 0.323), color=COL['human'], ms=8, connectionstyle='arc3,rad=-0.08')
    arrow(ax, (0.632, 0.396), (0.515, 0.323), color=COL['model'], ms=8, connectionstyle='arc3,rad=0.08')
    box(ax, 0.354, 0.286, 0.292, 0.050, 'Human–model comparison: behavioural similarity',
        fc=COL['accent_light'], ec=COL['accent'], fontsize=6.2, weight='bold', color=COL['accent'])
    arrow(ax, (0.50, 0.286), (0.50, 0.274), color=COL['accent'], ms=8)

    # Experiments.
    experiment_card(ax, 0.045, 'Experiment 1', 'Compression strength',
                    'Late-stage SE insertion\nR = 2, 4, 8, 16, 32\nvs. AlexNet and VGG16', COL['experiment'])
    experiment_card(ax, 0.364, 'Experiment 2', 'Insertion position',
                    'Early, middle, or late\nSE insertion × pre-training\n× compression ratio', '#5B7896')
    experiment_card(ax, 0.683, 'Experiment 3', 'Feature-use analysis',
                    'Grad-CAM and ROI saliency\nFull face vs. eye, nose,\nand mouth occlusion', COL['accent'])
    arrow(ax, (0.318, 0.178), (0.364, 0.178), color=COL['muted'], ms=8)
    arrow(ax, (0.637, 0.178), (0.683, 0.178), color=COL['muted'], ms=8)

    ax.text(0.50, 0.025, 'Compression strength and insertion stage determine behavioural similarity and model feature use.',
            ha='center', va='center', fontsize=6.1, color=COL['muted'], style='italic')

    base = OUT / 'experimental_workflow_overview'
    fig.savefig(base.with_suffix('.svg'), bbox_inches='tight', pad_inches=0.06)
    fig.savefig(base.with_suffix('.pdf'), bbox_inches='tight', pad_inches=0.06)
    fig.savefig(base.with_suffix('.png'), dpi=600, bbox_inches='tight', pad_inches=0.06)
    fig.savefig(base.with_suffix('.tiff'), dpi=600, bbox_inches='tight', pad_inches=0.06)


if __name__ == '__main__':
    main()
