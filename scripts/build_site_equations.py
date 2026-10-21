"""Render the manuscript equations as transparent PNGs for the project page."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1] / 'docs' / 'assets' / 'equations'
ROOT.mkdir(parents=True, exist_ok=True)
EQUATIONS = {
    '01-recalibration': r"x_i^{\prime}=x_i\,[\sigma(W_2\,\mathrm{ReLU}(W_1 x))]_i",
    '02-reduction': r"C_{\mathrm{out}}=\frac{C_{\mathrm{in}}}{R}",
    '03-psychometric': r"P(x)=\gamma+\frac{1-\delta-\gamma}{2}\left[1+\mathrm{erf}\left(\frac{x-\alpha}{\sqrt{2}\,\beta}\right)\right]",
    '04-rmse': r"\mathrm{RMSE}=\sqrt{\frac{1}{N}\sum_{i=1}^{N}(P_{\mathrm{model},i}-\overline{P}_{\mathrm{human},i})^2}",
    '05-delta-pse': r"\Delta\mathrm{PSE}_{ij}=|\mathrm{PSE}_{i}-\mathrm{PSE}_{\mathrm{human},j}|",
    '06-gradcam-weight': r"\alpha_k^c=\frac{1}{Z}\sum_{i,j}\frac{\partial y^c}{\partial A_{ij}^k}",
    '07-gradcam-map': r"L_{\mathrm{Grad-CAM}}^c=\mathrm{ReLU}\left(\sum_k\alpha_k^c A^k\right)",
    '08-pos': r"\mathrm{PoS}_{\mathrm{ROI}}=\frac{\sum_{(x,y)\in\mathrm{ROI}}L_{\mathrm{norm}}(x,y)}{\sum_{x,y}L_{\mathrm{norm}}(x,y)}",
    '09-diagnostic': r"\mathrm{DR}=\frac{\mathrm{PoS}_{\mathrm{eyes}}+\mathrm{PoS}_{\mathrm{mouth}}}{\mathrm{PoS}_{\mathrm{eyes}}+\mathrm{PoS}_{\mathrm{nose}}+\mathrm{PoS}_{\mathrm{mouth}}}",
}
plt.rcParams.update({'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix'})
for name, formula in EQUATIONS.items():
    fig = plt.figure(figsize=(12, 1.2), dpi=150)
    fig.patch.set_alpha(0)
    fig.text(.01, .5, f'${formula}$', fontsize=18, color='#24343a', va='center', ha='left')
    fig.savefig(ROOT / f'{name}.png', format='png', dpi=300,
                transparent=True, bbox_inches='tight', pad_inches=.03)
    plt.close(fig)
    print(f'{name}.png')
