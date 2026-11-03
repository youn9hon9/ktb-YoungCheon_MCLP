"""Plot report table 3; these are reported final scores, not new training results."""
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'kde'))
from reconstruct import IMPORTANCE
font = Path.home() / '.local/share/fonts/malgun.ttf'
if font.exists():
    font_manager.fontManager.addfont(font)
    plt.rcParams['font.family'] = 'Malgun Gothic'
else:
    plt.rcParams['font.family'] = ['Noto Sans CJK KR', 'DejaVu Sans']
rows = sorted(IMPORTANCE.items(), key=lambda p: p[1], reverse=True)
fig, ax = plt.subplots(figsize=(11, 7.2))
fig.subplots_adjust(left=.20, right=.95, top=.97, bottom=.08)
bars = ax.barh([r[0] for r in rows], [r[1] for r in rows], height=.65,
               color=['#2967ba' if i < 3 else '#a8c6e9' for i in range(len(rows))])
ax.invert_yaxis()
ax.set_xlim(0, 1)
ax.set_xticks([0, .2, .4, .6, .8, 1])
ax.xaxis.grid(True, color='#e9edf1', linewidth=.8)
ax.set_axisbelow(True)
ax.tick_params(axis='both', length=0, labelsize=11, colors='#34495e', pad=9)
for spine in ax.spines.values():
    spine.set_visible(False)
for bar, (_, value) in zip(bars, rows):
    ax.text(value+.012, bar.get_y()+bar.get_height()/2, f'{value:.4f}',
            va='center', fontsize=11, color='#34495e')
fig.savefig(Path(__file__).with_name('poi_importance.png'), dpi=200, facecolor='white')
