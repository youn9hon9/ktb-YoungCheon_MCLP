"""Explicit reconstruction, not a reproduction of the missing original KDE code."""
import argparse
import hashlib
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import intersects_xy

ROOT = Path(__file__).resolve().parents[2]
IMPORTANCE = {
    '노인복지시설': .7643, '편의점': .6409, '학교': .6246,
    '청소년복지시설': .4888, '체육시설': .4789, '문화공연시설': .4544,
    '아동돌봄시설': .4061, '은행': .3813, '학원': .3492, '대형마트': .3138,
    '주차장': .2557, '의료시설': .1484, '음식점': .1211, '미용실': .0358,
}
MAPPING = {k: k for k in IMPORTANCE}
MAPPING.update({'의료기관': '의료시설', '미용업': '미용실', '문화시설': '문화공연시설',
                '어린이집,유치원': '아동돌봄시설', '지역아동센터': '아동돌봄시설'})


def gaussian_density(points, xs, ys, bandwidth):
    """Per-category probability density; truncate each Gaussian at +/-5 sigma.

    Evaluate directly on cell centers (no raster binning of input points).
    The omitted continuous mass is less than 1.15e-6 per normalized category.
    """
    result = np.zeros((len(ys), len(xs)), dtype=np.float64)
    if not len(points):
        return result
    for x, y in points:
        ix0, ix1 = np.searchsorted(xs, [x - 5 * bandwidth, x + 5 * bandwidth])
        iy0, iy1 = np.searchsorted(ys, [y - 5 * bandwidth, y + 5 * bandwidth])
        dx = (xs[ix0:ix1] - x) / bandwidth
        dy = (ys[iy0:iy1] - y) / bandwidth
        result[iy0:iy1, ix0:ix1] += np.exp(-.5 * (dy[:, None] ** 2 + dx[None, :] ** 2))
    return result / (len(points) * 2 * np.pi * bandwidth ** 2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cell-size', type=float, default=25)
    parser.add_argument('--output', type=Path, default=Path(__file__).parent / 'output')
    args = parser.parse_args()
    if args.cell_size < 10:
        parser.error('Use a cell size of at least 10 m to limit memory use.')
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    poi_path = ROOT / 'data/영천시/영천시_POI_통합.csv'
    boundary_path = ROOT / 'data/행정동경계/bnd_sigungu_37070_2025_2Q.zip'
    boundary = gpd.read_file(boundary_path).to_crs(5179)
    geometry = boundary.geometry.union_all()
    raw = pd.read_csv(poi_path)
    raw['analysis_category'] = raw.category.map(MAPPING)
    known = raw[raw.analysis_category.notna()].copy()
    xy = known[['x', 'y']].apply(pd.to_numeric, errors='coerce')
    valid = xy.x.between(-180, 180) & xy.y.between(-90, 90)
    known = known[valid].copy()
    known[['x', 'y']] = xy[valid]
    points = gpd.GeoDataFrame(known, geometry=gpd.points_from_xy(known.x, known.y), crs=4326).to_crs(5179)
    inside = points.geometry.intersects(geometry)
    outside_count = int((~inside).sum())
    points = points[inside].copy()
    before_dedup = len(points)
    points = points.drop_duplicates(['analysis_category', 'x', 'y'])
    minx, miny, maxx, maxy = geometry.bounds
    step = args.cell_size
    left, bottom = np.floor(np.array([minx, miny]) / step) * step
    xs = np.arange(left + step / 2, maxx, step)
    ys = np.arange(bottom + step / 2, maxy, step)
    xx, yy = np.meshgrid(xs, ys)
    mask = intersects_xy(geometry, xx, yy)
    density = np.zeros_like(xx)
    rows = []
    for category, importance in IMPORTANCE.items():
        subset = points[points.analysis_category == category]
        coords = np.column_stack([subset.geometry.x, subset.geometry.y])
        component = gaussian_density(coords, xs, ys, importance * 1000)
        density += component
        rows.append({'category': category, 'importance': importance, 'bandwidth_m': importance * 1000,
                     'points': len(subset), 'integral_in_city': float(component[mask].sum() * step**2)})
    low, high = density[mask].min(), density[mask].max()
    score = (density - low) / (high - low)
    density[~mask] = np.nan
    score[~mask] = np.nan
    assert np.isfinite(score[mask]).all() and 0 <= np.nanmin(score) <= np.nanmax(score) <= 1
    np.savez_compressed(out / 'kde_grid.npz', x=xs, y=ys, density=density, score=score,
                        mask=mask, crs='EPSG:5179', cell_size=step)
    points.drop(columns='geometry').to_csv(out / 'included_poi.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(rows).to_csv(out / 'category_summary.csv', index=False, encoding='utf-8-sig')
    metadata = {'method': 'Gaussian per-category probability densities summed, then city-wide min-max normalization',
                'reconstruction_not_original': True, 'cell_size_m': step, 'crs': 'EPSG:5179',
                'kernel_cutoff_sigma': 5, 'mapping': MAPPING, 'importance': IMPORTANCE,
                'raw_rows': len(raw), 'included_unique_points': len(points),
                'excluded_categories': raw[raw.analysis_category.isna()].category.value_counts().to_dict(),
                'invalid_coordinates': int((~valid).sum()), 'outside_boundary': outside_count,
                'duplicates_removed': before_dedup - len(points),
                'missing_report_categories': [r['category'] for r in rows if r['points'] == 0],
                'source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in (poi_path, boundary_path)}}
    (out / 'metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    plot(out, boundary, points, xs, ys, score, step)
    print(json.dumps({k: metadata[k] for k in ('raw_rows', 'included_unique_points', 'duplicates_removed',
                                             'excluded_categories', 'missing_report_categories')}, ensure_ascii=False))


def plot(out, boundary, points, xs, ys, score, step):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager, colors
    from matplotlib.patches import Rectangle
    font = Path.home() / '.local/share/fonts/malgun.ttf'
    if font.exists():
        font_manager.fontManager.addfont(font)
        plt.rcParams['font.family'] = 'Malgun Gothic'
    else:
        plt.rcParams['font.family'] = ['Noto Sans CJK KR', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    fig, axes = plt.subplots(1, 2, figsize=(17, 8.7))
    fig.subplots_adjust(left=.04, right=.9, top=.81, bottom=.14, wspace=.12)
    fig.text(.04, .94, '영천시 POI 기반 활동 밀도', fontsize=24, color='#163447')
    fig.text(.04, .893, f'보고서의 유형별 대역폭을 적용한 KDE 재구성 · 공식 행정경계 · {step:g}m 계산 격자',
             fontsize=12, color='#657683')
    dong = gpd.read_file(ROOT / 'data/행정동경계/bnd_dong_37070_2025_2Q.zip').to_crs(5179)
    cmap = plt.get_cmap('YlOrRd').copy()
    cmap.set_bad('white')
    extent = [xs[0]-step/2, xs[-1]+step/2, ys[0]-step/2, ys[-1]+step/2]
    center = gpd.GeoSeries(gpd.points_from_xy([128.938], [35.975]), crs=4326).to_crs(5179).iloc[0]
    zoom = [center.x-6000, center.x+6000, center.y-4500, center.y+4500]
    for ax, title in zip(axes, ['① 영천시 전체', '② 도심 확대 · 같은 색상 기준']):
        im = ax.imshow(score, origin='lower', extent=extent, cmap=cmap,
                       norm=colors.PowerNorm(gamma=.5, vmin=0, vmax=1), interpolation='nearest')
        dong.boundary.plot(ax=ax, color='#a9b5b6', linewidth=.45, alpha=.7)
        boundary.boundary.plot(ax=ax, color='#243e53', linewidth=1.3)
        ax.set_title(title, loc='left', fontsize=14, color='#243e53', pad=14)
        ax.set_aspect('equal')
        ax.set_axis_off()
    minx, miny, maxx, maxy = boundary.total_bounds
    axes[0].set_xlim(minx-1500,maxx+1500);axes[0].set_ylim(miny-1500,maxy+1500)
    axes[0].add_patch(Rectangle((zoom[0],zoom[2]),zoom[1]-zoom[0],zoom[3]-zoom[2],
                              fill=False,edgecolor='#243e53',linestyle='--',linewidth=1.2))
    axes[1].set_xlim(zoom[:2]);axes[1].set_ylim(zoom[2:])
    # Scale bars and north arrows use the projected metric coordinates.
    for ax, length in zip(axes, [5000, 2000]):
        x0,x1=ax.get_xlim();y0,y1=ax.get_ylim();x=x0+.07*(x1-x0);y=y0+.035*(y1-y0)
        ax.plot([x,x+length],[y,y],lw=3,color='#243e53')
        ax.text(x+length/2,y+.022*(y1-y0),f'{length//1000} km',ha='center',fontsize=10,color='#243e53')
        ax.annotate('N',xy=(.94,.96),xytext=(.94,.85),xycoords='axes fraction',ha='center',
                    color='#243e53',arrowprops={'arrowstyle':'-|>','color':'#243e53'})
    cax=fig.add_axes([.925,.23,.014,.49])
    fig.colorbar(im,cax=cax,ticks=[0,.01,.05,.1,.25,.5,1]).set_label('상대 KDE 점수 (0–1) · 제곱근 색상 스케일',fontsize=10)
    fig.text(.04,.075,f'POI {len(points):,}개 · {points.analysis_category.nunique()}개 유형 | 가우시안 KDE, 유형별 밀도 정규화 후 합산 | 원래 분석의 재현 결과가 아닌 재구성입니다.',fontsize=10,color='#657683')
    fig.text(.04,.04,'점선: 도심 확대 범위. 카페·부동산 제외, 청소년복지시설 자료 없음. 색상은 실제 유동인구 수나 그늘막 필요 확률이 아닙니다.',fontsize=10,color='#657683')
    fig.savefig(out / 'yeongcheon_kde_wide.png', dpi=180, facecolor='white')
    plt.close(fig)


if __name__ == '__main__':
    main()
