"""Render matched POI/KDE panels with Folium, using the existing calculated grid."""
import json
from pathlib import Path

import contourpy
import folium
import geopandas as gpd
import numpy as np
import pandas as pd
from branca.element import Element, MacroElement, Template
from matplotlib import colormaps, colors
from shapely.geometry import Polygon

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).parent / 'output'
BLUE = '#1654cf'
LEVELS = [.002, .005, .01, .02, .05, .1, .2, .4, .7, 1.000001]


def contours(boundary):
    data = np.load(OUT / 'kde_grid.npz')
    # Extend the nearest evaluated values over boundary-adjacent NaN cells for
    # contour construction, then clip the resulting polygons to the exact city.
    # No values beyond two neighboring cells are filled or used in the analysis.
    z = data['score'].copy()
    for _ in range(2):
        total = np.zeros_like(z)
        count = np.zeros_like(z)
        for dy, dx in [(0,1),(0,-1),(1,0),(-1,0)]:
            neighbor = np.roll(z, (dy,dx), (0,1))
            if dy == 1: neighbor[0,:] = np.nan
            if dy == -1: neighbor[-1,:] = np.nan
            if dx == 1: neighbor[:,0] = np.nan
            if dx == -1: neighbor[:,-1] = np.nan
            finite = np.isfinite(neighbor)
            total += np.nan_to_num(neighbor)
            count += finite
        missing = ~np.isfinite(z) & (count > 0)
        z[missing] = total[missing] / count[missing]
    generator = contourpy.contour_generator(x=data['x'], y=data['y'], z=np.ma.masked_invalid(z),
                                            fill_type='OuterOffset')
    city = boundary.geometry.union_all()
    rows = []
    for low, high in zip(LEVELS[:-1], LEVELS[1:]):
        polygons, offsets = generator.filled(low, high)
        color = colors.to_hex(colormaps['Reds'](.25 + .7 * np.sqrt((low + min(high,1)) / 2)))
        for vertices, ring_offsets in zip(polygons, offsets):
            rings = [vertices[a:b] for a,b in zip(ring_offsets[:-1],ring_offsets[1:])]
            polygon = Polygon(rings[0], rings[1:]).intersection(city)
            if polygon.is_empty or polygon.area == 0:
                continue
            if polygon.geom_type not in ('Polygon','MultiPolygon'):
                continue
            assert polygon.difference(city).area < .01
            rows.append({'low': low, 'high': min(high,1), 'color': color, 'geometry': polygon})
    return gpd.GeoDataFrame(rows, crs=5179).to_crs(4326)


def panel(boundary, bounds, point_count, kde=False):
    m = folium.Map(tiles=None, zoom_control=False, prefer_canvas=True,
                   zoom_snap=.1, attribution_control=True)
    folium.TileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png',
                     attr='&copy; OpenStreetMap contributors', name='배경지도').add_to(m)
    folium.GeoJson(json.loads(boundary.to_crs(4326).to_json()),
                   style_function=lambda _: {'color': BLUE, 'weight': 2.2,
                                              'fillColor': '#9dc9fa', 'fillOpacity': .19}).add_to(m)
    m.fit_bounds(bounds, padding=(32,32))
    css = '''<style>
    .leaflet-container {background:#faf8f1!important; font-family:'Malgun Gothic',sans-serif;}
    .leaflet-tile-pane {filter:grayscale(1) sepia(.18); opacity:.32;}
    .map-legend {position:absolute;top:22px;right:22px;z-index:1000;background:rgba(255,255,255,.92);
    padding:14px 16px;border:1px solid #dfe5e8;border-radius:7px;color:#263a4b;font-size:14px;line-height:1.9;}
    .scale {position:absolute;bottom:30px;left:28px;z-index:1000;text-align:center;color:#243e53;font-size:13px;}
    .scale-bar {border-bottom:4px solid #243e53;border-left:2px solid #243e53;border-right:2px solid #243e53;height:6px;}
    .leaflet-control-attribution {font-size:9px!important;background:rgba(255,255,255,.65)!important;}
    </style>'''
    m.get_root().header.add_child(Element(css))
    legend = '<div class="map-legend"><span style="display:inline-block;width:22px;border-top:3px solid '+BLUE+';vertical-align:middle;margin-right:8px"></span>영천시 경계<br>'
    if kde:
        legend += '''상대 KDE 밀도 (0–1)<br><div style="width:175px;height:12px;background:linear-gradient(to right,#fee0d2,#fc9272,#ef3b2c,#99000d)"></div><div style="display:flex;justify-content:space-between;font-size:12px"><span>낮음</span><span>높음</span></div>'''
    else:
        legend += f'<span style="color:#df712a"><span style="font-size:19px;margin-right:8px">●</span>POI {point_count:,}개</span>'
    m.get_root().html.add_child(Element(legend+'</div><div class="scale">5 km<div class="scale-bar"></div></div>'))
    macro = MacroElement()
    macro._template = Template('''{% macro script(this, kwargs) %}
    function updateScale() {
      const m = {{this._parent.get_name()}};
      const y = m.getSize().y - 35;
      const a = m.containerPointToLatLng([28,y]);
      const b = m.containerPointToLatLng([128,y]);
      document.querySelector('.scale').style.width = (5000 / m.distance(a,b) * 100) + 'px';
    }
    {{this._parent.get_name()}}.on('moveend zoomend resize',updateScale);
    setTimeout(updateScale,300);
    {% endmacro %}''')
    m.add_child(macro)
    return m


def main():
    boundary = gpd.read_file(ROOT/'data/bnd_sigungu_37070_2025_2Q.zip').to_crs(5179)
    west,south,east,north = boundary.to_crs(4326).total_bounds
    bounds = [[south,west],[north,east]]
    points = pd.read_csv(OUT/'included_poi.csv')
    left = panel(boundary,bounds,len(points))
    for row in points.itertuples():
        folium.CircleMarker([row.y,row.x],radius=2.4,weight=0,fill=True,
                            fill_color='#e9792b',fill_opacity=.82).add_to(left)
    right = panel(boundary,bounds,len(points),kde=True)
    kde = contours(boundary)
    folium.GeoJson(json.loads(kde.to_json()),style_function=lambda f:{
        'color':f['properties']['color'],'weight':.75,'opacity':.82,
        'fillColor':f['properties']['color'],'fillOpacity':.5}).add_to(right)
    # Keep the city outline above the contour layer, with no second fill.
    folium.GeoJson(json.loads(boundary.to_crs(4326).to_json()),style_function=lambda _:{
        'color':BLUE,'weight':2.2,'fillOpacity':0}).add_to(right)
    left.save(OUT/'poi_map.html');right.save(OUT/'kde_map.html')
    (OUT/'maps.html').write_text('''<!doctype html><html><head><meta charset="utf-8"><style>
    html,body{margin:0;width:100%;height:100%;overflow:hidden;background:#fff}
    body{display:grid;grid-template-columns:1fr 1fr;gap:14px}
    iframe{width:100%;height:100%;border:0;display:block}
    </style></head><body><iframe src="poi_map.html"></iframe><iframe src="kde_map.html"></iframe></body></html>''',encoding='utf-8')
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={'width':1800,'height':900},device_scale_factor=2)
        page.goto((OUT/'maps.html').as_uri())
        for frame in page.frames[1:]:
            frame.wait_for_function('typeof L !== "undefined" && document.querySelectorAll(".leaflet-tile-loaded").length > 0',timeout=60000)
            frame.wait_for_function('Array.from(document.querySelectorAll(".leaflet-tile")).every(i=>i.complete && i.naturalWidth > 0)',timeout=60000)
        page.wait_for_timeout(2000)
        page.screenshot(path=str(OUT/'yeongcheon_kde_wide.png'))
        print('Rendered PNG; contour polygons:',len(kde))
        print('Tiles:',[f.locator('.leaflet-tile-loaded').count() for f in page.frames[1:]])
        browser.close()


if __name__ == '__main__':
    main()
