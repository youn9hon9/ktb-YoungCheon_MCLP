import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import folium
import geopandas as gpd
import pandas as pd
import streamlit as st
from shapely.geometry import box
from streamlit_folium import st_folium

BASE_DIR = Path(__file__).resolve().parent
SEED_PATH = BASE_DIR / 'data/영천시/영천시_횡단보도_좌표.csv'
SAVE_PATH = Path(os.environ.get('CROSSWALK_CSV_PATH', str(BASE_DIR / 'data/횡단보도_좌표.csv'))).resolve()
BOUNDARY_PATH = BASE_DIR / 'data/행정동경계/yeongcheon_dong_boundary.geojson'


def load_coords(path):
    frame = pd.read_csv(path)
    frame = frame[['위도', '경도']].apply(pd.to_numeric, errors='raise')
    if not (frame['위도'].between(-90, 90) & frame['경도'].between(-180, 180)).all():
        raise ValueError('유효하지 않은 위도·경도가 있습니다.')
    return list(frame.itertuples(index=False, name=None))


def save_coords(coords):
    """Write only on an explicit save, leaving the original seed CSV untouched."""
    SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with NamedTemporaryFile(mode='w', encoding='utf-8-sig', suffix='.csv',
                                dir=SAVE_PATH.parent, delete=False) as handle:
            temporary = Path(handle.name)
            pd.DataFrame(coords, columns=['위도', '경도']).to_csv(handle, index=False)
        temporary.replace(SAVE_PATH)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


@st.cache_data
def boundary_layers(raw):
    boundary = json.loads(raw)
    frame = gpd.GeoDataFrame.from_features(boundary['features'], crs='EPSG:4326')
    if frame.empty or not frame.geometry.geom_type.isin(['Polygon', 'MultiPolygon']).all():
        raise ValueError('Polygon 또는 MultiPolygon 경계가 필요합니다.')
    if not frame.is_valid.all():
        raise ValueError('유효하지 않은 경계 도형입니다.')
    minx, miny, maxx, maxy = frame.total_bounds
    if not (-180 <= minx < maxx <= 180 and -90 <= miny < maxy <= 90):
        raise ValueError('WGS84 경도·위도 좌표의 GeoJSON이 필요합니다.')
    step = 0.0045
    nx, ny = int((maxx - minx) / step) + 1, int((maxy - miny) / step) + 1
    if nx * ny > 100_000:
        raise ValueError('경계 범위가 너무 큽니다. 영천시 범위의 파일을 사용하세요.')
    cells = [box(minx + x * step, miny + y * step,
                 minx + (x + 1) * step, miny + (y + 1) * step)
             for x in range(nx) for y in range(ny)]
    grid = gpd.GeoDataFrame(geometry=cells, crs='EPSG:4326')
    grid = grid[grid.intersects(frame.geometry.union_all())]
    return json.loads(frame.to_json()), json.loads(grid.to_json())


st.set_page_config(layout='wide', page_title='영천시 횡단보도 수집')
st.title('영천시 횡단보도 추출')
st.caption('지도에서 횡단보도 위치를 클릭한 뒤 저장하세요.')

if 'coords' not in st.session_state:
    source = SAVE_PATH if SAVE_PATH.exists() else SEED_PATH
    try:
        st.session_state.coords = load_coords(source) if source.exists() else []
    except (OSError, ValueError, KeyError) as exc:
        st.error(f'좌표 CSV를 읽을 수 없습니다: {exc}')
        st.stop()

st.session_state.setdefault('last_center', [35.9749, 128.9461])
st.session_state.setdefault('zoom', 16)
st.session_state.setdefault('new_point', None)
st.session_state.setdefault('last_click', None)

raw_boundary = BOUNDARY_PATH.read_bytes() if BOUNDARY_PATH.exists() else None

m = folium.Map(location=st.session_state.last_center, zoom_start=st.session_state.zoom)
if raw_boundary is None:
    st.info('행정경계 파일이 없어 기본 지도로 시작합니다. 좌표 수집·저장은 가능하며, 경계 파일을 추가하면 경계와 탐색 격자가 표시됩니다.')
else:
    try:
        boundary, grid = boundary_layers(raw_boundary)
        folium.GeoJson(boundary, name='영천시 경계',
                       style_function=lambda _: {'fillOpacity': 0, 'color': 'red', 'weight': 2}).add_to(m)
        folium.GeoJson(grid, name='탐색 격자 (0.0045도)',
                       style_function=lambda _: {'fillOpacity': 0, 'color': 'gray', 'weight': 1, 'opacity': .5}).add_to(m)
    except (ValueError, KeyError, TypeError) as exc:
        st.warning(f'경계를 표시할 수 없어 기본 지도를 사용합니다: {exc}')

for lat, lon in st.session_state.coords:
    folium.CircleMarker(location=[lat, lon], radius=6, color='blue', fill=True).add_to(m)
folium.LayerControl().add_to(m)
m.add_child(folium.LatLngPopup())
map_data = st_folium(m, key='crosswalk_map', use_container_width=True, height=700,
                    returned_objects=['last_clicked', 'center', 'zoom'])
if map_data:
    if map_data.get('center'):
        center = map_data['center']
        st.session_state.last_center = [center['lat'], center['lng']]
    if map_data.get('zoom') is not None:
        st.session_state.zoom = map_data['zoom']
    click = map_data.get('last_clicked')
    if click:
        point = (click['lat'], click['lng'])
        if point != st.session_state.last_click:
            st.session_state.last_click = point
            st.session_state.new_point = point

if st.session_state.new_point is not None:
    point = st.session_state.new_point
    st.success(f'📍 {point[0]:.6f}, {point[1]:.6f}')
    if point in st.session_state.coords:
        st.info('이미 저장된 좌표입니다.')
    elif st.button('✅ 이 좌표 저장'):
        # Re-read saved coordinates so another session's additions are retained.
        try:
            existing = load_coords(SAVE_PATH) if SAVE_PATH.exists() else st.session_state.coords
            coords = existing if point in existing else existing + [point]
            save_coords(coords)
        except (OSError, ValueError, KeyError) as exc:
            st.error(f'저장하지 못했습니다. 다시 시도해 주세요: {exc}')
        else:
            st.session_state.coords = coords
            st.session_state.new_point = None
            st.rerun()

st.subheader(f'📋 저장된 좌표 목록 ({len(st.session_state.coords)}개)')
frame = pd.DataFrame(st.session_state.coords, columns=['위도', '경도'])
st.dataframe(frame, use_container_width=True)
st.download_button('📥 CSV 다운로드', frame.to_csv(index=False).encode('utf-8-sig'),
                   '횡단보도_좌표.csv', 'text/csv')
