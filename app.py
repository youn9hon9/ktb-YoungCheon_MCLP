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
    frame = pd.read_csv(path, float_precision='round_trip')
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

st.session_state.setdefault('new_point', None)
st.session_state.setdefault('last_click', None)
st.session_state.setdefault('undo_record', None)


def select_point():
    click = st.session_state.get('crosswalk_map', {}).get('last_clicked')
    if click:
        point = (click['lat'], click['lng'])
        if point != st.session_state.last_click:
            st.session_state.last_click = point
            st.session_state.new_point = point

raw_boundary = BOUNDARY_PATH.read_bytes() if BOUNDARY_PATH.exists() else None

# Keep the base script stable: st_folium hashes it into the component identity.
# Browser pan/zoom stays local; saved markers update through a dynamic layer.
m = folium.Map(location=[35.9749, 128.9461], zoom_start=16, prefer_canvas=True)
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

markers = folium.FeatureGroup(name='저장된 좌표')
for lat, lon in st.session_state.coords:
    folium.CircleMarker(location=[lat, lon], radius=6, color='blue', fill=True).add_to(markers)
if st.session_state.new_point is not None:
    folium.CircleMarker(location=st.session_state.new_point, radius=9,
                        color='#f97316', fill=True, fill_opacity=.9,
                        tooltip='선택한 위치').add_to(markers)
folium.LayerControl().add_to(m)
m.add_child(folium.LatLngPopup())
map_column, action_column = st.columns([3, 1], gap='medium')
with map_column:
    map_data = st_folium(m, key='crosswalk_map', use_container_width=True, height=700,
                        returned_objects=['last_clicked'], feature_group_to_add=markers,
                        on_change=select_point)
if map_data:
    # Also consume the returned value when restoring component state.
    click = map_data.get('last_clicked')
    if click:
        point = (click['lat'], click['lng'])
        if point != st.session_state.last_click:
            st.session_state.last_click = point
            st.session_state.new_point = point

with action_column:
    st.subheader('선택한 좌표')
    point = st.session_state.new_point
    duplicate = point is not None and point in st.session_state.coords
    st.markdown(f"위도 **{point[0]:.6f}**" if point else '위도 —')
    st.markdown(f"경도 **{point[1]:.6f}**" if point else '경도 —')
    st.caption('이미 저장된 좌표입니다.' if duplicate else
               '주황색 위치를 확인한 뒤 저장하세요.' if point else '지도에서 저장할 위치를 클릭하세요.')
    if st.button('이 좌표 저장', key='save_point', type='primary',
                 use_container_width=True, disabled=point is None or duplicate):
        try:
            existing = load_coords(SAVE_PATH) if SAVE_PATH.exists() else st.session_state.coords
            added = point not in existing
            coords = existing + [point] if added else existing
            save_coords(coords)
        except (OSError, ValueError, KeyError) as exc:
            st.error(f'저장하지 못했습니다. 다시 시도해 주세요: {exc}')
        else:
            st.session_state.coords = coords
            if added:
                st.session_state.undo_record = (len(existing), point)
            st.session_state.new_point = None
            st.session_state.notice = '좌표를 저장했습니다.' if added else '이미 저장된 좌표입니다.'
            st.rerun()

    if st.button('마지막 저장 취소', key='undo_point', use_container_width=True,
                 disabled=st.session_state.undo_record is None):
        index, saved_point = st.session_state.undo_record
        try:
            existing = load_coords(SAVE_PATH)
            if index >= len(existing) or existing[index] != saved_point:
                raise ValueError('저장 파일이 변경되어 취소할 좌표를 확인할 수 없습니다.')
            coords = existing[:index] + existing[index + 1:]
            save_coords(coords)
        except (OSError, ValueError, KeyError) as exc:
            st.error(f'취소하지 못했습니다: {exc}')
        else:
            st.session_state.coords = coords
            st.session_state.undo_record = None
            st.session_state.new_point = saved_point
            st.session_state.notice = '마지막 저장을 취소했습니다.'
            st.rerun()
    if notice := st.session_state.pop('notice', None):
        st.toast(notice)

    st.divider()
    st.subheader('최근 저장 목록')
    st.caption(f'전체 {len(st.session_state.coords):,}개 · 최근 5개')
    frame = pd.DataFrame(st.session_state.coords, columns=['위도', '경도'])
    st.dataframe(frame.tail(5).iloc[::-1], hide_index=True, use_container_width=True,
                 height=210, column_config={
                     '위도': st.column_config.NumberColumn(format='%.6f'),
                     '경도': st.column_config.NumberColumn(format='%.6f'),
                 })
    st.download_button('CSV 다운로드', frame.to_csv(index=False).encode('utf-8-sig'),
                       '횡단보도_좌표.csv', 'text/csv', use_container_width=True)
    st.caption('저장 취소는 이 세션에서 마지막으로 추가한 좌표에 적용됩니다.')

with st.expander(f'전체 좌표 목록 ({len(frame):,}개)', expanded=False):
    st.dataframe(frame, use_container_width=True, hide_index=True)
