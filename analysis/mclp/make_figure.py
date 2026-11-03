"""MCLP selection and a landscape blog figure from supplied local data."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import cKDTree, distance
from scipy.optimize import milp, Bounds, LinearConstraint
from scipy.sparse import lil_matrix
import folium
from branca.element import Element
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).parent/'output'; OUT.mkdir(exist_ok=True)
def points(d,x,y):
 return gpd.GeoDataFrame(d.copy(),geometry=gpd.points_from_xy(d[x],d[y]),crs=4326).to_crs(5179)
def xy(g): return np.column_stack([g.geometry.x,g.geometry.y])
def norm(v): return (v-v.min())/(np.ptp(v) or 1)
city=gpd.read_file(ROOT/'data/행정동경계/bnd_sigungu_37070_2025_2Q.zip').to_crs(5179)
c=points(pd.read_csv(ROOT/'data/영천시/영천시_횡단보도_좌표.csv').drop_duplicates(),'경도','위도')
c=c[c.intersects(city.geometry.union_all())].reset_index(drop=True); X=xy(c)
s=points(pd.read_csv(ROOT/'data/그늘막/영천시_그늘막.csv').drop_duplicates(['위도','경도']),'경도','위도'); S=xy(s)
p=points(pd.read_csv(ROOT/'analysis/kde/output/included_poi.csv'),'x','y')
w=points(pd.read_csv(ROOT/'data/영천시/영천시_WBGT.csv'),'lon','lat')
z=np.load(ROOT/'analysis/kde/output/kde_grid.npz')
ix=np.clip(np.searchsorted(z['x'],X[:,0]),1,len(z['x'])-1);iy=np.clip(np.searchsorted(z['y'],X[:,1]),1,len(z['y'])-1)
ix-=abs(z['x'][ix-1]-X[:,0])<abs(z['x'][ix]-X[:,0]);iy-=abs(z['y'][iy-1]-X[:,1])<abs(z['y'][iy]-X[:,1])
c['kde']=np.nan_to_num(z['score'][iy,ix]);c['poi_count']=cKDTree(xy(p)).query_ball_point(X,150,return_length=True)
c['wbgt']=w.WBGT.to_numpy()[cKDTree(xy(w)).query(X)[1]]
e=gpd.read_file('zip://'+str(next((ROOT/'data/고령인구/격자').glob('*고령*zip'))),encoding='utf-8').to_crs(5179)
joined=gpd.sjoin(c[['geometry']],e[['val','geometry']],how='left',predicate='intersects')
c['elderly']=joined.groupby(level=0).val.max().reindex(c.index).fillna(0)
c['score']=.3*norm(c.kde.to_numpy())+.3*norm(c.poi_count.to_numpy())+.3*norm(c.wbgt.to_numpy())+.1*norm(c.elderly.to_numpy())
demand_idx=c.nlargest(int(np.ceil(len(c)*.3)),'score').index.to_numpy(); D=X[demand_idx];weights=c.loc[demand_idx,'score'].to_numpy()
existing=distance.cdist(D,S).min(axis=1)<=200
eligible=np.flatnonzero(distance.cdist(X,S).min(axis=1)>=100); C=X[eligible];A=distance.cdist(D,C)<=200
n=len(C);m=len(D);pairs=np.argwhere(np.triu(distance.cdist(C,C)<100,1));B=lil_matrix((m+1+len(pairs),n+m));upper=np.ones(B.shape[0]);lower=np.full(B.shape[0],-np.inf)
for i in range(m): B[i,:n]=-A[i].astype(float); B[i,n+i]=1;upper[i]=int(existing[i])
B[m,:n]=1;lower[m]=upper[m]=36
for r,(a,b) in enumerate(pairs,m+1):B[r,a]=B[r,b]=1
res=milp(np.r_[np.zeros(n),-weights],integrality=np.ones(n+m),bounds=Bounds(0,1),constraints=LinearConstraint(B.tocsr(),lower,upper),options={'time_limit':90,'mip_rel_gap':.001})
if res.x is None:raise RuntimeError(res.message)
selected=eligible[np.flatnonzero(res.x[:n]>.5)]; covered=existing|A[:,res.x[:n]>.5].any(axis=1)
assert len(selected)==36
assert distance.cdist(X[selected],S).min()>=100-1e-6
sd=distance.cdist(X[selected],X[selected]);np.fill_diagonal(sd,np.inf);assert sd.min()>=100-1e-6
c['selected']=False;c.loc[selected,'selected']=True
c.drop(columns='geometry').to_csv(OUT/'candidate_scores.csv',index=False,encoding='utf-8-sig')
c.loc[selected].drop(columns='geometry').to_csv(OUT/'selected_sites.csv',index=False,encoding='utf-8-sig')
summary={'new_sites':len(selected),'existing_records':42,'existing_unique_coordinates':len(s),'candidates':len(c),'demand_points':m,'covered_before':int(existing.sum()),'covered_after':int(covered.sum()),'coverage_radius_m':200,'minimum_spacing_m':100,'solver_status':int(res.status),'mip_gap':float(res.mip_gap),'weights':{'kde':.3,'poi_count_150m':.3,'wbgt':.3,'elderly':.1},'missing_elderly_policy':'zero for this illustration','budget':36}
(OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)); print(summary,flush=True)
ll=c.to_crs(4326);sl=s.to_crs(4326);cl=city.to_crs(4326)
def panel(detail):
 mp=folium.Map(tiles=None,zoom_control=False,zoom_snap=.1,prefer_canvas=True,control_scale=True)
 folium.TileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png',attr='© OpenStreetMap contributors').add_to(mp)
 folium.GeoJson(json.loads(cl.to_json()),style_function=lambda _:dict(color='#8c9699',weight=1.3,fillOpacity=0)).add_to(mp)
 if detail:
  for i in selected:folium.Circle([ll.loc[i].geometry.y,ll.loc[i].geometry.x],radius=200,color='#db8a37',weight=.7,opacity=.35,fill=True,fill_color='#efb45c',fill_opacity=.07).add_to(mp)
  for j,i in enumerate(demand_idx):folium.CircleMarker([ll.loc[i].geometry.y,ll.loc[i].geometry.x],radius=2.2,weight=0,fill=True,fill_color='#73a38e' if covered[j] else '#aeb5b6',fill_opacity=.48 if covered[j] else .35).add_to(mp)
 for g in sl.geometry:folium.CircleMarker([g.y,g.x],radius=4 if detail else 2.5,color='white',weight=1.4,fill=True,fill_color='#345c73',fill_opacity=.95).add_to(mp)
 for i in selected:
  g=ll.loc[i].geometry;folium.CircleMarker([g.y,g.x],radius=6 if detail else 3.5,color='white',weight=1.8,fill=True,fill_color='#e37827',fill_opacity=1).add_to(mp)
 b=np.array([128.913,35.947,128.966,35.995]) if detail else cl.total_bounds
 mp.fit_bounds([[b[1],b[0]],[b[3],b[2]]],padding=(45,45))
 mp.get_root().header.add_child(Element('<style>.leaflet-container{background:#faf8f2!important}.leaflet-tile-pane{filter:grayscale(1) sepia(.14);opacity:.35}.leaflet-control-scale-line:nth-child(2){display:none}.leaflet-control-attribution{font-size:9px!important}.legend{position:absolute;top:24px;right:24px;z-index:1000;background:rgba(255,255,255,.94);border:1px solid #e5e5df;border-radius:9px;padding:14px 19px;font:15px/2 sans-serif;color:#35434a}.dot{display:inline-block;border-radius:50%;margin-right:10px;vertical-align:middle}</style>'))
 mp.save(OUT/('detail.html' if detail else 'overview.html'))
panel(False);panel(True)
(OUT/'map.html').write_text('<html><style>html,body{margin:0;width:100%;height:100%;background:#fff;overflow:hidden}body{display:grid;grid-template-columns:1fr}iframe{border:0;width:100%;height:100%}</style><body><iframe src="detail.html"></iframe></body></html>')
from playwright.sync_api import sync_playwright
with sync_playwright() as p:
 browser=p.chromium.launch(headless=True);page=browser.new_page(viewport={'width':1800,'height':950},device_scale_factor=2);page.goto((OUT/'map.html').as_uri())
 for f in page.frames[1:]:
  f.wait_for_function('typeof L !== "undefined" && document.querySelectorAll(".leaflet-tile-loaded").length > 0',timeout=60000)
  f.wait_for_function('Array.from(document.querySelectorAll(".leaflet-tile")).every(i=>i.complete && i.naturalWidth>0)',timeout=60000)
 page.wait_for_timeout(1500);page.screenshot(path=str(OUT/'yeongcheon_mclp.png'));browser.close()
