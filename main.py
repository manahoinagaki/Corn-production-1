import ee
import pandas as pd
import matplotlib.pyplot as plt
import sys
import requests

# --- 設定：プロジェクトIDとAPIキー ---
project_name = ''
USDA_API_KEY = '' 

try:
    ee.Initialize(project=project_name)
except Exception as e:
    ee.Authenticate()
    ee.Initialize(project=project_name)

# ==========================================
# 設定：地点と期間
# ==========================================
lon, lat = -100.55, 41.14
roi_point = ee.Geometry.Point([lon, lat])
roi_area = ee.Geometry.Rectangle([lon - 0.01, lat - 0.01, lon + 0.01, lat + 0.01])

START_DATE = '2015-04-01'
END_DATE = '2024-10-01'

# ==========================================
# 関数定義
# ==========================================
def get_location_info(geometry):
    try:
        counties = ee.FeatureCollection("TIGER/2018/Counties")
        county_feature = counties.filterBounds(geometry).first()
        name = county_feature.get('NAME').getInfo().upper()
        return name, "NEBRASKA"
    except:
        return "LINCOLN", "NEBRASKA"

def fetch_usda_yield_series(state, county, start_year, end_year):
    yield_list = []
    base_url = "https://quickstats.nass.usda.gov/api/api_GET/"
    print(f"USDA統計（{start_year}-{end_year}）を取得中...")
    for year in range(start_year, end_year + 1):
        params = {
            "key": USDA_API_KEY, "commodity_desc": "CORN", "year": year,
            "state_name": state, "county_name": county,
            "statisticcat_desc": "YIELD", "unit_desc": "BU / ACRE", "format": "JSON"
        }
        try:
            res = requests.get(base_url, params=params, timeout=10)
            if res.status_code == 200:
                val = res.json()['data'][0]['Value']
                yield_list.append({'date': pd.to_datetime(f'{year}-09-01'), 'yield_val': float(val.replace(',', ''))})
        except: continue
    return pd.DataFrame(yield_list)

def extract_data(collection, geometry, scale=30):
    valid = collection.filter(ee.Filter.notNull(['system:time_start']))
    def ex(img):
        stats = img.reduceRegion(reducer=ee.Reducer.mean(), geometry=geometry, scale=scale)
        return ee.Feature(None).set('date', img.date().format('YYYY-MM-dd')).set(stats)
    info = valid.map(ex).getInfo()
    return pd.DataFrame([f['properties'] for f in info['features'] if 'date' in f['properties']])

# ==========================================
# データ収集
# ==========================================
county_name, state_name = get_location_info(roi_point)
df_yield_stats = fetch_usda_yield_series(state_name, county_name, 2015, 2023)

print("衛星・気象データを抽出中...")
df_s2 = extract_data(ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').filterBounds(roi_area).filterDate(START_DATE, END_DATE).filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 50)).map(lambda img: img.addBands(img.normalizedDifference(['B8', 'B4']).rename('NDVI'))).select(['NDVI']), roi_area)
df_s1 = extract_data(ee.ImageCollection('COPERNICUS/S1_GRD').filterBounds(roi_area).filterDate(START_DATE, END_DATE).filter(ee.Filter.eq('instrumentMode', 'IW')).select(['VV', 'VH']), roi_area)
df_wx = extract_data(ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR").filterBounds(roi_area).filterDate(START_DATE, END_DATE).map(lambda img: img.addBands(img.select('temperature_2m').subtract(273.15).rename('Temp_C')).addBands(img.select('total_precipitation_sum').multiply(1000).rename('Precip_mm')).select(['Temp_C', 'Precip_mm'])), roi_area, scale=1000)

# ==========================================
# データ統合
# ==========================================
def clean(df):
    if df.empty: return df
    df['date'] = pd.to_datetime(df['date'])
    return df.groupby('date').mean().sort_index()

merged = pd.concat([clean(df_s2), clean(df_s1), clean(df_wx)], axis=1).sort_index()
# 指標の計算
if 'VH' in merged.columns and 'VV' in merged.columns:
    merged['RVI'] = 4 * merged['VH'] / (merged['VV'] + merged['VH'])

# 線形補間
merged[['NDVI', 'VH', 'VV', 'Temp_C', 'RVI']] = merged[['NDVI', 'VH', 'VV', 'Temp_C', 'RVI']].interpolate(method='linear')
merged['Precip_mm'] = merged['Precip_mm'].fillna(0)

## ==========================================
# 月次集計 & 年次Max & USDA収穫量の統合
# ==========================================
print("月次・年次統計（収穫量統合版）を計算中...")

# 1. 月次レポートの作成
df_monthly = merged.resample('MS').agg({
    'Temp_C': 'mean',
    'Precip_mm': 'sum',
    'NDVI': 'mean',
    'RVI': 'mean',
    'VH': 'mean'
})

# 2. 年次最大値とUSDA実績の統合
annual_summary_list = []
years = merged.index.year.unique()

for year in years:
    year_data = merged[merged.index.year == year]
    if not year_data.empty:
        # 最大値の特定
        max_ndvi = year_data['NDVI'].max()
        max_ndvi_date = year_data['NDVI'].idxmax()
        max_temp = year_data['Temp_C'].max()
        max_temp_date = year_data['Temp_C'].idxmax()
        
        # この年のUSDA収穫量をdf_yield_statsから探す
        # (df_yield_statsのdateから年を抽出して照合)
        year_yield = None
        if not df_yield_stats.empty:
            match = df_yield_stats[df_yield_stats['date'].dt.year == year]
            if not match.empty:
                year_yield = match['yield_val'].values[0]
        
        annual_summary_list.append({
            'Year': year,
            'USDA_Yield_bu_acre': year_yield, # ★ここに追加
            'Max_NDVI': max_ndvi,
            'Max_NDVI_Date': max_ndvi_date.strftime('%Y-%m-%d'),
            'Max_Temp_C': max_temp,
            'Max_Temp_Date': max_temp_date.strftime('%Y-%m-%d')
        })

df_annual_summary = pd.DataFrame(annual_summary_list)

# CSV保存（名前を summary に変更して分かりやすくしました）
df_monthly.to_csv('monthly_report.csv', encoding='utf-8-sig')
df_annual_summary.to_csv('annual_summary_report.csv', index=False, encoding='utf-8-sig')

print("CSVファイルを保存しました:")
print("- monthly_report.csv (月次平均/合計)")
print("- annual_summary_report.csv (年間の最大値とUSDA収穫量)")

# ターミナルにも結果を少し表示
print("\n--- 年次サマリー (抜粋) ---")
print(df_annual_summary[['Year', 'USDA_Yield_bu_acre', 'Max_NDVI']].to_string(index=False))

# ==========================================
# 可視化：気象統合4段ダッシュボード
# ==========================================
fig, axes = plt.subplots(4, 1, figsize=(12, 14), sharex=True)

# 1. NDVI + USDA Yield
axes[0].plot(merged.index, merged['NDVI'], color='green', alpha=0.6, label='NDVI')
if not df_yield_stats.empty:
    ax0_y = axes[0].twinx()
    ax0_y.bar(df_yield_stats['date'], df_yield_stats['yield_val'], width=30, color='red', alpha=0.2, label='USDA Yield')
    ax0_y.set_ylabel('Yield [bu/acre]', color='red')
axes[0].set_title(f'Combined Agricultural Dashboard: {county_name} Co., {state_name}', fontweight='bold')
axes[0].set_ylabel('NDVI Index')

# 2. VH (Volume)
axes[1].plot(merged.index, merged['VH'], color='blue', label='VH (Structure)')
axes[1].set_ylabel('VH [dB]')

# 3. RVI (Radar Index)
if 'RVI' in merged.columns:
    axes[2].plot(merged.index, merged['RVI'], color='purple', label='RVI')
axes[2].set_ylabel('RVI')

# 4. Weather (Temp + Precip)
axes[3].plot(merged.index, merged['Temp_C'], color='orange', label='Temp', alpha=0.8)
axes[3].set_ylabel('Temp [°C]', color='orange')
axes[3].tick_params(axis='y', labelcolor='orange')

ax4_right = axes[3].twinx()
ax4_right.bar(merged.index, merged['Precip_mm'], color='skyblue', alpha=0.4, label='Precip', width=2.0)
ax4_right.set_ylabel('Precip [mm]', color='skyblue')
ax4_right.tick_params(axis='y', labelcolor='skyblue')
axes[3].set_title('Weather: Temperature & Precipitation')

for i, ax in enumerate(axes):
    ax.grid(True, alpha=0.2)
    ax.legend(loc='upper left', fontsize=9)

plt.tight_layout()
plt.show()