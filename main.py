import ee
import pandas as pd
import matplotlib.pyplot as plt
import sqlite3  # データベース接続用
import requests
import os
import time


# ==========================================
# 設定
# ==========================================
# --- 設定：プロジェクトIDとAPIキー ---
project_name = 'my-project-1-rice'
USDA_API_KEY = '2FAEDCCD-2130-3902-9A67-DD087A9747FA' 
DB_NAME = "agri_data.db"

locations  = [
    (-100.55, 41.14), # 地点1 (既存)
    (-100.60, 41.15), # 地点2 (近隣)
    (-100.50, 41.10), # 地点3 (近隣)
    (-96.35, 40.85),  # 地点4 (ネブラスカ州別エリア)
    (-93.50, 42.00)   # 地点5 (アイオワ州: 最重要エリア)
]
# roi_point = ee.Geometry.Point([lon, lat])
# roi_area = ee.Geometry.Rectangle([lon - 0.01, lat - 0.01, lon + 0.01, lat + 0.01])

START_DATE = '2015-04-01'
END_DATE = '2024-10-01'
DB_NAME = "agri_data.db"  # 保存するデータベースファイル名

# ==========================================
# Earth Engine 初期化
# ==========================================
# Earth Engineの初期化
try:
    if not project_name:
        raise ValueError("project_name が空です。Google CloudのプロジェクトIDを入力してください。")
    ee.Initialize(project=project_name)
except Exception as e:
    print(f"EE初期化中: {e}")
    ee.Authenticate()
    ee.Initialize(project=project_name)

# ==========================================
# 関数定義
# ==========================================
def get_location_info(point):
    """郡名取得"""
    try:
        counties = ee.FeatureCollection("TIGER/2018/Counties")
        county = counties.filterBounds(point).first()
        name = county.get("NAME").getInfo().upper()
        return name, "NEBRASKA"
    except:
        return "UNKNOWN", "NEBRASKA"
    
def fetch_usda_yield(state, county, start_year, end_year):
    """USDA収穫量取得"""
    results = []
    url = "https://quickstats.nass.usda.gov/api/api_GET/"

    for year in range(start_year, end_year + 1):
        params = {
            "key": USDA_API_KEY,
            "commodity_desc": "CORN",
            "year": year,
            "state_name": state,
            "county_name": county,
            "statisticcat_desc": "YIELD",
            "unit_desc": "BU / ACRE",
            "format": "JSON"
        }

        try:
            r = requests.get(url, params=params, timeout=10)
            data = r.json()["data"]

            if data:
                val = float(data[0]["Value"].replace(",", ""))
                results.append({
                    "year": year,
                    "yield_val": val
                })
        except:
            continue

    return pd.DataFrame(results)
    

def extract_data(collection, geometry, scale=30):
    """Earth Engine データ抽出"""
    def reducer(img):
        stats = img.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=geometry,
            scale=scale
        )

        return ee.Feature(None).set(
            "date", img.date().format("YYYY-MM-dd")
        ).set(stats)

    info = collection.map(reducer).getInfo()

    return pd.DataFrame([
        f["properties"]
        for f in info["features"]
    ])

def clean(df):
    """日付整形"""
    if df.empty:
        return df

    df["date"] = pd.to_datetime(df["date"])
    return df.groupby("date").mean().sort_index()

def save_sql(df, table):
    conn = sqlite3.connect(DB_NAME)
    df.to_sql(table, conn, if_exists="replace", index=False)
    conn.close()

def plot_dashboard(df, county):
    """可視化"""
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    axes[0].plot(df.index, df["NDVI"])
    axes[0].set_title(f"{county} NDVI")

    axes[1].plot(df.index, df["RVI"])
    axes[1].set_title("RVI")

    axes[2].plot(df.index, df["Temp_C"])
    axes[2].bar(df.index, df["Precip_mm"], alpha=0.3)
    axes[2].set_title("Weather")

    plt.tight_layout()
    plt.show()

# def save_to_sqlite(df, table_name, db_name=DB_NAME):
#     """データフレームをSQLiteデータベースに保存する関数"""
#     try:
#         conn = sqlite3.connect(db_name)
#         df.to_sql(table_name, conn, if_exists='replace', index=True)
#         conn.close()
#         print(f"✅ Table '{table_name}' を {db_name} に保存しました。")
#     except Exception as e:
#         print(f"❌ データベース保存エラー ({table_name}): {e}")

# # 成長期（5月〜9月）の積算温度を計算
# def calculate_gdd(df):
#     growing_season = df[(df.index.month >= 5) & (df.index.month <= 9)]
#     # トウモロコシの基本温度 10度、上限 30度で計算
#     gdd = growing_season['Temp_C'].apply(lambda x: max(min(x, 30) - 10, 0))
#     return gdd.sum()

# ==========================================
# メイン
# ==========================================
all_results = []

for lon, lat in locations:

    print(f"\n解析中: {lon}, {lat}")

    point = ee.Geometry.Point([lon, lat])
    area = ee.Geometry.Rectangle([
        lon - 0.03,
        lat - 0.03,
        lon + 0.03,
        lat + 0.03
    ])

    county, state = get_location_info(point)

    print(f"場所: {county}")

    # USDA
    df_yield = fetch_usda_yield(state, county, 2015, 2023)

    # Sentinel-2
    s2 = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(area)
        .filterDate(START_DATE, END_DATE)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 50))
        .map(lambda img:
             img.addBands(
                 img.normalizedDifference(["B8", "B4"])
                 .rename("NDVI")
             ))
        .select(["NDVI"])
    )

    # Sentinel-1
    s1 = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(area)
        .filterDate(START_DATE, END_DATE)
        .select(["VV", "VH"])
    )

    # ERA5
    wx = (
        ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")
        .filterBounds(area)
        .filterDate(START_DATE, END_DATE)
        .map(lambda img:
             img.addBands(
                 img.select("temperature_2m")
                 .subtract(273.15)
                 .rename("Temp_C")
             ).addBands(
                 img.select("total_precipitation_sum")
                 .multiply(1000)
                 .rename("Precip_mm")
             ))
        .select(["Temp_C", "Precip_mm"])
    )

    df_s2 = extract_data(s2, area)
    df_s1 = extract_data(s1, area)
    df_wx = extract_data(wx, area, scale=1000)

    if df_s2.empty or df_s1.empty:
        print("データ不足")
        continue

    merged = pd.concat([
        clean(df_s2),
        clean(df_s1),
        clean(df_wx)
    ], axis=1)

    merged = merged.interpolate()
    merged["Precip_mm"] = merged["Precip_mm"].fillna(0)

    # RVI
    merged["RVI"] = 4 * merged["VH"] / (
        merged["VV"] + merged["VH"]
    )

    # 年次まとめ
    for year in merged.index.year.unique():

        data = merged[merged.index.year == year]

        # ★ 7月だけのデータを抽出
        july_data = data[data.index.month == 7]

        # 7月のデータがない場合は、その年の最大値で代用
        if not july_data.empty:
            july_ndvi = july_data['NDVI'].mean()
            july_rvi = july_data['RVI'].mean()
            july_precip = july_data['Precip_mm'].sum()
            july_gdd = july_data['Temp_C'].apply(lambda x: max(min(x, 30) - 10, 0)).sum()
        else:
            # データがない場合のデフォルト
            july_ndvi, july_rvi, july_precip, july_gdd = 0, 0, 0, 0


        y = None
        if not df_yield.empty:
            match = df_yield[df_yield["year"] == year]
            if not match.empty:
                y = match["yield_val"].iloc[0]

        all_results.append({
            "Location": f"{lon}_{lat}",
            "Year": year,
            "Yield": y,
            "July_NDVI": july_ndvi,  # 7月の平均を追加
            "July_RVI": july_rvi,    # 7月の平均を追加
            "July_Precip": july_precip, # 7月の合計降水量
            "July_GDD": july_gdd,  # 7月の有効積算温度 (10度〜30度の範囲で計算)
            "Max_NDVI": data["NDVI"].max(),
            "Max_RVI": data["RVI"].max(),
            "Max_Temp": data["Temp_C"].max()
        })

    # グラフ
    plot_dashboard(merged, county)

    time.sleep(2)

# ==========================================
# 保存
# ==========================================
final_df = pd.DataFrame(all_results)

save_sql(final_df, "annual_summary")
final_df.to_csv("annual_summary.csv", index=False)

print("完了")