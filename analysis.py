import sqlite3
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def run_advanced_analysis():
    conn = sqlite3.connect("agri_data.db")
    # raw_combined_dataにはS1とS2の両方が入っています
    # 年ごとの最大値・平均値として集計
    df = pd.read_sql("SELECT * FROM annual_summary", conn)
    
    # 別テーブルからS1の平均指標なども結合したい場合は、ここで処理
    # 今回は annual_summary に RVI の最大値などを追加するイメージで作成
    
    # 欠損値（None）がある行を削除
    df_clean = df.dropna(subset=['USDA_Yield_bu_acre']).copy()

    # グラフの作成 (3列構成にする)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Sentinel-2: NDVI (葉の緑さ)
    sns.regplot(x='Max_NDVI', y='USDA_Yield_bu_acre', data=df_clean, ax=axes[0], color='green')
    axes[0].set_title('Sentinel-2: NDVI vs Yield')

    # 2. Sentinel-1: ここをRVIやVHにしたい場合
    # もし annual_summary に Max_RVI がなければ、まずそれを作成する必要があります
    # 今回は一旦 Max_Temp で出ているので、ここを解析のポイントに書き換えます
    if 'Max_Temp_C' in df_clean.columns:
        sns.regplot(x='Max_Temp_C', y='USDA_Yield_bu_acre', data=df_clean, ax=axes[1], color='orange')
        axes[1].set_title('Weather: Temp vs Yield')

    # 3. 相関行列のヒートマップ（これが統合の証拠になります）
    # 解析したい列をピックアップ
    target_cols = ['USDA_Yield_bu_acre', 'Max_NDVI', 'Max_Temp_C']
    corr_matrix = df_clean[target_cols].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=axes[2])
    axes[2].set_title('Correlation Matrix')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_advanced_analysis()