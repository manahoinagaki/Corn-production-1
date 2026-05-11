import sqlite3
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def run_fusion_analysis():
    # 1. データベース読み込み
    conn = sqlite3.connect("agri_data.db")
    df = pd.read_sql("SELECT * FROM annual_summary", conn)
    conn.close()

    # 実績データがある年のみ抽出
    df_clean = df.dropna(subset=['USDA_Yield_bu_acre']).copy()

    # 2. 統合指標（S1 + S2）の作成
    # 光学(NDVI)とレーダー(RVI)を掛け合わせた「ハイブリッド指標」を作ってみる
    # これが生育の「質」と「量」を同時に表す簡易的なモデルになります
    df_clean['S1_S2_Fusion'] = df_clean['Max_NDVI'] * df_clean['Max_RVI']

    # 3. 可視化
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # (A) Sentinel-1: RVI vs Yield
    sns.regplot(x='Max_RVI', y='USDA_Yield_bu_acre', data=df_clean, ax=axes[0], color='blue')
    axes[0].set_title('Sentinel-1 (Radar): RVI vs Yield')
    axes[0].set_xlabel('Radar Vegetation Index (Volume)')

    # (B) S1 + S2 Fusion指標 vs Yield
    sns.regplot(x='S1_S2_Fusion', y='USDA_Yield_bu_acre', data=df_clean, ax=axes[1], color='purple')
    axes[1].set_title('Fusion (S1*S2): Quality * Volume vs Yield')
    axes[1].set_xlabel('Fusion Index (NDVI * RVI)')

    # (C) 統合相関行列（ヒートマップ）
    cols = ['USDA_Yield_bu_acre', 'Max_NDVI', 'Max_RVI', 'Max_Temp_C', 'S1_S2_Fusion']
    corr = df_clean[cols].corr()
    sns.heatmap(corr, annot=True, cmap='RdYlGn', center=0, ax=axes[2])
    axes[2].set_title('Integrated Correlation Matrix')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # seabornが入っていない場合は pip install seaborn してください
    run_fusion_analysis()