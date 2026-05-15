import sqlite3
import pandas as pd
import statsmodels.api as sm

def run_synergy_analysis():
    conn = sqlite3.connect("agri_data.db")
    df = pd.read_sql("SELECT * FROM annual_summary", conn)
    conn.close()

    required_cols = ['Yield', 'July_NDVI', 'July_RVI', 'July_GDD']
    df_clean = df.dropna(subset=required_cols).copy()
    df_clean = df_clean[df_clean['July_NDVI'] > 0]

    # --- RVIを活かすための工夫 ---
    # 1. NDVIとRVIの「相乗効果（Fusion）」を新しく作成
    df_clean['NDVI_RVI_Interaction'] = df_clean['July_NDVI'] * df_clean['July_RVI']
    
    # 2. 成長ストレス（GDD）
    y = df_clean['Yield']
    # 個別の指標だけでなく「相乗効果」を混ぜることで、RVIの重要性をモデルに再認識させます
    X = df_clean[['July_NDVI', 'July_RVI', 'NDVI_RVI_Interaction', 'July_GDD']]
    X = sm.add_constant(X)

    model = sm.OLS(y, X).fit()

    print("\n" + "="*60)
    print("      🌟 【S1+S2相乗効果版】トウモロコシ収穫量モデル")
    print("="*60)
    print(model.summary())
    print("="*60)

if __name__ == "__main__":
    run_synergy_analysis()