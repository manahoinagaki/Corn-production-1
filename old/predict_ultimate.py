import sqlite3
import pandas as pd
import statsmodels.api as sm

def run_ultimate_analysis():
    conn = sqlite3.connect("agri_data.db")
    df = pd.read_sql("SELECT * FROM annual_summary", conn)
    conn.close()

    # 水、熱、緑、ボリュームの4拍子で予測
    required_cols = ['Yield', 'July_NDVI', 'July_RVI', 'July_Precip', 'July_GDD']
    df_clean = df.dropna(subset=required_cols).copy()

    # 異常値の除外（7月に全くデータがない年などを省く）
    df_clean = df_clean[df_clean['July_NDVI'] > 0]

    y = df_clean['Yield']
    # 複数の視点から収穫量を説明する
    X = df_clean[['July_NDVI', 'July_RVI', 'July_Precip', 'July_GDD']]
    X = sm.add_constant(X)

    model = sm.OLS(y, X).fit()

    print("\n" + "="*60)
    print("      🚀 【究極版】トウモロコシ収穫量 予測モデル")
    print("="*60)
    print(model.summary())
    print("="*60)

if __name__ == "__main__":
    run_ultimate_analysis()