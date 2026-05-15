import sqlite3
import pandas as pd
import statsmodels.api as sm

def run_july_analysis():
    conn = sqlite3.connect("agri_data.db")
    df = pd.read_sql("SELECT * FROM annual_summary", conn)
    conn.close()

    # 7月の指標を使って分析
    required_cols = ['Yield', 'July_NDVI', 'July_RVI', 'July_Temp']
    df_clean = df.dropna(subset=required_cols).copy()

    print(f"✅ 7月のデータを用いた分析対象: {len(df_clean)} 件")

    y = df_clean['Yield']
    X = df_clean[['July_NDVI', 'July_RVI', 'July_Temp']]
    X = sm.add_constant(X)

    model = sm.OLS(y, X).fit()

    print("\n" + "="*60)
    print("      🌽 【7月限定】トウモロコシ収穫量 予測モデル")
    print("="*60)
    print(model.summary())
    print("="*60)

if __name__ == "__main__":
    run_july_analysis()