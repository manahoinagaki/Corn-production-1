import sqlite3
import pandas as pd
import statsmodels.api as sm

def run_regression():
    # 1. データベースからデータを読み込む
    conn = sqlite3.connect("agri_data.db")
    df = pd.read_sql("SELECT * FROM annual_summary", conn)
    conn.close()

    # 2. データのクリーニング（ここで df_clean を定義します）
    # 分析に使う4つの列に「None」が含まれている行をすべて削除します
    required_cols = ['USDA_Yield_bu_acre', 'Max_NDVI', 'Max_RVI', 'Max_Temp_C']
    
    # データベースに列があるか確認
    for col in required_cols:
        if col not in df.columns:
            print(f"❌ エラー: 列 '{col}' がデータベースに見つかりません。")
            print("💡 main.py を実行して最新のデータをデータベースに保存してください。")
            return

    df_clean = df.dropna(subset=required_cols).copy()

    # 3. データ件数のチェック
    if len(df_clean) < 4:
        print(f"⚠️ データ不足: 現在 {len(df_clean)} 年分しかありません。")
        print("💡 統計分析には最低でも4〜5年分以上のデータが必要です。")
        return

    # 4. 重回帰分析の実行
    # 目的変数 y (予測したいもの)
    y = df_clean['USDA_Yield_bu_acre']
    # 説明変数 X (予測の材料)
    X = df_clean[['Max_NDVI', 'Max_RVI', 'Max_Temp_C']]
    X = sm.add_constant(X)  # 統計計算に必要な「切片」を追加

    # モデルの構築と学習
    model = sm.OLS(y, X).fit()

    # 5. 結果の表示
    print(model.summary())

if __name__ == "__main__":
    run_regression()