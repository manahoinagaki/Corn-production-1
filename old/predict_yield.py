import sqlite3
import pandas as pd
import statsmodels.api as sm

def run_regression():
    # 1. データベースからデータを読み込む
    conn = sqlite3.connect("agri_data.db")
    df = pd.read_sql("SELECT * FROM annual_summary", conn)
    conn.close()

    # デバッグ用：現在保存されている列名を表示
    print(f"📊 現在のデータベースの列名: {list(df.columns)}")

    # 2. データのクリーニング（ここで df_clean を定義します）
    mapping = {
        'Yield': ['Yield', 'USDA_Yield_bu_acre'],
        'NDVI': ['Max_NDVI'],
        'RVI': ['Max_RVI'],
        'Temp': ['Max_Temp', 'Max_Temp_C']
    }
    
    final_cols = {}
    # データベースに列があるか確認
    for key, candidates in mapping.items():
        found = False
        for c in candidates:
            if c in df.columns:
                final_cols[key] = c
                found = True
                break
        if not found:
            print(f"❌ エラー: '{key}' に相当する列が見つかりません。候補: {candidates}")
            return
        
    # 3. データの抽出とクリーニング
    required_cols = list(final_cols.values())
    df_clean = df.dropna(subset=required_cols).copy()

    print(f"✅ 分析に使用する有効データ件数: {len(df_clean)} 件")

    if len(df_clean) < 4:
        print(f"⚠️ データ不足です（最低4件必要）。現在 {len(df_clean)} 件。")
        return
        return

    # 4. 重回帰分析の実行
    # 目的変数 y (予測したいもの)
    y = df_clean[final_cols['Yield']]
    # 説明変数 X (予測の材料)
    X = df_clean[[final_cols['NDVI'], final_cols['RVI'], final_cols['Temp']]]
    X = sm.add_constant(X)  # 統計計算に必要な「切片」を追加

    # モデルの構築と学習
    model = sm.OLS(y, X).fit()

    # 5. 結果の表示
    print("\n" + "="*60)
    print("           🌽 トウモロコシ収穫量 予測モデル結果")
    print("="*60)
    print(model.summary())

if __name__ == "__main__":
    run_regression()