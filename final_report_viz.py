import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import numpy as np

def create_pro_report():
    # 1. データ読み込み
    try:
        conn = sqlite3.connect("agri_data.db")
        df = pd.read_sql("SELECT * FROM annual_summary", conn)
        conn.close()
    except Exception as e:
        print(f"❌ データベース読み込みエラー: {e}")
        return

    # 必要な列の抽出
    required_cols = ['Yield', 'July_NDVI', 'July_RVI', 'July_GDD']
    df_clean = df.dropna(subset=required_cols).copy()
    
    if len(df_clean) < 5:
        print("⚠️ 分析に必要なデータ数が足りません。")
        return

    # 2. モデル再構築（標準化して比較しやすくする）
    y = df_clean['Yield']
    features = df_clean[['July_NDVI', 'July_RVI', 'July_GDD']]
    
    # 標準化（平均0, 分散1）: これにより「どの指標が一番強いか」が比較可能になります
    features_std = (features - features.mean()) / features.std()
    
    # 修正ポイント: 引数名を指定せず、直接渡す
    X = sm.add_constant(features_std) 
    model = sm.OLS(y, X).fit()

    # 3. 可視化のレイアウト設定
    fig = plt.figure(figsize=(15, 12))
    gs = fig.add_gridspec(2, 2)
    fig.suptitle('🌽 Corn Production Intelligence Report', fontsize=24, fontweight='bold', y=0.96)

    # --- 左上: 予測精度グラフ ---
    ax1 = fig.add_subplot(gs[0, 0])
    predicted = model.predict(X)
    sns.regplot(x=y, y=predicted, ax=ax1, scatter_kws={'alpha':0.6, 's':80}, line_kws={'color':'red', 'ls':'--'})
    ax1.set_title('Prediction Accuracy (Reality vs Model)', fontsize=15, pad=10)
    ax1.set_xlabel('Actual Yield (USDA)')
    ax1.set_ylabel('Predicted Yield')
    ax1.annotate(f'R-squared: {model.rsquared:.3f}', xy=(0.05, 0.9), xycoords='axes fraction', 
                 fontsize=13, fontweight='bold', bbox=dict(boxstyle="round", fc="w", ec="0.5"))

    # --- 右上: 指標の影響力 ---
    ax2 = fig.add_subplot(gs[0, 1])
    # 係数の比較
    coef_df = pd.DataFrame({
        'Feature': ['Greenness (NDVI)', 'Volume (RVI)', 'Heat Stress (GDD)'],
        'Impact': model.params[1:]
    })
    # インパクトの強さで色分け
    colors = ['#2ecc71' if x > 0 else '#e74c3c' for x in coef_df['Impact']]
    sns.barplot(x='Impact', y='Feature', data=coef_df, palette=colors, ax=ax2, hue='Feature', legend=False)
    ax2.set_title('Feature Importance (Impact on Yield)', fontsize=15, pad=10)
    ax2.axvline(0, color='black', lw=1.5)
    ax2.set_xlabel('Standardized Coefficient (Relative Impact)')

    # --- 下段: 年次・地点別の予測エラー（残差） ---
    ax3 = fig.add_subplot(gs[1, :])
    df_clean['Error'] = predicted - y
    df_clean['Year_Loc'] = df_clean['Year'].astype(str) + " (" + df_clean['Location'].str[:5] + ")"
    
    # エラーの大きさを可視化
    sns.barplot(x='Year_Loc', y='Error', data=df_clean, ax=ax3, palette='vlag', hue='Year_Loc', legend=False)
    ax3.set_title('Prediction Error analysis (Positive = Over-predicted / Negative = Under-predicted)', fontsize=15, pad=10)
    ax3.set_ylabel('Yield Error (bu/acre)')
    ax3.axhline(0, color='black', lw=1)
    plt.xticks(rotation=45, ha='right')

    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    
    # 画像として保存
    report_file = 'corn_intelligence_report.png'
    plt.savefig(report_file, dpi=300)
    print(f"✅ レポートを保存しました: {report_file}")
    plt.show()

if __name__ == "__main__":
    create_pro_report()