import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

def plot_prediction():
    conn = sqlite3.connect("agri_data.db")
    df = pd.read_sql("SELECT * FROM annual_summary", conn)
    conn.close()

    df_clean = df.dropna(subset=['Yield', 'July_NDVI', 'July_RVI', 'July_GDD']).copy()
    df_clean['Interaction'] = df_clean['July_NDVI'] * df_clean['July_RVI']
    
    y = df_clean['Yield']
    X = df_clean[['July_NDVI', 'July_RVI', 'Interaction', 'July_GDD']]
    X = sm.add_constant(X)

    model = sm.OLS(y, X).fit()
    # モデルによる予測値を計算
    df_clean['Predicted_Yield'] = model.predict(X)

    # 可視化：実績 vs 予測
    plt.figure(figsize=(10, 6))
    plt.scatter(df_clean['Yield'], df_clean['Predicted_Yield'], color='blue', alpha=0.6)
    
    # 理想的な線（予測が完璧ならこの線に乗る）
    line_min = min(df_clean['Yield'].min(), df_clean['Predicted_Yield'].min())
    line_max = max(df_clean['Yield'].max(), df_clean['Predicted_Yield'].max())
    plt.plot([line_min, line_max], [line_min, line_max], 'r--', lw=2, label='Ideal Prediction')

    plt.xlabel('Actual Yield (USDA Real Data)')
    plt.ylabel('Predicted Yield (S1+S2 Fusion Model)')
    plt.title(f'Corn Yield Prediction Accuracy (R-squared: {model.rsquared:.3f})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    plot_prediction()