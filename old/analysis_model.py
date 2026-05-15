import statsmodels.api as sm

# 収穫量を予測する「重回帰分析」モデル
# Yield = a * NDVI + b * RVI + c * GDD + d
X = df_clean[['Max_NDVI', 'Max_RVI', 'Max_Temp_C']]
X = sm.add_constant(X) # 切片の追加
y = df_clean['USDA_Yield_bu_acre']

model = sm.OLS(y, X).fit()
print(model.summary())