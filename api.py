from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse  # これが必要です
import sqlite3
import pandas as pd
import numpy as np
import json

app = FastAPI()
DB_NAME = "agri_data.db"

# --- 既存のコード ---
def get_db_data(table_name: str):
    conn = sqlite3.connect(DB_NAME)
    df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
    conn.close()

    data = df.to_dict(orient="records")

    # JSON標準規格に適合しない値を null (None) に置換する再帰処理
    def clean_data(obj):
        if isinstance(obj, list):
            return [clean_data(i) for i in obj]
        if isinstance(obj, dict):
            return {k: clean_data(v) for k, v in obj.items()}
        if isinstance(obj, float):
            if np.isnan(obj) or np.isinf(obj):
                return None
        return obj

    cleaned = clean_data(data)
    # FastAPIの自動変換に頼らず、自身でJSON化して返す
    return JSONResponse(content=cleaned)

@app.get("/")
def read_root():
    return {"message": "API is running", "endpoints": ["/monthly", "/annual", "/view-annual"]}

@app.get("/monthly")
def get_monthly():
    return get_db_data("monthly_data")

@app.get("/annual")
def get_annual():
    return get_db_data("annual_summary")

# --- 追加したコード ---
@app.get("/view-annual", response_class=HTMLResponse)
def view_annual_table():
    conn = sqlite3.connect(DB_NAME)
    df = pd.read_sql("SELECT * FROM annual_summary", conn)
    conn.close()

    # --- 見た目を整える処理 ---
    # 数値を丸める
    if 'Max_NDVI' in df.columns:
        df['Max_NDVI'] = df['Max_NDVI'].round(3)
    if 'Max_Temp_C' in df.columns:
        df['Max_Temp_C'] = df['Max_Temp_C'].round(1)
    
    # None (データなし) を分かりやすい表記に変える
    df = df.fillna('データなし')

    # index列（一番左の0,1,2...）を消してHTML化
    html_table = df.to_html(classes='table table-hover table-bordered', index=False)
    # -----------------------

    return f"""
    <html>
        <head>
            <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css">
            <style>
                body {{ background-color: #f8f9fa; }}
                .container {{ background-color: white; padding: 30px; border-radius: 15px; box-shadow: 0 0 20px rgba(0,0,0,0.1); }}
                thead {{ background-color: #2e7d32; color: white; }}
            </style>
        </head>
        <body class="container mt-5">
            <h2>🌾 年次サマリーレポート</h2>
            <p class="text-muted">衛星画像 (Sentinel) と気象データ (ERA5) の統合解析結果</p>
            {html_table}
            <div class="mt-4">
                <a href="/" class="btn btn-outline-secondary">API一覧へ戻る</a>
            </div>
        </body>
    </html>
    """

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)