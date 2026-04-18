from fastapi import FastAPI
from fastapi.responses import HTMLResponse  # これが必要です
import sqlite3
import pandas as pd

app = FastAPI()
DB_NAME = "agri_data.db"

# --- 既存のコード ---
def get_db_data(table_name: str):
    conn = sqlite3.connect(DB_NAME)
    df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
    conn.close()
    return df.to_dict(orient="records")

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
    html_table = df.to_html(classes='table table-striped', index=False)
    return f"""
    <html>
        <head>
            <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css">
        </head>
        <body class="container mt-5">
            <h2>🌾 年次サマリーレポート</h2>
            {html_table}
        </body>
    </html>
    """

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)