import streamlit as st
import pyodbc
import openai
import pandas as pd
import csv
from datetime import datetime
from io import BytesIO

# --------------------------
# Koneksi ke SQL Server
# --------------------------
@st.cache_resource
def connect_to_db(server, database, uid, pwd):
    conn_str = f"""
        DRIVER={{ODBC Driver 17 for SQL Server}};
        SERVER={server};
        DATABASE={database};
        UID={uid};
        PWD={pwd};
    """
    return pyodbc.connect(conn_str)

# --------------------------
# Ambil struktur schema database
# --------------------------
def get_db_schema(cursor):
    schema = ""
    cursor.execute("""
        SELECT TABLE_NAME, COLUMN_NAME
        FROM INFORMATION_SCHEMA.COLUMNS
        ORDER BY TABLE_NAME, ORDINAL_POSITION
    """)
    tables = {}
    for table_name, column_name in cursor.fetchall():
        tables.setdefault(table_name, []).append(column_name)

    for table, columns in tables.items():
        schema += f"- {table}({', '.join(columns)})\n"
    return schema.strip()

# --------------------------
# Generate SQL via OpenAI API
# --------------------------
def generate_sql_openai(user_input, schema, model_name, openai_api_key):
    openai.api_key = openai_api_key

    messages = [
        {
            "role": "system",
            "content": "You are an expert SQL assistant. Convert user questions into SQL Server queries using the database schema provided. Return ONLY the raw SQL without any markdown formatting, code blocks, or explanations."
        },
        {
            "role": "user",
            "content": f"""Database schema:
{schema}

User input: {user_input}

Generate only the SQL query without explanation or markdown formatting."""
        }
    ]

    try:
        response = openai.ChatCompletion.create(
            model=model_name,
            messages=messages,
            temperature=0.3,
            max_tokens=300
        )
        sql = response.choices[0].message.content.strip()
        
        # Remove markdown code blocks if present
        if sql.startswith("```") and "```" in sql[3:]:
            # Extract content between first ``` and last ```
            sql = sql[sql.find("\n", 3)+1:sql.rindex("```")].strip()
        
        return sql
    except Exception as e:
        return f"-- ERROR: {e} --"

# --------------------------
# Eksekusi query SQL
# --------------------------
def execute_query(cursor, sql):
    try:
        cursor.execute(sql)
        columns = [col[0] for col in cursor.description]
        rows = cursor.fetchall()
        return columns, rows
    except Exception as e:
        return [], [[str(e)]]

# --------------------------
# Simpan log pertanyaan + SQL
# --------------------------
def log_query(question, sql_query, log_path="query_log.csv"):
    with open(log_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([datetime.now().isoformat(), question, sql_query])

# --------------------------
# Export hasil ke CSV dan Excel
# --------------------------
def export_data(columns, rows):
    df = pd.DataFrame(rows, columns=columns)

    # Export ke CSV
    csv_data = df.to_csv(index=False).encode('utf-8')

    # Export ke Excel pakai BytesIO
    excel_buffer = BytesIO()
    df.to_excel(excel_buffer, index=False, engine="openpyxl")
    excel_data = excel_buffer.getvalue()

    st.download_button("📤 Download CSV", csv_data, file_name="query_result.csv", mime="text/csv")
    st.download_button("📤 Download Excel", excel_data, file_name="query_result.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# --------------------------
# Streamlit App
# --------------------------
st.set_page_config(page_title="SQL Chatbot with OpenAI", layout="wide")
st.title("🤖 SQL Chatbot (Natural Language → SQL) with OpenAI API")

# Sidebar untuk koneksi dan konfigurasi
with st.sidebar:
    st.header("🔐 SQL Server Connection")
    server = st.text_input("Server", "localhost")
    database = st.text_input("Database", "database")
    uid = st.text_input("Username", "sa")
    pwd = st.text_input("Password", type="password")

    st.header("🔑 OpenAI API Config")
    openai_api_key = st.text_input("API Key", type="password")
    model_name = st.selectbox("Model", ["gpt-3.5-turbo", "gpt-4"], index=0)

    if st.button("🔗 Connect"):
        try:
            conn = connect_to_db(server, database, uid, pwd)
            cursor = conn.cursor()
            schema = get_db_schema(cursor)
            st.session_state.conn = conn
            st.session_state.cursor = cursor
            st.session_state.schema = schema
            st.session_state.model_name = model_name
            st.session_state.openai_api_key = openai_api_key
            st.success("✅ Connected to database!")
        except Exception as e:
            st.error(f"❌ Connection failed: {e}")

# Main area untuk input dan eksekusi
if "conn" in st.session_state:
    st.subheader("💬 Tanyakan sesuatu ke database:")
    user_input = st.text_input("Contoh: Tampilkan semua order dari pelanggan bernama Budi")

    if st.button("🧠 Generate SQL"):
        sql_query = generate_sql_openai(
            user_input,
            st.session_state.schema,
            st.session_state.model_name,
            st.session_state.openai_api_key
        )
        st.session_state.sql_query = sql_query
        st.session_state.user_input = user_input
        st.success("✅ SQL berhasil di-generate.")

    if "sql_query" in st.session_state:
        st.subheader("📋 Query yang akan dijalankan:")
        with st.expander("Lihat SQL", expanded=True):
            st.code(st.session_state.sql_query, language="sql")

        # Validasi query aman
        sql_lower = st.session_state.sql_query.strip().lower()
        is_safe = sql_lower.startswith("select") and not any(
            keyword in sql_lower
            for keyword in ["delete", "drop", "update", "insert", "alter", "truncate"]
        )

        if not is_safe:
            st.error("🚨 Query ini tidak aman! Hanya perintah SELECT yang diperbolehkan.")
        else:
            st.success("✅ Query aman untuk dijalankan.")
            if st.button("🚀 Jalankan Query"):
                with st.spinner("Menjalankan query..."):
                    columns, rows = execute_query(st.session_state.cursor, st.session_state.sql_query)
                    if columns:
                        st.success(f"✅ Berhasil. Menampilkan {len(rows)} baris.")
                        st.dataframe([dict(zip(columns, row)) for row in rows])

                        # Simpan log dan export
                        log_query(st.session_state.user_input, st.session_state.sql_query)
                        # export_data(columns, rows)
                    else:
                        st.warning("⚠️ Query valid, tapi tidak mengembalikan data.")
