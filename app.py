import streamlit as st
import pyodbc
import openai
import pandas as pd
import csv
from io import BytesIO
from datetime import datetime
import requests
import os
import altair as alt

os.environ["STREAMLIT_WATCHDOG_IGNORE_PATH_CONTAINS"] = "torch,transformers"

@st.cache_resource
def connect_to_db(engine, server, database, uid, pwd, port=None):
    if engine == "SQL Server":
        conn_str = f"DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={server};DATABASE={database};UID={uid};PWD={pwd};"
    elif engine == "PostgreSQL":
        conn_str = f"DRIVER={{PostgreSQL Unicode}};SERVER={server};DATABASE={database};UID={uid};PWD={pwd};PORT={port or 5432};"
    elif engine == "MySQL":
        conn_str = f"DRIVER={{MySQL ODBC 8.0 Unicode Driver}};SERVER={server};DATABASE={database};UID={uid};PWD={pwd};PORT={port or 3306};"
    else:
        raise ValueError("Unsupported database engine.")
    return pyodbc.connect(conn_str)

def get_db_schema(cursor):
    schema = ""
    cursor.execute("SELECT TABLE_NAME, COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS ORDER BY TABLE_NAME, ORDINAL_POSITION")
    tables = {}
    for table_name, column_name in cursor.fetchall():
        tables.setdefault(table_name, []).append(column_name)
    for table, columns in tables.items():
        schema += f"- {table}({', '.join(columns)})\n"
    return schema.strip()

def generate_sql_openai(user_input, schema, model_name, openai_api_key):
    openai.api_key = openai_api_key
    messages = [
        {"role": "system", "content": "You are an expert SQL assistant. Convert natural language into SQL Server queries."},
        {"role": "user", "content": f"Schema:\n{schema}\n\nUser input: {user_input}\n\nGenerate only the SQL query without explanation or markdown formatting."}
    ]
    try:
        response = openai.ChatCompletion.create(
            model=model_name,
            messages=messages,
            temperature=0.3,
            max_tokens=300
        )
        sql = response.choices[0].message.content.strip()
        if sql.startswith("```") and "```" in sql[3:]:
            sql = sql[sql.find("\n", 3)+1:sql.rindex("```")].strip()
        return sql
    except Exception as e:
        return f"-- ERROR: {e} --"

def generate_sql_local_api(user_input, schema, model_name, api_url):
    prompt = f"You are an assistant that converts natural language into SQL for SQL Server.\nSchema:\n{schema}\n\nUser input: {user_input}\n\nSQL Query:"
    payload = {"model": model_name, "prompt": prompt, "stream": False}
    try:
        response = requests.post(api_url, json=payload)
        response.raise_for_status()
        result = response.json()
        return result.get("response", "").strip()
    except Exception as e:
        return f"-- ERROR: {e} --"

def generate_sql_transformers(user_input, schema, tokenizer, model):
    import torch
    prompt = f"You are an assistant that converts natural language into SQL for SQL Server.\nSchema:\n{schema}\n\nUser input: {user_input}\n\nSQL Query:"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    with torch.no_grad():
        output = model.generate(input_ids, max_new_tokens=256)
    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    if "SQL Query:" in decoded:
        return decoded.split("SQL Query:")[-1].strip()
    return decoded.strip()

def execute_query(cursor, sql):
    try:
        cursor.execute(sql)
        columns = [col[0] for col in cursor.description]
        rows = cursor.fetchall()
        return columns, rows
    except Exception as e:
        return [], [[str(e)]]

def log_query(question, sql_query, log_path="query_log.csv"):
    with open(log_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([datetime.now().isoformat(), question, sql_query])

def export_data(columns, rows):
    df = pd.DataFrame(rows, columns=columns)
    csv_data = df.to_csv(index=False).encode("utf-8")
    excel_buf = BytesIO()
    df.to_excel(excel_buf, index=False, engine="openpyxl")
    excel_data = excel_buf.getvalue()
    st.download_button("\U0001F4E4 Download CSV", csv_data, file_name="query_result.csv", mime="text/csv")
    st.download_button("\U0001F4E4 Download Excel", excel_data, file_name="query_result.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

def save_chat_history(log_path="chat_history.csv"):
    with open(log_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "user_input", "generated_sql"])
        for entry in st.session_state.chat_history:
            writer.writerow([datetime.now().isoformat(), entry['user'], entry['sql']])

st.set_page_config(page_title="LLM SQL Chatbot", layout="wide")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.title("\U0001F916 Natural Language SQL Chatbot")

with st.sidebar:
    st.header("\U0001F5C4️ Database Config")
    engine = st.selectbox("Database Engine", ["SQL Server", "PostgreSQL", "MySQL"])
    server = st.text_input("Server", "localhost")
    port = st.text_input("Port (opsional)", "")
    database = st.text_input("Database", "testdb")
    uid = st.text_input("Username", "sa")
    pwd = st.text_input("Password", type="password")

    st.header("\U0001F916 LLM Config")
    llm_engine = st.selectbox("Engine", ["OpenAI", "Local API", "Transformers"])
    model_name = st.text_input("Model Name", value="gpt-3.5-turbo" if llm_engine == "OpenAI" else "deepseek-ai/deepseek-coder-33b-instruct")

    openai_api_key = ""
    local_api_url = ""
    tokenizer = None
    model = None

    if llm_engine == "OpenAI":
        openai_api_key = st.text_input("OpenAI API Key", type="password")
    elif llm_engine == "Local API":
        local_api_url = st.text_input("Local API URL", value="http://localhost:11434/api/generate")
    elif llm_engine == "Transformers":
        st.caption("\U0001F680 Model lokal akan di-load menggunakan transformers.")
        @st.cache_resource
        def load_model(name):
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
            import warnings
            warnings.filterwarnings("ignore")
            try:
                tokenizer = AutoTokenizer.from_pretrained(name)
                model = AutoModelForCausalLM.from_pretrained(
                    name,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    low_cpu_mem_usage=True,
                    trust_remote_code=True
                )
                return tokenizer, model
            except Exception as e:
                st.error(f"Failed to load model: {str(e)}")
                return None, None
        tokenizer, model = load_model(model_name)

    if st.button("\U0001F517 Connect"):
        try:
            conn = connect_to_db(engine, server, database, uid, pwd, port)
            cursor = conn.cursor()
            schema = get_db_schema(cursor)
            st.session_state.conn = conn
            st.session_state.cursor = cursor
            st.session_state.schema = schema
            st.session_state.engine = engine
            st.success(f"\u2705 Connected to {engine} database!")
        except Exception as e:
            st.error(f"\u274C Connection failed: {e}")

if "conn" in st.session_state:
    st.subheader("\U0001F4AC Tanyakan sesuatu ke database:")
    user_input = st.text_input("Contoh: Tampilkan semua order dari pelanggan bernama Budi")

    if st.button("\U0001F9E0 Generate SQL"):
        if llm_engine == "OpenAI":
            sql_query = generate_sql_openai(user_input, st.session_state.schema, model_name, openai_api_key)
        elif llm_engine == "Local API":
            sql_query = generate_sql_local_api(user_input, st.session_state.schema, model_name, local_api_url)
        elif llm_engine == "Transformers":
            sql_query = generate_sql_transformers(user_input, st.session_state.schema, tokenizer, model)
        else:
            sql_query = "-- Unsupported engine --"

        st.session_state.sql_query = sql_query
        st.session_state.user_input = user_input
        st.session_state.chat_history.append({"user": user_input, "sql": sql_query})
        st.success("\u2705 SQL berhasil dihasilkan.")

    if "sql_query" in st.session_state:
        st.subheader("\U0001F4CB Query yang akan dijalankan:")
        with st.expander("Lihat SQL", expanded=True):
            st.code(st.session_state.sql_query, language="sql")

        sql_lower = st.session_state.sql_query.strip().lower()
        # is_safe = not any(kw in sql_lower for kw in ["delete", "drop", "update", "insert", "alter", "truncate"])

        # if not is_safe:
        #     st.error("\U0001F6A8 Query ini tidak aman! Hanya SELECT yang diizinkan.")
        # else:
        # st.success("\u2705 Query aman untuk dijalankan.")
        if st.button("\U0001F680 Jalankan Query"):
            with st.spinner("Menjalankan query..."):
                columns, rows = execute_query(st.session_state.cursor, st.session_state.sql_query)
                if columns:
                    st.success(f"\u2705 Menampilkan {len(rows)} baris.")
                    st.dataframe([dict(zip(columns, row)) for row in rows])
                    
                    # First, check the structure of 'rows' and 'columns'
                    st.write(f"Rows structure: {rows[:2]} (showing first 2)")
                    st.write(f"Columns: {columns}")
                    
                    # Instead of checking instance types, check if length matches
                    if rows and len(columns) > 0:
                        try:
                            # Just try to create the DataFrame directly
                            df = pd.DataFrame([dict(zip(columns, row)) for row in rows], columns=columns)
                            # If we get here, it worked without errors
                        except ValueError as e:
                            # If there's an error, log it and use a fallback approach
                            st.error(f"DataFrame creation error: {str(e)}")
                            if len(columns) == 1:
                                df = pd.DataFrame(rows, columns=columns)
                            else:
                                # Last resort: just use the first column
                                df = pd.DataFrame([[r[0] if hasattr(r, "__getitem__") else r] for r in rows], 
                                                columns=columns[:1])
                                st.warning("Data shape mismatch: Using only the first column")

                    # Chart
                    st.subheader("\U0001F4CA Visualisasi Otomatis")
                    chart_type = st.selectbox("Pilih tipe chart", ["Bar", "Line", "Area", "Scatter"])
                    if len(columns) >= 2:
                        x_col = st.selectbox("Kolom X", columns)
                        y_col = st.selectbox("Kolom Y", columns, index=1)
                        if chart_type == "Bar":
                            chart = alt.Chart(df).mark_bar().encode(x=x_col, y=y_col)
                        elif chart_type == "Line":
                            chart = alt.Chart(df).mark_line().encode(x=x_col, y=y_col)
                        elif chart_type == "Area":
                            chart = alt.Chart(df).mark_area().encode(x=x_col, y=y_col)
                        elif chart_type == "Scatter":
                            chart = alt.Chart(df).mark_circle(size=60).encode(x=x_col, y=y_col)
                        st.altair_chart(chart, use_container_width=True)
                    else:
                        st.info("\U0001F4CC Data terlalu sedikit untuk divisualisasikan.")

                    log_query(st.session_state.user_input, st.session_state.sql_query)
                else:
                    st.warning("\u26A0\uFE0F Query valid tapi tidak mengembalikan hasil.")

    st.subheader("\U0001F4DC Riwayat Chat")
    for i, chat in enumerate(st.session_state.chat_history[::-1]):
        with st.expander(f"Chat #{len(st.session_state.chat_history)-i}"):
            st.markdown(f"**\U0001F464 User:** {chat['user']}")
            st.code(chat['sql'], language="sql")

    if st.button("\U0001F9F9 Bersihkan Chat"):
        st.session_state.chat_history = []
        st.success("Histori chat dibersihkan.")
