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

def log_openai_interaction(user_input, schema, messages, response, log_path="openai_log.csv"):
    """Log OpenAI API request and response to a CSV file."""
    try:
        # Create file with headers if it doesn't exist
        if not os.path.exists(log_path):
            with open(log_path, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "user_input", "schema", "prompt", "response"])
        
        # Append the log entry
        with open(log_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            # For messages, we'll convert the list to a string representation
            prompt_text = str(messages)
            # For response, extract the content or store the full response if extraction fails
            if hasattr(response, 'choices') and len(response.choices) > 0:
                response_text = response.choices[0].message.content
            else:
                response_text = str(response)
                
            writer.writerow([
                datetime.now().isoformat(),
                user_input,
                schema,
                prompt_text,
                response_text
            ])
    except Exception as e:
        print(f"Failed to log OpenAI interaction: {str(e)}")

def get_custom_prompt(prompt_file="userprompt.md"):
    """Read custom prompt from userprompt.md if it exists."""
    try:
        if os.path.exists(prompt_file) and os.path.getsize(prompt_file) > 0:
            with open(prompt_file, "r", encoding="utf-8") as f:
                return f.read().strip()
    except Exception as e:
        print(f"Failed to read custom prompt: {str(e)}")
    return ""

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
    custom_prompt = get_custom_prompt()
    
    system_content = "You are an expert SQL assistant. Convert natural language into SQL Server queries."
    if custom_prompt:
        system_content += f"\n\n{custom_prompt}"
    
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": f"Schema:\n{schema}\n\nUser input: {user_input}\n\nGenerate only the SQL query without explanation or markdown formatting."}
    ]
    try:
        response = openai.ChatCompletion.create(
            model=model_name,
            messages=messages,
            temperature=0.3,
            max_tokens=300
        )

        log_openai_interaction(user_input, schema, messages, response)

        sql = response.choices[0].message.content.strip()
        if sql.startswith("```") and "```" in sql[3:]:
            sql = sql[sql.find("\n", 3)+1:sql.rindex("```")].strip()
        return sql
    except Exception as e:
        log_openai_interaction(user_input, schema, messages, f"ERROR: {e}")
        return f"-- ERROR: {e} --"

def generate_sql_local_api(user_input, schema, model_name, api_url):
    custom_prompt = get_custom_prompt()
    
    base_prompt = "You are an assistant that converts natural language into SQL for SQL Server."
    if custom_prompt:
        base_prompt += f"\n\n{custom_prompt}"
    
    prompt = f"{base_prompt}\nSchema:\n{schema}\n\nUser input: {user_input}\n\nSQL Query:"
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
    custom_prompt = get_custom_prompt()
    
    base_prompt = "You are an assistant that converts natural language into SQL for SQL Server."
    if custom_prompt:
        base_prompt += f"\n\n{custom_prompt}"
    
    prompt = f"{base_prompt}\nSchema:\n{schema}\n\nUser input: {user_input}\n\nSQL Query:"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    with torch.no_grad():
        output = model.generate(input_ids, max_new_tokens=256)
    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    if "SQL Query:" in decoded:
        return decoded.split("SQL Query:")[-1].strip()
    return decoded.strip()

def execute_query(cursor, sql):
    try:
        # Check if this is a multi-statement query
        statements = sql.strip().split(';')
        statements = [stmt.strip() for stmt in statements if stmt.strip()]
        
        # If there's only one statement, just execute it normally
        if len(statements) == 1:
            cursor.execute(sql)
            # If the query has results (like a SELECT)
            if cursor.description:
                columns = [col[0] for col in cursor.description]
                rows = cursor.fetchall()
                return columns, rows
            else:
                # For action queries (INSERT, UPDATE, DELETE) that don't return rows
                rows_affected = cursor.rowcount
                return ["RowsAffected"], [[rows_affected]]
        else:
            # For multiple statements, execute them one by one
            results = []
            total_rows_affected = 0
            
            for statement in statements:
                if not statement:
                    continue
                    
                cursor.execute(statement)
                
                if cursor.description:  # If the query returns rows (like SELECT)
                    columns = [col[0] for col in cursor.description]
                    rows = cursor.fetchall()
                    results.append({"type": "select", "columns": columns, "rows": rows})
                else:  # For action queries
                    rows_affected = cursor.rowcount
                    total_rows_affected += rows_affected
                    results.append({"type": "action", "rows_affected": rows_affected})
            
            # Return a summary for multiple statements
            if all(r["type"] == "action" for r in results):
                return ["RowsAffected"], [[total_rows_affected]]
            else:
                # If there was at least one SELECT, return the last SELECT result
                for result in reversed(results):
                    if result["type"] == "select":
                        return result["columns"], result["rows"]
                
                # If no SELECT statements were found, return the total affected rows
                return ["RowsAffected"], [[total_rows_affected]]
                
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



# ==================== FOREIGN KEY RESOLUTION FEATURE ====================
def auto_detect_table_relations(cursor):
    table_relations = {}

    fk_query = """
    SELECT 
        fk.TABLE_NAME AS main_table,
        fk.COLUMN_NAME AS fk_column,
        pk.TABLE_NAME AS ref_table
    FROM 
        INFORMATION_SCHEMA.REFERENTIAL_CONSTRAINTS AS rc
        JOIN INFORMATION_SCHEMA.KEY_COLUMN_USAGE AS fk ON rc.CONSTRAINT_NAME = fk.CONSTRAINT_NAME
        JOIN INFORMATION_SCHEMA.KEY_COLUMN_USAGE AS pk ON rc.UNIQUE_CONSTRAINT_NAME = pk.CONSTRAINT_NAME
    """
    cursor.execute(fk_query)
    fk_info = cursor.fetchall()

    for main_table, fk_column, ref_table in fk_info:
        label_column = "Name"
        try:
            cursor.execute(f"SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = ?", ref_table)
            columns = [row[0] for row in cursor.fetchall()]
            name_like = [c for c in columns if 'name' in c.lower()]
            desc_like = [c for c in columns if 'desc' in c.lower()]
            label_column = name_like[0] if name_like else desc_like[0] if desc_like else columns[0]
        except:
            pass
        table_relations.setdefault(main_table, {})[fk_column] = (ref_table, "Id", label_column)
    return table_relations

def resolve_foreign_keys(user_input, cursor, table_relations):
    modified_input = user_input.lower()
    for main_table, fk_map in table_relations.items():
        for col, (ref_table, ref_id, ref_label) in fk_map.items():
            try:
                cursor.execute(f"SELECT {ref_id}, {ref_label} FROM {ref_table}")
                for ref_value, label in cursor.fetchall():
                    label_lower = str(label).lower()
                    if label_lower in modified_input:
                        modified_input = modified_input.replace(label_lower, f"{col} {ref_value}")
            except:
                continue
    return modified_input
# ========================================================================


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

    st.subheader("🔧 Override Table Relations (Optional)")

    manual_relations = st.text_area(
        "Masukkan konfigurasi relasi manual (format Python dict):", 
        height=150, 
        placeholder='Contoh:\n{\n  "Ticket": {\n    "TicketSourceId": ["TicketSource", "Id", "Name"]\n  }\n}'
    )

    if st.button("Gunakan Relasi Manual"):
        try:
            st.session_state.table_relations = eval(manual_relations)
            st.success("✅ Relasi manual berhasil digunakan.")
        except Exception as e:
            st.error(f"⚠️ Gagal memproses input: {e}")


if "conn" in st.session_state:
    st.subheader("\U0001F4AC Tanyakan sesuatu ke database:")
    user_input_raw = st.text_input("Contoh: buat data tiket baru dengan ticket source email")

    if st.button("🧠 Generate SQL"):
        # Gunakan manual jika sudah tersedia
        if "table_relations" not in st.session_state:
            st.session_state.table_relations = auto_detect_table_relations(st.session_state.cursor)

        user_input = resolve_foreign_keys(user_input_raw, st.session_state.cursor, st.session_state.table_relations)

        # Debug view
        if "table_relations" in st.session_state:
            with st.sidebar.expander("🧭 Konfigurasi Relasi Aktif"):
                st.json(st.session_state.table_relations)

        # Generate SQL via LLM
        if llm_engine == "OpenAI":
            sql_query = generate_sql_openai(user_input, st.session_state.schema, model_name, openai_api_key)
        elif llm_engine == "Local API":
            sql_query = generate_sql_local_api(user_input, st.session_state.schema, model_name, local_api_url)
        elif llm_engine == "Transformers":
            sql_query = generate_sql_transformers(user_input, st.session_state.schema, tokenizer, model)
        else:
            sql_query = "-- Unsupported engine --"

        st.session_state.sql_query = sql_query
        st.session_state.user_input = user_input_raw
        st.session_state.chat_history.append({"user": user_input_raw, "sql": sql_query})
        st.success("✅ SQL berhasil dihasilkan.")


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
                # st.write(f"Rows structure: {rows[:2]} (showing first 2)")
                # st.write(f"Columns: {columns}")
                
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
                # st.subheader("\U0001F4CA Visualisasi Otomatis")
                # chart_type = st.selectbox("Pilih tipe chart", ["Bar", "Line", "Area", "Scatter"])
                # if len(columns) >= 2:
                #     x_col = st.selectbox("Kolom X", columns)
                #     y_col = st.selectbox("Kolom Y", columns, index=1)
                #     if chart_type == "Bar":
                #         chart = alt.Chart(df).mark_bar().encode(x=x_col, y=y_col)
                #     elif chart_type == "Line":
                #         chart = alt.Chart(df).mark_line().encode(x=x_col, y=y_col)
                #     elif chart_type == "Area":
                #         chart = alt.Chart(df).mark_area().encode(x=x_col, y=y_col)
                #     elif chart_type == "Scatter":
                #         chart = alt.Chart(df).mark_circle(size=60).encode(x=x_col, y=y_col)
                #     st.altair_chart(chart, use_container_width=True)
                # else:
                #     st.info("\U0001F4CC Data terlalu sedikit untuk divisualisasikan.")
                export_data(columns, [dict(zip(columns, row)) for row in rows])
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
