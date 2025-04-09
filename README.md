# LLM2DB - Natural Language to SQL Query Converter

LLM2DB is a Streamlit application that converts natural language questions to SQL queries using various LLM models. It allows users to connect to different database engines, generate SQL from natural language, and visualize the results.

## Features

- Connect to multiple database engines (SQL Server, PostgreSQL, MySQL)
- Generate SQL queries using different LLM providers:
  - OpenAI API
  - Local API endpoints
  - Local transformers models
- Execute and visualize query results
- Export results to CSV or Excel
- Track query history
- Automatic data visualization with charts

## Installation

### Prerequisites

- Python 3.8+
- Database drivers (ODBC drivers for SQL Server, PostgreSQL, or MySQL)
- An OpenAI API key (if using OpenAI mode)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/llm2db.git
cd llm2db
```

2. Install the dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

### Database Configuration

Configure your database connection in the sidebar:
- Select the database engine (SQL Server, PostgreSQL, MySQL)
- Enter server details, database name, username and password
- Click "Connect" to establish the connection

### LLM Configuration

Choose your preferred LLM engine:

1. **OpenAI**:
   - Enter your OpenAI API key
   - Specify the model (e.g., gpt-3.5-turbo, gpt-4)

2. **Local API**:
   - Enter the URL for your local LLM API endpoint
   - Default: http://localhost:11434/api/generate (Ollama)

3. **Transformers**:
   - Specify the model to load locally (e.g., deepseek-ai/deepseek-coder-33b-instruct)
   - Requires sufficient hardware resources

## Usage

1. Start the application:
```bash
streamlit run app.py
```

2. Connect to your database
3. Enter your question in natural language (e.g., "Show all orders from customers named John")
4. Click "Generate SQL" to create the SQL query
5. Click "Run Query" to execute and see the results
6. Visualize the results using the built-in chart options
7. Export the results as CSV or Excel if needed

## Query Logging

The application automatically logs all queries to [`query_log.csv`](query_log.csv ) with:
- Timestamp
- Original user question
- Generated SQL query

## System Requirements

For local model usage (Transformers mode):
- 16GB+ RAM
- GPU with 8GB+ VRAM (for larger models)
- SSD with sufficient space for model storage

## Troubleshooting

If you encounter connection issues:
- Verify your database credentials
- Check that the appropriate ODBC drivers are installed
- Ensure your database server allows remote connections

For model loading issues:
- Check available system resources
- Consider using a smaller model or switching to API mode

## License

MIT License