from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
import pyodbc
import openai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Set up OpenAI API key (Replace with your API key in .env file)
openai.api_key = os.getenv("OPENAI_API_KEY")

# Database connection function (MSSQL)
def get_db_connection():
    conn = pyodbc.connect(
        f"DRIVER={{SQL Server}};SERVER={os.getenv('DB_SERVER')};DATABASE={os.getenv('DB_NAME')};UID={os.getenv('DB_USER')};PWD={os.getenv('DB_PASSWORD')}"
    )
    return conn

# Request model for query input
class QueryRequest(BaseModel):
    question: str

# Security function (basic authentication)
def authenticate_user(api_key: str = Depends(lambda: os.getenv("API_ACCESS_KEY"))):
    if api_key != os.getenv("API_ACCESS_KEY"):
        raise HTTPException(status_code=403, detail="Unauthorized")

# Convert natural language to SQL
@app.post("/query", dependencies=[Depends(authenticate_user)])
def query_database(request: QueryRequest):
    prompt = f"Convert this natural language request into an SQL query: {request.question}"
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    sql_query = response["choices"][0]["message"]["content"].strip()
    
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(sql_query)
        result = cursor.fetchall()
        conn.commit()
    except Exception as e:
        conn.rollback()
        result = str(e)
    finally:
        conn.close()
    
    return {"query": sql_query, "result": result}

# Test route
@app.get("/")
def home():
    return {"message": "AI SQL Query App is running with MSSQL and Security!"}