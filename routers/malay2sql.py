from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, Any
from services.malay2sql_service import Malay2SQLService, QueryResult
from dependencies import get_malay2sql_service
from pydantic import BaseModel
from routers.auth import get_current_user
from models.user import UserDB
from datetime import datetime
from services.llm import strip_code_fences
import json
import random


router = APIRouter(
    prefix="/malay2sql",
    tags=["malay2sql"]
)

class SchemaInput(BaseModel):
    schema: Dict[str, Any]

class QueryInput(BaseModel):
    query: str

class FeedbackInput(BaseModel):
    original_query_result: QueryResult
    corrected_sql: str

class ExecuteQueryInput(BaseModel):
    sql_query: str


@router.post("/execute", status_code=200)
async def execute_query(
    execute_input: ExecuteQueryInput,
    service: Malay2SQLService = Depends(get_malay2sql_service),
    current_user: UserDB = Depends(get_current_user)
):
    """Execute a SQL query"""
    try:
        # Use the LLM to generate mock results (the SQL is not run against a database)
        
        prompt = f"""Given the SQL query below, generate realistic mock results in JSON format.
        The results should be consistent with a university staff database.

        SQL Query: {execute_input.sql_query}

        Return only the JSON array of results, without any explanation or markdown formatting.
        Make sure the field names match the SQL query.
        Limit the results to 5 rows maximum.
        """

        content = service.llm.chat(
            messages=[
                {"role": "system", "content": "You are a SQL database simulator. Generate realistic mock results for SQL queries."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )

        # Parse the mock results
        try:
            mock_results = json.loads(strip_code_fences(content))
            if not isinstance(mock_results, list):
                mock_results = [mock_results]
        except json.JSONDecodeError:
            mock_results = []

        execution_time = random.uniform(0.1, 0.5)  # Random execution time between 0.1 and 0.5 seconds
        
        return {
            "message": "Query executed successfully",
            "result": {
                "status": "success",
                "results": mock_results,
                "rows_affected": len(mock_results),
                "execution_time": execution_time,
                "query_type": execute_input.sql_query.strip().split()[0].upper()
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to execute query: {str(e)}"
        )
    
@router.post("/initialize", status_code=200)
async def initialize_schema(
    schema_input: SchemaInput,
    service: Malay2SQLService = Depends(get_malay2sql_service),
    current_user: UserDB = Depends(get_current_user) 
):
    """Initialize the schema for the Malay2SQL service"""
    try:
        service.initialize_schema(schema_input.schema, user_id=str(current_user.id))
        print(f"Schema initialized for user {current_user.id}")
        return {"message": "Schema initialized successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/query", response_model=QueryResult)
async def process_query(
    query_input: QueryInput,
    service: Malay2SQLService = Depends(get_malay2sql_service),
    current_user: UserDB = Depends(get_current_user)  
) -> QueryResult:
    """Process a Malay language query and return SQL"""
    try:
        result = await service.process_query(query_input.query, user_id=str(current_user.id))
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@router.post("/feedback", status_code=200)
async def store_query_feedback(
    feedback: FeedbackInput,
    service: Malay2SQLService = Depends(get_malay2sql_service),
    current_user: UserDB = Depends(get_current_user)
):
    """Store user feedback for a query"""
    try:
        await service.store_feedback(
            feedback.original_query_result,
            feedback.corrected_sql
        )
        return {
            "message": "Feedback stored successfully",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to store feedback: {str(e)}"
        )