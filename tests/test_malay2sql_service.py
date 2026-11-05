import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from services.malay2sql_service import Malay2SQLService
from config import Settings
import time
from datetime import datetime
import pytest
import asyncio

USER_ID = "test-user"


def _llm_config_or_skip():
    try:
        return Settings().llm_config()
    except ValueError as e:
        pytest.skip(f"needs an LLM key: {e}")

def test_malay2sql_service():
    asyncio.run(_run_pipeline())


async def _run_pipeline():
    
    
    # Print GPU information
    print("\nGPU Information:")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name()}")
        print(f"Device memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print("-" * 50)

    # Initialize service
    service = Malay2SQLService(
        llm_config=_llm_config_or_skip(),
        cache_client=None  # Disable cache for testing
    )

    # Test schema
    test_schema = {
        "table_name": "users",
        "columns": {
            "id": {
                "data_type": "Int64",
                "description": "Unique identifier for each user"
            },
            "name": {
                "data_type": "String",
                "description": "Full name of the user"
            },
            "email": {
                "data_type": "String",
                "description": "Email address of the user"
            },
            "age": {
                "data_type": "Int64",
                "description": "Age of the user in years"
            }
        }
    }

    # Initialize schema
    print("\nInitializing schema...")
    service.initialize_schema(test_schema, user_id=USER_ID)
    print("Schema initialized successfully")
    print("-" * 50)

    # Test cases with Malay queries
    test_cases = [
        "Tunjukkan semua pengguna",
        "Cari pengguna yang berumur lebih dari 25 tahun",
        "Senaraikan nama dan emel untuk semua pengguna",
        "Berapa ramai pengguna yang ada dalam sistem?"
    ]

    print("\nTesting full Malay2SQL pipeline:")
    print("-" * 50)
    
    for malay_query in test_cases:
        print(f"\nMalay Query: {malay_query}")
        
        # Time the execution
        start_time = time.time()
        
        # Process query
        result = await service.process_query(malay_query, user_id=USER_ID)
        
        # Print results
        print(f"English Translation: {result.english_translation}")
        print(f"SQL Query: {result.sql_query}")
        print("Relevant Columns:")
        for col, desc in result.relevant_columns.items():
            print(f"  - {col}: {desc}")
        print(f"Execution Time: {result.execution_time:.2f}s")
        print(f"Timestamp: {result.timestamp}")
        print("-" * 50)

        # Basic assertions
        assert isinstance(result.malay_query, str)
        assert isinstance(result.english_translation, str)
        assert isinstance(result.sql_query, str)
        assert isinstance(result.relevant_columns, dict)
        assert isinstance(result.execution_time, float)
        assert isinstance(result.timestamp, str)
        
        # SQL-specific assertions
        assert result.sql_query.upper().startswith("SELECT")
        assert "users" in result.sql_query.lower()
        
        # Timing assertions
        assert result.execution_time > 0
        assert datetime.fromisoformat(result.timestamp)

if __name__ == "__main__":
    asyncio.run(_run_pipeline())