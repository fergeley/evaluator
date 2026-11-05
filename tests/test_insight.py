import asyncio
import os
import sys
from dotenv import load_dotenv
from sqlalchemy import text
import pandas as pd
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.insight import Insight
from config import Settings

# Load environment variables
load_dotenv()

async def main():
    # Initialize with SQLite for testing
    insight = Insight(
        db_url='sqlite:///test.db',
        llm_config=Settings().llm_config()
    )
    
    try:
        # Test get_top_trends
        print("\nTesting get_top_trends...")
        trends_df = await insight.get_top_trends()
        if not trends_df.empty:
            print("\nTop Trends Results:")
            print(trends_df.to_string(index=False))
        else:
            print("No trends data returned")
        
        # Test schema generation
        print("\nTesting schema generation...")
        schema = insight.get_schema_json()
        sql = insight.create_sql_schema(schema)
        print("\nGenerated SQL Schema:")
        print(sql)
        
        # Print some sample queries to verify data
        print("\nSample Data Verification:")
        
        queries = {
            "Faculty count by state": """
                SELECT born_state, COUNT(*) as count, 
                       AVG(salary) as avg_salary
                FROM faculty 
                GROUP BY born_state
            """,
            "Age statistics": """
                SELECT 
                    MIN(age) as youngest,
                    MAX(age) as oldest,
                    AVG(age) as avg_age
                FROM faculty
            """,
            "Salary range": """
                SELECT 
                    MIN(salary) as min_salary,
                    MAX(salary) as max_salary,
                    AVG(salary) as avg_salary
                FROM faculty
            """
        }
        
        for title, query in queries.items():
            print(f"\n{title}:")
            with insight.engine.connect() as conn:
                result = conn.execute(text(query))
                df = pd.DataFrame(result.fetchall(), columns=result.keys())
                print(df.to_string(index=False))
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        print(traceback.format_exc())

# Test query efficiency
        print("\nTesting query efficiency...")
        for title, query in queries.items():
            print(f"\n{title} Efficiency Metrics:")
            efficiency_metrics = insight.evaluate_query_efficiency(query)
            print(efficiency_metrics)
        
        # Test query validation
        print("\nTesting query validation...")
        for title, query in queries.items():
            is_valid = insight.validate_sql_query(query)
            print(f"\n{title} is valid: {is_valid}")
            
if __name__ == "__main__":
    asyncio.run(main())