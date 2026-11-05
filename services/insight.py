import pandas as pd
from sqlalchemy import create_engine, text, MetaData, Table, Column, Integer, String, Float, DateTime
from services.malay2sql_service import Malay2SQLService
from config import LLMConfig
from typing import Dict, Any
import logging
import datetime
import time
import sqlalchemy as sa

class Insight:
    def __init__(self, db_url: str, llm_config: LLMConfig, cache_client=None):
        # Use SQLite for testing if no db_url is provided
        self.db_url = db_url or 'sqlite:///test.db'
        self.engine = create_engine(self.db_url)
        self.malay2sql_service = Malay2SQLService(llm_config, cache_client)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler()
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        
        # Create test data if using SQLite
        if 'sqlite' in self.db_url:
            self._create_test_data()

    def _create_test_data(self):
        """Create test database and populate with sample data"""
        metadata = MetaData()
        
        # Define faculty table
        faculty = Table('faculty', metadata,
            Column('id', Integer, primary_key=True),
            Column('name', String),
            Column('age', Integer),
            Column('born_state', String),
            Column('department_id', Integer),
            Column('join_date', DateTime),
            Column('salary', Float)
        )
        
        # Create tables
        metadata.create_all(self.engine)
        
        # Sample data
        test_data = [
            {
                'id': 1,
                'name': 'Dr. John Smith',
                'age': 45,
                'born_state': 'Texas',
                'department_id': 1,
                'join_date': datetime.datetime(2010, 1, 15),
                'salary': 95000.00
            },
            {
                'id': 2,
                'name': 'Dr. Maria Garcia',
                'age': 38,
                'born_state': 'California',
                'department_id': 2,
                'join_date': datetime.datetime(2015, 6, 1),
                'salary': 88000.00
            },
            {
                'id': 3,
                'name': 'Dr. Robert Chen',
                'age': 52,
                'born_state': 'New York',
                'department_id': 1,
                'join_date': datetime.datetime(2005, 9, 30),
                'salary': 105000.00
            },
            {
                'id': 4,
                'name': 'Dr. Sarah Johnson',
                'age': 41,
                'born_state': 'Texas',
                'department_id': 3,
                'join_date': datetime.datetime(2012, 3, 15),
                'salary': 92000.00
            },
            {
                'id': 5,
                'name': 'Dr. James Wilson',
                'age': 35,
                'born_state': 'California',
                'department_id': 2,
                'join_date': datetime.datetime(2018, 8, 1),
                'salary': 85000.00
            }
        ]
        
        # Insert test data
        with self.engine.connect() as conn:
            # Delete existing data
            conn.execute(text("DELETE FROM faculty"))
            
            # Insert new data
            for record in test_data:
                insert_query = text("""
                    INSERT INTO faculty (id, name, age, born_state, department_id, join_date, salary)
                    VALUES (:id, :name, :age, :born_state, :department_id, :join_date, :salary)
                """)
                conn.execute(insert_query, record)
            conn.commit()
        
        self.logger.info("Test data created successfully")

    async def initialize_schema(self):
        """Initialize schema for the default user"""
        try:
            schema = self.get_schema_json()
            # Initialize schema index
            self.malay2sql_service.schema_index.build_index(schema, "default_user")
            self.malay2sql_service.user_schemas["default_user"] = schema
            self.logger.info("Schema initialized for default user")
        except Exception as e:
            self.logger.error(f"Error initializing schema: {e}")
            raise

    async def get_top_trends(self) -> pd.DataFrame:
        prompt = """
        Generate a single SQL query to find the most interesting trend in the data.
        Choose one of these analyses:
        - Count the total number of faculty members by state and show their average salary
        - Find the oldest and youngest faculty members with their details
        - Show faculty members whose age is above average, ordered by salary
        - List the states with the most faculty members and their average age
        - Show the salary distribution across different age groups
        """
        try:
            # Make sure schema is initialized
            await self.initialize_schema()
            
            # First translate the prompt to English if it's in Malay
            english_translation = self.malay2sql_service.translate_malay_to_english(prompt)
            
            # Get schema for the user
            schema = self.get_schema_json()
            schema_sql = self.create_sql_schema(schema)
            
            # Get relevant columns
            relevant_columns = await self.malay2sql_service.get_relevant_columns(english_translation, "default_user")
            
            # Generate SQL query using the Malay2SQLService
            query = await self.malay2sql_service.generate_sql_query(
                english_translation,
                schema_sql,
                relevant_columns
            )
            
            self.logger.debug(f"Generated SQL query: {query}")
            
            # Split multiple statements and execute only the first one
            statements = [stmt.strip() for stmt in query.split(';') if stmt.strip()]
            if not statements:
                return pd.DataFrame()
            
            # Execute the first (or only) query
            first_query = statements[0]
            self.logger.debug(f"Executing query: {first_query}")
            
            with self.engine.connect() as connection:
                result = connection.execute(text(first_query))
                df = pd.DataFrame(result.fetchall(), columns=result.keys())
            
            if len(statements) > 1:
                self.logger.warning(f"Multiple statements detected. Only executed the first one: {first_query}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error in get_top_trends: {e}")
            raise

    def create_sql_schema(self, schema_json: Dict[str, Any]) -> str:
        """Convert JSON schema to SQL CREATE TABLE statement"""
        type_mapping = {
            "String": "VARCHAR(255)",
            "Int64": "INTEGER",
            "Float64": "DECIMAL(15,2)",
            "DateTime64(3)": "DATETIME"
        }
        
        columns = []
        for col_name, info in schema_json["columns"].items():
            data_type = type_mapping.get(info["data_type"], "VARCHAR(255)")
            comment = info["description"].replace("'", "''")
            columns.append(f"{col_name} {data_type} COMMENT '{comment}'")
            
        columns_str = ',\n    '.join(columns)
        create_statement = (
            f"CREATE TABLE {schema_json['table_name']} (\n"
            f"    {columns_str}\n"
            f");"
        )
        
        return create_statement

    def get_schema_json(self) -> Dict[str, Any]:
        """Retrieve the schema JSON"""
        return {
            "table_name": "faculty",
            "columns": {
                "id": {"data_type": "Int64", "description": "Faculty member ID"},
                "name": {"data_type": "String", "description": "Name of the faculty member"},
                "age": {"data_type": "Int64", "description": "Age of the faculty member"},
                "born_state": {"data_type": "String", "description": "State where the faculty member was born"},
                "department_id": {"data_type": "Int64", "description": "ID of the department"},
                "join_date": {"data_type": "DateTime64(3)", "description": "Date when faculty member joined"},
                "salary": {"data_type": "Float64", "description": "Faculty member's salary"}
            }
        }
    
    def evaluate_query_efficiency(self, sql_query: str) -> Dict[str, Any]:
        """Evaluate the efficiency of a SQL query."""
        efficiency_metrics = {}
        try:
            # Measure execution time
            start_time = time.time()
            with self.engine.connect() as conn:
                result = conn.execute(sa.text(sql_query))
                execution_time = time.time() - start_time
                efficiency_metrics['execution_time'] = execution_time

                # Analyze query plan
                query_plan = conn.execute(sa.text(f"EXPLAIN {sql_query}")).fetchall()
                efficiency_metrics['query_plan'] = query_plan

                # Check index usage
                index_usage = any("INDEX" in str(row) for row in query_plan)
                efficiency_metrics['index_usage'] = index_usage

                # Get result size
                result_size = len(result.fetchall())
                efficiency_metrics['result_size'] = result_size

                # Compare with golden query
                golden_query = self.malay2sql_service.golden_query
                similarity_score = golden_query.calculate_cosine_similarity(sql_query)
                efficiency_metrics['similarity_to_golden_query'] = similarity_score

        except Exception as e:
            self.logger.error(f"Error evaluating query efficiency: {e}")
            efficiency_metrics['error'] = str(e)

        return efficiency_metrics

    def validate_sql_query(self, sql_query: str) -> bool:
        """Validate if the SQL query is correct and functional."""
        try:
            with self.engine.connect() as conn:
                conn.execute(sa.text(sql_query))
            return True
        except Exception as e:
            self.logger.error(f"Invalid SQL query: {e}")
            return False

# Example usage:
# insight = Insight(db_url="your_database_url", llm_config=Settings().llm_config())
# trends_df = await insight.get_top_trends()

# Example usage:
# insight = Insight(db_url="your_database_url", llm_config=Settings().llm_config())
# trends_df = await insight.get_top_trends()