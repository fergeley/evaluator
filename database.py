from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker
import os

# Get database URL from environment variable
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://malay2sql-user:password@localhost/malay2sql"  # Default for local development
)

# Pin the driver to psycopg2 (what requirements.txt installs); SQLAlchemy 2.1
# otherwise picks psycopg 3 for a bare postgresql:// URL.
if DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = "postgresql+psycopg2://" + DATABASE_URL[len("postgresql://"):]

# Create engine without check_same_thread (not needed for PostgreSQL)
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def create_tables():
    Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()