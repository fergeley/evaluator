from functools import lru_cache
from services.malay2sql_service import Malay2SQLService
from config import Settings
import redis

@lru_cache()
def get_settings():
    return Settings()

@lru_cache()
def get_redis_client():
    settings = get_settings()
    try:
        client = redis.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            db=settings.redis_db
        )
        # Test connection
        client.ping()
        return client
    except:
        return None

@lru_cache()
def get_malay2sql_service() -> Malay2SQLService:
    settings = get_settings()
    try:
        redis_client = get_redis_client()
    except:
        # Fallback to no caching if Redis is unavailable
        redis_client = None
        
    return Malay2SQLService(
        llm_config=settings.llm_config(),
        cache_client=redis_client
    ) 