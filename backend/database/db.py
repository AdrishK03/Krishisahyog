import os

from motor.motor_asyncio import AsyncIOMotorClient

_client = None


def get_db():
    global _client
    if _client is None:
        _client = AsyncIOMotorClient(os.getenv("MONGODB_URL"))
    return _client[os.getenv("MONGODB_DB_NAME", "krishisahyog")]


def close_mongo_client() -> None:
    global _client
    if _client is not None:
        _client.close()
        _client = None
