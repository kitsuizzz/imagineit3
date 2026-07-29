from .db import init_db, get_db, get_db_url, get_base, db_session

__all__ = ["init_db", "get_db", "get_db_url", "get_base", "db_session"]


from dotenv import load_dotenv

load_dotenv()
init_db()
