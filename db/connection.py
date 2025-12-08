"""Database connection utilities for The Brain."""
import os
from contextlib import contextmanager

import psycopg2
from psycopg2.extras import RealDictCursor

DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://postgres:brain123@localhost:5432/brain")


def get_db():
    """Get a database connection with RealDictCursor."""
    conn = psycopg2.connect(DATABASE_URL)
    conn.cursor_factory = RealDictCursor
    return conn


@contextmanager
def get_db_cursor():
    """Context manager for database operations."""
    conn = get_db()
    try:
        cursor = conn.cursor()
        yield cursor
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        cursor.close()
        conn.close()
