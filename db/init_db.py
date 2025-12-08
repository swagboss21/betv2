#!/usr/bin/env python3
"""Initialize the PostgreSQL database with schema."""
import os
import sys
from pathlib import Path

import psycopg2

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://postgres:brain123@localhost:5432/brain")
SCHEMA_PATH = Path(__file__).parent / "schema.sql"


def init_db(force: bool = False):
    """Initialize database with schema.

    Args:
        force: If True, drop and recreate tables
    """
    print(f"Connecting to database...")
    conn = psycopg2.connect(DATABASE_URL)
    cursor = conn.cursor()

    if force:
        print("Dropping existing tables...")
        cursor.execute("""
            DROP TABLE IF EXISTS bets CASCADE;
            DROP TABLE IF EXISTS projections CASCADE;
            DROP TABLE IF EXISTS injuries CASCADE;
            DROP TABLE IF EXISTS users CASCADE;
            DROP TABLE IF EXISTS games CASCADE;
            DROP TABLE IF EXISTS house_bots CASCADE;
        """)
        conn.commit()

    print(f"Reading schema from {SCHEMA_PATH}...")
    schema_sql = SCHEMA_PATH.read_text()

    print("Executing schema...")
    cursor.execute(schema_sql)
    conn.commit()

    # Verify tables created
    cursor.execute("""
        SELECT table_name FROM information_schema.tables
        WHERE table_schema = 'public'
        ORDER BY table_name
    """)
    tables = [row[0] for row in cursor.fetchall()]

    print(f"\nCreated {len(tables)} tables:")
    for table in tables:
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        count = cursor.fetchone()[0]
        print(f"  - {table}: {count} rows")

    cursor.close()
    conn.close()
    print("\nDatabase initialized successfully!")


if __name__ == "__main__":
    force = "--force" in sys.argv
    if force:
        confirm = input("This will DROP all existing tables. Continue? [y/N] ")
        if confirm.lower() != 'y':
            print("Aborted.")
            sys.exit(0)

    init_db(force=force)
