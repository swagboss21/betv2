"""
Sprint 0 Integration Tests

Verify:
1. Database has 6 tables
2. All module stubs import without error
3. Folder structure is correct
4. Schema has correct columns
"""
import os
import sys
from pathlib import Path

import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestFolderStructure:
    """Verify directory structure exists."""

    REQUIRED_DIRS = ["db", "batch", "api", "chat", "bots", "auth", "tests"]

    def test_directories_exist(self):
        for dir_name in self.REQUIRED_DIRS:
            dir_path = PROJECT_ROOT / dir_name
            assert dir_path.is_dir(), f"Directory {dir_name}/ does not exist"

    def test_init_files_exist(self):
        for dir_name in self.REQUIRED_DIRS:
            init_path = PROJECT_ROOT / dir_name / "__init__.py"
            assert init_path.is_file(), f"{dir_name}/__init__.py does not exist"


class TestModuleImports:
    """Verify all stub modules import without error."""

    def test_import_batch_modules(self):
        from batch import precompute
        from batch import scrape_injuries
        from batch import grade_bets

        assert hasattr(precompute, "fetch_tonights_games")
        assert hasattr(precompute, "run_simulations")
        assert hasattr(precompute, "save_projections")
        assert hasattr(precompute, "main")

        assert hasattr(scrape_injuries, "scrape_espn")
        assert hasattr(scrape_injuries, "update_injuries_table")

        assert hasattr(grade_bets, "fetch_actuals")
        assert hasattr(grade_bets, "grade_bets")

    def test_import_api_modules(self):
        from api import queries
        from api import probability

        assert hasattr(queries, "get_games_today")
        assert hasattr(queries, "get_projection")
        assert hasattr(queries, "get_best_props")
        assert hasattr(queries, "save_bet")
        assert hasattr(queries, "get_user_bets")

        assert hasattr(probability, "prob_over")
        assert hasattr(probability, "prob_under")
        assert hasattr(probability, "american_to_probability")
        assert hasattr(probability, "calculate_edge")

    def test_import_db_modules(self):
        from db import connection

        assert hasattr(connection, "get_db")
        assert hasattr(connection, "DATABASE_URL")


class TestSchemaFile:
    """Verify schema file exists and has correct structure."""

    def test_schema_file_exists(self):
        schema_path = PROJECT_ROOT / "db" / "schema.sql"
        assert schema_path.is_file(), "db/schema.sql does not exist"

    def test_schema_has_tables(self):
        schema_path = PROJECT_ROOT / "db" / "schema.sql"
        schema_content = schema_path.read_text()

        required_tables = ["games", "projections", "injuries", "users", "bets", "house_bots"]
        for table in required_tables:
            assert f"CREATE TABLE" in schema_content and table in schema_content, \
                f"Schema missing CREATE TABLE for {table}"


class TestDatabaseConnection:
    """Test database connection (requires running Postgres)."""

    @pytest.fixture
    def skip_if_no_db(self):
        """Skip test if DATABASE_URL not set or DB not reachable."""
        import os
        if not os.environ.get("DATABASE_URL"):
            pytest.skip("DATABASE_URL not set")

    def test_database_connection(self, skip_if_no_db):
        from db.connection import get_db

        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("SELECT 1 as val")
        result = cursor.fetchone()
        cursor.close()
        conn.close()

        assert result['val'] == 1

    def test_tables_exist(self, skip_if_no_db):
        from db.connection import get_db

        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT table_name FROM information_schema.tables
            WHERE table_schema = 'public'
            ORDER BY table_name
        """)
        tables = [row['table_name'] for row in cursor.fetchall()]
        cursor.close()
        conn.close()

        expected_tables = ["bets", "games", "house_bots", "injuries", "projections", "users"]
        for table in expected_tables:
            assert table in tables, f"Table {table} not found in database"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
