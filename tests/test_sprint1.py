"""
Sprint 1 Integration Tests

Verify:
1. fetch_tonights_games() returns games with game_id
2. run_simulations() returns projections with mean/std
3. save_projections() writes to database
4. scrape_espn() returns injuries by team
"""
import os
import sys
from pathlib import Path
from datetime import date
from unittest.mock import patch, MagicMock

import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestFetchTonightsGames:
    """Test NBA API game fetching."""

    def test_fetch_returns_list(self):
        """fetch_tonights_games returns a list."""
        from batch.precompute import fetch_tonights_games

        # This makes real API call - may return empty on off days
        games = fetch_tonights_games()
        assert isinstance(games, list)

    def test_game_structure(self):
        """Each game has required fields."""
        from batch.precompute import fetch_tonights_games

        games = fetch_tonights_games()
        if games:  # Only test if games exist
            game = games[0]
            assert "game_id" in game
            assert "home_team" in game
            assert "away_team" in game
            assert "status" in game

            # Team abbreviations should be 3 characters
            assert len(game["home_team"]) == 3
            assert len(game["away_team"]) == 3


class TestRunSimulations:
    """Test Monte Carlo simulation engine."""

    @pytest.fixture
    def sample_game(self):
        """Create a sample game dict for testing."""
        return {
            "game_id": "TEST001",
            "home_team": "LAL",
            "away_team": "BOS",
            "status": "scheduled"
        }

    def test_simulation_returns_projections(self, sample_game):
        """run_simulations returns list of projections."""
        from batch.precompute import run_simulations

        projections = run_simulations(sample_game)
        assert isinstance(projections, list)

    def test_projection_structure(self, sample_game):
        """Each projection has required fields."""
        from batch.precompute import run_simulations

        projections = run_simulations(sample_game)
        if projections:
            proj = projections[0]
            assert "game_id" in proj
            assert "player_name" in proj
            assert "stat_type" in proj
            assert "mean" in proj
            assert "std" in proj
            assert "p10" in proj
            assert "p50" in proj
            assert "p90" in proj

    def test_projection_has_valid_stats(self, sample_game):
        """Projections have valid statistical values."""
        from batch.precompute import run_simulations

        projections = run_simulations(sample_game)
        if projections:
            proj = projections[0]
            # Mean should be non-negative for counting stats
            assert proj["mean"] >= 0
            # Std should be non-negative
            assert proj["std"] >= 0
            # Percentiles should be ordered
            assert proj["p10"] <= proj["p50"] <= proj["p90"]


class TestSaveProjections:
    """Test database save functionality."""

    @pytest.fixture
    def skip_if_no_db(self):
        """Skip if DATABASE_URL not set."""
        if not os.environ.get("DATABASE_URL"):
            pytest.skip("DATABASE_URL not set")

    def test_save_empty_list(self, skip_if_no_db):
        """Saving empty list returns 0."""
        from batch.precompute import save_projections

        count = save_projections([])
        assert count == 0

    def test_database_has_projections(self, skip_if_no_db):
        """Database should have projections after precompute."""
        from db.connection import get_db

        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) as cnt FROM projections")
        count = cursor.fetchone()["cnt"]
        cursor.close()
        conn.close()

        # Should have projections from precompute run
        assert count > 0


class TestScrapeInjuries:
    """Test ESPN injury scraper."""

    def test_scrape_returns_dict(self):
        """scrape_espn returns a dictionary."""
        from batch.scrape_injuries import scrape_espn

        injuries = scrape_espn()
        assert isinstance(injuries, dict)

    def test_injury_structure(self):
        """Injuries have correct structure."""
        from batch.scrape_injuries import scrape_espn

        injuries = scrape_espn()
        if injuries:
            # Get first team's injuries
            team = list(injuries.keys())[0]
            team_injuries = injuries[team]

            # Team key should be 3-letter abbreviation
            assert len(team) == 3

            if team_injuries:
                injury = team_injuries[0]
                assert "player" in injury
                assert "status" in injury
                assert injury["status"] in ["OUT", "DOUBTFUL", "QUESTIONABLE", "PROBABLE"]


class TestIntegration:
    """Full pipeline integration tests."""

    @pytest.fixture
    def skip_if_no_db(self):
        """Skip if DATABASE_URL not set."""
        if not os.environ.get("DATABASE_URL"):
            pytest.skip("DATABASE_URL not set")

    def test_games_in_database(self, skip_if_no_db):
        """Games table has entries."""
        from db.connection import get_db

        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) as cnt FROM games")
        count = cursor.fetchone()["cnt"]
        cursor.close()
        conn.close()

        assert count > 0

    def test_injuries_in_database(self, skip_if_no_db):
        """Injuries table has entries."""
        from db.connection import get_db

        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) as cnt FROM injuries")
        count = cursor.fetchone()["cnt"]
        cursor.close()
        conn.close()

        assert count > 0

    def test_projections_have_valid_game_fk(self, skip_if_no_db):
        """All projections reference valid games."""
        from db.connection import get_db

        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT COUNT(*) as cnt FROM projections p
            LEFT JOIN games g ON p.game_id = g.id
            WHERE g.id IS NULL
        """)
        orphan_count = cursor.fetchone()["cnt"]
        cursor.close()
        conn.close()

        assert orphan_count == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
