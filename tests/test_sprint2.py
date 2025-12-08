#!/usr/bin/env python3
"""
Sprint 2 Integration Tests - API Layer

Tests probability math, database queries, and CRUD operations.
Run: pytest tests/test_sprint2.py -v
"""
import os
import pytest
from datetime import datetime

# Skip all tests if DATABASE_URL not set
pytestmark = pytest.mark.skipif(
    not os.environ.get("DATABASE_URL"),
    reason="DATABASE_URL not set"
)


class TestProbabilityMath:
    """S2-T1: Test probability calculation functions."""

    def test_prob_over_basic(self):
        """prob_over returns expected value for known inputs."""
        from api.probability import prob_over
        # mean=26.3, std=5.2, line=25.5 should give ~56%
        p = prob_over(26.3, 5.2, 25.5)
        assert 0.55 < p < 0.60, f"Expected ~0.56, got {p}"

    def test_prob_over_goblin_line(self):
        """Lower line (Goblin style) gives higher probability."""
        from api.probability import prob_over
        p = prob_over(26.3, 5.2, 22.5)
        assert p > 0.70, f"Goblin line should be >70%, got {p}"

    def test_prob_over_demon_line(self):
        """Higher line (Demon style) gives lower probability."""
        from api.probability import prob_over
        p = prob_over(26.3, 5.2, 30.5)
        assert p < 0.30, f"Demon line should be <30%, got {p}"

    def test_prob_over_plus_under_equals_one(self):
        """prob_over + prob_under should equal 1.0."""
        from api.probability import prob_over, prob_under
        p_over = prob_over(26.3, 5.2, 25.5)
        p_under = prob_under(26.3, 5.2, 25.5)
        total = p_over + p_under
        assert abs(total - 1.0) < 0.001, f"Sum should be 1.0, got {total}"

    def test_prob_over_zero_std(self):
        """Edge case: zero std returns deterministic result."""
        from api.probability import prob_over
        assert prob_over(26.3, 0, 25.5) == 1.0  # mean > line
        assert prob_over(26.3, 0, 27.0) == 0.0  # mean < line

    def test_american_to_probability_negative(self):
        """-115 converts to ~53.5%."""
        from api.probability import american_to_probability
        p = american_to_probability("-115")
        assert 0.53 < p < 0.54, f"Expected ~0.535, got {p}"

    def test_american_to_probability_positive(self):
        """+150 converts to 40%."""
        from api.probability import american_to_probability
        p = american_to_probability("+150")
        assert abs(p - 0.40) < 0.01, f"Expected 0.40, got {p}"

    def test_calculate_edge(self):
        """Edge calculation is simple subtraction."""
        from api.probability import calculate_edge
        edge = calculate_edge(0.56, 0.524)
        assert abs(edge - 0.036) < 0.001, f"Expected 0.036, got {edge}"

    def test_format_edge_positive(self):
        """format_edge shows +X% for positive edge >= 5%."""
        from api.probability import format_edge
        assert format_edge(0.08) == "+8.0%"

    def test_format_edge_pass(self):
        """format_edge shows PASS for small edge."""
        from api.probability import format_edge
        assert format_edge(0.02) == "PASS"

    def test_devig_odds_sums_to_one(self):
        """Devigged odds sum to 1.0."""
        from api.probability import devig_odds
        over, under = devig_odds("-115", "-105")
        total = over + under
        assert abs(total - 1.0) < 0.001, f"Devig should sum to 1.0, got {total}"


class TestProjectionQueries:
    """S2-T2: Test database projection queries."""

    def test_get_games_today_returns_list(self):
        """get_games_today returns a list."""
        from api.queries import get_games_today
        games = get_games_today()
        assert isinstance(games, list)

    def test_get_projection_returns_dict_or_none(self):
        """get_projection returns dict or None."""
        from api.queries import get_projection
        proj = get_projection("LeBron James", "pts")
        assert proj is None or isinstance(proj, dict)

    def test_get_projection_with_line_has_probabilities(self):
        """get_projection with line parameter includes prob_over/prob_under."""
        from api.queries import get_projection
        proj = get_projection("Stephen Curry", "pts", line=25.5)
        if proj:
            assert "prob_over" in proj, "Should have prob_over"
            assert "prob_under" in proj, "Should have prob_under"
            assert "line" in proj, "Should have line"
            assert proj["line"] == 25.5

    def test_get_all_projections_returns_list(self):
        """get_all_projections returns a list."""
        from api.queries import get_all_projections, get_games_today
        games = get_games_today()
        if games:
            projections = get_all_projections(games[0]["id"])
            assert isinstance(projections, list)

    def test_get_injuries_returns_list(self):
        """get_injuries returns a list."""
        from api.queries import get_injuries
        injuries = get_injuries("LAL")
        assert isinstance(injuries, list)


class TestBestProps:
    """S2-T3: Test best props detection."""

    def test_get_best_props_returns_list(self):
        """get_best_props returns a list."""
        from api.queries import get_best_props
        props = get_best_props(min_edge=0.01, limit=5)
        assert isinstance(props, list)

    def test_get_best_props_sorted_by_edge(self):
        """get_best_props is sorted by edge descending."""
        from api.queries import get_best_props
        props = get_best_props(min_edge=0.01, limit=10)
        if len(props) > 1:
            edges = [p["edge"] for p in props]
            assert edges == sorted(edges, reverse=True), "Should be sorted by edge desc"

    def test_get_best_props_respects_limit(self):
        """get_best_props respects limit parameter."""
        from api.queries import get_best_props
        props = get_best_props(min_edge=0.0, limit=3)
        assert len(props) <= 3

    def test_get_best_props_has_required_fields(self):
        """get_best_props results have required fields."""
        from api.queries import get_best_props
        props = get_best_props(min_edge=0.01, limit=1)
        if props:
            required = ["player_name", "stat_type", "game_id", "mean", "std",
                       "line", "direction", "probability", "edge"]
            for field in required:
                assert field in props[0], f"Missing field: {field}"


class TestBetOperations:
    """S2-T4: Test bet CRUD operations."""

    @pytest.fixture
    def test_user(self):
        """Create a test user for bet operations."""
        from api.queries import get_or_create_user
        return get_or_create_user(f"test-bet-{datetime.now().timestamp()}@example.com")

    def test_save_bet_returns_id(self, test_user):
        """save_bet returns a positive bet ID."""
        from api.queries import save_bet, get_games_today

        games = get_games_today()
        if not games:
            pytest.skip("No games today")

        bet_id = save_bet(test_user["id"], {
            "game_id": games[0]["id"],
            "player_name": "Test Player",
            "stat_type": "pts",
            "line": 20.5,
            "direction": "OVER",
            "odds": "-110",
            "edge_pct": 0.05
        })
        assert bet_id > 0, f"Expected positive ID, got {bet_id}"

    def test_get_user_bets_returns_list(self, test_user):
        """get_user_bets returns a list."""
        from api.queries import get_user_bets
        bets = get_user_bets(test_user["id"])
        assert isinstance(bets, list)

    def test_get_user_record_has_required_fields(self, test_user):
        """get_user_record returns dict with required fields."""
        from api.queries import get_user_record
        record = get_user_record(test_user["id"])
        assert "total" in record
        assert "wins" in record
        assert "losses" in record
        assert "pushes" in record
        assert "win_rate" in record


class TestUserOperations:
    """S2-T5: Test user CRUD operations."""

    def test_create_user_returns_uuid(self):
        """create_user returns a UUID string."""
        from api.queries import create_user
        user_id = create_user(f"test-{datetime.now().timestamp()}@example.com")
        assert user_id is not None
        assert len(user_id) == 36  # UUID format

    def test_get_user_returns_dict_or_none(self):
        """get_user returns dict or None."""
        from api.queries import get_user
        user = get_user("nonexistent@example.com")
        assert user is None or isinstance(user, dict)

    def test_get_or_create_user_creates_new(self):
        """get_or_create_user creates user if not exists."""
        from api.queries import get_or_create_user
        email = f"test-new-{datetime.now().timestamp()}@example.com"
        user = get_or_create_user(email)
        assert user["email"] == email

    def test_get_or_create_user_returns_existing(self):
        """get_or_create_user returns existing user."""
        from api.queries import get_or_create_user
        email = f"test-existing-{datetime.now().timestamp()}@example.com"
        user1 = get_or_create_user(email)
        user2 = get_or_create_user(email)
        assert user1["id"] == user2["id"]

    def test_increment_message_count(self):
        """increment_message_count increases count."""
        from api.queries import get_or_create_user, increment_message_count
        user = get_or_create_user(f"test-count-{datetime.now().timestamp()}@example.com")
        new_count = increment_message_count(user["id"])
        assert new_count >= 1

    def test_is_user_paid_default_false(self):
        """New users are not paid by default."""
        from api.queries import get_or_create_user, is_user_paid
        user = get_or_create_user(f"test-paid-{datetime.now().timestamp()}@example.com")
        assert is_user_paid(user["id"]) == False

    def test_set_user_paid(self):
        """set_user_paid updates paid status."""
        from api.queries import get_or_create_user, set_user_paid, is_user_paid
        user = get_or_create_user(f"test-setpaid-{datetime.now().timestamp()}@example.com")
        set_user_paid(user["id"], True)
        assert is_user_paid(user["id"]) == True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
