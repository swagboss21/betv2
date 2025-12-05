"""
Feature transformer for converting live NBA data into model-ready features.

Uses nba_api to fetch fresh player and team statistics.
"""

import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np

from nba_api.stats.endpoints import (
    PlayerGameLog,
    TeamGameLog,
    CommonTeamRoster,
    LeagueGameFinder
)
from nba_api.stats.static import teams, players

from .models import PlayerFeatures, GameContext


# Team abbreviation mapping (nba_api uses different abbrevs sometimes)
TEAM_ABBREV_MAP = {
    "PHX": "PHO",  # Phoenix
    "BKN": "BRK",  # Brooklyn
    "CHA": "CHO",  # Charlotte
    "NOP": "NOP",  # New Orleans
}


def get_team_id(abbrev: str) -> int:
    """Get team ID from abbreviation."""
    # Check mapping
    mapped = TEAM_ABBREV_MAP.get(abbrev, abbrev)

    for team in teams.get_teams():
        if team["abbreviation"] == abbrev or team["abbreviation"] == mapped:
            return team["id"]

    raise ValueError(f"Unknown team abbreviation: {abbrev}")


def get_team_abbrev(team_id: int) -> str:
    """Get team abbreviation from ID."""
    for team in teams.get_teams():
        if team["id"] == team_id:
            return team["abbreviation"]
    return "UNK"


def find_player_id(player_name: str) -> Optional[int]:
    """Find player ID by name (fuzzy match)."""
    player_name_lower = player_name.lower()

    for player in players.get_active_players():
        full_name = player["full_name"].lower()
        if full_name == player_name_lower:
            return player["id"]

    # Try partial match
    for player in players.get_active_players():
        full_name = player["full_name"].lower()
        if player_name_lower in full_name or full_name in player_name_lower:
            return player["id"]

    return None


class FeatureTransformer:
    """Transforms live NBA data into model-ready features."""

    # Current NBA season
    CURRENT_SEASON = "2024-25"

    def __init__(self, request_delay: float = 0.6):
        """
        Initialize transformer.

        Args:
            request_delay: Delay between API requests to avoid rate limiting
        """
        self.request_delay = request_delay
        self._player_cache: Dict[int, dict] = {}
        self._team_cache: Dict[int, dict] = {}

    def _api_delay(self):
        """Sleep to avoid rate limiting."""
        time.sleep(self.request_delay)

    def get_player_game_logs(
        self,
        player_id: int,
        n_games: int = 10
    ) -> List[dict]:
        """Fetch recent game logs for a player."""
        self._api_delay()

        try:
            logs = PlayerGameLog(
                player_id=player_id,
                season=self.CURRENT_SEASON,
                season_type_all_star="Regular Season"
            )
            df = logs.get_data_frames()[0]

            if df.empty:
                return []

            # Take most recent n games
            df = df.head(n_games)

            return df.to_dict("records")

        except Exception as e:
            print(f"Error fetching player {player_id} logs: {e}")
            return []

    def get_team_game_logs(
        self,
        team_id: int,
        n_games: int = 10
    ) -> List[dict]:
        """Fetch recent game logs for a team."""
        self._api_delay()

        try:
            logs = TeamGameLog(
                team_id=team_id,
                season=self.CURRENT_SEASON,
                season_type_all_star="Regular Season"
            )
            df = logs.get_data_frames()[0]

            if df.empty:
                return []

            df = df.head(n_games)
            return df.to_dict("records")

        except Exception as e:
            print(f"Error fetching team {team_id} logs: {e}")
            return []

    def get_team_roster(self, team_abbrev: str) -> List[dict]:
        """Fetch current roster for a team."""
        self._api_delay()

        try:
            team_id = get_team_id(team_abbrev)
            roster = CommonTeamRoster(
                team_id=team_id,
                season=self.CURRENT_SEASON
            )
            df = roster.get_data_frames()[0]
            return df.to_dict("records")

        except Exception as e:
            print(f"Error fetching roster for {team_abbrev}: {e}")
            return []

    def calculate_player_features(
        self,
        player_id: int,
        player_name: str,
        team_id: int,
        team_abbrev: str
    ) -> Optional[PlayerFeatures]:
        """
        Calculate all features for a player from their recent games.

        Returns None if insufficient data.
        """
        games = self.get_player_game_logs(player_id, n_games=10)

        if len(games) < 3:
            return None

        # L5 averages (most recent 5 games)
        l5_games = games[:5]

        def avg(key: str, game_list: List[dict]) -> float:
            vals = [g.get(key, 0) or 0 for g in game_list]
            return sum(vals) / len(vals) if vals else 0.0

        min_l5 = avg("MIN", l5_games)
        pts_l5 = avg("PTS", l5_games)
        reb_l5 = avg("REB", l5_games)
        ast_l5 = avg("AST", l5_games)
        stl_l5 = avg("STL", l5_games)
        blk_l5 = avg("BLK", l5_games)
        tov_l5 = avg("TOV", l5_games)
        fg3m_l5 = avg("FG3M", l5_games)

        # Season averages (all games)
        min_szn = avg("MIN", games)
        pts_szn = avg("PTS", games)

        games_played = len(games)

        # Determine if starter based on minutes
        is_starter = 1 if min_szn >= 20 else 0

        # Calculate usage proxy: (FGA + 0.44*FTA + TOV) / MIN
        # This matches the training data calculation
        fga_avg = avg("FGA", games)
        fta_avg = avg("FTA", games)
        tov_avg = avg("TOV", games)
        if min_szn > 0:
            usage_proxy = (fga_avg + 0.44 * fta_avg + tov_avg) / min_szn
        else:
            usage_proxy = 0.0

        # Team points share: player_pts / team_pts
        # Approximate using league average team PPG (~115)
        team_pts_share = pts_szn / 115.0

        # Determine tier
        if min_szn >= 30:
            tier = "Stars"
        elif min_szn >= 20:
            tier = "Starters"
        elif min_szn >= 10:
            tier = "Rotation"
        else:
            tier = "Bench"

        return PlayerFeatures(
            player_id=player_id,
            player_name=player_name,
            team_id=team_id,
            team_abbr=team_abbrev,
            min_L5_avg=min_l5,
            min_szn_avg=min_szn,
            is_starter=is_starter,
            games_played_szn=games_played,
            player_usage_proxy=usage_proxy,
            player_team_pts_share=team_pts_share,
            pts_L5_avg=pts_l5,
            reb_L5_avg=reb_l5,
            ast_L5_avg=ast_l5,
            stl_L5_avg=stl_l5,
            blk_L5_avg=blk_l5,
            tov_L5_avg=tov_l5,
            fg3m_L5_avg=fg3m_l5,
            tier=tier
        )

    def get_player_features(self, player_name: str) -> Optional[PlayerFeatures]:
        """
        Get features for a player by name.

        Returns None if player not found or insufficient data.
        """
        player_id = find_player_id(player_name)
        if not player_id:
            print(f"Player not found: {player_name}")
            return None

        # Get player info for team
        for p in players.get_active_players():
            if p["id"] == player_id:
                # Need to look up current team from game logs
                games = self.get_player_game_logs(player_id, n_games=1)
                if games:
                    team_abbrev = games[0].get("MATCHUP", "")[:3]
                    team_id = get_team_id(team_abbrev)
                    return self.calculate_player_features(
                        player_id, player_name, team_id, team_abbrev
                    )

        return None

    def get_roster(self, team_abbrev: str) -> List[PlayerFeatures]:
        """Get features for all players on a team's roster."""
        roster_data = self.get_team_roster(team_abbrev)
        team_id = get_team_id(team_abbrev)

        player_features = []

        for player in roster_data:
            player_id = player.get("PLAYER_ID")
            player_name = player.get("PLAYER", "")

            if not player_id:
                continue

            features = self.calculate_player_features(
                player_id, player_name, team_id, team_abbrev
            )

            if features:
                player_features.append(features)

        return player_features

    def get_team_stats(self, team_abbrev: str) -> dict:
        """Get L10 team statistics."""
        team_id = get_team_id(team_abbrev)
        games = self.get_team_game_logs(team_id, n_games=10)

        if not games:
            # Return league averages as fallback
            return {
                "pts_L10": 115.0,
                "opp_pts_L10": 115.0,
                "pace_L10": 100.0,
                "fg_pct_L10": 0.46
            }

        def avg(key: str) -> float:
            vals = [g.get(key, 0) or 0 for g in games]
            return sum(vals) / len(vals) if vals else 0.0

        pts = avg("PTS")
        opp_pts = avg("PLUS_MINUS")  # Approximate opponent pts
        opp_pts = pts - opp_pts if opp_pts else 115.0

        fg_pct = avg("FG_PCT")
        # Pace approximation: possessions ~ FGA + 0.44*FTA - OREB + TOV
        fga = avg("FGA")
        fta = avg("FTA")
        oreb = avg("OREB")
        tov = avg("TOV")
        pace = fga + 0.44 * fta - oreb + tov

        return {
            "pts_L10": pts,
            "opp_pts_L10": opp_pts,
            "pace_L10": pace,
            "fg_pct_L10": fg_pct
        }

    def get_team_rest_days(self, team_abbrev: str) -> int:
        """Calculate rest days since last game."""
        team_id = get_team_id(team_abbrev)
        games = self.get_team_game_logs(team_id, n_games=1)

        if not games:
            return 1  # Default

        last_game_date_str = games[0].get("GAME_DATE", "")

        try:
            # Parse date (format: "DEC 04, 2024")
            last_game_date = datetime.strptime(last_game_date_str, "%b %d, %Y")
            today = datetime.now()
            rest_days = (today - last_game_date).days - 1
            return max(0, min(rest_days, 7))  # Cap at 7

        except Exception:
            return 1

    def get_game_context(
        self,
        home_team: str,
        away_team: str,
        game_date: str = None
    ) -> GameContext:
        """Build game context from team features."""
        if game_date is None:
            game_date = datetime.now().strftime("%Y-%m-%d")

        home_stats = self.get_team_stats(home_team)
        away_stats = self.get_team_stats(away_team)

        home_rest = self.get_team_rest_days(home_team)
        away_rest = self.get_team_rest_days(away_team)

        return GameContext(
            home_team_abbr=home_team,
            away_team_abbr=away_team,
            game_date=game_date,
            home_pts_L10=home_stats["pts_L10"],
            away_pts_L10=away_stats["pts_L10"],
            home_opp_pts_L10=home_stats["opp_pts_L10"],
            away_opp_pts_L10=away_stats["opp_pts_L10"],
            home_pace_L10=home_stats["pace_L10"],
            away_pace_L10=away_stats["pace_L10"],
            home_fg_pct_L10=home_stats["fg_pct_L10"],
            away_fg_pct_L10=away_stats["fg_pct_L10"],
            home_rest_days=home_rest,
            away_rest_days=away_rest
        )

    def build_minutes_features(
        self,
        player: PlayerFeatures,
        rest_days: int,
        is_home: bool
    ) -> np.ndarray:
        """
        Build feature array for minutes model.

        Order: min_L5_avg, min_szn_avg, is_starter, games_played_szn, rest_days, is_home
        """
        return np.array([
            player.min_L5_avg,
            player.min_szn_avg,
            player.is_starter,
            player.games_played_szn,
            rest_days,
            1 if is_home else 0
        ])

    def build_stats_features(
        self,
        player: PlayerFeatures,
        stat_name: str,
        opp_pts_allowed_L10: float,
        is_home: bool
    ) -> np.ndarray:
        """
        Build feature array for stats model.

        Order: min_L5_avg, player_usage_proxy, player_team_pts_share,
               opp_pts_allowed_L10, is_home, {stat}_L5_avg
        """
        # Get the stat-specific L5 average
        stat_l5_map = {
            "pts": player.pts_L5_avg,
            "reb": player.reb_L5_avg,
            "ast": player.ast_L5_avg,
            "stl": player.stl_L5_avg,
            "blk": player.blk_L5_avg,
            "tov": player.tov_L5_avg,
            "fg3m": player.fg3m_L5_avg
        }

        stat_l5 = stat_l5_map.get(stat_name, 0.0)

        return np.array([
            player.min_L5_avg,
            player.player_usage_proxy,
            player.player_team_pts_share,
            opp_pts_allowed_L10,
            1 if is_home else 0,
            stat_l5
        ])
