"""
Injury adjustment module for redistributing minutes and usage.

When key players are OUT, remaining players get more minutes and usage.
"""

from typing import Dict, List
from copy import deepcopy
from .models import PlayerFeatures


class InjuryAdjustmentModule:
    """Adjusts player features based on injury report."""

    # Minutes boost factors when a player is out
    # Key: tier of injured player, Value: boost factors by tier of healthy player
    MINUTES_BOOST = {
        "Stars": {      # When a star (30+ min) is OUT
            "Stars": 1.08,       # Other stars get +8%
            "Starters": 1.12,    # Starters get +12%
            "Rotation": 1.20,    # Rotation gets +20%
            "Bench": 1.35        # Bench gets +35%
        },
        "Starters": {   # When a starter (20-30 min) is OUT
            "Stars": 1.03,
            "Starters": 1.08,
            "Rotation": 1.15,
            "Bench": 1.25
        },
        "Rotation": {   # When rotation player (10-20 min) is OUT
            "Stars": 1.00,
            "Starters": 1.02,
            "Rotation": 1.10,
            "Bench": 1.15
        },
        "Bench": {      # When bench player (<10 min) is OUT
            "Stars": 1.00,
            "Starters": 1.00,
            "Rotation": 1.02,
            "Bench": 1.08
        }
    }

    # Usage boost factors (affects pts_share, usage_proxy)
    USAGE_BOOST = {
        "Stars": {
            "Stars": 1.10,       # Other stars see +10% usage
            "Starters": 1.08,
            "Rotation": 1.05,
            "Bench": 1.02
        },
        "Starters": {
            "Stars": 1.05,
            "Starters": 1.06,
            "Rotation": 1.04,
            "Bench": 1.02
        },
        "Rotation": {
            "Stars": 1.02,
            "Starters": 1.02,
            "Rotation": 1.03,
            "Bench": 1.02
        },
        "Bench": {
            "Stars": 1.00,
            "Starters": 1.00,
            "Rotation": 1.01,
            "Bench": 1.02
        }
    }

    # Stat boost factors (L5 averages increase when usage increases)
    STAT_BOOST_FROM_USAGE = {
        "pts_L5_avg": 0.8,    # 80% of usage boost applies to points
        "reb_L5_avg": 0.3,    # 30% to rebounds
        "ast_L5_avg": 0.5,    # 50% to assists
        "stl_L5_avg": 0.1,
        "blk_L5_avg": 0.1,
        "tov_L5_avg": 0.4,    # More usage = more turnovers
        "fg3m_L5_avg": 0.6
    }

    def __init__(self):
        self.injury_report: Dict[str, str] = {}

    def adjust_roster(
        self,
        roster: List[PlayerFeatures],
        injuries: Dict[str, str]
    ) -> List[PlayerFeatures]:
        """
        Adjust roster features based on injuries.

        Args:
            roster: List of PlayerFeatures for all players
            injuries: Dict mapping player_name to status ("OUT", "QUESTIONABLE", etc.)

        Returns:
            Adjusted roster with boosted minutes/usage for healthy players
        """
        if not injuries:
            return roster

        # Deep copy to avoid modifying original
        adjusted_roster = [deepcopy(p) for p in roster]

        # Identify OUT players and their tiers
        out_players = []
        healthy_players = []

        for player in adjusted_roster:
            status = injuries.get(player.player_name, "").upper()
            if status == "OUT":
                out_players.append(player)
            elif status != "QUESTIONABLE":  # QUESTIONABLE still plays
                healthy_players.append(player)

        # Apply boosts for each OUT player
        for out_player in out_players:
            out_tier = out_player.get_tier()
            minutes_boosts = self.MINUTES_BOOST.get(out_tier, {})
            usage_boosts = self.USAGE_BOOST.get(out_tier, {})

            for healthy in healthy_players:
                # Only boost teammates (same team)
                if healthy.team_abbr != out_player.team_abbr:
                    continue

                healthy_tier = healthy.get_tier()

                # Apply minutes boost
                minutes_factor = minutes_boosts.get(healthy_tier, 1.0)
                healthy.min_L5_avg *= minutes_factor
                healthy.min_szn_avg *= minutes_factor

                # Cap at 42 minutes (realistic max for heavy workload)
                healthy.min_L5_avg = min(healthy.min_L5_avg, 42.0)
                healthy.min_szn_avg = min(healthy.min_szn_avg, 42.0)

                # Apply usage boost
                usage_factor = usage_boosts.get(healthy_tier, 1.0)
                healthy.player_usage_proxy *= usage_factor
                healthy.player_team_pts_share *= usage_factor

                # Cap usage at reasonable maxes
                healthy.player_usage_proxy = min(healthy.player_usage_proxy, 0.40)
                healthy.player_team_pts_share = min(healthy.player_team_pts_share, 0.35)

                # Boost stat averages proportionally
                usage_boost_pct = usage_factor - 1.0  # e.g., 0.10 for 10% boost
                for stat, stat_factor in self.STAT_BOOST_FROM_USAGE.items():
                    if hasattr(healthy, stat):
                        current = getattr(healthy, stat)
                        boost = current * usage_boost_pct * stat_factor
                        setattr(healthy, stat, current + boost)

        # Remove OUT players from roster
        return [p for p in adjusted_roster if p.player_name not in
                [op.player_name for op in out_players]]

    def get_missing_minutes(
        self,
        roster: List[PlayerFeatures],
        injuries: Dict[str, str]
    ) -> float:
        """Calculate total minutes missing from OUT players."""
        missing = 0.0
        for player in roster:
            if injuries.get(player.player_name, "").upper() == "OUT":
                missing += player.min_szn_avg
        return missing

    def get_usage_vacuum(
        self,
        roster: List[PlayerFeatures],
        injuries: Dict[str, str]
    ) -> float:
        """Calculate total usage share missing from OUT players."""
        missing = 0.0
        for player in roster:
            if injuries.get(player.player_name, "").upper() == "OUT":
                missing += player.player_team_pts_share
        return missing
