"""
Data models for the Monte Carlo simulation engine.

These dataclasses define the interfaces between components.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np


@dataclass
class PlayerFeatures:
    """Player features for model input."""
    player_id: int
    player_name: str
    team_id: int
    team_abbr: str

    # Minutes features
    min_L5_avg: float
    min_szn_avg: float
    is_starter: int
    games_played_szn: int

    # Stats features
    player_usage_proxy: float
    player_team_pts_share: float
    pts_L5_avg: float
    reb_L5_avg: float
    ast_L5_avg: float
    stl_L5_avg: float
    blk_L5_avg: float
    tov_L5_avg: float
    fg3m_L5_avg: float

    # Tier for variance lookup
    tier: str = "Rotation"  # Stars, Starters, Rotation, Bench

    def get_tier(self) -> str:
        """Determine player tier based on minutes."""
        if self.min_szn_avg >= 30:
            return "Stars"
        elif self.min_szn_avg >= 20:
            return "Starters"
        elif self.min_szn_avg >= 10:
            return "Rotation"
        else:
            return "Bench"


@dataclass
class GameContext:
    """Game context for simulation."""
    home_team_abbr: str
    away_team_abbr: str
    game_date: str

    # Team L10 stats
    home_pts_L10: float
    away_pts_L10: float
    home_opp_pts_L10: float  # Points allowed
    away_opp_pts_L10: float
    home_pace_L10: float
    away_pace_L10: float
    home_fg_pct_L10: float
    away_fg_pct_L10: float

    # Rest days
    home_rest_days: int
    away_rest_days: int

    def build_game_features(self) -> np.ndarray:
        """Build feature array for game model (10 features)."""
        return np.array([
            self.home_pts_L10,
            self.home_opp_pts_L10,
            self.home_fg_pct_L10,
            self.home_pace_L10,
            self.home_rest_days,
            self.away_pts_L10,
            self.away_opp_pts_L10,
            self.away_fg_pct_L10,
            self.away_pace_L10,
            self.away_rest_days
        ])


@dataclass
class StatDistribution:
    """Distribution of a simulated statistic."""
    mean: float
    std: float
    p10: float
    p25: float
    p50: float  # median
    p75: float
    p90: float

    @classmethod
    def from_simulations(cls, values: np.ndarray) -> "StatDistribution":
        """Create distribution from simulation array."""
        return cls(
            mean=float(np.mean(values)),
            std=float(np.std(values)),
            p10=float(np.percentile(values, 10)),
            p25=float(np.percentile(values, 25)),
            p50=float(np.percentile(values, 50)),
            p75=float(np.percentile(values, 75)),
            p90=float(np.percentile(values, 90))
        )

    def to_dict(self) -> dict:
        return {
            "mean": round(self.mean, 2),
            "std": round(self.std, 2),
            "p10": round(self.p10, 1),
            "p25": round(self.p25, 1),
            "p50": round(self.p50, 1),
            "p75": round(self.p75, 1),
            "p90": round(self.p90, 1)
        }


@dataclass
class PlayerPrediction:
    """Prediction results for a single player."""
    player_id: int
    player_name: str
    team_abbr: str

    minutes: StatDistribution
    pts: StatDistribution
    reb: StatDistribution
    ast: StatDistribution
    stl: StatDistribution
    blk: StatDistribution
    tov: StatDistribution
    fg3m: StatDistribution

    def to_dict(self) -> dict:
        return {
            "player_id": self.player_id,
            "player_name": self.player_name,
            "team_abbr": self.team_abbr,
            "minutes": self.minutes.to_dict(),
            "pts": self.pts.to_dict(),
            "reb": self.reb.to_dict(),
            "ast": self.ast.to_dict(),
            "stl": self.stl.to_dict(),
            "blk": self.blk.to_dict(),
            "tov": self.tov.to_dict(),
            "fg3m": self.fg3m.to_dict()
        }


@dataclass
class PropBet:
    """A player prop bet."""
    player_name: str
    stat: str  # pts, reb, ast, etc.
    line: float
    over_odds: str = "-110"
    under_odds: str = "-110"


@dataclass
class PropAnalysis:
    """Analysis result for a prop bet."""
    prop: PropBet
    model_prob_over: float
    model_prob_under: float
    implied_prob_over: float
    implied_prob_under: float
    edge_over: float
    edge_under: float
    recommendation: str  # "OVER", "UNDER", "PASS"
    confidence: str  # "HIGH", "MEDIUM", "LOW"

    # Distribution info
    predicted_mean: float
    predicted_std: float

    def to_dict(self) -> dict:
        return {
            "player": self.prop.player_name,
            "stat": self.prop.stat,
            "line": self.prop.line,
            "over_odds": self.prop.over_odds,
            "under_odds": self.prop.under_odds,
            "model_prob_over": round(self.model_prob_over, 4),
            "model_prob_under": round(self.model_prob_under, 4),
            "implied_prob_over": round(self.implied_prob_over, 4),
            "implied_prob_under": round(self.implied_prob_under, 4),
            "edge_over": round(self.edge_over, 4),
            "edge_under": round(self.edge_under, 4),
            "recommendation": self.recommendation,
            "confidence": self.confidence,
            "predicted_mean": round(self.predicted_mean, 2),
            "predicted_std": round(self.predicted_std, 2)
        }


@dataclass
class SimulationResult:
    """Full game simulation result."""
    home_team: str
    away_team: str
    game_date: str
    n_simulations: int

    # Score distributions
    home_score: StatDistribution
    away_score: StatDistribution

    # Player predictions keyed by player_name
    players: Dict[str, PlayerPrediction] = field(default_factory=dict)

    # Prop analyses if requested
    prop_analyses: List[PropAnalysis] = field(default_factory=list)

    # Injury scenarios (e.g., {"if_ad_out": {...}})
    scenarios: Dict[str, dict] = field(default_factory=dict)

    # Raw simulation arrays for empirical probability calculation
    # Structure: {player_name: {stat: np.ndarray of 10K values}}
    raw_simulations: Dict[str, Dict[str, np.ndarray]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "game": {
                "home_team": self.home_team,
                "away_team": self.away_team,
                "game_date": self.game_date,
                "n_simulations": self.n_simulations
            },
            "scores": {
                "home": self.home_score.to_dict(),
                "away": self.away_score.to_dict()
            },
            "players": {
                name: pred.to_dict()
                for name, pred in self.players.items()
            },
            "props": [p.to_dict() for p in self.prop_analyses],
            "scenarios": self.scenarios
        }
