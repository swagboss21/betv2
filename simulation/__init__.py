"""
Monte Carlo Simulation Engine for NBA Player Props

Usage:
    from simulation import MonteCarloEngine

    engine = MonteCarloEngine("models/")

    # Full game simulation
    result = engine.simulate_game("LAL", "BOS", injuries={"AD": "OUT"})
    print(result.players["LeBron James"].pts.mean)

    # Single prop analysis
    prop = engine.simulate_player_prop(
        player_name="LeBron James",
        stat="pts",
        line=25.5,
        opponent="BOS",
        over_odds="-115"
    )
    print(prop.recommendation)  # "OVER", "UNDER", or "PASS"
"""

from .engine import MonteCarloEngine
from .models import (
    PlayerFeatures,
    GameContext,
    StatDistribution,
    PlayerPrediction,
    PropBet,
    PropAnalysis,
    SimulationResult
)
from .edge_calculator import EdgeCalculator
from .injury_adjustment import InjuryAdjustmentModule
from .feature_transformer import FeatureTransformer
from .parlay_analyzer import ParlayAnalyzer, ParlayLeg, ParlayAnalysis

__all__ = [
    "MonteCarloEngine",
    "PlayerFeatures",
    "GameContext",
    "StatDistribution",
    "PlayerPrediction",
    "PropBet",
    "PropAnalysis",
    "SimulationResult",
    "EdgeCalculator",
    "InjuryAdjustmentModule",
    "FeatureTransformer",
    "ParlayAnalyzer",
    "ParlayLeg",
    "ParlayAnalysis"
]

__version__ = "1.0.0"
