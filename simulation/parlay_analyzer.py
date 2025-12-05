"""
Parlay Analyzer - Detect correlations and build thesis-based parlays.

The key insight: Good parlays aren't random combinations - they're CORRELATED
picks that rise and fall together.

Thesis types:
- Shootout: High-scoring game → both teams' stars over points
- Blowout: One-sided game → winning team's bench gets minutes
- Pace: Fast pace → more possessions → more counting stats
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass
class ParlayLeg:
    """A single leg in a parlay."""
    player_name: str
    stat: str
    line: float
    direction: str  # "OVER" or "UNDER"
    team: str
    opponent: str
    is_home: bool
    model_prob: float
    edge: float


@dataclass
class CorrelationInfo:
    """Information about correlation between parlay legs."""
    correlation_type: str  # "shootout", "blowout", "pace", "independent"
    correlation_strength: float  # -1 to 1, where positive = positively correlated
    explanation: str


@dataclass
class ParlayAnalysis:
    """Full analysis of a multi-leg parlay."""
    legs: List[ParlayLeg]
    individual_probs: List[float]
    naive_combined_prob: float  # P(A) * P(B) * ... assuming independence
    adjusted_combined_prob: float  # Accounting for correlation
    correlations: List[CorrelationInfo]
    thesis: str  # Human-readable explanation of why these picks hang together
    overall_edge: float
    recommendation: str  # "STRONG", "MODERATE", "WEAK", "PASS"


class ParlayAnalyzer:
    """
    Analyze multi-leg parlays for correlation and generate thesis explanations.
    """

    # Correlation adjustments by thesis type
    # These are heuristic multipliers - positive correlation means legs move together
    CORRELATION_ADJUSTMENTS = {
        "shootout": 0.15,  # Same game, both overs on points → correlated
        "blowout": 0.10,   # Same game, blowout scenario → bench minutes correlated
        "pace": 0.08,      # Same game, pace-dependent stats → correlated
        "same_player": 0.25,  # Same player different stats → highly correlated
        "same_team": 0.05,    # Same team different players → slightly correlated
        "opposite_sides": -0.05,  # Opposing players same stat → slightly negative
    }

    def __init__(self):
        pass

    def analyze_parlay(self, legs: List[ParlayLeg]) -> ParlayAnalysis:
        """
        Analyze a multi-leg parlay.

        Args:
            legs: List of ParlayLeg objects

        Returns:
            ParlayAnalysis with combined probability and thesis
        """
        if not legs:
            return self._empty_analysis()

        if len(legs) == 1:
            return self._single_leg_analysis(legs[0])

        # Calculate individual probabilities
        individual_probs = [leg.model_prob for leg in legs]

        # Calculate naive combined probability (assuming independence)
        naive_combined = np.prod(individual_probs)

        # Detect correlations between legs
        correlations = self._detect_correlations(legs)

        # Adjust for correlations
        adjusted_combined = self._adjust_for_correlations(
            individual_probs, correlations, legs
        )

        # Generate thesis
        thesis = self._generate_thesis(legs, correlations)

        # Calculate overall edge
        # For parlays, we compare to implied odds which are harder to calculate
        # Simplified: average edge across legs weighted by correlation
        avg_edge = np.mean([leg.edge for leg in legs])
        correlation_boost = sum(c.correlation_strength for c in correlations) * 0.1
        overall_edge = avg_edge + correlation_boost

        # Recommendation
        if overall_edge > 0.10 and adjusted_combined > 0.15:
            recommendation = "STRONG"
        elif overall_edge > 0.05 and adjusted_combined > 0.10:
            recommendation = "MODERATE"
        elif overall_edge > 0.02:
            recommendation = "WEAK"
        else:
            recommendation = "PASS"

        return ParlayAnalysis(
            legs=legs,
            individual_probs=individual_probs,
            naive_combined_prob=naive_combined,
            adjusted_combined_prob=adjusted_combined,
            correlations=correlations,
            thesis=thesis,
            overall_edge=overall_edge,
            recommendation=recommendation
        )

    def _detect_correlations(self, legs: List[ParlayLeg]) -> List[CorrelationInfo]:
        """Detect correlations between parlay legs."""
        correlations = []

        for i, leg1 in enumerate(legs):
            for leg2 in legs[i + 1:]:
                correlation = self._detect_pair_correlation(leg1, leg2)
                if correlation:
                    correlations.append(correlation)

        return correlations

    def _detect_pair_correlation(
        self,
        leg1: ParlayLeg,
        leg2: ParlayLeg
    ) -> Optional[CorrelationInfo]:
        """Detect correlation between two legs."""

        # Same player, different stats
        if leg1.player_name == leg2.player_name:
            return CorrelationInfo(
                correlation_type="same_player",
                correlation_strength=self.CORRELATION_ADJUSTMENTS["same_player"],
                explanation=f"{leg1.player_name}'s stats are highly correlated - "
                           f"if they play well, multiple stats benefit"
            )

        # Same game (check if they're playing each other)
        same_game = (
            (leg1.team == leg2.opponent and leg2.team == leg1.opponent) or
            (leg1.opponent == leg2.team and leg2.opponent == leg1.team)
        )

        if same_game:
            # Check for shootout thesis
            if (leg1.stat == "pts" and leg2.stat == "pts" and
                leg1.direction == "OVER" and leg2.direction == "OVER"):
                return CorrelationInfo(
                    correlation_type="shootout",
                    correlation_strength=self.CORRELATION_ADJUSTMENTS["shootout"],
                    explanation=f"Shootout thesis: If {leg1.team} vs {leg2.team} is high-scoring, "
                               f"both {leg1.player_name} and {leg2.player_name} benefit"
                )

            # Check for pace thesis (rebounds, assists also benefit from pace)
            pace_stats = {"pts", "reb", "ast", "fg3m"}
            if (leg1.stat in pace_stats and leg2.stat in pace_stats and
                leg1.direction == "OVER" and leg2.direction == "OVER"):
                return CorrelationInfo(
                    correlation_type="pace",
                    correlation_strength=self.CORRELATION_ADJUSTMENTS["pace"],
                    explanation=f"Pace thesis: Fast game = more possessions = more opportunities "
                               f"for both {leg1.player_name} and {leg2.player_name}"
                )

            # Same team
            if leg1.team == leg2.team:
                return CorrelationInfo(
                    correlation_type="same_team",
                    correlation_strength=self.CORRELATION_ADJUSTMENTS["same_team"],
                    explanation=f"Same team: {leg1.player_name} and {leg2.player_name} "
                               f"share court time and game flow"
                )

            # Opposite sides, same stat
            if leg1.stat == leg2.stat:
                return CorrelationInfo(
                    correlation_type="opposite_sides",
                    correlation_strength=self.CORRELATION_ADJUSTMENTS["opposite_sides"],
                    explanation=f"Opposing stars: {leg1.player_name} and {leg2.player_name} "
                               f"compete for the same stat category"
                )

        return None

    def _adjust_for_correlations(
        self,
        individual_probs: List[float],
        correlations: List[CorrelationInfo],
        legs: List[ParlayLeg]
    ) -> float:
        """
        Adjust combined probability for correlations.

        Uses a simplified adjustment where positive correlations increase
        combined probability (since legs move together).
        """
        naive_prob = np.prod(individual_probs)

        if not correlations:
            return naive_prob

        # Calculate average correlation adjustment
        total_adjustment = sum(c.correlation_strength for c in correlations)
        avg_adjustment = total_adjustment / len(correlations)

        # Adjust probability
        # Positive correlation → higher combined prob than naive
        # Negative correlation → lower combined prob than naive
        adjustment_factor = 1 + (avg_adjustment * 0.5)  # Dampen the effect
        adjusted_prob = naive_prob * adjustment_factor

        # Clamp to valid range
        return min(max(adjusted_prob, 0.01), 0.99)

    def _generate_thesis(
        self,
        legs: List[ParlayLeg],
        correlations: List[CorrelationInfo]
    ) -> str:
        """Generate a human-readable thesis for the parlay."""
        if not correlations:
            return "Independent picks - these legs don't share obvious correlation"

        # Find the dominant thesis type
        thesis_counts = {}
        for c in correlations:
            thesis_counts[c.correlation_type] = thesis_counts.get(c.correlation_type, 0) + 1

        dominant_type = max(thesis_counts.keys(), key=lambda k: thesis_counts[k])

        # Build thesis explanation
        if dominant_type == "shootout":
            teams = list(set(leg.team for leg in legs))
            return (f"SHOOTOUT THESIS: Betting on a high-scoring {' vs '.join(teams)} game. "
                    f"All picks benefit if the game is fast-paced and offensively focused. "
                    f"Risk: If it's a defensive grind, all legs struggle together.")

        elif dominant_type == "pace":
            return ("PACE THESIS: These picks all benefit from a fast-paced game with "
                    "more possessions. They rise and fall together based on game tempo.")

        elif dominant_type == "same_player":
            player = legs[0].player_name
            return (f"PLAYER PERFORMANCE THESIS: Betting on {player} having a big game. "
                    f"If they're cooking, multiple stats benefit. Risk: Bad game tanks all legs.")

        elif dominant_type == "same_team":
            team = legs[0].team
            return (f"TEAM SUCCESS THESIS: Betting on {team} having a strong offensive night. "
                    f"These players share the same game flow.")

        else:
            return "MIXED THESIS: These picks have some correlation but different drivers."

    def _empty_analysis(self) -> ParlayAnalysis:
        """Return empty analysis for no legs."""
        return ParlayAnalysis(
            legs=[],
            individual_probs=[],
            naive_combined_prob=0,
            adjusted_combined_prob=0,
            correlations=[],
            thesis="No legs provided",
            overall_edge=0,
            recommendation="PASS"
        )

    def _single_leg_analysis(self, leg: ParlayLeg) -> ParlayAnalysis:
        """Return analysis for single leg (straight bet)."""
        return ParlayAnalysis(
            legs=[leg],
            individual_probs=[leg.model_prob],
            naive_combined_prob=leg.model_prob,
            adjusted_combined_prob=leg.model_prob,
            correlations=[],
            thesis="Single leg - consider as straight bet instead of parlay",
            overall_edge=leg.edge,
            recommendation="STRONG" if leg.edge > 0.05 else "MODERATE" if leg.edge > 0.02 else "PASS"
        )

    def to_dict(self, analysis: ParlayAnalysis) -> dict:
        """Convert ParlayAnalysis to dict for JSON serialization."""
        return {
            "legs": [
                {
                    "player": leg.player_name,
                    "stat": leg.stat,
                    "line": leg.line,
                    "direction": leg.direction,
                    "team": leg.team,
                    "opponent": leg.opponent,
                    "model_prob": round(leg.model_prob, 4),
                    "edge": round(leg.edge, 4)
                }
                for leg in analysis.legs
            ],
            "individual_probs": [round(p, 4) for p in analysis.individual_probs],
            "naive_combined_prob": round(analysis.naive_combined_prob, 4),
            "adjusted_combined_prob": round(analysis.adjusted_combined_prob, 4),
            "correlations": [
                {
                    "type": c.correlation_type,
                    "strength": round(c.correlation_strength, 3),
                    "explanation": c.explanation
                }
                for c in analysis.correlations
            ],
            "thesis": analysis.thesis,
            "overall_edge": round(analysis.overall_edge, 4),
            "recommendation": analysis.recommendation
        }
