"""
Edge calculation utilities for comparing model predictions to sportsbook odds.

Converts American odds to implied probabilities and calculates betting edge.
"""

import numpy as np
from typing import Tuple
from .models import PropBet, PropAnalysis


class EdgeCalculator:
    """Utility class for odds conversion and edge calculation."""

    # Minimum edge threshold for recommendations
    MIN_EDGE_MEDIUM = 0.05  # 5% edge for MEDIUM confidence
    MIN_EDGE_HIGH = 0.10    # 10% edge for HIGH confidence

    @staticmethod
    def american_to_implied(odds: str) -> float:
        """
        Convert American odds to implied probability.

        Examples:
            -110 -> 0.524 (52.4%)
            +150 -> 0.400 (40.0%)
            -200 -> 0.667 (66.7%)
        """
        odds_val = float(str(odds).replace("+", ""))

        if odds_val > 0:
            # Positive odds (underdog)
            return 100.0 / (100.0 + odds_val)
        else:
            # Negative odds (favorite)
            return abs(odds_val) / (abs(odds_val) + 100.0)

    @staticmethod
    def american_to_decimal(odds: str) -> float:
        """
        Convert American odds to decimal odds.

        Examples:
            -110 -> 1.909
            +150 -> 2.500
        """
        odds_val = float(str(odds).replace("+", ""))

        if odds_val > 0:
            return (odds_val / 100.0) + 1.0
        else:
            return (100.0 / abs(odds_val)) + 1.0

    @staticmethod
    def devig_odds(over_odds: str, under_odds: str) -> Tuple[float, float]:
        """
        Remove the vig from a two-way market to get true probabilities.

        Standard -110/-110 has ~4.5% vig (implied 52.4% + 52.4% = 104.8%)
        This normalizes to 50%/50%.

        Returns:
            (true_over_prob, true_under_prob)
        """
        over_implied = EdgeCalculator.american_to_implied(over_odds)
        under_implied = EdgeCalculator.american_to_implied(under_odds)

        total = over_implied + under_implied

        return over_implied / total, under_implied / total

    @staticmethod
    def calculate_prop_probability(
        simulated_values: np.ndarray,
        line: float,
        direction: str = "over"
    ) -> float:
        """
        Calculate probability of hitting a prop from simulations.

        Args:
            simulated_values: Array of simulated stat values
            line: The betting line (e.g., 25.5)
            direction: "over" or "under"

        Returns:
            Probability of hitting the prop (0-1)
        """
        if direction == "over":
            return float(np.mean(simulated_values > line))
        else:
            return float(np.mean(simulated_values < line))

    @staticmethod
    def calculate_edge(model_prob: float, implied_prob: float) -> float:
        """
        Calculate edge as difference between model and implied probability.

        Positive edge = model thinks more likely than market (value bet)
        Negative edge = market has it right or overpriced
        """
        return model_prob - implied_prob

    @staticmethod
    def calculate_kelly_fraction(
        edge: float,
        decimal_odds: float,
        kelly_multiplier: float = 0.25
    ) -> float:
        """
        Calculate recommended bet size using Kelly Criterion.

        Args:
            edge: Edge as decimal (e.g., 0.05 for 5%)
            decimal_odds: Decimal odds (e.g., 1.91)
            kelly_multiplier: Fraction of Kelly to use (0.25 = quarter Kelly)

        Returns:
            Recommended bet as fraction of bankroll (0 if negative edge)
        """
        if edge <= 0:
            return 0.0

        # Kelly formula: f = (bp - q) / b
        # where b = decimal_odds - 1, p = model_prob, q = 1 - p
        b = decimal_odds - 1
        if b <= 0:
            return 0.0

        kelly = edge / b
        return max(0, kelly * kelly_multiplier)

    def analyze_prop(
        self,
        simulated_values: np.ndarray,
        prop: PropBet,
        use_devigged: bool = True
    ) -> PropAnalysis:
        """
        Full analysis of a prop bet.

        Args:
            simulated_values: Array of simulated stat values
            prop: PropBet with line and odds
            use_devigged: If True, use devigged odds for edge calc

        Returns:
            PropAnalysis with recommendation
        """
        # Calculate model probabilities
        over_prob = self.calculate_prop_probability(
            simulated_values, prop.line, "over"
        )
        under_prob = 1.0 - over_prob

        # Calculate implied probabilities
        if use_devigged:
            over_implied, under_implied = self.devig_odds(
                prop.over_odds, prop.under_odds
            )
        else:
            over_implied = self.american_to_implied(prop.over_odds)
            under_implied = self.american_to_implied(prop.under_odds)

        # Calculate edges
        over_edge = self.calculate_edge(over_prob, over_implied)
        under_edge = self.calculate_edge(under_prob, under_implied)

        # Determine recommendation and confidence
        if over_edge >= self.MIN_EDGE_HIGH:
            recommendation = "OVER"
            confidence = "HIGH"
        elif over_edge >= self.MIN_EDGE_MEDIUM:
            recommendation = "OVER"
            confidence = "MEDIUM"
        elif under_edge >= self.MIN_EDGE_HIGH:
            recommendation = "UNDER"
            confidence = "HIGH"
        elif under_edge >= self.MIN_EDGE_MEDIUM:
            recommendation = "UNDER"
            confidence = "MEDIUM"
        else:
            recommendation = "PASS"
            confidence = "LOW"

        return PropAnalysis(
            prop=prop,
            model_prob_over=over_prob,
            model_prob_under=under_prob,
            implied_prob_over=over_implied,
            implied_prob_under=under_implied,
            edge_over=over_edge,
            edge_under=under_edge,
            recommendation=recommendation,
            confidence=confidence,
            predicted_mean=float(np.mean(simulated_values)),
            predicted_std=float(np.std(simulated_values))
        )
