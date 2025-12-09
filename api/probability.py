"""
Probability and edge calculation utilities.

Pure math functions - no database access.
"""
import numpy as np
from scipy import stats


def prob_over(mean: float, std: float, line: float) -> float:
    """
    Calculate probability of exceeding the line.

    P(X > line) using normal distribution survival function.

    Args:
        mean: Distribution mean
        std: Distribution standard deviation
        line: Betting line to compare against

    Returns:
        Probability (0-1) of going over

    Example:
        mean=26.3, std=5.2, line=25.5
        returns 0.561 (56.1% chance of going over)
    """
    if std <= 0:
        return 1.0 if mean > line else 0.0
    return float(stats.norm.sf(line, loc=mean, scale=std))


def prob_under(mean: float, std: float, line: float) -> float:
    """
    Calculate probability of staying under the line.

    P(X < line) using normal distribution CDF.

    Args:
        mean: Distribution mean
        std: Distribution standard deviation
        line: Betting line to compare against

    Returns:
        Probability (0-1) of going under
    """
    if std <= 0:
        return 1.0 if mean < line else 0.0
    return float(stats.norm.cdf(line, loc=mean, scale=std))


def american_to_probability(odds: str) -> float:
    """
    Convert American odds to implied probability.

    Args:
        odds: American odds string (e.g., "-115", "+150")

    Returns:
        Implied probability (0-1)

    Examples:
        "-115" -> 0.535 (53.5%)
        "+150" -> 0.40 (40%)
    """
    odds_val = float(odds.replace('+', ''))
    if odds_val > 0:
        return 100.0 / (100.0 + odds_val)
    else:
        return abs(odds_val) / (abs(odds_val) + 100.0)


def calculate_edge(model_prob: float, book_prob: float) -> float:
    """
    Calculate betting edge.

    Edge = model_prob - book_prob

    Positive edge means model thinks outcome is more likely than book implies.

    Args:
        model_prob: Probability from our model
        book_prob: Implied probability from betting odds

    Returns:
        Edge as decimal (e.g., 0.05 = 5% edge)

    Example:
        model says 56% over, book implies 53.5%
        edge = 0.025 (2.5% edge)
    """
    return model_prob - book_prob


def format_edge(edge: float) -> str:
    """
    Format edge as human-readable string.

    Args:
        edge: Edge as decimal

    Returns:
        Formatted string like "+5.2%" or "PASS"
    """
    if edge >= 0.05:
        return f"+{edge*100:.1f}%"
    elif edge <= -0.05:
        return f"{edge*100:.1f}%"
    else:
        return "PASS"


def devig_odds(over_odds: str, under_odds: str) -> tuple[float, float]:
    """
    Remove vig from odds to get true probabilities.

    Args:
        over_odds: American odds for over
        under_odds: American odds for under

    Returns:
        Tuple of (true_prob_over, true_prob_under)
    """
    over_prob = american_to_probability(over_odds)
    under_prob = american_to_probability(under_odds)
    total = over_prob + under_prob
    return (over_prob / total, under_prob / total)


# ============================================================
# EMPIRICAL PROBABILITY FUNCTIONS
# These use stored histograms/percentiles instead of normal distribution
# Better for count data like fg3m, stl, blk
# ============================================================

def prob_over_empirical(histogram: dict, line: float) -> float:
    """
    Calculate P(X > line) using stored histogram.

    This is more accurate than normal distribution for count data (fg3m, stl, blk)
    because it uses the actual empirical distribution from Monte Carlo simulations.

    Args:
        histogram: Dict with "counts" (list) and "edges" (list) from simulation
        line: Betting line to compare against

    Returns:
        Probability (0-1) of going over

    Example:
        histogram = {"counts": [100, 500, 300, 100], "edges": [0, 1, 2, 3, 4]}
        prob_over_empirical(histogram, 1.5) -> ~0.40 (300+100 out of 1000)
    """
    if not histogram or "counts" not in histogram or "edges" not in histogram:
        return 0.5  # Fallback

    counts = np.array(histogram["counts"])
    edges = np.array(histogram["edges"])
    total = counts.sum()

    if total == 0:
        return 0.5

    # Special case for line = 0 (common for count data like fg3m)
    # "Over 0" means scoring at least 1, so we count everything outside the first bin
    if line <= 0:
        # First bin [edges[0], edges[1]] typically contains zeros
        # Sum all bins except the first one
        if len(counts) > 1:
            over_count = counts[1:].sum()
            return float(over_count / total)
        return 0.5

    # For positive lines, use bin midpoint comparison
    bin_midpoints = (edges[:-1] + edges[1:]) / 2

    # Count values in bins where midpoint > line
    over_mask = bin_midpoints > line
    over_count = counts[over_mask].sum()

    return float(over_count / total)


def prob_under_empirical(histogram: dict, line: float) -> float:
    """Calculate P(X < line) using stored histogram."""
    return 1.0 - prob_over_empirical(histogram, line)


def prob_over_from_percentiles(
    p10: float, p25: float, p50: float, p75: float, p90: float, line: float
) -> float:
    """
    Approximate probability using stored percentiles (fallback method).

    Uses linear interpolation between percentiles when histogram is not available.
    Less accurate than empirical but works without histogram storage.

    Args:
        p10, p25, p50, p75, p90: Percentile values from simulation
        line: Betting line to compare against

    Returns:
        Approximate probability (0-1) of going over
    """
    # Build percentile lookup table
    # Each tuple is (percentile_rank, value)
    percentiles = [
        (10, p10),
        (25, p25),
        (50, p50),
        (75, p75),
        (90, p90)
    ]

    # Handle edge cases
    if line <= p10:
        # Line is below p10, so P(over) > 90%
        # Extrapolate assuming some distribution below p10
        if p10 > 0:
            return min(0.95, 0.90 + 0.10 * (p10 - line) / p10)
        return 0.95

    if line >= p90:
        # Line is above p90, so P(over) < 10%
        # Extrapolate assuming some distribution above p90
        if p90 > 0:
            return max(0.05, 0.10 * (p90 / max(line, 0.01)))
        return 0.05

    # Interpolate between percentiles
    for i in range(len(percentiles) - 1):
        pct_low, val_low = percentiles[i]
        pct_high, val_high = percentiles[i + 1]

        if val_low <= line <= val_high:
            # Line falls in this range
            if val_high - val_low > 0:
                # Interpolate position within range
                frac = (line - val_low) / (val_high - val_low)
            else:
                frac = 0.5

            # P(over) at val_low is (100 - pct_low)/100
            # P(over) at val_high is (100 - pct_high)/100
            prob_at_low = (100 - pct_low) / 100
            prob_at_high = (100 - pct_high) / 100

            # Linear interpolation
            return prob_at_low + frac * (prob_at_high - prob_at_low)

    # Fallback (shouldn't reach here)
    return 0.5


def prob_under_from_percentiles(
    p10: float, p25: float, p50: float, p75: float, p90: float, line: float
) -> float:
    """Calculate P(X < line) using percentile interpolation."""
    return 1.0 - prob_over_from_percentiles(p10, p25, p50, p75, p90, line)
