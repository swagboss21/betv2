"""
Probability and edge calculation utilities.

Pure math functions - no database access.
"""
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
