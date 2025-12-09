"""
Player name matching utilities for SGO API integration.

Matches SGO player IDs (e.g., "LEBRON_JAMES_LOS_ANGELES_LAKERS_NBA")
to our player names (e.g., "LeBron James").
"""

from difflib import SequenceMatcher
from typing import Optional, Tuple
import re


def extract_player_name_from_sgo_id(sgo_player_id: str) -> str:
    """
    Extract readable player name from SGO player ID.

    SGO format varies:
    - "LEBRON_JAMES_LOS_ANGELES_LAKERS_NBA" -> "LeBron James"
    - "Anthony Davis 1" -> "Anthony Davis"  (newer format with suffix)

    Args:
        sgo_player_id: SGO format player ID

    Returns:
        Human-readable player name
    """
    # Check if it's the newer format (already readable with suffix)
    # Format: "FirstName LastName N" where N is a numeric suffix
    if re.match(r'^[A-Z][a-z]+ [A-Z][a-z]', sgo_player_id):
        # Remove numeric suffix at end
        name = re.sub(r'\s+\d+$', '', sgo_player_id)
        return name.strip()

    # Legacy format: FIRST_LAST_TEAM_CITY_TEAM_NAME_LEAGUE
    parts = sgo_player_id.split('_')

    # Find where the name ends (before team city/name)
    # Usually names are 2-3 parts, teams are 2-4 parts
    # Try to find common team indicators
    team_indicators = {
        'LOS', 'NEW', 'SAN', 'GOLDEN', 'OKLAHOMA', 'MINNESOTA', 'PORTLAND',
        'MIAMI', 'CHICAGO', 'BOSTON', 'ATLANTA', 'CLEVELAND', 'DENVER',
        'DETROIT', 'HOUSTON', 'INDIANA', 'MEMPHIS', 'MILWAUKEE', 'BROOKLYN',
        'ORLANDO', 'PHILADELPHIA', 'PHOENIX', 'SACRAMENTO', 'TORONTO',
        'UTAH', 'WASHINGTON', 'CHARLOTTE', 'DALLAS', 'NBA', 'WNBA', 'NCAAB'
    }

    name_parts = []
    for part in parts:
        if part.upper() in team_indicators:
            break
        name_parts.append(part)

    if not name_parts:
        return sgo_player_id  # Fallback

    # Convert to title case
    name = ' '.join(part.title() for part in name_parts)

    # Handle special cases (Mc, O', etc.)
    name = re.sub(r'\bMc([a-z])', lambda m: f"Mc{m.group(1).upper()}", name)
    name = re.sub(r"\bO'([a-z])", lambda m: f"O'{m.group(1).upper()}", name)

    return name


def normalize_name(name: str) -> str:
    """
    Normalize player name for comparison.

    - Lowercase
    - Remove punctuation
    - Remove suffixes (Jr., III, etc.)
    - Standardize common name variations
    """
    name = name.lower().strip()

    # Remove common suffixes
    name = re.sub(r'\s+(jr\.?|sr\.?|iii|ii|iv)$', '', name, flags=re.IGNORECASE)

    # Remove punctuation
    name = re.sub(r"[.'`\-]", '', name)

    # Remove extra whitespace
    name = ' '.join(name.split())

    return name


def similarity_score(name1: str, name2: str) -> float:
    """
    Calculate similarity between two names.

    Returns:
        Score 0-1 where 1 is exact match
    """
    norm1 = normalize_name(name1)
    norm2 = normalize_name(name2)

    # Exact match after normalization
    if norm1 == norm2:
        return 1.0

    # SequenceMatcher for fuzzy matching
    return SequenceMatcher(None, norm1, norm2).ratio()


def match_player(sgo_player_id: str, candidate_names: list[str],
                 threshold: float = 0.85) -> Optional[Tuple[str, float]]:
    """
    Match SGO player ID to the best candidate from our database.

    Args:
        sgo_player_id: SGO format player ID
        candidate_names: List of player names from our database
        threshold: Minimum similarity score to accept (default 0.85 = 85%)

    Returns:
        Tuple of (matched_name, score) or None if no match above threshold
    """
    sgo_name = extract_player_name_from_sgo_id(sgo_player_id)

    best_match = None
    best_score = 0.0

    for candidate in candidate_names:
        score = similarity_score(sgo_name, candidate)
        if score > best_score:
            best_score = score
            best_match = candidate

    if best_score >= threshold:
        return (best_match, best_score)

    return None


def build_player_lookup(player_names: list[str]) -> dict[str, str]:
    """
    Build a lookup dict from normalized names to original names.

    Useful for quick exact-match lookups.
    """
    return {normalize_name(name): name for name in player_names}


# Common name variations mapping (SGO name -> canonical name)
NAME_OVERRIDES = {
    "nicolas claxton": "Nic Claxton",
    "pj washington": "P.J. Washington",
    "og anunoby": "OG Anunoby",
    "mo bamba": "Mohamed Bamba",
    "jt thor": "JT Thor",
    "aj green": "A.J. Green",
    "gg jackson": "GG Jackson II",
    "rj barrett": "RJ Barrett",
}


def apply_name_override(name: str) -> str:
    """Apply known name corrections."""
    norm = normalize_name(name)
    return NAME_OVERRIDES.get(norm, name)
