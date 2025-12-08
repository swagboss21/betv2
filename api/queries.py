"""
Database query functions for The Brain.

All database operations go through this module.
Functions return dicts with RealDictCursor.
"""
from datetime import datetime
from typing import Optional
import uuid

from db.connection import get_db_cursor


def get_games_today() -> list[dict]:
    """
    Get all games scheduled for today.

    Returns:
        List of game dicts with id, home_team, away_team, starts_at, status
    """
    with get_db_cursor() as cursor:
        cursor.execute("""
            SELECT id, home_team, away_team, starts_at, status
            FROM games
            WHERE DATE(starts_at) = CURRENT_DATE
               OR status = 'scheduled'
            ORDER BY starts_at
        """)
        return [dict(row) for row in cursor.fetchall()]


def get_projection(player_name: str, stat_type: str, game_id: Optional[str] = None, line: Optional[float] = None) -> Optional[dict]:
    """
    Get projection for a specific player/stat with optional line probability.

    Args:
        player_name: Full player name
        stat_type: pts, reb, ast, stl, blk, tov, fg3m
        game_id: Optional game ID (uses most recent if not provided)
        line: Optional line to calculate probability for (ANY value - supports alternate lines)

    Returns:
        Projection dict with mean, std, p10-p90, and if line provided: prob_over, prob_under
    """
    from api.probability import prob_over, prob_under

    with get_db_cursor() as cursor:
        if game_id:
            cursor.execute("""
                SELECT game_id, player_name, stat_type, mean, std, p10, p25, p50, p75, p90
                FROM projections
                WHERE LOWER(player_name) = LOWER(%s) AND stat_type = %s AND game_id = %s
            """, (player_name, stat_type, game_id))
        else:
            cursor.execute("""
                SELECT game_id, player_name, stat_type, mean, std, p10, p25, p50, p75, p90
                FROM projections
                WHERE LOWER(player_name) = LOWER(%s) AND stat_type = %s
                ORDER BY computed_at DESC
                LIMIT 1
            """, (player_name, stat_type))

        row = cursor.fetchone()
        if not row:
            return None

        result = dict(row)

        # If line provided, calculate probabilities using normal distribution
        if line is not None:
            result['line'] = line
            result['prob_over'] = prob_over(result['mean'], result['std'], line)
            result['prob_under'] = prob_under(result['mean'], result['std'], line)

        return result


def get_all_projections(game_id: str) -> list[dict]:
    """
    Get all projections for a game.

    Args:
        game_id: NBA game ID

    Returns:
        List of projection dicts for all players/stats
    """
    with get_db_cursor() as cursor:
        cursor.execute("""
            SELECT game_id, player_name, stat_type, mean, std, p10, p25, p50, p75, p90
            FROM projections
            WHERE game_id = %s
            ORDER BY player_name, stat_type
        """, (game_id,))
        return [dict(row) for row in cursor.fetchall()]


def get_best_props(min_edge: float = 0.05, limit: int = 10) -> list[dict]:
    """
    Get props with highest edge across all tonight's games.

    Uses p50 (median) as market line placeholder (v1).
    Assumes -110 juice (52.4% implied probability) for edge calculation.

    Args:
        min_edge: Minimum edge threshold (default 5%)
        limit: Max number of props to return

    Returns:
        List sorted by edge descending with:
        - player_name, stat_type, game_id
        - mean, std, line, direction
        - probability, edge, edge_formatted
    """
    from api.probability import prob_over, prob_under, format_edge

    with get_db_cursor() as cursor:
        # Get all projections for today's games
        cursor.execute("""
            SELECT p.game_id, p.player_name, p.stat_type, p.mean, p.std, p.p50
            FROM projections p
            JOIN games g ON p.game_id = g.id
            WHERE g.status = 'scheduled' OR DATE(g.starts_at) = CURRENT_DATE
        """)
        projections = cursor.fetchall()

    # Calculate edge for each projection
    results = []
    book_prob = 0.524  # -110 implied probability

    for proj in projections:
        mean = proj['mean']
        std = proj['std']
        line = proj['p50']  # Use median as "market line"

        if std <= 0 or line is None:
            continue

        # Calculate probabilities
        p_over = prob_over(mean, std, line)
        p_under = prob_under(mean, std, line)

        # Pick better direction
        if p_over >= p_under:
            direction = "OVER"
            probability = p_over
        else:
            direction = "UNDER"
            probability = p_under

        # Calculate edge
        edge = probability - book_prob

        # Filter by min_edge
        if edge >= min_edge:
            results.append({
                "player_name": proj['player_name'],
                "stat_type": proj['stat_type'],
                "game_id": proj['game_id'],
                "mean": mean,
                "std": std,
                "line": line,
                "direction": direction,
                "probability": round(probability, 4),
                "edge": round(edge, 4),
                "edge_formatted": format_edge(edge)
            })

    # Sort by edge descending and limit
    results.sort(key=lambda x: x['edge'], reverse=True)
    return results[:limit]


def get_injuries(team: str) -> list[dict]:
    """
    Get injury report for a team.

    Args:
        team: 3-letter team code

    Returns:
        List of injury dicts with player_name, status, injury
    """
    with get_db_cursor() as cursor:
        cursor.execute("""
            SELECT player_name, status, injury, source
            FROM injuries
            WHERE team = %s
            ORDER BY player_name
        """, (team,))
        return [dict(row) for row in cursor.fetchall()]


def save_bet(user_id: str, bet: dict) -> int:
    """
    Save a locked bet.

    Args:
        user_id: UUID of user
        bet: Dict with game_id, player_name, stat_type, line, direction, odds, edge_pct

    Returns:
        Bet ID
    """
    with get_db_cursor() as cursor:
        cursor.execute("""
            INSERT INTO bets (user_id, game_id, player_name, stat_type, line, direction, odds, edge_pct)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
        """, (
            user_id,
            bet.get('game_id'),
            bet.get('player_name'),
            bet.get('stat_type'),
            bet.get('line'),
            bet.get('direction'),
            bet.get('odds', '-110'),
            bet.get('edge_pct')
        ))
        row = cursor.fetchone()
        return row['id'] if row else 0


def get_user_bets(user_id: str, pending_only: bool = False) -> list[dict]:
    """
    Get all bets for a user.

    Args:
        user_id: UUID of user
        pending_only: If True, only return ungraded bets

    Returns:
        List of bet dicts sorted by locked_at desc
    """
    with get_db_cursor() as cursor:
        if pending_only:
            cursor.execute("""
                SELECT * FROM bets
                WHERE user_id = %s AND result IS NULL
                ORDER BY locked_at DESC
            """, (user_id,))
        else:
            cursor.execute("""
                SELECT * FROM bets
                WHERE user_id = %s
                ORDER BY locked_at DESC
            """, (user_id,))
        return [dict(row) for row in cursor.fetchall()]


def get_user_record(user_id: str) -> dict:
    """
    Get user's betting record.

    Args:
        user_id: UUID of user

    Returns:
        Dict with total, wins, losses, pushes, win_rate
    """
    with get_db_cursor() as cursor:
        cursor.execute("""
            SELECT
                COUNT(*) as total,
                COUNT(CASE WHEN result = 'WIN' THEN 1 END) as wins,
                COUNT(CASE WHEN result = 'LOSS' THEN 1 END) as losses,
                COUNT(CASE WHEN result = 'PUSH' THEN 1 END) as pushes
            FROM bets
            WHERE user_id = %s AND result IS NOT NULL
        """, (user_id,))
        row = cursor.fetchone()

        if not row or row['total'] == 0:
            return {"total": 0, "wins": 0, "losses": 0, "pushes": 0, "win_rate": 0.0}

        total = row['total']
        wins = row['wins']
        losses = row['losses']
        pushes = row['pushes']

        # win_rate excludes pushes
        decided = wins + losses
        win_rate = wins / decided if decided > 0 else 0.0

        return {
            "total": total,
            "wins": wins,
            "losses": losses,
            "pushes": pushes,
            "win_rate": round(win_rate, 3)
        }


# User operations

def create_user(email: str, auth_provider: str = "google") -> str:
    """
    Create a new user.

    Args:
        email: User email
        auth_provider: google, apple, etc.

    Returns:
        User ID (UUID)
    """
    user_id = str(uuid.uuid4())
    with get_db_cursor() as cursor:
        cursor.execute("""
            INSERT INTO users (id, email, auth_provider)
            VALUES (%s, %s, %s)
        """, (user_id, email, auth_provider))
    return user_id


def get_user(email: str) -> Optional[dict]:
    """
    Get user by email.

    Args:
        email: User email

    Returns:
        User dict or None
    """
    with get_db_cursor() as cursor:
        cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
        row = cursor.fetchone()
        return dict(row) if row else None


def get_or_create_user(email: str, auth_provider: str = "google") -> dict:
    """
    Get existing user or create new one.

    Args:
        email: User email
        auth_provider: google, apple, etc.

    Returns:
        User dict
    """
    user = get_user(email)
    if user:
        return user
    user_id = create_user(email, auth_provider)
    return get_user(email)


def increment_message_count(user_id: str) -> int:
    """
    Increment user's message count.

    Args:
        user_id: UUID of user

    Returns:
        New message count
    """
    with get_db_cursor() as cursor:
        cursor.execute("""
            UPDATE users
            SET message_count = message_count + 1, last_active_at = NOW()
            WHERE id = %s
            RETURNING message_count
        """, (user_id,))
        row = cursor.fetchone()
        return row['message_count'] if row else 0


def is_user_paid(user_id: str) -> bool:
    """
    Check if user has paid subscription.

    Args:
        user_id: UUID of user

    Returns:
        True if paid
    """
    with get_db_cursor() as cursor:
        cursor.execute("SELECT is_paid FROM users WHERE id = %s", (user_id,))
        row = cursor.fetchone()
        return row['is_paid'] if row else False


def set_user_paid(user_id: str, is_paid: bool = True) -> None:
    """
    Update user's paid status.

    Args:
        user_id: UUID of user
        is_paid: New paid status
    """
    with get_db_cursor() as cursor:
        cursor.execute("UPDATE users SET is_paid = %s WHERE id = %s", (is_paid, user_id))
