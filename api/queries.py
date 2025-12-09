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
        Projection dict with mean, std, p10-p90, deviation, and if line provided: prob_over, prob_under
    """
    from api.probability import (
        prob_over, prob_under,
        prob_over_empirical, prob_under_empirical,
        prob_over_from_percentiles
    )

    with get_db_cursor() as cursor:
        if game_id:
            cursor.execute("""
                SELECT game_id, player_name, stat_type, mean, std, p10, p25, p50, p75, p90,
                       sim_histogram, l5_avg, szn_avg, l5_std
                FROM projections
                WHERE LOWER(player_name) = LOWER(%s) AND stat_type = %s AND game_id = %s
            """, (player_name, stat_type, game_id))
        else:
            cursor.execute("""
                SELECT game_id, player_name, stat_type, mean, std, p10, p25, p50, p75, p90,
                       sim_histogram, l5_avg, szn_avg, l5_std
                FROM projections
                WHERE LOWER(player_name) = LOWER(%s) AND stat_type = %s
                ORDER BY computed_at DESC
                LIMIT 1
            """, (player_name, stat_type))

        row = cursor.fetchone()
        if not row:
            return None

        result = dict(row)

        # Calculate deviation signal (hot/cold indicator)
        # Formula: deviation = (l5_avg - szn_avg) / max(l5_std, 0.5)
        l5_avg = result.get('l5_avg')
        szn_avg = result.get('szn_avg')
        l5_std = result.get('l5_std')

        if l5_avg is not None and szn_avg is not None:
            std_safe = max(l5_std or 0.5, 0.5)
            result['deviation'] = round((l5_avg - szn_avg) / std_safe, 2)
        else:
            result['deviation'] = None

        # If line provided, calculate probabilities
        if line is not None:
            result['line'] = line

            # Use empirical probability if histogram available (more accurate for count data)
            if result.get('sim_histogram'):
                result['prob_over'] = prob_over_empirical(result['sim_histogram'], line)
                result['prob_under'] = prob_under_empirical(result['sim_histogram'], line)
            else:
                # Fallback: use percentile interpolation (more accurate than normal for count data)
                result['prob_over'] = prob_over_from_percentiles(
                    result['p10'], result['p25'], result['p50'], result['p75'], result['p90'], line
                )
                result['prob_under'] = 1.0 - result['prob_over']

        # Don't return internal fields to the caller
        result.pop('sim_histogram', None)
        result.pop('l5_std', None)  # Keep l5_avg and szn_avg for transparency

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
    from api.probability import (
        format_edge,
        prob_over_empirical,
        prob_over_from_percentiles
    )

    with get_db_cursor() as cursor:
        # Get all projections for today's games (including histogram and percentiles)
        cursor.execute("""
            SELECT p.game_id, p.player_name, p.stat_type, p.mean, p.std, p.p50,
                   p.p10, p.p25, p.p75, p.p90, p.sim_histogram
            FROM projections p
            JOIN games g ON p.game_id = g.id
            WHERE g.status = 'scheduled' OR DATE(g.starts_at) = CURRENT_DATE
        """)
        projections = cursor.fetchall()

    # Calculate edge for each projection
    results = []
    book_prob = 0.524  # -110 implied probability

    # Minimum realistic lines by stat type (sportsbooks don't offer lines below these)
    min_lines = {
        'pts': 5.5,
        'reb': 2.5,
        'ast': 1.5,
        'stl': 0.5,
        'blk': 0.5,
        'tov': 0.5,
        'fg3m': 0.5
    }

    for proj in projections:
        mean = proj['mean']
        std = proj['std']
        line = proj['p50']  # Use median as "market line"

        if std <= 0 or line is None:
            continue

        # Skip unrealistic lines - sportsbooks don't offer these
        min_line = min_lines.get(proj['stat_type'], 0.5)
        if line < min_line:
            continue

        # Calculate probabilities using empirical method (more accurate for count data)
        if proj.get('sim_histogram'):
            p_over = prob_over_empirical(proj['sim_histogram'], line)
        else:
            # Fallback to percentile interpolation
            p_over = prob_over_from_percentiles(
                proj.get('p10', 0), proj.get('p25', 0), proj.get('p50', 0),
                proj.get('p75', 0), proj.get('p90', 0), line
            )
        p_under = 1.0 - p_over

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


def get_tonight_injuries() -> list[dict]:
    """Get injuries for all teams playing tonight, grouped by game.

    Returns:
        List of dicts with structure:
        {
            "game_id": "0022400123",
            "matchup": "BOS @ LAL",
            "starts_at": datetime or None,
            "home_team": "LAL",
            "away_team": "BOS",
            "home_injuries": [...],
            "away_injuries": [...]
        }
    """
    with get_db_cursor() as cursor:
        cursor.execute("""
            SELECT id, home_team, away_team, starts_at
            FROM games
            WHERE status = 'scheduled'
            ORDER BY starts_at NULLS LAST
        """)
        games = [dict(row) for row in cursor.fetchall()]

        result = []
        for game in games:
            home_injuries = get_injuries(game['home_team'])
            away_injuries = get_injuries(game['away_team'])
            result.append({
                "game_id": game['id'],
                "matchup": f"{game['away_team']} @ {game['home_team']}",
                "starts_at": game['starts_at'],
                "home_team": game['home_team'],
                "away_team": game['away_team'],
                "home_injuries": home_injuries,
                "away_injuries": away_injuries
            })
        return result


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


def get_tonight_analysis() -> dict:
    """
    Get complete analysis for tonight's games in one call.

    Returns all games, players, projections, odds, and injuries
    needed for the agent to make recommendations.

    Returns:
        Dict with:
        - games: List of tonight's games
        - players: List of player analysis dicts
    """
    from api.probability import (
        prob_over_empirical,
        prob_over_from_percentiles
    )

    with get_db_cursor() as cursor:
        # Get tonight's games
        cursor.execute("""
            SELECT id, home_team, away_team, starts_at, status
            FROM games
            WHERE status = 'scheduled' OR DATE(starts_at) = CURRENT_DATE
            ORDER BY starts_at
        """)
        games = [dict(row) for row in cursor.fetchall()]

        if not games:
            return {"games": [], "players": []}

        game_ids = [g['id'] for g in games]

        # Get all projections for tonight's games with deviation columns
        cursor.execute("""
            SELECT p.game_id, p.player_name, p.stat_type,
                   p.mean, p.std, p.p10, p.p25, p.p50, p.p75, p.p90,
                   p.l5_avg, p.szn_avg, p.l5_std, p.sim_histogram,
                   g.home_team, g.away_team
            FROM projections p
            JOIN games g ON p.game_id = g.id
            WHERE p.game_id = ANY(%s)
            ORDER BY p.player_name, p.stat_type
        """, (game_ids,))
        projections = cursor.fetchall()

        # Get odds if available
        cursor.execute("""
            SELECT game_id, player_name, stat_type, line, over_odds, under_odds, book
            FROM odds
            WHERE game_id = ANY(%s)
        """, (game_ids,))
        odds_rows = cursor.fetchall()

        # Build odds lookup: (game_id, player_name, stat_type) -> odds
        odds_lookup = {}
        for o in odds_rows:
            key = (o['game_id'], o['player_name'], o['stat_type'])
            odds_lookup[key] = {
                'line': o['line'],
                'over_odds': o['over_odds'],
                'under_odds': o['under_odds'],
                'book': o['book']
            }

        # Get injuries for teams playing tonight
        teams = set()
        for g in games:
            teams.add(g['home_team'])
            teams.add(g['away_team'])

        cursor.execute("""
            SELECT player_name, team, status, injury
            FROM injuries
            WHERE team = ANY(%s)
        """, (list(teams),))
        injuries = cursor.fetchall()

        # Build injury lookup: player_name -> status
        injury_lookup = {i['player_name']: i['status'] for i in injuries}

    # Aggregate projections by player
    players_dict = {}

    for proj in projections:
        player_name = proj['player_name']

        if player_name not in players_dict:
            # Determine team from game context
            game_id = proj['game_id']
            game = next((g for g in games if g['id'] == game_id), None)
            team = None
            if game:
                # Player could be on either team - we don't have team info in projections
                # For now, leave team as None or infer from another source
                pass

            players_dict[player_name] = {
                'name': player_name,
                'game_id': game_id,
                'injury_status': injury_lookup.get(player_name),
                'projections': {},
            }

        stat_type = proj['stat_type']

        # Calculate deviation
        l5_avg = proj['l5_avg']
        szn_avg = proj['szn_avg']
        l5_std = proj['l5_std']
        deviation = None
        if l5_avg is not None and szn_avg is not None:
            std_safe = max(l5_std or 0.5, 0.5)
            deviation = round((l5_avg - szn_avg) / std_safe, 2)

        # Get odds for this player/stat
        odds_key = (proj['game_id'], player_name, stat_type)
        odds_data = odds_lookup.get(odds_key)

        # Build projection data
        proj_data = {
            'mean': round(proj['mean'], 1),
            'p50': round(proj['p50'], 1) if proj['p50'] else None,
            'p10': round(proj['p10'], 1) if proj['p10'] else None,
            'p90': round(proj['p90'], 1) if proj['p90'] else None,
            'deviation': deviation,
            'l5_avg': round(l5_avg, 1) if l5_avg else None,
            'szn_avg': round(szn_avg, 1) if szn_avg else None,
        }

        # Add odds and probability if available
        if odds_data:
            line = odds_data['line']
            proj_data['odds'] = {
                'line': line,
                'over': odds_data['over_odds'],
                'under': odds_data['under_odds'],
            }

            # Calculate probability for the line
            if proj.get('sim_histogram'):
                prob_over = prob_over_empirical(proj['sim_histogram'], line)
            else:
                prob_over = prob_over_from_percentiles(
                    proj['p10'] or 0, proj['p25'] or 0, proj['p50'] or 0,
                    proj['p75'] or 0, proj['p90'] or 0, line
                )
            proj_data['prob_over'] = round(prob_over, 3)
            proj_data['prob_under'] = round(1.0 - prob_over, 3)

        players_dict[player_name]['projections'][stat_type] = proj_data

    # Format games
    formatted_games = []
    for g in games:
        formatted_games.append({
            'id': g['id'],
            'matchup': f"{g['away_team']} @ {g['home_team']}",
            'home_team': g['home_team'],
            'away_team': g['away_team'],
            'starts_at': g['starts_at'].isoformat() if g['starts_at'] else None,
        })

    return {
        'games': formatted_games,
        'players': list(players_dict.values()),
        'injury_count': len(injuries),
        'odds_count': len(odds_rows),
    }
