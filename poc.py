"""
The Brain v0.1 - NBA Betting Co-Pilot
=====================================

LLM + Monte Carlo simulation engine integration.
Core principle: MODEL does MATH, LLM does TALKING.

Usage:
    python poc.py

Requirements:
    pip install anthropic beautifulsoup4 requests

    Set ANTHROPIC_API_KEY environment variable
"""

import json
import os
import sys
from datetime import date, datetime
from typing import Any, Dict, List, Optional

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from simulation import MonteCarloEngine, ParlayAnalyzer, ParlayLeg

# =============================================================================
# ENGINE INITIALIZATION
# =============================================================================

print("Loading Monte Carlo engine...")
engine = MonteCarloEngine("models/")
parlay_analyzer = ParlayAnalyzer()
print("Engine ready.\n")

# =============================================================================
# SESSION STATE
# =============================================================================

# Injury overrides from user (takes priority over scraped data)
INJURY_OVERRIDES: Dict[str, Dict] = {}

# Cached ESPN injury data
ESPN_INJURY_CACHE: Dict[str, List] = {}
ESPN_CACHE_TIME: Optional[datetime] = None

# Team abbreviation mappings
TEAM_ABBREVS = {
    "ATL": "Atlanta Hawks", "BOS": "Boston Celtics", "BKN": "Brooklyn Nets",
    "CHA": "Charlotte Hornets", "CHI": "Chicago Bulls", "CLE": "Cleveland Cavaliers",
    "DAL": "Dallas Mavericks", "DEN": "Denver Nuggets", "DET": "Detroit Pistons",
    "GSW": "Golden State Warriors", "HOU": "Houston Rockets", "IND": "Indiana Pacers",
    "LAC": "LA Clippers", "LAL": "Los Angeles Lakers", "MEM": "Memphis Grizzlies",
    "MIA": "Miami Heat", "MIL": "Milwaukee Bucks", "MIN": "Minnesota Timberwolves",
    "NOP": "New Orleans Pelicans", "NYK": "New York Knicks", "OKC": "Oklahoma City Thunder",
    "ORL": "Orlando Magic", "PHI": "Philadelphia 76ers", "PHX": "Phoenix Suns",
    "POR": "Portland Trail Blazers", "SAC": "Sacramento Kings", "SAS": "San Antonio Spurs",
    "TOR": "Toronto Raptors", "UTA": "Utah Jazz", "WAS": "Washington Wizards"
}

# Reverse mapping for lookups
TEAM_NAME_TO_ABBREV = {v.lower(): k for k, v in TEAM_ABBREVS.items()}

# =============================================================================
# SYSTEM PROMPT
# =============================================================================

SYSTEM_PROMPT = """You are "The Brain" - an AI sports betting co-pilot for NBA player props.

YOUR ROLE:
- Help users find value bets using data and probability, not hunches
- Explain recommendations conversationally, like a smart friend who knows stats
- Present options and analysis - NEVER make the final decision for them

TOOLS AVAILABLE:
- simulate_prop: Get probability and edge for any player prop
- get_player_projection: Get full stat projections for a player
- check_injuries: Get injury report for a team (ALWAYS check before recommending)
- set_injury: Record an injury status the user tells you about
- get_lineup: Get current roster for a team
- get_tonight_games: See what games are on tonight

INJURY PROTOCOL (CRITICAL):
1. ALWAYS call check_injuries(team) before recommending any player
2. If user tells you about an injury, call set_injury() to record it
3. Trust user overrides over scraped data
4. If a player is OUT, do not recommend them
5. If QUESTIONABLE, mention the uncertainty in your response

CRITICAL RULES:
1. NEVER discuss specific dollar amounts or bet sizing (legal requirement)
2. NEVER recommend assist props unless user specifically asks (historically low hit rate)
3. If you can't find a player or data, SAY SO - never make up numbers
4. Always cite "my model" or "the projections" - never say "I think" for numbers

COMMUNICATION STYLE:
- Casual but knowledgeable, like a sharp friend at a sportsbook
- Explain the "why" behind recommendations
- Use phrases like "my model shows" or "the numbers say"
- Acknowledge uncertainty when edge is small (<5% = "basically a coin flip")
- Keep responses concise - users want signal, not essays

ENTANGLEMENT (for parlays):
When building multi-leg parlays, look for CORRELATED picks:
- "Shootout thesis" = both teams' stars go over points if game is high-scoring
- "Blowout thesis" = bench players get more minutes if game is lopsided
- Explain the thesis so user understands why picks rise/fall together

EXAMPLE INTERACTION:
User: "Who should I bet on in the Celtics game?"
You: [check_injuries for both teams first]
     [simulate_prop for 2-3 interesting players]
     "With Porzingis out, I like Tatum over 27.5 tonight. My model has him
      at 28.2 projected with about 56% chance of hitting - that's a 6% edge
      over the book. Jaylen Brown's an interesting pivot at 22.5 if you want
      a safer play..."
"""

# =============================================================================
# TOOL DEFINITIONS
# =============================================================================

TOOLS = [
    {
        "name": "simulate_prop",
        "description": "Simulate a player prop bet and get probability of hitting over/under, plus edge vs the sportsbook. Returns projection, hit probability, and recommendation.",
        "input_schema": {
            "type": "object",
            "properties": {
                "player_name": {
                    "type": "string",
                    "description": "Full player name (e.g., 'LeBron James', 'Jayson Tatum')"
                },
                "stat": {
                    "type": "string",
                    "enum": ["pts", "reb", "ast", "stl", "blk", "tov", "fg3m"],
                    "description": "The stat type to simulate"
                },
                "line": {
                    "type": "number",
                    "description": "The betting line (e.g., 25.5)"
                },
                "opponent": {
                    "type": "string",
                    "description": "Opponent team abbreviation (e.g., 'BOS', 'LAL')"
                },
                "is_home": {
                    "type": "boolean",
                    "description": "Is the player's team playing at home?",
                    "default": True
                },
                "over_odds": {
                    "type": "string",
                    "description": "American odds for over (e.g., '-110')",
                    "default": "-110"
                },
                "under_odds": {
                    "type": "string",
                    "description": "American odds for under (e.g., '-110')",
                    "default": "-110"
                }
            },
            "required": ["player_name", "stat", "line", "opponent"]
        }
    },
    {
        "name": "get_player_projection",
        "description": "Get full stat projections for a player in an upcoming game. Returns predicted mean and distribution for all stats (pts, reb, ast, etc.).",
        "input_schema": {
            "type": "object",
            "properties": {
                "player_name": {
                    "type": "string",
                    "description": "Full player name"
                },
                "opponent": {
                    "type": "string",
                    "description": "Opponent team abbreviation"
                },
                "is_home": {
                    "type": "boolean",
                    "description": "Is the player's team playing at home?",
                    "default": True
                }
            },
            "required": ["player_name", "opponent"]
        }
    },
    {
        "name": "check_injuries",
        "description": "Get the current injury report for a team. ALWAYS call this before recommending any player to avoid suggesting players who are OUT.",
        "input_schema": {
            "type": "object",
            "properties": {
                "team": {
                    "type": "string",
                    "description": "Team abbreviation (e.g., 'LAL', 'BOS', 'MIA')"
                }
            },
            "required": ["team"]
        }
    },
    {
        "name": "set_injury",
        "description": "Manually set a player's injury status. Use when user provides injury info like 'AD is out tonight'.",
        "input_schema": {
            "type": "object",
            "properties": {
                "player_name": {
                    "type": "string",
                    "description": "Full player name"
                },
                "status": {
                    "type": "string",
                    "enum": ["OUT", "QUESTIONABLE", "PROBABLE", "AVAILABLE"],
                    "description": "Injury status"
                },
                "note": {
                    "type": "string",
                    "description": "Optional injury details (e.g., 'knee')"
                }
            },
            "required": ["player_name", "status"]
        }
    },
    {
        "name": "get_lineup",
        "description": "Get current roster for a team.",
        "input_schema": {
            "type": "object",
            "properties": {
                "team": {
                    "type": "string",
                    "description": "Team abbreviation (e.g., 'LAL')"
                }
            },
            "required": ["team"]
        }
    },
    {
        "name": "get_tonight_games",
        "description": "Get list of NBA games happening tonight with home/away teams.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "build_parlay",
        "description": "Analyze a multi-leg parlay for correlation and combined probability. Returns thesis explanation of why picks are correlated and recommendation.",
        "input_schema": {
            "type": "object",
            "properties": {
                "legs": {
                    "type": "array",
                    "description": "Array of prop bet legs to combine",
                    "items": {
                        "type": "object",
                        "properties": {
                            "player_name": {"type": "string"},
                            "stat": {"type": "string", "enum": ["pts", "reb", "ast", "stl", "blk", "tov", "fg3m"]},
                            "line": {"type": "number"},
                            "direction": {"type": "string", "enum": ["OVER", "UNDER"]},
                            "opponent": {"type": "string"},
                            "is_home": {"type": "boolean", "default": True}
                        },
                        "required": ["player_name", "stat", "line", "direction", "opponent"]
                    }
                }
            },
            "required": ["legs"]
        }
    }
]

# =============================================================================
# TOOL IMPLEMENTATIONS
# =============================================================================

def simulate_prop(
    player_name: str,
    stat: str,
    line: float,
    opponent: str,
    is_home: bool = True,
    over_odds: str = "-110",
    under_odds: str = "-110"
) -> dict:
    """
    Call the Monte Carlo engine to simulate a prop.
    """
    try:
        # Normalize opponent abbreviation
        opponent = opponent.upper()
        if opponent not in TEAM_ABBREVS:
            return {"error": f"Unknown team: {opponent}. Use 3-letter abbreviation like 'BOS', 'LAL'."}

        # Check if player is injured
        player_injury = INJURY_OVERRIDES.get(player_name.lower())
        if player_injury and player_injury.get("status") == "OUT":
            return {
                "error": f"{player_name} is marked as OUT. Cannot simulate prop.",
                "injury_info": player_injury
            }

        # Run simulation
        result = engine.simulate_player_prop(
            player_name=player_name,
            stat=stat,
            line=line,
            opponent=opponent,
            is_home=is_home,
            over_odds=over_odds,
            under_odds=under_odds
        )

        return result.to_dict()

    except ValueError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": f"Simulation failed: {str(e)}"}


def get_player_projection(
    player_name: str,
    opponent: str,
    is_home: bool = True
) -> dict:
    """
    Get full stat projections for a player.
    Runs simulation and returns distributions for all stats.
    """
    try:
        opponent = opponent.upper()
        if opponent not in TEAM_ABBREVS:
            return {"error": f"Unknown team: {opponent}"}

        # Check if player is injured
        player_injury = INJURY_OVERRIDES.get(player_name.lower())
        if player_injury and player_injury.get("status") == "OUT":
            return {
                "error": f"{player_name} is marked as OUT.",
                "injury_info": player_injury
            }

        # Get player features to find their team
        player = engine.transformer.get_player_features(player_name)
        if not player:
            return {"error": f"Could not find player: {player_name}"}

        # Determine home/away teams
        home_team = player.team_abbr if is_home else opponent
        away_team = opponent if is_home else player.team_abbr

        # Run full game simulation
        result = engine.simulate_game(
            home_team=home_team,
            away_team=away_team,
            n_simulations=5000  # Fewer sims for faster response
        )

        # Extract this player's predictions
        player_pred = result.players.get(player_name)
        if not player_pred:
            return {"error": f"Player {player_name} not found in simulation results"}

        return {
            "player": player_name,
            "team": player.team_abbr,
            "opponent": opponent,
            "is_home": is_home,
            "projections": {
                "minutes": player_pred.minutes.to_dict(),
                "pts": player_pred.pts.to_dict(),
                "reb": player_pred.reb.to_dict(),
                "ast": player_pred.ast.to_dict(),
                "stl": player_pred.stl.to_dict(),
                "blk": player_pred.blk.to_dict(),
                "tov": player_pred.tov.to_dict(),
                "fg3m": player_pred.fg3m.to_dict()
            }
        }

    except Exception as e:
        return {"error": f"Projection failed: {str(e)}"}


def scrape_espn_injuries() -> Dict[str, List]:
    """
    Scrape ESPN injury page for current injury data.
    Returns {team_abbrev: [{player, status, injury, details}]}
    """
    global ESPN_INJURY_CACHE, ESPN_CACHE_TIME

    # Use cache if less than 30 minutes old
    if ESPN_CACHE_TIME and (datetime.now() - ESPN_CACHE_TIME).seconds < 1800:
        return ESPN_INJURY_CACHE

    try:
        import requests
        from bs4 import BeautifulSoup

        url = "https://www.espn.com/nba/injuries"
        headers = {"User-Agent": "Mozilla/5.0"}

        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        injuries = {}

        # Find injury tables by team
        tables = soup.find_all("div", class_="ResponsiveTable")

        for table in tables:
            # Get team name from header
            header = table.find_previous("div", class_="injuries__teamName")
            if not header:
                continue

            team_name = header.get_text(strip=True).lower()
            team_abbrev = TEAM_NAME_TO_ABBREV.get(team_name)

            if not team_abbrev:
                continue

            injuries[team_abbrev] = []

            # Get injury rows
            rows = table.find_all("tr", class_="Table__TR")
            for row in rows:
                cols = row.find_all("td")
                if len(cols) >= 3:
                    player = cols[0].get_text(strip=True)
                    status = cols[1].get_text(strip=True)
                    injury = cols[2].get_text(strip=True) if len(cols) > 2 else ""

                    injuries[team_abbrev].append({
                        "player": player,
                        "status": status.upper(),
                        "injury": injury,
                        "source": "ESPN"
                    })

        ESPN_INJURY_CACHE = injuries
        ESPN_CACHE_TIME = datetime.now()
        return injuries

    except Exception as e:
        print(f"ESPN scrape failed: {e}")
        return ESPN_INJURY_CACHE or {}


def check_injuries(team: str) -> dict:
    """
    Get injury report for a team.
    Combines user overrides with ESPN scraped data.
    """
    team = team.upper()

    if team not in TEAM_ABBREVS:
        return {"error": f"Unknown team: {team}"}

    injuries = []

    # Get ESPN data
    espn_data = scrape_espn_injuries()
    espn_injuries = espn_data.get(team, [])

    # Track which players we've seen
    seen_players = set()

    # Add user overrides first (they take priority)
    for player_key, override in INJURY_OVERRIDES.items():
        # Check if this player is on the requested team
        # (For now, include all overrides - could be refined with team lookup)
        if override.get("status") != "AVAILABLE":
            injuries.append({
                "player": override.get("player_name", player_key),
                "status": override.get("status"),
                "injury": override.get("note", ""),
                "source": "USER_OVERRIDE"
            })
            seen_players.add(player_key)

    # Add ESPN injuries (if not overridden)
    for inj in espn_injuries:
        player_key = inj["player"].lower()
        if player_key not in seen_players:
            injuries.append(inj)

    return {
        "team": team,
        "team_name": TEAM_ABBREVS[team],
        "injuries": injuries,
        "count": len(injuries),
        "note": "No injuries reported" if not injuries else None
    }


def set_injury(player_name: str, status: str, note: str = "") -> dict:
    """
    Set a player's injury status (user override).
    """
    player_key = player_name.lower()

    if status == "AVAILABLE":
        # Remove any existing override
        if player_key in INJURY_OVERRIDES:
            del INJURY_OVERRIDES[player_key]
        return {
            "success": True,
            "message": f"{player_name} marked as AVAILABLE (cleared injury status)"
        }

    INJURY_OVERRIDES[player_key] = {
        "player_name": player_name,
        "status": status.upper(),
        "note": note,
        "source": "USER_OVERRIDE"
    }

    return {
        "success": True,
        "message": f"Recorded: {player_name} is {status}" + (f" ({note})" if note else "")
    }


def get_lineup(team: str) -> dict:
    """
    Get current roster for a team using nba_api.
    """
    team = team.upper()

    if team not in TEAM_ABBREVS:
        return {"error": f"Unknown team: {team}"}

    try:
        from nba_api.stats.endpoints import CommonTeamRoster
        from nba_api.stats.static import teams

        # Get team ID
        nba_teams = teams.get_teams()
        team_info = next((t for t in nba_teams if t["abbreviation"] == team), None)

        if not team_info:
            return {"error": f"Could not find team ID for {team}"}

        roster = CommonTeamRoster(team_id=team_info["id"], season="2024-25")
        df = roster.get_data_frames()[0]

        players = []
        for _, row in df.iterrows():
            player_name = row["PLAYER"]

            # Check injury status
            player_key = player_name.lower()
            injury_info = INJURY_OVERRIDES.get(player_key)

            players.append({
                "name": player_name,
                "number": row.get("NUM", ""),
                "position": row.get("POSITION", ""),
                "injury_status": injury_info.get("status") if injury_info else "AVAILABLE"
            })

        return {
            "team": team,
            "team_name": TEAM_ABBREVS[team],
            "roster": players,
            "count": len(players)
        }

    except Exception as e:
        return {"error": f"Failed to get roster: {str(e)}"}


def get_tonight_games() -> dict:
    """
    Get tonight's NBA games using nba_api.
    """
    try:
        from nba_api.stats.endpoints import ScoreboardV2

        today = date.today().strftime("%m/%d/%Y")
        scoreboard = ScoreboardV2(game_date=today)

        games_df = scoreboard.get_data_frames()[0]  # GameHeader

        if games_df.empty:
            return {
                "date": str(date.today()),
                "games": [],
                "message": "No NBA games scheduled for today"
            }

        games = []
        for _, row in games_df.iterrows():
            games.append({
                "home": row.get("HOME_TEAM_ABBREVIATION", row.get("HOME_TEAM_ID", "")),
                "away": row.get("VISITOR_TEAM_ABBREVIATION", row.get("VISITOR_TEAM_ID", "")),
                "status": row.get("GAME_STATUS_TEXT", ""),
                "game_id": row.get("GAME_ID", "")
            })

        return {
            "date": str(date.today()),
            "games": games,
            "count": len(games)
        }

    except Exception as e:
        return {"error": f"Failed to get schedule: {str(e)}"}


def build_parlay(legs: List[dict]) -> dict:
    """
    Analyze a multi-leg parlay for correlation and combined probability.
    """
    try:
        if not legs or len(legs) < 2:
            return {"error": "Parlay requires at least 2 legs"}

        parlay_legs = []

        # Simulate each leg and build ParlayLeg objects
        for leg_data in legs:
            player_name = leg_data["player_name"]
            stat = leg_data["stat"]
            line = leg_data["line"]
            direction = leg_data["direction"].upper()
            opponent = leg_data["opponent"].upper()
            is_home = leg_data.get("is_home", True)

            # Check if player is injured
            player_injury = INJURY_OVERRIDES.get(player_name.lower())
            if player_injury and player_injury.get("status") == "OUT":
                return {
                    "error": f"Cannot include {player_name} in parlay - marked as OUT",
                    "injury_info": player_injury
                }

            # Get player's team
            player = engine.transformer.get_player_features(player_name)
            if not player:
                return {"error": f"Could not find player: {player_name}"}

            # Simulate the prop
            result = engine.simulate_player_prop(
                player_name=player_name,
                stat=stat,
                line=line,
                opponent=opponent,
                is_home=is_home
            )

            # Determine which probability to use based on direction
            if direction == "OVER":
                model_prob = result.model_prob_over
                edge = result.edge_over
            else:
                model_prob = result.model_prob_under
                edge = result.edge_under

            parlay_legs.append(ParlayLeg(
                player_name=player_name,
                stat=stat,
                line=line,
                direction=direction,
                team=player.team_abbr,
                opponent=opponent,
                is_home=is_home,
                model_prob=model_prob,
                edge=edge
            ))

        # Analyze the parlay
        analysis = parlay_analyzer.analyze_parlay(parlay_legs)

        return parlay_analyzer.to_dict(analysis)

    except Exception as e:
        return {"error": f"Parlay analysis failed: {str(e)}"}


def execute_tool(tool_name: str, tool_input: dict) -> Any:
    """Route tool calls to implementations."""
    if tool_name == "simulate_prop":
        return simulate_prop(**tool_input)
    elif tool_name == "get_player_projection":
        return get_player_projection(**tool_input)
    elif tool_name == "check_injuries":
        return check_injuries(**tool_input)
    elif tool_name == "set_injury":
        return set_injury(**tool_input)
    elif tool_name == "get_lineup":
        return get_lineup(**tool_input)
    elif tool_name == "get_tonight_games":
        return get_tonight_games()
    elif tool_name == "build_parlay":
        return build_parlay(**tool_input)
    else:
        return {"error": f"Unknown tool: {tool_name}"}


# =============================================================================
# MAIN CHAT LOOP
# =============================================================================

def chat(user_message: str, history: list) -> tuple[str, list]:
    """
    Send a message and get a response, handling tool calls.

    Returns: (response_text, updated_history)
    """
    try:
        import anthropic
        client = anthropic.Anthropic()
    except ImportError:
        print("ERROR: pip install anthropic")
        return "Error: Anthropic SDK not installed", history

    # Add user message to history
    messages = history + [{"role": "user", "content": user_message}]

    # Initial API call
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        tools=TOOLS,
        messages=messages
    )

    # Handle tool use loop
    while response.stop_reason == "tool_use":
        # Extract tool calls
        tool_calls = [block for block in response.content if block.type == "tool_use"]

        # Execute each tool
        tool_results = []
        for tool_call in tool_calls:
            print(f"  [Calling {tool_call.name}...]")
            result = execute_tool(tool_call.name, tool_call.input)
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tool_call.id,
                "content": json.dumps(result)
            })

        # Add assistant response and tool results to messages
        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": tool_results})

        # Continue conversation
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages
        )

    # Extract text response
    text_response = ""
    for block in response.content:
        if hasattr(block, "text"):
            text_response += block.text

    # Update history
    messages.append({"role": "assistant", "content": response.content})

    return text_response, messages


def main():
    """Simple CLI interface."""
    print("=" * 60)
    print("  THE BRAIN v0.1 - NBA Betting Co-Pilot")
    print("=" * 60)
    print("\nCommands:")
    print("  'quit' - Exit")
    print("  'clear' - Reset conversation")
    print("  'injuries' - Clear all injury overrides")
    print()

    history = []

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() == "quit":
            print("Goodbye!")
            break

        if user_input.lower() == "clear":
            history = []
            print("Conversation cleared.")
            continue

        if user_input.lower() == "injuries":
            INJURY_OVERRIDES.clear()
            print("Injury overrides cleared.")
            continue

        print("\nBrain: ", end="", flush=True)
        response, history = chat(user_input, history)
        print(response)


if __name__ == "__main__":
    main()
