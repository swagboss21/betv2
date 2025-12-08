"""
LLM Tool definitions and execution for The Brain chatbot.
"""
import json
from typing import Any

from api.queries import (
    get_games_today,
    get_projection,
    get_best_props,
    get_injuries,
    save_bet,
)


# Tool definitions for Claude function calling
TOOLS = [
    {
        "name": "get_games_today",
        "description": "Get list of NBA games scheduled for today with home/away teams and status.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "get_projection",
        "description": "Get projection for a player and stat. If line provided, returns prob_over/prob_under for that specific line.",
        "input_schema": {
            "type": "object",
            "properties": {
                "player_name": {
                    "type": "string",
                    "description": "Full player name (e.g., 'Stephen Curry')"
                },
                "stat_type": {
                    "type": "string",
                    "enum": ["pts", "reb", "ast", "stl", "blk", "tov", "fg3m"],
                    "description": "The stat type to get projection for"
                },
                "line": {
                    "type": "number",
                    "description": "Optional betting line to calculate probability (supports any line value)"
                }
            },
            "required": ["player_name", "stat_type"]
        }
    },
    {
        "name": "get_best_props",
        "description": "Get props with highest edge for tonight. Returns sorted by edge descending.",
        "input_schema": {
            "type": "object",
            "properties": {
                "min_edge": {
                    "type": "number",
                    "description": "Minimum edge threshold (default 0.05 = 5%)"
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results (default 10)"
                }
            },
            "required": []
        }
    },
    {
        "name": "get_injuries",
        "description": "Get injury report for a team. ALWAYS call this before recommending players.",
        "input_schema": {
            "type": "object",
            "properties": {
                "team": {
                    "type": "string",
                    "description": "3-letter team code (LAL, BOS, GSW, etc.)"
                }
            },
            "required": ["team"]
        }
    },
    {
        "name": "lock_bet",
        "description": "Save a locked bet for the user. Call this when user confirms they want to bet.",
        "input_schema": {
            "type": "object",
            "properties": {
                "player_name": {"type": "string"},
                "stat_type": {"type": "string"},
                "line": {"type": "number"},
                "direction": {"type": "string", "enum": ["OVER", "UNDER"]}
            },
            "required": ["player_name", "stat_type", "line", "direction"]
        }
    }
]


def execute_tool(tool_name: str, tool_input: dict, user_id: str = None) -> Any:
    """
    Execute a tool and return the result.

    Args:
        tool_name: Name of the tool to execute
        tool_input: Tool input parameters
        user_id: User ID for bet locking (optional)

    Returns:
        Tool result (dict or list)
    """
    try:
        if tool_name == "get_games_today":
            games = get_games_today()
            return {"games": games, "count": len(games)}

        elif tool_name == "get_projection":
            projection = get_projection(
                player_name=tool_input.get("player_name"),
                stat_type=tool_input.get("stat_type"),
                line=tool_input.get("line")
            )
            if projection is None:
                return {"error": f"No projection found for {tool_input.get('player_name')} {tool_input.get('stat_type')}"}
            return projection

        elif tool_name == "get_best_props":
            props = get_best_props(
                min_edge=tool_input.get("min_edge", 0.05),
                limit=tool_input.get("limit", 10)
            )
            return {"props": props, "count": len(props)}

        elif tool_name == "get_injuries":
            injuries = get_injuries(team=tool_input.get("team", "").upper())
            return {
                "team": tool_input.get("team", "").upper(),
                "injuries": injuries,
                "count": len(injuries)
            }

        elif tool_name == "lock_bet":
            if not user_id:
                return {"error": "User not authenticated"}
            bet_id = save_bet(user_id, {
                "player_name": tool_input.get("player_name"),
                "stat_type": tool_input.get("stat_type"),
                "line": tool_input.get("line"),
                "direction": tool_input.get("direction"),
                "odds": "-110",
                "edge_pct": tool_input.get("edge_pct", 0)
            })
            return {
                "success": True,
                "bet_id": bet_id,
                "message": f"Locked: {tool_input.get('player_name')} {tool_input.get('direction')} {tool_input.get('line')} {tool_input.get('stat_type')}"
            }

        else:
            return {"error": f"Unknown tool: {tool_name}"}

    except Exception as e:
        return {"error": str(e)}
