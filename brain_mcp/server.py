"""
MCP Server for The Brain - NBA Betting Co-Pilot.

Exposes Monte Carlo projections and betting tools to Claude.
Run with: python -m mcp.server
"""
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from api.queries import (
    get_games_today,
    get_projection,
    get_best_props,
    get_injuries,
    get_tonight_injuries,
    save_bet,
)

# Create MCP server
server = Server("the-brain")


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools."""
    return [
        Tool(
            name="get_games_today",
            description="Get NBA games scheduled for today with home/away teams and game status. Call this first to see what games are available.",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        Tool(
            name="get_projection",
            description="Get Monte Carlo projection for a player's stat. Returns mean, std dev, percentiles (p10-p90). If line provided, returns prob_over/prob_under for that line. Use this to analyze specific player props.",
            inputSchema={
                "type": "object",
                "properties": {
                    "player_name": {
                        "type": "string",
                        "description": "Full player name (e.g., 'Stephen Curry', 'LeBron James')"
                    },
                    "stat_type": {
                        "type": "string",
                        "enum": ["pts", "reb", "ast", "stl", "blk", "tov", "fg3m"],
                        "description": "Stat type: pts=points, reb=rebounds, ast=assists, stl=steals, blk=blocks, tov=turnovers, fg3m=3-pointers made"
                    },
                    "line": {
                        "type": "number",
                        "description": "Optional betting line to calculate over/under probability"
                    }
                },
                "required": ["player_name", "stat_type"]
            }
        ),
        Tool(
            name="get_best_props",
            description="Get player props with highest edge for tonight's games. Returns props sorted by edge (model probability - implied book probability). Use this to find value bets.",
            inputSchema={
                "type": "object",
                "properties": {
                    "min_edge": {
                        "type": "number",
                        "description": "Minimum edge threshold as decimal (default 0.05 = 5% edge)"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of props to return (default 10)"
                    }
                },
                "required": []
            }
        ),
        Tool(
            name="get_injuries",
            description="Get injury report for a team. CRITICAL: Always check injuries before recommending player props. Returns player name, status (OUT/DOUBTFUL/QUESTIONABLE/PROBABLE), and injury description.",
            inputSchema={
                "type": "object",
                "properties": {
                    "team": {
                        "type": "string",
                        "description": "3-letter team code: ATL, BOS, BKN, CHA, CHI, CLE, DAL, DEN, DET, GSW, HOU, IND, LAC, LAL, MEM, MIA, MIL, MIN, NOP, NYK, OKC, ORL, PHI, PHX, POR, SAC, SAS, TOR, UTA, WAS"
                    }
                },
                "required": ["team"]
            }
        ),
        Tool(
            name="get_tonight_injuries",
            description="Get injury reports for all teams playing tonight, grouped by game. Returns injuries for both home and away teams for each scheduled game. More efficient than calling get_injuries for each team separately.",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        Tool(
            name="lock_bet",
            description="Save a bet that the user wants to track. Only call this when the user explicitly confirms they want to lock/save a bet.",
            inputSchema={
                "type": "object",
                "properties": {
                    "player_name": {
                        "type": "string",
                        "description": "Player name"
                    },
                    "stat_type": {
                        "type": "string",
                        "enum": ["pts", "reb", "ast", "stl", "blk", "tov", "fg3m"],
                        "description": "Stat type"
                    },
                    "line": {
                        "type": "number",
                        "description": "The betting line"
                    },
                    "direction": {
                        "type": "string",
                        "enum": ["OVER", "UNDER"],
                        "description": "OVER or UNDER the line"
                    }
                },
                "required": ["player_name", "stat_type", "line", "direction"]
            }
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Execute a tool and return results."""
    import json

    try:
        if name == "get_games_today":
            games = get_games_today()
            # Format games nicely
            if not games:
                result = {"games": [], "message": "No games scheduled for today"}
            else:
                result = {
                    "games": [
                        {
                            "matchup": f"{g['away_team']} @ {g['home_team']}",
                            "starts_at": str(g['starts_at']),
                            "status": g['status'],
                            "game_id": g['id']
                        }
                        for g in games
                    ],
                    "count": len(games)
                }

        elif name == "get_projection":
            player_name = arguments.get("player_name")
            stat_type = arguments.get("stat_type")
            line = arguments.get("line")

            projection = get_projection(player_name, stat_type, line=line)

            if projection is None:
                result = {"error": f"No projection found for {player_name} {stat_type}. Check spelling or player may not be playing today."}
            else:
                result = {
                    "player": projection['player_name'],
                    "stat": projection['stat_type'],
                    "projection": {
                        "mean": round(projection['mean'], 1),
                        "std": round(projection['std'], 1),
                        "p10": round(projection['p10'], 1),
                        "p50": round(projection['p50'], 1),
                        "p90": round(projection['p90'], 1)
                    }
                }
                if line is not None:
                    result["line_analysis"] = {
                        "line": line,
                        "prob_over": f"{projection['prob_over']*100:.1f}%",
                        "prob_under": f"{projection['prob_under']*100:.1f}%"
                    }

        elif name == "get_best_props":
            min_edge = arguments.get("min_edge", 0.05)
            limit = arguments.get("limit", 10)

            props = get_best_props(min_edge=min_edge, limit=limit)

            result = {
                "props": [
                    {
                        "player": p['player_name'],
                        "stat": p['stat_type'],
                        "line": p['line'],
                        "direction": p['direction'],
                        "probability": f"{p['probability']*100:.1f}%",
                        "edge": p['edge_formatted']
                    }
                    for p in props
                ],
                "count": len(props),
                "min_edge_used": f"{min_edge*100:.0f}%"
            }

        elif name == "get_injuries":
            team = arguments.get("team", "").upper()
            injuries = get_injuries(team)

            result = {
                "team": team,
                "injuries": [
                    {
                        "player": inj['player_name'],
                        "status": inj['status'],
                        "injury": inj['injury']
                    }
                    for inj in injuries
                ],
                "count": len(injuries)
            }
            if not injuries:
                result["message"] = f"No injuries reported for {team}"

        elif name == "get_tonight_injuries":
            games_injuries = get_tonight_injuries()

            result = {
                "games": [
                    {
                        "game_id": g['game_id'],
                        "matchup": g['matchup'],
                        "starts_at": str(g['starts_at']),
                        "home_team": g['home_team'],
                        "away_team": g['away_team'],
                        "home_injuries": [
                            {
                                "player": inj['player_name'],
                                "status": inj['status'],
                                "injury": inj['injury']
                            }
                            for inj in g['home_injuries']
                        ],
                        "away_injuries": [
                            {
                                "player": inj['player_name'],
                                "status": inj['status'],
                                "injury": inj['injury']
                            }
                            for inj in g['away_injuries']
                        ],
                        "total_injuries": len(g['home_injuries']) + len(g['away_injuries'])
                    }
                    for g in games_injuries
                ],
                "count": len(games_injuries)
            }
            if not games_injuries:
                result["message"] = "No games scheduled for tonight"

        elif name == "lock_bet":
            # Use a default user_id for MCP (can be enhanced later)
            default_user_id = "mcp-user-001"

            bet_id = save_bet(default_user_id, {
                "player_name": arguments.get("player_name"),
                "stat_type": arguments.get("stat_type"),
                "line": arguments.get("line"),
                "direction": arguments.get("direction"),
                "odds": "-110",
                "edge_pct": 0
            })

            result = {
                "success": True,
                "bet_id": bet_id,
                "locked": f"{arguments.get('player_name')} {arguments.get('direction')} {arguments.get('line')} {arguments.get('stat_type')}"
            }

        else:
            result = {"error": f"Unknown tool: {name}"}

        return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]

    except Exception as e:
        return [TextContent(type="text", text=json.dumps({"error": str(e)}))]


async def main():
    """Run the MCP server."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
