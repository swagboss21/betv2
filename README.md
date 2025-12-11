# The Brain v2 - NBA Betting Co-Pilot

Data-first NBA player prop betting assistant. Aggregates odds from multiple sportsbooks to find the best available lines.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set up PostgreSQL
docker run -d --name brain-postgres \
  -e POSTGRES_PASSWORD=brain123 \
  -e POSTGRES_DB=brain \
  -p 5432:5432 postgres:15

# 3. Initialize database
python db/init_db.py

# 4. Run pipeline (fetches games, injuries, props)
/opt/homebrew/bin/python3.11 batch/orchestrate.py

# 5. Use with Claude Desktop
# Add .mcp.json config to Claude Desktop settings
```

## How It Works

```
User Query → Claude → MCP Tools → PostgreSQL → Response
                           ↑
                     SGO API (batch)
```

1. **User asks**: "Best LeBron prop tonight?" or "Where's the best line on Curry points?"
2. **Claude calls tools**: Fetches props, checks injuries
3. **Data layer returns**: Best lines across books, consensus odds
4. **Response**: Line shopping recommendations with injury context

## Project Structure

```
the-brain/
├── brain_mcp/      # MCP server (Claude Desktop integration)
├── api/            # Database queries
├── batch/          # Pipeline jobs (games, injuries, props)
├── db/             # PostgreSQL schema
└── tests/          # Test suite
```

## Key Files

| File | Purpose |
|------|---------|
| `CLAUDE.md` | Project context, architecture, API reference |
| `brain_mcp/server.py` | MCP server (5 tools) |
| `batch/orchestrate.py` | Pipeline entry point |
| `api/queries.py` | Database query functions |

## MCP Tools

| Tool | Description |
|------|-------------|
| `get_games` | Tonight's NBA schedule |
| `get_props` | Props with best lines across books |
| `get_injuries` | Team injury report |
| `get_tonight_injuries` | All injuries for tonight |
| `get_player_analysis` | Deep dive on one player |

## Example Prompts

Try these with Claude Desktop:

- "What games are on tonight?"
- "Best line on LeBron points?"
- "Where do books disagree most on Curry?"
- "Check Lakers injuries"
- "Show me all Jokic props"

## Current Status

| Component | Status |
|-----------|--------|
| Database (PostgreSQL) | Ready |
| SGO Integration | Ready |
| MCP Server | Ready |
| Pipeline | Ready |

## Philosophy

Sportsbooks have already done the math. Our value is:
- **Curation**: Filter noise, surface what matters
- **Structure**: Clean data for LLM reasoning
- **Line shopping**: Find best odds across books
