# The Brain - NBA Betting Co-Pilot

AI-powered NBA player prop betting assistant using Monte Carlo simulation + LLM.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set up PostgreSQL (Docker)
docker run -d --name brain-postgres \
  -e POSTGRES_PASSWORD=brain123 \
  -e POSTGRES_DB=brain \
  -p 5432:5432 postgres:15

# 3. Initialize database
python db/init_db.py

# 4. Use with Claude Desktop (MCP)
# Add to Claude Desktop config (~/.config/claude/claude_desktop_config.json):
# See .mcp.json for server configuration
```

## How It Works

```
User Query → LLM (Claude) → Tools → Monte Carlo Engine → Response
```

1. **User asks**: "Best props tonight?" or "4-leg parlay for Lakers game"
2. **LLM calls tools**: Fetches projections, checks injuries
3. **Engine runs**: 10K Monte Carlo simulations per prop
4. **Response**: Concise recommendation with probability + edge

## Project Structure

```
the-brain/
├── brain_mcp/      # MCP server (Claude Desktop integration)
├── simulation/     # Monte Carlo engine (10K sims)
├── api/            # Database queries
├── batch/          # Scheduled jobs (precompute, injuries)
├── db/             # PostgreSQL schema
├── models/         # Trained XGBoost models
├── tests/          # Test suite (pytest)
├── scripts/        # Model training pipeline
└── docs/           # Sprint plans & architecture
```

## Key Files

| File | Purpose |
|------|---------|
| `CLAUDE.md` | Project context & locked decisions |
| `brain_mcp/server.py` | MCP server (5 tools for Claude Desktop) |
| `simulation/engine.py` | Monte Carlo simulation engine |
| `api/queries.py` | Database query functions |

## Documentation

- `CLAUDE.md` - Project context, architecture, locked decisions
- `docs/files/SPRINT_3.md` - MCP server implementation (complete)
- `docs/files/SPRINTS_4_5_6.md` - Future sprints (results, bots, auth)

## Example Prompts

Try these with Claude Desktop:
- "What games are on tonight?"
- "Should I bet LeBron over 25.5 points vs Boston?"
- "Check Lakers injuries"
- "Best props for tonight?"

## Current Status

**Sprint 3 Complete** - MCP server ready for Claude Desktop

| Component | Status |
|-----------|--------|
| Monte Carlo Engine | ✅ Complete |
| Database + API | ✅ Complete |
| LLM Integration | ✅ Complete |
| MCP Server | ✅ Complete |

## What NOT to Do

- Don't rebuild the simulation engine (it works)
- Don't switch LLM providers without testing
- Don't add features before validating core loop with real users
- Don't make the LLM do math (engine does math, LLM explains)

## GitHub

https://github.com/swagboss21/betv2.git
