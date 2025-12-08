# SPRINT 3: THE SALESPERSON (Chat MVP)

**Goal:** Chat interface that users actually interact with.
**Duration:** 3-4 days
**Blocker Risk:** Medium — LLM tuning

**THIS IS YOUR MVP.** After Sprint 3, you can share with friends and get feedback.

---

## Sprint Summary

| Task ID | Task | Parallel? | Depends On | Output |
|---------|------|-----------|------------|--------|
| S3-T1 | Create Streamlit app shell | Yes | — | `chat/app.py` |
| S3-T2 | Create LLM tools (function calling) | Yes | Sprint 2 | `chat/tools.py` |
| S3-T3 | Create system prompt | Yes | — | `chat/prompts.py` |
| S3-T4 | Wire LLM + tools together | No | S3-T1-T3 | Updated `chat/app.py` |
| S3-T5 | Implement "Lock it" flow | No | S3-T4 | Updated `chat/app.py` |

---

## 🔲 HUMAN CHECKPOINTS

Before executing Sprint 3, confirm:
- [ ] Using Streamlit (not Flask/Next.js)
- [ ] Using Claude Haiku for cost ($0.25/1M input, $1.25/1M output)
- [ ] OK with guided first message (buttons to reduce garbage input)

After S3-T3, confirm:
- [ ] Review and approve system prompt before proceeding

---

## Task Cards

---

### S3-T1: Create Streamlit App Shell

**Type:** UI Setup
**Input:** Streamlit docs
**Output:** `chat/app.py`

**Task Prompt for Sub-Agent:**
```
Create a basic Streamlit chat app in chat/app.py:

import streamlit as st
from datetime import datetime

# Page config
st.set_page_config(
    page_title="The Brain - NBA Betting Co-Pilot",
    page_icon="🧠",
    layout="wide"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "user_id" not in st.session_state:
    st.session_state.user_id = None  # Will be set after auth in Sprint 6

# Sidebar
with st.sidebar:
    st.title("🧠 The Brain")
    st.caption("NBA Betting Co-Pilot")
    
    # Tonight's games placeholder
    st.subheader("Tonight's Games")
    st.info("Games will appear here after connecting to API")
    
    # Locked bets placeholder
    st.subheader("Your Locked Bets")
    st.caption("No bets locked yet")

# Main chat area
st.title("💬 Chat")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Guided first message (if no history)
if not st.session_state.messages:
    st.write("**Quick Start:**")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("📊 Tonight's Best Props"):
            # Will be handled in S3-T4
            pass
    with col2:
        if st.button("🏀 What Games Tonight?"):
            pass
    with col3:
        if st.button("🎯 Help Me Build a Parlay"):
            pass

# Chat input
if prompt := st.chat_input("Ask about tonight's games..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
    
    # Placeholder response
    response = "I'm not connected to the brain yet. Coming in S3-T4!"
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.write(response)

# Run with: streamlit run chat/app.py
```

**Verification:**
```bash
streamlit run chat/app.py &
sleep 3
curl -s http://localhost:8501 | grep -q "The Brain" && echo "✅ PASS: Streamlit running" || echo "❌ FAIL"
kill %1
```

---

### S3-T2: Create LLM Tools (Function Calling)

**Type:** Tool Definitions
**Input:** Sprint 2 API functions
**Output:** `chat/tools.py`

**Task Prompt for Sub-Agent:**
```
Create chat/tools.py with tool definitions for Claude:

from api import queries, probability
from typing import Any

# Tool definitions for Claude function calling
TOOLS = [
    {
        "name": "get_games_today",
        "description": "Get list of NBA games scheduled for today",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "get_projection",
        "description": "Get projection for a specific player and stat. Returns mean, std, percentiles.",
        "input_schema": {
            "type": "object",
            "properties": {
                "player_name": {"type": "string", "description": "Full player name, e.g. 'LeBron James'"},
                "stat_type": {"type": "string", "enum": ["pts", "reb", "ast", "stl", "blk", "tov", "fg3m"]}
            },
            "required": ["player_name", "stat_type"]
        }
    },
    {
        "name": "get_best_props",
        "description": "Get the best betting props for tonight based on edge. Use this when user asks for recommendations.",
        "input_schema": {
            "type": "object",
            "properties": {
                "min_edge": {"type": "number", "description": "Minimum edge threshold (default 0.05 = 5%)"},
                "limit": {"type": "integer", "description": "Max number of props to return (default 5)"}
            },
            "required": []
        }
    },
    {
        "name": "get_injuries",
        "description": "Get injury report for a team",
        "input_schema": {
            "type": "object",
            "properties": {
                "team": {"type": "string", "description": "3-letter team code, e.g. 'LAL', 'BOS'"}
            },
            "required": ["team"]
        }
    },
    {
        "name": "calculate_probability",
        "description": "Calculate probability of hitting a line given projection. Use after getting projection.",
        "input_schema": {
            "type": "object",
            "properties": {
                "mean": {"type": "number"},
                "std": {"type": "number"},
                "line": {"type": "number"},
                "direction": {"type": "string", "enum": ["over", "under"]}
            },
            "required": ["mean", "std", "line", "direction"]
        }
    },
    {
        "name": "lock_bet",
        "description": "Lock a bet for the user. Call this when user confirms they want to bet.",
        "input_schema": {
            "type": "object",
            "properties": {
                "player_name": {"type": "string"},
                "stat_type": {"type": "string"},
                "line": {"type": "number"},
                "direction": {"type": "string", "enum": ["OVER", "UNDER"]},
                "odds": {"type": "string", "description": "American odds, e.g. '-115'"}
            },
            "required": ["player_name", "stat_type", "line", "direction"]
        }
    }
]

def execute_tool(tool_name: str, args: dict, user_id: str = None) -> Any:
    """Execute a tool and return result."""
    
    if tool_name == "get_games_today":
        games = queries.get_games_today()
        if not games:
            return "No games found for today. The precompute may not have run yet."
        return games
    
    elif tool_name == "get_projection":
        proj = queries.get_projection(args["player_name"], args["stat_type"])
        if not proj:
            return f"No projection found for {args['player_name']} {args['stat_type']}. Check spelling or they may not be playing tonight."
        return proj
    
    elif tool_name == "get_best_props":
        return queries.get_best_props(
            min_edge=args.get("min_edge", 0.05),
            limit=args.get("limit", 5)
        )
    
    elif tool_name == "get_injuries":
        return queries.get_injuries(args["team"])
    
    elif tool_name == "calculate_probability":
        if args["direction"].lower() == "over":
            prob = probability.prob_over(args["mean"], args["std"], args["line"])
        else:
            prob = probability.prob_under(args["mean"], args["std"], args["line"])
        return {"probability": round(prob, 3), "percentage": f"{prob*100:.1f}%"}
    
    elif tool_name == "lock_bet":
        if not user_id:
            return {"error": "User not authenticated. Cannot lock bet."}
        
        # Find game_id for this player
        proj = queries.get_projection(args["player_name"], args["stat_type"])
        if not proj:
            return {"error": f"No projection found for {args['player_name']}"}
        
        bet_id = queries.save_bet(user_id, {
            "game_id": proj["game_id"],
            "player_name": args["player_name"],
            "stat_type": args["stat_type"],
            "line": args["line"],
            "direction": args["direction"],
            "odds": args.get("odds", "-110")
        })
        return {"success": True, "bet_id": bet_id, "message": f"🔒 Locked: {args['player_name']} {args['stat_type']} {args['direction']} {args['line']}"}
    
    else:
        return {"error": f"Unknown tool: {tool_name}"}
```

**Verification:**
```bash
python -c "
from chat.tools import TOOLS, execute_tool
print(f'Defined {len(TOOLS)} tools')
# Test a tool
result = execute_tool('get_games_today', {})
print(f'get_games_today result: {result}')
print('✅ PASS')
"
```

---

### S3-T3: Create System Prompt

**Type:** Prompt Engineering
**Input:** Existing poc.py system prompt
**Output:** `chat/prompts.py`

**Task Prompt for Sub-Agent:**
```
Create chat/prompts.py with the system prompt:

Reference the existing SYSTEM_PROMPT in poc.py and adapt it for the new tool-based architecture.

SYSTEM_PROMPT = '''You are "The Brain" - an AI sports betting co-pilot for NBA player props.

## Your Role
You help users make informed betting decisions using pre-computed Monte Carlo simulations.
You are NOT a guaranteed winner. You provide analysis, not financial advice.

## How You Work
1. Projections are pre-computed daily using 10,000 Monte Carlo simulations
2. You have tools to look up projections, calculate probabilities, and lock bets
3. The MODEL does the MATH. You do the TALKING (explaining results in plain English).

## Your Personality
- Confident but humble (you're data-driven, not psychic)
- Concise (users want quick answers, not essays)
- Honest about uncertainty (if edge is small, say so)
- Fun (sprinkle in sports banter when appropriate)

## Edge Thresholds
- 5%+ edge: Strong recommendation
- 3-5% edge: Lean / slight edge
- <3% edge: PASS (no clear edge)

## Injury Awareness
Always check injuries before making recommendations. An injured star changes everything.

## When User Asks for Recommendations
1. Call get_best_props to get highest-edge opportunities
2. Explain WHY each prop has edge (mean vs line, injury impact, etc.)
3. Ask if they want to "lock" any bet

## When User Asks About Specific Player
1. Call get_projection for that player/stat
2. If they mention a line, call calculate_probability
3. Give a clear recommendation: OVER, UNDER, or PASS

## Lock Flow
When user says "lock it", "I'll take it", or similar:
1. Confirm the exact bet (player, stat, line, direction)
2. Call lock_bet tool
3. Confirm the bet is saved

## Important Rules
- Never fabricate projections. If tool returns no data, say so.
- Never guarantee wins. Betting involves risk.
- Be responsive to follow-up questions.
- Keep responses under 200 words unless user asks for detail.
'''

def get_system_prompt() -> str:
    return SYSTEM_PROMPT
```

**Verification:**
```bash
python -c "
from chat.prompts import get_system_prompt
prompt = get_system_prompt()
assert 'Monte Carlo' in prompt
assert 'lock_bet' in prompt
print('✅ PASS: System prompt contains key elements')
"
```

---

### S3-T4: Wire LLM + Tools Together

**Type:** Integration
**Input:** S3-T1, S3-T2, S3-T3
**Output:** Updated `chat/app.py`

**Task Prompt for Sub-Agent:**
```
Update chat/app.py to connect to Claude API with tools:

Add to imports:
import anthropic
import json
from chat.tools import TOOLS, execute_tool
from chat.prompts import get_system_prompt

Add API key handling (before page config):
import os
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
if not ANTHROPIC_API_KEY:
    st.error("ANTHROPIC_API_KEY not set!")
    st.stop()

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

Replace the chat input handler:

if prompt := st.chat_input("Ask about tonight's games..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
    
    # Build messages for Claude
    messages = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
    
    # Call Claude with tools
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = client.messages.create(
                model="claude-3-haiku-20240307",  # Use Haiku for cost
                max_tokens=1024,
                system=get_system_prompt(),
                tools=TOOLS,
                messages=messages
            )
            
            # Handle tool use
            while response.stop_reason == "tool_use":
                # Get tool call
                tool_block = next(b for b in response.content if b.type == "tool_use")
                tool_name = tool_block.name
                tool_args = tool_block.input
                tool_id = tool_block.id
                
                # Execute tool
                tool_result = execute_tool(
                    tool_name, 
                    tool_args, 
                    user_id=st.session_state.get("user_id")
                )
                
                # Add assistant message with tool use
                messages.append({
                    "role": "assistant",
                    "content": response.content
                })
                
                # Add tool result
                messages.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": tool_id,
                        "content": json.dumps(tool_result) if not isinstance(tool_result, str) else tool_result
                    }]
                })
                
                # Continue conversation
                response = client.messages.create(
                    model="claude-3-haiku-20240307",
                    max_tokens=1024,
                    system=get_system_prompt(),
                    tools=TOOLS,
                    messages=messages
                )
            
            # Get final text response
            final_text = next((b.text for b in response.content if hasattr(b, 'text')), "")
            st.write(final_text)
    
    # Save to history
    st.session_state.messages.append({"role": "assistant", "content": final_text})
```

**Verification:**
```bash
# Manual test:
streamlit run chat/app.py
# Then ask: "What games are on tonight?"
# Should see a response based on database data
```

---

### S3-T5: Implement "Lock It" Flow

**Type:** UX Enhancement
**Input:** S3-T4
**Output:** Updated `chat/app.py`

**Task Prompt for Sub-Agent:**
```
Enhance chat/app.py with:

1. Sidebar showing locked bets:

# In sidebar section, replace placeholder:
st.subheader("Your Locked Bets")
if st.session_state.user_id:
    from api.queries import get_user_bets
    bets = get_user_bets(st.session_state.user_id, pending_only=True)
    if bets:
        for bet in bets:
            st.write(f"🔒 {bet['player_name']} {bet['stat_type']} {bet['direction']} {bet['line']}")
    else:
        st.caption("No bets locked yet")
else:
    st.caption("Sign in to lock bets (Sprint 6)")

2. Sidebar showing tonight's games:

# In sidebar, replace games placeholder:
from api.queries import get_games_today
games = get_games_today()
if games:
    for g in games:
        st.write(f"🏀 {g['away_team']} @ {g['home_team']}")
        st.caption(f"{g['starts_at']}")
else:
    st.caption("No games today (or precompute hasn't run)")

3. Quick start buttons that inject prompts:

# Update the quick start buttons:
if not st.session_state.messages:
    st.write("**Quick Start:**")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("📊 Tonight's Best Props"):
            st.session_state.quick_prompt = "What are the best props to bet on tonight?"
            st.rerun()
    with col2:
        if st.button("🏀 What Games Tonight?"):
            st.session_state.quick_prompt = "What NBA games are on tonight?"
            st.rerun()
    with col3:
        if st.button("🎯 Build Me a Parlay"):
            st.session_state.quick_prompt = "Help me build a 3-leg parlay for tonight"
            st.rerun()

# Handle quick prompt (before chat input):
if "quick_prompt" in st.session_state:
    prompt = st.session_state.pop("quick_prompt")
    # ... trigger the same flow as chat_input
```

**Verification:**
```bash
# Manual test:
streamlit run chat/app.py
# Click "Tonight's Best Props" button
# Verify it triggers a chat message
# Try: "Lock LeBron over 25.5 points"
# Verify bet appears in sidebar
```

---

## Success Criteria

Sprint 3 is COMPLETE when:
- [ ] Streamlit app runs without error
- [ ] Can ask about tonight's games and get response
- [ ] Can ask about specific player props and get probability
- [ ] Can lock a bet and see it in sidebar
- [ ] Quick start buttons work
- [ ] Human has tested the full flow

---

## Key Insight

This is your MVP. After this sprint:
- Share with 5 friends
- Get feedback on the conversation flow
- Identify pain points before building more features

Everything after this sprint (Sprints 4-6) is enhancement, not core functionality.

---

## Proceed to Sprint 4?

User testing checklist:
- [ ] Tested "What games tonight?"
- [ ] Tested "Should I bet X over Y?"
- [ ] Tested "Lock it"
- [ ] Tested with a friend (not just yourself)
- [ ] Collected at least 3 pieces of feedback
