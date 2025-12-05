# Phase 4 Planning: LLM Integration

## The Golden Rule (from your Claude.md)

```
MODEL does the MATH. LLM does the TALKING.
```

This is your North Star. Every decision should be evaluated against this.

---

## ❌ Common Mistakes (What You'll Be Tempted To Do)

### Mistake #1: Making the LLM Do Math

**Wrong approach:**
```
User: "Should I bet LeBron over 25.5 points?"
LLM: "Based on his recent averages of 26.2 PPG and the Celtics 
      allowing 114.5 PPG, I calculate a 54% chance..."
```

**Why it's wrong:**
- LLMs hallucinate numbers
- No reproducibility
- Can't validate the math
- You already HAVE a Monte Carlo engine that does this correctly

**Right approach:**
```
User: "Should I bet LeBron over 25.5 points?"
LLM: [calls engine.simulate_player_prop("LeBron James", "pts", 25.5, "BOS")]
     → Receives: {p_over: 0.431, recommendation: "UNDER", edge: -0.069}
LLM: "I'd lean under on this one. My model has him around 24 points 
      tonight, and the line is set a bit high at 25.5."
```

---

### Mistake #2: Building a Multi-Agent System

**Wrong approach:**
```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ Parser Agent│ →  │Research Agent│ → │ Advisor Agent│
└─────────────┘    └─────────────┘    └─────────────┘
       ↓                  ↓                  ↓
   Intent DB         Web Search         Response Gen
```

**Why it's wrong:**
- Complexity explosion
- Hard to debug
- Latency stacks up
- You don't need it

**Right approach:**
```
┌─────────────────────────────────────────────────────┐
│                   ONE LLM CALL                       │
│                                                      │
│  System prompt: "You are a betting co-pilot.        │
│                  Use these tools: [engine API]       │
│                  Always verify injuries first."      │
│                                                      │
│  Tools available:                                    │
│    - simulate_prop(player, stat, line)              │
│    - check_injuries(team)                           │
│    - get_player_projection(player)                   │
└─────────────────────────────────────────────────────┘
```

---

### Mistake #3: Trying to "Train" or "Fine-Tune" for This

**Wrong approach:**
"I should fine-tune an LLM on betting conversations..."

**Why it's wrong:**
- Expensive
- Unnecessary
- Your edge is in the MODEL, not the LLM
- Prompt engineering is sufficient for this use case

**Right approach:**
A well-crafted system prompt + function calling = done.

---

### Mistake #4: Over-Engineering the Conversation Flow

**Wrong approach:**
```python
class ConversationState:
    def __init__(self):
        self.stage = "greeting"
        self.collected_info = {}
        self.next_required_field = None
        
    def transition(self, user_input):
        if self.stage == "greeting":
            self.stage = "collect_bankroll"
        elif self.stage == "collect_bankroll":
            # ... 50 more states
```

**Why it's wrong:**
- Rigid conversation flow
- Users don't follow scripts
- Maintenance nightmare

**Right approach:**
Let the LLM handle conversation naturally. Give it tools and constraints.

---

### Mistake #5: Forgetting the Failure Modes You Already Discovered

From your early testing (Phase 1), you learned:
- **Injury hallucinations** = Primary failure mode
- **Stale roster data** = Players traded, AI doesn't know
- **Assist props never hit** = 0% in your sample

These lessons should be BAKED INTO the system prompt, not rediscovered.

---

## ✅ The Right Architecture

### Layer 1: System Prompt (The "Personality")

```markdown
You are "The Brain" - an AI sports betting co-pilot.

YOUR ROLE:
- Help users find value bets using data, not hunches
- Explain recommendations conversationally
- NEVER make the final decision for them

YOUR TOOLS:
- simulate_prop(player, stat, line, opponent) → probability + edge
- get_game_projections(home_team, away_team) → all player projections
- check_injuries(team) → current injury report

CRITICAL RULES:
1. ALWAYS check injuries before recommending a player
2. NEVER discuss specific dollar amounts (legal requirement)
3. NEVER recommend assist props (historically 0% hit rate)
4. If asked about a player you can't find, SAY SO - don't guess

COMMUNICATION STYLE:
- Casual but confident
- Explain the "why" behind recommendations
- Use phrases like "my model shows" not "I think"
- Acknowledge uncertainty when edge is small (<5%)
```

### Layer 2: Function Definitions (The "Tools")

```python
TOOLS = [
    {
        "name": "simulate_prop",
        "description": "Get probability and edge for a player prop bet",
        "parameters": {
            "player_name": "Full player name (e.g., 'LeBron James')",
            "stat": "One of: pts, reb, ast, stl, blk, tov, fg3m",
            "line": "The betting line (e.g., 25.5)",
            "opponent": "Opponent team abbreviation (e.g., 'BOS')",
            "is_home": "Boolean, is player's team at home",
            "over_odds": "American odds for over (e.g., '-110')",
            "under_odds": "American odds for under (e.g., '-110')"
        }
    },
    {
        "name": "get_tonight_games",
        "description": "Get list of NBA games happening tonight",
        "parameters": {}
    },
    {
        "name": "check_injuries",
        "description": "Get injury report for a team. ALWAYS CALL THIS before recommending players.",
        "parameters": {
            "team": "Team abbreviation (e.g., 'LAL')"
        }
    },
    {
        "name": "build_parlay",
        "description": "Analyze a multi-leg parlay for correlation and combined probability",
        "parameters": {
            "legs": "List of prop bets to combine"
        }
    }
]
```

### Layer 3: The Orchestration (Simple!)

```python
def chat(user_message: str, conversation_history: list) -> str:
    """
    Single function that handles everything.
    No state machines. No multi-agent nonsense.
    """
    response = llm.chat(
        model="claude-sonnet-4-20250514",  # or gpt-4o
        system=SYSTEM_PROMPT,
        messages=conversation_history + [{"role": "user", "content": user_message}],
        tools=TOOLS
    )
    
    # If LLM wants to use a tool, execute it
    while response.tool_calls:
        tool_results = execute_tools(response.tool_calls)
        response = llm.chat(
            messages=conversation_history + [response, tool_results],
            tools=TOOLS
        )
    
    return response.content
```

That's it. That's the whole architecture.

---

## The Entanglement Concept (Your Secret Sauce)

From your early notes, you identified that good parlays aren't random combinations - they're **correlated picks that rise and fall together**.

Example "Shootout Thesis":
- If you think LAL vs BOS will be high-scoring...
- LeBron OVER points makes sense
- Tatum OVER points makes sense
- These picks are CORRELATED (both benefit from same game environment)

This is where the LLM adds real value - it can EXPLAIN the thesis:

```
User: "Give me a 3-leg parlay for tonight"

LLM: [calls get_tonight_games()]
     [calls simulate_prop() for several players]
     [identifies correlated opportunities]

LLM: "Here's a shootout parlay for LAL-BOS:

      1. LeBron OVER 24.5 pts (my model: 58% vs book's 52%)
      2. Tatum OVER 26.5 pts (my model: 55% vs book's 52%)  
      3. Game OVER 225.5 total (my model: 54% vs book's 51%)

      These hang together - if the game is fast-paced and high-scoring,
      all three legs benefit. If it's a grind-it-out defensive battle,
      they all struggle together.
      
      That's the thesis. You in?"
```

---

## Implementation Phases

### Phase 4.1: Proof of Concept (1-2 days)
- Single Python script
- Hardcoded system prompt
- Call your existing MonteCarloEngine
- Test with 5-10 real conversations
- **Goal:** Validate the approach works

### Phase 4.2: Injury Integration (1 day)  
- Add injury checking tool
- Scrape ESPN/Rotowire or use API
- Make injury check MANDATORY before recommendations
- **Goal:** Eliminate the #1 failure mode

### Phase 4.3: Conversation Polish (2-3 days)
- Refine system prompt based on testing
- Add edge cases handling
- Improve response formatting
- **Goal:** Make it feel natural

### Phase 4.4: Parlay Logic (2-3 days)
- Implement correlation detection
- Build "thesis" generator
- Test multi-leg recommendations
- **Goal:** Enable the "entanglement" feature

### Phase 4.5: UI (Optional, Later)
- Don't build this until 4.1-4.4 are solid
- Could be CLI, Streamlit, or web app
- **Goal:** Make it accessible

---

## What NOT to Build Yet

- User accounts / authentication
- Bankroll tracking
- Bet history database
- Mobile app
- Real money integration
- Multi-sport support

These are all **Phase 5+** concerns. Focus on making the core loop work.

---

## Decision Framework

When you're unsure what to build next, ask:

1. **Does this help the MODEL do math better?**
   → If yes, maybe do it
   → If no, probably skip it

2. **Does this help the LLM communicate better?**
   → If yes, it's probably a system prompt tweak
   → If no, probably skip it

3. **Am I building infrastructure or product?**
   → Infrastructure without users = waste
   → Get the core loop working first

4. **Would a user notice this?**
   → If no, skip it
   → Ship something they can feel

---

## Your Specific Risks (Based on Your History)

From what I've seen in this project:

1. **You tend to go deep on data/ML before validating UX**
   - Counter: Build the simplest possible chat interface FIRST
   - Test with real humans before optimizing models

2. **You research thoroughly (good!) but sometimes too long**
   - Counter: Set a 2-hour timebox for any research spike
   - Then build something, even if imperfect

3. **You've already solved problems you might re-solve**
   - Counter: Reference your Claude.md and past docs
   - Your injury failure mode insight is GOLD - use it

---

## Concrete Next Step

Don't build anything complex. Build this:

```python
# phase4_poc.py - The whole thing in one file

import anthropic
from simulation import MonteCarloEngine

client = anthropic.Anthropic()
engine = MonteCarloEngine("models/")

SYSTEM = """You are The Brain, a sports betting co-pilot.
Use simulate_prop to get probabilities. Be conversational.
ALWAYS verify player is not injured before recommending."""

def simulate_prop(player: str, stat: str, line: float, opponent: str):
    """Tool the LLM can call."""
    result = engine.simulate_player_prop(
        player_name=player,
        stat=stat,
        line=line,
        opponent=opponent
    )
    return result.to_dict()

# ... add tool handling ...

if __name__ == "__main__":
    print("The Brain v0.1")
    while True:
        user_input = input("\nYou: ")
        response = chat(user_input)
        print(f"\nBrain: {response}")
```

That's your Phase 4.1. One file. One afternoon. Ship it.