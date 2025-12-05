# The Brain - Testing Analysis (Nov 29-30, 2024)

## Executive Summary

Two days of testing across 3 prompt strategies and 3 AI models revealed:

1. **Perplexity sonar-pro** is the most reliable for live sports data
2. **Minimal prompts** are faster but produce more errors
3. **Player availability** is the #1 failure mode across all configs

---

## Day 1 Findings (Nov 29) - Perplexity Only

### Prompt Evolution

| Version | Style | Pros | Cons |
|---------|-------|------|------|
| v1-v4 | Collaborative | Good engagement, user feels ownership | Too slow, too many questions |
| v5.x | Phase-based | Structured, catches conflicts | Still some friction |
| v6-v7 | Minimal | Fast, direct | Hallucinations, less validation |

### Key Observations

1. **"Cooper Flagg" hallucination** - AI recommended a player who doesn't exist in NBA
2. **Kevin Durant trade error** - AI thought KD was still on Suns (traded to Houston in prompt's reality)
3. **Steph Curry injury miss** - Recommended Curry when he was OUT
4. **Jayson Tatum injury miss** - Same issue

### What Worked
- Book-agnostic stat projections (e.g., "28.5 points" instead of "over 28.5")
- Single clarifying questions
- "Compare to your book's lines" ending

---

## Day 2 Findings (Nov 30) - Multi-Model via OpenRouter

### Model Comparison

| Model | Search Quality | Player Accuracy | Speed | Verdict |
|-------|---------------|-----------------|-------|---------|
| Perplexity sonar-pro | ★★★★★ | ★★★☆☆ | Fast | Best overall |
| GPT-4o:online | ★★★★☆ | ★★☆☆☆ | Medium | Good backup |
| Grok-3-fast:online | N/A | N/A | N/A | **BROKEN** |

### Critical Issues

1. **Grok model ID invalid** - `x-ai/grok-3-fast:online` threw error immediately
2. **GPT-4o availability gaps** - Recommended players not on PrizePicks
3. **Date confusion** - Search results mixing 2024 and 2025 data

---

## Failure Mode Analysis

### Most Common Failures

| Failure Type | Frequency | Root Cause | Fix Needed |
|--------------|-----------|------------|------------|
| Player not available | Very High | AI doesn't check PrizePicks inventory | Pre-fetch available players |
| Player injured/out | High | Injury data stale or not checked | Mandatory injury search |
| Wrong team | Medium | Trade data outdated | Roster verification step |
| Line doesn't exist | Medium | AI invents plausible lines | Validate against actual book |

### Example Failure Chain (Chat #27)

```
User: "5 leg parlay, you pick the game"
AI: Recommends Steph Curry (OUT - quad injury)
User: "steph is out"
AI: Pivots to Jayson Tatum (OUT - Achilles)
User: "jayson is out"
AI: Pivots to Kevin Durant on Suns (WRONG - he's on Rockets now)
User: "kevin plays for houston"
AI: Finally gets valid players
```

**4 corrections needed for 1 parlay** = terrible UX

---

## Recommendations

### Immediate Fixes

1. **Add injury verification as STEP 1** - Before any picks, search "[team] injury report today"
2. **Add roster verification** - Before using a player, confirm they're on the team
3. **Reduce hallucination** - Tell AI "Only use players you found in search results"

### Architecture Changes

1. **Pre-fetch layer** - Before AI responds, fetch:
   - Today's games
   - Injury reports for those games
   - Available props on user's book (if API exists)
   
2. **Validation layer** - After AI responds, check:
   - Are these real players?
   - Are they playing today?
   - Does the line exist?

### Prompt Recommendation

Use a **hybrid of v5 (phased) + v7 (minimal output)**:
- Keep the search-first behavior
- Keep the stat projection format
- Add explicit "verify injuries before recommending" rule
- Remove the confirmation step (adds friction without enough value)

---

## Files Reference

| File | Contents |
|------|----------|
| `/prompts/prompt-v1-collaborative.md` | Day 1 original prompt |
| `/prompts/prompt-v5-phased.md` | Day 1 phase-based prompt |
| `/prompts/prompt-v7-minimal.md` | Day 1 minimal prompt |
| `/prompts/prompt-v2-openrouter.md` | Day 2 multi-model prompt |
| `/analysis/experiments.csv` | Experiment tracking spreadsheet |
| `/data/chats/*.json` | Raw chat exports |

---

## Next Steps

1. [ ] Get bet results spreadsheet from Noah
2. [ ] Match bets to specific chats/experiments
3. [ ] Calculate win rate per prompt version
4. [ ] Calculate win rate per model
5. [ ] Identify which prompt + model combo performs best
