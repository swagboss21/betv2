# The Brain - Organized Testing Data

## Quick Start

```
/the-brain-organized
├── /prompts                    # System prompt versions (Git these)
│   ├── prompt-v1-collaborative.md
│   ├── prompt-v5-phased.md
│   ├── prompt-v7-minimal.md
│   └── prompt-v2-openrouter.md
│
├── /data                       # Test data (DON'T Git - too large)
│   ├── /chats
│   │   ├── day1-perplexity-main.json      # 35 chats, Nov 29
│   │   ├── day2-openrouter-test1.json     # Multi-model test 1
│   │   └── day2-openrouter-test2.json     # Multi-model test 2
│   └── /bets
│       └── (add bet tracking CSV here)
│
└── /analysis                   # Analysis files (Git these)
    ├── experiments.csv         # Master experiment tracker
    └── KEY_FINDINGS.md         # Summary of learnings
```

---

## What Goes Where

| Type | Location | Git? |
|------|----------|------|
| System prompts | `/prompts/*.md` | ✅ Yes |
| Experiment tracker | `/analysis/experiments.csv` | ✅ Yes |
| Analysis docs | `/analysis/*.md` | ✅ Yes |
| Chat JSON exports | `/data/chats/*.json` | ❌ No |
| Bet results | `/data/bets/*.csv` | ❌ No |
| HTML versions | Keep 1 current + archive old | ✅ Yes (current only) |

---

## Testing Timeline

### Day 1: Nov 29, 2024 (Perplexity Only)
- Morning: v1-v4 (Collaborative prompt)
- Evening: v5-v7 (Phased → Minimal)
- 35 total chats in `day1-perplexity-main.json`

### Day 2: Nov 30, 2024 (Multi-Model)
- OpenRouter integration
- Tested: Perplexity, GPT-4o, Grok (failed)
- Chats in `day2-openrouter-test1.json` and `test2.json`

---

## Next: Add Your Bet Results

Create `/data/bets/bet-results.csv` with columns:

```csv
date,chat_id,experiment_id,pick,line,book,result,notes
2024-11-29,1764452278714,exp-001,"Booker O 26.5 pts",26.5,PrizePicks,win,
2024-11-29,1764452278714,exp-001,"Jokic O 24.5 pts",24.5,PrizePicks,loss,Foul trouble
```

Then we can calculate:
- Win rate per experiment
- Win rate per model
- Win rate per prompt version

---

## .gitignore

```
/data/*
*.log
.DS_Store
```
