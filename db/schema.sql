-- The Brain - PostgreSQL Schema
-- Sprint 0: Foundation
-- 6 tables for NBA betting co-pilot SaaS

-- 1. games - Tonight's games
CREATE TABLE IF NOT EXISTS games (
    id TEXT PRIMARY KEY,  -- SGO eventID format
    home_team VARCHAR(3) NOT NULL,
    away_team VARCHAR(3) NOT NULL,
    starts_at TIMESTAMPTZ,
    home_score INTEGER,
    away_score INTEGER,
    status VARCHAR(20) DEFAULT 'scheduled',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_games_starts_at ON games(starts_at);
CREATE INDEX IF NOT EXISTS idx_games_status ON games(status);

-- 2. projections - Monte Carlo simulation results
CREATE TABLE IF NOT EXISTS projections (
    id SERIAL PRIMARY KEY,
    game_id TEXT REFERENCES games(id) ON DELETE CASCADE,
    player_name TEXT NOT NULL,
    stat_type VARCHAR(10) NOT NULL,  -- pts, reb, ast, stl, blk, tov, fg3m
    mean REAL NOT NULL,
    std REAL NOT NULL,
    p10 REAL,
    p25 REAL,
    p50 REAL,
    p75 REAL,
    p90 REAL,
    computed_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(game_id, player_name, stat_type)
);

CREATE INDEX IF NOT EXISTS idx_projections_player ON projections(player_name);
CREATE INDEX IF NOT EXISTS idx_projections_game ON projections(game_id);

-- 3. injuries - Current injury status
CREATE TABLE IF NOT EXISTS injuries (
    id SERIAL PRIMARY KEY,
    player_name TEXT NOT NULL,
    team VARCHAR(3) NOT NULL,
    status VARCHAR(20) NOT NULL,  -- OUT, DOUBTFUL, QUESTIONABLE, PROBABLE
    injury TEXT,
    source VARCHAR(10) DEFAULT 'ESPN',  -- ESPN, USER
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(player_name, team)
);

CREATE INDEX IF NOT EXISTS idx_injuries_team ON injuries(team);

-- 4. users
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY,
    email TEXT UNIQUE NOT NULL,
    auth_provider VARCHAR(20) DEFAULT 'google',
    is_paid BOOLEAN DEFAULT FALSE,
    message_count INTEGER DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    last_active_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);

-- 5. bets - User locked bets
CREATE TABLE IF NOT EXISTS bets (
    id SERIAL PRIMARY KEY,
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    game_id TEXT REFERENCES games(id) ON DELETE CASCADE,
    player_name TEXT NOT NULL,
    stat_type VARCHAR(10) NOT NULL,
    line REAL NOT NULL,
    direction VARCHAR(5) NOT NULL,  -- OVER, UNDER
    odds TEXT DEFAULT '-110',
    edge_pct REAL,
    locked_at TIMESTAMPTZ DEFAULT NOW(),
    result VARCHAR(10),  -- WIN, LOSS, PUSH (null until graded)
    actual_value REAL,
    graded_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_bets_user ON bets(user_id);
CREATE INDEX IF NOT EXISTS idx_bets_game ON bets(game_id);
CREATE INDEX IF NOT EXISTS idx_bets_result ON bets(result);

-- 6. house_bots - Automated bettors for track record
CREATE TABLE IF NOT EXISTS house_bots (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    strategy VARCHAR(20) NOT NULL,  -- aggressive, conservative, balanced
    bankroll REAL DEFAULT 1000.0,
    starting_bankroll REAL DEFAULT 1000.0,
    is_alive BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Seed initial house bots
INSERT INTO house_bots (id, name, strategy) VALUES
    ('bot-aggressive', 'The Degen', 'aggressive'),
    ('bot-conservative', 'The Grinder', 'conservative'),
    ('bot-balanced', 'The Brain Bot', 'balanced')
ON CONFLICT (id) DO NOTHING;
