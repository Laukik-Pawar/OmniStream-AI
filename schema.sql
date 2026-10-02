-- =============================================
-- OmniStream-AI Schema (matches existing code)
-- =============================================

CREATE TABLE IF NOT EXISTS users (
    id              SERIAL PRIMARY KEY,
    reddit_username VARCHAR(150) UNIQUE,
    reddit_refresh_token TEXT,
    created_at      TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS user_oauth_tokens (
    user_identifier VARCHAR(150) PRIMARY KEY,
    youtube_token   JSONB,
    reddit_token    JSONB,
    last_updated    TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS reddit_interactions (
    id                    SERIAL PRIMARY KEY,
    user_id               INTEGER REFERENCES users(id) ON DELETE CASCADE,
    reddit_post_id        VARCHAR(100),
    title                 TEXT,
    content               TEXT,
    url                   TEXT,
    subreddit             VARCHAR(100),
    interaction_timestamp TIMESTAMP WITH TIME ZONE,
    UNIQUE (user_id, reddit_post_id)
);

CREATE TABLE IF NOT EXISTS post_genres (
    post_url  TEXT PRIMARY KEY,
    genre     VARCHAR(100) NOT NULL
);

CREATE TABLE IF NOT EXISTS item_mappings (
    url         TEXT PRIMARY KEY,
    source      VARCHAR(50),
    title       TEXT,
    content     TEXT,
    timestamp   TIMESTAMP WITH TIME ZONE,
    time_block  VARCHAR(20),
    ml_genre    VARCHAR(100)
);

CREATE TABLE IF NOT EXISTS temporal_clusters (
    id           SERIAL PRIMARY KEY,
    time_block   VARCHAR(20),
    genre        VARCHAR(100),
    search_query TEXT,
    last_updated TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS recommendations (
    id             SERIAL PRIMARY KEY,
    title          TEXT,
    url            TEXT UNIQUE,
    snippet        TEXT,
    source_domain  VARCHAR(150),
    genre          VARCHAR(100),
    matched_query  TEXT,
    image_url      TEXT,
    time_block     VARCHAR(20),
    created_at     TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_reddit_interactions_user ON reddit_interactions(user_id);
CREATE INDEX IF NOT EXISTS idx_item_mappings_time_block ON item_mappings(time_block);
CREATE INDEX IF NOT EXISTS idx_recommendations_genre ON recommendations(genre);
CREATE INDEX IF NOT EXISTS idx_recommendations_time_block ON recommendations(time_block);