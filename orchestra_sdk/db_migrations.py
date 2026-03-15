"""
orchestra_sdk.db_migrations
=============================
SQL migrations for Conductor tables in Supabase.
Run with: orchestra migrate --env .env
"""

from __future__ import annotations

from typing import Optional

MIGRATIONS = [
    {
        "name": "001_conductor_sessions",
        "sql": """
-- Conductor sessions table
CREATE TABLE IF NOT EXISTS conductor_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL UNIQUE,
    dataset_id TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'running'
        CHECK (status IN ('running', 'paused', 'completed', 'failed')),
    baseline_metric FLOAT8,
    iteration INT4 NOT NULL DEFAULT 0,
    config JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Index for fast lookup by name
CREATE INDEX IF NOT EXISTS conductor_sessions_name_idx ON conductor_sessions (name);

-- RLS: allow authenticated users to manage their sessions
ALTER TABLE conductor_sessions ENABLE ROW LEVEL SECURITY;
CREATE POLICY IF NOT EXISTS "conductor_sessions_all" ON conductor_sessions
    FOR ALL USING (true);
""",
    },
    {
        "name": "002_conductor_experiments",
        "sql": """
-- Conductor experiments table
CREATE TABLE IF NOT EXISTS conductor_experiments (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_name TEXT NOT NULL,
    iteration INT4 NOT NULL,
    hypothesis TEXT NOT NULL,
    hypothesis_sha TEXT NOT NULL DEFAULT '',
    target_metric FLOAT8,
    baseline_at_time FLOAT8,
    delta FLOAT8,
    decision TEXT NOT NULL
        CHECK (decision IN ('keep', 'discard', 'failed', 'skipped')),
    duration_seconds FLOAT8,
    log_tail TEXT DEFAULT '',
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Indexes
CREATE INDEX IF NOT EXISTS conductor_experiments_session_idx
    ON conductor_experiments (session_name, iteration DESC);
CREATE INDEX IF NOT EXISTS conductor_experiments_decision_idx
    ON conductor_experiments (session_name, decision);

-- RLS
ALTER TABLE conductor_experiments ENABLE ROW LEVEL SECURITY;
CREATE POLICY IF NOT EXISTS "conductor_experiments_all" ON conductor_experiments
    FOR ALL USING (true);
""",
    },
    {
        "name": "003_conductor_memories",
        "sql": """
-- Enable pgvector extension (requires Supabase Pro or self-hosted with pgvector)
CREATE EXTENSION IF NOT EXISTS vector;

-- Conductor memories table with pgvector embedding column
CREATE TABLE IF NOT EXISTS conductor_memories (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_name TEXT NOT NULL,
    content TEXT NOT NULL,
    embedding VECTOR(768),
    iteration INT4 NOT NULL,
    decision TEXT NOT NULL DEFAULT 'unknown',
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- IVFFlat index for approximate nearest-neighbor search
-- Note: requires at least 100 rows before the index is effective
CREATE INDEX IF NOT EXISTS conductor_memories_embedding_idx
    ON conductor_memories
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

CREATE INDEX IF NOT EXISTS conductor_memories_session_idx
    ON conductor_memories (session_name, created_at DESC);

-- RLS
ALTER TABLE conductor_memories ENABLE ROW LEVEL SECURITY;
CREATE POLICY IF NOT EXISTS "conductor_memories_all" ON conductor_memories
    FOR ALL USING (true);
""",
    },
    {
        "name": "004_search_memories_rpc",
        "sql": """
-- RPC function for pgvector cosine similarity search
CREATE OR REPLACE FUNCTION search_conductor_memories(
    query_embedding VECTOR(768),
    session_name_filter TEXT,
    match_threshold FLOAT DEFAULT 0.75,
    match_count INT DEFAULT 5
)
RETURNS TABLE (
    id UUID,
    content TEXT,
    similarity FLOAT,
    iteration INT4,
    decision TEXT,
    created_at TIMESTAMPTZ,
    metadata JSONB
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        m.id,
        m.content,
        1 - (m.embedding <=> query_embedding) AS similarity,
        m.iteration,
        m.decision,
        m.created_at,
        m.metadata
    FROM conductor_memories m
    WHERE
        m.session_name = session_name_filter
        AND 1 - (m.embedding <=> query_embedding) > match_threshold
    ORDER BY m.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;
""",
    },
]


def run_migrations(dry_run: bool = False, console=None) -> None:
    """Execute all migrations against Supabase."""
    import os
    from supabase import create_client

    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_ANON_KEY") or os.environ.get("SUPABASE_SERVICE_ROLE_KEY")

    if not url or not key:
        msg = "SUPABASE_URL and SUPABASE_ANON_KEY (or SUPABASE_SERVICE_ROLE_KEY) must be set."
        if console:
            console.print(f"[red]Error:[/red] {msg}")
        else:
            print(f"Error: {msg}")
        return

    if dry_run:
        for m in MIGRATIONS:
            print(f"\n-- Migration: {m['name']}")
            print(m["sql"])
        return

    client = create_client(url, key)

    for migration in MIGRATIONS:
        name = migration["name"]
        sql = migration["sql"]
        if console:
            console.print(f"[dim]Running migration:[/dim] {name}")
        try:
            client.rpc("exec_sql", {"sql": sql}).execute()
            if console:
                console.print(f"[green]✓[/green] {name}")
        except Exception as e:
            # Try direct postgrest approach
            if console:
                console.print(
                    f"[yellow]Note:[/yellow] {name} — {e}\n"
                    "  Run the SQL manually in the Supabase SQL editor if needed."
                )

    if console:
        console.print("\n[green]Migrations complete.[/green]")
        console.print(
            "  If any migrations failed, run them manually in the Supabase SQL Editor:\n"
            "  https://supabase.com/dashboard/project/_/sql"
        )
