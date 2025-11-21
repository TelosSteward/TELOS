#!/usr/bin/env python3
"""
Apply Supabase Schema Extension for Validation Telemetric Sessions.

This script applies the schema changes needed for:
1. Validation telemetric sessions tables
2. Telemetric signature columns in existing tables
3. Views and functions for analysis

Usage:
    python3 apply_supabase_schema.py
"""

import os
from pathlib import Path
from supabase import create_client

# Load credentials from environment or .streamlit/secrets.toml
def load_supabase_credentials():
    """Load Supabase credentials."""
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")

    if not url or not key:
        # Try loading from secrets.toml
        secrets_path = Path(".streamlit/secrets.toml")
        if secrets_path.exists():
            print("Loading credentials from .streamlit/secrets.toml...")
            import tomli
            with open(secrets_path, "rb") as f:
                secrets = tomli.load(f)
                url = secrets.get("SUPABASE_URL")
                key = secrets.get("SUPABASE_KEY")

    if not url or not key:
        raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set")

    return url, key


def read_sql_file(filepath):
    """Read SQL file and return as string."""
    with open(filepath, 'r') as f:
        return f.read()


def main():
    print("=" * 80)
    print("SUPABASE SCHEMA EXTENSION - VALIDATION TELEMETRIC SESSIONS")
    print("=" * 80)
    print()

    # Load credentials
    print("1. Loading Supabase credentials...")
    try:
        url, key = load_supabase_credentials()
        print(f"   ✓ URL: {url}")
        print(f"   ✓ Key: {key[:20]}...")
    except Exception as e:
        print(f"   ✗ Failed to load credentials: {e}")
        return

    # Connect to Supabase
    print()
    print("2. Connecting to Supabase...")
    try:
        client = create_client(url, key)
        # Test connection
        result = client.table("governance_deltas").select("id").limit(1).execute()
        print(f"   ✓ Connected successfully")
    except Exception as e:
        print(f"   ✗ Connection failed: {e}")
        return

    # Read SQL file
    print()
    print("3. Reading schema SQL file...")
    sql_file = Path("supabase_validation_telemetric_extension.sql")
    if not sql_file.exists():
        print(f"   ✗ SQL file not found: {sql_file}")
        return

    sql_content = read_sql_file(sql_file)
    print(f"   ✓ Loaded {len(sql_content)} characters of SQL")

    # Note about manual application
    print()
    print("4. Schema Application:")
    print("   ⚠️  IMPORTANT: SQL must be applied via Supabase Dashboard SQL Editor")
    print()
    print("   The schema includes:")
    print("   - CREATE TABLE statements")
    print("   - CREATE VIEW statements")
    print("   - CREATE FUNCTION statements")
    print("   - CREATE TRIGGER statements")
    print("   - Row Level Security policies")
    print()
    print("   These require elevated privileges that the service role key doesn't have.")
    print()
    print("   TO APPLY:")
    print("   1. Go to: https://supabase.com/dashboard")
    print("   2. Select your project")
    print("   3. Go to: SQL Editor (left sidebar)")
    print("   4. Create new query")
    print(f"   5. Copy contents of: {sql_file.absolute()}")
    print("   6. Paste into SQL Editor")
    print("   7. Click 'Run' button")
    print()

    # Check if tables already exist
    print("5. Checking current schema state...")
    tables_to_check = [
        "validation_telemetric_sessions",
        "validation_sessions",
        "validation_counterfactual_comparisons"
    ]

    existing_tables = []
    missing_tables = []

    for table in tables_to_check:
        try:
            result = client.table(table).select("*").limit(1).execute()
            existing_tables.append(table)
            print(f"   ✓ {table} - EXISTS")
        except Exception as e:
            if "not found" in str(e).lower() or "relation" in str(e).lower():
                missing_tables.append(table)
                print(f"   ✗ {table} - NOT FOUND")
            else:
                print(f"   ? {table} - ERROR: {str(e)[:50]}")

    print()
    if missing_tables:
        print(f"   Schema needs to be applied: {len(missing_tables)} tables missing")
        print()
        print("   Next steps:")
        print(f"   1. Apply the schema via Supabase Dashboard (see instructions above)")
        print(f"   2. Re-run this script to verify")
    else:
        print(f"   ✓ All validation tables exist! Schema already applied.")
        print()
        print("   You can now:")
        print("   1. Run validation studies with: python3 run_ollama_validation_suite.py")
        print("   2. Generate signed validation data")
        print("   3. Query validation results")

    print()
    print("=" * 80)
    print("SCHEMA CHECK COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
