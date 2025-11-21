# Supabase Setup Guide for TELOS Observatory

## Quick Summary

This guide helps you set up Supabase to collect governance deltas (mathematical metrics) for research while keeping conversations private.

**What gets stored**: Fidelity scores, distances, intervention counts
**What NEVER gets stored**: User messages, AI responses, conversation content

---

## Step 1: Create Supabase Account (5 minutes)

1. Go to https://supabase.com
2. Click "Start your project"
3. Sign up with GitHub or email
4. Create a new project:
   - **Project name**: `telos-observatory`
   - **Database password**: Generate a strong password (save it!)
   - **Region**: Choose closest to your users
   - **Pricing**: Free tier is fine for beta testing

Wait ~2 minutes for project to provision.

---

## Step 2: Run the Schema SQL (2 minutes)

1. In Supabase dashboard, go to **SQL Editor** (left sidebar)
2. Click "New query"
3. Copy the entire contents of `SUPABASE_SCHEMA.sql`
4. Paste into the SQL editor
5. Click **RUN** button

You should see:
- ✅ 4 tables created
- ✅ Indexes created
- ✅ Triggers created
- ✅ Views created

---

## Step 3: Get API Credentials (2 minutes)

1. In Supabase dashboard, go to **Settings** → **API**
2. Copy these values:
   - **Project URL**: `https://xxxxx.supabase.co`
   - **anon public key**: `eyJhbG...` (long string)
   - **service_role key**: `eyJhbG...` (different long string)

---

## Step 4: Configure Environment Variables (2 minutes)

Create `.streamlit/secrets.toml` in your project:

```toml
# Supabase Configuration
SUPABASE_URL = "https://your-project.supabase.co"
SUPABASE_KEY = "your-service-role-key-here"

# Mistral API (you already have this)
MISTRAL_API_KEY = "your-mistral-key"
```

**IMPORTANT**: Use the `service_role` key, not the `anon` key, for server-side operations.

---

## Step 5: Install Python Client (1 minute)

Add to `requirements.txt`:

```
supabase-py>=2.0.0
```

Then install:

```bash
pip install supabase-py
```

---

## Step 6: Test Connection

Run the test script I'll create next to verify everything works.

---

## What This Architecture Does

### ✅ Privacy-Preserving
- Conversations stay in browser (never sent to Supabase)
- Only mathematical metrics transmitted
- Research team sees numbers, not content

### ✅ Research-Enabling
- Collect fidelity scores across all sessions
- Analyze intervention effectiveness
- Track governance quality over time

### ✅ Deployment-Ready
- Works on Streamlit Cloud (ephemeral containers)
- No filesystem dependency
- Scales with Supabase free tier (50,000 rows/month)

---

## Database Tables Overview

**`governance_deltas`**: Per-turn metrics (fidelity, distance, interventions)
**`session_summaries`**: Aggregated session stats (avg fidelity, intervention rate)
**`beta_consent_log`**: Immutable audit trail of consent
**`primacy_attractor_configs`**: PA settings (structure, not content)

---

## Sample Data Flow

```
User sends message
    ↓
AI generates response
    ↓
TELOS calculates fidelity = 0.87, distance = 0.23
    ↓
App sends to Supabase:
{
  "session_id": "abc-123",
  "turn_number": 5,
  "fidelity_score": 0.87,
  "distance_from_pa": 0.23,
  "intervention_triggered": false
}
    ↓
Supabase stores metrics
    ↓
Conversation content stays in browser only
```

---

## Research Queries You Can Run

Once data is flowing, you can analyze it:

```sql
-- Average fidelity by mode
SELECT mode, AVG(avg_fidelity_score)
FROM session_summaries
WHERE beta_consent_given = TRUE
GROUP BY mode;

-- Intervention effectiveness
SELECT intervention_type, AVG(fidelity_score)
FROM governance_deltas
WHERE intervention_triggered = TRUE
GROUP BY intervention_type;

-- Quality trends over time
SELECT DATE(created_at), AVG(avg_fidelity_score)
FROM session_summaries
GROUP BY DATE(created_at);
```

---

## Troubleshooting

### "Connection refused"
- Check SUPABASE_URL is correct
- Verify internet connection
- Confirm project is not paused (free tier pauses after 1 week inactivity)

### "Invalid API key"
- Use `service_role` key, not `anon` key
- Check for extra spaces in secrets.toml
- Regenerate key in Supabase dashboard if needed

### "Row level security" errors
- RLS policies are defined in schema
- Service role bypasses RLS
- Make sure you're using service_role key

---

## Next Steps

1. ✅ Complete this setup
2. Run the test script to verify connection
3. Integrate delta transmission in conversation flow
4. Deploy to Streamlit Cloud
5. Start collecting research data!

---

## Cost Estimate

**Supabase Free Tier**:
- 500MB database
- 50,000 monthly active users
- 2GB bandwidth
- 1GB file storage

**For TELOS Beta**:
- Each delta ~200 bytes
- 100 users × 50 turns each = 5,000 deltas/month
- Total: ~1MB/month
- **Cost: $0 (well within free tier)**

---

## Support

Questions? Check:
- Supabase docs: https://supabase.com/docs
- Python client: https://github.com/supabase-community/supabase-py
- TELOS issues: (your repo)
