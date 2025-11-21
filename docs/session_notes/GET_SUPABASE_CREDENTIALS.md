# Get Your Supabase Credentials

Follow these steps to get your Supabase credentials and complete the setup.

## Step 1: Open Your Supabase Project

1. Go to https://supabase.com/dashboard
2. Click on your `telos-observatory` project (or whatever you named it)

## Step 2: Get Your API Credentials

1. In the left sidebar, click **Settings** (⚙️ icon at bottom)
2. Click **API** in the settings menu
3. You'll see two important values:

### Project URL
Look for "Project URL" - it will look like:
```
https://ukqrwjowlchhwznefboj.supabase.co
```

### Service Role Key (Secret)
Scroll down to "Project API keys"

You'll see two keys:
- **anon public** - DON'T use this one
- **service_role** - USE THIS ONE (click "Reveal" to show it)

The service_role key starts with `eyJhbG...` and is very long.

**IMPORTANT**: Use the `service_role` key, NOT the `anon` key!

## Step 3: Copy Your Credentials

Copy both values. You'll paste them in the next step.

---

## What to Do Next

Once you have both credentials, paste them here and I'll update your secrets file:

1. **SUPABASE_URL** = (paste your Project URL)
2. **SUPABASE_KEY** = (paste your service_role key)

I'll then run the test to verify everything is working!
