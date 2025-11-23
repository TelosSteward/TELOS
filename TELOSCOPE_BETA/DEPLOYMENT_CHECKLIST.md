# TELOS Observatory - Streamlit Cloud Deployment Checklist

**Status**: ✅ Ready to Deploy with Mistral Large
**Date**: November 21, 2025
**API Key**: Configured and ready

---

## Pre-Deployment Checklist

### ✅ Code Updates Complete

- [x] Mistral API client updated to use `mistral-large-latest`
- [x] Requirements.txt includes all dependencies
- [x] Streamlit config.toml has correct colors (#F4D03F)
- [x] BETA/DEMO mode configured
- [x] API key ready: `iYsJab8PibuqxWgOFFLQ3WcMrTguE3X8`

### ⬜ Repository Setup

- [ ] Push all changes to GitHub
- [ ] Verify `.streamlit/secrets.toml` is in `.gitignore` (**IMPORTANT**)
- [ ] Confirm repository is accessible (public or private with Streamlit access)

### ⬜ Supabase Setup

- [ ] Get Supabase anon key from: https://supabase.com/dashboard/project/ukqrwjowlchhwznefboj/settings/api
- [ ] Test Supabase connection locally (optional)

---

## Deployment Steps

### Step 1: Push to GitHub

```bash
cd /Users/brunnerjf/Desktop/Privacy_PreCommit/TELOSCOPE_BETA

# Check status
git status

# Add all files (EXCEPT .streamlit/secrets.toml if it exists)
git add .

# Commit
git commit -m "Deploy TELOS Observatory BETA - Mistral Large integration"

# Push
git push origin main
```

### Step 2: Deploy to Streamlit Cloud

1. **Go to Streamlit Cloud**:
   - Visit: https://share.streamlit.io/
   - Sign in with GitHub

2. **Create New App**:
   - Click "New app"
   - Repository: Select your TELOS repository
   - Branch: `main`
   - Main file path: `TELOSCOPE_BETA/main.py`

3. **Configure Secrets**:
   - Click "Advanced settings"
   - Click "Secrets"
   - Copy and paste from `STREAMLIT_CLOUD_SECRETS.txt`:

```toml
SUPABASE_URL = "https://ukqrwjowlchhwznefboj.supabase.co"
SUPABASE_KEY = "YOUR_SUPABASE_ANON_KEY_HERE"
MISTRAL_API_KEY = "iYsJab8PibuqxWgOFFLQ3WcMrTguE3X8"
```

   - **IMPORTANT**: Replace `YOUR_SUPABASE_ANON_KEY_HERE` with actual key

4. **Deploy**:
   - Click "Deploy!"
   - Wait 2-5 minutes for installation
   - App will be live at: `https://[your-app-name].streamlit.app`

---

## Post-Deployment Testing

### Test 1: App Loads

- [ ] App loads without errors
- [ ] DEMO tab is default and shows slideshow
- [ ] Can navigate through all 12 slides
- [ ] Colors match TELOS brand (muted gold)

### Test 2: DEMO Mode

- [ ] Progressive slideshow works
- [ ] Can advance through slides
- [ ] Slide 12 (completion) shows
- [ ] BETA tab unlocks after completion

### Test 3: BETA Mode

- [ ] BETA tab is accessible
- [ ] PA onboarding loads
- [ ] Can create a Primacy Attractor
- [ ] Consent forms display
- [ ] Data saves to Supabase

### Test 4: Mistral Integration

- [ ] Mistral API connects
- [ ] Using Mistral Large model
- [ ] Responses are high quality
- [ ] No rate limit errors
- [ ] Monitor usage at: https://console.mistral.ai/usage

---

## Monitoring

### Track Your $125 Credits

**Mistral Console**: https://console.mistral.ai/

Watch for:
- API calls count
- Tokens used
- Cost per request
- Remaining balance

**Expected Usage** (Mistral Large):
- Cost: ~$8 per 1M tokens
- Your $125 = ~15.6M tokens
- Average conversation: ~1,000-5,000 tokens
- Estimate: **3,000-15,000 conversations** before credits run out

### Streamlit Analytics

**Dashboard**: https://share.streamlit.io/

Monitor:
- Active users
- Error logs
- Resource usage
- Uptime

---

## Troubleshooting

### Error: "ModuleNotFoundError"

**Cause**: Missing dependency
**Fix**: Add to `requirements.txt` and redeploy

### Error: "MissingAPIKeyError: MISTRAL"

**Cause**: API key not in secrets
**Fix**:
1. Go to Streamlit app settings
2. Add/update MISTRAL_API_KEY in secrets
3. Restart app

### Error: "Supabase connection failed"

**Cause**: Wrong Supabase credentials
**Fix**:
1. Get correct anon key from Supabase dashboard
2. Update in Streamlit secrets
3. Restart app

### Error: "Rate limit exceeded"

**Cause**: Credits not linked to API key
**Fix**:
1. Check Mistral Console: https://console.mistral.ai/
2. Verify paid tier is active
3. Check API key is correct
4. Contact Mistral support if needed

### Slow Performance

**Cause**: Free tier limitations
**Fix**: Upgrade to Streamlit Cloud Pro ($20/month) for:
- 4GB RAM (vs 1GB)
- Faster response times
- Priority support

---

## Security Checklist

### Before Going Public

- [ ] `.streamlit/secrets.toml` is in `.gitignore`
- [ ] No API keys in code (only in secrets/environment)
- [ ] Supabase RLS (Row Level Security) is enabled
- [ ] Test with incognito browser (fresh session)
- [ ] No sensitive data in logs

### API Key Safety

✅ **Your Mistral key is**:
- In Streamlit Cloud secrets (encrypted)
- Not in git repository
- Not visible to users
- Only accessible server-side

---

## Next Steps After Deployment

### Immediate (Today)

1. Test all features thoroughly
2. Fix any deployment errors
3. Verify Mistral Large is being used
4. Check one conversation in Mistral Console

### Short-term (This Week)

1. Share with 5-10 beta testers
2. Gather feedback via BETA forms
3. Monitor credit usage
4. Fix bugs as they arise

### Medium-term (Next 2 Weeks)

1. Deploy to Hetzner for production
2. Keep Streamlit Cloud for demos
3. Optimize based on feedback
4. Add requested features

---

## Cost Tracking

### Current Setup

- **Streamlit Cloud**: Free tier (1 app, 1GB RAM)
- **Mistral API**: $125 in credits (paid tier)
- **Supabase**: Free tier (25k rows)
- **Total**: $0/month (using your credits)

### When Credits Run Out

**Option A**: Buy more Mistral credits
- $125 = 15.6M tokens (Mistral Large)
- Good for heavy usage

**Option B**: Deploy to Hetzner with Ollama
- €10/month for server
- Unlimited local Ollama usage
- Zero API costs
- Better long-term solution

---

## Support Contacts

### Streamlit
- Docs: https://docs.streamlit.io/
- Forum: https://discuss.streamlit.io/
- Status: https://status.streamlit.io/

### Mistral
- Console: https://console.mistral.ai/
- Usage: https://console.mistral.ai/usage
- Support: support@mistral.ai

### Supabase
- Dashboard: https://supabase.com/dashboard
- Project: ukqrwjowlchhwznefboj
- Docs: https://supabase.com/docs

---

## Files to Reference

- `SESSION_HANDOFF_NOV21_2025.md` - Complete session summary
- `STREAMLIT_DEPLOYMENT_GUIDE.md` - Full deployment guide
- `STREAMLIT_CLOUD_SECRETS.txt` - Secrets template
- `MISTRAL_API_SETUP.md` - Mistral configuration details

---

**Status**: 🟢 Ready to deploy!

**Your API Key**: `iYsJab8PibuqxWgOFFLQ3WcMrTguE3X8`

**Model**: `mistral-large-latest` (automatically configured)

**Next Action**: Push to GitHub → Deploy to Streamlit Cloud → Test!
