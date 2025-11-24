# TELOSCOPE Streamlit Cloud Deployment Guide

**Purpose**: Deploy TELOSCOPE_BETA to public Streamlit Cloud for grant reviewers

**Estimated Time**: 15-30 minutes

---

## Why This is Critical

**All grant applications require**:
- Live demo URL (reviewers can test TELOS themselves)
- Public accessibility (no auth barriers)
- Stable hosting (not localhost)

**Streamlit Cloud benefits**:
- Free tier available (Community plan)
- Auto-deploy from GitHub
- Custom subdomain (telos-observatory.streamlit.app)
- SSL certificate included
- No DevOps required

---

## Step 1: Prepare GitHub Repository

### Option A: TELOSCOPE_BETA Already in GitHub

**If TELOS repository already contains TELOSCOPE_BETA**:
- GitHub URL: https://github.com/[YourUsername]/TELOS
- Path: `/TELOSCOPE_BETA/main.py`
- ✅ Skip to Step 2

### Option B: Push TELOSCOPE_BETA to GitHub (if not there yet)

**Check if git repository exists**:
```bash
cd /Users/brunnerjf/Desktop/Privacy_PreCommit/TELOSCOPE_BETA
git status
```

**If NOT a git repository**:
```bash
# Initialize git
cd /Users/brunnerjf/Desktop/Privacy_PreCommit/TELOSCOPE_BETA
git init

# Add all files (EXCEPT secrets.toml)
git add .
git commit -m "Initial commit: TELOSCOPE_BETA v3"

# Create GitHub repository (via GitHub web interface or gh CLI)
# Then link and push:
git remote add origin https://github.com/[YourUsername]/TELOS.git
git branch -M main
git push -u origin main
```

**CRITICAL**: Ensure `.streamlit/secrets.toml` is in `.gitignore`:
```bash
# Check .gitignore
cat .gitignore | grep secrets.toml

# If not there, add it:
echo ".streamlit/secrets.toml" >> .gitignore
git add .gitignore
git commit -m "Ignore secrets.toml"
git push
```

---

## Step 2: Create Streamlit Cloud Account

1. **Go to**: https://share.streamlit.io
2. **Sign up** with GitHub account (click "Sign up")
3. **Authorize Streamlit** to access GitHub repositories
4. **Confirm email** (check inbox)

✅ Account created (free Community tier)

---

## Step 3: Deploy TELOSCOPE from GitHub

### In Streamlit Cloud Dashboard

1. **Click "New app"** (top right)

2. **Fill deployment form**:
   - **Repository**: Select your TELOS repository
   - **Branch**: `main` (or whatever branch has TELOSCOPE_BETA)
   - **Main file path**: `TELOSCOPE_BETA/main.py`
   - **App URL**: Choose subdomain (e.g., `telos-observatory.streamlit.app`)

3. **Advanced settings** → Click "Advanced settings"

4. **Python version**: Select `3.9` (or `3.10`, `3.11` - all compatible)

5. **Click "Deploy"**

⏳ Streamlit Cloud will:
- Clone your GitHub repository
- Install dependencies from `requirements.txt`
- Start the app
- Assign public URL

**Expected deployment time**: 3-5 minutes

---

## Step 4: Configure Secrets

**Problem**: App will fail on first deploy because secrets are missing (Mistral API key, Supabase credentials)

### Add Secrets in Streamlit Cloud Dashboard

1. **In Streamlit Cloud**, click your deployed app
2. **Click "Settings"** (gear icon, bottom right)
3. **Click "Secrets"** tab
4. **Paste secrets** in TOML format:

```toml
# Copy content from STREAMLIT_CLOUD_SECRETS.txt
# (but with YOUR actual API keys, not placeholders)

[default]
# Mistral API - PAID TIER
MISTRAL_API_KEY = "your-actual-mistral-key-here"

# Supabase - Production Database
SUPABASE_URL = "https://ukqrwjowlchhwznefboj.supabase.co"
SUPABASE_KEY = "your-actual-supabase-key-here"

# Optional: Anthropic Claude (if using)
ANTHROPIC_API_KEY = "your-anthropic-key-here"
```

5. **Click "Save"**

6. **App auto-restarts** with secrets loaded

⏳ Wait 30-60 seconds for restart

---

## Step 5: Verify Deployment

### Check App is Live

1. **Visit your app URL**: `https://telos-observatory.streamlit.app` (or your chosen subdomain)

2. **Should see**: TELOSCOPE Observatory interface

3. **Test Demo Mode**:
   - Click "DEMO MODE"
   - Should load 12-slide slideshow
   - Navigate through slides

4. **Test BETA Mode** (optional, requires governance setup):
   - Click "BETA MODE"
   - Try establishing PA
   - Submit a query

### Expected Behavior

✅ **Demo Mode**: Works immediately (no API calls required)
✅ **BETA Mode**: Requires Mistral API key and Supabase connection

**If errors occur**:
- Check Streamlit Cloud logs (Settings → Logs)
- Verify secrets are correct
- Check requirements.txt has all dependencies

---

## Step 6: Grant-Ready Configuration

### Make App Grant-Reviewer Friendly

**Update main.py title/description** (if needed):
```python
# At top of main.py
st.set_page_config(
    page_title="TELOS Observatory - Live Demo",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🛡️ TELOS Observatory")
st.markdown("""
**For Grant Reviewers**: This is a live demonstration of TELOS AI governance.
- **Demo Mode**: 12-slide introduction (no API keys required)
- **Beta Mode**: Live governance with dual-attractor system

**Validation Results**: 0% Attack Success Rate across 2,000 adversarial attacks
""")
```

**Add "About" section** for reviewers:
```python
with st.expander("ℹ️ About TELOS (For Grant Reviewers)"):
    st.markdown("""
    ### Technical Details
    - **Validation**: 2,000 attacks, 0% ASR, 99.9% CI [0%, 0.37%]
    - **Cryptography**: Quantum-resistant SHA3-512 + HMAC-SHA512 signatures
    - **Reproducibility**: 15-minute setup (see REPRODUCTION_GUIDE.md)

    ### Links
    - **GitHub**: https://github.com/[YourUsername]/TELOS
    - **Validation Data**: Zenodo DOI [pending]
    - **Documentation**: Full whitepapers in `/docs/`
    """)
```

---

## Step 7: Get Public URL for Grant Applications

### Your Deployment URLs

**Primary**: `https://telos-observatory.streamlit.app`
**Alternative**: `https://[custom-subdomain].streamlit.app`

### Include in All Grant Applications

**AIgrant.org**:
- Demo URL: [Streamlit Cloud URL]

**Emergent Ventures**:
- Live Demo: [Streamlit Cloud URL]

**NSF SBIR Phase I**:
- Supplementary Materials: Link to live demo

**All Others**:
- Include in "Preliminary Results" section

---

## Troubleshooting

### Issue 1: App Won't Start

**Symptom**: Red error message, app crashes on load

**Causes**:
1. Missing dependencies in `requirements.txt`
2. Python version mismatch
3. Import errors

**Fix**:
```bash
# Test locally first
cd /Users/brunnerjf/Desktop/Privacy_PreCommit/TELOSCOPE_BETA
streamlit run main.py

# If it works locally, check requirements.txt includes ALL dependencies
pip freeze > requirements-check.txt
# Compare with requirements.txt
```

### Issue 2: Secrets Not Loading

**Symptom**: "Missing API key" errors

**Causes**:
1. Secrets not added in Streamlit Cloud dashboard
2. Secrets format incorrect (must be valid TOML)
3. Secret keys don't match code

**Fix**:
1. Go to Settings → Secrets in Streamlit Cloud
2. Verify TOML syntax (no quotes around section headers)
3. Check code uses correct key names:
   ```python
   # Code should use:
   st.secrets["MISTRAL_API_KEY"]  # NOT st.secrets["mistral_api_key"]
   ```

### Issue 3: App is Slow

**Symptom**: Long load times, timeouts

**Causes**:
1. Streamlit Cloud Community tier has limited resources
2. Model downloads on first run
3. Large validation datasets

**Fix**:
1. **Upgrade to Pro tier** ($20/month) for better performance
2. **Lazy-load models**:
   ```python
   @st.cache_resource
   def load_embedding_model():
       return SentenceTransformer('all-MiniLM-L6-v2')
   ```
3. **Limit validation data** in public demo (optional)

### Issue 4: Streamlit Cloud Free Tier Limitations

**Community Tier Limits**:
- 1GB RAM
- Sleep after 7 days inactivity
- Public repos only
- No custom domain

**If hitting limits**:
- **Pro tier**: $20/month (8GB RAM, no sleep, custom domain)
- **Team tier**: $250/month (dedicated resources)

**For grant applications**: Community tier is usually sufficient. Upgrade to Pro if reviewers report slowness.

---

## Alternative: Deploy to Other Platforms

### If Streamlit Cloud Doesn't Work

**Option B: Hugging Face Spaces** (Free)
- Similar to Streamlit Cloud
- More generous free tier (2 CPU cores, 16GB RAM)
- URL: `https://huggingface.co/spaces/[username]/telos-observatory`

**Option C: AWS/GCP** (Paid)
- Full control, better performance
- Requires DevOps setup
- Cost: ~$50-100/month

**Recommendation**: Start with Streamlit Cloud Community tier. It's the fastest path to a public URL.

---

## Security Considerations

### ⚠️ NEVER Commit Secrets to GitHub

**Verify `.gitignore` includes**:
```
.streamlit/secrets.toml
.env
*.key
credentials.json
```

**Check for exposed secrets**:
```bash
# In your repo
git log --all --full-history --source -- .streamlit/secrets.toml

# If secrets were EVER committed:
# 1. Rotate ALL API keys immediately
# 2. Use git-filter-branch or BFG Repo-Cleaner to remove from history
```

### Public Demo Considerations

**Your Streamlit Cloud app is PUBLIC** - anyone can access it.

**Implications**:
- ✅ Grant reviewers can test easily
- ⚠️ API usage from your Mistral key (potential cost)
- ⚠️ Public can see demo mode (this is fine)

**To limit API costs**:
1. **Mistral API rate limits**: Set daily spending cap in Mistral dashboard
2. **Demo mode only**: Make BETA mode require password (optional)
3. **Monitor usage**: Check Mistral dashboard daily during grant review period

---

## Post-Deployment Checklist

- [ ] App loads at public URL
- [ ] Demo Mode works (12 slides)
- [ ] BETA Mode works (if testing governance)
- [ ] No secrets exposed in GitHub
- [ ] URL added to all grant applications
- [ ] Monitoring setup (check Streamlit Cloud logs daily)
- [ ] Mistral API spending cap configured
- [ ] App listed in grant materials:
  - README.md has link
  - Grant applications have link
  - Demo video mentions URL

---

## Maintenance

### Keeping App Updated

**Auto-deploy from GitHub**:
- Every push to `main` branch → Streamlit Cloud auto-redeploys
- No manual steps required

**Manual redeploy**:
- Streamlit Cloud dashboard → Click "Reboot"

### Monitoring

**Check logs**:
- Streamlit Cloud dashboard → Settings → Logs
- Shows errors, API calls, user activity

**Check usage**:
- Mistral dashboard: API calls, costs
- Supabase dashboard: Database queries
- Streamlit Cloud analytics: Visitors, pageviews

---

## Expected Costs

### Streamlit Cloud
- **Community tier**: FREE (sufficient for grant applications)
- **Pro tier**: $20/month (if performance issues)

### Mistral API
- **Your usage**: ~$5-20/month (demo + testing)
- **Grant reviewer usage**: ~$10-50/month (if 10-20 reviewers test)
- **Safety**: Set $100/month spending cap

### Total Expected Cost
- **Minimal**: $0-20/month (Community tier + API usage)
- **Recommended**: $20-40/month (Pro tier + API usage)

**For grant applications**: This is negligible ($240-480/year) compared to $275K+ grant awards.

---

## Timeline to Live Demo

**Optimistic** (everything works):
- Setup GitHub: 10 min
- Create Streamlit Cloud account: 5 min
- Deploy app: 5 min
- Configure secrets: 5 min
- Test and verify: 10 min
- **Total**: 35 minutes

**Realistic** (troubleshooting):
- Setup: 20 min
- Deploy: 10 min
- Fix errors: 30-60 min
- Test: 15 min
- **Total**: 1-2 hours

**Worst case** (major issues):
- Debugging dependency issues: 2-4 hours
- Switching to alternative platform: +2 hours
- **Total**: 4-6 hours

**Recommendation**: Budget 2 hours to be safe.

---

## Next Steps After Deployment

1. **Test public URL** thoroughly
2. **Add URL to all grant applications**
3. **Record demo video** showing live deployment
4. **Monitor usage** during grant review period
5. **Respond to reviewer feedback** (if they report issues)

---

**Document Version**: 1.0
**Last Updated**: November 24, 2025
**Estimated Completion Time**: 1-2 hours
**Cost**: $0-20/month

**End of Deployment Guide**
