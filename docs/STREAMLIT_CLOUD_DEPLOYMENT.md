# Streamlit Cloud Deployment Guide

## Quick Setup for Streamlit Cloud

###  Step 1: Push to GitHub

Your code is already on GitHub. Make sure the latest changes are pushed:

```bash
git push origin experimental/dual-attractor
```

### Step 2: Go to Streamlit Cloud

1. Go to **https://share.streamlit.io/**
2. Sign in with your GitHub account
3. Click "New app"

### Step 3: Configure Your App

**Repository**: Select your TELOS repository
**Branch**: `experimental/dual-attractor`
**Main file path**: `telos_observatory_v3/main.py`

### Step 4: Add Secrets

In the Streamlit Cloud dashboard, go to **App settings** > **Secrets** and add:

```toml
[default]
MISTRAL_API_KEY = "your-actual-mistral-api-key"
```

### Step 5: Deploy

Click "Deploy!" and wait for the app to build.

---

## Files Already Set Up

✅ `requirements.txt` - Python dependencies
✅ `.streamlit/config.toml` - Streamlit configuration
✅ `.streamlit/secrets.toml.example` - Example secrets file

---

## Accessing Secrets in Your Code

Streamlit Cloud automatically makes secrets available via `st.secrets`:

```python
import streamlit as st

# Access secrets
api_key = st.secrets["default"]["MISTRAL_API_KEY"]
```

---

## Current App Location

**Local**: http://localhost:8502
**Cloud** (after deployment): https://your-app-name.streamlit.app

---

## Troubleshooting

### Build Fails

- Check `requirements.txt` has all dependencies
- Check that `telos_observatory_v3/main.py` exists and is valid
- Check logs in Streamlit Cloud dashboard

### Secrets Not Working

- Make sure secrets are added in Streamlit Cloud dashboard
- Format must be TOML (like example)
- No quotes around keys, quotes around values

### App Crashes

- Check Python version compatibility (Streamlit Cloud uses Python 3.9-3.11)
- Check that all imports work
- Review logs in dashboard

---

## What Gets Deployed

When you deploy to Streamlit Cloud, it will include:
- `telos_observatory_v3/` - Main app code
- `telos_purpose/` - Core TELOS modules
- `dual_pa_research_briefs/` - Validation evidence
- Any other files in the repository

**Note**: Large data files (>100MB) should be stored separately (e.g., GitHub LFS, cloud storage)

---

## App URL

After deployment, your app will be available at:
**https://telos-observatory.streamlit.app** (or similar)

You can customize the URL in Streamlit Cloud settings.

---

## Updating the App

Any push to the `experimental/dual-attractor` branch will automatically trigger a redeploy.

```bash
git add .
git commit -m "Update observatory"
git push origin experimental/dual-attractor
```

Streamlit Cloud will detect the push and rebuild the app.

---

## Next Steps After Deployment

1. Test the deployed app thoroughly
2. Share the URL with collaborators
3. Monitor usage in Streamlit Cloud dashboard
4. Consider custom domain (available on paid plans)

---

**Ready to deploy?** Just go to https://share.streamlit.io/ and follow steps 2-5 above!
