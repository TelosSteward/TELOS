# TELOS Observatory - Streamlit Cloud Deployment Guide

**Date**: November 21, 2025
**Status**: Ready for Deployment
**Mode**: BETA/DEMO Only (Production features disabled for initial launch)

---

## Prerequisites

### 1. GitHub Repository
- Code must be in a GitHub repository
- Repository can be private or public
- Streamlit will read from main/master branch

### 2. Streamlit Cloud Account
- Sign up at https://streamlit.io/cloud
- Free tier: 1 app, 1GB resources, community support
- Link to your GitHub account

### 3. Supabase Credentials
You'll need these environment variables:
- `SUPABASE_URL`
- `SUPABASE_KEY`

Get them from: https://supabase.com/dashboard/project/YOUR_PROJECT/settings/api

---

## Deployment Steps

### Step 1: Prepare Repository

1. **Ensure files are committed**:
   ```bash
   cd /Users/brunnerjf/Desktop/Privacy_PreCommit/TELOSCOPE_BETA
   git status
   ```

2. **Push to GitHub** (if not already):
   ```bash
   git add .
   git commit -m "Prepare for Streamlit deployment - BETA/DEMO mode"
   git push origin main
   ```

### Step 2: Deploy to Streamlit Cloud

1. **Go to Streamlit Cloud**:
   - Visit https://share.streamlit.io/
   - Click "New app"

2. **Connect Repository**:
   - Repository: `your-username/your-repo-name`
   - Branch: `main`
   - Main file path: `TELOSCOPE_BETA/main.py`

3. **Add Secrets**:
   Click "Advanced settings" → "Secrets"

   Paste this format:
   ```toml
   # Supabase Configuration
   SUPABASE_URL = "https://ukqrwjowlchhwznefboj.supabase.co"
   SUPABASE_KEY = "your-anon-key-here"

   # Mistral API (Optional - for Steward features)
   MISTRAL_API_KEY = "your-mistral-key-here"

   # Anthropic API (Optional - for comparisons)
   ANTHROPIC_API_KEY = "your-anthropic-key-here"
   ```

4. **Deploy**:
   - Click "Deploy!"
   - Wait 2-5 minutes for installation
   - App will be live at: `https://your-app-name.streamlit.app`

---

## What's Enabled in BETA/DEMO Mode

### ✅ DEMO Tab (Default)
- Progressive demo slideshow (12 slides)
- Interactive Q&A with demo data
- No login required
- Perfect for showcasing TELOS

### ✅ BETA Tab (Unlocked after demo)
- PA onboarding (create Primacy Attractor)
- User consent collection
- Beta feedback forms
- Session tracking in Supabase

### ❌ LIVE Tab (Disabled)
- Commented out in `main.py`
- Requires full LLM API integration
- Will be enabled post-beta

### ❌ DEV/OPS Tabs (Hidden)
- Developer tools
- Only shown locally

---

## Environment Variables

### Required for Basic Functionality
```bash
SUPABASE_URL=https://ukqrwjowlchhwznefboj.supabase.co
SUPABASE_KEY=your-anon-public-key
```

### Optional (Enhanced Features)
```bash
MISTRAL_API_KEY=your-key  # For Steward LLM
ANTHROPIC_API_KEY=your-key  # For Claude comparisons
```

---

## Post-Deployment Checklist

### Immediate Testing
- [ ] App loads without errors
- [ ] DEMO tab shows progressive slideshow
- [ ] Can navigate through all 12 slides
- [ ] BETA tab unlocks after demo completion
- [ ] PA onboarding flow works
- [ ] Consent forms submit to Supabase
- [ ] No console errors

### User Experience
- [ ] Colors match TELOS brand (muted gold #F4D03F)
- [ ] Dark mode enabled
- [ ] Responsive on mobile
- [ ] No broken images
- [ ] Smooth transitions

### Database
- [ ] Supabase connection works
- [ ] PA submissions save correctly
- [ ] Beta consent records created
- [ ] Session tracking functional

---

## Monitoring

### Streamlit Cloud Dashboard
- View logs in real-time
- Monitor resource usage
- Track app status
- See visitor analytics

### Supabase Dashboard
- Monitor database queries
- Check user signups
- View PA submissions
- Track beta consents

---

## Troubleshooting

### Error: "ModuleNotFoundError"
**Fix**: Add missing package to `requirements.txt`

### Error: "Supabase connection failed"
**Fix**: Check secrets are correctly formatted (no quotes around values in Streamlit Cloud)

### Error: "App won't start"
**Fix**: Check Streamlit Cloud logs for Python errors

### Performance Issues
**Fix**: Free tier has limited resources. Upgrade to $20/month for better performance.

---

## Custom Domain (Optional)

1. Upgrade to Streamlit Cloud Pro ($20/month)
2. Go to app settings → Domain
3. Add custom domain (e.g., `observatory.telos.ai`)
4. Update DNS records as instructed
5. SSL certificate automatically provisioned

---

## Scaling Plan

### Phase 1: Free Tier (Current)
- 1 app
- 1GB RAM
- Community support
- Perfect for beta testing

### Phase 2: Streamlit Cloud Pro
- Multiple apps
- 4GB RAM per app
- Priority support
- Custom domains
- **Cost**: $20/month per app

### Phase 3: Self-Hosted (Future)
- Deploy to Hetzner VPS
- Full control
- No Streamlit Cloud costs
- More scalable
- **Cost**: ~€10/month

---

## Current Repository Structure

```
TELOSCOPE_BETA/
├── main.py                 # Entry point
├── requirements.txt        # Python dependencies
├── .streamlit/
│   ├── config.toml        # Theme configuration
│   └── secrets.toml       # Local secrets (DO NOT COMMIT)
├── components/            # UI components
├── core/                  # Core logic
├── services/              # API integrations
├── demo_mode/             # Demo slideshow
└── config/                # Configuration files
```

---

## Security Notes

### What's Public
- DEMO mode content
- UI code
- Business logic
- Color scheme

### What's Private
- Supabase keys (stored in Streamlit secrets)
- API keys (Mistral, Anthropic)
- User PAs
- Beta feedback
- Session data

### Best Practices
- ✅ Use environment variables for all secrets
- ✅ Never commit `.streamlit/secrets.toml` to GitHub
- ✅ Use Supabase RLS (Row Level Security) for data isolation
- ✅ Rotate API keys if exposed

---

## Next Steps After Deployment

1. **Test thoroughly** on mobile and desktop
2. **Share beta link** with 10-20 users
3. **Gather feedback** via beta forms
4. **Monitor Supabase** for PA submissions
5. **Iterate** based on user feedback
6. **Enable LIVE tab** when ready for production

---

## Support

### Streamlit Issues
- Docs: https://docs.streamlit.io
- Forum: https://discuss.streamlit.io
- Status: https://status.streamlit.io

### TELOS Issues
- Check app logs in Streamlit Cloud
- Review Supabase logs
- Test locally first: `streamlit run main.py`

---

**Ready to Deploy!** 🚀

The app is configured for Streamlit Cloud with BETA/DEMO mode enabled. Just push to GitHub and deploy via Streamlit Cloud dashboard.
