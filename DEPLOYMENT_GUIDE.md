# TELOS Beta Deployment Guide

**Status:** ✅ Ready for Deployment
**Target:** Streamlit Cloud
**Entry Point:** `observatory/main.py`
**Branch:** `main`

---

## 🚀 Quick Deploy (5 Minutes)

### Prerequisites
- ✅ Code pushed to GitHub: `https://github.com/TelosSteward/TELOS.git`
- ✅ Mistral API key ready
- ✅ Streamlit Cloud account

### Steps

#### 1. Go to Streamlit Cloud
Navigate to: **https://share.streamlit.io/**

#### 2. Click "New app"
- **Repository:** `TelosSteward/TELOS`
- **Branch:** `main`
- **Main file path:** `observatory/main.py`
- **App URL:** Choose your subdomain (e.g., `telos-beta.streamlit.app`)

#### 3. Configure Secrets
Click **Advanced settings** → **Secrets**

Paste this configuration (replace with your actual Mistral API key):

```toml
[default]
MISTRAL_API_KEY = "your-actual-mistral-api-key-here"

[beta]
enabled = true
data_dir = "beta_testing/data"
min_turns_for_completion = 50
beta_duration_days = 14
```

#### 4. Click "Deploy!"
Streamlit will:
- Install dependencies from `requirements.txt`
- Start the Observatory app
- Give you a public URL

#### 5. Test Beta Flow
Once deployed:
1. Visit your app URL
2. Complete beta consent
3. Send a test message
4. Verify everything works

---

## 📋 Detailed Deployment Checklist

### Pre-Deployment

- [x] ✅ Beta implementation complete
- [x] ✅ Automated tests passed (5/5 scenarios)
- [x] ✅ Code committed to GitHub
- [x] ✅ `.streamlit/config.toml` configured
- [x] ✅ `.streamlit/secrets.toml` in `.gitignore`
- [x] ✅ `requirements.txt` up to date
- [ ] ⏳ Mistral API key obtained

### During Deployment

- [ ] Streamlit Cloud account logged in
- [ ] New app created with correct settings
- [ ] Secrets configured with actual API key
- [ ] Deployment initiated

### Post-Deployment

- [ ] App URL accessible
- [ ] Beta onboarding page loads
- [ ] Consent flow works
- [ ] First message sends successfully
- [ ] No console errors
- [ ] Share URL with first beta testers

---

## 🔑 Obtaining Mistral API Key

### If You Don't Have One:

1. Go to: **https://console.mistral.ai/**
2. Sign up or log in
3. Navigate to **API Keys**
4. Click **Create new key**
5. Copy the key (starts with `...`)
6. Add to Streamlit Cloud secrets

### Pricing Note:
- Mistral offers free tier with API credits
- Beta testing should stay within free limits
- Monitor usage in Mistral console

---

## 🎯 Configuration Reference

### Repository Settings
```
Repository: TelosSteward/TELOS
Branch: main
Main file: observatory/main.py
Python version: 3.9 (auto-detected)
```

### Required Secrets
```toml
[default]
MISTRAL_API_KEY = "sk-..." # Your actual key
```

### Optional Secrets
```toml
[beta]
enabled = true                    # Enable beta testing mode
data_dir = "beta_testing/data"    # Where to store feedback
min_turns_for_completion = 50     # Turns needed to unlock
beta_duration_days = 14           # OR days to unlock
```

---

## 📦 What Gets Deployed

### Core Application
- `observatory/main.py` - Main entry point
- `observatory/components/` - UI components
- `observatory/beta_testing/` - Beta session management
- `telos/` - TELOS governance core
- `.streamlit/config.toml` - Theme and settings

### Dependencies (from requirements.txt)
- `streamlit>=1.28.0` - Web framework
- `mistralai>=0.1.0` - LLM backend
- `sentence-transformers>=2.2.0` - Embeddings
- `torch>=2.0.0` - ML framework
- `numpy`, `pandas` - Data processing
- Plus others (see requirements.txt)

### Not Deployed
- ❌ `.streamlit/secrets.toml` (in .gitignore)
- ❌ `screenshots/` (test artifacts)
- ❌ `tests/` (not needed for production)
- ❌ `venv/` (Streamlit Cloud creates its own)

---

## 🧪 Testing After Deployment

### Critical Path Test (5 minutes)

**1. Beta Onboarding**
- [ ] Visit deployed URL
- [ ] See "Welcome to TELOS Beta"
- [ ] Read consent information
- [ ] Click consent checkbox
- [ ] Click "Continue to Beta"
- [ ] Redirected to BETA tab

**2. Tab Locking**
- [ ] BETA tab is highlighted
- [ ] DEMO tab is grayed out
- [ ] TELOS tab is grayed out
- [ ] Message: "Complete beta testing to unlock..."

**3. First Conversation**
- [ ] Type test message
- [ ] Click Send
- [ ] See "Contemplating..." response
- [ ] TELOS responds within 30 seconds
- [ ] Turn number shows: 1
- [ ] PA Status shows: "Calibrating"
- [ ] Fidelity shows: 0.000

**4. PA Calibration (turns 1-10)**
- [ ] Send 10 messages total
- [ ] Each response completes
- [ ] NO feedback buttons appear (turns 1-10)
- [ ] PA status updates

**5. Phase Transition (turn 11)**
- [ ] Send 11th message
- [ ] See phase transition message (if implemented)
- [ ] "🎯 PA Established!"
- [ ] Feedback buttons appear

**6. Feedback Collection**
- [ ] Thumbs up/down buttons visible
- [ ] Click thumbs up
- [ ] See confirmation: "✓ Thank you for your feedback!"
- [ ] Button changes to confirmation
- [ ] Cannot rate same turn twice

**7. Progress Tracking**
- [ ] Check sidebar
- [ ] See "Beta Progress" section
- [ ] Shows "Days: 0/14"
- [ ] Shows "Feedback: 1/50"

If all tests pass: ✅ **Ready for beta users!**

---

## 🐛 Troubleshooting

### App Won't Start

**Error:** "Module not found"
- **Fix:** Check requirements.txt includes all dependencies
- **Fix:** Redeploy from Streamlit Cloud dashboard

**Error:** "MISTRAL_API_KEY not found"
- **Fix:** Add key to Streamlit Cloud secrets
- **Fix:** Ensure format: `MISTRAL_API_KEY = "sk-..."`

**Error:** "Port already in use"
- **Fix:** Streamlit Cloud handles ports automatically, ignore locally

### App Crashes During Use

**Error:** Rate limit exceeded
- **Fix:** Check Mistral API quota
- **Fix:** Upgrade Mistral plan if needed

**Error:** Memory issues
- **Fix:** Reduce embedding model size (check embedding_provider.py)
- **Fix:** Clear session state more frequently

### Beta Features Not Working

**Issue:** Consent page doesn't appear
- **Fix:** Clear browser cache and reload
- **Fix:** Check if beta_consent_given is set (shouldn't be initially)

**Issue:** Tabs not locking
- **Fix:** Check browser console for JavaScript errors
- **Fix:** Verify CSS is loading correctly

**Issue:** Feedback buttons don't appear
- **Fix:** Ensure you've completed 11+ turns
- **Fix:** Check beta_phase_transition_shown in session state

---

## 📊 Monitoring Beta Launch

### Week 1 Metrics to Track

**User Engagement:**
- Beta consent rate (target: >80%)
- Average turns per user
- Completion rate within 2 weeks
- Drop-off points

**Technical Performance:**
- Response times (should be <30s)
- Error rates (target: <1%)
- API usage vs limits
- Session persistence issues

**Feedback Quality:**
- Feedback submission rate (target: >60%)
- Positive vs negative feedback ratio
- Completion criteria (50 turns vs 2 weeks)
- User comments (if collected)

### How to Access Data

**Streamlit Cloud Logs:**
1. Go to Streamlit Cloud dashboard
2. Click your app
3. View **Logs** tab
4. Monitor errors and warnings

**Beta Feedback Data:**
- Stored in `st.session_state.beta_feedback`
- Export via "Export Evidence" button
- Analyze feedback patterns
- Identify improvement areas

---

## 👥 Inviting Beta Testers

### First Wave (5-10 users)

**Ideal Beta Testers:**
- AI/ML researchers
- AI safety enthusiasts
- Technical users who understand LLMs
- People who can provide detailed feedback

**How to Invite:**
1. Share deployed URL
2. Explain beta testing purpose
3. Set expectations:
   - "Takes 10-15 minutes"
   - "Provide thumbs up/down feedback"
   - "Helps improve AI governance"
4. Ask for qualitative feedback too

### Message Template:

```
🔭 TELOS Beta Testing Invitation

You're invited to test TELOS Observatory - a new approach to AI governance
through geometric alignment in embedding spaces.

What you'll do:
• Have a conversation with an AI (10-15 minutes)
• Provide thumbs up/down feedback on responses
• Help validate TELOS governance system

Your data:
✓ Conversation content stays private
✓ Only mathematical measurements collected
✓ No login or personal info required
✓ Anonymous testing session

Try it here: [YOUR-STREAMLIT-URL]

Questions? Reply to this message!
```

---

## 🎯 Success Criteria

### Day 1 (Launch Day)
- [ ] App deployed successfully
- [ ] At least 1 complete test session
- [ ] No critical bugs
- [ ] URL shared with 5-10 testers

### Week 1
- [ ] 10+ beta users started
- [ ] 5+ beta users completed (50 turns or 2 weeks)
- [ ] 100+ feedback items collected
- [ ] Consent rate >80%
- [ ] Feedback submission rate >60%

### Week 2
- [ ] 25+ beta users started
- [ ] 15+ completed sessions
- [ ] 500+ feedback items
- [ ] Analyze preference patterns
- [ ] Identify improvements

---

## 📈 Next Steps After Beta

### Analyze Results
1. Export all beta feedback data
2. Calculate preference rates
3. Identify correlation with fidelity scores
4. Document findings

### Improve System
1. Address common user complaints
2. Refine PA calibration if needed
3. Improve response quality
4. Enhance UI/UX based on feedback

### Prepare for Full Launch
1. Write up beta results
2. Include in grant applications
3. Prepare public release
4. Scale infrastructure if needed

---

## 🔗 Quick Reference Links

**Streamlit Cloud:**
- Dashboard: https://share.streamlit.io/
- Docs: https://docs.streamlit.io/streamlit-community-cloud

**Mistral AI:**
- Console: https://console.mistral.ai/
- Docs: https://docs.mistral.ai/

**GitHub Repository:**
- TELOS: https://github.com/TelosSteward/TELOS

**Documentation:**
- Beta Implementation Plan: `BETA_IMPLEMENTATION_PLAN.md`
- Test Report: `BETA_TESTING_AUTOMATION_REPORT.md`
- Quick Start: `QUICK_START_GUIDE.md`

---

## ✅ Deployment Checklist

**Pre-Flight:**
- [x] Code tested locally
- [x] Automated tests passed
- [x] Code pushed to GitHub
- [x] Mistral API key ready
- [ ] Streamlit Cloud account ready

**Deployment:**
- [ ] App created on Streamlit Cloud
- [ ] Repository connected
- [ ] Main file path set: `observatory/main.py`
- [ ] Secrets configured
- [ ] Deploy clicked

**Post-Launch:**
- [ ] App accessible at public URL
- [ ] Beta flow tested end-to-end
- [ ] No critical errors
- [ ] Invitation message prepared
- [ ] First testers invited

---

**Ready to launch?** Follow the **Quick Deploy** steps at the top of this guide! 🚀

**Last Updated:** 2025-11-08
**Status:** ✅ Ready for Production
