# TELOS Observatory - Streamlit Cloud Deployment Guide

This guide walks you through deploying TELOS Observatory to Streamlit Cloud for public access with the 5-message demo mode.

## Prerequisites

- GitHub account
- Streamlit Cloud account (free tier available at share.streamlit.io)
- Anthropic API key pool for demo mode (optional but recommended)

## Deployment Steps

### 1. Prepare Repository

Push your latest changes to GitHub on the `experimental/dual-attractor` branch:

```bash
git push origin experimental/dual-attractor
```

### 2. Deploy to Streamlit Cloud

1. Visit [share.streamlit.io](https://share.streamlit.io)
2. Click "New app"
3. Select your repository: `your-username/telos`
4. Set branch: `experimental/dual-attractor`
5. Set main file path: `telos_observatory_v3/main.py`
6. Click "Deploy"

### 3. Configure Secrets (Optional - for Demo Mode API Pool)

If you want to provide pooled API keys for demo mode users:

1. In Streamlit Cloud, go to your app's settings
2. Click "Secrets" in the sidebar
3. Add your pooled API keys in TOML format:

```toml
# Demo Mode Pooled API Keys
[demo_api_keys]
keys = [
    "sk-ant-api03-...",  # Key 1
    "sk-ant-api03-...",  # Key 2
    "sk-ant-api03-..."   # Key 3
]
```

**Note**: Without pooled keys, demo mode will show metrics/visualization only (no live generation).

### 4. Configure Environment (Auto-detected)

Streamlit Cloud automatically:
- Installs dependencies from `requirements.txt`
- Applies configuration from `.streamlit/config.toml`
- Sets Python 3.8+

## App Configuration

### Demo Mode Limits

- **5-message cap**: Users get 5 free messages in demo mode
- **No API key required**: Demo mode works without user API keys (if pool configured)
- **Upgrade path**: Clear prompt to "Exit Demo Mode" and add own API key

### Theme

Dark mode with TELOS branding:
- Primary color: Gold (#FFD700)
- Background: Dark (#1a1a1a)
- See `.streamlit/config.toml` for full theme

## Post-Deployment

### Test the Deployment

1. Visit your app URL (e.g., `https://your-app.streamlit.app`)
2. Verify demo mode loads with intro message
3. Send a few test messages (count should increment)
4. Verify warning at message 4
5. Verify lock at message 5
6. Test "Exit Demo Mode" → counter should reset

### Monitor Usage

In Streamlit Cloud dashboard:
- View app analytics
- Monitor resource usage
- Check error logs
- Track concurrent users

### Update Deployment

Simply push to the branch - Streamlit Cloud auto-deploys:

```bash
git push origin experimental/dual-attractor
```

Changes deploy in ~2-3 minutes.

## Sharing the App

### Public URL

Your app is public at: `https://your-app-name.streamlit.app`

Share this with:
- Technical communities (Discord, Reddit)
- AI/ML forums
- Colleagues and beta testers

**Avoid**: Mass promotion on X/Twitter (per project strategy)

### Embedding

You can embed the app in documentation or websites:

```html
<iframe
  src="https://your-app-name.streamlit.app/?embed=true"
  height="800"
  width="100%"
></iframe>
```

## Troubleshooting

### App Won't Start

**Check logs** in Streamlit Cloud dashboard:
- Look for import errors
- Verify all dependencies in `requirements.txt`
- Check Python version compatibility

### High Memory Usage

Torch + sentence-transformers can use significant memory.

**Solution**: Streamlit Cloud free tier has 1GB RAM
- If needed, upgrade to paid tier for more resources
- Or use lighter embedding model

### Slow Cold Starts

First load downloads sentence-transformer models (~500MB).

**Solution**:
- Normal on first run
- Subsequent loads are cached
- Paid tier offers persistent storage

### Demo Mode Pool Empty

If all pooled API keys hit rate limits:

**Solution**:
- Add more keys to secrets
- Implement rate limiting per key
- Or disable live generation in demo mode

## Security Considerations

### API Keys

- **Never commit** API keys to Git
- Store in Streamlit Cloud secrets only
- Rotate keys periodically

### Rate Limiting

Demo mode with pooled keys needs rate limiting:
- Limit requests per IP
- Implement key rotation
- Monitor for abuse

### User Data

- Demo sessions are temporary (not persisted)
- User API keys stored in session state only (not logged)
- Clear privacy policy in UI

## Cost Estimates

### Streamlit Cloud

- **Free tier**: Perfect for initial launch
  - Public apps
  - 1GB RAM
  - Community support

- **Paid tier** ($20/month if needed):
  - Private apps option
  - More resources
  - Priority support

### Anthropic API (Demo Pool)

Example costs with 100 users/day:
- 5 messages each = 500 messages/day
- Avg 500 tokens per message = 250K tokens/day
- Claude 3.5 Sonnet: ~$1-2/day
- ~$30-60/month for demo pool

**Budget recommendation**: Start with $50/month API budget

## Next Steps After Deployment

1. **Monitor first 24 hours**
   - Check for errors
   - Watch usage patterns
   - Gather initial feedback

2. **Collect Metrics**
   - Track demo→paid conversion
   - Monitor message counts
   - Identify popular features

3. **Iterate Based on Feedback**
   - Fix reported issues
   - Add requested features
   - Optimize performance

4. **Plan Discord Bot** (per roadmap)
   - After dual attractor architecture is stable
   - Use deployment learnings
   - Build autonomous TELOS Discord manager

## Support

For deployment issues:
- Streamlit Cloud: [docs.streamlit.io](https://docs.streamlit.io)
- TELOS specific: Check `/tests` and `DUAL_ATTRACTOR_ARCHITECTURE.md`

---

**Status**: Ready for deployment (experimental/dual-attractor branch)
**Last Updated**: 2025-11-02
**Next Milestone**: Discord bot integration
