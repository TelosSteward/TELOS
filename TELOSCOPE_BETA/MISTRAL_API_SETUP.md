# Using Mistral API Credits in Streamlit Deployment

**Issue**: You have $125 in Mistral API credits that you want to use instead of hitting free tier rate limits.

**Status**: Need to configure Mistral API key and ensure app uses it properly.

---

## Step 1: Get Your Mistral API Key

1. Go to https://console.mistral.ai/
2. Navigate to API Keys section
3. Create a new API key or copy existing one
4. **IMPORTANT**: Call Mistral support to ensure your paid credits are applied to this key

### Mistral Support Contact:
- Email: support@mistral.ai
- Console: https://console.mistral.ai/support

**What to tell them:**
> "I have $125 in credits but my API key is hitting free tier rate limits. Please ensure my API key [YOUR_KEY] is linked to my paid account with the credit balance."

---

## Step 2: Add to Streamlit Cloud Secrets

When deploying to Streamlit Cloud:

1. Click "Advanced settings" → "Secrets"
2. Add your Mistral key:

```toml
# Supabase Configuration
SUPABASE_URL = "https://ukqrwjowlchhwznefboj.supabase.co"
SUPABASE_KEY = "your-supabase-anon-key"

# Mistral API (PAID TIER with $125 credits)
MISTRAL_API_KEY = "your-mistral-api-key-here"
```

---

## Step 3: Verify App Uses Mistral Correctly

Check that the app is actually using your API key. Let me verify the code:

### Check services/mistral_client.py

The app should be reading the API key from environment variables:

```python
import os
from mistralai.client import MistralClient

# This should read from Streamlit secrets
api_key = os.getenv("MISTRAL_API_KEY")
client = MistralClient(api_key=api_key)
```

### Usage Tracking

Once deployed with your paid API key, you can monitor usage at:
- https://console.mistral.ai/usage
- See how your $125 credits are being consumed
- Track API calls and costs

---

## Rate Limits Comparison

### Free Tier (What you're hitting now)
- **Rate limit**: 1 request/second
- **Monthly quota**: Limited
- **Models**: Basic models only

### Paid Tier (With your $125 credits)
- **Rate limit**: Much higher (varies by model)
- **Monthly quota**: Based on credit balance
- **Models**: All models available including Mistral Large

---

## Mistral Pricing (FYI)

Your $125 gets you approximately:

| Model | Cost per 1M tokens | Your Credit Gets |
|-------|-------------------|------------------|
| Mistral Small | $2 | 62.5M tokens |
| Mistral Medium | $2.70 | 46M tokens |
| Mistral Large | $8 | 15.6M tokens |

**Example**: If using Mistral Small, $125 = ~60M tokens = ~45 million words = A LOT of conversations!

---

## Testing After Deployment

1. **Deploy to Streamlit Cloud** with MISTRAL_API_KEY in secrets
2. **Test the app** - try a conversation
3. **Check Mistral Console** - verify API calls show up
4. **Check rate limits** - should be much higher than before
5. **Monitor credit usage** - watch your $125 balance

---

## Fallback: Local Ollama (Always Free)

If you run out of Mistral credits or want to save them:

**Option A: Use Ollama on Streamlit Cloud**
- Not possible (Streamlit Cloud doesn't allow you to install Ollama)

**Option B: Deploy to Hetzner with Ollama**
- Install Ollama on Hetzner server
- Zero API costs
- Your $125 Mistral credits saved for emergencies

---

## Recommended Strategy

**Phase 1: Streamlit Cloud Beta (Use Mistral Credits)**
- Deploy to Streamlit Cloud for easy sharing
- Use your $125 Mistral credits
- Get 10-20 beta testers
- Gather feedback
- **Cost**: Free (using your credits)

**Phase 2: Hetzner Production (Use Ollama)**
- Deploy to Hetzner with local Ollama
- Zero ongoing API costs
- Unlimited usage
- Better performance
- **Cost**: €10/month server only

**Phase 3: Hybrid (Best of Both)**
- Keep Streamlit Cloud for demos/marketing
- Use Hetzner for actual users
- Save Mistral credits for special features

---

## Next Steps

1. ⬜ Get Mistral API key
2. ⬜ Call Mistral support to link paid credits
3. ⬜ Add key to Streamlit Cloud secrets
4. ⬜ Deploy and test
5. ⬜ Monitor usage in Mistral Console

---

**Bottom Line**: Your $125 credits will work great on Streamlit Cloud once you add the API key and confirm with Mistral support that it's linked to your paid account.
