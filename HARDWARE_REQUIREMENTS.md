# TELOS Hardware & Computational Requirements

**For Reproducibility and Independent Validation**

---

## Minimum System Requirements

### For Development & Local Testing

**CPU**:
- Intel i5/i7 or AMD Ryzen 5/7 (4+ cores)
- Apple Silicon (M1/M2/M3) works excellently

**RAM**:
- **Minimum**: 8GB
- **Recommended**: 16GB
- **Optimal**: 32GB (for large-scale validation runs)

**Storage**:
- **Minimum**: 20GB free space
- **Recommended**: 50GB (includes validation datasets, forensic logs)

**GPU**:
- **Not required** for basic TELOS governance
- **Optional**: NVIDIA GPU (CUDA 11+) for faster embedding computations
- CPU-only mode fully supported

**Operating System**:
- macOS 11+ (primary development environment)
- Linux (Ubuntu 20.04+, Debian 11+)
- Windows 10/11 (via WSL2 recommended)

---

## Computational Requirements by Use Case

### 1. **Basic TELOSCOPE Demo** (Streamlit Observatory)
- **CPU**: 2-4 cores
- **RAM**: 4-8GB
- **Time**: Instant startup
- **Cost**: Free (Streamlit Cloud) or $0-50/month (self-hosted)

### 2. **Research Validation** (Reproducing 2,000 Attack Results)
- **CPU**: 8+ cores (parallel processing)
- **RAM**: 16-32GB
- **Storage**: 10GB (validation datasets + forensic logs)
- **Time**: ~12 seconds (165.7 attacks/second as documented)
- **GPU**: Optional (2-3x speedup for embeddings)

### 3. **Institutional Deployment** (Multi-User Research Environment)
- **CPU**: 16+ cores
- **RAM**: 64GB+
- **Storage**: 500GB SSD
- **Network**: 1Gbps+
- **Load**: 100-500 concurrent users
- **Cost**: ~$200-500/month (cloud) or on-prem server

### 4. **Production Enterprise** (Fortune 500 Scale)
- **Kubernetes Cluster**: 3-10 nodes
- **CPU per node**: 32+ cores
- **RAM per node**: 128GB+
- **Storage**: 2TB+ distributed
- **Network**: 10Gbps+
- **Load**: 10,000+ concurrent users
- **Cost**: $5K-50K/month (scaled)

---

## Software Dependencies

### Core Requirements

**Python**: 3.9, 3.10, or 3.11 (3.9 tested extensively)

**Key Libraries** (versions in `requirements.txt`):
- `streamlit >= 1.28.0` - Web interface
- `sentence-transformers >= 2.2.0` - Embedding models
- `torch >= 2.0.0` - PyTorch backend
- `supabase >= 2.0.0` - Database client
- `mistralai >= 1.0.0` - LLM API client

**LLM Backend** (choose one):
- **Mistral API** (cloud) - $125 credits available, paid tier
- **Ollama** (local) - Free, runs mistral:7b locally
- **Anthropic Claude** (cloud) - Optional, for comparisons

**Embedding Model**:
- **Default**: `sentence-transformers/all-MiniLM-L6-v2` (90MB download)
- **Alternative**: `sentence-transformers/all-mpnet-base-v2` (420MB, higher quality)
- Downloads automatically on first run

---

## Validation Benchmark Compute Requirements

### Reproducibility Specifications

**To reproduce our 2,000 attack validation**:

1. **MedSafetyBench** (900 attacks):
   - Time: ~5-6 seconds
   - RAM: 8GB
   - Storage: 490KB output JSON

2. **HarmBench** (400 attacks):
   - Time: ~2-3 seconds
   - RAM: 8GB
   - Storage: Supabase (cloud)

3. **AgentHarm** (176 attacks):
   - Time: ~1-2 seconds
   - RAM: 8GB
   - Storage: 75KB output JSON

4. **Telemetric Keys Cryptographic Validation** (2,000 attacks):
   - Time: ~12 seconds total (165.7 attacks/second)
   - RAM: 4GB
   - Storage: Minimal (signatures in Supabase)

**Total Time**: ~12 seconds for full 2,000 attack suite
**Parallelizable**: Yes (multi-core speeds up proportionally)

---

## Cloud Environment Specifications

### Streamlit Cloud (Free Tier)
- **CPU**: Shared, 2 virtual cores
- **RAM**: 1GB (can upgrade to 8GB on Pro tier)
- **Storage**: 1GB
- **Limitations**: Sleep after 7 days inactivity
- **Cost**: Free (Community tier)

### Recommended Cloud Providers

**For Research Deployment**:
- **Streamlit Cloud Pro**: $20/month, 8GB RAM, custom domain
- **Hetzner VPS**: €10/month, 4GB RAM, 2 vCPU
- **DigitalOcean Droplet**: $24/month, 4GB RAM, 2 vCPU

**For Institutional Deployment**:
- **AWS EC2**: t3.xlarge ($133/month), 4 vCPU, 16GB RAM
- **Google Cloud**: e2-standard-4 ($122/month), 4 vCPU, 16GB RAM
- **Azure**: Standard_D4s_v3 ($140/month), 4 vCPU, 16GB RAM

---

## Network Requirements

**Outbound API Calls**:
- Mistral API: `https://api.mistral.ai`
- Supabase: `https://ukqrwjowlchhwznefboj.supabase.co`
- Hugging Face: `https://huggingface.co` (model downloads)

**Bandwidth**:
- **Initial setup**: ~500MB (model downloads)
- **Runtime**: ~10-50MB/hour (API calls, minimal)
- **Validation runs**: ~100MB (benchmark datasets)

**Firewall Rules**:
- Outbound HTTPS (443) required
- Inbound HTTP/HTTPS (80/443) for web interface
- No special ports required

---

## Reproducibility Environment

### Exact Configuration Used for Published Results

**Hardware**:
- **CPU**: Apple M2 Pro (10 cores)
- **RAM**: 32GB
- **Storage**: 512GB SSD
- **OS**: macOS 14.3

**Software**:
- **Python**: 3.9.18
- **PyTorch**: 2.1.0
- **Sentence-Transformers**: 2.2.2
- **Streamlit**: 1.28.0

**LLM Backend**:
- **Mistral API**: mistral-large-latest (paid tier)
- **Ollama**: mistral:7b (local fallback)

**Random Seeds**:
- Not set explicitly (validation is deterministic via cryptographic signatures)
- For exact reproduction, contact authors for seed configuration

**Execution Time**:
- 2,000 attacks: 12.07 seconds
- Throughput: 165.7 attacks/second

---

## Performance Benchmarks

### Expected Execution Times

| Task | Minimum Spec | Recommended Spec | Our Hardware |
|------|-------------|------------------|--------------|
| **TELOSCOPE Startup** | 10-15 sec | 5-8 sec | 3-5 sec |
| **Single LLM Query** | 2-5 sec | 1-2 sec | 0.5-1 sec |
| **PA Establishment** | 30-60 sec | 15-30 sec | 10-20 sec |
| **2,000 Attack Validation** | 30-60 sec | 15-30 sec | 12 sec |
| **Embedding Computation** | 100-200 ms | 50-100 ms | 20-50 ms |

### Bottlenecks

**Common Performance Limiters**:
1. **LLM API latency** (1-3 seconds per query) - dominates total time
2. **Embedding computation** (50-200ms) - scales with input length
3. **Network bandwidth** - affects model downloads, API calls
4. **RAM** - insufficient RAM causes swapping, 10x slowdown

**Optimization Tips**:
- Use GPU for embeddings (2-3x speedup)
- Enable caching for repeated queries
- Parallelize validation runs across cores
- Use local Ollama to eliminate API latency

---

## Docker/Container Specifications

**Docker Image Size**: ~2-3GB (includes Python, PyTorch, models)
**Container RAM**: 4GB minimum, 8GB recommended
**Container CPU**: 2+ cores

**Docker Compose**:
```yaml
version: '3.8'
services:
  telos:
    image: telos/teloscope:latest
    ports:
      - "8501:8501"
    environment:
      - MISTRAL_API_KEY=${MISTRAL_API_KEY}
      - SUPABASE_URL=${SUPABASE_URL}
      - SUPABASE_KEY=${SUPABASE_KEY}
    volumes:
      - ./data:/app/data
    mem_limit: 8g
    cpus: 2
```

**Note**: Docker containerization is planned for Phase 2 (post-grant funding). Current deployment uses native Python environment.

---

## Reproducibility Checklist

To independently reproduce TELOS validation results, you need:

- [ ] Hardware meeting minimum requirements (8GB RAM, 4+ cores)
- [ ] Python 3.9+ installed
- [ ] Git clone of repository
- [ ] `pip install -r requirements.txt`
- [ ] Mistral API key OR Ollama installed locally
- [ ] Supabase credentials (read-only provided for validation data)
- [ ] Internet connection (model downloads, API calls)
- [ ] ~1 hour for initial setup + validation runs

**Expected Output**:
- 0% Attack Success Rate across all benchmarks
- Forensic JSON files matching published results
- Telemetric signatures verifying governance actions

---

## Questions or Issues?

**For reproducibility support**:
- GitHub Issues: https://github.com/TelosSteward/TELOS/issues
- Email: research@telos-project.org (once established)
- Documentation: `/docs/` directory in repository

**For grant reviewers/peer reviewers**:
We are committed to supporting independent validation. Contact us for:
- Extended compute time estimates for your hardware
- Alternative LLM backend configuration (if Mistral API unavailable)
- Assistance with environment setup
- Access to validation datasets

---

**Last Updated**: November 24, 2025
**Document Version**: 1.0
**Corresponding Paper**: TELOS Technical Paper (docs/whitepapers/)
