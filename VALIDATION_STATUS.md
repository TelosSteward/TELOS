# Validation Status

This document describes the status and availability of benchmark data referenced in the README.

## Benchmark Data Included in Repository

| Benchmark | Location | Scenarios | Status |
|-----------|----------|-----------|--------|
| AILuminate | `validation/ailuminate/` | 1,200 | Included (MLCommons prompt set) |
| Nearmap Property Intel | `validation/nearmap/` | 235 | Included |
| Healthcare | `validation/healthcare/` | 280 | Included |
| OpenClaw | `validation/openclaw/` | 100 | Included |
| Civic Services | `validation/civic/` | 75 | Included |
| Agentic Benchmarks | `validation/agentic/` | See below | Included (metadata + results) |

## Benchmark Data Not Included

The following benchmarks were used during validation but their source data is not included in this repository due to licensing restrictions or because the data is available directly from the original authors:

| Benchmark | Scenarios | Source | How to Obtain |
|-----------|-----------|--------|---------------|
| HarmBench | 400 | CAIS (Center for AI Safety) | [github.com/centerforaisafety/HarmBench](https://github.com/centerforaisafety/HarmBench) |
| MedSafetyBench | 900 | NeurIPS 2024 | Available from original authors via the published paper |
| SB 243 Child Safety | 50 | California Legislature | Derived from SB 243 legislative text; contact authors for prompt set |
| XSTest (calibration) | 250 | Röttger et al. | [github.com/paul-rottger/exaggerated-safety](https://github.com/paul-rottger/exaggerated-safety) |

## Reproduction

For benchmarks included in the repository:
```bash
telos benchmark run -b nearmap --forensic
telos benchmark run -b healthcare --forensic
telos benchmark run -b openclaw --forensic
```

For benchmarks not included: obtain the source data from the links above, then follow the methodology described in `docs/REPRODUCTION_GUIDE.md`.

## Zenodo Deposits

Results and evaluation artifacts are archived with persistent DOIs:
- AILuminate: [10.5281/zenodo.18370263](https://doi.org/10.5281/zenodo.18370263)
- PropensityBench: [10.5281/zenodo.18562833](https://doi.org/10.5281/zenodo.18562833)
- AgentHarm: [10.5281/zenodo.18564855](https://doi.org/10.5281/zenodo.18564855)
- AgentDojo: [10.5281/zenodo.18565869](https://doi.org/10.5281/zenodo.18565869)

Note: Zenodo is a self-archiving service. These deposits provide persistent identifiers for evaluation artifacts. They are not peer-reviewed publications.
