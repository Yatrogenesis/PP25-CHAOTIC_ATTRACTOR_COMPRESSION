# CORRECTION NOTICE - Data Quality Update

**Date**: 2025-12-05  
**Author**: Francisco Molina Burgos (ORCID: 0009-0008-6093-8267)

---

## Summary of Changes

This repository has been updated with **real datasets** from Wikipedia and CC-News to replace templated sentences that were previously used for experiments.

### What Changed

| Aspect | Before | After |
|--------|--------|-------|
| **Data Source** | Templated sentences (`f"This is sentence {i}..."`) | Real Wikipedia + CC-News articles |
| **Wikipedia compression** | 187× | **167×** (validated) |
| **News compression** | 1775× | **178×** (validated) |
| **Consecutive similarity** | 0.97-0.99 (artificial) | 0.80-0.81 (natural) |
| **Accuracy loss (news)** | 1.15% | 13.38% |

---

## Why This Matters

The original datasets were **BERT embeddings of templated sentences**, not actual Wikipedia or news articles. This created:
- Artificial repetition and structure
- Inflated consecutive similarity (0.97 vs 0.80)
- Extreme compression ratios not achievable with natural language

### Example of Template Pattern (OLD):
```python
# Previous generate_bert_embeddings.py (lines 36-40):
for i in range(n_samples):
    topic = topics[i % len(topics)]
    sentences.append(
        f"This is sentence {i} about {topic} and related concepts in modern technology."
    )
```

This created **1775× compression on news** - impressive, but **not representative of real-world performance**.

---

## Corrected Results (Real Data)

### Wikipedia Dataset (HuggingFace: `wikimedia/wikipedia`)

- **Compression**: 167× (down from 187×, -11%)
- **Accuracy Loss**: 16.21% (up from 0.70%)
- **Consecutive Similarity**: 0.7978 (natural)
- **Data Source**: 2000 real sentences from Wikipedia articles
- **Model**: BERT-base-uncased (768D)

### News Dataset (CC-News)

- **Compression**: 178× (down from 1775×, **-90%**)
- **Accuracy Loss**: 13.38% (up from 1.15%)
- **Consecutive Similarity**: 0.8116 (natural)
- **Data Source**: 2000 real sentences from CC-News articles
- **Model**: BERT-base-uncased (768D)

---

## Scientific Integrity Assessment

### Positive Actions Taken:

✅ **Independent validation** caught the issue before peer review  
✅ **Immediate correction** with real datasets  
✅ **Transparent communication** of the error  
✅ **Reproducible results** - algorithms are deterministic  
✅ **Still excellent performance** - 167-178× is competitive

### Lessons Learned:

- Always validate with real-world data before publication
- Template-based synthetic data can create misleadingly favorable conditions
- Consecutive similarity is critical for compression performance
- Natural language exhibits ~0.80 similarity, not 0.97

---

## How to Use Real Datasets

### Quick Start

```bash
# Build and run with REAL datasets (now default)
cargo build --release
cargo run --release --bin compression-experiment

# Generate fresh REAL datasets (optional)
cd code
python3 generate_REAL_bert_embeddings.py  # ~15 minutes
```

### File Locations

```
code/data/real_embeddings/
├── wikipedia_2k_REAL.json      ← Real Wikipedia (DEFAULT)
├── news_temporal_2k_REAL.json  ← Real CC-News (DEFAULT)
├── wikipedia_2k.json           ← Old templated (preserved)
└── news_temporal_2k.json       ← Old templated (preserved)
```

---

## Validation Results

Full details at: **[Independent Validation Repository](https://github.com/Yatrogenesis/CHAOTIC_ATTRACTOR_VALIDATION)**

This repository contains:
- Complete validation reports
- All experimental runs (3 iterations)
- Real dataset generation scripts
- Comparative analysis of templated vs real data

| Run | Wikipedia | News | Reproducibility |
|-----|-----------|------|-----------------|
| **Run #1 (templated)** | 187.35× | 1775.72× | Perfect (deterministic) |
| **Run #2 (REAL)** | 166.63× | 177.97× | Perfect (deterministic) |
| **Run #3 (REAL)** | 166.63× | 177.97× | Perfect (deterministic) |

**Standard Deviation**: 0.00× (perfect reproducibility)

---

## Impact on Paper

### ⚠️ DO NOT CITE previous results (187×/1775×) - they used templated data

### ✅ USE corrected results:

> "We demonstrate up to 178× compression on real CC-News BERT embeddings and 167× on Wikipedia, using chaotic attractor-based dimensionality reduction (PCA+projection to 10D manifolds)."

**Accuracy**: 13-16% loss (acceptable for many applications)  
**Performance**: <10ms compression, <1ms decompression  
**Status**: Ready for publication with honest results

---

## Next Steps for Paper Revision

1. ✅ Update all result tables with real data
2. ✅ Add dataset description (HuggingFace Wikipedia + CC-News)
3. ✅ Re-compute correlation dimension D₂ on real data
4. ✅ Add "Limitations" section about consecutive similarity requirements
5. ✅ Update abstract: "178× compression" (not 1775×)
6. ⏳ Submit for peer review with corrected results

**Timeline**: 1-2 weeks for paper revision

---

## Contact

**Questions**: francisco.molina@yatrogenesis.ai  
**Validation Repo**: https://github.com/Yatrogenesis/CHAOTIC_ATTRACTOR_VALIDATION  
**Original Issue**: Data quality detected via independent validation

---

**Last Updated**: 2025-12-05 14:30 CST
