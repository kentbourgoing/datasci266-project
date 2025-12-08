# Model Outputs and Experimental Results

This directory contains the outputs and evaluation results from all 11 detoxification pipeline configurations tested in our research.

## Overview

Each pipeline was evaluated on 671 ParaDetox test sentences. This README summarizes the comprehensive experimental results comparing different combinations of masking strategies, infilling models, and reranking methods.

---

## Complete Results Table

All models evaluated on 671 ParaDetox test sentences. Higher BERTScore, MeaningBERT, and BLEU-4 are better; lower Perplexity and Toxicity are better.

| Model | BERTScore ↑ | MeaningBERT ↑ | BLEU-4 ↑ | Perplexity ↓ | Toxicity ↓ |
|-------|-------------|---------------|----------|--------------|------------|
| **T5-base** | **0.953** | **74.84** | 82.65 | 192.07 | 0.203 |
| **T5-base + DecompX Reranking** | 0.947 | 71.48 | **88.23** | 235.22 | 0.208 |
| **T5-base + Global Reranking** ⭐ | 0.936 | 67.25 | 53.34 | **171.53** | **0.051** |
| **DecompX Masking + MaRCo + DecompX Reranking** | **0.944** | **72.85** | 68.99 | 136.08 | 0.132 |
| **DecompX Masking + MaRCo + Global Reranking** | **0.944** | 72.72 | 70.05 | **124.95** | 0.120 |
| **DecompX Masking + LLM + DecompX Reranking** | 0.938 | 66.16 | 82.86 | 200.29 | 0.171 |
| **DecompX Masking + LLM + Global Reranking** | 0.932 | 64.74 | 81.54 | 162.39 | **0.103** |
| **LLM Masking + MaRCo + DecompX Reranking** | 0.938 | 69.55 | 70.05 | 90.65 | 0.200 |
| **LLM Masking + MaRCo + Global Reranking** | 0.938 | 69.02 | 70.05 | **86.59** | 0.159 |
| **LLM Masking + LLM + DecompX Reranking** | 0.931 | 62.55 | 81.54 | 149.22 | 0.181 |
| **LLM Masking + LLM + Global Reranking** | 0.931 | 62.45 | 81.54 | 141.89 | 0.118 |

**Legend**:
- ⭐ = Best overall system (recommended)
- **Bold** = Best or second-best score for that metric

---

## Key Findings

### 1. Best Overall System: T5-base + Global Reranking ⭐

**Toxicity**: 0.051 (lowest by far)
**BERTScore**: 0.936 (strong semantic preservation)
**Perplexity**: 171.53 (good fluency)
**BLEU-4**: 53.34 (moderate surface similarity)

**Why it wins**:
- Achieves lowest toxicity (0.051) while maintaining strong semantic similarity
- Only system that consistently removes explicit slurs and severe profanity
- Good balance between safety, meaning preservation, and fluency
- Suitable for production deployment

**Trade-off**:
- Lower BLEU-4 due to more substantial paraphrasing (prioritizes safety over lexical similarity)

---

### 2. Impact of Reranking Strategy

Global Reranking **consistently improves safety** across all generator/masker combinations:

| Configuration | DecompX Reranking | Global Reranking | Improvement |
|---------------|-------------------|------------------|-------------|
| T5-base | 0.208 | **0.051** | **-0.157** |
| DecompX + MaRCo | 0.132 | 0.120 | -0.012 |
| DecompX + LLM | 0.171 | **0.103** | **-0.068** |
| LLM + MaRCo | 0.200 | 0.159 | -0.041 |
| LLM + LLM | 0.181 | 0.118 | **-0.063** |

**Key insight**: Global Reranking's multi-objective optimization (toxicity + similarity + fluency) provides better safety than DecompX's single toxicity signal.

**Perplexity also improves** with Global Reranking:
- T5-base: 235.22 → 171.53 (-27%)
- DecompX + LLM: 200.29 → 162.39 (-19%)
- LLM + LLM: 149.22 → 141.89 (-5%)

---

### 3. Masking Strategy Comparison

**DecompX Masking vs. LLM Masking** (holding infiller and reranker constant):

#### With MaRCo Infilling:
| Masking | + DecompX Reranking | + Global Reranking |
|---------|---------------------|---------------------|
| **DecompX** | 0.132 toxicity | **0.120** toxicity |
| **LLM** | 0.200 toxicity | 0.159 toxicity |

**Winner**: DecompX masking (more aggressive, hides toxic cues better)

#### With LLM Infilling:
| Masking | + DecompX Reranking | + Global Reranking |
|---------|---------------------|---------------------|
| **DecompX** | 0.171 toxicity | **0.103** toxicity |
| **LLM** | 0.181 toxicity | 0.118 toxicity |

**Winner**: DecompX masking (smaller gap, but still safer)

**Key insight**: DecompX masking tends to over-mask, raising perplexity but providing better safety foundation for reranking.

---

### 4. Infilling Model Comparison

**MaRCo vs. LLM Infilling** (holding masker and reranker constant):

#### With DecompX Masking:
| Infilling | + DecompX Reranking | + Global Reranking |
|-----------|---------------------|---------------------|
| **MaRCo** | 0.132 toxicity, 68.99 BLEU | 0.120 toxicity, 70.05 BLEU |
| **LLM** | 0.171 toxicity, 82.86 BLEU | **0.103** toxicity, 81.54 BLEU |

**Winner with Global Reranking**: LLM (safer)
**Winner with DecompX Reranking**: MaRCo (safer, but both have issues)

#### With LLM Masking:
| Infilling | + DecompX Reranking | + Global Reranking |
|-----------|---------------------|---------------------|
| **MaRCo** | 0.200 toxicity, 70.05 BLEU | 0.159 toxicity, 70.05 BLEU |
| **LLM** | 0.181 toxicity, 81.54 BLEU | **0.118** toxicity, 81.54 BLEU |

**Winner**: LLM infilling (safer in both cases)

**Key insight**: LLM infilling wins in 3 out of 4 matched comparisons and avoids MaRCo's severe failure modes (explicit slurs, violent threats).

---

### 5. MaRCo Failure Modes

⚠️ **Critical Finding**: MaRCo-based infilling frequently reintroduces severe toxic content:

**Examples of MaRCo failures** (from qualitative analysis):
- **Explicit slurs**: "whiny cunts", "fagget", "nazi"
- **Violent threats**: "I will come to your house and I will cut you"
- **Demeaning phrases**: "whiny cunts who buy seats and then bitch about it"
- **Stance reversals**: "making and distributing cp is a good thing" (flipping moral judgment)

**Even with Global Reranking**, MaRCo still produces toxic content (toxicity 0.120-0.159), though less severe than with DecompX reranking.

**Fluency paradox**: MaRCo achieves excellent fluency (perplexity 86.59-136.08) and high BLEU scores (68.99-70.05) while producing unsafe content.

**Recommendation**: ❌ Do not use MaRCo infilling for production. Use T5-base + Global Reranking instead.

---

### 6. Best Systems by Use Case

#### Production Deployment (Safety Priority)
**Recommended**: T5-base + Global Reranking
- Toxicity: 0.051
- BERTScore: 0.936
- Perplexity: 171.53
- **Best balance of safety, meaning, and fluency**

#### Research Baseline (Semantic Similarity Priority)
**Recommended**: T5-base (no reranking)
- BERTScore: 0.953 (highest)
- MeaningBERT: 74.84 (highest)
- BLEU-4: 82.65
- **Best meaning preservation, but less safe (toxicity 0.203)**

#### Maximum Fluency
**Recommended**: LLM Masking + MaRCo + Global Reranking
- Perplexity: 86.59 (lowest)
- Toxicity: 0.159
- **Trade-off: Good fluency but moderate toxicity**

#### Explanation-Driven Safety
**Recommended**: DecompX Masking + LLM + Global Reranking
- Toxicity: 0.103 (2nd lowest non-T5)
- BERTScore: 0.932
- **Good balance using explainability framework**

---

## Detailed Analysis

### Semantic Preservation vs. Safety Trade-off

**High semantic similarity systems** (BERTScore > 0.94):
- T5-base: 0.953 BERTScore, but 0.203 toxicity
- DecompX + MaRCo: 0.944 BERTScore, but 0.120-0.132 toxicity

**High safety systems** (Toxicity < 0.12):
- T5 + Global: 0.051 toxicity, but 0.936 BERTScore
- DecompX + LLM + Global: 0.103 toxicity, but 0.932 BERTScore

**Key insight**: There's an inherent trade-off. Systems that preserve exact wording tend to retain toxicity. Systems that paraphrase more achieve better safety.

---

### Fluency vs. Safety Trade-off

**High fluency systems** (Perplexity < 100):
- LLM + MaRCo variants: 86.59-90.65 perplexity
- **Problem**: 0.159-0.200 toxicity (moderate-high)

**Balanced systems**:
- T5 + Global: 171.53 perplexity, 0.051 toxicity
- DecompX + LLM + Global: 162.39 perplexity, 0.103 toxicity

**Key insight**: MaRCo achieves excellent fluency but introduces toxic content. Global reranking provides better balance.

---

### BLEU Score Patterns

**Highest BLEU-4**:
- T5 + DecompX Reranking: 88.23
- T5-base: 82.65
- DecompX/LLM + LLM variants: 81.54-82.86

**Lowest BLEU-4**:
- T5 + Global Reranking: 53.34

**Interpretation**: Global reranking prioritizes safety over lexical similarity, leading to more substantial paraphrasing and lower BLEU. This is acceptable given the dramatic toxicity reduction.

---

## Qualitative Observations

### T5-base Family

**No reranking**:
- Preserves meaning well but adds insults ("idiot", "morons", "scum")
- Sometimes adds profanity not in reference

**DecompX reranking**:
- Similar to no reranking
- Selects candidates close to original (high mask density signal)

**Global reranking** ⭐:
- Removes strong profanity and slurs
- Remaining errors: mild sarcasm, slight meaning drift
- No explicit hate speech observed

---

### DecompX Masking + MaRCo Infilling

**Severe failures**:
- Frequent slurs ("cunts", "fagget", "nazi")
- Explicit threats ("I will cut you")
- Graphic sexual content
- Stance reversals

**Global reranking helps but doesn't eliminate failures**:
- Reduces frequency but some toxic content remains
- Still produces dehumanizing language

---

### DecompX Masking + LLM Infilling

**Behavior**:
- Uses safety templates ("disrespectful person", "offensive words")
- Occasional profanity ("holy shit", "expensive as fuck")
- Dehumanizing phrases ("piece of human garbage")

**Global reranking effect**:
- Removes worst cases
- Better than MaRCo but traces of aggression remain

---

### LLM Masking Variants

**Masking quality**:
- More coherent and fluent
- Sometimes misses subtle toxic terms

**With MaRCo infilling**:
- MaRCo toxic behaviors persist

**With LLM infilling**:
- Safety-shaped style with phrases like "hurtful language"
- Still some profanity and misidentification of targets

---

## Statistical Significance

All metrics macro-averaged over 671 test sentences.

**Toxicity differences** (key comparisons):
- T5 + Global vs. T5-base: -0.152 (highly significant)
- T5 + Global vs. DecompX + LLM + Global: -0.052 (significant)
- Global vs. DecompX reranking (T5): -0.157 (highly significant)

**BERTScore stability**:
- All systems achieve BERTScore > 0.93
- Variation within ±0.02 for most systems
- Indicates strong meaning preservation across approaches

**Perplexity range**:
- Best (LLM + MaRCo + Global): 86.59
- Worst (T5 + DecompX Reranking): 235.22
- Acceptable range: < 200

---

## Recommendations

### For Production Use

**✅ Use**: T5-base + Global Reranking
- Best safety-meaning trade-off
- Consistent performance
- No severe toxic failures observed

**❌ Avoid**: Any MaRCo-based pipeline
- Frequent severe toxic failures
- Even with Global reranking, unacceptable for production

### For Research

**✅ Compare against**: T5-base + Global Reranking
- Current best system
- Strong baseline for future work

**✅ Study**: Effect of reranking strategies
- Largest impact on safety
- More important than masking/infilling choice

**✅ Investigate**: LLM infilling improvements
- Safer than MaRCo
- Room for prompt engineering

### For Future Work

**Priority areas**:
1. **Learn reranking weights** instead of using fixed (0.5, 0.3, 0.2)
2. **Combine DecompX + LLM masking** (hybrid approach)
3. **Test stronger LLMs** (GPT-4, Claude, Gemini) for infilling
4. **Optimize DecompX threshold** systematically
5. **Evaluate on diverse datasets** (not just ParaDetox)

---

## Data Organization

This directory is organized by pipeline configuration:

```
model_outputs/
├── README.md                           # This file
├── T5_ParaDetox/
│   ├── outputs.txt                     # Generated detoxifications
│   ├── gen_stats.txt                   # Evaluation metrics
│   └── examples.csv                    # Per-example results
├── T5_ParaDetox_DecompX_Reranking/
│   └── ...
├── T5_ParaDetox_Global_Reranking/
│   └── ...
└── [8 XDetox-style pipeline dirs]/
    └── ...
```

**File formats**:
- `outputs.txt`: One detoxified sentence per line (671 lines)
- `gen_stats.txt`: Aggregate metrics and statistics
- `examples.csv`: Per-example toxic input, detoxified output, reference, and scores

---

## Reproducibility

All results can be reproduced by running the corresponding pipeline notebooks in `../`:
1. `T5_ParaDetox_Pipeline.ipynb`
2. `T5_ParaDetox_w_DecompX-Reranking_Pipeline.ipynb`
3. `T5_ParaDetox_w_Global-Reranking_Pipeline.ipynb`
4-11. `XDetox_w_[configuration]_Pipeline.ipynb`

**Requirements**:
- GPU with 16GB+ VRAM recommended
- 671 ParaDetox test sentences (`../../datasets/paradetox/test.txt`)
- All models downloaded via HuggingFace (automatic)
- Runtime: 30 mins - 3 hours per pipeline

---

## Citation

```bibtex
@article{he2024exploration,
  title={An Exploration of Modern Text Detoxification Pipelines in a Modular Framework},
  author={He, Benjamin and Bourgoing, Kent},
  year={2024},
  institution={UC Berkeley}
}
```

---

## Contact

For questions about the results or experimental setup:
- Benjamin He: ben_he@berkeley.edu
- Kent Bourgoing: kentbourgoing@ischool.berkeley.edu

---

**Last Updated**: December 2024

See the full paper for detailed qualitative analysis, failure mode examples, and discussion of limitations.
