# Evaluation Metrics for Text Detoxification

This directory contains comprehensive evaluation metrics used to assess the quality of detoxified text across three key dimensions: **safety** (toxicity reduction), **semantic preservation** (meaning retention), and **fluency** (naturalness).

## Overview

All 11 pipeline configurations in our research are evaluated using the same standardized metrics, allowing direct comparison across different masking, infilling, and reranking strategies.

---

## Metrics Implemented

### 1. Toxicity (`toxicity.py`)

**Purpose**: Measure the predicted toxicity probability of generated text

**Implementation**: XLM-R toxicity classifier (`textdetox/xlmr-large-toxicity-classifier-v2`)

**Usage**:
```python
from evaluation.toxicity import get_toxicity

texts = ["This is a safe sentence.", "This is offensive content."]
toxicity_scores = get_toxicity(texts)
# Returns: [0.05, 0.85]  (probabilities from 0 to 1)
```

**Interpretation**:
- **0.0 - 0.5**: Likely non-toxic
- **0.5 - 1.0**: Likely toxic
- Lower scores are better
- We report both mean toxicity and percentage above 0.5 threshold

**Notes**:
- Based on XLM-R large model fine-tuned on toxicity data
- Handles multiple languages but optimized for English
- May have biases in edge cases or context-dependent language

---

### 2. Perplexity (`perplexity.py`)

**Purpose**: Measure fluency and naturalness of generated text

**Implementation**: GPT-2-XL language model

**Usage**:
```python
from evaluation.perplexity import get_perplexity

texts = ["This is a well-formed sentence.", "This grammr bad is."]
perplexities = get_perplexity(texts)
# Returns: [45.2, 312.8]  (lower is better)
```

**Interpretation**:
- **< 100**: Very fluent, natural text
- **100 - 300**: Acceptable fluency
- **> 300**: Unnatural or poorly formed text
- Lower scores are better

**Notes**:
- Default model: `gpt2-xl` (can be changed via parameters)
- Automatically omits sentences with perplexity > 1e4 (edge cases)
- GPU recommended for faster computation

---

### 3. BLEU-4 (`bleu.py`)

**Purpose**: Measure n-gram overlap with human reference paraphrases

**Implementation**: SacreBLEU

**Usage**:
```python
from evaluation.bleu import get_bleu

hypotheses = ["You are very annoying sometimes."]
references = ["You can be frustrating at times."]
bleu_score = get_bleu(hypotheses, references)
# Returns: 15.3  (score from 0 to 100)
```

**Interpretation**:
- **> 80**: Very close to reference (high lexical overlap)
- **50 - 80**: Moderate similarity
- **< 50**: Substantial paraphrasing or divergence
- Higher scores indicate closer match to reference

**Notes**:
- Measures 1-gram through 4-gram overlap
- Sensitive to word choice (not semantic meaning)
- High BLEU doesn't guarantee safety or correctness

---

### 4. BERTScore (`bertscore.py`)

**Purpose**: Measure semantic similarity between generated text and reference

**Implementation**: BERTScore using contextual embeddings

**Usage**:
```python
from evaluation.bertscore import get_bertscore

hypotheses = ["You are very annoying sometimes."]
references = ["You can be frustrating at times."]
precision, recall, f1 = get_bertscore(hypotheses, references)
# Returns: (0.92, 0.88, 0.90)  (scores from 0 to 1)
```

**Interpretation**:
- **> 0.95**: Very high semantic similarity
- **0.90 - 0.95**: Strong semantic similarity
- **0.85 - 0.90**: Moderate similarity
- **< 0.85**: Significant semantic divergence
- Higher F1 scores are better

**Notes**:
- Based on BERT contextual embeddings
- Captures semantic meaning beyond word overlap
- More robust than BLEU to paraphrasing
- We report F1 score in our results

---

### 5. MeaningBERT (used via external model)

**Purpose**: Specialized metric for meaning preservation in paraphrases

**Implementation**: `davebulaval/MeaningBERT` from HuggingFace

**Usage**: Integrated into pipeline notebooks

**Interpretation**:
- **> 70**: Strong meaning preservation
- **60 - 70**: Moderate meaning preservation
- **< 60**: Significant meaning drift
- Higher scores are better

**Notes**:
- Specifically designed for paraphrase evaluation
- Complementary to BERTScore
- Used in all 11 pipeline evaluations

---

## Running Evaluation

### Single Metric Evaluation

Each metric can be run independently:

```python
# Toxicity
from evaluation.toxicity import get_toxicity
scores = get_toxicity(["Generated text here"])

# Perplexity
from evaluation.perplexity import get_perplexity
ppls = get_perplexity(["Generated text here"])

# BLEU
from evaluation.bleu import get_bleu
bleu = get_bleu(["Generated text"], ["Reference text"])

# BERTScore
from evaluation.bertscore import get_bertscore
p, r, f1 = get_bertscore(["Generated text"], ["Reference text"])
```

### Comprehensive Evaluation (`evaluate_all.py`)

Run all metrics at once:

```bash
python3 -m evaluation.evaluate_all \
    --orig_path path/to/original_texts.txt \
    --gen_path path/to/generated_texts.txt
```

**Input Format**:
- `orig_path`: Text file with one original (toxic) sentence per line
- `gen_path`: Text file with one generated (detoxified) sentence per line
- Lines in both files must correspond (same order)

**Output**:
- Metrics saved to `{gen_path}_stats.txt`
- Includes: toxicity, BLEU, BERTScore, perplexity
- Both aggregate statistics and per-example scores

**Example**:
```bash
python3 -m evaluation.evaluate_all \
    --orig_path datasets/paradetox/test_toxic.txt \
    --gen_path data/model_outputs/T5_ParaDetox_outputs.txt
```

Output file `data/model_outputs/T5_ParaDetox_outputs_stats.txt`:
```
=== Evaluation Results ===
Mean Toxicity: 0.051
Percent Toxic (>0.5): 2.3%
Mean Perplexity: 171.53
BLEU-4: 53.34
BERTScore F1: 0.936
MeaningBERT: 67.25
...
```

---

## Integration with Pipeline Notebooks

All 11 pipeline notebooks automatically run evaluation:

1. **Generate detoxified outputs** for test set
2. **Save outputs** to `data/model_outputs/{pipeline_name}/`
3. **Run evaluation** via `evaluate_all.py` or inline code
4. **Save metrics** to CSV and text files
5. **Display results** in notebook

**Typical notebook evaluation cell**:
```python
# After generation
outputs_df = pd.DataFrame({
    'toxic_input': toxic_inputs,
    'detoxified_output': detoxified_outputs,
    'reference': references
})

# Compute metrics
toxicity_scores = get_toxicity(outputs_df['detoxified_output'].tolist())
bertscore_f1 = get_bertscore(
    outputs_df['detoxified_output'].tolist(),
    outputs_df['reference'].tolist()
)[2]  # F1 score
perplexities = get_perplexity(outputs_df['detoxified_output'].tolist())
bleu_score = get_bleu(
    outputs_df['detoxified_output'].tolist(),
    outputs_df['reference'].tolist()
)

# Display results
print(f"Mean Toxicity: {np.mean(toxicity_scores):.3f}")
print(f"BERTScore F1: {np.mean(bertscore_f1):.3f}")
print(f"BLEU-4: {bleu_score:.2f}")
print(f"Perplexity: {np.mean(perplexities):.2f}")
```

---

## Evaluation Results Summary

### Best Performers by Metric

| Metric | Best System | Score |
|--------|-------------|-------|
| **Toxicity** ⭐ | T5-base + Global Reranking | 0.051 |
| **BERTScore** | T5-base (no reranking) | 0.953 |
| **MeaningBERT** | T5-base (no reranking) | 74.84 |
| **BLEU-4** | T5-base + DecompX Reranking | 88.23 |
| **Perplexity** | LLM Masking + MaRCo + Global | 86.59 |

### Trade-offs

**High semantic similarity ↔ High toxicity reduction**:
- Systems with highest BERTScore/BLEU tend to retain more toxicity
- Systems with lowest toxicity accept more paraphrasing
- T5-base + Global Reranking offers best balance

**Fluency ↔ Safety**:
- MaRCo infilling achieves excellent fluency but introduces toxic content
- Global Reranking improves safety with modest fluency cost
- LLM infilling is safer but sometimes less fluent than MaRCo

---

## Metric Limitations

### Toxicity Classifier Biases
- May flag non-toxic content containing identity terms
- Less accurate on indirect toxicity or sarcasm
- Context-dependent harm not fully captured

### BLEU Limitations
- Only measures word overlap, not meaning
- Penalizes valid paraphrases
- High BLEU doesn't guarantee safety or quality

### Perplexity Limitations
- GPT-2 may have different linguistic preferences
- Doesn't directly measure semantic correctness
- Can be affected by rare but valid constructions

### BERTScore Limitations
- May score toxic paraphrases highly if semantically similar
- Doesn't explicitly measure safety
- Can be sensitive to model choice

**Recommendation**: Always use multiple metrics together and perform qualitative analysis.

---

## Custom Evaluation

### Adding a New Metric

1. Create new file `your_metric.py`:
```python
def get_your_metric(texts, references=None):
    """
    Compute your custom metric.

    Args:
        texts: List[str] - generated texts
        references: Optional[List[str]] - reference texts

    Returns:
        scores: List[float] - metric scores
    """
    # Your implementation here
    return scores
```

2. Import in `evaluate_all.py`:
```python
from evaluation.your_metric import get_your_metric
```

3. Add to evaluation loop:
```python
your_scores = get_your_metric(generated_texts, references)
```

### Customizing Evaluation Weights

For Global Reranking, you can adjust metric weights:

```python
# In pipeline notebook
toxicity_weight = 0.5    # Prioritize safety
similarity_weight = 0.3  # Moderate meaning preservation
fluency_weight = 0.2     # Lower priority for fluency

# Custom weighting for specific use case
toxicity_weight = 0.7    # Maximum safety (e.g., children's content)
similarity_weight = 0.2
fluency_weight = 0.1
```

---

## Troubleshooting

**Problem**: Perplexity computation is very slow

**Solution**:
- Use GPU: `export CUDA_VISIBLE_DEVICES=0`
- Reduce batch size in `perplexity.py`
- Consider using smaller model (`gpt2` instead of `gpt2-xl`)

**Problem**: BERTScore gives unexpected results

**Solution**:
- Check reference text quality (must be non-toxic paraphrases)
- Ensure texts are properly formatted (no extra whitespace)
- Try different BERT model variants

**Problem**: Toxicity scores seem biased

**Solution**:
- Use multiple toxicity classifiers for comparison
- Perform manual review of flagged examples
- Consider context-specific classifiers if available

---

## Citation

**Our framework**:
```bibtex
@article{he2024exploration,
  title={An Exploration of Modern Text Detoxification Pipelines in a Modular Framework},
  author={He, Benjamin and Bourgoing, Kent},
  year={2024},
  institution={UC Berkeley}
}
```

**Evaluation methods**:
- BERTScore: Zhang et al. (2020)
- SacreBLEU: Post (2018)
- MeaningBERT: Bulaval (2023)

---

## Further Reading

- Full results table: See paper Table 1
- Qualitative analysis: See paper Section 4.4
- Model outputs: See `../data/model_outputs/README.md`
- Pipeline details: See `../README.md`
