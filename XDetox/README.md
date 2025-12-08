# XDetox: Modular Text Detoxification Framework

This directory contains the core implementation of our modular text detoxification framework, including all 11 pipeline configurations tested in our research.

## Directory Structure

```
XDetox/
├── README.md                                    # This file
├── [11 Pipeline Notebooks]                      # Experimental configurations
├── DecompX/                                     # DecompX explainability framework
├── datasets/                                    # Toxicity evaluation datasets
├── data/                                        # Experiment outputs and results
├── rewrite/                                     # Core detoxification components
└── evaluation/                                  # Evaluation metrics
```

---

## Pipeline Notebooks

The directory contains **11 Jupyter notebooks**, each implementing a different detoxification pipeline configuration. Each notebook is self-contained and can be run independently.

### T5-based Baselines (3 notebooks)

#### 1. `T5_ParaDetox_Pipeline.ipynb`
- **Description**: Baseline T5-base model fine-tuned on ParaDetox dataset
- **Method**: Direct sequence-to-sequence generation with beam search (beam_size=5)
- **Use Case**: Simple, fast detoxification without reranking
- **Results**: High semantic similarity (BERTScore 0.953) but moderate toxicity (0.203)

#### 2. `T5_ParaDetox_w_DecompX-Reranking_Pipeline.ipynb`
- **Description**: T5-base with DecompX-based reranking
- **Method**:
  - Generates 10 candidates using top-k/nucleus sampling
  - Reranks by summing token-level toxicity scores from DecompX
  - Selects candidate with lowest cumulative toxicity
- **Use Case**: Testing whether explanation-based reranking improves T5
- **Results**: Similar to baseline, sometimes keeps toxic content (toxicity 0.208)

#### 3. `T5_ParaDetox_w_Global-Reranking_Pipeline.ipynb` ⭐
- **Description**: T5-base with Global Reranking (our novel method)
- **Method**:
  - Generates 10 candidates
  - Scores each by: 0.5×(1-toxicity) + 0.3×similarity + 0.2×fluency
  - Selects highest-scoring candidate
- **Use Case**: Best overall system for production deployment
- **Results**: **Lowest toxicity (0.051)** while maintaining good meaning preservation
- **⭐ Recommended for most applications**

---

### XDetox-style Pipelines: DecompX Masking (4 notebooks)

These pipelines use **DecompX token-level explanations** to identify and mask toxic spans before infilling.

#### 4. `XDetox_w_DecompX-Masking-DecompX-Reranking_Pipeline.ipynb`
- **Masking**: DecompX RoBERTa (threshold=0.2)
- **Infilling**: MaRCo Product-of-Experts (BART-based)
- **Reranking**: DecompX cumulative toxicity
- **Results**: Toxicity 0.132, good fluency (perplexity 136.08)
- **Note**: MaRCo can reintroduce severe toxic content

#### 5. `XDetox_w_DecompX-Masking-Global-Reranking_Pipeline.ipynb`
- **Masking**: DecompX RoBERTa (threshold=0.2)
- **Infilling**: MaRCo Product-of-Experts
- **Reranking**: Global multi-objective
- **Results**: Toxicity 0.120, improved safety over DecompX reranking
- **Improvement**: Global reranking reduces MaRCo failures

#### 6. `XDetox_w_DecompX-Masking_LLM-Infilling_DecompX-Reranking_Pipeline.ipynb`
- **Masking**: DecompX RoBERTa (threshold=0.2)
- **Infilling**: Mistral-7B-Instruct with custom prompts
- **Reranking**: DecompX cumulative toxicity
- **Results**: Toxicity 0.171, high BLEU-4 (82.86)
- **Behavior**: LLM uses safety-shaped templates ("disrespectful person")

#### 7. `XDetox_w_DecompX-Masking_LLM-Infilling_Global-Reranking_Pipeline.ipynb`
- **Masking**: DecompX RoBERTa (threshold=0.2)
- **Infilling**: Mistral-7B-Instruct
- **Reranking**: Global multi-objective
- **Results**: Toxicity 0.103 (2nd best non-T5 system)
- **Trade-off**: Lower toxicity, slightly reduced semantic similarity

---

### XDetox-style Pipelines: LLM Masking (4 notebooks)

These pipelines use **Mistral-7B-Instruct** with custom prompts to identify toxic spans, replacing DecompX masking.

#### 8. `XDetox_w_LLM-Masking_DecompX-Reranking_Pipeline.ipynb`
- **Masking**: LLM prompt-based (Mistral-7B)
- **Infilling**: MaRCo Product-of-Experts
- **Reranking**: DecompX cumulative toxicity
- **Results**: Toxicity 0.200, excellent fluency (perplexity 90.65)
- **Note**: LLM masking is more selective, sometimes misses subtle toxic terms

#### 9. `XDetox_w_LLM-Masking_Global-Reranking_Pipeline.ipynb`
- **Masking**: LLM prompt-based (Mistral-7B)
- **Infilling**: MaRCo Product-of-Experts
- **Reranking**: Global multi-objective
- **Results**: Toxicity 0.159, best fluency (perplexity 86.59)
- **Improvement**: Global reranking significantly improves safety

#### 10. `XDetox_w_LLM-Masking_LLM-Infilling_DecompX-Reranking_Pipeline.ipynb`
- **Masking**: LLM prompt-based (Mistral-7B)
- **Infilling**: LLM prompt-based (Mistral-7B)
- **Reranking**: DecompX cumulative toxicity
- **Results**: Toxicity 0.181, high BLEU-4 (81.54)
- **Behavior**: End-to-end LLM pipeline, coherent but sometimes overly safe

#### 11. `XDetox_w_LLM-Masking_LLM-Infilling_Global-Reranking_Pipeline.ipynb`
- **Masking**: LLM prompt-based (Mistral-7B)
- **Infilling**: LLM prompt-based (Mistral-7B)
- **Reranking**: Global multi-objective
- **Results**: Toxicity 0.118, balanced metrics
- **Use Case**: Fully LLM-based pipeline with global safety optimization

---

## Folders

### `DecompX/`
**Purpose**: Implementation of the DecompX explainability framework (ACL 2023)

**Contents**:
- `src/` - Core DecompX implementation
  - `modeling_bert.py` - BERT with decomposition support
  - `modeling_roberta.py` - RoBERTa with decomposition support
  - `decompx_utils.py` - Utilities for token-level attribution
- `experiments/` - Original DecompX research experiments
- `notebooks/` - DecompX demonstration notebooks
- `requirements.txt` - Python dependencies

**Description**: DecompX decomposes transformer predictions into token-level contributions, providing faithful explanations for toxicity classification. Used for both masking (identifying toxic tokens) and reranking (selecting least toxic candidate).

**Key Paper**: Modarressi et al. (2023) - "DecompX: Explaining Transformers Decisions by Propagating Token Decomposition"

---

### `datasets/`
**Purpose**: Collection of 7 toxicity datasets for evaluation

**Contents**:
- `paradetox/` (670 test examples) - Main benchmark, parallel toxic/non-toxic pairs
- `microagressions/` - Subtle offensive language, test/val splits
- `sbf/` (Social Bias Frames) - Social bias annotations with README
- `dynabench/` - Dynamically collected adversarial examples
- `jigsaw_full_30/` - Jigsaw Kaggle competition data subset
- `appdia/` - Discourse-aware annotations
- `_subsets/` - Runtime-generated dataset subsets for experiments

**Dataset Selection**: Each pipeline notebook can be configured to run on any of these datasets. ParaDetox is used for all main results in the paper.

**Format**: Varies by dataset (txt, csv, tsv). Each has specific loading utilities in the pipeline notebooks.

---

### `data/`
**Purpose**: Experiment outputs and cached results

**Contents**:
- `dexp_outputs/` - DecompX experiment outputs
  - Token-level attribution scores
  - Masked sentence intermediates
  - Toxicity importance values

- `model_outputs/` - Results from all 11 pipeline configurations
  - Generated detoxified text
  - Evaluation metrics (toxicity, BERTScore, BLEU, perplexity)
  - Per-example statistics
  - See `model_outputs/README.md` for detailed results summary

**Note**: Some model outputs may be large. Outputs are organized by pipeline configuration and dataset.

---

### `rewrite/`
**Purpose**: Core detoxification pipeline components (masking and infilling)

**Key Files**:
- `masking.py` - MaRCo masking using expert/anti-expert divergence
- `generation.py` - Product-of-Experts infilling with BART models
- `infilling.py` - Text infilling utilities and helpers
- `gen_utils.py` - Generation helper functions
- `generation_logits_process.py` - Custom logit processors
- `rewrite_example.py` - End-to-end pipeline example script

**Description**: Implements the mask-and-infill pattern for detoxification. Can be run interactively or imported into notebooks.

**See**: `rewrite/README.md` for detailed usage instructions and API documentation.

---

### `evaluation/`
**Purpose**: Comprehensive evaluation metrics for detoxification quality

**Metrics Implemented**:
- `toxicity.py` - XLM-R toxicity classifier scoring
- `perplexity.py` - GPT-2-XL fluency measurement
- `bleu.py` - BLEU-4 n-gram overlap
- `bertscore.py` - BERTScore semantic similarity
- `evaluate_all.py` - Run all metrics together

**Usage**: Each pipeline notebook automatically calls these evaluation functions. Can also be run standalone via command line.

**See**: `evaluation/README.md` for detailed metric descriptions and API usage.

---

## Running the Pipelines

### Prerequisites
1. Install dependencies:
   ```bash
   cd DecompX
   pip install -r requirements.txt
   ```

2. GPU recommended (CUDA support) for reasonable runtime

### Basic Workflow

1. **Choose a pipeline notebook** based on your needs:
   - For best results: `T5_ParaDetox_w_Global-Reranking_Pipeline.ipynb`
   - For fastest execution: `T5_ParaDetox_Pipeline.ipynb`
   - For experimentation: Any XDetox-style variant

2. **Open the notebook** in Jupyter:
   ```bash
   jupyter notebook T5_ParaDetox_w_Global-Reranking_Pipeline.ipynb
   ```

3. **Configure parameters** (optional):
   - Dataset selection (default: ParaDetox)
   - Masking threshold (default: 0.2 for DecompX)
   - Generation hyperparameters (temperature, top-k, etc.)
   - Reranking weights (for Global Reranking)

4. **Run all cells** to execute the pipeline:
   - Loads dataset and models
   - Performs masking (if applicable)
   - Generates candidate detoxifications
   - Reranks and selects final outputs
   - Evaluates with all metrics
   - Saves results to `data/model_outputs/`

5. **Review results**:
   - Metrics printed in notebook
   - Outputs saved to text files
   - Statistics saved to CSV

---

## Pipeline Comparison Quick Reference

| Pipeline | Toxicity | Best For |
|----------|----------|----------|
| **T5 + Global Reranking** | **0.051** ⭐ | Production deployment |
| T5 baseline | 0.203 | Fast prototyping |
| DecompX + LLM + Global | 0.103 | Explanation-driven safety |
| LLM + LLM + Global | 0.118 | Fully LLM-based workflow |
| DecompX + MaRCo + DecompX | 0.132 | Research on masking |

---

## Key Hyperparameters

### Global Reranking Weights
```python
w_toxicity = 0.5      # Weight for (1 - toxicity_prob)
w_similarity = 0.3    # Weight for semantic similarity
w_fluency = 0.2       # Weight for fluency (GPT-2 perplexity)
```

### DecompX Masking
```python
threshold = 0.2       # Higher = less masking, more meaning preservation
                      # Lower = more masking, more aggressive detoxification
```

### MaRCo Generation (Product-of-Experts)
```python
alpha_a = 1.5         # Anti-expert weight (toxic model)
alpha_e = 4.25-4.75   # Expert weight (non-toxic model)
alpha_b = 1.0         # Base model weight
temperature = 2.5     # Sampling temperature
```

### LLM Infilling (Mistral-7B)
```python
temperature = 0.7     # Lower = more deterministic
top_p = 0.95         # Nucleus sampling threshold
num_candidates = 10   # Number of candidates for reranking
```

---

## Customization

### Adding a New Dataset
1. Add dataset files to `datasets/your_dataset/`
2. Create loading function in pipeline notebook
3. Ensure format: toxic text + optional reference
4. Run pipeline with your dataset

### Adding a New Reranking Method
1. Implement scoring function for candidates
2. Add to reranking cell in notebook
3. Compare with DecompX and Global methods
4. Evaluate on ParaDetox test set

### Tuning Hyperparameters
- Masking threshold: Affects safety vs. meaning trade-off
- Reranking weights: Adjust based on application priorities
- Generation temperature: Higher = more diverse, lower = safer
- Number of candidates: More candidates = better reranking, slower runtime

---

## Citation

If you use this framework, please cite our paper:

```bibtex
@article{he2024exploration,
  title={An Exploration of Modern Text Detoxification Pipelines in a Modular Framework},
  author={He, Benjamin and Bourgoing, Kent},
  year={2024},
  institution={UC Berkeley}
}
```

---

## Acknowledgments

This framework integrates components from:
- **XDetox** (Lee et al., 2024)
- **DecompX** (Modarressi et al., 2023)
- **MaRCo** (Hallinan et al., 2023)
- **ParaDetox** (Logacheva et al., 2022)

---

**Note**: Notebooks may take 30 minutes to several hours to run depending on dataset size and GPU availability. Results are cached in `data/model_outputs/` for reuse.
