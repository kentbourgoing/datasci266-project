# Rewrite Module: Masking and Infilling for Text Detoxification

This directory contains the core masking and infilling components used in our modular text detoxification framework. These components implement the MaRCo (Hallinan et al., 2023) approach for identifying and replacing toxic content.

## Overview

The rewrite module follows a **mask-and-infill pattern**:
1. **Masking**: Identify potentially toxic tokens and replace them with `<mask>`
2. **Infilling**: Generate safe replacements for masked positions using Product-of-Experts

This approach is used in 8 of our 11 pipeline configurations (the XDetox-style pipelines), combined with either DecompX-based or LLM-based masking, and either DecompX-based or Global reranking.

---

## Files in this Directory

| File | Description |
|------|-------------|
| `masking.py` | MaRCo masking using expert/anti-expert divergence |
| `generation.py` | Product-of-Experts infilling with BART models |
| `infilling.py` | Text infilling utilities and helpers |
| `gen_utils.py` | Generation helper functions |
| `generation_logits_process.py` | Custom logit processors for PoE |
| `rewrite_example.py` | End-to-end pipeline example script |

---

## Components

### 1. Masking (`masking.py`)

The masking component identifies potentially toxic tokens by comparing token probabilities under expert and anti-expert models.

#### Usage

**Instantiate the Masker**:
```python
from rewrite.masking import Masker

masker = Masker(
    seed=0,
    base_path="facebook/bart-base",
    antiexpert_path="hallisky/bart-base-toxic-antiexpert",
    expert_path="hallisky/bart-base-toxic-antiexpert",
    tokenizer="facebook/bart-base"
)
```

**Mask toxic tokens**:
```python
inputs = ["I'm surprised you got it done, seeing as you're all girls!"]
decoded_masked_inputs = masker.mask(inputs, thresh=1.2)
# Output: ["I'm surprised you got it done, seeing as you're all<mask>!"]
```

**Remove BOS/EOS tokens** (recommended):
```python
decoded_masked_inputs = [d.replace("<s>", "").replace("</s>", "")
                          for d in decoded_masked_inputs]
```

#### Parameters

- `thresh` (float): Divergence threshold for masking
  - **Lower threshold** (e.g., 0.8): More aggressive masking, higher safety, lower meaning preservation
  - **Higher threshold** (e.g., 1.5): Less masking, better meaning preservation, may retain some toxicity
  - **Default in paper**: 1.2
  - **Our experiments**: Use DecompX masking instead (threshold=0.2 on token importance)

#### Interactive Mode

Test masking interactively from the project root:
```bash
python3 -m rewrite.masking --thresh 1.2
```

Example output:
```
inputs: ["I'm surprised you got it done, seeing as you're all girls!",
         'You are a human']
masked inputs: ["I'm surprised you got it done, seeing as you're all<mask>!",
                'You are a<mask>']
```

---

### 2. Infilling (`generation.py`)

The infilling component fills masked positions using a **Product-of-Experts (PoE)** approach that combines three BART models:
- **Base model**: General language model
- **Expert model**: Trained on non-toxic text
- **Anti-expert model**: Trained on toxic text

The PoE formula combines logits as:
```
output_logits = α_b × base_logits + α_e × expert_logits - α_a × antiexpert_logits
```

#### Usage

**Instantiate the Infiller**:
```python
from rewrite.generation import Infiller

infiller = Infiller(
    seed=0,
    base_path="facebook/bart-base",
    antiexpert_path="hallisky/bart-base-toxic-antiexpert",
    expert_path="hallisky/bart-base-toxic-antiexpert",
    base_type="base",
    antiexpert_type="antiexpert",
    expert_type="expert",
    tokenizer="facebook/bart-base"
)
```

**Generate detoxified outputs**:
```python
outputs, decoded_outputs = infiller.generate(
    inputs,                    # Original toxic inputs
    decoded_masked_inputs,     # Masked versions from masker
    max_length=128,
    sample=False,              # True for sampling, False for greedy
    filter_p=1.0,
    k=0,                       # top-k (0 = disabled)
    p=1.0,                     # nucleus sampling threshold
    temperature=2.5,           # Sampling temperature
    alpha_a=1.5,              # Anti-expert weight
    alpha_e=4.25,             # Expert weight
    alpha_b=1.0,              # Base model weight
    repetition_penalty=1.0,
    batch_size=50,
    verbose=False
)
```

#### Generation Strategies

**Greedy Decoding** (used for microaggressions dataset):
```python
outputs, decoded = infiller.generate(
    inputs, masked_inputs,
    max_length=128,
    sample=False,
    temperature=2.5,
    alpha_a=1.5,
    alpha_e=4.25,
    alpha_b=1.0,
    repetition_penalty=1.0
)
```

**Sampling Decoding** (for candidate generation in reranking):
```python
outputs, decoded = infiller.generate(
    inputs, masked_inputs,
    max_length=128,
    sample=True,
    k=50,                     # top-k sampling
    p=0.95,                   # nucleus sampling
    temperature=2.5,
    alpha_a=1.5,
    alpha_e=4.75,
    alpha_b=1.0
)
```

#### Key Hyperparameters

| Parameter | Description | Typical Values | Effect |
|-----------|-------------|----------------|--------|
| `alpha_a` | Anti-expert weight | 1.5 | Higher = stronger toxic suppression |
| `alpha_e` | Expert weight | 4.25-4.75 | Higher = stronger non-toxic bias |
| `alpha_b` | Base weight | 1.0 | Usually kept at 1.0 |
| `temperature` | Sampling temp | 2.5-2.9 | Higher = more diverse outputs |
| `k` | Top-k | 50 | Limits vocabulary per step |
| `p` | Nucleus (top-p) | 0.95 | Cumulative probability threshold |

**Dataset-specific tuning**:
- ParaDetox: `alpha_e=4.75`, `temperature=2.5`
- Microaggressions: `alpha_e=4.25`, `temperature=2.5`
- SBF: `alpha_e=4.5`, `temperature=2.7`

#### Interactive Mode

Test infilling interactively:
```bash
python3 -m rewrite.generation
```

Example output:
```
inputs: ["I'm surprised you got it done, seeing as you're all girls!",
         'You are a human']
masked inputs: ["I'm surprised you got it done, seeing as you're all<mask>!",
                'You are a<mask>']
outputs: ["I'm surprised you got it done, seeing as you're all so busy!",
          'You are a human.']
```

You can specify custom arguments (see `argparser` at the bottom of `generation.py`).

---

### 3. End-to-End Pipeline (`rewrite_example.py`)

A complete example script that demonstrates the full masking → infilling pipeline.

#### Features

- **Data loading**: Loads evaluation datasets (dynabench, sbf, microaggressions)
- **Masking**: Initializes and runs the Masker
- **Infilling**: Initializes and runs the Infiller
- **Output saving**: Saves original inputs, masked inputs, and detoxified outputs

#### Usage

```bash
python3 -m rewrite.rewrite_example \
    --data_path datasets/paradetox/test.txt \
    --thresh 1.2 \
    --alpha_e 4.75 \
    --temperature 2.5
```

See the `argparser` in `rewrite_example.py` for all available parameters.

#### Special Mode: Multiple Hyperparameter Sets

Use `--gen_many` to try multiple generation hyperparameters on the same masked inputs:
```bash
python3 -m rewrite.rewrite_example \
    --data_path datasets/paradetox/test.txt \
    --gen_many
```

This iterates through predefined hyperparameter combinations to generate multiple candidate rewrites.

---

## Integration with Our Research Framework

### How This Module Fits Into the 11 Pipelines

**XDetox-style pipelines using MaRCo infilling** (4 pipelines):
1. DecompX Masking + **MaRCo Infilling** + DecompX Reranking
2. DecompX Masking + **MaRCo Infilling** + Global Reranking
3. LLM Masking + **MaRCo Infilling** + DecompX Reranking
4. LLM Masking + **MaRCo Infilling** + Global Reranking

**Alternative approaches** (7 pipelines):
- T5 baselines (3): Don't use masking/infilling, direct seq2seq
- LLM infilling pipelines (4): Use Mistral-7B instead of MaRCo

### Key Findings About MaRCo Infilling

Our research revealed important limitations:

⚠️ **MaRCo frequently reintroduces severe toxic content**, including:
- Explicit slurs ("cunts", "fagget")
- Violent threats ("I will cut you")
- Graphic sexual content
- Even stance reversals (flipping the moral judgment)

✅ **Global Reranking helps**, but doesn't fully eliminate failures:
- Toxicity drops from 0.132 → 0.120 (DecompX masking variant)
- Toxicity drops from 0.200 → 0.159 (LLM masking variant)

📊 **LLM infilling is safer** (3 out of 4 matched comparisons):
- Uses templates like "disrespectful person" instead of slurs
- Still produces some profanity ("holy shit", "as hell")
- Better controlled but slightly less fluent

### Recommended Usage

**For production**:
- ❌ Don't use MaRCo infilling alone
- ✅ Use T5-base + Global Reranking instead (toxicity 0.051)
- ✅ If using MaRCo, always apply Global Reranking

**For research**:
- ✅ MaRCo provides a useful baseline for comparison
- ✅ Demonstrates importance of reranking
- ✅ Shows limitations of pure Product-of-Experts approaches

---

## Auxiliary Files

### `gen_utils.py`
Helper functions for generation:
- Tokenization utilities
- Batch processing helpers
- Output formatting

### `generation_logits_process.py`
Custom logit processors for Product-of-Experts:
- Expert/anti-expert combination
- Temperature scaling
- Repetition penalty
- Top-k and nucleus filtering

### `infilling.py`
Additional infilling utilities:
- Alternative infilling strategies
- Debugging tools
- Visualization helpers

---

## Technical Details

### Product-of-Experts Formula

For each token position, the output distribution is:
```
P(token | context) = softmax(α_b × z_base + α_e × z_expert - α_a × z_antiexpert)
```

where:
- `z_base`, `z_expert`, `z_antiexpert` are logits from the three models
- `α_b`, `α_e`, `α_a` are learnable or tunable weights

**Intuition**:
- Expert model pushes toward non-toxic vocabulary
- Anti-expert model pushes away from toxic vocabulary
- Base model provides general fluency

### Masking by Divergence

The masker computes Jensen-Shannon divergence between expert and anti-expert predictions:
```
JSD(P_expert || P_antiexpert) > threshold → mask token
```

**Intuition**: Tokens where the two models disagree strongly are likely toxic.

---

## Troubleshooting

**Problem**: MaRCo generates toxic content even with high expert weight

**Solution**:
- Increase `alpha_e` (try 5.0 or higher)
- Use Global Reranking instead of DecompX reranking
- Consider switching to LLM infilling or T5-base

**Problem**: Outputs are too generic or lose meaning

**Solution**:
- Decrease `alpha_e` (try 4.0 or lower)
- Increase masking threshold (less aggressive masking)
- Increase temperature for more diversity

**Problem**: Repetitive outputs

**Solution**:
- Increase `repetition_penalty` (try 1.2-1.5)
- Use sampling instead of greedy decoding
- Increase temperature

---

## Citation

This implementation is based on:

**MaRCo**:
```bibtex
@inproceedings{hallinan2023detoxifying,
  title={Detoxifying Text with MaRCo: Controllable Revision with Experts and Anti-Experts},
  author={Hallinan, Skyler and Liu, Alisa and Choi, Yejin and Sap, Maarten},
  booktitle={ACL 2023},
  year={2023}
}
```

**Our framework**:
```bibtex
@article{he2024exploration,
  title={An Exploration of Modern Text Detoxification Pipelines in a Modular Framework},
  author={He, Benjamin and Bourgoing, Kent},
  year={2024},
  institution={UC Berkeley}
}
```

---

## Further Reading

- Original MaRCo repository: https://github.com/shallinan1/MarcoDetoxification
- Our full paper: `../An Exploration of Modern Text Detoxification Pipelines in a Modular Framework.pdf`
- XDetox paper: Lee et al. (2024)
- DecompX paper: Modarressi et al. (2023)
