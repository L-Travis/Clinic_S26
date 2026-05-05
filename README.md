# Clinic S26 — LLM Reasoning Benchmark

> **Harvey Mudd Clinic Program · Spring 2026**
> Evaluating reasoning strategies for Large Language Models on deductive logic puzzles.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Repository Structure](#repository-structure)
3. [Dataset](#dataset)
4. [Reasoning Modes](#reasoning-modes)
5. [Testbench Architecture](#testbench-architecture)
6. [Grading Methodology](#grading-methodology)
7. [Experiment Results](#experiment-results)
   - [Run Overview](#run-overview)
   - [v1 Results — step-3.5-flash, Thinking ON](#v1-results--step-35-flash-thinking-on)
   - [v2 Results — step-3.5-flash, Thinking OFF](#v2-results--step-35-flash-thinking-off)
   - [Qwen 3.5 4B Results](#qwen-35-4b-results)
   - [Key Findings](#key-findings)
8. [Setup & Usage](#setup--usage)
9. [Prompt Engineering Details](#prompt-engineering-details)
10. [Changelog](#changelog)

---

## Project Overview

This repository contains the core testbench for the **Harvey Mudd Clinic Spring 2026** project, investigating how different **prompt-engineering strategies** and **model inference modes** affect reasoning accuracy on deductive logic puzzles.

The benchmark targets room-assignment problems where an LLM must find a bijective mapping of people to rooms and pets given a set of symbolic constraints. Five prompting strategies are evaluated across **multiple model configurations**, producing a controlled comparison across:

- Prompt structure (Zero-Shot → Chain → Tree → Verify → Few-Shot)
- Internal reasoning toggle (Thinking ON vs. Thinking OFF)
- Model scale (large API models vs. local 4B models)

---

## Repository Structure

```
Clinic_S26/
├── Model_testbench_colab.ipynb     # Main Colab notebook (v2 testbench)
├── benchmark_results2.json         # v1 run — 5,000 records, step-3.5-flash, thinking ON
├── benchmark_results.json          # Partial v1 run — 378 records, step-3.5-flash
├── benchmark_results3.json         # Partial v1 run — 155 records, ZERO_SHOT only
├── benchmark_1775711863.json       # Qwen 3.5 4B local model run
├── results_zero_shot.json          # v2 run — ZERO_SHOT,  1,000 records, thinking OFF
├── results_chain.json              # v2 run — CHAIN,      1,000 records, thinking OFF
├── results_tree.json               # v2 run — TREE,       1,000 records, thinking OFF
├── results_verify.json             # v2 run — VERIFY,     1,000 records, thinking OFF
├── results_few_shot.json           # v2 run — FEW_SHOT,   1,000 records, thinking OFF
└── README.md                       # This file
```

> **API key note:** The notebook ships with placeholder tokens. Fill them in Cell 1 before running.

---

## Dataset

| Property | Value |
|---|---|
| **Name** | `emunah/deductive_logical_reasoning-room_assignment` |
| **Source** | Hugging Face Hub |
| **Task type** | Multi-constraint deductive room assignment |
| **Puzzle sizes** | 7 – 19 rooms (N people + N pets) |
| **Sampling** | 1,000 examples, shuffled `seed=42` |
| **Answer column** | `completion` (fallback: `answer`) |

### Puzzle Format

Each puzzle provides N rooms, N people, and N pets with symbolic constraints:

| Symbol | Meaning |
|---|---|
| `X @ Y` | X and Y are in the **same** room |
| `X <> Y` | X and Y are in **different** rooms |
| `X: o o # o` | X is in the room at `#` position (e.g. Room 3) |
| `X: o x o o` | X is **not** in the room at `x` position (e.g. Room 2) |
| `=` / `!=` | Adjacency or specific exclusion constraint |

Required output:
```
Room 1: [Person], [Pet]
Room 2: [Person], [Pet]
...
```

---

## Reasoning Modes

| Mode | Description |
|---|---|
| **Zero-Shot** | Direct answer — no reasoning scaffold |
| **Chain-of-Thought** | 4-step: list constraints → deduce → explore → construct |
| **Tree-of-Thought** | Search-and-prune: branch → evaluate → select → conclude |
| **Self-Verification** | Solve, then explicitly re-verify against every constraint |
| **Few-Shot** | Prepend a worked 3-room example before the puzzle |

All modes share a Symbol Legend header and a format instruction footer.

---

## Testbench Architecture

```
Cell 1  Config (API keys, model, output path)
Cell 2  !pip install datasets requests tqdm
Cell 3  Google Drive mount
Cell 4  Core logic
        ├── ReasoningMode enum (5 modes)
        ├── get_prompt(question, mode)
        ├── extract_entities(prompt)          — parses people & pets from preamble
        ├── parse_assignments(text, ppl, pts) — extracts Room N: … patterns
        ├── grade(pred, gt, prompt)           — positional v2 grader
        ├── call_api(prompt)                  — NVIDIA NIM with retry + 429 backoff
        └── save_results(results, path)
Cell 5  run(limit=1000)
        └── Per-mode ThreadPoolExecutor(max_workers=5)
            └── saves results_{mode}.json per mode to Drive
```

**Key parameters:**
- `MAX_WORKERS = 5` — avoids 429 rate-limit errors
- `REQUEST_TIMEOUT = 120s`
- `temperature = 0.3`, `max_tokens = 1024`
- Retry: up to 3 attempts, backoff `(attempt+1) × 3s` on 429

---

## Grading Methodology

### v2 — Entity-Aware Positional Grader *(current — used for `results_*.json`)*

Scores each room slot independently for person correctness, pet correctness, and correct pairing:

```python
def extract_entities(prompt):
    # Parses people and pet lists from puzzle preamble via regex

def parse_assignments(text, people_set, pets_set):
    # Strips <think>/Reasoning blocks, finds "Room N: ..." patterns
    # Falls back to line-by-line matching

def grade(pred, gt, prompt):
    ppl, pts = extract_entities(prompt)
    gt_map   = parse_assignments(gt,   ppl, pts)
    res_map  = parse_assignments(pred, ppl, pts)
    # +1 correct person in correct room
    # +1 correct pet   in correct room
    # +1 correct person-pet pair
    return earned / total
```

**This grader is stricter** than v1: correct names in wrong rooms score near zero.

### v1 — Token-Overlap Recall *(used for `benchmark_results*.json`)*

```python
def grade(pred, gt):
    p, g = token_set(pred), token_set(gt)
    return len(p & g) / max(len(g), 1)
```

Measures what fraction of ground-truth tokens appear *anywhere* in the response. Does not check positional correctness — a response with right names in wrong rooms still scores high.

> ⚠️ **Direct comparison caveat:** v1 and v2 scores are **not comparable** — use each only within its own run.

---

## Experiment Results

### Run Overview

| File(s) | Model | Thinking | Grader | n / mode | Notes |
|---|---|---|---|---|---|
| `benchmark_results2.json` | step-3.5-flash | **ON** | v1 token-overlap | 1,000 | Main v1 run |
| `benchmark_results.json` | step-3.5-flash | **ON** | v1 token-overlap | ~80–92 | Partial run |
| `benchmark_results3.json` | step-3.5-flash | **ON** | v1 token-overlap | 155 | Zero-Shot only |
| `benchmark_1775711863.json` | Qwen 3.5 4B (local) | N/A | positional | 17,997 | Full dataset |
| `results_*.json` | step-3.5-flash | **OFF** | v2 positional | 1,000 | Main v2 run |

---

### v1 Results — step-3.5-flash, Thinking ON

> **Grader:** Token-overlap recall · **n=1,000 per mode** · Full run: `benchmark_results2.json`

#### Accuracy

| Mode | n | Mean | Median | Std | Perfect (1.0) | Partial | Zero |
|---|---|---|---|---|---|---|---|
| **Zero-Shot** | 977 | **0.9131** | **1.000** | 0.124 | **547** | 430 | 0 |
| Self-Verification | 979 | 0.8795 | 0.955 | 0.160 | 456 | 523 | 0 |
| Tree-of-Thought | 978 | 0.8658 | 0.919 | 0.170 | 425 | 553 | 0 |
| Chain-of-Thought | 979 | 0.8594 | 0.946 | 0.188 | 452 | 527 | 0 |
| Few-Shot | 969 | 0.8529 | 0.864 | 0.163 | 354 | 615 | 0 |
| **Overall** | 4,882 | **0.8742** | — | — | 2,234 | 2,648 | 0 |

#### Latency

| Mode | Mean (s) | Median (s) |
|---|---|---|
| **Zero-Shot** | **30.34** | **26.38** |
| Tree-of-Thought | 37.71 | 34.06 |
| Self-Verification | 37.81 | 34.00 |
| Chain-of-Thought | 38.37 | 34.40 |
| Few-Shot | 39.81 | 36.42 |

**Interpretation:** With thinking enabled, Zero-Shot dominates — the model's internal CoT (`reasoning_content`) already performs structured deduction. External scaffolding adds latency without improving accuracy. No zero scores were observed across 4,882 responses.

---

### v2 Results — step-3.5-flash, Thinking OFF

> **Grader:** v2 positional · **n=1,000 per mode** · Files: `results_*.json`  
> **Config:** `enable_thinking: False`, `thinking_budget: 0`, system msg `"detailed thinking off"`

#### Accuracy

| Mode | n | Mean | Median | Std | Partial | Zero | Errors |
|---|---|---|---|---|---|---|---|
| **Few-Shot** | 1,000 | **0.0609** | 0.037 | 0.073 | 719 | 277 | 4 |
| Chain-of-Thought | 1,000 | 0.0575 | 0.035 | 0.072 | 687 | 312 | 1 |
| Zero-Shot | 1,000 | 0.0536 | 0.037 | 0.066 | 695 | 304 | 1 |
| Self-Verification | 1,000 | 0.0521 | 0.028 | 0.072 | 640 | 360 | 0 |
| Tree-of-Thought | 1,000 | 0.0475 | 0.028 | 0.062 | 641 | 359 | 0 |

#### Latency

| Mode | Mean (s) | Median (s) |
|---|---|---|
| **Zero-Shot** | **11.49** | **9.57** |
| Tree-of-Thought | 11.80 | 10.47 |
| Chain-of-Thought | 11.81 | 10.20 |
| Self-Verification | 12.29 | 9.86 |
| **Few-Shot** | **15.97** | **12.73** |

**Interpretation:** Disabling thinking causes a dramatic accuracy collapse — mean scores drop from ~0.88 (v1) to ~0.054 (v2). No perfect scores were achieved across all 5,000 v2 responses. This confirms that `step-3.5-flash`'s accuracy on this task class is **almost entirely dependent on its internal reasoning chain**. Without it, the model produces structurally plausible but positionally incorrect room assignments.

Notably, the mode rankings **reverse**: Few-Shot leads marginally (0.061) while Tree-of-Thought falls last (0.048) — the opposite of v1. When the model cannot think through the problem, the worked example provides more signal than structured step instructions.

---

### Qwen 3.5 4B Results

> **File:** `benchmark_1775711863.json` · **Model:** Qwen 3.5 4B (local inference)  
> **Grader:** Positional · **Dataset:** Full dataset (~18,000 samples)

| Configuration | Accuracy | Points |
|---|---|---|
| Qwen 3.5 4B — Zero-Shot | **4.24%** | 1,525 / 35,994 |

**Interpretation:** The 4B local model achieves only 4.24% positional accuracy on Zero-Shot, establishing a clear baseline for small-model performance. This contrasts sharply with step-3.5-flash's 91.3% token-overlap (v1) and contextualises how much larger models' internal reasoning contributes to deductive task performance.

---

### Key Findings

1. **Internal reasoning is the dominant factor.** step-3.5-flash with thinking ON scores 0.913 (v1 token-overlap); with thinking OFF, it collapses to 0.054 (v2 positional). The model's accuracy is almost entirely driven by its chain-of-thought reasoning trace.

2. **Prompt scaffolding matters only without internal reasoning.** With thinking ON, Zero-Shot wins. With thinking OFF, Few-Shot leads — providing a concrete example compensates partially for the removed reasoning capacity.

3. **Zero-Shot is fastest regardless of thinking mode.** 30.3 s (thinking ON) → 11.5 s (thinking OFF), always the lowest latency mode.

4. **No perfect scores with thinking disabled.** Across 5,000 v2 responses, zero achieved a perfect positional score, vs. 2,234 / 4,882 in v1.

5. **Small models (Qwen 3.5 4B) cannot solve these puzzles.** 4.24% positional accuracy on the full dataset makes it unsuitable as a baseline reasoner for this task.

6. **The v2 grader reveals true difficulty.** The token-overlap grader inflated v1 scores — responses with correct names in wrong positions still scored high. The positional grader exposes that correct structure matters, not just vocabulary.

---

## Setup & Usage

### Prerequisites

- Google Colab with Drive mounted
- NVIDIA NIM API key (`nvapi-...`)
- Hugging Face token (read access to dataset)

### Running the Notebook

1. Open `Model_testbench_colab.ipynb` in Colab.
2. Fill in **Cell 1**:
   ```python
   os.environ['HF_TOKEN']       = "hf_..."
   os.environ['NVIDIA_API_KEY'] = "nvapi-..."
   MODEL_NAME       = "stepfun-ai/step-3.5-flash"
   DRIVE_OUTPUT_DIR = '/content/drive/MyDrive/ClinicSpring2026_Results'
   ```
3. Run all cells. Per-mode files are saved to Drive:
   ```
   results_zero_shot.json  results_chain.json  results_tree.json
   results_verify.json     results_few_shot.json
   ```

### Enabling / Disabling Thinking

In `call_api`, toggle the thinking budget:
```python
# Thinking ON  (v1 behaviour — high accuracy, slower)
# Remove or comment out:  "enable_thinking": False, "thinking_budget": 0

# Thinking OFF (v2 behaviour — fast, low accuracy)
payload["enable_thinking"] = False
payload["thinking_budget"] = 0
```

---

## Prompt Engineering Details

**Symbol Legend** (prepended to all modes):
```
- 'X @ Y'      means X and Y are in the same room.
- 'X <> Y'     means X and Y are in DIFFERENT rooms.
- 'X: o o # o' refers to the room at the '#' position.
- 'X: o x o o' means X is NOT in the room at the 'x' position.
- '=' and '!=' refer to adjacency or exclusion constraints.
```

**Format instruction** (appended to all modes):
```
Keep your reasoning concise. Clearly state the final arrangement using ONLY:
Room 1: [Person], [Pet]
Room 2: [Person], [Pet]
...
```

---

## Changelog

### v2 — May 2026
- **New entity-aware positional grader** replacing token-overlap recall
- **Thinking disabled** (`enable_thinking: False`, `thinking_budget: 0`, system: `"detailed thinking off"`)
- **Per-mode output files** (`results_{mode}.json`) saved independently to Drive
- **Compact single f-string prompts** replacing multi-line concatenation
- **Simplified few-shot example** — single inline string
- **Key finding:** thinking=OFF causes ~94% accuracy drop; prompt scaffolding rankings reverse

### v1 — April 2026
- Initial 5,000-record benchmark (`benchmark_results2.json`)
- Token-overlap recall grader
- Thinking enabled — model uses internal `reasoning_content` chain
- Zero-Shot wins all metrics; no zero scores observed
