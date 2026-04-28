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
6. [Experiment Results](#experiment-results)
   - [Summary Table](#summary-table)
   - [Per-Mode Analysis](#per-mode-analysis)
   - [Key Findings](#key-findings)
7. [Setup & Usage](#setup--usage)
8. [Prompt Engineering Details](#prompt-engineering-details)
9. [Grading Methodology](#grading-methodology)
10. [Acknowledgements](#acknowledgements)

---

## Project Overview

This repository contains the core testbench for the **Harvey Mudd Clinic Spring 2026** project, which investigates how different **prompt-engineering strategies** affect the reasoning accuracy of state-of-the-art Large Language Models (LLMs).

The benchmark targets **deductive logical reasoning** — specifically room-assignment puzzles where an LLM must infer a valid bijective mapping of people to rooms and pets given a set of symbolic constraints. Five distinct prompting strategies are evaluated head-to-head on **1,000 randomly sampled puzzles** (5,000 total API calls across all modes).

**Model under test:** `stepfun-ai/step-3.5-flash` via the NVIDIA NIM API.

**Core research questions:**
- Does adding structured reasoning steps (CoT, ToT, Self-Verification) improve accuracy over a naive Zero-Shot baseline?
- How do different reasoning frameworks trade off accuracy against latency?
- Does providing in-context examples (Few-Shot) help or hurt on complex symbolic puzzles?

---

## Repository Structure

```
Clinic_S26/
├── Model_testbench_colab.ipynb   # Main Google Colab notebook (self-contained testbench)
├── benchmark_results2.json       # Full results — 5,000 records across 5 reasoning modes
└── README.md                     # This file
```

> **Note:** The notebook is designed to run end-to-end in **Google Colab** with Google Drive mounted for output persistence. All dependencies are installed inline via `!pip`.

---

## Dataset

| Property | Value |
|---|---|
| **Name** | `emunah/deductive_logical_reasoning-room_assignment` |
| **Source** | Hugging Face Hub |
| **Task type** | Multi-constraint deductive room assignment |
| **Puzzle sizes** | 7 – 19 rooms (people + pets) |
| **Sampling** | 1,000 examples, shuffled with `seed=42` |
| **Answer column** | `completion` (or `answer` as fallback) |

### Puzzle Format

Each puzzle presents:
- **N rooms** assigned to N people and N pets (a bijective mapping).
- A list of symbolic constraints using a compact notation:

| Symbol | Meaning |
|---|---|
| `X @ Y` | X and Y are in the **same** room |
| `X <> Y` | X and Y are in **different** rooms |
| `X: o o # o` | X is in the room matching `#` position |
| `X: o x o o` | X is **not** in the room matching `x` position |
| `=` | Adjacency or same-room equivalence |
| `!=` | Specific exclusion constraint |

The model must output the final assignment in the strict format:
```
Room 1: [Person], [Pet]
Room 2: [Person], [Pet]
...
```

---

## Reasoning Modes

Five reasoning strategies are benchmarked. All modes prepend the **Symbol Legend** and append a common **format instruction** to the raw puzzle prompt.

| Mode | Enum | Description |
|---|---|---|
| **Zero-Shot** | `ZERO_SHOT` | Asks for the answer directly with no reasoning scaffold. |
| **Chain-of-Thought** | `CHAIN` | 4-step structured reasoning: list constraints → deduce implications → eliminate contradictions → construct solution. |
| **Tree-of-Thought** | `TREE` | Explicit search-and-prune: create branches → evaluate each → select valid branch → conclude. |
| **Self-Verification** | `VERIFY` | Solve first, then verify solution against every constraint before committing. |
| **Few-Shot** | `FEW_SHOT` | Prepends a worked 3-room example before posing the actual puzzle. |

---

## Testbench Architecture

```
┌─────────────────────────────────────────────────────┐
│                Model_testbench_colab.ipynb           │
│                                                     │
│  Config ──► get_prompt(question, mode)              │
│                     │                               │
│                     ▼                               │
│             call_api(prompt)                        │
│          (max_retries=3, 429 backoff)               │
│                     │                               │
│                     ▼                               │
│  ThreadPoolExecutor(max_workers=5)                  │
│  ├── Concurrent requests per mode                   │
│  ├── Checkpoint save every 20 results               │
│  └── Graceful retry on rate limits                  │
│                     │                               │
│                     ▼                               │
│             grade(response, ground_truth)           │
│          (token-overlap F1 scoring)                 │
│                     │                               │
│                     ▼                               │
│          benchmark_results.json (Google Drive)      │
└─────────────────────────────────────────────────────┘
```

**Key implementation details:**
- **Parallelism:** `ThreadPoolExecutor` with `MAX_WORKERS=5` (tuned to avoid NVIDIA API 429 rate limits).
- **Retry logic:** Up to 3 retries with exponential backoff `(attempt+1) × 3s` on HTTP 429.
- **Reasoning content:** The API response captures both `content` and `reasoning_content`/`reasoning` fields, allowing visibility into the model's internal thinking trace.
- **Checkpointing:** Results are saved to disk every 20 completions to survive partial runs.

---

## Experiment Results

> **Full results file:** `benchmark_results2.json`
> **Total records:** 5,000 (1,000 puzzles × 5 modes)
> **Model:** `stepfun-ai/step-3.5-flash` · **Temperature:** 0.3 · **Max tokens:** 1,024

### Summary Table

| Reasoning Mode | n (scored) | Errors | Mean Score | Median Score | Std Dev | Perfect (1.0) | Partial | Zero |
|---|---|---|---|---|---|---|---|---|
| **Zero-Shot** | 977 | 23 | **0.9131** | **1.0000** | 0.1238 | **547** | 430 | 0 |
| **Self-Verification** | 979 | 21 | 0.8795 | 0.9545 | 0.1595 | 456 | 523 | 0 |
| **Tree-of-Thought** | 978 | 22 | 0.8658 | 0.9189 | 0.1704 | 425 | 553 | 0 |
| **Chain-of-Thought** | 979 | 21 | 0.8594 | 0.9459 | 0.1884 | 452 | 527 | 0 |
| **Few-Shot** | 969 | 31 | 0.8529 | 0.8636 | 0.1625 | 354 | 615 | 0 |
| **Overall** | 4,882 | 118 | **0.8742** | — | — | 2,234 | 2,648 | 0 |

### Latency Breakdown

| Reasoning Mode | Mean Latency (s) | Median Latency (s) |
|---|---|---|
| **Zero-Shot** | **30.34** | **26.38** |
| Tree-of-Thought | 37.71 | 34.06 |
| Self-Verification | 37.81 | 34.00 |
| Chain-of-Thought | 38.37 | 34.40 |
| Few-Shot | 39.81 | 36.42 |

---

### Per-Mode Analysis

#### Zero-Shot (Mean: 0.9131 | Median: 1.000)

The **highest accuracy** of all five modes, and the **fastest** (mean 30.3s, median 26.4s). The model — which appears to possess strong internal chain-of-thought capabilities via its `reasoning_content` field — naturally decomposes the problem without explicit scaffolding from the prompt. The high median of **1.000** means the majority of responses are perfect. 

This is the strongest result and strongly suggests the model's internal reasoning is comprehensive enough that external structured prompts may *constrain* rather than *help*.

#### Self-Verification (Mean: 0.8795 | Median: 0.9545)

Second-best accuracy at 0.8795 mean. The verification pass — checking the solution against all constraints — occasionally catches and corrects errors. However, it adds ~7.5s of mean latency vs. Zero-Shot. The mode still achieves 456 perfect scores, the second-highest count.

#### Tree-of-Thought (Mean: 0.8658 | Median: 0.9189)

The explicit search-and-prune strategy performs slightly below Self-Verification. The branching prompt structure can introduce verbosity that pushes the model toward truncated outputs within the 1,024-token limit, which may account for the reduced perfect-score count (425 vs. 547 for Zero-Shot).

#### Chain-of-Thought (Mean: 0.8594 | Median: 0.9459)

CoT places fourth. Despite having the second-highest median (0.9459), the mean is pulled down by more variance (`std=0.1884`, the highest of all modes). This suggests CoT works well on mid-complexity puzzles but occasionally over-complicates simple ones.

#### Few-Shot (Mean: 0.8529 | Median: 0.8636)

**Lowest accuracy** and the **most errors** (31). The prepended 3-room worked example may mislead the model on larger puzzles (up to 19 rooms), creating format anchoring bias. The lowest median (0.8636) and lowest perfect-score count (354) confirm that the example provides inconsistent signal across puzzle sizes.

---

### Key Findings

1. **Zero-Shot wins across all metrics** — both highest accuracy (0.9131) and lowest latency (30.3s). This is counter-intuitive given conventional wisdom around structured prompting.

2. **Reasoning-capable models may not benefit from explicit CoT prompts.** The `step-3.5-flash` model returns a `reasoning_content` field, indicating it already performs internal chain-of-thought. External scaffolding may thus *interfere* with an optimized internal process.

3. **No zero scores were observed** across all 4,882 graded responses, reflecting a high floor for the model's raw capability on this task class.

4. **Few-Shot hurts on variable-size puzzles.** A single 3-room example proved an inconsistent prior for puzzles ranging from 7–19 rooms.

5. **Error rates are uniform (~2%)** across modes, suggesting errors are primarily network/rate-limit transients rather than model failures.

6. **All modes produced substantial partial credit** (scores between 0 and 1), indicating the model typically identifies *most* room assignments correctly even when it does not achieve a perfect score.

---

## Setup & Usage

### Prerequisites

- A **Google Colab** account with Google Drive mounted.
- A valid **NVIDIA NIM API key** (`nvapi-...`).
- A **Hugging Face token** with access to the dataset.

### Running the Notebook

1. Open `Model_testbench_colab.ipynb` in Google Colab.
2. In **Cell 1**, update the following constants:
   ```python
   os.environ['HF_TOKEN'] = "YOUR_HF_TOKEN"
   os.environ['NVIDIA_API_KEY'] = "YOUR_NVIDIA_API_KEY"
   MODEL_NAME = "stepfun-ai/step-3.5-flash"   # or swap model here
   DRIVE_OUTPUT_DIR = '/content/drive/MyDrive/ClinicSpring2026_Results'
   ```
3. Run all cells in order (Runtime → Run all).
4. Results are saved to `benchmark_results.json` in your Drive folder.

### Changing the Sample Size

```python
run(limit=1000)  # Change 1000 to any number ≤ len(dataset)
```

### Swapping the Model

Update `MODEL_NAME` and `API_URL` in Cell 1:
```python
API_URL   = "https://integrate.api.nvidia.com/v1/chat/completions"
MODEL_NAME = "meta/llama-3.1-70b-instruct"  # example swap
```

---

## Prompt Engineering Details

All prompts share a common **Symbol Legend** header that explains the constraint notation to the model, and a common **format instruction** suffix that specifies the exact output format required for grading.

```
Symbol Legend:
- 'X @ Y' means X and Y are in the same room.
- 'X <> Y' means X and Y are in DIFFERENT rooms.
- 'X: o o # o' refers to the room matching the '#' position (e.g., Room 3).
- 'X: o x o o' means X is NOT in the room matching the 'x' position (e.g., Room 2).
- '=' and '!=' often refer to adjacency or specific exclusion constraints.
```

**Format instruction** (appended to all modes):
```
Keep your reasoning concise and focused. Avoid unnecessary repetition.
Clearly state the final arrangement at the end using ONLY this format:
Room 1: [Person], [Pet]
Room 2: [Person], [Pet]
...
```

---

## Grading Methodology

Scoring uses **token-overlap recall** against the ground truth:

```python
def extract_words(text):
    return set(re.findall(r"\b\w+\b", str(text).lower()))

def grade(pred, gt):
    p, g = extract_words(pred), extract_words(gt)
    return len(p & g) / max(len(g), 1)
```

- A score of **1.0** means every token in the ground truth appears in the model's response (perfect or superset).
- Scores below 1.0 reflect missing person/pet/room tokens.
- **No zero scores** were observed, consistent with the model always producing well-formed structured output.

> **Limitation:** This metric counts token overlap, not positional correctness. A model that assigns the right names but to wrong rooms may still score high. A stricter positional grader would likely lower all mode scores uniformly.

---

## Acknowledgements

- **Harvey Mudd College Clinic Program** for sponsoring this research.
- **NVIDIA** for API access via the NIM platform.
- **Hugging Face** for hosting the `emunah/deductive_logical_reasoning-room_assignment` dataset.
- **StepFun AI** for the `step-3.5-flash` model.
