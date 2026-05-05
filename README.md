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
   - [Summary Table](#summary-table)
   - [Per-Mode Analysis](#per-mode-analysis)
   - [Key Findings](#key-findings)
8. [Setup & Usage](#setup--usage)
9. [Prompt Engineering Details](#prompt-engineering-details)
10. [Changelog](#changelog)

---

## Project Overview

This repository contains the core testbench for the **Harvey Mudd Clinic Spring 2026** project, which investigates how different **prompt-engineering strategies** affect the reasoning accuracy of state-of-the-art Large Language Models (LLMs).

The benchmark targets **deductive logical reasoning** — specifically room-assignment puzzles where an LLM must infer a valid bijective mapping of people to rooms and pets given a set of symbolic constraints. Five distinct prompting strategies are evaluated head-to-head on **1,000 randomly sampled puzzles** (5,000 total API calls across all modes).

**Model under test:** `stepfun-ai/step-3.5-flash` via the NVIDIA NIM API  
**Thinking budget:** Disabled (`enable_thinking: False`, `thinking_budget: 0`) — pure fast-inference mode.

**Core research questions:**
- Does adding structured reasoning steps (CoT, ToT, Self-Verification) improve accuracy over a naive Zero-Shot baseline?
- How do different reasoning frameworks trade off accuracy against latency?
- Does providing in-context examples (Few-Shot) help or hurt on complex symbolic puzzles?

---

## Repository Structure

```
Clinic_S26/
├── Model_testbench_colab.ipynb   # Main Google Colab notebook (self-contained testbench)
├── benchmark_results2.json       # v1 results — 5,000 records, token-overlap grader
└── README.md                     # This file
```

> **Note on API keys:** The notebook ships with placeholder tokens (`YOUR_HF_TOKEN_HERE`, `YOUR_NVIDIA_API_KEY_HERE`). Fill these in Cell 1 before running in Colab.

> **Note on results files:** The current testbench saves **per-mode** result files (`results_zero_shot.json`, `results_chain.json`, etc.) to Google Drive. Download and add them here when available.

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

Each puzzle presents **N rooms** assigned to N people and N pets (a bijective mapping), with a list of symbolic constraints:

| Symbol | Meaning |
|---|---|
| `X @ Y` | X and Y are in the **same** room |
| `X <> Y` | X and Y are in **different** rooms |
| `X: o o # o` | X is in the room at the `#` position (e.g. Room 3) |
| `X: o x o o` | X is **not** in the room at the `x` position (e.g. Room 2) |
| `=` / `!=` | Adjacency or specific exclusion constraint |

Required output format:
```
Room 1: [Person], [Pet]
Room 2: [Person], [Pet]
...
```

---

## Reasoning Modes

Five reasoning strategies are benchmarked. All modes prepend the **Symbol Legend** and append a common **format instruction**.

| Mode | Enum | Prompt Strategy |
|---|---|---|
| **Zero-Shot** | `ZERO_SHOT` | Direct answer request — no reasoning scaffold. |
| **Chain-of-Thought** | `CHAIN` | 4-step: list constraints → deduce implications → explore possibilities → construct solution. |
| **Tree-of-Thought** | `TREE` | Search-and-prune: tree search & branching → candidate evaluation → branch selection → final conclusion. |
| **Self-Verification** | `VERIFY` | Solve first, then explicitly verify solution against every constraint. |
| **Few-Shot** | `FEW_SHOT` | Prepends a worked 3-room example before the actual puzzle. |

---

## Testbench Architecture

```
┌──────────────────────────────────────────────────────┐
│              Model_testbench_colab.ipynb              │
│                                                      │
│  Cell 1: Config (API keys, model, output path)       │
│  Cell 2: !pip install datasets requests tqdm         │
│  Cell 3: Google Drive mount                          │
│  Cell 4: Core logic                                  │
│    ├── ReasoningMode enum                            │
│    ├── get_prompt(question, mode)                    │
│    ├── extract_entities(prompt)  ─── NEW             │
│    ├── parse_assignments(text, people, pets) ─ NEW   │
│    ├── grade(pred, gt, prompt)   ─── NEW (v2 grader) │
│    ├── call_api(prompt)          ─── thinking=OFF    │
│    └── save_results(results, path)                   │
│  Cell 5: run(limit=1000)                             │
│    └── Per-mode ThreadPoolExecutor(max_workers=5)    │
│        └── Saves results_{mode}.json per mode        │
└──────────────────────────────────────────────────────┘
```

**Key implementation details:**
- **Parallelism:** `ThreadPoolExecutor` with `MAX_WORKERS=5` (tuned to avoid NVIDIA API 429 rate limits)
- **Thinking disabled:** System message `"detailed thinking off"` + API params `enable_thinking: False, thinking_budget: 0` — forces the model into fast non-reasoning mode
- **Retry logic:** Up to 3 retries with backoff `(attempt+1) × 3s` on HTTP 429
- **Per-mode checkpointing:** Each mode saves its own `results_{mode}.json` to Drive on completion

---

## Grading Methodology

### v2 — Entity-Aware Positional Grader *(current)*

The updated grader is **position-aware** and scores each room slot individually:

```python
def extract_entities(prompt):
    # Parses people and pets lists directly from the puzzle preamble
    ...

def parse_assignments(text, people_set, pets_set):
    # Extracts "Room N: ..." patterns, strips <think> / Reasoning blocks
    # Falls back to line-by-line matching if no Room headers found
    ...

def grade(pred, gt, prompt):
    ppl, pts = extract_entities(prompt)
    gt_map  = parse_assignments(gt,   ppl, pts)
    res_map = parse_assignments(pred, ppl, pts)
    # Scores: +1 correct person in correct room
    #         +1 correct pet in correct room
    #         +1 correct person-pet pair
    return earned / total
```

**Score interpretation:**
- `1.0` — every room has the correct person, correct pet, and correct pairing
- Partial credit awarded per slot and per pair
- This is **stricter** than v1: a response with right names in wrong rooms now scores lower

### v1 — Token-Overlap Recall *(used for `benchmark_results2.json`)*

```python
def grade(pred, gt):
    p, g = set(re.findall(r"\b\w+\b", pred.lower())), set(re.findall(r"\b\w+\b", gt.lower()))
    return len(p & g) / max(len(g), 1)
```

- Measures what fraction of ground-truth tokens appear anywhere in the response
- **Does not check positional correctness** — right names in wrong rooms still score high
- Results in `benchmark_results2.json` used this grader

---

## Experiment Results

### v1 Results — `benchmark_results2.json`

> **Grader:** Token-overlap recall · **Model:** `stepfun-ai/step-3.5-flash` · **n=1,000 per mode**  
> **Note:** The v1 grader is lenient — scores reflect token presence, not positional accuracy.

#### Summary Table

| Reasoning Mode | n (scored) | Errors | Mean Score | Median | Std Dev | Perfect (1.0) | Partial |
|---|---|---|---|---|---|---|---|
| **Zero-Shot** | 977 | 23 | **0.9131** | **1.000** | 0.1238 | **547** | 430 |
| **Self-Verification** | 979 | 21 | 0.8795 | 0.9545 | 0.1595 | 456 | 523 |
| **Tree-of-Thought** | 978 | 22 | 0.8658 | 0.9189 | 0.1704 | 425 | 553 |
| **Chain-of-Thought** | 979 | 21 | 0.8594 | 0.9459 | 0.1884 | 452 | 527 |
| **Few-Shot** | 969 | 31 | 0.8529 | 0.8636 | 0.1625 | 354 | 615 |
| **Overall** | 4,882 | 118 | **0.8742** | — | — | 2,234 | 2,648 |

#### Latency

| Mode | Mean Latency (s) | Median Latency (s) |
|---|---|---|
| **Zero-Shot** | **30.34** | **26.38** |
| Tree-of-Thought | 37.71 | 34.06 |
| Self-Verification | 37.81 | 34.00 |
| Chain-of-Thought | 38.37 | 34.40 |
| Few-Shot | 39.81 | 36.42 |

---

### Per-Mode Analysis

#### Zero-Shot (Mean: 0.9131 | Median: 1.000)

Highest accuracy of all five modes and the fastest (mean 30.3 s). The model's internal chain-of-thought (visible via `reasoning_content`) naturally decomposes the problem without external scaffolding. The median of **1.000** means the majority of responses are perfect under the v1 grader.

#### Self-Verification (Mean: 0.8795 | Median: 0.9545)

Second-best. The verification pass occasionally catches and corrects errors but adds ~7.5 s of mean latency vs. Zero-Shot.

#### Tree-of-Thought (Mean: 0.8658 | Median: 0.9189)

Explicit search-and-prune performs slightly below Self-Verification. The branching prompt can produce verbose outputs that risk hitting the 1,024-token limit, reducing perfect-score count.

#### Chain-of-Thought (Mean: 0.8594 | Median: 0.9459)

Fourth place. High median but the largest variance (`std=0.1884`), suggesting CoT works well on mid-complexity puzzles but occasionally over-complicates simpler ones.

#### Few-Shot (Mean: 0.8529 | Median: 0.8636)

Lowest accuracy and most errors (31). The single 3-room worked example creates format-anchoring bias on larger puzzles (7–19 rooms). Lowest perfect-score count (354).

---

### Key Findings

1. **Zero-Shot wins across all metrics** — both highest accuracy (0.9131) and lowest latency (30.3 s).
2. **Reasoning-capable models may not benefit from explicit CoT scaffolding.** The model's own internal reasoning (`reasoning_content` / `thinking`) already performs structured deduction; external scaffolding can interfere.
3. **No zero scores observed** across 4,882 graded responses — the model reliably produces well-formed structured output.
4. **Few-Shot hurts on variable-size puzzles.** A single 3-room example is an inconsistent prior for 7–19 room problems.
5. **Error rates are uniform (~2%)** across modes — failures are network/rate-limit transients, not model failures.

> ⚠️ **v2 grader caveat:** Because the v2 grader checks positional correctness (not just token overlap), new results run with the updated notebook are expected to show lower absolute scores while better reflecting true accuracy. Rankings between modes may shift.

---

## Setup & Usage

### Prerequisites

- A **Google Colab** account with Google Drive mounted
- A valid **NVIDIA NIM API key** (`nvapi-...`)
- A **Hugging Face token** with read access to the dataset

### Running the Notebook

1. Open `Model_testbench_colab.ipynb` in Google Colab.
2. In **Cell 1**, fill in your credentials:
   ```python
   os.environ['HF_TOKEN']       = "hf_..."
   os.environ['NVIDIA_API_KEY'] = "nvapi-..."
   MODEL_NAME       = "stepfun-ai/step-3.5-flash"
   DRIVE_OUTPUT_DIR = '/content/drive/MyDrive/ClinicSpring2026_Results'
   ```
3. Run all cells (**Runtime → Run all**).
4. Per-mode result files are saved to your Drive folder:
   ```
   results_zero_shot.json
   results_chain.json
   results_tree.json
   results_verify.json
   results_few_shot.json
   ```

### Changing Sample Size or Model

```python
run(limit=500)   # change sample count

# swap model in Cell 1:
MODEL_NAME = "meta/llama-3.1-70b-instruct"
API_URL    = "https://integrate.api.nvidia.com/v1/chat/completions"
```

---

## Prompt Engineering Details

All modes share a **Symbol Legend** header and a **format instruction** footer.

```
Symbol Legend:
- 'X @ Y' means X and Y are in the same room.
- 'X <> Y' means X and Y are in DIFFERENT rooms.
- 'X: o o # o' refers to the room matching the '#' position (e.g., Room 3).
- 'X: o x o o' means X is NOT in the room matching the 'x' position (e.g., Room 2).
- '=' and '!=' often refer to adjacency or specific exclusion constraints.
```

**Format instruction** (all modes):
```
Keep your reasoning concise and focused. Avoid unnecessary repetition.
Clearly state the final arrangement at the end using ONLY this format:
Room 1: [Person], [Pet]
Room 2: [Person], [Pet]
...
```

---

## Changelog

### v2 — May 2026
- **New entity-aware positional grader** (`extract_entities` + `parse_assignments`) replacing token-overlap recall
- **Thinking disabled** on API calls: system message `"detailed thinking off"` + `enable_thinking: False, thinking_budget: 0` — tests model in pure fast-inference mode
- **Compact prompt format**: prompts refactored to single f-strings
- **Per-mode output files**: each mode now saves `results_{mode}.json` independently to Drive
- **Simplified few-shot example**: reduced from multi-line block to single inline string

### v1 — April 2026
- Initial release with token-overlap recall grader
- 5,000-record benchmark (`benchmark_results2.json`)
- Multi-line prompt construction with explicit 4-step CoT, ToT, Verify scaffolding
- Reasoning content captured from `reasoning_content` / `reasoning` API fields
