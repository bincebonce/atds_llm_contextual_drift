# Hallucination Under Contextual and Semantic Drift in Large Language Models

**Authors:** Ryan Han, Vincent Qiu, Terence Xu
**Course:** DS-UA 301 — Advanced Topics in Data Science, NYU


This repository contains the code, data, and results for our 12-week class project investigating how progressive context perturbations affect hallucination behavior in instruction-tuned LLMs on the TruthfulQA benchmark.

**Headline finding.** Across four models (GPT-3.5 Turbo, Llama-3.2-1B, Qwen-2.5-1.5B, Falcon3-1B) and two qualitatively different drift paradigms — totalling **4,800 model queries** — we find no statistically significant hallucination increase under context drift. We interpret this as evidence that instruction-tuning hardens models against shallow context perturbations and that adversarial-misconception benchmarks like TruthfulQA are dominated by pretraining biases rather than prompt-level signal.

---

## Motivation

Hallucinations in large language models present a significant challenge to their reliable deployment in high-stakes domains where incorrect or fabricated information can lead to serious consequences. They arise when models generate responses that are plausible but factually incorrect, unsupported by the provided context, or inconsistent with established knowledge. Prior research has shown they emerge from multiple sources, including training-data biases, misaligned reasoning, and contextual perturbations during inference.

Existing work has primarily focused on developing **diagnostic** frameworks to detect hallucination after it occurs, and often stops short of robust cross-model, cross-paradigm evaluation. Most "LLMs hallucinate under X" claims are based on a single model, a single drift paradigm, and scoring that does not survive close inspection. In this project we set out to characterize the relationship between contextual drift and hallucination with **multi-model, multi-paradigm, paired-statistical evaluation on an adversarial-truthfulness benchmark using a semantic-similarity scorer**. The negative result we report is, we believe, the more honest answer.

---

## Related Work

The two papers our experimental design draws from are listed below. Both inspired one of our two experiments.

### Natural Context Drift Undermines the Natural Language Understanding of Large Language Models (Wu et al., 2025)

Wu, Y., Schlegel, V., & Batista-Navarro, R. (2025). *Natural context drift undermines the natural language understanding of large language models.* arXiv:2509.01093. https://arxiv.org/abs/2509.01093

**Methods.** Wu et al. propose a framework to study how natural contextual drift affects the reading-comprehension ability of LLMs. They leverage Wikipedia revision histories to collect human-edited versions of passages used in existing QA benchmarks, compare edited and original versions via semantic similarity, and evaluate model performance across similarity ranges.

**Why we used it.** Their use of Wikipedia revisions as a natural-drift signal directly inspired our Experiment 2.

**Limitation we wanted to extend.** Their evaluation is on NLU reading-comprehension tasks; it is not run on adversarial-truthfulness benchmarks like TruthfulQA where the failure mode is reproducing misconceptions rather than mis-comprehending text.

### Shadows in the Attention: Contextual Perturbation and Representation Drift in the Dynamics of Hallucination in LLMs (Wei et al., 2025)

Wei, Z., Wang, S., Rong, X., Liu, X., & Li, H. (2025). *Shadows in the attention: Contextual perturbation and representation drift in the dynamics of hallucination in LLMs.* arXiv:2505.16894. https://arxiv.org/abs/2505.16894

**Methods.** Wei et al. investigate how contextual perturbations influence hallucination using TruthfulQA. They use a multi-round context injection framework where each question undergoes incremental context augmentation along two tracks (relevant-but-misleading vs. irrelevant-distraction) and track hallucination alongside hidden-state and attention dynamics.

**Why we used it.** Their multi-round context-injection design directly inspired our Experiment 1, which we generalised to six rounds spanning a relevance-to-distraction gradient.

**Limitation we wanted to extend.** Single model, mechanism-focused, and primarily diagnostic. We extended to four models, added a second drift paradigm, and added paired statistical testing.

---

## Methodology

We run two experiments on the same 100-question TruthfulQA subsample. Both experiments share the same scorer, the same statistical tests, and the same four models, so within-model and cross-paradigm comparisons are paired at the question level.

### Experiment 1 — Prompt-template context injection

For each question, we evaluate six conditions (Rounds 0–5):
- **R0 (baseline):** no context.
- **R1 — relevant grounding:** a factual framing referencing the question's TruthfulQA category.
- **R2 — ambiguous context:** acknowledges both evidence-based and popular beliefs.
- **R3 — tangential context:** cross-domain framing.
- **R4 — misconception-reinforcing:** explicitly endorses "the commonly accepted answer."
- **R5 — irrelevant distraction:** a fabricated passage about ambient temperature and analog clocks.

Same prompts for all four models so cross-model differences are attributable to the model.

### Experiment 2 — Temporal Wikipedia revision drift

For each question we (i) query Wikipedia for the most relevant article, (ii) fetch up to 50 historical revisions via the MediaWiki API, (iii) clean wikitext, (iv) sample five revisions at evenly-spaced chronological indices (R1 = newest, R5 = oldest). Each revision is injected as context. Revisions are fetched once and persisted to a JSON cache so subsequent model runs reuse the same contexts.

### Truthfulness scoring

We use sentence-embedding cosine similarity (`all-MiniLM-L6-v2`) against TruthfulQA's correct and incorrect answer pools:
- s⁺ = max cos(response, correct_i)
- s⁻ = max cos(response, incorrect_j)
- **correct** if s⁺ ≥ τ and s⁺ > s⁻
- **hallucinated** if s⁻ ≥ τ and s⁻ > s⁺
- threshold τ = 0.55 (calibrated on a 20-response sample)
- continuous truth margin = s⁺ − s⁻

This replaces the substring-matching scorer used in our Milestone 2 pipeline, which produced a near-floor accuracy artefact (8–11%). The semantic scorer lifts baseline accuracy to 41–55% — confirming the floor was a measurement artefact, not a model property.

### Statistical analysis

Because the same 100 questions are evaluated under every round, each round-vs.-baseline comparison is a paired binary outcome. We use **McNemar's exact binomial test** on the discordant pairs and report **bootstrap 95% confidence intervals** (2,000 resamples) on per-round accuracy and hallucination rate.

---

## Models Evaluated

| Model | Size | Backend |
|---|---|---|
| GPT-3.5 Turbo | ~175B | OpenAI Chat Completions API |
| Llama-3.2-1B-Instruct | 1B | Local, Colab T4 / fp16 |
| Qwen-2.5-1.5B-Instruct | 1.5B | Local, Colab T4 / fp16 |
| Falcon3-1B-Instruct | 1B | Local, Colab T4 / fp16 |

All models are queried at `temperature=0` with a max of 200 generated tokens. We initially planned to also run 7–8B open-source models (Llama-3-8B, Qwen-2.5-7B, Falcon3-7B) via the HuggingFace Inference Router, but free-tier quota limits and Colab T4 memory both made full 600-row runs infeasible. We treat the 1–2B scale as our open-source comparison tier.

---

## Datasets

**TruthfulQA** (Lin et al., 2022) — 817 question–answer pairs across 38 categories, specifically designed to elicit incorrect but plausible responses. We sample 100 questions across 33 categories with `seed=42` and reuse this subset across every model and every experiment.

**Wikipedia revision histories** — fetched on-the-fly via the MediaWiki `action=query&prop=revisions` API. For each TruthfulQA question, we identify the most relevant article via the Wikipedia search API, then fetch up to 50 revisions. Revisions shorter than 300 characters are discarded; the remainder are cleaned of wikitext markup and capped at 3,500 characters per revision before injection.

---

## Repository Layout

```
ATDS_Milestone_2.ipynb              # Experiment 1 — prompt-template injection (final version)
ATDS_Semantic_Drift.ipynb           # Experiment 2 — Wikipedia revision drift

results_<model>.csv                 # Per-row Experiment 1 results
results_temporal_<model>.csv        # Per-row Experiment 2 results
summary_<model>.csv                 # Round-level aggregates (Experiment 1)
ci_<model>.csv                      # Bootstrap 95% CIs (Experiment 1)
mcnemar_acc_<model>.csv             # McNemar tests, accuracy (Experiment 1)
mcnemar_halluc_<model>.csv          # McNemar tests, hallucination (Experiment 1)
category_<model>.csv                # Per-category breakdown (Experiment 1)

drift_analysis_<model>.png          # Accuracy & hallucination with 95% CIs
truth_margin_<model>.png            # Truth-margin violin plots
category_heatmap_<model>.png        # Per-category accuracy × round heatmap

<model> tags: gpt35, llama32_1b, qwen25_1_5b, falcon3_1b
```

---

## Key Results

### Experiment 1 — Prompt-template injection

| Model | Baseline acc | R1–R5 acc range | Baseline halluc | R1–R5 halluc range |
|---|---|---|---|---|
| GPT-3.5 Turbo  | 0.55 | 0.49–0.56 | 0.34 | 0.36–0.43 |
| Llama-3.2-1B   | 0.47 | 0.43–0.53 | 0.43 | 0.36–0.44 |
| Qwen-2.5-1.5B  | 0.41 | 0.44–0.49 | 0.51 | 0.43–0.47 |
| Falcon3-1B     | 0.41 | 0.42–0.50 | 0.49 | 0.41–0.52 |

Of 40 paired McNemar tests (4 models × 5 rounds × 2 metrics), only 2 reach α=0.05 — both showing accuracy *improving* relative to baseline under injection.

### Experiment 2 — Wikipedia revision drift

| Model | Baseline acc | R1–R5 acc range | Notes |
|---|---|---|---|
| GPT-3.5 Turbo  | 0.55 | 0.52–0.59 | Best round = R5 (oldest revision) |
| Llama-3.2-1B   | 0.57 | 0.46–0.50 | Flat 7–11 pt drop, not monotone with age |
| Qwen-2.5-1.5B  | 0.50 | 0.46–0.51 | Indistinguishable from baseline |
| Falcon3-1B     | 0.46 | 0.39–0.43 | Mild decline paired with *lower* hallucination |

Zero of 20 McNemar accuracy tests reach α=0.05.

---

## Why Drift Doesn't Move the Needle

Three reinforcing explanations, discussed in detail in the final report:

1. **Pretraining beats prompts.** TruthfulQA targets memorised misconceptions. Errors live in the weights, and prompt-level context can't edit weights.
2. **Instruction-tuned models hedge.** Under noisy context, RLHF-trained models abstain rather than confabulate, landing in the "ambiguous" bucket — neither correct nor hallucinated.
3. **The drift signal is too soft.** Even the oldest Wikipedia revision of most TruthfulQA articles is factually correct; TruthfulQA contains very few questions whose true answer has changed over time.

---

## Limitations & Future Work

**Limitations.** (1) n=100 gives ±8–10 pt CIs — limited power for small effects. (2) Open-source models capped at 1–2B due to quota/memory constraints. (3) Round number is a noisy proxy for true semantic distance. (4) TruthfulQA is the regime *least* sensitive to prompt-level drift.

**Future work.** (i) Full TruthfulQA-817 on 7–8B open-source models, ideally on NYU HPC. (ii) Continuous semantic-distance covariate fit with a GLMM. (iii) FreshQA or similar recency-sensitive benchmarks where temporal drift should actually bite. (iv) Mechanistic layer for the open-source models — attention to injected context, per-token entropy.

---

## How to Reproduce

1. Open `ATDS_Milestone_2.ipynb` in Colab (or Jupyter with a CUDA GPU).
2. Set `ACTIVE_MODEL` in the registry cell to the model you want to run.
3. Set `OPENAI_API_KEY` and `HF_TOKEN` in Colab Secrets (or environment).
4. Run all cells. Outputs land in `results/results_<tag>.csv` and friends.
5. For Experiment 2, open `ATDS_Semantic_Drift.ipynb`, set `MODEL_ID`, and run. Wikipedia contexts are cached in `wiki_contexts_cache.json` so subsequent model runs reuse them.

---

## References

Lin, S., Hilton, J., & Evans, O. (2022). TruthfulQA: Measuring how models mimic human falsehoods. In *Proceedings of ACL*.

Wei, Z., Wang, S., Rong, X., Liu, X., & Li, H. (2025). Shadows in the attention: Contextual perturbation and representation drift in the dynamics of hallucination in LLMs. *arXiv:2505.16894.*

Wu, Y., Schlegel, V., & Batista-Navarro, R. (2025). Natural context drift undermines the natural language understanding of large language models. *arXiv:2509.01093.*

---

*Thanks to our TA Satyapragnya for the initial paper pointers.*
