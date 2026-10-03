# 06 — Evaluation Protocol

**Purpose:** state before any experiment runs exactly what will be measured, what
it will be compared against, and what would count as failure.

Written in advance deliberately. A protocol fixed after seeing results is a
protocol fitted to them.

---

## 1. What Must Be True for This Project to Have Succeeded

Stated as falsifiable claims, so the answer can be *no*:

| # | Claim | Falsified if |
|---|---|---|
| **H1** | Claim-level NLI detects hallucinations better than answer-level scoring | PRAMANA does not beat the RAGAS-threshold and sentence-NLI baselines on detection F1 |
| **H2** | Detection quality is comparable across the three languages | Hindi or Tamil F1 falls more than 10% relative to English |
| **H3** | The fused confidence score is calibrated | ECE > 0.10 in any language |
| **H4** | Evidence-guided correction reduces hallucination without breaking answers | Hallucination reduction < 40%, **or** regression rate > 10%, **or** relevance drops > 5% |
| **H5** | Cross-lingual verification strategy matters | The three arms are statistically indistinguishable |

**H5 falsified is still a result.** "Translate-then-verify buys nothing over native
verification" is useful to the field and must be reported as readily as the
opposite.

---

## 2. Metrics

### 2.1 Retrieval (stage 1)

| Metric | Definition | Why |
|---|---|---|
| Recall@k | Share of questions whose gold chunk is in the top k | Upper bound on everything downstream — an answer cannot be grounded in evidence never retrieved |
| nDCG@k | Rank-weighted relevance | Rewards ranking the right chunk first |
| MRR | Mean reciprocal rank of the first gold chunk | Single-number summary |
| Empty rate | Share returning nothing | Should correlate with answerability class C |

**Reported per language.** Retrieval is the first place the cross-lingual gap can
open, and attributing a generation failure to the model when retrieval never
supplied the evidence would be wrong.

### 2.2 Detection (stage 3) — the primary metric

Claim-level, three-way, against human annotation.

| Metric | Detail |
|---|---|
| Precision / Recall / F1 | Per verdict class **and** macro-averaged |
| Binary F1 | SUPPORTED vs. not — the headline hallucination-detection number |
| Confusion matrix | Per language. The CONTRADICTED↔UNVERIFIABLE cell matters most: confusing them sends the wrong correction |
| AUROC | Threshold-free, so results do not depend on the tuned cut-points |
| Claim-alignment rate | Share of system claims matching an annotator claim — isolates decomposition quality from verification quality |

> **Macro-F1 is the headline, not accuracy.** SUPPORTED will dominate the label
> distribution, so accuracy rewards a detector that flags nothing. Fixture F03 is
> the same guard at unit-test level.

### 2.3 Confidence (stage 4)

| Metric | Detail |
|---|---|
| **ECE** | 10 bins, **per language**. The headline calibration number |
| Brier score | Combines calibration and sharpness |
| Reliability diagram | Per language. Shows *how* it is miscalibrated, not just that it is |
| AUROC (confidence vs. correctness) | Does the score rank good answers above bad ones? |
| Selective risk / coverage | Error rate among answers above threshold t, as t varies — the curve an operator actually uses to set a routing threshold |

**Never averaged across languages.** A score calibrated in English but not Tamil is
a failure of the multilingual claim, and averaging conceals precisely what is under
study.

### 2.4 Correction (stage 5)

| Metric | Definition |
|---|---|
| Hallucination-rate reduction | Relative change in claim-level hallucination rate, before → after |
| **Regression rate** | Share of correction attempts that made the answer worse |
| Net improvement | Improved − regressed. The honest headline |
| Relevance retention | Post-correction relevance ÷ pre-correction |
| Abstention rate | Per language — see §2.5 |
| Abstention precision | Of answers abstained on, the share that genuinely should have been |
| Iterations to convergence | Mean loop depth |

> Regression rate is reported with equal prominence to the improvement figure. A
> system that improves 60% and degrades 20% is not the same as one that improves 45%
> and degrades none, and the headline number alone cannot distinguish them.

### 2.5 Abstention — promoted to a first-class metric

Added after finding **P1**, which showed abstention behaviour differing sharply by
language under identical prompts.

| Metric | Why |
|---|---|
| Abstention rate per language | Directly measures the P1 asymmetry |
| Abstention precision | Correct refusals ÷ all refusals |
| Abstention recall | Correct refusals ÷ questions that should be refused (classes C and D) |
| **Over-abstention rate** | Class A questions refused. **A system that never answers has a hallucination rate of zero and is useless** |

Hallucination rate and abstention rate must be read together. Optimising either
alone is meaningless.

### 2.6 Efficiency

Latency p50/p95 per stage; API requests and tokens per query; end-to-end overhead
versus the vanilla baseline (target ≤ 2×).

---

## 3. Baselines

Six systems on identical inputs. Baselines 4 and 5 exist to justify specific design
decisions, not to pad the table.

| # | System | Isolates |
|---|---|---|
| 1 | **Vanilla RAG** — no verification | Lower bound: what happens with no assurance layer |
| 2 | **RAGAS faithfulness threshold** | Comparison against the standard tool |
| 3 | **LLM-as-judge** — "is this supported? yes/no" | The obvious cheap alternative |
| 4 | **Sentence-level NLI** — no decomposition | **The value of claim decomposition** |
| 5 | **Self-consistency only** — no NLI | **The value of NLI grounding** |
| 6 | **PRAMANA (full)** | — |

If PRAMANA does not beat 4 and 5, the corresponding component is not earning its
cost and the report must say so.

---

## 4. Ablations

Each removes one component and re-runs the full evaluation.

| # | Ablation | Question answered |
|---|---|---|
| A1 | Rule-based instead of LLM decomposition | How much does LLM decomposition contribute? |
| A2 | Remove S1 / S2 / S4 in turn | Contribution of each confidence signal |
| A3 | No calibration (raw fusion score) | Does post-hoc calibration help? |
| A4 | Isotonic instead of temperature scaling | Is one parameter enough? |
| A5 | Disable each correction action | Contribution of each policy branch |
| A6 | Global instead of per-language NLI thresholds | Does per-language tuning matter, and by how much for Tamil? |
| A7 | Vary decomposition granularity | Optimal atomicity — an open question in the literature |
| A8 | Swap generator (Qwen / Llama-3 / Gemma) | Is the assurance layer generator-independent? |
| A9 | Swap verifier (transformer NLI / LLM / keyword) | Verifier sensitivity, with the keyword backend as the lexical floor |
| A10 | Sparse-only vs. dense-only vs. hybrid retrieval | Justifies the hybrid design |
| A11 | Verifier aggregation: support-wins vs. contradiction-wins | Documented as a judgement call; measure it |
| **A12** | **Prompt variants, held fixed across languages** | **Separates genuine language effects from prompt sensitivity — the P1 confound** |

A12 was added after P1: a cross-lingual difference measured under a single prompt
could be prompt sensitivity rather than a language effect, and RQ4 depends on
telling them apart.

---

## 5. Cross-Lingual Arms (RQ4)

| Arm | Mechanism |
|---|---|
| **A — Native** | Multilingual NLI verifies in the source language |
| **B — Translate-then-verify** | Claims and evidence translated to English, verified with a stronger English NLI model |
| **C — Hybrid** | Retrieval in source language, evidence translated post-retrieval, verification in English, verdicts mapped back to source-language claim spans |

Compared on detection F1 and ECE, per language, with significance testing. The
arms are run on the **same** generated drafts (from the cache), so differences come
from verification and nothing else.

**A negative result is publishable.** If native verification matches
translate-then-verify on Tamil, that is worth knowing — and it would mean the
simpler architecture is sufficient.

### 5.1 Cross-language setting (exploratory extension)

The arms above keep question and document in one language. A common enterprise
case breaks that assumption: the policy is in English and the employee asks in
Hindi or Tamil. This setting is an exploratory extension of RQ4, not a new
research question.

**Mechanism.** An uploaded document's language is detected, and every question is
routed to that document's index. When the question's language differs, one model
call adds a translated query variant, fused with the original by rank. The answer
is generated in the question's language. Claims are then verified directly
against the document-language evidence by the LLM judge, which is told that
premise and hypothesis may be in different languages. This is cross-lingual
native verification (arm A); arms B and C are not implemented for this setting.

**Procedure.**

1. The sample policies in `examples/corpus/` state the same facts in English,
   Hindi and Tamil, so the document's language is the only variable. They were
   aligned on 2026-10-04. Before that, the Tamil text lacked two facts.
2. `examples/eval/crosslingual.template.jsonl` holds 31 items in 10 parallel groups
   (answerability A 18, B 3, C 6, D 3). Each item lists `evidence_languages`, the
   document languages that state its answer. Items new in version
   `crosslingual-0.1` are machine-drafted and marked `human_reviewed: false`.
3. `python scripts/run_crosslingual.py` answers every question against the
   document in every language. It writes `raw.jsonl`, `summary.json`,
   `manifest.json` and `review.csv`.
4. A person marks `human_mark` in `review.csv` as `correct`, `partial` or
   `incorrect`. Then `--score review.csv` reports accuracy per condition and
   language pair with 95% bootstrap intervals.

**What may be reported.** `summary.json` describes behaviour against the draft
labels: answering answerable questions, abstaining on unanswerable ones, and
releasing only supported claims. It is not accuracy. Accuracy comes only from
human marks, and is reported with *n* and its interval. On this corpus every
figure is a pilot observation. The corpus is three fictional paragraphs, so
retrieval is close to trivial. The judge shares a provider with the generator.
The Hindi and Tamil items are machine-drafted. Each condition is a single run.

---

## 6. Statistical Protocol

| Aspect | Choice |
|---|---|
| Significance test | Paired bootstrap, 10,000 resamples, paired on `parallel_id` |
| Confidence intervals | 95% bootstrap percentile, reported for every headline metric |
| Multiple comparisons | Holm–Bonferroni across the six baselines |
| Effect size | Reported alongside p-values — a significant 0.5-point gain is not an interesting one |
| Seeds | Fixed and recorded; three seeds for anything stochastic, with variance reported |

**Point estimates are never reported without an interval.** With a few hundred
questions per language, the interval will often be wide enough to change the
conclusion, and hiding that would be the most likely way this project reports
something false.

---

## 7. Experiment Execution

```
scripts/run_experiment.py --config experiments/main.yaml
   │
   ├─ load eval set (dev or test — test is touched once)
   ├─ load cached drafts        ← identical inputs for every system
   ├─ for system in baselines + PRAMANA:
   │     for language in en, hi, ta:
   │         run → collect AssuranceResult
   ├─ join to annotations on qid
   ├─ compute metrics (§2), per language and pooled
   ├─ paired bootstrap (§6)
   └─ write reports/{run_id}/{tables.csv, figures/, raw.jsonl, manifest.json}
```

**Every ablation replays against the same cached drafts.** Live generation would
inject sampling noise into every comparison; replay means an observed difference
comes from the component under test.

**Manifest.** Config hash, dataset version, model strings and generation dates, seeds,
git commit, host profile, and the draft-cache version. Without the last field, split
execution silently destroys reproducibility — the same config could replay against
different drafts.

---

## 8. Threats to Validity

Stated in advance, because a threats section written after results is a defence
rather than an analysis.

| # | Threat | Mitigation | Residual |
|---|---|---|---|
| **T1** | Synthetic corpus lacks real-world messiness | Realistic domains; external validation on BHRAM-IL | Real enterprise noise is not represented. **Stated as a limitation** |
| **T2** | Tamil data is MT-derived | Four-stage QC (`05_DATASET_SPEC.md` §5) | If human review is impossible, Tamil is reported as indicative only |
| **T3** | Small samples | Bootstrap CIs; no claim without an interval | Wide intervals limit what can be concluded |
| **T4** | Annotator subjectivity | ≥ 2 annotators, blind, κ reported, adjudication | Genuinely ambiguous claims remain |
| **T5** | Hosted models may change mid-project | Cache is the artefact; model string and date in every manifest | Live reproduction may differ; the cache reproduces exactly |
| **T6** | Prompt sensitivity confounds RQ4 | Ablation A12 | Prompt space cannot be exhausted |
| **T7** | **Pipeline artefacts masquerading as language effects** | Regression suite; eight such defects already found (`08_RESULTS.md` §P4–P6) | Undiscovered artefacts may remain. **The audit is reported so readers can judge** |
| **T8** | S3 unavailable on free endpoints | Fitted without S3; dispersion substituted | Results are not directly comparable to work using token probabilities |
| **T9** | Test-set contamination through repeated tuning | Thresholds and calibrators fitted on dev only; test touched once | — |

**T7 deserves emphasis.** Eight defects were found before any result was produced,
each of which would have appeared as a number in a results table rather than as an
error — and three would have concentrated in Hindi or Tamil specifically. Any
cross-lingual gap this project reports must be shown not to be a pipeline artefact
before it is attributed to the model. The report will say so, and cite the audit.

---

## 9. Reporting Standard

The results section will include, without exception:

1. **Per-language tables** for every metric. No cross-language averaging.
2. **Confidence intervals** on every point estimate.
3. **Regression rate** beside every correction improvement figure.
4. **Abstention rate** beside every hallucination rate.
5. **Negative results** — H5 falsified, a language where calibration fails to
   transfer, an ablation showing a component does not earn its cost.
6. **The defect audit** (§T7), as a methodological contribution rather than an
   appendix.

> In a project about reliability, concealing a failure mode would contradict the
> work's own premise. Every metric in this document that could embarrass the system
> is reported with the same prominence as the ones that flatter it.
