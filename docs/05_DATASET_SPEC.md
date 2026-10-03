# 05 — Dataset Specification

**Purpose:** define the corpus, the question set, and the annotation protocol
precisely enough that another researcher could rebuild the dataset and get
comparable numbers.

This is deliverable **D3** and contribution **C4**: the first RAG-specific
hallucination resource covering Tamil.

---

## 1. Design Principles

| # | Principle | Rationale |
|---|---|---|
| **D1** | **Parallel across languages** | Every question exists in all three languages with equivalent meaning. Without parallelism a cross-lingual difference could be a difference in the questions rather than in the model — which would sink RQ4 |
| **D2** | **Synthetic corpus, released** | Real enterprise data cannot be published, and an unreleasable dataset cannot be a contribution. Authoring the corpus also lets unanswerable questions be constructed deliberately rather than hoped for |
| **D3** | **Answerability is a controlled variable** | Finding P1 showed abstention behaviour differs sharply by language. A set containing only answerable questions cannot measure that |
| **D4** | **Script variants included** | Real users write Hindi in Devanagari *and* Roman. Most academic sets omit this; every deployment meets it |
| **D5** | **Claim-level labels, not answer-level** | The unit of the contribution is the claim. Answer-level labels cannot evaluate claim-level detection |
| **D6** | **Provenance recorded per item** | Human-authored, machine-translated, or MT-then-verified. Results must be reportable separately by provenance (risk R2) |

---

## 2. Corpus

### 2.1 Domains

Four enterprise domains, chosen because each generates the failure mode this
project targets — confident answers about rules with exceptions:

| Domain | Example content | Why included |
|---|---|---|
| **Health insurance** | Claim deadlines, appeals, exclusions, waiting periods | Numeric conditions and negations — the two error classes that matter most |
| **HR policy** | Leave, reimbursement, notice periods, grievance process | Employees ask personal-case questions the corpus cannot answer — the P1 scenario |
| **Banking products** | Account types, charges, eligibility, dispute windows | Product codes and identifiers, which dense retrieval blurs and BM25 catches |
| **IT support** | Access requests, VPN, password policy, escalation | Procedural, multi-step answers that decompose into several claims |

### 2.2 Scale

| Unit | Target | Minimum viable |
|---|---|---|
| Documents per domain | 8–12 | 5 |
| Total documents per language | ~40 | 20 |
| Chunks per language (after chunking) | 250–400 | 150 |
| Words per document | 300–600 | — |

The minimum column is the fallback if annotation time runs short (risk R4).
Reducing the **question** set is preferred over reducing the corpus: a thin corpus
makes retrieval trivially easy and inflates every downstream metric.

### 2.3 Deliberate corpus properties

Written into the documents on purpose, because each exercises a specific pathway:

- **Numeric conditions** ("within 30 days", "up to Rs. 25,000") — the substrate for
  intrinsic hallucination.
- **Explicit negations** ("not available during the first policy year") — negation
  loss is both a model failure and the most common MT error (§5.2).
- **General rules with exceptions** — the case where one chunk supports a claim and
  another appears to contradict it, which the verifier's aggregation rule must
  handle (`verifier.py`, an ablation candidate).
- **Near-duplicate passages across domains** ("30 days" appearing in both claims and
  IT access) — forces retrieval to discriminate rather than keyword-match.
- **Identifiers** (`PX-4471`) — exact-match retrieval, which embeddings blur.
- **Deliberate gaps** — topics referenced but never specified, so unanswerable
  questions have a genuine basis.

### 2.4 Authoring and translation order

```
English source (authored)
        │
        ├──► Hindi   ── translate ──► QC pipeline ──► native-script corpus
        │                                   │
        │                                   └──► Roman transliteration ──► script variant
        │
        └──► Tamil   ── translate ──► QC pipeline ──► native-script corpus
                                            │
                                            └──► Roman transliteration ──► script variant
```

English is authored first and is the semantic reference. Every translated item
carries `provenance` and `qc_status`.

---

## 3. Question Set

### 3.1 Answerability classes — the controlled variable

| Class | Share | Definition | Correct behaviour |
|---|---|---|---|
| **A — Answerable** | 50% | The corpus states the answer, directly or as a general rule | Answer, all claims SUPPORTED |
| **B — Partially answerable** | 20% | Part is covered; part is not | Answer the covered part, abstain or omit the rest |
| **C — Unanswerable (out of scope)** | 20% | The topic is absent from the corpus | ABSTAIN |
| **D — Unanswerable (personal instance)** | 10% | Asks about a specific user case the corpus cannot know | ABSTAIN |

> **Class D exists because of finding P1.** Asked *"Why was **my** claim rejected?"*
> against a corpus stating only the general rule, Hindi and Tamil answered with an
> inferred personal circumstance — an extrinsic hallucination — while English
> abstained. Class D isolates that behaviour instead of letting it contaminate
> class A results.

### 3.2 Question types

Balanced across classes so a system cannot score well by handling one type:

| Type | Share | Example | Stresses |
|---|---|---|---|
| Factual lookup | 30% | "What is the claim submission deadline?" | Retrieval precision |
| Numeric | 20% | "How many days do I have to appeal?" | Number preservation |
| Negation / exclusion | 15% | "Is maternity covered in year one?" | Negation handling |
| Multi-hop | 15% | "If my claim is rejected, how long to appeal?" | Multi-chunk grounding |
| Procedural | 10% | "How do I request VPN access?" | Multi-claim decomposition |
| Comparative | 10% | "Which account has lower charges?" | Cross-document reasoning |

### 3.3 Scale

| | Target | Minimum viable |
|---|---|---|
| Questions per language | 300 | 150 |
| Total (3 languages, parallel) | 900 | 450 |
| Romanised variants (hi, ta) | 60 each | 30 each |
| Dev / test split | 30 / 70 | — |

The **dev split fits thresholds and calibrators**; the test split is touched once.
Per-language NLI thresholds and the confidence calibrator are both fitted on dev —
tuning either on test would invalidate every reported number.

### 3.4 Item schema

```jsonc
{
  "qid": "hi-ins-042",
  "parallel_id": "ins-042",        // same across languages — the join key for RQ4
  "language": "hi",
  "script": "native",              // native | roman
  "domain": "insurance",
  "question": "क्लेम जमा करने की समय सीमा क्या है?",
  "answerability": "A",            // A | B | C | D
  "question_type": "numeric",
  "gold_answer": "डिस्चार्ज की तारीख से 30 दिन।",
  "gold_chunk_ids": ["ins_hi_003#c1"],
  "should_abstain": false,
  "provenance": "mt_verified",     // human_authored | mt_raw | mt_verified
  "qc": { "roundtrip_similarity": 0.91, "entities_preserved": true, "human_reviewed": true },
  "split": "test"
}
```

`parallel_id` is what makes the cross-lingual comparison valid: results are joined
on it, so English and Tamil are compared on *the same question*, not on two
questions about the same topic.

---

## 4. Hallucination Annotation

The core of contribution C4, and the part that cannot be automated.

### 4.1 What is annotated

Not the questions — the **model outputs**. For each system-generated answer:

1. Decompose into atomic claims (the system's decomposition is shown; annotators
   may correct it).
2. Label each claim against the retrieved evidence.
3. Record an answer-level judgement.

### 4.2 Claim label definitions

| Label | Definition | Test the annotator applies |
|---|---|---|
| **SUPPORTED** | The evidence states or directly entails the claim | "Can I point at the sentence that says this?" |
| **CONTRADICTED** | The evidence states something incompatible | "Does the evidence say otherwise?" |
| **UNVERIFIABLE** | The evidence neither supports nor refutes | "Is the evidence simply silent on this?" |

**The instruction that matters most:** judge against the **evidence**, not against
the world. A claim that is true in reality but absent from the retrieved passages is
UNVERIFIABLE. Annotators consistently want to mark such claims SUPPORTED, and
allowing it would make the labels measure world knowledge rather than grounding.

### 4.3 Answer-level fields

```jsonc
{
  "answer_id": "...",
  "qid": "hi-ins-042",
  "system": "pramana_full",
  "claims": [
    { "text": "...", "label": "SUPPORTED",    "evidence_chunk_ids": ["ins_hi_003#c1"] },
    { "text": "...", "label": "UNVERIFIABLE", "evidence_chunk_ids": [] }
  ],
  "fully_faithful": false,           // the confidence model's training target
  "answer_relevance": 4,             // 1-5: does it address the question?
  "abstained": false,
  "abstention_correct": null,        // set when abstained, or when it should have
  "annotator_id": "a2",
  "annotation_time_s": 95
}
```

`answer_relevance` exists because faithfulness alone is gameable: "I cannot answer"
is perfectly faithful and useless. Correction must be shown not to trade relevance
for faithfulness.

### 4.4 Protocol

| Step | Detail |
|---|---|
| Annotators | ≥ 2 per language, independent, blind to which system produced the answer |
| Pilot | 30 items, then reconcile the guidelines before the main pass |
| Agreement | Cohen's κ per language, reported. Target ≥ 0.70; below 0.60 the guidelines are revised and the batch redone |
| Adjudication | Disagreements resolved by discussion; unresolved items excluded and counted |
| Blinding | System identity hidden — otherwise "the proposed system" is annotated more generously |
| Order | Randomised, so fatigue does not correlate with system |

**Annotation budget.** At ~90 s per answer, 300 answers × 3 languages × 2
annotators ≈ **45 hours**. This is the project's real bottleneck (risk R4).

### 4.5 Human-verified LLM pre-annotation

The 45-hour figure is reduced to **7–15 hours** by having a **different model**
propose labels and a human verify a stratified sample. Implemented in
`src/pramana/evaluation/annotation.py` and `scripts/annotate.py`.

**The distinction that makes this valid.**

| | |
|---|---|
| **Circular — invalid** | The system labels its own output and is then scored against those labels. F1 is 1.0 by construction. Worse in this project than most: the thesis *is* that LLM verification is unreliable in Indic languages, so using an LLM to build the ground truth assumes the conclusion |
| **Sound** | A different provider proposes labels, a human verifies a stratified sample, and **agreement between them is reported** |

Judge separation is **enforced in code**, not advised: `LLMAnnotator` raises
`JudgeConflictError` when the judge shares a provider or model with the system, and
refuses a multi-provider router that could dispatch to the system's own backend.

**The agreement figure is a result, not a process detail.**

> *"LLM-as-judge agrees with human annotators at κ = 0.82 in English but κ = 0.51
> in Tamil."*

That directly supports the project's argument, and gives baseline #3
(`06_EVALUATION_PROTOCOL.md` §3) a **reliability figure per language** rather than
only an F1. It is reported per language and never averaged.

**Sampling is stratified by (language, proposed label).** Uniform sampling would
draw almost no CONTRADICTED items — the rare class, and the one whose validation
matters most — leaving the judge's behaviour on the critical class unmeasured.

**Workflow.**

```
python scripts/annotate.py propose  --raw reports/<run>/raw.jsonl                                     --system-provider groq --judge google
python scripts/annotate.py review   --limit 40      # the irreducible human part
python scripts/annotate.py agreement --out reports/agreement.json
python scripts/annotate.py export   --out data/eval/annotations.jsonl
```

Review hides the judge's label until the annotator decides, so it cannot anchor
them; revealing it is recorded on the annotation. Progress is flushed after every
keystroke, so the work survives being done across several sittings.

**Every exported label carries `label_source`** (`human` or `llm`). Results must be
reported separately for human-verified and judge-only labels — required wherever no
human reader was available, which for this project means Tamil (risk R2).

**Budget.**

| Approach | Human hours | Valid? |
|---|---|---|
| Full human annotation | ~45 | ✅ |
| **Judge proposes + 30% human review** | **~15** | ✅ Report agreement |
| Minimum viable: 150/language + 30% review | **~7** | ✅ State the limitation |
| Judge only, no review | 0 | ❌ **Circular — invalid** |

---

## 5. Translation Quality Control

Risk **R2** is the project's primary methodological risk: no native Tamil speaker is
available, so machine translation will be used. The full argument is in
`03_PROPOSAL.md` §7.1; this section specifies the mechanism.

### 5.1 Why the eval set is held to a higher standard than the corpus

| | Effect of a translation error |
|---|---|
| **Corpus** | The system performs slightly worse. Degradation, measurable |
| **Evaluation set** | The *ground truth* is wrong. **The measurement is invalid** |

A translation error in the eval set is indistinguishable from a hallucination in the
results. The Tamil arm would then measure translation quality while reporting it as
hallucination detection — and Tamil is where the contribution lives.

### 5.2 The four-stage pipeline

| Stage | Method | Catches | Human effort |
|---|---|---|---|
| **1. Prefer human-authored data** | Source Tamil QA from existing corpora (IndicQA, XQuAD-ta; index via BhashaSutra, arXiv:2604.18423) | Everything — never translated | None |
| **2. Round-trip check** | Tamil → English back-translation, compare to the source by embedding similarity, flag below threshold | Gross semantic drift, dropped clauses | Automated |
| **3. Entity & number preservation** | Rule check that every number, date, currency amount and named entity survives | **Number corruption — the most damaging class for a factuality benchmark** | Automated |
| **4. Targeted human review** | Only flagged items are read by a Tamil speaker; failures are removed, not repaired | Residual meaning errors | ~10–15% of the set |

Stage 1 is the highest-value action and should be attempted **before** translating
anything. Stages 2–3 reduce the human requirement from "read 300 items" to "read
about 40" — one sitting rather than an ongoing commitment.

### 5.3 Negation is the specific danger

Negation loss is the most common MT error class and the most damaging here: a
dropped "not" converts a SUPPORTED claim into a CONTRADICTED one, which is exactly
what the system is being measured on. Fixture **F05** exists to test this pathway,
and stage 3 includes an explicit negation-marker count on both sides.

### 5.4 If no human review is possible

The project degrades gracefully rather than reporting an invalid result:

- Tamil is reported as **MT-derived and unvalidated**, stated prominently in the
  results rather than buried in limitations.
- Tamil results are **indicative, not confirmatory**; no headline claim rests on
  them alone.
- The round-trip similarity distribution is reported as a quantitative proxy, so
  readers can judge the data quality themselves.
- The report explicitly separates the two explanations for any Tamil–English gap:
  genuine cross-lingual degradation versus translation artefact.

**Naming a confound you cannot remove is sound methodology. Ignoring it is not.**

---

## 6. External Validation

Anchoring against published resources guards against a dataset that only this
system does well on:

| Resource | Use |
|---|---|
| **BHRAM-IL** (BHASHA 2025) | Hindi and English splits as an external hallucination check. Its headline — primary score 0.23 across 14 multilingual LLMs — is the strongest available evidence that the problem is unsolved |
| **Hindi-BEIR** (arXiv:2408.09437) | Hindi retrieval baseline, so retrieval quality is comparable to published numbers |
| **IndicQA / XQuAD-ta** | Human-authored Tamil QA — the stage-1 source that avoids translation entirely |
| **RAGAS** | Reference-free faithfulness on the same outputs, as the baseline to beat |

---

## 7. Layout and Versioning

```
data/
├── raw/{en,hi,ta}/{insurance,hr,banking,it}/*.md
├── processed/{en,hi,ta}/chunks.jsonl
└── eval/
    ├── questions.jsonl          # §3.4 schema, all languages
    ├── annotations.jsonl        # §4.3 schema
    ├── guidelines/{en,hi,ta}.md # annotator instructions
    └── qc_report.json           # §5 outputs, per item
```

Every file carries `dataset_version`. Results reference it, because a metric is
meaningless without knowing which version produced it.

**Release.** Corpus, questions and annotations are published together. The corpus is
synthetic and authored for this project, so there is no confidentiality barrier —
one of the reasons §D2 chose synthetic data.

---

## 8. Construction Checklist

- [ ] Author English corpus, 4 domains, with the §2.3 properties present
- [ ] Verify §2.3 properties are actually present (numbers, negations, identifiers, gaps)
- [ ] **Search for existing human-authored Tamil/Hindi QA before translating anything**
- [ ] Translate the corpus; run QC stages 2–3; record `qc_status` per item
- [ ] Generate Romanised variants for Hindi and Tamil
- [ ] Author the English question set, balanced by §3.1 and §3.2
- [ ] Translate questions; run QC; assign `parallel_id`
- [ ] Split dev/test (30/70), stratified by answerability and type
- [ ] Write annotation guidelines in all three languages
- [ ] Pilot-annotate 30 items; measure κ; revise guidelines
- [ ] Main annotation pass, blind and randomised
- [ ] Report κ per language; adjudicate; record exclusions
- [ ] Publish with a version tag
