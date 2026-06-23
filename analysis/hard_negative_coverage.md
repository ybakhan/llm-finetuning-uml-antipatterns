# Hard-negative coverage and failure analysis (FD-include)

Generated from `output/run_20260325_012118` (388 samples / 194 domains) and the
verified review files. Re-run `scripts/analyze_hard_negatives.py` to refresh the
coverage tables after regenerating data.

Sources:
- coverage: `scripts/analyze_hard_negatives.py`
- test-set failures: `output/run_20260325_012118/results_review_20260527_071224.json`
- MAPSTEDI failures: `evaluation/mapstedi_eval_20260529_102437.json` vs
  `evaluation/mapstedi_sample_label.json`

## Why this matters

FD-include requires an inclusion use case (the target of an `<<include>>`) to
satisfy ALL of the following. Violating any one disqualifies it, so a use case
that violates a criterion is a **hard negative**: it looks include-related but
must NOT be flagged.

- (i) included by exactly one base use case
- (ii) has no direct actor association
- (iii) neither includes nor extends any other use case
- (iv) is not extended by any use case
- (v) is not involved in any generalization

If the training set contains few examples of a disqualifying structure, the model
never learns the corresponding exclusion criterion and false-positives when a real
diagram contains it.

## 1. Hard-negative coverage

Counts over all 388 samples, and split by the 80/20 domain-stratified split the
model actually trained on (155 train domains / 39 test domains).

| Disqualifying structure | Criterion | All samples | All domains | Train domains | Test domains |
|---|---|---|---|---|---|
| Actor-associated inclusion use case | (ii) | 26/388 | 13/194 | 10/155 | 3/39 |
| Inclusion use case that includes another | (iii) | 7/388 | 7/194 | 5/155 | 2/39 |
| Shared inclusion (included by >=2 bases) | (i) | 4/388 | 2/194 | 2/155 | 0/39 |
| Inclusion use case extended by another | (iv) | 2/388 | 1/194 | 1/155 | 0/39 |
| Inclusion use case that extends another | (iii) | 0/388 | 0/194 | 0/155 | 0/39 |
| Inclusion use case in a generalization | (v) | 0/388 | 0/194 | 0/155 | 0/39 |

Two disqualifying structures have zero training exposure (an inclusion use case
that extends another, and one in a generalization). The dataset contains 48
samples with some `<<extend>>` relationship, but in none is a use case both an
include target and an extend source.

## 2. Held-out test-set failures (verified)

Official counts from the review file: TP 68, TN 32, FP 16, FN 10 (precision 0.81,
recall 0.87, F1 0.84, MCC 0.55). Failure causes from the reviewer notes:

False positives:
- **Actor-associated inclusion flagged as FD-include** (dominant): domains 017,
  069, 118, 139, 192 (Check Available Slots/Receptionist; Process Payment/Payment
  Gateway x2; View Photo Details/two actors; Border Officer).
- **Extend misread as include**: 070 (Apply Discount extends Book Group Visit),
  067_re (Report Bug extended by Reopen Bug).
- **Hallucinated / unrelated pair** (elements share no relationship): 025, 067, 159.

False negatives:
- **Missed a standard FD-include (incomplete scanning)**: 025, 067, 083, 119, 154.
- **Missed a chain link (A->B->C, second link)**: 010, 154.

## 3. MAPSTEDI failures (verified)

Model vs ground truth: TP 1, FP 4, FN 2 (precision 0.20, recall 0.33).

- FP *Search Collections Data* -> actor-associated inclusion (PublicUser, ResearchUser)
- FP *Edit Collections Data* -> actor-associated inclusion (DBIntegrator, DataEditor)
- FP *Query Remote Database* -> shared inclusion (included by Integrate and RunQC)
- FP *RunQC Tests* -> inclusion that includes another (RunQC includes QueryRemote, Upload)
- FN *Upload DBG and UCOM Data* -> incomplete scanning (missed standard)
- FN *Find Locality* -> incomplete scanning (missed standard)

## 4. Which failures appear in both test set and MAPSTEDI

| Failure | Held-out test set | MAPSTEDI |
|---|---|---|
| Actor-associated inclusion FP | yes (017, 069, 118, 139, 192) | yes (Search, Edit) |
| Incomplete-scanning FN | yes (025, 067, 083, 119, 154) | yes (Upload, Find Locality) |
| Shared inclusion FP | no (0/39 test domains) | yes (Query Remote) |
| Inclusion-that-includes-another FP | no (not flagged in test) | yes (RunQC) |

Two failure modes are shared (actor-association FP, incomplete-scanning FN), and
MAPSTEDI additionally exposes two structural FPs (shared inclusion,
inclusion-that-includes-another) that the synthetic test set was too sparse to
reveal. These are exactly the starved hard negatives from section 1.

## 5. Implication for the data-generation pipeline

The structural false positives are traceable to under-covered hard negatives, not
random error. Because the data is generated rather than collected, the pipeline
can be steered to oversample these disqualifying configurations (actor-associated
inclusion, shared inclusion, inclusion that includes or extends another, and the
currently absent generalization case) so the model learns every exclusion
criterion. This is a targeted fix a fixed real-world dataset could not offer.

## Per-structure example domains

- Shared inclusion: 021 (Movie Ticket Booking, Process Payment), 181 (Irrigation
  Control, Validate Sensor Readings).
- Inclusion that includes another: 008, 010, 036, 093, 115, 154, 161.
- Inclusion extended by another: 059 (Notify Applicant).
- Actor-associated inclusion: 002, 008, 017, and 10 other domains.
