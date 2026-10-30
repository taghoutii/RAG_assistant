# Error analysis

The final system is plain hybrid retrieval (BM25 + dense, RRF), top 8 chunks, answered by
`openai/gpt-oss-120b`. This page sorts 13 failures, 9 from the held-out test run and 4 from dev,
into four categories: **retrieval miss**, **reranker drop**, **hallucination** and **wrong
abstention**. Each failure comes with its evidence: whether the gold evidence quote was among
the 8 chunks the model saw, what the model answered, and what the judge flagged. Judge labels
for citation correctness and correctness are indicative only (agreement with the hand grades:
kappa 0.52 and 0.46), so every judge-flagged case below was checked against the chunk text.

Nothing was changed or re-run on test after these results were seen.

## Summary

Across all 36 test questions, 16 hybrid answers have a flagged outcome (a wrong abstention, or
any judge label of 0). Their main causes:

| Cause | Test questions | Count |
|---|---|---|
| Retrieval miss: the gold evidence was not in the top 8 | q103, q207, q301, q313, q412 (abstained); q105, q203, q213, q411 (partial answer); q212 (hallucination, see below) | 10 |
| Generation: citation points at the wrong chunk | q505 | 1 |
| Judge strictness: correct, cited answer marked 0 for a minor omission or a split claim | q112, q310, q403, q407, q506 | 5 |

Retrieval misses dominate. The generator behaves as instructed when evidence is missing:
on test, all 7 unanswerable questions abstained, and every false abstention on an answerable
question (5 of 29) happened with no gold evidence in the top 8. Only one test answer states
a fact that is not in the context (q212).

## Retrieval miss

**Abstained because the evidence was not retrieved (test, hybrid).**

- **q103:** *Why does adding more retrieved passages not make the Fusion-in-Decoder encoder's
  cost explode?* Gold (FiD, 2007.01282): "it only performs self attention over one context at a
  time". Top 8: T5 architecture chunks, RETRO, DPR and one FiD experiments chunk, but not the
  one with the quote. The model abstained. The reranker's top 8 would have contained it.
- **q301:** *Why do Kaplan et al. leave the embedding matrices out when counting parameters?*
  Gold (2001.08361): "performance of models with different depths converge to a single trend".
  Top 8: mostly Chinchilla chunks *about* Kaplan et al., and one Kaplan chunk without the
  quote. The model abstained. The reranker's top 8 would have contained it.
- **q207:** *How does GPT-3's training ramp up the batch size?* Gold (2005.14165): "gradually
  increase the batch size linearly from a small value (32k tokens)". The top 8 were crowded by
  other training-hyperparameter chunks (LoRA's GPT-3 settings, Toolformer, RoBERTa's large
  batches). The model abstained. The reranker missed it too.
- **q313, q412 (comparisons):** for q313 all 8 chunks came from the Flan-T5 paper and none held
  either gold quote (FLAN's "62 text datasets", the "1,836 tasks"). For q412 neither the E5
  prefix quote nor the BGE instruction quote was retrieved. Both abstained.

**Answered only the retrieved half (test, hybrid).** This is rule 3 of the prompt working as
designed:

- **q411:** *How differently do Sentence-BERT and supervised SimCSE use NLI data?* All 8
  chunks are from SimCSE (1 of 3 evidence items). The answer explains SimCSE's entailment
  positives and contradiction hard negatives, cited, then says "The paper does not describe
  how Sentence-BERT utilizes NLI data". That is true of the context. Correctness 0 is right
  (the SBERT half is missing). Citation correctness 0 is a **judge error**: the judge counted
  the "not covered" sentence as an uncited claim, which its own rubric excludes.
- **q213:** *How big is T5's C4 compared with GPT-3's filtered Common Crawl?* C4's "about
  750 GB" was retrieved and cited. GPT-3's 45TB/570GB figures are in none of the 8 chunks
  (checked), and the answer says so. This is the same judge error on citation correctness as
  q411.

**Answered off-target (dev, hybrid + reranker).**

- **q402:** *Why doesn't prefix-tuning optimize the prefix vectors directly, and what does it
  do instead?* Neither gold quote (unstable optimization; the MLP reparametrization) was in
  the top 8. The answer explains instead that the model weights stay frozen, a different
  question. Relevance 0, the only one in any setup.

## Reranker drop

- **q305 (dev, hybrid + reranker):** *How many instruction templates did the FLAN authors
  write for each dataset?* Plain hybrid had the gold quote ("we manually compose ten unique
  templates") in its top 8 and answered. The cross-encoder pushed it below rank 8 (its top 8
  has QLoRA's FLAN v2 setup, FLAN results and Flan-T5 chunks), and the model abstained.

## Hallucination

- **q212 (test, hybrid):** *How do the original Transformer and RoFormer give the model
  information about word order?* All 8 chunks are from RoFormer (1 of 2 evidence items). The
  answer's first claim, that the Transformer adds an absolute-position embedding, is broadly
  true, but it comes from the model's memory. It is cited to chunk [5], which never mentions
  it (checked: no "absolute", "Vaswani" or additive encoding in that chunk). This is the only
  unfaithful test answer. It is retrieval-induced: without the Transformer passage, the model
  filled the gap instead of saying that half was not covered.
- **q505 (test, hybrid), citation error:** *Which sampled Toolformer API calls are kept?* The
  answer's filtering rule (keep a call if it reduces the loss by at least tau_f) is supported
  by chunk [2], the appendix with "we keep API calls if they reduce the loss by at least 1.0".
  But it is cited to [3] and [7], the "Sampling API Calls" chunks, which contain no filtering
  criterion (checked). The content is grounded; the pointer is wrong.

## Wrong abstention

- **q102 (dev, BM25), false abstention:** *How long can monoBERT's query be before it is cut
  off?* The gold sentence ("We truncate the query to have at most 64 tokens") was at rank 7 of
  the 8 chunks, surrounded by ColBERT and Lost-in-the-Middle chunks, yet the model abstained.
  This is the only generation-side false abstention found: in every other false abstention,
  across all four setups on dev and on test, the evidence was missing.
- **q215 (dev, hybrid), missed abstention:** *Why does ELECTRA work best when the generator
  is larger than the discriminator?* The premise is false. The answer does not use the
  abstention sentence: it rejects the premise and cites the passage stating the opposite
  ("models perform best when the generator is only 1/4-1/2 the size of the discriminator").
  The content is faithful and arguably the most useful reply, but the exact-sentence rule
  scores it as an error. BM25 and hybrid + reranker also answered rather than abstain; only
  dense abstained.

## A note on the correctness metric

Five correct, faithful, correctly cited test answers were marked 0 for omitting a secondary
detail of the reference answer:

- **q403:** LoRA's W_q + W_v result, without "W_q or W_k alone is worse";
- **q407:** BGE's in-batch negatives and batch of 19,200, without "hard negatives come later";
- **q506:** Mistral's rolling buffer and 8x saving, without "without hurting quality";
- **q310:** both scaling recommendations, without Chinchilla's 70B / 1.4T example;
- **q112:** ColBERT's 128 dimensions. Here the 0 was on citation correctness: the judge split
  the cited sentence into an uncited first half.

The judge used for evaluation was stricter than the hand grades on correctness during
validation. The test correctness score (0.625) therefore understates complete answers, and it
is reported as indicative only.

## Finding: the reranker on dev vs test

Paired difference, hybrid + reranker minus hybrid, with 95% paired-bootstrap intervals:

| Split | Recall@3 | Recall@5 | Recall@10 | MRR | nDCG@10 |
|---|---|---|---|---|---|
| Dev (30 answerable) | +0.006 [-0.15, +0.17] | -0.017 [-0.15, +0.11] | -0.094 [-0.21, +0.01] | -0.032 [-0.17, +0.10] | -0.037 [-0.13, +0.06] |
| Test (29 answerable) | +0.138 [-0.07, +0.34] | +0.115 [-0.07, +0.29] | +0.069 [+0.00, +0.17] | +0.064 [-0.10, +0.22] | +0.087 [-0.04, +0.22] |

On dev, the reranker was slightly below plain hybrid. On test it is above on every metric.
Neither split shows a significant difference: every interval includes 0. This is the
expected behaviour of two small samples of about 30 questions around a true difference near
zero. The system was frozen on dev before the test run, so the final system stays plain hybrid.
The test result is reported as found: the reranker may help, but this evaluation is too small
to show it, and it costs about 1.4 s per query on CPU. Two of the five test retrieval misses
(q103, q301) would have been recovered by the reranker's top 8.
