# Hand check: questions where reranker (a) lost the most reciprocal rank

Evidence for a human judgement; not used to choose anything. For each question: where the gold evidence ranks, and (a)'s top 3 chunks.

## q305 (factual): RR hybrid 1.000 -> (a) 0.111

**Question:** How many instruction templates did the FLAN authors write for each dataset?

- Gold (2109.01652): "For each dataset, we manually compose ten unique templates that use natural language instructions to describe the task for that dataset." | hybrid rank 1, (a) rank 9

**(a) #1** `2305.14314:0069` QLoRA: Efficient Finetuning of Quantized LLMs > Appendix B Training a State-of-the-art Chatbot Experimental Setup Details > B.1 Datasets > FLAN v2

> The FLAN v2 collection [ref] is a collection of 1836 tasks augmented with hundreds of manually curated templates and rich formatting patterns into over 15M examples. The authors show that models trained on this collection outperform other public collections including the original FLAN 2021 [ref], T0++ [ref], Super-Natural Instructions [ref], and OPT-IML [ref]. We used the same task mixtures described by the authors with the exception of some datasets that were not freely available at the time of writing.

**(a) #2** `2109.01652:0010` Finetuned Language Models Are Zero-Shot Learners > 3 Results

> and even surpasses GPT-3’s few-shot performance on 10 datasets. With the best dev-template, zero-shot FLAN outperforms zero-shot GLaM on 13 of 19 available datasets and one-shot GLaM on 11 of 19 datasets. Overall, we observe that instruction tuning is very effective on tasks naturally verbalized as instructions (e.g., NLI, QA, translation, struct-to-text) and is less effective on tasks directly formulated as language modeling, where instructions would be largely redundant (e.g., commonsense reasoning and coreference resolution tasks that are formatted as finishing an incomplete sentence or paragraph). Results on natural language inference, reading comprehension, closed-book QA, and translati ...

**(a) #3** `2109.01652:0009` Finetuned Language Models Are Zero-Shot Learners > 3 Results

> We evaluate FLAN on natural language inference, reading comprehension, closed-book QA, translation, commonsense reasoning, coreference resolution, and struct-to-text. As described in § 2.2, we evaluate on unseen tasks by grouping datasets into task clusters and holding out each cluster for evaluation while instruction tuning on all remaining clusters (i.e., each evaluation task cluster uses a different checkpoint). For each dataset, we evaluate the mean of performance on all templates, which proxies the expected performance given a typical natural language instruction. As a dev set is sometimes available for manual prompt engineering (Brown et al., 2020), for each dataset we also obtain the  ...

## q204 (factual): RR hybrid 1.000 -> (a) 0.250

**Question:** On which long-range benchmark did FlashAttention let a Transformer beat chance for the first time, and at what sequence length?

- Gold (2205.14135): "FlashAttention yields the first Transformer that can achieve better-than-random performance on the challenging Path-X task (sequence length 16K)" | hybrid rank 1, (a) rank 4

**(a) #1** `2205.14135:0006` FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness > 1 Introduction

> We empirically validate that FlashAttention speeds up model training and improves model quality by modeling longer context. We also benchmark the runtime and memory footprint of FlashAttention and block-sparse FlashAttention compared to prior attention implementations. • Faster Model Training. FlashAttention trains Transformer models faster in wall-clock time. We train BERT-large (seq. length 512) 15% faster than the training speed record in MLPerf 1.1 [ref], GPT2 (seq. length 1K) 3$\times$ faster than baseline implementations from HuggingFace [ref] and Megatron-LM [ref], and long-range arena (seq. length 1K-4K) 2.4$\times$ faster than baselines. • Higher Quality Models. FlashAttention scale ...

**(a) #2** `2205.14135:0037` FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness > 4 Experiments > 4.2 Better Models with Longer Sequences > Path-X and Path-256.

> The Path-X and Path-256 benchmarks are challenging tasks from the long-range arena benchmark designed to test long context. The task is to classify whether two points in a black and white 128$\times$128 (or 256$\times$256) image have a path connecting them, and the images are fed to the transformer one pixel at a time. In prior work, all transformer models have either run out of memory, or only achieved random performance [ref]. There has been a search for alternative architectures that can model such long context [ref]. We present here the first result of Transformer models being able to solve Path-X and Path-256 (Table 6). We pretrain a transformer on Path-64, and then transfer to Path-X b ...

**(a) #3** `2205.14135:0032` FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness > 4 Experiments > 4.1 Faster Models with FlashAttention > Long-range Arena.

> We compare vanilla Transformer (with either standard implementation or FlashAttention) on the long-range arena (LRA [ref]) benchmark. We measure accuracy, throughput, and training time of all models. Each task has a different sequence length varying between 1024 and 4096. We follow the implementation and experimental setting in Tay et al. [80]and Xiong et al. [90]. Table 3 shows that FlashAttention achieves up 2.4$\times$ speed-up compared to standard attention. Block-sparse FlashAttention is faster than all of the approximate attention methods that we have tested. [Footnote: LRA accuracy results are known to be highly dependent on the tuning procedure [ref]. Our reproduced baselines perform ...

## q209 (factual): RR hybrid 1.000 -> (a) 0.250

**Question:** To save compute, which sequence lengths did BERT pre-train with, and for what share of the steps?

- Gold (1810.04805): "we pre-train the model with sequence length of 128 for 90% of the steps" | hybrid rank 1, (a) rank 4
- Gold (1810.04805): "Then, we train the rest 10% of the steps of sequence of 512 to learn the positional embeddings." | hybrid rank 1, (a) rank 4

**(a) #1** `2003.10555:0016` ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators > 3 Experiments > 3.3 Small Models

> As a goal of this work is to improve the efficiency of pre-training, we develop a small model that can be quickly trained on a single GPU. Starting with the BERT-Base hyperparameters, we shortened the sequence length (from 512 to 128), reduced the batch size (from 256 to 128), reduced the model’s hidden dimension size (from 768 to 256), and used smaller token embeddings (from 768 to 128). To provide a fair comparison, we also train a BERT-Small model using the same hyperparameters. We train BERT-Small for 1.5M steps, so it uses the same training FLOPs as ELECTRA-Small, which was trained for 1M steps. In addition to BERT, we compare against two less resource-intensive pre-training methods bas ...

**(a) #2** `1907.11692:0014` RoBERTa: A Robustly Optimized BERT Pretraining Approach > 3 Experimental Setup > 3.1 Implementation

> We reimplement BERT in fairseq Ott et al. (2019). We primarily follow the original BERT optimization hyperparameters, given in Section 2, except for the peak learning rate and number of warmup steps, which are tuned separately for each setting. We additionally found training to be very sensitive to the Adam epsilon term, and in some cases we obtained better performance or improved stability after tuning it. Similarly, we found setting $\beta_{2}=0.98$ to improve stability when training with large batch sizes. We pretrain with sequences of at most $T=512$ tokens. Unlike Devlin et al. (2019), we do not randomly inject short sequences, and we do not train with a reduced sequence length for the  ...

**(a) #3** `1910.10683:0106` Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer > 3 Experiments > 3.7 Putting It All Together

> (Joshi et al., 2019). Specifically, we use a mean span length of $3$ and corrupt $15\%$ of the original sequence. We found that this objective produced marginally better performance (Table 7) while being slightly more computationally efficient due to shorter target sequence lengths. Longer training Our baseline model uses a relatively small amount of pre-training ($\nicefrac{{1}}{{4}}$ as much as BERT (Devlin et al., 2018), $\nicefrac{{1}}{{16}}$ as much as XLNet (Yang et al., 2019), $\nicefrac{{1}}{{64}}$ as much as RoBERTa (Liu et al., 2019c), etc.). Fortunately, C4 is big enough that we can train for substantially longer without repeating data (which can be detrimental, as shown in Sectio ...

## q206 (factual): RR hybrid 1.000 -> (a) 0.500

**Question:** ELECTRA's generator is smaller than its discriminator. Which weights do the two networks still share?

- Gold (2003.10555): "we only share the embeddings (both the token and positional embeddings) of the generator and discriminator" | hybrid rank 1, (a) rank 2

**(a) #1** `2003.10555:0013` ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators > 3 Experiments > 3.2 Model Extensions

> vocabulary densely updates all token embeddings. On the other hand, tying all encoder weights caused little improvement while incurring the significant disadvantage of requiring the generator and discriminator to be the same size. Based on these findings, we use tied embeddings for further experiments in this paper. Smaller Generators If the generator and discriminator are the same size, training ELECTRA would take around twice as much compute per step as training only with masked language modeling. We suggest using a smaller generator to reduce this factor. Specifically, we make models smaller by decreasing the layer sizes while keeping the other hyperparameters constant. We also explore us ...

**(a) #2** `2003.10555:0012` ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators > 3 Experiments > 3.2 Model Extensions

> We improve our method by proposing and evaluating several extensions to the model. Unless stated otherwise, these experiments use the same model size and training data as BERT-Base. Weight Sharing We propose improving the efficiency of the pre-training by sharing weights between the generator and discriminator. If the generator and discriminator are the same size, all of the transformer weights can be tied. However, we found it to be more efficient to have a small generator, in which case we only share the embeddings (both the token and positional embeddings) of the generator and discriminator. In this case we use embeddings the size of the discriminator’s hidden states. The “input” and “out ...

**(a) #3** `2003.10555:0005` ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators > 2 Method

> Figure 2: An overview of replaced token detection. The generator can be any model that produces an output distribution over tokens, but we usually use a small masked language model that is trained jointly with the discriminator. Although the models are structured like in a GAN, we train the generator with maximum likelihood rather than adversarially due to the difficulty of applying GANs to text. After pre-training, we throw out the generator and only fine-tune the discriminator (the ELECTRA model) on downstream tasks. We first describe the replaced token detection pre-training task; see Figure 2 for an overview. We suggest and evaluate several modeling improvements for this method in Sectio ...

## q309 (factual): RR hybrid 1.000 -> (a) 0.500

**Question:** How does DPO's summary quality change as the decoding temperature is raised, compared with PPO?

- Gold (2305.18290): "we find DPO to be much more robust to the sampling temperature than PPO, the performance of which can degrade to that of the base GPT-J model at high temperatures" | hybrid rank 1, (a) rank 2

**(a) #1** `2305.18290:0029` Direct Preference Optimization: Your Language Model is Secretly a Reward Model > 6 Experiments

> demonstrating the quality of the optimization. Right. TL;DR summarization win rates vs. human-written summaries, using GPT-4 as evaluator. DPO exceeds PPO’s best-case performance on summarization, while being more robust to changes in the sampling temperature. Evaluation. Our experiments use two different approaches to evaluation. In order to analyze the effectiveness of each algorithm in optimizing the constrained reward maximization objective, in the controlled sentiment generation setting we evaluate each algorithm by its frontier of achieved reward and KL-divergence from the reference policy; this frontier is computable because we have acccess to the ground-truth reward function (a senti ...

**(a) #2** `2305.18290:0035` Direct Preference Optimization: Your Language Model is Secretly a Reward Model > 6 Experiments > 6.2 Can DPO scale to real preference datasets?

> Next, we evaluate fine-tuning performance of DPO on summarization and single-turn dialogue. For summarization, automatic evaluation metrics such as ROUGE can be poorly correlated with human preferences [ref], and prior work has found that fine-tuning LMs using PPO on human preferences to provide more effective summaries. We evaluate different methods by sampling completions on the test split of TL;DR summarization dataset, and computing the average win rate against reference completions in the test set. The completions for all methods are sampled at temperatures varying from 0.0 to 1.0, and the win rates are shown in Figure 2 (right). DPO, PPO and Preferred-FT all fine-tune the same GPT-J SF ...

**(a) #3** `2305.18290:0074` Direct Preference Optimization: Your Language Model is Secretly a Reward Model > Appendix D Additional Empirical Results > D.2 Sample Responses and GPT-4 Judgments

> In this section, we present examples of comparisons between DPO and the baseline (PPO temp 0. for summarization, and the ground truth chosen response for dialogue). See Tables 4-6 for summarization examples, and Tables 7-10 for dialogue examples. Table 4: Sample summaries to a post from the TL;DR test set. DPO sample generated with a temperature 0.25; PPO sample generated at a temperature of 0. The order in which summaries are presented in randomized when evaluating with GPT-4, so the order in the Judgment may not correspond to the order in the table. For clarity, post-hoc annotations are included in bold, formatted as [annotation]. These annotations are not part of the model generations. Pr ...

## Judgement (by hand; reported only, nothing was tuned on it)

Each of (a)'s top-3 chunks, judged against the question, with the full chunk text read where the excerpt above is cut:

| Question | (a) #1 | (a) #2 | (a) #3 | Verdict |
|---|---|---|---|---|
| q305 | wrong paper (QLoRA on FLAN v2; its header contains "FLAN") | right paper, on topic, no count | right paper, on topic, no count | **real loss**: none answers "ten templates" |
| q204 | **answers fully**: Introduction says the same fact in other words ("better-than-chance ... Path-X ... (16K)"); the gold quote is the abstract's wording | on topic: Path-X, no sequence length | on topic (LRA), does not answer | **measurement artefact**: the fact appears twice, gold names one place |
| q209 | wrong paper (ELECTRA-Small shortening 512 to 128) | partial: RoBERTa restates BERT's "reduced sequence length for the first 90% of updates", second-hand, no lengths | off topic (T5) | **real loss**: cross-paper distractors promoted |
| q206 | partial: "we use tied embeddings" (no token + positional detail) | **gold chunk** | on topic, does not answer | **mostly benign**: gold at #2 |
| q309 | **answers**: "more robust to changes in the sampling temperature" than PPO (without the "degrades to base GPT-J" detail) | **gold chunk** | on topic, does not answer | **measurement artefact**: the fact appears twice |

Tally over the 15 chunks: 2 answer fully without the gold quote, 3 answer partly, 2 are the gold chunk, 5 are on topic in the right paper without answering, 3 are distractors (2 from other papers, 1 off topic).

Reading: two of the five "losses" (q204, q309) are artefacts of single-location gold evidence. The paper states the fact twice, the reranker picks the other place, and quote matching (rapidfuzz partial ratio at 90) does not count a paraphrase. This undercounting can affect every setup, not only the reranker. The two real losses (q305, q209) promote chunks from *other* papers that share the question's key term (FLAN, BERT), so paper identity matters more than the cross-encoder weighs it.
