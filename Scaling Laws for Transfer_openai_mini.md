# Scaling Laws for Transfer

## Executive Summary
Problem being solved: The paper asks how and by how much pretraining transfers to downstream fine-tuning — in particular, can we put transfer into intuitive, quantitative units ("effective data transferred") and predict how transfer scales with model size, fine‑tuning data, and pretraining? The goal is to make fine‑tuning performance, compute cost, and data requirements predictable so practitioners can decide when to pretrain, when to collect more task data, and how to trade model size against labeling effort, especially in the low‑data regime where transfer is most important.

Proposed approach: The authors run a controlled empirical program that compares pretrained transformers (pretrained on ~24B text characters) fine‑tuned on a Python code target to matched from‑scratch models across four orders of magnitude in model size and dataset size, using KMH+20‑style optimization and careful early‑stopping. They measure performance in units of effective data (DE = DF + DT), define D(N) (the downstream data needed to reach near‑infinite‑data performance), and fit a compact power‑law model for transfer: DT = k · (DF)^α · N^β. They build this global fit by “fitting the fits” across DF values (logit transforms) and validate behavior across pretraining corpora and ablations (including a controlled small Python injection).

Key results and contributions: The central empirical law is DT = k · (DF)^α · N^β, with a model‑size exponent β ≈ 0.38 that is surprisingly stable across pretraining corpora and an α that quantifies distributional proximity (how transfer decays as DF increases). Major findings include: (a) pretraining multiplicatively increases effective fine‑tuning data, explaining large few‑shot gains for big models; (b) "ossification": for small models in the high‑data regime pretraining can hurt final performance (pretraining can be a poor prior when abundant task data exists); (c) fine‑tuning is generally compute‑efficient if pretraining cost is ignored; and (d) the low‑data transfer fit breaks down entering the medium‑data regime. The paper also proposes a tentative unified fine‑tuning scaling law (by substituting DT into KMH+20 from‑scratch form), offers a manifold‑tiling intuition for the power laws, and gives a practical experimental recipe (small DF fractions at multiple sizes) to estimate the tradeoff between collecting more labeled data versus increasing model size. The authors are explicit about limitations (hyperparameters tuned for from‑scratch, single downstream domain—Python—possible warmup confounds, and limited LR scans for largest models).

Why this matters: By expressing transfer as an effective amount of data that scales predictably with model size and task data, the paper gives a practical, quantitative toolkit for planning both research and engineering: deciding whether to invest in pretraining, collecting more labeled data, or scaling model size; predicting few‑shot behavior of large pretrained models; and estimating compute/data tradeoffs with cheap pilot experiments. The results unify and extend scaling‑law thinking to transfer (not just from‑scratch training), surface failure modes (ossification) that affect small models with abundant task data, and point to concrete future work to broaden domains, tasks, and architectures so these empirical laws can guide deployment decisions across ML applications.

## Key Contributions
- Derivation and empirical validation of a low-data transfer scaling law: the effective data transferred from pretraining scales as DT = k · DF^α · N^β (measured β ≈ 0.38), so pretraining acts multiplicatively to increase effective fine‑tuning data; α quantifies distributional proximity and governs how transfer decays with increasing DF.
- A concrete operationalization of transfer in units of data — DE = DT + DF — and a measurement protocol (including the D(N) metric: data to reach 99% infinite-data performance) that lets one decompose overall performance into pretraining-contributed versus fine-tuning-contributed effective data.
- Identification and characterization of “ossification”: an empirical regime in which pretraining can worsen final performance for small models with abundant target data (D/D(N) → 1), showing pretraining can act as a poor prior/initialization and that this effect is predictable with the D(N) regime framing.
- Empirical demonstration that, when ignoring pretraining cost, fine‑tuning is usually more compute‑efficient than training from scratch (fine‑tuning learning curves lie tangent to the compute‑efficient Pareto frontier), with the ossification regime as a notable caveat.
- A tentative unified fine‑tuning scaling proposal: substitute the measured DT into the KMH+20 from‑scratch scaling form to predict loss (equation 6.1), together with a manifold‑tiling interpretation that explains why transfer follows power laws and speculative quantitative links to few‑shot behavior.
- A practical, reproducible methodology for estimating transfer coefficients (the “fit‑the‑fits” logit approach) and a cheap experimental recipe — small-scale fine‑tunes on fractions of a candidate dataset plus model‑size sweeps — that lets practitioners quantify the tradeoff between collecting more fine‑tuning data and increasing model size.

## Section Summaries

### 1 Introduction
The introduction frames fine-tuning as a form of transfer that can dramatically improve sample efficiency, especially in the low-data regime. The authors introduce a quantitative lens—"effective data transferred" (DT)—and describe experiments (varying model size and three curricula) that lead to a simple scaling law (equation 1.1) relating pretraining, fine-tuning data, and performance.
- Introduce the concept of effective data transferred (DT) to measure how much in-task data pretraining effectively supplies; DT can be orders of magnitude larger than the actual fine-tuning data (DF) — e.g., a 40M transformer fine-tuned on 3e5 characters had DT ≈ 1000× DF.
- Focus on low-data fine-tuning (python code) because many tasks lack large datasets; code provides an interesting domain intermediate between natural text and other code distributions.
- Methodological novelty: analyze units of data while holding model size and performance constant, which yields surprisingly clean fits summarized by a simple scaling relation (equation 1.1).
- Experiments train transformers of various sizes under three curricula — from-scratch on python, pretrain on natural language then fine-tune on python, and pretrain on mixed text+code then fine-tune — to quantify transfer effects.

  ### 1.1 Key Results
  The section presents an empirical scaling law for transfer in the low-data regime: the effective data transferred (DT) from pre-training follows DT = k (DF)^α (N)^β, where DF is fine-tuning data and N model parameters. Key implications are that pre-training multiplicatively increases effective fine-tuning data, the model-size exponent β (≈0.38) appears architecture/target-dependent and stable across pretraining corpora, while α measures distributional proximity and governs how transfer diminishes with more DF.
  - Scaling law: DT = k (DF)^α (N)^β accurately fits across ~4 orders of magnitude in model size and ~3 in DF in the low-data regime, with an effective-data multiplier approximation DE/DF ≈ DT/DF = k N^β / (DF)^{1−α}.
  - Empirical coefficients (Table 1): Text→Python: k≈1.9e4, α≈0.18, β≈0.38; 50% Text+50% non-python→Python: k≈2.1e5, α≈0.096, β≈0.38 — same β but different k and α, indicating corpus affects magnitude and proximity.
  - Practical trade-offs: for text→python they find β ≈ 2α, so increasing DF by factor C is roughly equivalent to increasing N by √C (e.g., 10× model size ≈ 100× fine-tuning data).
  - Implications: pre-training substantially helps in the low-data regime (DT >> DF), can move models out of data-constrained plateaus, and measuring α,β,k requires few experiments and can guide decisions between collecting task data versus scaling models.

  ### 1.2 Notation
  This section defines the notation used for measuring transfer in characters and model-size units: DE (total effective data), DT (effective data transferred), DF (fine-tuning data), N (parameter count), and the scaling-law constants α, β, and k. It also specifies loss L (cross-entropy in nats/token), compute units C (FLOPs), and references the KMH+20 scaling parameters (αN, αD, DC, NC) and D(N) (data to reach 99% of infinite-data performance).
  - DE and DT are measured in characters of python-equivalent data: DE is the total effective data a same-size from-scratch model would need to match a pre-trained model; DT is the extra python-equivalent data contributed by pretraining.
  - DF is the fine-tuning dataset size (characters); N is model parameter count (excluding vocabulary and positional embeddings).
  - α, β, and k are the power-law exponents/constants governing effective data transferred; L is cross-entropy loss in nats/token and C denotes FLOP compute units.
  - D(N) is defined as the data required to reach 99% of infinite-data performance for a given N; αN, αD, DC, NC are the KMH+20 loss-scaling exponents/constants referenced for loss behavior.

### 2 Methods
The Methods section describes a controlled experimental protocol that trains models pre-trained on large text corpora, fine-tuned on Python, and trained from scratch on Python across four orders of magnitude in model and dataset size. Training follows KMH+20-style optimization (Adam, batch 256, seq length 2048, 3k-step warmup, vocab 50,257) to convergence/optimal early stopping; pretraining used ~24B text characters and the Python corpus was ~22B characters (3% held out).
- Experiments span 4 orders of magnitude in both model parameter count (N) and dataset size, enabling scaling-law analysis of transfer.
- Training protocol mirrors KMH+20: Adam optimizer, batch size 256, sequence length 2048, 3000-step warmup, reversible tokenizer, and vocabulary of 50,257; models were trained to convergence with early stopping.
- Pretraining corpora: WebText2, Common Crawl, English Wikipedia, and Internet Books totaling ~24 billion characters; code corpus: ~22 billion Python characters from public GitHub with 3% held out for evaluation.
- Design isolates transfer effects by keeping optimization and compute measurement consistent (FLOPs) across pre-trained, fine-tuned, and from-scratch runs.

### 3 Results
Section 3 reports two main empirical findings: (1) a phenomenon called “ossification,” where pre-training can hurt fine-tuning performance in the high-data regime for small models, and (2) that fine-tuning is generally compute-efficient (when ignoring the cost of pre-training). The section formalizes data regimes via D(N) (data to reach 99% of infinite-data performance) and focuses on the low-data regime where pre-training is most beneficial.
- Ossification: pre-training can act like a poor initialization that reduces effective fine-tuning data for small models when D/D(N) approaches 1; in this regime some small models trained from scratch outperform their fine-tuned counterparts even with 10–100× D(N).
- D(N) and regimes: D(N) is defined as the data needed to reach 99% of infinite-data performance; the low-data regime is D/D(N) < 0.10, and the paper concentrates on this regime because pre-training helps most there.
- Compute efficiency of fine-tuning: ignoring pre-training compute, fine-tuning is easier to keep on the compute-efficient frontier — fine-tuning training curves remain tangent to the frontier for most of training, whereas from-scratch models only touch it briefly.
- Caveats: ossification might be recoverable with additional tuning (not explored) and larger models may exhibit different behavior in the high-data regime.

  ### 3.1 Ossification – can pre-training harm performance?
  This subsection identifies and characterizes “ossification”: an empirical phenomenon where pre-training can hurt fine-tuning performance for small models in the high-data regime. Using D(N) (data required to reach 99% of infinite-data performance) to define data regimes, the authors show that as D/D(N) → 1 pre-trained models often exhibit reduced effective data and fail to match from-scratch training, suggesting pre-training can act as a poor initialization (with caveats about tuning and scale).
  - Ossification defined: pre-training can ‘ossify’ weights so models adapt worse to the fine-tuning distribution in the high-data regime.
  - Empirical finding: for small models, fine-tuning after large amounts of task data (e.g., >1e8 chars, or 10×–100× D(N)) can underperform models trained from scratch on the same task.
  - Regime formalism: D(N) measures data to reach 99% of infinite-data performance; low-data regime is D/D(N)<0.10, where pre-training is most beneficial, while harm appears as D/D(N) approaches 1.
  - Interpretation and caveats: authors view ossification as a poor initialization problem that might be recoverable with tuning; behavior may differ for larger models and wasn’t explored further.

  ### 3.2 Fine-tuning is usually compute efficient (ignoring pre-training compute)
  Ignoring the cost of pre-training, fine-tuning is typically more compute-efficient than training from scratch: fine-tuning learning curves lie tangent to the compute‑efficient (Pareto) frontier for most of training, whereas from‑scratch curves only touch that frontier briefly. A key caveat is that for small models in the high‑data regime (near or above D(N)) fine‑tuning can converge to worse performance (ossification) and thus be less compute‑efficient.
  - Main empirical result: when not counting pre‑training FLOPs, fine‑tuning more easily attains the compute‑efficient frontier than training from scratch — its training curves are tangent to the frontier for most of training.
  - From‑scratch training only aligns with the compute frontier over a narrow window of compute; thus it is harder to be compute‑efficient when training from scratch for a given dataset size.
  - Caveat: for small models with as much or more data than required to reach near‑infinite‑data performance (D(N)), fine‑tuned models can converge worse than from‑scratch (ossification) and become less compute‑efficient.
  - When evaluating converged training (not intermediate checkpoints): pre‑trained models are more compute‑efficient than from‑scratch for small datasets, and across dataset sizes it is generally easier to land on the compute frontier when fine‑tuning.

### 4 Related Work
This Related Work section situates the paper within prior studies of scaling laws, transfer, and meta-learning. It emphasizes that while many works have studied power-law scaling and transfer (in language, vision, few-shot learning, and sim-to-real), the present paper’s novelty is analyzing how transfer itself scales with compute, data, and model size rather than just performance when training from scratch.
- Power-law and scaling behavior in neural networks have been studied broadly (e.g., HNA+17) and can arise from many sources; the closest prior empirical studies are RRBS19, KMH+20, HKK+20.
- The paper’s contribution differs by focusing on how transfer (pre-training → fine-tuning) scales with compute, data, and parameters, not just scaling of from-scratch training.
- Related literature spans pre-training gains in vision (Instagram, ImageNet), CLIP’s zero-shot transfer, few-shot/meta-learning work, and sim-to-real transfer (e.g., Rubik’s cube robotic manipulation) that motivated studying transfer when fine-tuning data is costly.
- The authors note this is not a comprehensive survey and point to recent reviews (TSK+18, Wen18); they also highlight benchmark development as relevant (e.g., harder language benchmarks).

### 5 Limitations
The authors identify three primary limitations: hyperparameters were not tailored for fine-tuning or code tasks (they used settings tuned for from‑scratch natural language training), small‑dataset experiments may be confounded because training sometimes finished before warmup completed, and transfer experiments were limited to Python so results may not generalize across distribution pairs. They note limited learning‑rate scans for some larger models but acknowledge those searches were not comprehensive.
- Hyperparameters were optimized for from‑scratch natural language training, not for fine‑tuning or code; only a few learning‑rate scans were run for larger models and these were not exhaustive.
- Small‑dataset runs often ended before the warmup phase completed, so the learning schedule could have biased results for low‑data regimes.
- Transfer was evaluated only on Python fine‑tuning, so observed scaling/transfer behaviors may not hold across other source/target distribution pairs.

### 6 Discussion
The discussion proposes a tentative unified scaling law for fine-tuning by substituting an effective transferred dataset size into the KMH+20 from‑scratch scaling form (equation 6.1), and offers a manifold‑tiling interpretation for why transfer follows power laws. It gives speculative quantitative estimates that pretraining can correspond to very large effective data amounts (explaining few‑shot behavior) and suggests a practical experiment to decide whether collecting more fine‑tuning data is worth the cost. The authors emphasize important caveats about extrapolation, scope (unsupervised only, transformers only), and measurement limitations.
- Unified fine‑tuning law: substituting the transfer effective data (DT) into KMH+20 yields equation 6.1 describing overall loss in the low‑data regime, providing a candidate closed‑form for fine‑tuned performance relative to from‑scratch training.
- Manifold interpretation: speculatively, pretraining ‘‘tiles’’ parts of the downstream data manifold at lower density, which could explain the observed power‑law transfer behavior and why pretraining gives effective data multipliers.
- Quantitative few‑shot estimates and caveats: extrapolating their fits suggests large pretrained models (e.g., GPT‑3 scale) can be equivalent to hundreds of millions–billions of characters of in‑domain data (e.g., 3.7e8 chars for python; 4.1e9 chars for text+code), and a few‑shot context (≈300 chars) can multiply effective data modestly; these numbers are speculative because they rely on large extrapolations and potential dataset contamination (addressed in Appendix F).
- Practical application and limitations: they propose cheap diagnostic fine‑tuning experiments (e.g., 1% vs 10% of data plus model‑size sweeps) to decide whether to collect more costly fine‑tuning data; major limitations include only unsupervised transfers, transformer models only, no robust closed‑form from‑scratch fit in this work, issues handling exact zero‑shot in the formulas, and unexplored cheaper measurement alternatives (e.g., KL‑based proxies).

  ### 6.1 Potential unified scaling law for fine-tuning
  The authors propose a tentative unified scaling law for fine-tuning by substituting the effective transferred dataset size (DT) from their transfer scaling relation into the KMH+20 from‑scratch scaling form, yielding equation 6.1 that predicts overall loss in the low‑data regime. They interpret this as consistent with a manifold‑tiling picture—pretraining covers parts of the downstream data manifold at lower density than direct training—while emphasizing the result is speculative and subject to important caveats.
  - Equation 6.1 is obtained by replacing the dataset size D in KMH+20 with the effective transfer data DT (from eq. 1.1), giving a closed‑form prediction for loss for fine‑tuned models in the low‑data regime.
  - Conceptual insight: transfer behaves like adding effective training data because pretraining ‘‘tiles’’ the downstream manifold, explaining why transfer follows power laws and why few‑shot behavior can be strong.
  - Practical implication: the unified form gives a way to trade off collecting more fine‑tuning data versus increasing model size (e.g., by estimating effective data gains from pretraining).
  - Key caveats: the law is derived by substitution and extrapolation (often far beyond measured ranges), was validated only in unsupervised transformer experiments, and does not naturally handle true zero‑shot without approximations.

  ### 6.2 Speculative estimates for why large models would be few-shot learners
  The authors give speculative quantitative estimates linking their transfer scaling law to why large pretrained models behave as few‑shot learners. Using their effective-data multiplier extrapolated to near zero-shot and few‑shot contexts, they estimate equivalent from‑scratch dataset sizes for GPT‑3–scale models and note substantial caveats about the extrapolations.
  - Extrapolating the transfer law to ~1 character (zero‑shot) yields an effective training-equivalent of ~3.7×10^8 Python characters for a GPT‑3–sized model; 300‑character few‑shot prompts increase that effective data by ~2.8×, which helps explain observed few‑shot gains (e.g., 10–15 SuperGLUE points).
  - For text+code pretraining the estimate is larger: DE ≈ 4.1×10^9 characters, with 300‑character examples worth ~1.7× effective data.
  - The estimates are highly speculative: they extrapolate model size by ~100× and data amount by ~10^5×, and they assume context examples are equivalent to fine‑tuned data; potential contamination from trace Python in pretraining data is addressed in Appendix F.

  ### 6.3 Potential applications of these scaling laws
  The section proposes a practical use of the transfer scaling laws: perform small, cheap fine-tuning experiments on fractions (e.g., 1% and 10%) of a candidate dataset and vary model size on the full dataset to empirically estimate the power-law tradeoff between additional fine-tuning data and model size. From that fit one can quantify the expected performance loss (expressed as an equivalent reduction in model size) and thus decide if collecting more expensive fine-tuning data is worth the cost.
  - A concrete, low-cost experiment is recommended: fine-tune on small subsets (1% and 10%) and measure how performance scales with model size on the full dataset to infer the transfer power law.
  - Use the inferred relation to express the value of additional fine-tuning data in terms of equivalent model-size reduction (i.e., estimate lost performance if you don’t collect the data).
  - This is especially useful when fine-tuning data is costly to obtain (example: human preference labels for summaries), enabling principled cost–benefit decisions about data collection.
  - Applicability is subject to the paper’s caveats: the observed scaling is from unsupervised pretraining on transformers, involves extrapolation, and depends on measurement choices, so results may not generalize to all settings.

  ### 6.4 Core factors vs details
  Building on from‑scratch scaling work that identified data, compute, and parameter count as the dominant factors, the authors argue that fine‑tuned performance will follow a similar pattern but with data split into pre‑training and fine‑tuning components. They expect architectural details and modest hyperparameter choices (e.g., depth/width/attention heads, fine‑tuning hyperparameter tweaks, or number of pre‑train epochs) to have much smaller effects than the amounts and distributions of pre‑training and fine‑tuning data, and call for empirical tests of this hypothesis.
  - Core factors for performance remain data, compute, and parameters; fine‑tuning likely follows the same principle but with data partitioned into pre‑training and fine‑tuning sets.
  - The amounts and distributions of pre‑training and fine‑tuning data are expected to dominate performance differences, more so than architectural details (depth/width/heads) or modest hyperparameter changes.
  - Changes like varying fine‑tuning hyperparameters, pre‑train epoch count, or smoothly shifting distributions should have smaller effects than changing dataset sizes or the pre‑training distribution.
  - The claim is presented as a hypothesis; the authors highlight the need for future empirical evaluation to confirm these relative effect sizes.

  ### 6.5 How similar are Python and English?
  The authors argue that transfer between Python code and English text is a useful case study for transfer across distant distributions. Although Python contains English elements (docstrings, comments, identifiers), its formal, instruction-oriented nature contrasts with informal natural language, so studying code↔text transfer can illuminate behavior for other far-apart and near-apart distribution pairs.
  - Python and English are meaningfully different: Python is a formal instruction language while English is an informal human language, though Python includes English fragments (comments, docstrings, names).
  - Code↔text transfer is taken as representative of transfer between distant distributions and can therefore provide insight into other challenging pairs (e.g., English vs Math) as well as closer pairs (e.g., English vs French).
  - Studying transfer across such gaps helps characterize how pretrained models generalize across distributional distance and informs expectations for future transfer problems.

  ### 6.6 Ossification
  The authors report that small models fine-tuned from pre-trained weights failed to reach the performance of models trained from scratch even when given 10x–100x more fine-tuning data (Figure 6). They interpret this as evidence of “ossification,” where strong pre-trained priors reduce a model’s ability to absorb new distribution-specific information, an effect that appears to scale predictably and can make heavy pre‑training counterproductive when abundant target data is available.
  - Empirical observation: fine-tuned small models did not match from‑scratch performance despite large increases (10x–100x) in fine‑tuning data, suggesting a persistent deficit from pre‑training.
  - Ossification defined: pre-trained weights can ‘‘saturate’’—losing plasticity—so the learned prior becomes counterproductive for learning a new distribution.
  - Practical implication: when ample target‑distribution data exists, training from scratch may be preferable to heavy pre‑training, since pre‑training can be computationally wasteful and hinder adaptation.
  - Broader analogy and consequence: ossification scales predictably and resembles human phenomena (early training inducing hard‑to‑change habits), pointing to a need to study how to measure and mitigate ossification across scales.

  ### 6.7 Future work we're particularly excited about
  The authors outline several concrete research directions to deepen and apply transfer scaling laws: expanding empirical measurements of transfer coefficients across more and broader target distributions (including task sets), extending transfer scaling to supervised and reinforcement settings and to comparisons across architectures, and deriving unified predictive equations for transfer. They’re particularly excited about practical methods to predict optimal pre‑training data mixes—both a simple pre‑train ratio for two datasets relative to a target and an algorithm to allocate pre‑training data when there are multiple limited downstream targets.
  - Measure transfer coefficients across many unsupervised distributions and for broader/more general target distributions or task sets.
  - Develop transfer scaling laws in supervised and reinforcement‑learning settings and compare Transformers to other architectures; seek unified predictive equations for transfer.
  - Create cheap, predictive methods to choose the ideal pre‑training ratio between datasets A and B to maximize performance on a limited‑data target C, potentially based on transfer coefficients.
  - Use multiple alpha (transfer) measurements to construct optimal pre‑training data splits when aiming to serve multiple downstream distributions/tasks with limited data.

### 7 Conclusion
The authors demonstrate that transfer between distributions in language models is measurable across a wide range of model sizes and follows predictable scaling behavior. They frame transfer in intuitive units of data, provide a cheap method to estimate transfer in low-data regimes, and present scaling laws for fine-tuning that help predict performance, compute, and data requirements for larger fine‑tuned models.
- Transfer is measurable and scales predictably across model sizes, enabling quantitative comparison of cross-distribution transfer.
- Transfer is expressed in units of data, an intuitive metric; given pre-trained models this allows inexpensive measurements in low-data regimes.
- They introduce scaling laws for fine‑tuning that can predict performance, compute, and data needs when scaling up fine‑tuned models.
- The approach offers a novel way to reason about data as an ML ingredient and the generality of AI systems.

### A Data Regime
This section defines a data-regime metric D(N): the amount of downstream data required for a model of size N to reach near–infinite-data performance (taken as 99% for from-scratch, 95% for fine-tuned). It presents the algebraic decomposition of total effective data (DE = DT + DF), gives an explicit form for the fraction of effective data contributed by transfer, and describes how a global fit (using an empirical exponent ≈0.38) was built by “fitting the fits” across DF values to produce the predictive transfer curve.
- Data-regime definition: D(N) is the dataset size to reach ~infinite-data performance; DF ≤ 10% of D(N) is considered the low-data regime.
- Effective-data decomposition: total effective data DE = DT + DF, and fraction from transfer = DT/(DF+DT), which can be written in parametric form involving k, α, β and N.
- Global predictive fit: empirical logit-style fits across DF showed a roughly constant exponent, leading to the compact form fraction = 1 / (1 + (N*/N)^{0.38}), with N* fitted to produce the global transfer scaling law.
- Practical choices: the 99%/95% intersection thresholds for estimating D(N) were judgment calls based on noisier fine-tuning curves; fits give equal weight to each DF when producing the global curve.

### B Supplementary equations
This section defines total effective data as DE = DT + DF and gives an explicit closed-form for the fraction of effective data coming from transfer. Using equation 1.1 and the decomposition, the fraction plotted in Figure 2 is written as DT/(DF+DT) = k (DF)^{α−1} N^{β} / (1 + k (DF)^{α−1} N^{β}), making the dependence on fine-tuning data DF and model size N explicit.
- Total effective data decomposes as DE = DT + DF, separating transfer-derived and fine-tuning contributions.
- The fraction of effective data from transfer is given by B.2: DT/(DF+DT) = k (DF)^{α−1} N^{β} / (1 + k (DF)^{α−1} N^{β}), which is the quantity plotted on the vertical axis of Figure 2.
- This formula makes explicit how transfer scales with model size (via N^{β}) and with the fine-tuning dataset size (via the exponent α−1) with k, α, and β determined by empirical fits.

### C Generating fit for Equation 1.2
This section explains how the authors produced a global predictive fit for the fraction of effective data coming from transfer by “fitting the fits.” They found that the logit-transformed fraction DT/(DF+DT) produced nearly straight lines with a common exponent ≈0.38, leading to the compact form (Eq. C.1) and a global curve (Eq. 1.1) obtained by fitting the per-DF threshold N* and aggregating across DF. Alternative parametrizations were explored but the logit-fit approach proved most robust.
- On logit axes the per-DF curves are approximately straight with a shared exponent ≈0.38, motivating the model DT/(DF+DT)=1/(1+(N*/N)^0.38) (Eq. C.1).
- They obtain the global predictive curve (Eq. 1.1) by fitting N* for each DF and then aggregating those fits (“fitting the fits”), giving equal weight to each fine-tuning dataset size.
- Other representations (raw effective data, DT/DF normalized by D(N)) were tested but were noisier; the logit-linearization yielded the most reliable, interpretable regularity.
- They note practical details: the largest model shows signs of overfitting late in training, and the resulting global fit improves predictive capability for transfer across model sizes and DF regimes.

### D Figure 3, including medium data regime
The authors report that the simple transfer fit (equation 3.2) that describes low-data transfer breaks down once the medium-data regime (D(N) > 0.10) is reached: the transfer behavior becomes less regular and is no longer well-fit by a straight line. The breakdown is particularly pronounced for models pre-trained on programming or mixed-source data, while text-only pre-training shows less deviation; analyses only include points with DT > 0 and the fine-tuning shown used 5.5e9 Python characters.
- Equation 3.2 fails in the medium/high data regime (D(N) > 0.10): transfer curves are no longer approximately linear in the fitted representation.
- Breakdown is stronger for models pre-trained on code or mixed modalities than for text-only pre-training; plotted points exclude DT = 0 cases.
- Two likely causes: (1) estimating effective data by interpolation is poorly conditioned when curves for different models lie nearly on top of each other (small performance noise causes large parameter swings), and (2) fine-tuning hyperparameter choices matter more in the high-data regime and the authors performed relatively little tuning.

### E Attempt to generate global fit to our from-scratch python runs
The authors attempted to fit a global power-law (Equation E.1, the form used in [KMH+20]) to their from‑scratch Python training runs but found the resulting fit too poor to support deriving a unified fine‑tuning scaling law L(N,DF). They note this failure could reflect insufficient experimental tuning or that a different functional form (e.g., variants proposed in [RRBS19]) might be more appropriate.
- Tried to apply the KMH+20-style global power‑law (Eq. E.1) to Python models trained from scratch and to use that as a basis for a unified fine‑tuning law L(N,DF).
- The empirical fit was inadequate, so it could not reliably serve as the foundation for a combined scaling law across pretraining and fine‑tuning. 
- Authors suggest that improved experimental tuning or an alternative parametrization (as in [RRBS19]) might yield a better fit. 
- By "unified scaling law" they mean a single L(N,D) form whose coefficients also apply when replacing D by DF for fine‑tuning; this was not achieved here.

### F Addressing potential concern about small amounts of python in text dataset
The authors added a controlled 0.3% of Python data into pretraining to test whether zero-shot Python performance came from a small uncontrolled Python fraction in the text corpus. Instead of the expected ~2× increase in effective Python data (if simply additive), they observed a ~4.5× increase, implying the zero-shot gains are largely due to transfer from the text distribution to Python rather than contamination alone.
- Experiment: injected a known 0.3% Python fraction into pretraining and measured changes in zero-shot Python performance for larger models.
- Expectation vs observation: if gains were just from added data, effective Python data would double (~2×); observed increase was ~4.5×, much larger than additive effect.
- Conclusion: majority of the previously reported zero-shot Python ability arises from transfer from text to Python, not from an uncontrolled small amount of Python in the pretraining data.
- Supports prior results in Section 6.2 by demonstrating transfer is the dominant factor driving zero-shot performance improvements.

### G Analysis of best epoch
Fine-tuning on large code datasets requires roughly the same number of epochs as training from scratch (typically 1–10 epochs), with epoch counts declining as model size and dataset size increase. For small fine-tuning sets, convergence is faster—about 2–5× fewer epochs—so early stopping is practical, though warmup and learning-rate schedules confound the exact optimal epoch and produce Z-shaped validation curves similar to prior work.
- When fine-tuning on substantial amounts of data, required training time (in epochs) is similar to from-scratch training—about 1–10 epochs for code—and epochs generally decrease with larger models and more data.
- For very small fine-tuning datasets, models converge 2–5× faster than from-scratch, making early stopping a simple practical strategy.
- Expressing results in terms of D(N) (data fraction relative to model) makes patterns clearer; validation curves can be Z-shaped, mirroring observations in NKB+19.
- Estimating the best epoch for small datasets is confounded by warmup and the learning-rate schedule, since much learning happens during warmup periods.
