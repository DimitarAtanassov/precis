# Provided proper attribution is provided, Google hereby grants permission to

## Executive Summary
The Transformer architecture addresses a fundamental computational bottleneck in sequence-to-sequence modeling: the sequential processing requirement of recurrent neural networks (RNNs), which prevents parallelization and hinders the ability to model long-range dependencies efficiently. While RNNs and convolutional alternatives process sequences with linear or logarithmic dependency paths, they suffer from either computational inefficiency or reduced parallelizability. The paper introduces a novel approach that eliminates recurrence and convolution entirely, relying instead on self-attention mechanisms to establish constant-time dependencies between any positions in a sequence.

The Transformer employs a fully attention-based encoder-decoder architecture where input sequences are mapped to continuous representations through multiple stacked layers of multi-head self-attention and position-wise feed-forward networks. The key innovation is the Scaled Dot-Product Attention mechanism, which computes weighted value sums based on query-key interactions and scales by 1/√dk to maintain gradient stability. Multi-head attention extends this by performing eight parallel attention operations with different learned projections, enabling the model to attend to information from multiple representation subspaces simultaneously. The architecture incorporates residual connections, layer normalization, causal masking for auto-regressive decoding, sinusoidal positional encodings to inject sequence order information, and position-wise feed-forward networks with dimensionality 512→2048→512. This design achieves O(n²·d) computational complexity per layer while enabling O(1) sequential operations, contrasting sharply with RNNs that require O(n) sequential steps.

The Transformer achieves state-of-the-art performance on WMT 2014 machine translation benchmarks, with the "big" model attaining 28.4 BLEU on English-German translation and 41.8 BLEU on English-French translation while requiring substantially lower training costs than previous methods. The base model trains to completion in 12 hours on 8 NVIDIA P100 GPUs, while larger models complete training in 3.5 days—dramatically faster than recurrent and convolutional baselines. The model demonstrates generalizability beyond machine translation, successfully applying the architecture to English constituency parsing on Penn Treebank with competitive results. Ablation studies isolate the contribution of individual components, confirming the importance of multi-head attention and attention mechanisms broadly to overall performance.

This work is significant because it fundamentally reshapes sequence modeling by demonstrating that self-attention alone—without recurrence—can achieve superior translation quality while enabling full parallelization and requiring significantly less training time. The constant-time dependency paths enable more effective learning of long-range linguistic phenomena compared to sequential architectures. The Transformer's success on multiple tasks beyond machine translation indicates broad applicability as a general-purpose sequence transduction architecture, establishing a new foundation for deep learning on sequential data that has proven influential for diverse downstream applications.

## Key Contributions
- Introduced the Transformer architecture, the first sequence transduction model to rely entirely on self-attention mechanisms without recurrence or convolution, eliminating sequential computation constraints and enabling full parallelization of sequence processing.
- Developed Scaled Dot-Product Attention mechanism with 1/√dk scaling to efficiently compute weighted sums of values based on query-key interactions while preventing gradient saturation, providing better efficiency than additive attention alternatives.
- Introduced Multi-Head Attention that performs multiple parallel attention operations with different learned linear projections, allowing the model to attend to information from multiple representation subspaces simultaneously rather than using a single attention function.
- Designed encoder-decoder architecture with stacked multi-head attention and position-wise feed-forward networks (FFN) with residual connections and layer normalization, achieving constant-time dependencies between distant positions compared to linear or logarithmic complexity in CNN and RNN alternatives.
- Introduced sinusoidal positional encodings that enable extrapolation to longer sequences and represent relative position offsets as linear transformations, replacing the need for recurrent or convolutional mechanisms to capture sequence order.
- Achieved state-of-the-art results on WMT 2014 machine translation benchmarks (28.4 BLEU English-German, 41.8 BLEU English-French) while requiring substantially reduced training costs on modest hardware (8 NVIDIA P100 GPUs), demonstrating significant practical advantages over prior recurrent and convolutional approaches.
- Demonstrated generalizability of the Transformer architecture beyond machine translation by successfully applying it to English constituency parsing on Penn Treebank with minimal task-specific hyperparameter tuning.

## Section Summaries

### Introduction
This section introduces the Transformer architecture as a novel approach to sequence modeling that eliminates recurrent computation in favor of self-attention mechanisms. The authors motivate this innovation by highlighting the computational limitations of RNNs (which must process sequences sequentially, preventing parallelization) and claim the Transformer achieves state-of-the-art translation results with significantly reduced training time on modest hardware.
- RNNs/LSTMs are limited by inherent sequential computation that prevents parallelization and creates memory bottlenecks, whereas the Transformer enables full parallelization through attention alone
- The Transformer reduces computational distance between any two positions in a sequence to constant operations (vs. linear/logarithmic for convolutional alternatives), improving dependency learning
- Self-attention allows the model to relate all positions in a sequence without sequential processing, making it the first pure self-attention transduction model without RNNs or convolutions
- The architecture achieves state-of-the-art machine translation performance in just 12 hours on 8 P100 GPUs, demonstrating significant efficiency gains over existing approaches

### Background
The Background section positions the Transformer as an innovation addressing the fundamental limitation of sequential computation in RNNs and convolutional alternatives. While prior work explored convolutional models (ConvS2S, ByteNet) and self-attention has been used in various tasks, the Transformer is the first transduction model to rely entirely on self-attention mechanisms without RNNs or convolution, achieving constant-time dependencies between distant positions compared to linear or logarithmic scaling in prior approaches.
- Prior architectures using convolutions required operations that grew linearly or logarithmically with distance between input positions, making long-range dependencies harder to learn
- Self-attention has been successfully applied to reading comprehension, summarization, and entailment, but always alongside recurrent networks until the Transformer
- The Transformer achieves O(1) operations to relate distant positions, though using Multi-Head Attention to mitigate resolution loss from attention averaging
- This is the first sequence transduction model to eliminate both RNNs and convolutions entirely in favor of pure self-attention

### Model Architecture
The Model Architecture section describes a novel encoder-decoder framework that abandons recurrence entirely in favor of self-attention mechanisms. The Transformer maps input sequences to continuous representations through an encoder, then auto-regressively generates output sequences through a decoder, achieving constant-time dependencies between distant positions compared to linear or logarithmic complexity in CNN-based alternatives.
- Eliminates sequential computation bottleneck of RNNs by replacing recurrence with pure self-attention mechanisms, enabling parallelization and 12-hour training on modest hardware
- First transduction model to rely entirely on self-attention without RNNs or convolution, achieving O(1) dependency complexity versus O(n) or O(log n) in ConvS2S and ByteNet
- Standard encoder-decoder architecture where encoder maps input symbols to continuous representations and decoder auto-regressively generates output, mitigating attention resolution loss through Multi-Head Attention

  ### Encoder and Decoder Stacks
  The Encoder and Decoder Stacks section details the core architectural components of the Transformer. Both encoder and decoder consist of N=6 identical stacked layers, where each layer contains multi-head self-attention followed by feed-forward networks. A key innovation is the use of residual connections and layer normalization around each sub-layer. The decoder includes an additional cross-attention sub-layer for attending to encoder outputs and employs causal masking to ensure predictions depend only on preceding positions.
  - Both encoder and decoder use 6 identical stacked layers with residual connections and layer normalization, maintaining a consistent dimension (dmodel=512) throughout
  - Encoder contains two sub-layers per layer: multi-head self-attention and position-wise feed-forward networks
  - Decoder adds a third sub-layer for multi-head cross-attention over encoder outputs, with causal masking to enforce autoregressive generation (predictions at position i depend only on positions < i)
  - The architecture eliminates recurrence entirely, enabling parallel computation across sequence positions

  ### Attention
  The Attention section introduces the foundational concept of attention mechanisms as a mapping function that computes weighted sums of values based on query-key interactions. This function forms the basis for the Transformer's multi-head self-attention mechanism, replacing sequential processing entirely.
  - Attention maps queries and key-value pairs to outputs through weighted summation
  - The mechanism enables the Transformer to replace RNN/CNN-based sequential processing with parallel self-attention computation
  - This approach achieves constant-time dependencies between distant sequence positions, a key advantage over prior sequential models

    ### Scaled Dot-Product Attention
    The Scaled Dot-Product Attention section introduces the core attention mechanism used in Transformers. It computes weighted sums of values based on query-key dot products, scaled by 1/√dk to prevent gradient saturation in softmax for large key dimensions. This mechanism is more efficient than alternatives like additive attention while maintaining comparable theoretical complexity.
    - Scaled Dot-Product Attention computes Attention(Q, K, V) = softmax(QK^T/√dk)V, enabling efficient parallel computation on matrix-packed queries, keys, and values
    - The 1/√dk scaling factor prevents dot products from becoming too large and pushing softmax into regions with vanishing gradients when dk is large
    - Dot-product attention is chosen over additive attention due to superior practical efficiency via optimized matrix multiplication, despite similar theoretical complexity
    - Multi-Head Attention applies the attention function h times in parallel with different learned linear projections, allowing the model to attend to different representation subspaces simultaneously

    ### Multi-Head Attention
    Multi-Head Attention extends the Scaled Dot-Product Attention mechanism by performing multiple parallel attention operations with different learned linear projections. This approach allows the model to attend to information from multiple representation subspaces simultaneously, rather than relying on a single attention function with full-dimensional keys, values, and queries.
    - Multi-head attention performs h parallel attention operations, each with independently learned linear projections to lower dimensions (dk, dk, dv), enabling diverse feature extraction
    - This design allows the model to jointly attend to information from different representation subspaces, enhancing expressiveness compared to single-head attention
    - The multi-head framework builds directly on Scaled Dot-Product Attention, which uses 1/√dk scaling to prevent softmax gradient saturation with large key dimensions

    ### Applications of Attention in our Model
    This section describes three distinct applications of multi-head attention in the Transformer architecture: encoder-decoder attention (allowing decoder positions to attend to all input positions), self-attention in the encoder (allowing positions to attend to previous encoder layer positions), and masked self-attention in the decoder (restricting attention to current and previous positions to maintain auto-regressive generation). The model employs 8 parallel attention heads with reduced dimensionality (dk = dv = 64) for computational efficiency comparable to single-head attention.
    - Encoder-decoder attention enables cross-sequence alignment where decoder queries attend to encoder outputs
    - Encoder self-attention allows hierarchical processing where each position attends to all positions in the previous layer
    - Decoder self-attention uses causal masking (setting illegal connections to −∞ in softmax) to preserve auto-regressive properties and prevent information leakage from future positions
    - 8 parallel attention heads with reduced dimensionality maintain computational efficiency while capturing information from multiple representation subspaces

  ### Position-wise Feed-Forward Networks
  Position-wise Feed-Forward Networks are fully connected feed-forward sublayers that complement attention mechanisms in the Transformer encoder and decoder. Each layer applies the same feed-forward network to every position independently, consisting of two linear transformations with ReLU activation in between (FFN(x) = max(0, xW₁ + b₁)W₂ + b₂), with input/output dimensionality of 512 and inner dimensionality of 2048.
  - The FFN consists of two linear transformations with ReLU activation, equivalent to 1D convolutions, applied identically across positions but with unique parameters per layer
  - Input and output dimensionality is dmodel=512, with an inner layer expansion to dff=2048, providing non-linear transformations alongside attention
  - This design complements multi-head attention by enabling position-wise feature transformation, capturing complex non-linear relationships in the data representation

  ### Embeddings and Softmax
  The Embeddings and Softmax section describes how the Transformer converts input and output tokens to continuous vector representations using learned embeddings with dimension dmodel, and converts decoder outputs to next-token probability distributions through a linear transformation and softmax function. The model employs weight sharing between the embedding layers and pre-softmax linear transformation, with embedding weights scaled by √dmodel to balance their magnitude relative to other model components.
  - Learned embeddings convert input/output tokens to vectors of dimension dmodel = 512
  - Weight matrix is shared between embedding layers and pre-softmax linear transformation for parameter efficiency
  - Embedding weights are scaled by √dmodel to prevent magnitude imbalance in the model

  ### Positional Encoding
  The Positional Encoding section describes how the Transformer injects sequence order information through sinusoidal positional encodings added to input embeddings. Since the model lacks recurrence and convolution, positional encodings using sine and cosine functions of varying frequencies are applied, with wavelengths forming a geometric progression from 2π to 10000·2π. Sinusoidal encodings were chosen over learned embeddings because they enable the model to extrapolate to longer sequences and represent relative position offsets as linear functions of base positions.
  - Sinusoidal positional encodings (sin/cos functions) are added to embeddings to capture sequence order in the absence of recurrence or convolution
  - The encoding design allows relative position offsets to be expressed as linear functions of base positions, facilitating attention over relative distances
  - Sinusoidal encodings were chosen over learned embeddings because they enable extrapolation to sequence lengths longer than training examples
  - Empirical results showed learned and fixed positional encodings produced nearly identical performance (Table 3, row E)

### Why Self-Attention
The "Why Self-Attention" section compares self-attention mechanisms to recurrent and convolutional layers across three key criteria: computational complexity per layer, parallelizability (sequential operations), and path length for long-range dependencies. Self-attention achieves O(n²·d) complexity but enables O(1) sequential operations and constant maximum path length, compared to recurrent layers requiring O(n) sequential steps, making it superior for learning long-range dependencies while allowing full parallelization.
- Self-attention connects all sequence positions with constant O(1) sequential operations, versus O(n) for recurrent layers, enabling full parallelization
- Self-attention maintains O(1) maximum path length between any input-output positions, facilitating learning of long-range dependencies compared to O(n) for recurrence
- Trade-off: self-attention has O(n²·d) per-layer complexity versus O(n·d²) for recurrence, but superior parallelism and path length compensate for sequences of practical lengths

### Training
The Training section details the experimental setup for the Transformer model across four key aspects: (1) Training data uses WMT 2014 datasets (4.5M English-German and 36M English-French sentence pairs) with byte-pair and word-piece encoding, batched by approximate sequence length with ~25K tokens per batch; (2) Hardware consists of 8 NVIDIA P100 GPUs with base models trained for 100K steps (12 hours) and larger models for 300K steps (3.5 days); (3) Optimization employs Adam optimizer with a custom learning rate schedule that linearly increases for 4000 warmup steps then decreases by inverse square root of step number; (4) Three regularization techniques are employed during training.
- Training on standard WMT 2014 datasets: 4.5M English-German pairs and 36M English-French pairs with shared vocabularies (37K and 32K tokens respectively)
- Custom learning rate schedule with linear warmup (4000 steps) followed by inverse square root decay, optimized with Adam (β1=0.9, β2=0.98)
- Hardware: 8 P100 GPUs with training times of 12 hours for base models (100K steps) and 3.5 days for larger models (300K steps)
- Systematic batching strategy using approximate sequence length with ~25K source and target tokens per batch to optimize computational efficiency

  ### Training Data and Batching
  The Training Data and Batching section describes the dataset preparation and batch construction strategy for the Transformer model. The authors trained on WMT 2014 datasets (4.5M English-German and 36M English-French sentence pairs) with byte-pair and word-piece encoding schemes, organizing sentence pairs into batches containing approximately 25,000 source and target tokens each to balance computational efficiency with gradient stability.
  - Used standard WMT 2014 datasets with shared vocabularies (37K tokens for English-German byte-pair encoding, 32K tokens for English-French word-piece encoding)
  - Batched sentence pairs by approximate sequence length with ~25K tokens per batch to optimize training efficiency
  - This batching strategy enabled effective parallel processing on the 8 NVIDIA P100 GPU setup described in subsequent sections

  ### Hardware and Schedule
  The Hardware and Schedule section reports that Transformer models were trained on a single machine with 8 NVIDIA P100 GPUs. Base models completed 100,000 training steps in 12 hours (0.4 seconds per step), while larger models required 300,000 steps over 3.5 days (1.0 seconds per step).
  - Training infrastructure: 8 NVIDIA P100 GPUs on a single machine
  - Base model training: 100K steps completed in 12 hours at 0.4 sec/step
  - Larger model training: 300K steps completed in 3.5 days at 1.0 sec/step

  ### Optimizer
  The Optimizer section describes the use of Adam optimizer with custom hyperparameters (β1 = 0.9, β2 = 0.98, ϵ = 10−9) and a learning rate schedule that implements linear warmup for 4000 steps followed by inverse square root decay, enabling stable training of Transformer models on large-scale machine translation datasets.
  - Adam optimizer with specific hyperparameters designed for stable convergence
  - Learning rate schedule: linear warmup for 4000 steps, then inverse square root decay over step number
  - Formula-based scheduling (Equation 3) balances exploration during early training with progressive reduction in later stages
  - Warmup phase prevents optimization instabilities common in large-scale deep learning

  ### Regularization
  The Regularization section (5.4) indicates that the Transformer model employs three types of regularization techniques during training, though the specific details of these techniques are not fully provided in the excerpt shown. This section follows the discussion of the Adam optimizer and its custom learning rate schedule, completing the description of the model's training regime.
  - Three regularization techniques are employed during training
  - This section concludes the Training methodology description
  - Specific regularization methods are referenced but details appear in the following content not shown in this excerpt

### Results
The Results section demonstrates that the Transformer architecture achieves state-of-the-art performance on WMT 2014 machine translation benchmarks, with the big model attaining 28.4 BLEU on English-German and 41.8 BLEU on English-French while requiring substantially lower training costs (measured in FLOPs) than previous methods. The results are obtained using checkpoint averaging and beam search decoding strategies during inference.
- Transformer (big) surpasses previous state-of-the-art by >2.0 BLEU on EN-DE (28.4) and achieves 41.8 on EN-FR, both with significantly reduced training costs (3.3×10^18 to 2.3×10^19 FLOPs versus 10^20 FLOPs for competitors)
- Base model achieves 27.3 BLEU on EN-DE and 38.1 on EN-FR, outperforming all published models at a fraction of competitive training costs
- Inference uses checkpoint averaging (last 5 checkpoints for base, last 20 for big models), beam search (size 4), and length penalty optimization, with hyperparameters tuned on development sets

  ### Machine Translation
  The Machine Translation section demonstrates that the Transformer achieves state-of-the-art performance on WMT 2014 benchmarks, with the big model attaining 28.4 BLEU on English-German and 41.8 BLEU on English-French while requiring substantially lower training costs than previous methods. The regularization techniques employed include residual dropout (0.1-0.3 rate applied to sub-layer outputs, embeddings, and positional encodings) and label smoothing (ϵls = 0.1), with inference using checkpoint averaging and beam search decoding.
  - Transformer big model outperforms all previous models by >2.0 BLEU on EN-DE and at 1/4 the training cost on EN-FR, establishing new state-of-the-art results
  - Residual dropout and label smoothing are the primary regularization techniques; label smoothing trades perplexity for improved BLEU scores
  - Inference uses checkpoint averaging (5 checkpoints for base, 20 for big) and beam search (beam size 4, length penalty α=0.6)
  - Training efficiency is exceptional: base model requires 3.3·10^18 FLOPs (EN-DE) vs 9.6·10^18 FLOPs for ConvS2S, demonstrating computational advantage

  ### Model Variations
  The Model Variations section evaluates the importance of different Transformer components through ablation studies, systematically modifying the base model to measure performance changes on English-to-German translation tasks. This empirical approach isolates the contribution of individual architectural elements and design choices to overall model performance.
  - Ablation studies modify the base Transformer model in controlled ways to assess component importance
  - Performance impact is measured on English-to-German translation benchmarks
  - This systematic evaluation validates design choices and identifies critical architectural elements for the Transformer's success

  ### English Constituency Parsing
  The English Constituency Parsing section evaluates the Transformer's generalizability beyond machine translation by applying it to constituency parsing on the Penn Treebank. The task presents unique challenges including strong structural output constraints, output longer than input, and historical difficulties for RNN models in small-data settings. The authors trained a 4-layer Transformer (dmodel=1024) on ~40K WSJ sentences and additionally in a semi-supervised setting with ~17M sentences, with minimal hyperparameter tuning beyond dropout, learning rates, and beam size.
  - Demonstrates Transformer's ability to generalize to non-translation tasks with structurally constrained outputs that exceed input length
  - Addresses a domain where RNN sequence-to-sequence models had struggled, particularly in low-data regimes
  - Employs minimal task-specific tuning, keeping most hyperparameters from the English-to-German baseline while only adjusting dropout, learning rates, and beam size
  - Evaluates both supervised (40K WSJ) and semi-supervised (17M sentence) training scenarios to test scalability

### Conclusion
The Conclusion section presents the Transformer as the first attention-based sequence transduction model, demonstrating significant advances in machine translation speed and quality. The authors achieve state-of-the-art results on WMT 2014 benchmarks (28.4 BLEU for English-German, 41.8 BLEU for English-French) while training substantially faster than recurrent or convolutional architectures, and outline future research directions including application to multimodal inputs and more efficient attention mechanisms.
- Introduces the Transformer as the first pure attention-based sequence transduction model, eliminating recurrent layers from encoder-decoder architectures
- Achieves state-of-the-art translation performance with significantly faster training compared to RNN/CNN baselines
- Demonstrates generalization beyond translation tasks with strong constituency parsing results (92.7 F1 semi-supervised)
- Future work targets multimodal extensions, local attention mechanisms for handling large inputs/outputs, and less sequential generation approaches
