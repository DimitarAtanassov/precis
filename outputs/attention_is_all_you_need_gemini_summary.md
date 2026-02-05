# Provided proper attribution is provided, Google hereby grants permission to

## Executive Summary
The paper introduces the Transformer, a novel network architecture designed to overcome the fundamental limitations of recurrent and convolutional neural networks in sequence transduction tasks. Traditional recurrent models (RNNs, LSTMs, GRUs) suffer from inherently sequential computation, which prevents parallelization during training and makes it computationally expensive to model dependencies between distant positions in a sequence. While convolutional models can parallelize, they often struggle to relate signals across long distances. The Transformer solves these challenges by eschewing recurrence and convolution entirely, relying instead on a self-attention mechanism to model global dependencies in constant time.

The proposed architecture utilizes a dual-stack encoder-decoder structure composed of identical layers. Central to its success is Multi-Head Attention, which allows the model to simultaneously attend to information from different representation subspaces at different positions. This is supported by Scaled Dot-Product Attention, which maintains numerical stability through a scaling factor, and position-wise feed-forward networks. To preserve sequence order in the absence of recurrence, the authors introduce sinusoidal positional encodings that are added to the input embeddings. This design enables the model to achieve superior parallelization and significantly shorter path lengths for long-range dependencies compared to previous state-of-the-art architectures.

The Transformer achieved record-breaking results on the WMT 2014 English-to-German and English-to-French translation tasks, establishing new state-of-the-art benchmarks (28.4 BLEU on En-De). Crucially, these results were achieved with a fraction of the training cost of previous models; the base model outperformed prior ensembles after only 12 hours of training on eight GPUs. Furthermore, the paper demonstrates the architecture's robust generalization capabilities by successfully applying it to English constituency parsing, where it performed competitively even with minimal task-specific tuning.

This work represents a paradigm shift in sequence modeling, moving the field away from sequential processing toward highly parallelizable attention-based architectures. By proving that "attention is all you need" for high-quality sequence transduction, the authors provided the foundation for the current era of large-scale language models. The Transformer’s efficiency and effectiveness in handling long-range dependencies make it a superior standard for a wide range of applications beyond machine translation, including document summarization, parsing, and generative pre-training.

## Key Contributions
- Proposes the Transformer, the first sequence transduction model to rely entirely on self-attention, eliminating recurrent and convolutional layers to allow for significantly greater parallelization.
- Introduces Multi-Head Attention, a mechanism that enables the model to jointly attend to information from different representation subspaces at different positions in parallel.
- Develops Scaled Dot-Product Attention, which utilizes a scaling factor to maintain numerical stability and gradient flow when computing compatibility between high-dimensional queries and keys.
- Utilizes sinusoidal positional encodings to inject sequence order information into the architecture, allowing the model to account for token positions without the inherent sequential nature of RNNs.
- Demonstrates that self-attention reduces the maximum path length between any two input and output positions to a constant O(1) operation, facilitating the learning of long-range dependencies.
- Establishes new state-of-the-art benchmarks in machine translation (WMT 2014 English-to-German and English-to-French) with drastically reduced training costs compared to previous recurrent and convolutional ensembles.

## Section Summaries

### Introduction
The introduction and background define the Transformer as the first transduction model to rely entirely on self-attention, eschewing traditional recurrence and convolution. By eliminating sequential dependencies in computation, the architecture allows for superior parallelization and efficient modeling of global dependencies regardless of their distance in the sequence.
- The Transformer replaces recurrent layers with a self-attention mechanism, overcoming the sequential computation bottleneck that limits parallelization in RNNs and LSTMs.
- Unlike convolutional models (e.g., ConvS2S), the Transformer relates distant positions in a constant number of operations, facilitating the learning of long-range dependencies.
- The architecture follows an encoder-decoder structure where the decoder is auto-regressive, generating output symbols one at a time based on previously generated symbols.
- The model achieves state-of-the-art results in translation quality with significantly reduced training time (e.g., twelve hours on eight GPUs).

### Background
The Background section highlights the limitations of existing recurrent and convolutional models, specifically their inability to parallelize computation and their difficulty in modeling long-range dependencies efficiently. The Transformer overcomes these constraints by relying solely on self-attention, allowing for constant-time operations for global dependencies and significantly higher training efficiency.
- Recurrent models (RNNs, LSTMs, GRUs) are limited by their inherently sequential computation, which prevents parallelization during training.
- Convolutional architectures like ConvS2S reduce sequential dependency but require an increasing number of operations to relate distant positions as sequences grow.
- The Transformer uses self-attention to relate any two positions in a constant number of operations, facilitating the learning of global dependencies regardless of distance.
- The Transformer is the first transduction model to eschew recurrence and convolution entirely in favor of an attention-only mechanism.

### Model Architecture
The Transformer is the first transduction model to rely entirely on self-attention, discarding recurrent and convolutional layers to solve the bottleneck of sequential computation. This design allows for more efficient modeling of global dependencies and reaches state-of-the-art performance with drastically reduced training time.
- The Transformer replaces the sequential nature of RNNs with a pure attention mechanism, enabling significantly higher parallelization during training.
- Unlike CNN-based models (e.g., ConvS2S), where the operations to relate distant positions grow linearly or logarithmically, the Transformer achieves this in a constant number of operations.
- The model utilizes a standard encoder-decoder structure, where the decoder generates outputs auto-regressively by consuming previously generated symbols.

  ### Encoder and Decoder Stacks
  The Transformer architecture utilizes a dual-stack design where both the encoder and decoder are composed of six identical layers featuring residual connections and layer normalization. While the encoder focuses on self-attention and feed-forward processing, the decoder introduces cross-attention to the encoder's output and employs masking to ensure predictions are based only on preceding tokens.
  - The encoder and decoder each consist of a stack of $N=6$ identical layers, using a constant dimensionality of $d_{model}=512$ for all sub-layers.
  - Each encoder layer contains two sub-layers: a multi-head self-attention mechanism and a position-wise feed-forward network, both wrapped in residual connections and layer normalization.
  - The decoder adds a third sub-layer for multi-head attention over the encoder's output and uses masking in its self-attention layer to preserve the auto-regressive property by preventing attention to future positions.

  ### Attention
  The Transformer architecture utilizes six-layer stacks for both the encoder and decoder, employing residual connections and layer normalization to stabilize training across sub-layers. While the encoder focuses on self-attention, the decoder integrates cross-attention and causal masking to ensure predictions remain auto-regressive. Central to this design is the definition of attention as a functional mapping of queries and key-value pairs into a weighted output.
  - The encoder and decoder are each composed of six identical layers featuring residual connections and layer normalization with a consistent internal dimension ($d_{model} = 512$).
  - The decoder includes an additional multi-head attention sub-layer to process the encoder's output and uses causal masking to prevent positions from attending to subsequent tokens.
  - Attention is formally defined as a mapping of a query and a set of key-value pairs to an output calculated as a weighted sum of the values.

    ### Scaled Dot-Product Attention
    Scaled Dot-Product Attention calculates the compatibility of queries and keys through a dot-product mechanism, scaled by the inverse square root of their dimension to maintain training stability. This approach provides a more space-efficient and faster alternative to additive attention while avoiding the performance degradation typically seen in unscaled dot-products at high dimensions.
    - Introduces a scaling factor of $1/\sqrt{d_k}$ to prevent the dot products from reaching large magnitudes that would result in vanishing gradients during the softmax operation.
    - Utilizes dot-product (multiplicative) attention instead of additive attention, enabling the use of highly optimized matrix multiplication for superior computational efficiency.
    - Computes attention simultaneously for a set of queries packed into a matrix $Q$, allowing for parallelized processing across the entire sequence.

    ### Multi-Head Attention
    Multi-Head Attention improves upon standard attention by linearly projecting queries, keys, and values into multiple subspaces, allowing the model to attend to diverse features in parallel. This is supported by Scaled Dot-Product Attention, which incorporates a scaling factor to ensure numerical stability and efficient computation during training.
    - Multi-Head Attention uses parallel 'heads' to attend to information from different representation subspaces simultaneously.
    - The Scaled Dot-Product Attention mechanism uses a $1/\sqrt{d_k}$ scaling factor to prevent vanishing gradients caused by large dot-product magnitudes.
    - Dot-product attention is computationally more efficient than additive attention due to optimized matrix multiplication kernels.
    - Linear projections are applied to queries, keys, and values before they are processed by the parallel attention heads.

    ### Applications of Attention in our Model
    This section details the specific deployment of multi-head attention across the model's architecture and the auxiliary components that support representation learning. It explains how masking and cross-attention facilitate sequence generation, defines the position-independent feed-forward sub-layers, and outlines the weight-sharing strategy used for embeddings and final predictions.
    - The Transformer employs three types of attention: encoder self-attention, encoder-decoder attention (linking the stacks), and masked decoder self-attention to maintain auto-regressive properties.
    - Each layer contains a position-wise feed-forward network (FFN) consisting of two linear transformations and a ReLU activation, applied independently to each position.
    - The model uses shared weight matrices between the input embeddings, output embeddings, and the pre-softmax linear transformation, scaling embedding weights by the square root of the model dimension.

  ### Position-wise Feed-Forward Networks
  This section defines the Position-wise Feed-Forward Networks and the embedding strategies used in the Transformer. The FFN processes each sequence position through a two-layer linear transformation with an internal ReLU activation, while a shared weight matrix strategy is employed across embeddings and the final output layer to improve efficiency.
  - The Position-wise Feed-Forward Network (FFN) applies two linear transformations and a ReLU activation to each position independently and identically.
  - While the transformations are consistent across positions within a layer, the parameters differ from layer to layer, functioning similarly to two convolutions with kernel size 1.
  - The architecture uses a d_model of 512 and an inner-layer dimensionality (d_ff) of 2048.
  - Weight sharing is implemented between the input/output embedding layers and the pre-softmax linear transformation, with embedding weights scaled by the square root of d_model.

  ### Embeddings and Softmax
  This section details the conversion of input and output tokens into vector representations using learned embeddings and the final prediction process via softmax. A key architectural optimization is the sharing of a single weight matrix across both embedding layers and the pre-softmax linear transformation, with embedding weights being scaled by the square root of the model dimension.
  - The model utilizes learned embeddings to represent input and output tokens in a $d_{model}$-dimensional space (512).
  - A shared weight matrix is used across both embedding layers and the pre-softmax linear transformation, reducing total parameters and improving performance.
  - Embedding weights are scaled by $\sqrt{d_{model}}$ to maintain consistency with the attention mechanisms and stabilize training.
  - A standard linear transformation followed by a softmax function is applied to the decoder output to predict next-token probabilities.

  ### Positional Encoding
  To account for token order without recurrence or convolution, the Transformer adds sinusoidal positional encodings to input embeddings, enabling the model to learn relative positions through linear transformations. An analysis of layer types reveals that self-attention minimizes sequential operations and path lengths for long-range dependencies, making it more parallelizable and effective for sequence transduction than traditional architectures.
  - Transformers utilize sinusoidal positional encodings (sine and cosine functions of varying frequencies) to inject sequence order information into the position-agnostic attention mechanism.
  - Sinusoidal encodings were chosen over learned embeddings because they allow the model to potentially extrapolate to sequence lengths longer than those seen during training.
  - Self-attention provides a constant O(1) maximum path length between any two positions, significantly improving the model's ability to learn long-range dependencies compared to recurrent (O(n)) or convolutional (O(log(n))) layers.
  - Computationally, self-attention is more efficient than recurrent layers when the sequence length (n) is smaller than the representation dimensionality (d).

### Why Self-Attention
This section justifies the use of self-attention by comparing it to recurrent and convolutional layers across three dimensions: computational complexity, parallelization, and path length for long-range dependencies. Self-attention excels by providing constant path lengths and sequential operations, while sinusoidal positional encodings are introduced to provide essential sequence order information without the need for recurrence.
- Self-attention minimizes the maximum path length between long-range dependencies to O(1), facilitating easier learning compared to the O(n) paths in recurrent layers.
- The architecture enables maximum parallelization with O(1) sequential operations, avoiding the linear computational bottleneck inherent in recurrent networks.
- Self-attention is computationally more efficient than recurrent layers when the sequence length (n) is smaller than the representation dimensionality (d).
- Sinusoidal positional encodings are utilized to inject sequence order, allowing the model to represent relative positions and extrapolate to longer sequences than those seen in training.

### Training
This section details the training configuration of the Transformer, highlighting the use of standard WMT datasets and subword tokenization. It specifies a unique learning rate schedule with warmup and decay, while theoretically justifying self-attention over convolutions based on computational complexity and path length for long-range dependencies.
- Contrasts self-attention's O(1) path length with the O(log n) or O(n/k) required by convolutional layers, noting self-attention's superior interpretability.
- Utilizes WMT 2014 English-German (4.5M pairs) and English-French (36M pairs) datasets with subword tokenization (BPE/WordPiece).
- Employs the Adam optimizer with a custom learning rate schedule featuring a 4,000-step linear warmup followed by inverse square root decay.
- Training was performed on 8 NVIDIA P100 GPUs, taking 12 hours for the base model and 3.5 days for the 'big' model.

  ### Training Data and Batching
  This section details the Transformer's training environment, specifically its use of large-scale WMT datasets and a specialized Adam-based optimization schedule. It emphasizes the computational efficiency of the architecture, which allows for relatively short training times on standard GPU hardware compared to previous state-of-the-art models.
  - Training utilized the WMT 2014 English-German (4.5M pairs) and English-French (36M pairs) datasets with subword tokenization (BPE and word-piece).
  - Implemented a custom learning rate schedule for the Adam optimizer, featuring a 4,000-step linear warmup followed by an inverse square root decay.
  - Training was highly efficient on 8 NVIDIA P100 GPUs, requiring only 12 hours for the base model and 3.5 days for the 'big' model.
  - Self-attention mechanisms are highlighted for their potential interpretability, with individual heads often attending to syntactic and semantic structures.

  ### Hardware and Schedule
  This section details the Transformer's hardware requirements and training timeline, emphasizing its computational efficiency compared to recurrent or convolutional architectures. It also introduces a specific learning rate scheduler designed to stabilize training through a warmup phase and subsequent decay.
  - Training utilized 8 NVIDIA P100 GPUs, with the base model completing 100,000 steps in 12 hours and the 'big' model 300,000 steps in 3.5 days.
  - A custom Adam optimizer schedule was implemented, featuring a linear learning rate warmup for 4,000 steps followed by an inverse square root decay.
  - The model achieves superior efficiency compared to convolutions, as self-attention provides constant path lengths while maintaining a computational complexity similar to separable convolutions.

  ### Optimizer
  The Transformer utilizes the Adam optimizer with a specialized learning rate schedule designed to stabilize training. This schedule increases the learning rate linearly during an initial warmup phase before decreasing it proportionally to the inverse square root of the step number, ensuring efficient convergence across different model scales.
  - Employs the Adam optimizer with specific hyperparameters: β1 = 0.9, β2 = 0.98, and ε = 10^-9.
  - Implements a custom learning rate schedule that features a linear warmup for the first 4,000 steps followed by an inverse square root decay.
  - The learning rate is scaled by the inverse square root of the model dimensionality (d_model).

  ### Regularization
  The Transformer architecture leverages self-attention to achieve superior computational efficiency and shorter dependency path lengths compared to recurrent and convolutional layers. The training regime involves high-performance hardware, large-scale translation datasets, and a specialized learning rate schedule designed to stabilize convergence through an initial warmup phase.
  - Self-attention provides a constant path length of O(1) between any two positions, outperforming the logarithmic or linear scaling found in convolutional layers.
  - Training utilizes large-scale WMT datasets with subword tokenization (BPE and word-piece) and batches organized by token count (approx. 25,000 tokens per batch).
  - The model employs a specific Adam-based learning rate schedule that features a 4,000-step linear warmup followed by an inverse square root decay.
  - Self-attention distributions are shown to be interpretable, often capturing complex syntactic and semantic structures across heads.

### Results
The Transformer architecture establishes new state-of-the-art benchmarks on WMT 2014 translation tasks, specifically surpassing previous recurrent and convolutional ensembles in both accuracy and training efficiency. By achieving superior BLEU scores with significantly lower computational costs, the results validate the effectiveness of the self-attention mechanism and the specific regularization strategies employed.
- The Transformer (big) achieved a new state-of-the-art BLEU score of 28.4 on WMT 2014 English-to-German, outperforming previous ensembles by over 2.0 points.
- Training efficiency improved significantly, with the Transformer reaching SOTA results at a fraction of the FLOPs required by models like GNMT or ConvS2S.
- Regularization via 10% dropout and label smoothing ($\epsilon_{ls} = 0.1$) was critical for performance, with the latter improving BLEU scores despite increasing model uncertainty.
- Final performance was boosted by averaging multiple checkpoints (5 for base, 20 for big) and utilizing beam search with a length penalty during inference.

  ### Machine Translation
  The Transformer achieves record-breaking performance on WMT 2014 English-to-German and English-to-French tasks, outperforming existing benchmarks while being significantly faster to train. By utilizing residual dropout, label smoothing, and efficient beam search, the model provides a new standard for translation quality and computational economy.
  - The Transformer (big) model established a new state-of-the-art BLEU score of 28.4 on WMT 2014 English-to-German, surpassing previous ensembles by over 2.0 BLEU.
  - The architecture demonstrates superior training efficiency, achieving these results at a fraction of the floating-point operations (FLOPs) required by previous models like GNMT or ConvS2S.
  - Regularization techniques included a 0.1 dropout rate on sub-layers and embeddings, alongside label smoothing, which improved accuracy despite increasing model uncertainty.
  - Inference utilized beam search (size 4) and checkpoint averaging to further refine translation quality.

  ### Model Variations
  This section presents empirical results demonstrating that the Transformer architecture outperforms existing recurrent and convolutional models in both translation quality and computational efficiency. It also details the specific regularization techniques—residual dropout and label smoothing—and inference configurations that contribute to its state-of-the-art performance.
  - The Transformer (big) achieved a new state-of-the-art BLEU score of 28.4 on EN-DE and 41.8 on EN-FR, significantly outperforming previous ensembles at a fraction of the training cost.
  - Residual dropout is applied to sub-layer outputs and embedding-positional encoding sums, while label smoothing ($\epsilon = 0.1$) is used to improve accuracy and BLEU at the expense of perplexity.
  - Model performance was optimized through checkpoint averaging (last 5 for base, last 20 for big) and beam search with a beam size of 4 and a length penalty of 0.6.

  ### English Constituency Parsing
  This section evaluates the Transformer’s ability to generalize beyond translation by applying it to English constituency parsing, where it manages complex structural constraints and long sequences. Through empirical variations, the authors demonstrate that the architecture's success is driven by multi-head attention and model scale, while maintaining performance even when switching between fixed and learned positional embeddings.
  - The Transformer generalizes effectively to English constituency parsing, handling structural constraints and long output sequences despite minimal task-specific tuning.
  - Architectural variations indicate that multi-head attention is superior to single-head attention, and larger models combined with dropout significantly improve performance.
  - Sinusoidal positional encodings perform nearly identically to learned positional embeddings, suggesting the fixed approach is robust.
  - Experiments on the Wall Street Journal dataset (40K sentences) and a semi-supervised corpus (17M sentences) demonstrate the model's capability in both small-data and large-scale regimes.

### Conclusion
The authors conclude that the Transformer’s reliance on self-attention allows for superior performance and training efficiency compared to traditional architectures. Its ability to generalize across different tasks, such as translation and parsing, combined with its capacity to learn complex structural relationships, establishes it as a robust new standard for sequence modeling.
- The Transformer is the first sequence transduction model built entirely on multi-headed self-attention, successfully replacing recurrent and convolutional layers.
- The architecture achieves state-of-the-art results on WMT 2014 translation tasks while being significantly faster to train than previous models.
- The model demonstrates strong generalization by outperforming most specialized systems in English constituency parsing, even without task-specific tuning.
- Attention visualizations reveal that individual heads learn to resolve specific linguistic structures, including long-distance dependencies and anaphora.
