# 1. 1 Introduction

## Executive Summary
## Executive Summary: Recursive Language Models (RLMs)

### Problem
Large language models (LLMs) face a fundamental limitation: restricted context windows and "context rot"—the degradation of model quality as context length increases, even in frontier models like GPT-5. While architectural improvements are expanding context lengths, this problem is increasingly critical as LLMs are deployed for long-horizon tasks requiring processing of tens to hundreds of millions of tokens. Existing solutions either rely on training-time architectural modifications or fail when the input itself exceeds context limits. Moreover, effective context handling depends on task complexity, not just length—tasks where complexity scales with input size (e.g., requiring examination of nearly every token) experience degradation at much shorter lengths than simple needle-in-a-haystack retrieval.

### Proposed Approach
This paper introduces Recursive Language Models (RLMs), a novel inference-time scaling strategy that enables LLMs to process prompts significantly longer than their trained context windows. RLMs treat long prompts as external environments that the model can programmatically interact with through a Read-Eval-Print Loop (REPL). The LLM writes code to examine, decompose, and recursively call itself (or a sub-LM) on prompt snippets, with the full context stored as variables rather than loaded into the context window. Drawing inspiration from out-of-core algorithms, this approach fundamentally differs from prior recursive decomposition methods by allowing the input itself to scale beyond the context window, not just the task decomposition. The implementation uses a Python REPL environment where GPT-5 or Qwen3-Coder-480B serve as the root model, with GPT-5-mini handling recursive calls for cost efficiency.

### Key Results and Contributions
RLMs demonstrate the ability to scale to 10M+ token inputs—up to 100x beyond model context limits—while outperforming base LLMs by up to 2× on performance metrics at comparable or lower costs:

- **Ultra-long contexts**: On BrowseComp-Plus (6-11M tokens, 1000 documents), RLM(GPT-5) achieves near-perfect performance while base GPT-5 shows clear degradation, at up to 3× lower cost than summarization baselines
- **Information-dense tasks**: On OOLONG-Pairs (quadratically scaling complexity), RLMs achieve 58% F1 where base GPT-5 achieves <0.1%, with recursive sub-calling providing 10-59% improvements
- **Graceful degradation**: RLM performance degrades more slowly than base models as input length increases, especially for complex tasks beyond 2^14 tokens
- **Cost efficiency**: Median inference costs remain comparable to base models despite high variance due to adaptive processing; costs scale log-linearly with input size

The work introduces three benchmark tasks (S-NIAH, BrowseComp-Plus, OOLONG-Pairs) designed to evaluate different complexity-to-length scaling relationships. Analysis reveals emergent behaviors including intelligent input filtering—RLMs can filter massive contexts without reading everything by using code execution, regex searches, and model priors—explaining how they scale to huge inputs without proportional cost increases.

### Significance
This work demonstrates that inference-time scaling can dramatically extend context handling by orders of magnitude without waiting for training-time architectural solutions. RLMs address the critical limitation that prior approaches fail when inputs themselves exceed context windows, enabling LLMs to handle real-world long-horizon tasks involving tens to hundreds of millions of tokens. The framework is model-agnostic (working with both GPT-5 and Qwen3-Coder) and task-agnostic (using fixed prompts across diverse tasks). By offloading context to an external environment and enabling programmatic, recursive processing, RLMs represent a new paradigm for reasoning that could be further improved through explicit training. The approach has immediate practical value for applications requiring massive document corpus analysis, large codebase understanding, and complex multi-hop reasoning over extensive contexts, while maintaining reasonable computational costs.

## Key Contributions
- Introduction of Recursive Language Models (RLMs), a novel inference-time scaling strategy that enables LLMs to process prompts up to 100× beyond their trained context windows by treating long prompts as external environments accessible through programmatic interaction
- A Read-Eval-Print Loop (REPL)-based framework where LLMs can write code to examine, decompose, and recursively call themselves on prompt snippets, addressing the fundamental limitation that the input itself (not just task decomposition) can scale beyond context windows
- Demonstration that RLMs outperform both base LLMs and existing long-context methods across four diverse tasks at comparable or lower cost, achieving up to 2× performance improvements while maintaining cost efficiency
- A task characterization framework arguing that effective context window cannot be understood independently of task complexity, with evidence that complex problems exhibit degradation at shorter context lengths than simpler ones
- Three benchmark tasks with different information density patterns: S-NIAH (constant complexity), BrowseComp-Plus (moderate complexity with multi-hop reasoning over 1K documents at 6-11M tokens), and OOLONG-Pairs (quadratic complexity requiring pairwise aggregation)
- Empirical evidence that RLMs scale to 10M+ token inputs while achieving 58% F1 on information-dense tasks where base GPT-5 achieves <0.1%, with performance degradation that is slower than base models as input length increases beyond 2^14 tokens
- Discovery of emergent intelligent input filtering behavior where RLMs filter massive contexts without explicitly reading everything by using code execution, regex searches, and model priors to process fewer tokens while maintaining performance
- Demonstration of model-dependent context management strategies: RLM(GPT-5) nearly solves all BrowseComp-Plus tasks while RLM(Qwen3-Coder) struggles with half despite nearly identical system prompts, revealing distinct emergent behaviors across different base models
- A hierarchical approach using GPT-5-mini for recursive calls and GPT-5 as root model to balance capability and cost, with inference costs remaining comparable to base models in median cases and up to 3× cheaper than summarization baselines
- Evidence that the REPL environment with recursive sub-calling provides 10-59% improvements on information-dense tasks compared to ablations without recursion, validating both components of the approach

## Section Summaries
### Recursive Language Models
This paper introduces Recursive Language Models (RLMs), a novel inference strategy that enables LLMs to process prompts significantly longer than their trained context windows. RLMs treat long prompts as external environments that the model can programmatically examine, decompose, and recursively process in smaller snippets. The approach successfully handles inputs up to 100x beyond model context limits and outperforms both base LLMs and existing long-context methods across four diverse tasks at comparable or lower cost. The work addresses the critical "context rot" problem where model quality degrades with increasing context length, even in frontier models like GPT-5, which is increasingly important as LLMs are deployed for long-horizon tasks requiring processing of tens to hundreds of millions of tokens.
- Proposes Recursive Language Models (RLMs) as an inference-time scaling strategy for handling arbitrarily long prompts
- RLMs treat long prompts as external environments and allow programmatic examination, decomposition, and recursive self-calling over prompt snippets
- Successfully processes inputs up to 2 orders of magnitude beyond model context windows
- Outperforms base LLMs and common long-context scaffolds across four diverse tasks at comparable or lower cost
- Addresses 'context rot' phenomenon where model quality degrades as context length increases
- Motivated by need to process tens to hundreds of millions of tokens in long-horizon LLM applications

  ### Abstract
  The Introduction motivates the work by highlighting the fundamental limitation of LLMs: restricted context windows and inevitable "context rot" where model quality degrades as context length increases, even in frontier models like GPT-5. While context lengths are expected to grow through architectural and infrastructure improvements, the authors propose an inference-time scaling approach to dramatically extend context handling by orders of magnitude. This is critical as LLMs are increasingly deployed for long-horizon tasks requiring processing of tens to hundreds of millions of tokens. The paper frames the problem of handling arbitrarily long prompts through the lens of inference-time scaling rather than waiting for training-time solutions.
  - Modern LLMs suffer from limited context lengths and context rot, where quality degrades as context increases, even in GPT-5
  - The paper approaches long-context processing through inference-time scaling rather than architectural/training improvements
  - Motivation driven by LLM deployment in long-horizon tasks requiring processing of tens to hundreds of millions of tokens
  - Goal is to scale context size of general-purpose LLMs by orders of magnitude

### 1 Introduction
The Introduction motivates Recursive Language Models (RLMs) as an inference-time scaling solution to LLMs' fundamental limitation: restricted context windows and "context rot" where quality degrades with increasing context length, even in frontier models like GPT-5. Rather than waiting for training-time solutions to expand context lengths, RLMs treat long prompts as external environments that the LLM can programmatically interact with through a Read-Eval-Print Loop (REPL). The approach draws inspiration from out-of-core algorithms and enables processing of prompts by allowing the LLM to write code that examines, decomposes, and recursively calls itself on prompt snippets. This design addresses a key limitation of prior recursive decomposition methods: the input itself can now scale beyond the context window, not just the task decomposition. RLMs successfully handle inputs up to 100x beyond model context limits while outperforming base LLMs and existing long-context approaches across diverse tasks at comparable or lower cost.
- Identifies 'context rot' as a fundamental LLM limitation where quality degrades with increasing context length, even in GPT-5
- Frames the problem through inference-time scaling rather than waiting for training-time architectural improvements
- Introduces RLMs which treat long prompts as external environments accessible via REPL programming interface
- Draws inspiration from out-of-core algorithms that process datasets larger than main memory
- Key innovation: prompts are not fed directly into the neural network but accessed symbolically through code
- LLM writes code to examine, decompose prompts and recursively invoke itself on sub-tasks
- Addresses limitation of prior recursive approaches which couldn't scale inputs beyond context windows
- Achieves 100x extension beyond context limits with superior quality at comparable cost

### 2 Scaling Long Context Tasks
This section argues that effective context window cannot be understood independently of task complexity. The authors hypothesize that more complex problems exhibit degradation at shorter context lengths than simpler ones, requiring task characterization based on how complexity scales with prompt length. They contrast needle-in-a-haystack (NIAH) tasks—where frontier models succeed even at 1M+ tokens because "needles" remain constant—with tasks like OOLONG where answers depend on almost every line, causing models to struggle at shorter lengths. This task-dependent degradation explains observed patterns where GPT-5 scales well on constant-complexity tasks (S-NIAH) but fails on tasks where complexity grows with context length.
- Effective context window depends on task complexity, not just model capacity
- More complex problems degrade at shorter context lengths than simpler ones
- NIAH tasks maintain constant needle size, allowing frontier models to succeed at 1M+ tokens
- OOLONG requires processing nearly every line, causing failure at shorter lengths
- Task complexity must be characterized by how it scales with prompt length

  ### 2.1 Tasks
  This section describes three benchmark tasks designed to evaluate model performance across different information density patterns. The tasks vary both prompt length and problem complexity scaling: (1) S-NIAH (Single Needle-in-a-Haystack) requires finding one specific phrase/number in large texts, with constant processing cost regardless of input size; (2) BrowseComp-Plus uses 150 multi-hop QA tasks over 1K documents that require reasoning across multiple documents to piece together answers, making it more complex than S-NIAH despite constant document requirements; (3) OOLONG tests long reasoning by requiring semantic examination and transformation of input chunks followed by aggregation, with scoring using exponential decay (0.75^|y-ŷ|) for numerical answers and exact match for others. The task design enables systematic evaluation of how models handle different complexity-to-length scaling relationships.
  - Three tasks designed with varying information density patterns: constant (S-NIAH), linear (OOLONG), and multi-document reasoning (BrowseComp-Plus)
  - S-NIAH: 50 single needle-in-haystack tasks with constant complexity regardless of input length
  - BrowseComp-Plus: 150 multi-hop QA tasks over 1K documents from 100K document corpus, requiring cross-document reasoning
  - OOLONG: Long reasoning benchmark requiring semantic transformation and aggregation with exponential scoring (0.75^|y-ŷ|)
  - Task characterization focuses on information density - how much processing is required relative to input size

  ### 2.2 Methods and Baselines
  This section details the experimental setup, introducing additional task variants and comparing Recursive Language Models (RLMs) against baselines. OOLONG-Pairs extends the trec_coarse split with 20 new queries requiring pairwise chunk aggregation, scaling quadratically with input length. LongBench-v2 CodeQA tests code repository understanding with fixed-file reasoning. The evaluation uses two frontier models: GPT-5 (medium reasoning) and Qwen3-Coder-480B-A35B. The RLM implementation uses a Python REPL environment where context is loaded as a string and a sub-LM can be queried; for GPT-5 experiments, GPT-5-mini handles recursive calls while GPT-5 serves as the root model to balance capability and cost.
  - OOLONG-Pairs adds 20 manually created queries requiring quadratic-scaling pairwise chunk aggregation to the trec_coarse split
  - LongBench-v2 CodeQA provides multi-choice code understanding tasks challenging for frontier models with fixed-file reasoning requirements
  - Evaluation compares RLMs against baselines using GPT-5 (medium reasoning) and open-source Qwen3-Coder-480B-A35B
  - RLM implementation uses Python REPL environment with context loaded as string and sub-LM query capability
  - GPT-5 experiments use GPT-5-mini for recursive calls and GPT-5 as root model to optimize capability-cost tradeoff

### 3 Results and Discussion
This section presents experimental results comparing Recursive Language Models (RLMs) against baselines across multiple long-context benchmarks. RLMs demonstrate the ability to scale to 10M+ token inputs while outperforming base models by up to 2× on performance metrics at comparable or lower costs. Key findings include: (1) RLMs excel on both ultra-long contexts (BrowseComp-Plus at 6-11M tokens) and information-dense tasks (OOLONG-Pairs), achieving 58% F1 where base GPT-5 achieves <0.1%; (2) The REPL environment enables scaling beyond context limits, while recursive sub-calling provides 10-59% improvements on information-dense tasks; (3) RLM performance degrades more slowly than base models as input length increases, especially for complex tasks beyond 2^14 tokens; (4) RLM inference costs remain comparable to base models in median cases but show high variance due to adaptive iteration lengths, and are up to 3× cheaper than summarization baselines while maintaining superior performance.
- RLMs scale to 10M+ tokens and outperform base models by up to 2× on long-context tasks at comparable/lower costs
- On information-dense OOLONG-Pairs task, RLMs achieve 58% F1 (GPT-5) and 23% F1 (Qwen3-Coder) vs <0.1% for base models
- REPL environment enables beyond-context-limit scaling; recursive sub-calling provides 10-59% gains on information-dense tasks
- RLM performance degrades slower than base models as context grows, especially beyond 2^14 tokens and on complex tasks
- Median RLM costs comparable to base models but with high variance; up to 3× cheaper than summarization baselines

  ### 3.1 Emergent Patterns in RLM Trajectories
  This section analyzes emergent behaviors in RLM trajectories that arise without explicit training. A key finding is that RLMs exhibit model-dependent behavior (Observation 5): while both GPT-5 and Qwen3-Coder perform well as RLMs relative to baselines, they show distinct context management strategies—RLM(GPT-5) nearly solves all BrowseComp-Plus tasks while RLM(Qwen3-Coder) struggles with half, despite using nearly identical system prompts. The main emergent pattern discussed is intelligent input filtering: RLMs can filter massive input contexts without explicitly reading everything by using code execution and model priors. For example, RLM(GPT-5) employs regex searches for relevant keywords from the prompt and leverages prior knowledge to narrow the search space, processing fewer tokens while maintaining performance—explaining how RLMs scale to huge inputs without proportional cost increases.
  - RLMs show model-dependent behavior: GPT-5 and Qwen3-Coder exhibit different context management strategies despite identical prompts
  - RLM(GPT-5) nearly solves all BrowseComp-Plus tasks while RLM(Qwen3-Coder) solves only half
  - GPT-5 is conservative with sub-calls while Qwen3-Coder makes more granular sub-queries
  - Emergent pattern: RLMs filter input context using code execution without explicitly reading all content
  - Model priors enable intelligent search space narrowing (e.g., using regex for relevant keywords)
  - This filtering mechanism explains RLMs' ability to handle massive inputs cost-effectively

### 4 Related Works
This section begins to review related work on long-context language model systems, identifying two primary approaches: (1) architectural modifications and retraining of base models to handle longer contexts, and (2) alternative methods not yet specified in the provided excerpt. The section cites Press et al. (2022) as an example of the architectural retraining approach.
- Two main orthogonal directions exist for long-context management in LM systems
- First direction: directly modifying architecture and retraining base LMs for longer contexts (e.g., Press et al., 2022)
- Section appears incomplete in the provided excerpt

### 5 Limitations and Future Work
This section discusses two main limitations and future research directions for RLMs. First, while RLMs demonstrate strong performance on long-context tasks beyond base model context window limits with reasonable inference costs, the optimal implementation mechanism remains under-explored. The current work uses synchronous sub-calls within a Python REPL environment, but alternative approaches could improve efficiency. Specifically, the authors suggest that asynchronous sub-calls and sandboxed REPLs could potentially reduce both runtime and inference costs significantly.
- RLMs achieve strong performance on tasks exceeding context window limits at reasonable inference costs
- The optimal mechanism for implementing RLMs is still under-explored
- Current implementation uses synchronous sub-calls in a Python REPL environment
- Future work could explore asynchronous sub-calls and sandboxed REPLs to reduce runtime and inference costs

### 6 Conclusion
This section concludes the paper on Recursive Language Models (RLMs), a novel inference framework that offloads input context to enable language models to recursively sub-query other LMs before generating outputs. The implementation uses a Python REPL environment where context is stored as variables, allowing the model to reason using code and recursive calls rather than purely in token space. Results demonstrate RLMs are effective for both long-context tasks and general reasoning across multiple settings and models. The authors identify three key areas for future work: (1) exploring alternative implementation mechanisms beyond synchronous sub-calls, (2) investigating deeper recursion beyond the max depth of one used in current experiments, and (3) explicitly training models to function as RLMs, hypothesizing that RLM trajectories can be viewed as a form of reasoning that could be improved through bootstrapping from existing frontier models. The authors express excitement about future scaling potential through explicit RLM training.
- Introduces Recursive Language Models (RLMs) as a general inference framework that offloads context and enables recursive sub-querying
- Implementation uses Python REPL environment to store context as variables, enabling reasoning in code rather than token space
- Demonstrates effectiveness across multiple settings for long-context problems and general reasoning
- Current implementation limited to max recursion depth of one (sub-calls are standard LMs)
- Future work directions: asynchronous sub-calls, deeper recursion layers, and explicit training of models as RLMs
- Hypothesizes RLM trajectories represent a form of reasoning that could be trained via bootstrapping from frontier models

### Acknowledgments
This acknowledgments section recognizes research support from the Laude Institute and thanks multiple individuals and research groups for their contributions. The authors specifically thank Noah Ziems, Jacob Li, James Moore, and members of MIT OASYS and MIT DSG labs for discussions during the project, as well as Matej Sirovatka, Ofir Press, Sebastian Müller, Simon Guo, and Zed Li for providing helpful feedback.
- Research partially supported by the Laude Institute
- Collaborators from MIT OASYS and MIT DSG labs contributed through discussions
- Multiple individuals provided feedback on the work

### References
This References section contains a comprehensive bibliography of 30+ citations supporting the Recursive Language Models (RLMs) paper. Key referenced work includes: recent long-context benchmarks (LongBench v2, Oolong, BrowseComp-plus), reasoning models (DeepSeek-R1, OpenAI o1, GPT-5), agent frameworks and memory systems (Anthropic Claude subagents, MemGPT, Mem0, THREAD), context management approaches (context condensation, ReSum, context-folding), code-based reasoning systems (SWE-bench, executable code actions), and foundational technical papers on efficient sequence modeling and multi-hop reasoning. The references span work from 2018-2025, with heavy emphasis on 2024-2025 publications, reflecting the cutting-edge nature of long-context processing, agentic systems, and reasoning capabilities in large language models.
- Includes citations for major long-context evaluation benchmarks (LongBench v2, Oolong, RULER, BrowseComp-plus)
- References recent reasoning models including DeepSeek-R1, OpenAI o1, and GPT-5 system cards
- Cites related agent and memory system work (Claude subagents, MemGPT, Mem0, THREAD, ROMA)
- Includes context management techniques (context condensation, context-folding, ReSum)
- References code-based reasoning and execution frameworks (SWE-bench, executable code actions, ViperGPT)
- Covers foundational work on efficient sequence modeling and multi-hop reasoning (Baleen, Infini-attention, structured state spaces)
- Primarily draws from 2024-2025 publications, indicating recent developments in the field

### Appendix A Negative Results: Things we Tried that Did Not Work.
This appendix documents unsuccessful approaches and implementation challenges encountered during RLM development. Key failures include: (1) using identical system prompts across different models led to undesirable behavior—Qwen3-Coder required prompt modifications to prevent excessive recursive calls; (2) smaller models like Qwen3-8B lacked sufficient coding capabilities for the REPL-based RLM framework; (3) thinking models with limited output token budgets (e.g., Qwen3-235B-A22B) experienced trajectory truncation due to thinking tokens consuming the maximum output length, limiting performance gains; (4) synchronous/blocking LM calls caused significant slowdowns compared to base models; (5) the FINAL()/FINAL_VAR() tagging mechanism for distinguishing final answers from intermediate thoughts proved brittle, causing models to incorrectly output plans as final answers. The authors suggest future model training specifically for RLM behavior could resolve some issues.
- Model-specific system prompt engineering is necessary—identical prompts cause divergent behavior across models
- Smaller models without strong coding abilities fail in REPL-based RLM implementations
- Output token limits in thinking models cause trajectory truncation and performance degradation
- Sequential LM calls create significant latency; asynchronous implementation needed
- Structured output tags (FINAL/FINAL_VAR) for answer detection are brittle and cause erroneous outputs
- Future RLM-specific model training could address structural output issues

### Appendix B Additional RLM Trajectories
Appendix B presents example execution trajectories of frontier language models operating as Recursive Language Models (RLMs) to illustrate their qualitative behavior. The trajectories are too long to include fully in the paper but are available in the codebase with a visualizer. Key observations reveal RLMs often make suboptimal decisions despite strong quantitative performance: Qwen3-Coder sometimes constructs answers through recursive calls and code execution but then discards this information and wastes additional sub-calls; different models show dramatically different calling patterns (Qwen3-Coder makes hundreds-to-thousands of recursive calls for simple tasks while GPT-5 makes only ~10). Example B.1 shows RLM(GPT-5) solving a BrowseComp-Plus multi-hop query task requiring finding answers from 1000 documents (8.3M tokens) at a cost of $0.079. These qualitative insights aim to inform future RLM improvements.
- Provides example execution trajectories of frontier models as RLMs with visualizations in codebase
- RLMs often make non-optimal choices despite strong benchmark results
- Qwen3-Coder sometimes constructs answers then discards information and wastes sub-calls
- Dramatic differences in calling patterns: Qwen3-Coder makes 100s-1000s of calls vs GPT-5 making ~10 for same tasks
- Example B.1: GPT-5 solving multi-hop query over 1000 documents (8.3M tokens) for $0.079
- Qualitative analysis intended to guide future RLM improvements

  ### B.1 RLM(GPT-5) on BrowseComp-Plus-Query_74
  This section presents example execution trajectories of frontier language models operating as Recursive Language Models (RLMs) to illustrate their qualitative behavior. The trajectories demonstrate that RLMs often make suboptimal decisions despite strong quantitative performance. Specifically, Example B.1 shows RLM(GPT-5) solving a BrowseComp-Plus multi-hop query task that requires finding answers from 1000 documents (8.3M tokens total) at a cost of $0.079. The task involves a complex multi-hop query about a vegetable stew and a related township celebration.
  - RLMs often make non-optimal choices despite strong quantitative results, such as Qwen3-Coder constructing answers through recursive calls and code execution but then discarding the information and wasting additional sub-calls
  - Different models exhibit dramatically different calling patterns: Qwen3-Coder makes hundreds-to-thousands of recursive calls for simple tasks while GPT-5 makes only ~10
  - Example B.1 demonstrates RLM(GPT-5) solving a BrowseComp-Plus task with 1000 documents (8.3M tokens) at $0.079 cost
  - The qualitative insights are intended to inform future RLM improvements
  - Full trajectories and a visualizer are provided in the codebase due to their length

  ### B.2 RLM(Qwen3-Coder) on OOLONG-Pairs-Query_3
  This section presents an execution trajectory of RLM(Qwen3-Coder) solving an OOLONG-Pairs task at a cost of $1.12. The task requires identifying all pairs of user IDs meeting specific criteria from a dense 32k token input with long output requirements. The model initially employs an effective programmatic strategy: it probes the data format, semantically classifies chunks using recursive sub-LM calls to avoid context rot, then programmatically identifies qualifying users and forms unique pairs—successfully deriving the correct answer. However, Qwen3-Coder exhibits a critical inefficiency characteristic observed across trajectories: despite having the correct answer stored in a variable, it repeatedly re-verifies and regenerates the answer 5 times using additional sub-LM calls before finally returning. This behavior likely stems from the model not being specifically trained for the RLM paradigm and prompt misalignment (failed FINAL_VAR() tag acceptance), illustrating the suboptimal decision-making patterns that RLMs display despite strong quantitative performance.
  - Task: Extract all user ID pairs meeting specific label criteria from 32k token dense dataset
  - Cost: $1.12 for trajectory execution
  - Model uses effective initial strategy: data probing → semantic classification via sub-LM calls → programmatic pair extraction
  - Successfully derives correct answer and stores in variable
  - Critical inefficiency: Model repeats answer generation process 5 times despite already having correct result
  - Demonstrates Qwen3-Coder tendency to output multiple code blocks per step and excessively verify answers
  - Suboptimal behavior attributed to lack of RLM-specific training and prompt tuning

  ### B.3 RLM(Qwen3-Coder) on OOLONG-Query_212
  This section presents an execution trajectory of RLM(Qwen3-Coder) on an OOLONG aggregate query task that cost $0.38. The task requires classifying thousands of questions into 6 semantic categories (numeric value, entity, location, description and abstract concept, abbreviation, human being) and comparing the frequency of two specific categories. Unlike rule-based approaches, this requires semantic transformations that cannot be done programmatically. The model begins by probing the input context with code snippets to explore the data format. A behavioral difference is noted: Qwen3-Coder-480B-A35B tends to output multiple code blocks in a single step, unlike GPT-5 which operates more iteratively. The previous context mentions a case where the RLM failed by generating a wrong answer and not returning the correct answer it had built up in its code environment.
  - Task involves semantic classification of questions into 6 categories and frequency comparison at $0.38 cost
  - Requires semantic transformations that cannot be performed with rule-based syntax rules
  - Qwen3-Coder exhibits different execution style than GPT-5: outputs multiple code blocks per step vs. iterative approach
  - Model starts by probing context with exploratory code snippets
  - Previous example showed RLM failure where correct answer was computed but wrong answer was returned

  ### B.4 RLM(GPT-5) on CodeQA-Query_44
  This section presents an execution trajectory of RLM(GPT-5) on a CodeQA task costing $0.27. The task requires the model to answer a multiple-choice question about a large codebase (~900k tokens) used for fine-tuning text-to-image models and training LoRA models. The query presents 4 different statements (0-3) about the repository's training process, dataset preparation, annotation tools, GPU support, and execution methods, and asks the model to identify which statement is correct based on the stored code context. The section appears to continue beyond what's shown, as statement 3 is incomplete.
  - Task involves understanding a 900k token codebase for text-to-image model fine-tuning and LoRA training
  - Total execution cost was $0.27
  - Model must select one correct statement from 4 choices about repository functionality
  - Statements cover: parallel processing architecture, dataset preparation, annotation tools with Florence model and multi-GPU support, and UI/terminal execution methods
  - Demonstrates RLM's ability to comprehend and reason over large-scale code repositories

### Appendix C Additional Runtime and Cost Analysis of RLMs
This appendix provides supplementary runtime and cost analysis for Reasoning Language Models (RLMs). It includes fine-grained visualizations through histograms (Figures 7, 8) showing the cost distribution of different methods across tasks for both GPT-5 and Qwen3-Coder implementations. The analysis reveals that RLMs exhibit long-tailed, high-variance cost distributions across both models. The appendix also presents log-scaled runtime plots for each method, with a note that runtime performance could be significantly improved through asynchronous LM calls and prompt engineering to limit lengthy sub-LM calls or code generation. Average API cost per task data is provided to supplement the scaling analysis from Figure 1.
- Provides fine-grained cost histograms for RLM methods on GPT-5 and Qwen3-Coder across all tasks
- Observes long-tailed, high-variance cost trajectories for RLMs in both models
- Includes log-scaled runtime plots for each method
- Notes potential for significant runtime improvements through asynchronous LM calls and prompt optimization
- Supplements main scaling analysis with average API cost per task data

### Appendix D Additional Methods and Baseline Details
This section (Appendix D) primarily describes additional methodological details and baselines, beginning with D.1 on experimental prompts. The preceding content shows the conclusion of an RLM execution example where the model successfully solves a CodeQA task by partitioning a large codebase (~900k tokens), recursively sub-querying language models on each partition to gather clues, then aggregating results to arrive at the correct answer (choice '1'). The approach demonstrates that for tasks with low information density, decomposition and recursive querying can be effective strategies.
- Appendix D provides additional methods and baseline details for the experiments
- Includes D.1 subsection on prompts used in experiments
- Demonstrates successful RLM strategy of partitioning large codebases and using recursive sub-queries for low information-density tasks
- Example shows correct answer achieved through aggregation of clues from sub-queries

  ### D.1 Prompts for Experiments
  Section D.1 presents the complete system prompts used for experimental methods in the paper. The authors emphasize task-agnostic prompts that remain fixed across all tasks. For RLMs, the main difference between GPT-5 and Qwen3-Coder implementations is an added warning for Qwen3-Coder to limit sub-LM calls, as without this constraint, Qwen3-Coder tends to make thousands of unnecessary subcalls even for basic tasks. The section provides four main prompt variants: (1a) RLM with REPL for GPT-5, which includes detailed instructions for using recursive LLM queries with a REPL environment to handle large contexts (up to 500K chars per sub-LLM), multiple code examples demonstrating chunking strategies, and final answer formatting with FINAL() or FINAL_VAR(); (1b) the Qwen3-Coder variant adding explicit batching guidance to aim for ~200k characters per call; (2) RLM with REPL (no sub-calls) that removes the llm_query function; and (3a-3b) CodeAct prompts with and without BM25 retrieval, following the original CodeAct paper's methodology, which use a THINK-ACT loop format with Python code execution capabilities.
  - All methods use task-agnostic prompts fixed across all tasks
  - Qwen3-Coder requires explicit warnings to prevent excessive sub-LM calls (thousands for basic tasks without constraint)
  - RLM prompts provide detailed chunking strategies with examples for handling large contexts (500K chars per sub-query)
  - Four main prompt variants: RLM with REPL (GPT-5), RLM with REPL (Qwen3-Coder with batching guidance), RLM without sub-calls, and CodeAct with/without BM25
  - CodeAct follows original paper methodology with THINK-ACT loop and Python execution
  - Prompts include specific formatting requirements (FINAL/FINAL_VAR for RLM, ANSWER for CodeAct)

  ### D.2 Summary agent baseline
  Section D.2 describes a "Summary agent baseline" used for comparison in the paper's experiments. The content shown appears to be the tail end of a CodeAct-style prompt (Section D.1) that demonstrates a THINK-ACT workflow with Python code execution. The example shows counting words with exactly 2 r's using regex, emphasizing strategic reasoning to avoid context window limits, explicit code execution with all necessary imports and data included directly (no placeholders), and the ability to answer simple questions without code when appropriate. Variables persist across executions but each code block must be self-contained. The actual D.2 baseline description follows this prompt example but is not included in the provided content.
  - Section D.2 introduces a 'Summary agent baseline' for experimental comparison
  - The shown content is the conclusion of a CodeAct prompt example from D.1
  - Example demonstrates THINK-ACT pattern with Python code execution for counting words with specific character patterns
  - Key requirements: start with THINK step, avoid context window limits, include all code context explicitly (no placeholders like 'FILL IN WITH REAL DATA')
  - Code blocks must be self-contained with imports and data, though variables persist across executions
  - Simple factual questions can be answered directly without code execution

### Appendix E Additional Benchmark Details
Appendix E provides additional details on benchmarks used to evaluate RLMs. Section E.1 describes OOLONG-Pairs, a synthetic benchmark created from OOLONG (Bertsch et al., 2025) with 20 new tasks based on the trec_coarse split across 11 different input context lengths (ranging from 1024 to 1,048,576 tokens). Each question requires correctly predicting semantic mappings for each entry. To ensure quadratic information density (I ≈ O(N²)), the authors explicitly designed questions asking for all pairs satisfying certain properties, as they found many pair-aggregation tasks could be solved linearly using techniques like inclusion-exclusion, which would not properly stress-test the models' ability to handle pairwise relationships.
- OOLONG-Pairs benchmark synthetically generates 20 tasks from OOLONG's trec_coarse split across 11 context lengths (1K to 1M tokens)
- Each task requires predicting semantic mappings for entries
- Questions explicitly ask for pairs satisfying properties to ensure O(N²) information density
- Design prevents linear shortcuts (e.g., inclusion-exclusion) that would avoid genuine pairwise reasoning

  ### E.1 OOLONG-Pairs Benchmark
  Section E.1 describes OOLONG-Pairs, a synthetic benchmark derived from OOLONG (Bertsch et al., 2025) designed to evaluate RLMs' ability to handle quadratic information density (I ≈ O(N²)). The benchmark contains 20 new tasks based on the trec_coarse split, tested across 11 input context lengths ranging from 1,024 to 1,048,576 tokens. Each task requires predicting semantic mappings by identifying pairs of user IDs that satisfy specific properties based on semantic labels (description and abstract concept, entity, human being, numeric value, location, abbreviation). The authors explicitly designed pair-aggregation questions to ensure quadratic complexity, as they discovered many pair-based tasks could be solved linearly using techniques like inclusion-exclusion, which would fail to properly stress-test models' pairwise relationship handling capabilities. The 20 tasks vary in complexity, from simple pair matching based on shared label categories to more complex constraints involving specific counts, temporal conditions, and asymmetric requirements between users.
  - OOLONG-Pairs is a synthetic benchmark with 20 tasks across 11 context lengths (1K to 1M tokens)
  - Designed to ensure quadratic information density I ≈ O(N²) to properly test pairwise relationship handling
  - Tasks require identifying user ID pairs satisfying constraints based on 6 semantic labels from OOLONG trec_coarse split
  - Explicitly uses pair-aggregation questions to prevent linear solutions via inclusion-exclusion principles
  - Tasks increase in complexity from simple shared label matching to multi-constraint conditions with temporal filters and asymmetric requirements

  ### E.2 Scaling Huge Document Corpuses in BrowseComp+
  This section presents additional BrowseComp+ experiments evaluating RLM scalability on document corpuses beyond the k=1000 results in the main paper. A subset of 20 tasks from the original 150 was used to analyze performance degradation as input size increases. Two new baselines were added: ReAct w/ GPT-5 + BM25 (CodeAct variant without code environment) and GPT-5 + pre-query BM25. Key findings: (1) RLM(GPT-5) is the only model achieving and maintaining perfect performance at 1000-document scale, with the no-recursion ablation achieving 90%; (2) Base GPT-5 approaches show clear performance dropoff with increasing documents regardless of conditioning; (3) RLM inference costs scale log-linearly and are reasonably bounded compared to strategies like ReAct + BM25, with extrapolated costs being cheaper than GPT-5 with infinite context window.
  - RLM(GPT-5) achieves perfect performance at 1000-document scale; no-recursion ablation reaches 90%
  - Base GPT-5 models show performance degradation as document count increases, regardless of conditioning method
  - RLM inference costs scale log-linearly and are more cost-effective than extrapolated GPT-5 with infinite context
  - Experiments conducted on 20-task subset from original 150 BrowseComp+ tasks
  - Two new baselines added: ReAct w/ GPT-5 + BM25 and GPT-5 + pre-query BM25
