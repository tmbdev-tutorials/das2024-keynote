Can Large Language Models Reason? A Characterization via 3-SAT

Rishi Hazra, Gabriele Venturato, Pedro Zuidberg Dos Martires, Luc De Raedt

2024-08-13 [entry](http://arxiv.org/abs/2408.07215v1) [pdf](http://arxiv.org/pdf/2408.07215v1)

Large Language Models (LLMs) are said to possess advanced reasoning abilities.
However, some skepticism exists as recent works show how LLMs often bypass true
reasoning using shortcuts. Current methods for assessing the reasoning abilities
of LLMs typically rely on open-source benchmarks that may be overrepresented in
LLM training data, potentially skewing performance. We instead provide a
computational theory perspective of reasoning, using 3-SAT -- the prototypical
NP-complete problem that lies at the core of logical reasoning and constraint
satisfaction tasks. By examining the phase transitions in 3-SAT, we empirically
characterize the reasoning abilities of LLMs and show how they vary with the
inherent hardness of the problems. Our experimental evidence shows that LLMs
cannot perform true reasoning, as is required for solving 3-SAT problems.

---

IDEA: Enhancing the rule learning ability of language agent through Induction, DEuction, and Abduction

Kaiyu He, Zhiyu Chen

2024-08-19 [entry](http://arxiv.org/abs/2408.10455v1) [pdf](http://arxiv.org/pdf/2408.10455v1)

While large language models (LLMs) have been thoroughly evaluated for deductive
and inductive reasoning, their proficiency in abductive reasoning and holistic
rule learning in interactive environments remains less explored. This work
introduces RULEARN, a novel benchmark specifically designed to assess the rule-
learning ability of LLMs in interactive settings. In RULEARN, agents interact
with the environment to gather observations and discern patterns, using these
insights to solve problems. To further enhance the rule-learning capabilities of
LLM agents within this benchmark, we propose IDEA agent, which integrates
Induction, Deduction, and Abduction processes. IDEA agent refines this approach
by leveraging a structured reasoning sequence: generating hypotheses through
abduction, testing them via deduction, and refining them based on induction
feedback. This sequence enables agents to dynamically establish and apply rules,
mimicking human-like reasoning processes. Our evaluation of five representative
LLMs indicates that while these models can generate plausible initial
hypotheses, they often struggle with strategic interaction within the
environment, effective incorporation of feedback, and adaptive refinement of
their hypotheses. IDEA agent demonstrates significantly improved performance on
the RULEARN benchmark, offering valuable insights for the development of agents
capable of human-like rule-learning in real-world scenarios. We will release our
code and data.

---

Do Large Language Models Have Compositional Ability? An Investigation into Limitations and Scalability

Zhuoyan Xu, Zhenmei Shi, Yingyu Liang

2024-07-22 [entry](http://arxiv.org/abs/2407.15720v2) [pdf](http://arxiv.org/pdf/2407.15720v2)

Large language models (LLMs) have emerged as powerful tools for many AI problems
and exhibit remarkable in-context learning (ICL) capabilities. Compositional
ability, solving unseen complex tasks that combine two or more simple tasks, is
an essential reasoning ability for Artificial General Intelligence. Despite the
tremendous success of LLMs, how they approach composite tasks, especially those
not encountered during the pretraining phase, remains an open and largely
underexplored question. In this study, we delve into the ICL capabilities of
LLMs on composite tasks, with only simple tasks as in-context examples. We
develop a test suite of composite tasks including linguistic and logical
challenges and perform empirical studies across different LLM families. We
observe that models exhibit divergent behaviors: (1) For simpler composite tasks
that apply distinct mapping mechanisms to different input segments, the models
demonstrate decent compositional ability, while scaling up the model enhances
this ability; (2) for more complex composite tasks involving reasoning multiple
steps, where each step represents one task, models typically underperform, and
scaling up generally provides no improvements. We offer theoretical analysis in
a simplified setting, explaining that models exhibit compositional capability
when the task handles different input parts separately. We believe our work
sheds new light on the capabilities of LLMs in solving composite tasks regarding
the nature of the tasks and model scale. Our dataset and code are available at
{\url{https://github.com/OliverXUZY/LLM_Compose}}.

---

SubgoalXL: Subgoal-based Expert Learning for Theorem Proving

Xueliang Zhao, Lin Zheng, Haige Bo, Changran Hu, Urmish Thakker, Lingpeng Kong

2024-08-20 [entry](http://arxiv.org/abs/2408.11172v1) [pdf](http://arxiv.org/pdf/2408.11172v1)

Formal theorem proving, a field at the intersection of mathematics and computer
science, has seen renewed interest with advancements in large language models
(LLMs). This paper introduces SubgoalXL, a novel approach that synergizes
subgoal-based proofs with expert learning to enhance LLMs' capabilities in
formal theorem proving within the Isabelle environment. SubgoalXL addresses two
critical challenges: the scarcity of specialized mathematics and theorem-proving
data, and the need for improved multi-step reasoning abilities in LLMs. By
optimizing data efficiency and employing subgoal-level supervision, SubgoalXL
extracts richer information from limited human-generated proofs. The framework
integrates subgoal-oriented proof strategies with an expert learning system,
iteratively refining formal statement, proof, and subgoal generators. Leveraging
the Isabelle environment's advantages in subgoal-based proofs, SubgoalXL
achieves a new state-of-the-art performance of 56.1\% in Isabelle on the
standard miniF2F dataset, marking an absolute improvement of 4.9\%. Notably,
SubgoalXL successfully solves 41 AMC12, 9 AIME, and 3 IMO problems from miniF2F.
These results underscore the effectiveness of maximizing limited data utility
and employing targeted guidance for complex reasoning in formal theorem proving,
contributing to the ongoing advancement of AI reasoning capabilities. The
implementation is available at \url{https://github.com/zhaoxlpku/SubgoalXL}.

---

ONSEP: A Novel Online Neural-Symbolic Framework for Event Prediction Based on Large Language Model

Xuanqing Yu, Wangtao Sun, Jingwei Li, Kang Liu, Chengbao Liu, Jie Tan

2024-08-14 [entry](http://arxiv.org/abs/2408.07840v1) [pdf](http://arxiv.org/pdf/2408.07840v1)

In the realm of event prediction, temporal knowledge graph forecasting (TKGF)
stands as a pivotal technique. Previous approaches face the challenges of not
utilizing experience during testing and relying on a single short-term history,
which limits adaptation to evolving data. In this paper, we introduce the Online
Neural-Symbolic Event Prediction (ONSEP) framework, which innovates by
integrating dynamic causal rule mining (DCRM) and dual history augmented
generation (DHAG). DCRM dynamically constructs causal rules from real-time data,
allowing for swift adaptation to new causal relationships. In parallel, DHAG
merges short-term and long-term historical contexts, leveraging a bi-branch
approach to enrich event prediction. Our framework demonstrates notable
performance enhancements across diverse datasets, with significant Hit@k
(k=1,3,10) improvements, showcasing its ability to augment large language models
(LLMs) for event prediction without necessitating extensive retraining. The
ONSEP framework not only advances the field of TKGF but also underscores the
potential of neural-symbolic approaches in adapting to dynamic data
environments.

---

Reasoning Factual Knowledge in Structured Data with Large Language Models

Sirui Huang, Yanggan Gu, Xuming Hu, Zhonghao Li, Qing Li, Guandong Xu

2024-08-22 [entry](http://arxiv.org/abs/2408.12188v1) [pdf](http://arxiv.org/pdf/2408.12188v1)

Large language models (LLMs) have made remarkable progress in various natural
language processing tasks as a benefit of their capability to comprehend and
reason with factual knowledge. However, a significant amount of factual
knowledge is stored in structured data, which possesses unique characteristics
that differ from the unstructured texts used for pretraining. This difference
can introduce imperceptible inference parameter deviations, posing challenges
for LLMs in effectively utilizing and reasoning with structured data to
accurately infer factual knowledge. To this end, we propose a benchmark named
StructFact, to evaluate the structural reasoning capabilities of LLMs in
inferring factual knowledge. StructFact comprises 8,340 factual questions
encompassing various tasks, domains, timelines, and regions. This benchmark
allows us to investigate the capability of LLMs across five factual tasks
derived from the unique characteristics of structural facts. Extensive
experiments on a set of LLMs with different training strategies reveal the
limitations of current LLMs in inferring factual knowledge from structured data.
We present this benchmark as a compass to navigate the strengths and weaknesses
of LLMs in reasoning with structured data for knowledge-sensitive tasks, and to
encourage advancements in related real-world applications. Please find our code
at https://github.com/EganGu/StructFact.

---

Reasoning or Simply Next Token Prediction? A Benchmark for Stress-Testing Large Language Models

Wentian Wang, Paul Kantor, Jacob Feldman, Lazaros Gallos, Hao Wang

2024-06-15 [entry](http://arxiv.org/abs/2406.15468v1) [pdf](http://arxiv.org/pdf/2406.15468v1)

We propose MMLU-SR, a novel dataset designed to measure the true comprehension
abilities of Large Language Models (LLMs) by challenging their performance in
question-answering tasks with modified terms. We reasoned that an agent that
``truly'' understands a concept can still evaluate it when key terms are
replaced by suitably defined alternate terms, and sought to differentiate such
comprehension from mere text replacement. In our study, we modified standardized
test questions by replacing a key term with a dummy word along with its
definition. The key term could be in the context of questions, answers, or both
questions and answers.   Notwithstanding the high scores achieved by recent
popular LLMs on the MMLU leaderboard, we found a substantial reduction in model
performance after such replacement, suggesting poor comprehension. This new
benchmark provides a rigorous benchmark for testing true model comprehension,
and poses a challenge to the broader scientific community.

---

A Benchmark Suite for Systematically Evaluating Reasoning Shortcuts

Samuele Bortolotti, Emanuele Marconato, Tommaso Carraro, Paolo Morettin, Emile van Krieken, Antonio Vergari, Stefano Teso, Andrea Passerini

2024-06-14 [entry](http://arxiv.org/abs/2406.10368v1) [pdf](http://arxiv.org/pdf/2406.10368v1)

The advent of powerful neural classifiers has increased interest in problems
that require both learning and reasoning. These problems are critical for
understanding important properties of models, such as trustworthiness,
generalization, interpretability, and compliance to safety and structural
constraints. However, recent research observed that tasks requiring both
learning and reasoning on background knowledge often suffer from reasoning
shortcuts (RSs): predictors can solve the downstream reasoning task without
associating the correct concepts to the high-dimensional data. To address this
issue, we introduce rsbench, a comprehensive benchmark suite designed to
systematically evaluate the impact of RSs on models by providing easy access to
highly customizable tasks affected by RSs. Furthermore, rsbench implements
common metrics for evaluating concept quality and introduces novel formal
verification procedures for assessing the presence of RSs in learning tasks.
Using rsbench, we highlight that obtaining high quality concepts in both purely
neural and neuro-symbolic models is a far-from-solved problem. rsbench is
available at: https://unitn-sml.github.io/rsbench.

---

Planetarium: A Rigorous Benchmark for Translating Text to Structured Planning Languages

Max Zuo, Francisco Piedrahita Velez, Xiaochen Li, Michael L. Littman, Stephen H. Bach

2024-07-03 [entry](http://arxiv.org/abs/2407.03321v1) [pdf](http://arxiv.org/pdf/2407.03321v1)

Many recent works have explored using language models for planning problems. One
line of research focuses on translating natural language descriptions of
planning tasks into structured planning languages, such as the planning domain
definition language (PDDL). While this approach is promising, accurately
measuring the quality of generated PDDL code continues to pose significant
challenges. First, generated PDDL code is typically evaluated using planning
validators that check whether the problem can be solved with a planner. This
method is insufficient because a language model might generate valid PDDL code
that does not align with the natural language description of the task. Second,
existing evaluation sets often have natural language descriptions of the
planning task that closely resemble the ground truth PDDL, reducing the
challenge of the task. To bridge this gap, we introduce \benchmarkName, a
benchmark designed to evaluate language models' ability to generate PDDL code
from natural language descriptions of planning tasks. We begin by creating a
PDDL equivalence algorithm that rigorously evaluates the correctness of PDDL
code generated by language models by flexibly comparing it against a ground
truth PDDL. Then, we present a dataset of $132,037$ text-to-PDDL pairs across 13
different tasks, with varying levels of difficulty. Finally, we evaluate several
API-access and open-weight language models that reveal this task's complexity.
For example, $87.6\%$ of the PDDL problem descriptions generated by GPT-4o are
syntactically parseable, $82.2\%$ are valid, solve-able problems, but only
$35.1\%$ are semantically correct, highlighting the need for a more rigorous
benchmark for this problem.

---

LLASP: Fine-tuning Large Language Models for Answer Set Programming

Erica Coppolillo, Francesco Calimeri, Giuseppe Manco, Simona Perri, Francesco Ricca

2024-07-26 [entry](http://arxiv.org/abs/2407.18723v1) [pdf](http://arxiv.org/pdf/2407.18723v1)

Recently, Large Language Models (LLMs) have showcased their potential in various
natural language processing tasks, including code generation. However, while
significant progress has been made in adapting LLMs to generate code for several
imperative programming languages and tasks, there remains a notable gap in their
application to declarative formalisms, such as Answer Set Programming (ASP). In
this paper, we move a step towards exploring the capabilities of LLMs for ASP
code generation. First, we perform a systematic evaluation of several state-of-
the-art LLMs. Despite their power in terms of number of parameters, training
data and computational resources, empirical results demonstrate inadequate
performances in generating correct ASP programs. Therefore, we propose LLASP, a
fine-tuned lightweight model specifically trained to encode fundamental ASP
program patterns. To this aim, we create an ad-hoc dataset covering a wide
variety of fundamental problem specifications that can be encoded in ASP. Our
experiments demonstrate that the quality of ASP programs generated by LLASP is
remarkable. This holds true not only when compared to the non-fine-tuned
counterpart but also when compared to the majority of eager LLM candidates,
particularly from a semantic perspective. All the code and data used to perform
the experiments are publicly available at
https://anonymous.4open.science/r/LLASP-D86C/.

---

Misinforming LLMs: vulnerabilities, challenges and opportunities

Bo Zhou, Daniel Geißler, Paul Lukowicz

2024-08-02 [entry](http://arxiv.org/abs/2408.01168v1) [pdf](http://arxiv.org/pdf/2408.01168v1)

Large Language Models (LLMs) have made significant advances in natural language
processing, but their underlying mechanisms are often misunderstood. Despite
exhibiting coherent answers and apparent reasoning behaviors, LLMs rely on
statistical patterns in word embeddings rather than true cognitive processes.
This leads to vulnerabilities such as "hallucination" and misinformation. The
paper argues that current LLM architectures are inherently untrustworthy due to
their reliance on correlations of sequential patterns of word embedding vectors.
However, ongoing research into combining generative transformer-based models
with fact bases and logic programming languages may lead to the development of
trustworthy LLMs capable of generating statements based on given truth and
explaining their self-reasoning process.

---

QirK: Question Answering via Intermediate Representation on Knowledge Graphs

Jan Luca Scheerer, Anton Lykov, Moe Kayali, Ilias Fountalis, Dan Olteanu, Nikolaos Vasiloglou, Dan Suciu

2024-08-14 [entry](http://arxiv.org/abs/2408.07494v1) [pdf](http://arxiv.org/pdf/2408.07494v1)

We demonstrate QirK, a system for answering natural language questions on
Knowledge Graphs (KG). QirK can answer structurally complex questions that are
still beyond the reach of emerging Large Language Models (LLMs). It does so
using a unique combination of database technology, LLMs, and semantic search
over vector embeddings. The glue for these components is an intermediate
representation (IR). The input question is mapped to IR using LLMs, which is
then repaired into a valid relational database query with the aid of a semantic
search on vector embeddings. This allows a practical synthesis of LLM
capabilities and KG reliability.   A short video demonstrating QirK is available
at https://youtu.be/6c81BLmOZ0U.

---

Does Reasoning Emerge? Examining the Probabilities of Causation in Large Language Models

Javier González, Aditya V. Nori

2024-08-15 [entry](http://arxiv.org/abs/2408.08210v1) [pdf](http://arxiv.org/pdf/2408.08210v1)

Recent advances in AI have been significantly driven by the capabilities of
large language models (LLMs) to solve complex problems in ways that resemble
human thinking. However, there is an ongoing debate about the extent to which
LLMs are capable of actual reasoning. Central to this debate are two key
probabilistic concepts that are essential for connecting causes to their
effects: the probability of necessity (PN) and the probability of sufficiency
(PS). This paper introduces a framework that is both theoretical and practical,
aimed at assessing how effectively LLMs are able to replicate real-world
reasoning mechanisms using these probabilistic measures. By viewing LLMs as
abstract machines that process information through a natural language interface,
we examine the conditions under which it is possible to compute suitable
approximations of PN and PS. Our research marks an important step towards
gaining a deeper understanding of when LLMs are capable of reasoning, as
illustrated by a series of math examples.

---

AutoML-guided Fusion of Entity and LLM-based representations

Boshko Koloski, Senja Pollak, Roberto Navigli, Blaž Škrlj

2024-08-19 [entry](http://arxiv.org/abs/2408.09794v1) [pdf](http://arxiv.org/pdf/2408.09794v1)

Large semantic knowledge bases are grounded in factual knowledge. However,
recent approaches to dense text representations (embeddings) do not efficiently
exploit these resources. Dense and robust representations of documents are
essential for effectively solving downstream classification and retrieval tasks.
This work demonstrates that injecting embedded information from knowledge bases
can augment the performance of contemporary Large Language Model (LLM)-based
representations for the task of text classification. Further, by considering
automated machine learning (AutoML) with the fused representation space, we
demonstrate it is possible to improve classification accuracy even if we use
low-dimensional projections of the original representation space obtained via
efficient matrix factorization. This result shows that significantly faster
classifiers can be achieved with minimal or no loss in predictive performance,
as demonstrated using five strong LLM baselines on six diverse real-life
datasets.

---

A Percolation Model of Emergence: Analyzing Transformers Trained on a Formal Language

Ekdeep Singh Lubana, Kyogo Kawaguchi, Robert P. Dick, Hidenori Tanaka

2024-08-22 [entry](http://arxiv.org/abs/2408.12578v1) [pdf](http://arxiv.org/pdf/2408.12578v1)

Increase in data, size, or compute can lead to sudden learning of specific
capabilities by a neural network -- a phenomenon often called "emergence".
Beyond scientific understanding, establishing the causal factors underlying such
emergent capabilities is crucial to enable risk regulation frameworks for AI. In
this work, we seek inspiration from study of emergent properties in other fields
and propose a phenomenological definition for the concept in the context of
neural networks. Our definition implicates the acquisition of specific
structures underlying the data-generating process as a cause of sudden
performance growth for specific, narrower tasks. We empirically investigate this
definition by proposing an experimental system grounded in a context-sensitive
formal language and find that Transformers trained to perform tasks on top of
strings from this language indeed exhibit emergent capabilities. Specifically,
we show that once the language's underlying grammar and context-sensitivity
inducing structures are learned by the model, performance on narrower tasks
suddenly begins to improve. We then analogize our network's learning dynamics
with the process of percolation on a bipartite graph, establishing a formal
phase transition model that predicts the shift in the point of emergence
observed in experiment when changing the data structure. Overall, our
experimental and theoretical frameworks yield a step towards better defining,
characterizing, and predicting emergence in neural networks.

---

Exploiting Large Language Models Capabilities for Question Answer-Driven Knowledge Graph Completion Across Static and Temporal Domains

Rui Yang, Jiahao Zhu, Jianping Man, Li Fang, Yi Zhou

2024-08-20 [entry](http://arxiv.org/abs/2408.10819v1) [pdf](http://arxiv.org/pdf/2408.10819v1)

Knowledge graph completion (KGC) aims to identify missing triples in a knowledge
graph (KG). This is typically achieved through tasks such as link prediction and
instance completion. However, these methods often focus on either static
knowledge graphs (SKGs) or temporal knowledge graphs (TKGs), addressing only
within-scope triples. This paper introduces a new generative completion
framework called Generative Subgraph-based KGC (GS-KGC). GS-KGC employs a
question-answering format to directly generate target entities, addressing the
challenge of questions having multiple possible answers. We propose a strategy
that extracts subgraphs centered on entities and relationships within the KG,
from which negative samples and neighborhood information are separately obtained
to address the one-to-many problem. Our method generates negative samples using
known facts to facilitate the discovery of new information. Furthermore, we
collect and refine neighborhood path data of known entities, providing
contextual information to enhance reasoning in large language models (LLMs). Our
experiments evaluated the proposed method on four SKGs and two TKGs, achieving
state-of-the-art Hits@1 metrics on five datasets. Analysis of the results shows
that GS-KGC can discover new triples within existing KGs and generate new facts
beyond the closed KG, effectively bridging the gap between closed-world and
open-world KGC.

---

Diagnosing and Remedying Knowledge Deficiencies in LLMs via Label-free Curricular Meaningful Learning

Kai Xiong, Xiao Ding, Li Du, Jiahao Ying, Ting Liu, Bing Qin, Yixin Cao

2024-08-21 [entry](http://arxiv.org/abs/2408.11431v1) [pdf](http://arxiv.org/pdf/2408.11431v1)

Large Language Models (LLMs) are versatile and demonstrate impressive
generalization ability by mining and learning information from extensive
unlabeled text. However, they still exhibit reasoning mistakes, often stemming
from knowledge deficiencies, which can affect their trustworthiness and
reliability. Although users can provide diverse and comprehensive queries,
obtaining sufficient and effective feedback is demanding. Furthermore,
evaluating LLMs comprehensively with limited labeled samples is difficult. This
makes it a challenge to diagnose and remedy the deficiencies of LLMs through
rich label-free user queries. To tackle this challenge, we propose a label-free
curricular meaningful learning framework (LaMer). LaMer first employs relative
entropy to automatically diagnose and quantify the knowledge deficiencies of
LLMs in a label-free setting. Next, to remedy the diagnosed knowledge
deficiencies, we apply curricular meaningful learning: first, we adopt
meaningful learning to adaptively synthesize augmentation data according to the
severity of the deficiencies, and then design a curricular deficiency remedy
strategy to remedy the knowledge deficiencies of LLMs progressively. Experiments
show that LaMer efficiently and effectively diagnoses and remedies knowledge
deficiencies in LLMs, improving various LLMs across seven out-of-distribution
(OOD) reasoning and language understanding benchmarks, achieving comparable
results to baselines with just 40\% training data. LaMer even surpasses methods
that rely on labeled datasets for deficiency diagnosis. In application, our
label-free method can offer an effective knowledge deficiency diagnostic tool
for efficient LLM development.

---

Exploring and Benchmarking the Planning Capabilities of Large Language Models

Bernd Bohnet, Azade Nova, Aaron T Parisi, Kevin Swersky, Katayoon Goshvadi, Hanjun Dai, Dale Schuurmans, Noah Fiedel, Hanie Sedghi

2024-06-18 [entry](http://arxiv.org/abs/2406.13094v1) [pdf](http://arxiv.org/pdf/2406.13094v1)

We seek to elevate the planning capabilities of Large Language Models
(LLMs)investigating four main directions. First, we construct a comprehensive
benchmark suite encompassing both classical planning domains and natural
language scenarios. This suite includes algorithms to generate instances with
varying levels of difficulty, allowing for rigorous and systematic evaluation of
LLM performance. Second, we investigate the use of in-context learning (ICL) to
enhance LLM planning, exploring the direct relationship between increased
context length and improved planning performance. Third, we demonstrate the
positive impact of fine-tuning LLMs on optimal planning paths, as well as the
effectiveness of incorporating model-driven search procedures. Finally, we
investigate the performance of the proposed methods in out-of-distribution
scenarios, assessing the ability to generalize to novel and unseen planning
challenges.

---

