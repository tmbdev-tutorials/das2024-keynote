
## TaskMatrix.AI: Completing Tasks by Connecting Foundation Models with Millions of APIs

Yaobo Liang, Chenfei Wu, Ting Song, Wenshan Wu, Yan Xia, Yu Liu, Yang Ou, Shuai Lu, Lei Ji, Shaoguang Mao, Yun Wang, Linjun Shou, Ming Gong, Nan Duan

Category: symbolic
Keywords: Artificial Intelligence, Foundation Models, APIs, Task Completion, TaskMatrix.AI
Year: 2023

Artificial Intelligence (AI) has made incredible progress recently. On the one
hand, advanced foundation models like ChatGPT can offer powerful conversation,
in-context learning and code generation abilities on a broad range of open-
domain tasks. They can also generate high-level solution outlines for domain-
specific tasks based on the common sense knowledge they have acquired. However,
they still face difficulties with some specialized tasks because they lack
enough domain-specific data during pre-training or they often have errors in
their neural network computations on those tasks that need accurate executions.
On the other hand, there are also many existing models and systems (symbolic-
based or neural-based) that can do some domain-specific tasks very well.
However, due to the different implementation or working mechanisms, they are not
easily accessible or compatible with foundation models. Therefore, there is a
clear and pressing need for a mechanism that can leverage foundation models to
propose task solution outlines and then automatically match some of the sub-
tasks in the outlines to the off-the-shelf models and systems with special
functionalities to complete them. Inspired by this, we introduce TaskMatrix.AI
as a new AI ecosystem that connects foundation models with millions of APIs for
task completion. Unlike most previous work that aimed to improve a single AI
model, TaskMatrix.AI focuses more on using existing foundation models (as a
brain-like central system) and APIs of other AI models and systems (as sub-task
solvers) to achieve diversified tasks in both digital and physical domains. As a
position paper, we will present our vision of how to build such an ecosystem,
explain each key component, and use study cases to illustrate both the
feasibility of this vision and the main challenges we need to address next.
---

## Neural Theorem Proving in Lean using Proof Artifact Co-training and Language Models

Jason Rute

Category: symbolic
Keywords: neural theorem proving, Lean, proof artifact co-training, language models
Year: 2023

The paper presents a method for neural theorem proving in the Lean interactive
theorem prover by leveraging proof artifact co-training and language models. The
approach aims to improve the efficiency and accuracy of theorem proving tasks by
utilizing state-of-the-art deep learning techniques.
---

## PLASMA: Making Small Language Models Better Procedural Knowledge Models for (Counterfactual) Planning

Faeze Brahman, Chandra Bhagavatula, Valentina Pyatkin, Jena D. Hwang, Xiang Lorraine Li, Hirona J. Arai, Soumya Sanyal, Keisuke Sakaguchi, Xiang Ren, Yejin Choi

Category: symbolic
Keywords: procedural planning, small language models, counterfactual planning, symbolic procedural knowledge distillation, structured reasoning
Year: 2023

Procedural planning, which entails decomposing a high-level goal into a sequence
of temporally ordered steps, is an important yet intricate task for machines. It
involves integrating common-sense knowledge to reason about complex
contextualized situations that are often counterfactual, e.g., 'scheduling a
doctor’s appointment without a phone'. While current approaches show encouraging
results using large language models (LLMs), they are hindered by drawbacks such
as costly API calls and reproducibility issues. In this paper, we advocate
planning using smaller language models. We present PLASMA, a novel two-pronged
approach to endow small language models with procedural knowledge and
(counterfactual) planning capabilities. More concretely, we develop symbolic
procedural knowledge distillation to enhance the implicit knowledge in small
language models and an inference-time algorithm to facilitate more structured
and accurate reasoning. In addition, we introduce a novel task, Counterfactual
Planning, that requires a revision of a plan to cope with a counterfactual
situation. In both the original and counterfactual setting, we show that orders-
of-magnitude smaller models (770M-11B parameters) can compete and often surpass
their larger teacher models’ capabilities.
---

## Learning to Generate Novel Scientific Directions with Contextualized Literature-based Discovery

Qingyun Wang, Doug Downey, Heng Ji, Tom Hope

Category: symbolic
Keywords: Literature-Based Discovery, contextualized LBD, scientific hypothesis generation, knowledge graph, large language models
Year: 2023

Literature-Based Discovery (LBD) aims to discover new scientific knowledge by
mining papers and generating hypotheses. Standard LBD is limited to predicting
pairwise relations between discrete concepts (e.g., drug-disease links). LBD
also ignores critical contexts like experimental settings (e.g., a specific
patient population where a drug is evaluated) and background knowledge and
motivations that human scientists consider (e.g., to find a drug candidate
without specific side effects). We address these limitations with a novel
formulation of contextualized-LBD (C-LBD): generating scientific hypotheses in
natural language, while grounding them in a context that controls the hypothesis
search space. We present a new modeling framework using retrieval of
'inspirations' from a heterogeneous network of citations and knowledge graph
relations, and create a new dataset derived from papers. In automated and human
evaluations, our models improve over baselines, including powerful large
language models (LLMs), but also reveal challenges on the road to building
machines that generate new scientific knowledge.
---

## On Suspicious Coincidences and Pointwise Mutual Information

Christopher K. I. Williams

Category: symbolic
Keywords: suspicious coincidences, pointwise mutual information, mutual information, association measures, contingency tables, odds ratio
Year: 2022

Barlow (1985) hypothesized that the co-occurrence of two events A and B is
'suspicious' if P(A, B) ≫ P(A)P(B). We first review classical measures of
association for 2 × 2 contingency tables, including Yule’s Y (Yule, 1912), which
depends only on the odds ratio λ and is independent of the marginal
probabilities of the table. We then discuss the mutual information (MI) and
pointwise mutual information (PMI), which depend on the ratio P(A, B)/P(A)P(B),
as measures of association. We show that once the effect of the marginals is
removed, MI and PMI behave similarly to Y as functions of λ. The pointwise
mutual information is used extensively in some research communities for flagging
suspicious coincidences. We discuss the pros and cons of using it in this way,
bearing in mind the sensitivity of the PMI to the marginals, with increased
scores for sparser events.
---

## Consistent ultrafinitist logic

Michal J. Gajda

Category: symbolic
Keywords: ultrafinitism, finitist logic, proof system, Curry-Howard isomorphism, transfinite reasoning
Year: 2021

Ultrafinitism postulates that we can only compute on relatively short objects,
and numbers beyond a certain value are not available. This approach would also
forbid many forms of infinitary reasoning and allow the removal of certain
paradoxes stemming from enumeration theorems. However, philosophers still
disagree on whether such a finitist logic would be consistent. We present
preliminary work on a proof system based on the Curry-Howard isomorphism. We
also try to present some well-known theorems that stop being true in such
systems, whereas opposite statements become provable. This approach presents
certain impossibility results as logical paradoxes stemming from a profligate
use of transfinite reasoning.
---

## Learning Neural Causal Models with Active Interventions

Nino Scherrer, Olexa Bilaniuk, Yashas Annadani, Anirudh Goyal, Patrick Schwab, Bernhard Schölkopf, Michael C. Mozer, Yoshua Bengio, Stefan Bauer, Nan Rosemary Ke

Category: symbolic
Keywords: causal structure learning, neural networks, active interventions, differentiable causal discovery, directed acyclic graph
Year: 2021

Discovering causal structures from data is a challenging inference problem of
fundamental importance in all areas of science. The appealing scaling properties
of neural networks have recently led to a surge of interest in differentiable
neural network-based methods for learning causal structures from data. So far
differentiable causal discovery has focused on static datasets of observational
or interventional origin. In this work, we introduce an active intervention-
targeting mechanism which enables a quick identification of the underlying
causal structure of the data-generating process. Our method significantly
reduces the required number of interactions compared with random intervention
targeting and is applicable for both discrete and continuous optimization
formulations of learning the underlying directed acyclic graph (DAG) from data.
We examine the proposed method across a wide range of settings and demonstrate
superior performance on multiple benchmarks from simulated to real-world data.
---

## Symbolic Knowledge Distillation: from General Language Models to Commonsense Models

Peter West, Chandra Bhagavatula, Jack Hessel, Jena D. Hwang, Liwei Jiang, Ronan Le Bras, Ximing Lu, Sean Welleck, Yejin Choi

Category: symbolic
Keywords: symbolic knowledge distillation, commonsense models, general language models, knowledge graphs, GPT-3
Year: 2021

The common practice for training commonsense models has gone from human to
corpus to machine: humans author commonsense knowledge graphs in order to train
commonsense models. In this work, we investigate an alternative, from machine to
corpus to machine: general language models author these commonsense knowledge
graphs to train commonsense models. Our study leads to a new framework, Symbolic
Knowledge Distillation. As with prior art in Knowledge Distillation, our
approach uses larger models to teach smaller models. A key difference is that we
distill knowledge symbolically—as text—in addition to the neural model. We also
distill only one aspect—the commonsense of a general language model teacher,
allowing the student to be a different type, a commonsense model. Altogether, we
show that careful prompt engineering and a separately trained critic model allow
us to selectively distill high-quality causal commonsense from GPT-3, a general
language model. Empirical results demonstrate that, for the first time, a human-
authored commonsense knowledge graph is surpassed by our automatically distilled
variant in all three criteria: quantity, quality, and diversity. In addition, it
results in a neural commonsense model that surpasses the teacher model’s
commonsense capabilities despite its 100x smaller size. We apply this to the
ATOMIC resource, and share our new symbolic knowledge graph and commonsense
models.
---

## WikiGraphs: A Wikipedia Text - Knowledge Graph Paired Dataset

Luyu Wang, Yujia Li, Ozlem Aslan, Oriol Vinyals

Category: symbolic
Keywords: Wikipedia, knowledge graph, text generation, graph generation, graph representation learning
Year: 2021

We present a new dataset of Wikipedia articles each paired with a knowledge
graph, to facilitate research in conditional text generation, graph generation,
and graph representation learning. Existing graph-text paired datasets typically
contain small graphs and short text (1 or few sentences), thus limiting the
capabilities of the models that can be learned on the data. Our new dataset
WikiGraphs is collected by pairing each Wikipedia article from the established
WikiText-103 benchmark with a subgraph from the Freebase knowledge graph. This
makes it easy to benchmark against other state-of-the-art text generative models
that are capable of generating long paragraphs of coherent text. Both the graphs
and the text data are of significantly larger scale compared to prior graph-text
paired datasets. We present baseline graph neural network and transformer model
results on our dataset for 3 tasks: graph → text generation, graph → text
retrieval, and text → graph retrieval. We show that better conditioning on the
graph provides gains in generation and retrieval quality but there is still
large room for improvement.
---

## Causality for Machine Learning

Bernhard Schölkopf

Category: symbolic
Keywords: causality, machine learning, graphical causal inference, artificial intelligence, causal modeling
Year: 2020

Graphical causal inference as pioneered by Judea Pearl arose from research on
artificial intelligence (AI), and for a long time had little connection to the
field of machine learning. This article discusses where links have been and
should be established, introducing key concepts along the way. It argues that
the hard open problems of machine learning and AI are intrinsically related to
causality, and explains how the field is beginning to understand them.
---

## Causal Induction from Visual Observations for Goal Directed Tasks

Suraj Nair, Yuke Zhu, Silvio Savarese, Li Fei-Fei

Category: symbolic
Keywords: causal reasoning, goal-directed tasks, causal induction, directed acyclic graphs, attention mechanisms
Year: 2020

Causal reasoning has been an indispensable capability for humans and other
intelligent animals to interact with the physical world. In this work, we
propose to endow an artificial agent with the capability of causal reasoning for
completing goal-directed tasks. We develop learning-based approaches to inducing
causal knowledge in the form of directed acyclic graphs, which can be used to
contextualize a learned goal-conditional policy to perform tasks in novel
environments with latent causal structures. We leverage attention mechanisms in
our causal induction model and goal-conditional policy, enabling us to
incrementally generate the causal graph from the agent’s visual observations and
to selectively use the induced graph for determining actions. Our experiments
show that our method effectively generalizes towards completing new tasks in
novel environments with previously unseen causal structures.
---

## Discovering Symbolic Models from Deep Learning with Inductive Biases

Miles Cranmer, Alvaro Sanchez-Gonzalez, Peter Battaglia, Rui Xu, Kyle Cranmer, David Spergel, Shirley Ho

Category: symbolic
Keywords: symbolic regression, graph neural networks, inductive biases, cosmology, dark matter, interpretability
Year: 2020

We develop a general approach to distill symbolic representations of a learned
deep model by introducing strong inductive biases. We focus on Graph Neural
Networks (GNNs). The technique works as follows: we first encourage sparse
latent representations when we train a GNN in a supervised setting, then we
apply symbolic regression to components of the learned model to extract explicit
physical relations. We find the correct known equations, including force laws
and Hamiltonians, can be extracted from the neural network. We then apply our
method to a non-trivial cosmology example—a detailed dark matter simulation—and
discover a new analytic formula which can predict the concentration of dark
matter from the mass distribution of nearby cosmic structures. The symbolic
expressions extracted from the GNN using our technique also generalized to out-
of-distribution data better than the GNN itself. Our approach offers alternative
directions for interpreting neural networks and discovering novel physical
principles from the representations they learn.
---

## Well-tempered ZX and ZH calculi

Niel de Beaudrap

Category: symbolic
Keywords: ZX calculus, ZH calculus, quantum operations, tensor networks, bialgebra
Year: 2020

The ZX calculus is a mathematical tool to represent and analyse quantum
operations by manipulating diagrams which in effect represent tensor networks.
Two families of nodes of these networks are ones which commute with either Z
rotations or X rotations, usually called “green nodes” and “red nodes”
respectively. The original formulation of the ZX calculus was motivated in part
by properties of the algebras formed by the green and red nodes: notably, that
they form a bialgebra — but only up to scalar factors. As a consequence, the
diagram transformations and notation for certain unitary operations involve
“scalar gadgets” which denote contributions to a normalising factor. We present
renormalised generators for the ZX calculus, which form a bialgebra precisely.
As a result, no scalar gadgets are required to represent the most common unitary
transformations, and the corresponding diagram transformations are generally
simpler. We also present a similar renormalised version of the ZH calculus. We
obtain these results by an analysis of conditions under which various
“idealised” rewrites are sound, leveraging the existing presentations of the ZX
and ZH calculi.
---

## DualSMC: Tunneling Differentiable Filtering and Planning under Continuous POMDPs

Yunbo Wang, Bo Liu, Jiajun Wu, Yuke Zhu, Simon S. Du, Li Fei-Fei, Joshua B. Tenenbaum

Category: symbolic
Keywords: POMDPs, Sequential Monte Carlo, Adversarial Particle Filter, Planning Algorithm, Continuous Domains
Year: 2020

A major difficulty of solving continuous POMDPs is to infer the multi-modal
distribution of the unobserved true states and to make the planning algorithm
dependent on the perceived uncertainty. We cast POMDP filtering and planning
problems as two closely related Sequential Monte Carlo (SMC) processes, one over
the real states and the other over the future optimal trajectories, and combine
the merits of these two parts in a new model named the DualSMC network. In
particular, we first introduce an adversarial particle filter that leverages the
adversarial relationship between its internal components. Based on the filtering
results, we then propose a planning algorithm that extends the previous SMC
planning approach [Piche et al., 2018] to continuous POMDPs with an uncertainty-
dependent policy. Crucially, not only can DualSMC handle complex observations
such as image input but also it remains highly interpretable. It is shown to be
effective in three continuous POMDP domains: the floor positioning domain, the
3D light-dark navigation domain, and a modified Reacher domain.
---

## Language Models are Open Knowledge Graphs

Chenguang Wang, Xiao Liu, Dawn Song

Category: symbolic
Keywords: knowledge graphs, language models, unsupervised learning, GPT-3, BERT
Year: 2020

This paper shows how to construct knowledge graphs (KGs) from pre-trained
language models (e.g., BERT, GPT-2/3), without human supervision. Popular KGs
(e.g., Wikidata, NELL) are built in either a supervised or semi-supervised
manner, requiring humans to create knowledge. Recent deep language models
automatically acquire knowledge from large-scale corpora via pre-training. The
stored knowledge has enabled the language models to improve downstream NLP
tasks, e.g., answering questions, and writing code and articles. In this paper,
we propose an unsupervised method to cast the knowledge contained within
language models into KGs. We show that KGs are constructed with a single forward
pass of the pre-trained language models (without fine-tuning) over the corpora.
We demonstrate the quality of the constructed KGs by comparing to two KGs
(Wikidata, TAC KBP) created by humans. Our KGs also provide open factual
knowledge that is new in the existing KGs. Our code and KGs will be made
publicly available.
---

## Tensor Network Rewriting Strategies for Satisfiability and Counting

Niel de Beaudrap, Aleks Kissinger, Konstantinos Meichanetzidis

Category: symbolic
Keywords: SAT, #SAT, tensor networks, ZH-calculus, complexity theory
Year: 2020

We provide a graphical treatment of SAT and #SAT on equal footing. Instances of
#SAT can be represented as tensor networks in a standard way. These tensor
networks are interpreted by diagrams of the ZH-calculus: a system to reason
about tensors over C in terms of diagrams built from simple generators, in which
computation may be carried out by transformations of diagrams alone. In general,
nodes of ZH diagrams take parameters over C which determine the tensor
coefficients; for the standard representation of #SAT instances, the
coefficients take the value 0 or 1. Then, by choosing the coefficients of a
diagram to range over B, we represent the corresponding instance of SAT. Thus,
by interpreting a diagram either over the boolean semiring or the complex
numbers, we instantiate either the decision or counting version of the problem.
We find that for classes known to be in P, such as 2SAT and #XORSAT, the
existence of appropriate rewrite rules allows for efficient simplification of
the diagram, producing the solution in polynomial time. In contrast, for classes
known to be NP-complete, such as 3SAT, or #P-complete, such as #2SAT, the
corresponding rewrite rules introduce hyperedges to the diagrams, in numbers
which are not easily bounded above by a polynomial. This diagrammatic approach
unifies the diagnosis of the complexity of CSPs and #CSPs and shows promise in
aiding tensor network contraction-based algorithms.
---

## Web Table Extraction, Retrieval and Augmentation: A Survey

Shuo Zhang, Krisztian Balog

Category: symbolic
Keywords: table extraction, table search, table retrieval, table mining, table augmentation, table interpretation
Year: 2020

Tables are a powerful and popular tool for organizing and manipulating data. A
vast number of tables can be found on the Web, which represent a valuable
knowledge resource. The objective of this survey is to synthesize and present
two decades of research on web tables. In particular, we organize existing
literature into six main categories of information access tasks: table
extraction, table interpretation, table search, question answering, knowledge
base augmentation, and table augmentation. For each of these tasks, we identify
and describe seminal approaches, present relevant resources, and point out
interdependencies among the different tasks.
---

## Tensor Network Rewriting Strategies for Satisfiability and Counting

Niel de Beaudrap, Aleks Kissinger, Konstantinos Meichanetzidis

Category: symbolic
Keywords: SAT, #SAT, tensor networks, ZH-calculus, complexity theory
Year: 2020

We provide a graphical treatment of SAT and #SAT on equal footing. Instances of
#SAT can be represented as tensor networks in a standard way. These tensor
networks are interpreted by diagrams of the ZH-calculus: a system to reason
about tensors over C in terms of diagrams built from simple generators, in which
computation may be carried out by transformations of diagrams alone. In general,
nodes of ZH diagrams take parameters over C which determine the tensor
coefficients; for the standard representation of #SAT instances, the
coefficients take the value 0 or 1. Then, by choosing the coefficients of a
diagram to range over B, we represent the corresponding instance of SAT. Thus,
by interpreting a diagram either over the boolean semiring or the complex
numbers, we instantiate either the decision or counting version of the problem.
We find that for classes known to be in P, such as 2SAT and #XORSAT, the
existence of appropriate rewrite rules allows for efficient simplification of
the diagram, producing the solution in polynomial time. In contrast, for classes
known to be NP-complete, such as 3SAT, or #P-complete, such as #2SAT, the
corresponding rewrite rules introduce hyperedges to the diagrams, in numbers
which are not easily bounded above by a polynomial. This diagrammatic approach
unifies the diagnosis of the complexity of CSPs and #CSPs and shows promise in
aiding tensor network contraction-based algorithms.
---

## Tensor Network Rewriting Strategies for Satisfiability and Counting

Niel de Beaudrap, Aleks Kissinger, Konstantinos Meichanetzidis

Category: symbolic
Keywords: SAT, #SAT, tensor networks, ZH-calculus, complexity theory
Year: 2020

We provide a graphical treatment of SAT and #SAT on equal footing. Instances of
#SAT can be represented as tensor networks in a standard way. These tensor
networks are interpreted by diagrams of the ZH-calculus: a system to reason
about tensors over C in terms of diagrams built from simple generators, in which
computation may be carried out by transformations of diagrams alone. In general,
nodes of ZH diagrams take parameters over C which determine the tensor
coefficients; for the standard representation of #SAT instances, the
coefficients take the value 0 or 1. Then, by choosing the coefficients of a
diagram to range over B, we represent the corresponding instance of SAT. Thus,
by interpreting a diagram either over the boolean semiring or the complex
numbers, we instantiate either the decision or counting version of the problem.
We find that for classes known to be in P, such as 2SAT and #XORSAT, the
existence of appropriate rewrite rules allows for efficient simplification of
the diagram, producing the solution in polynomial time. In contrast, for classes
known to be NP-complete, such as 3SAT, or #P-complete, such as #2SAT, the
corresponding rewrite rules introduce hyperedges to the diagrams, in numbers
which are not easily bounded above by a polynomial. This diagrammatic approach
unifies the diagnosis of the complexity of CSPs and #CSPs and shows promise in
aiding tensor network contraction-based algorithms.
---

## The Structure of Concurrent Process Histories

Chad Nester

Category: symbolic
Keywords: concurrent systems, algebraic structure, resource convertibility, categorical theories, string diagrams
Year: 2020

We identify the algebraic structure of the material histories generated by
concurrent processes. Specifically, we extend existing categorical theories of
resource convertibility to capture concurrent interaction. Our formalism admits
an intuitive graphical presentation via string diagrams for proarrow equipments.
---

## AI-KG: an Automatically Generated Knowledge Graph of Artificial Intelligence

Danilo Dessì, Francesco Osborne, Diego Reforgiato Recupero, Davide Buscaldi, Enrico Motta, Harald Sack

Category: symbolic
Keywords: Knowledge Graph, Artificial Intelligence, Natural Language Processing, Machine Learning, Information Retrieval, Knowledge Discovery
Year: 2020

This document introduces AI-KG, an automatically generated knowledge graph that
covers the domain of Artificial Intelligence. The graph is created by applying
natural language processing and machine learning techniques to a large corpus of
research papers. The resulting knowledge graph provides a structured
representation of the concepts and relationships within the AI domain,
facilitating various applications such as information retrieval, question
answering, and knowledge discovery.
---

## Discovering Symbolic Models from Deep Learning with Inductive Biases

Miles Cranmer, Alvaro Sanchez-Gonzalez, Peter Battaglia, Rui Xu, Kyle Cranmer, David Spergel, Shirley Ho

Category: symbolic
Keywords: symbolic regression, deep learning, inductive biases, graph neural networks, cosmology
Year: 2020

We develop a general approach to distill symbolic representations of a learned
deep model by introducing strong inductive biases. We focus on Graph Neural
Networks (GNNs). The technique works as follows: we first encourage sparse
latent representations when we train a GNN in a supervised setting, then we
apply symbolic regression to components of the learned model to extract explicit
physical relations. We find the correct known equations, including force laws
and Hamiltonians, can be extracted from the neural network. We then apply our
method to a non-trivial cosmology example—a detailed dark matter simulation—and
discover a new analytic formula which can predict the concentration of dark
matter from the mass distribution of nearby cosmic structures. The symbolic
expressions extracted from the GNN using our technique also generalized to out-
of-distribution-data better than the GNN itself. Our approach offers alternative
directions for interpreting neural networks and discovering novel physical
principles from the representations they learn.
---

## Quantum Tensor Networks, Stochastic Processes, and Weighted Automata

Siddarth Srinivasan, Sandesh Adhikary, Jacob Miller, Guillaume Rabusseau, Byron Boots

Category: symbolic
Keywords: quantum tensor networks, stochastic processes, weighted automata, matrix product states, probabilistic modeling
Year: 2020

Modeling joint probability distributions over sequences has been studied from
many perspectives. The physics community developed matrix product states, a
tensor-train decomposition for probabilistic modeling, motivated by the need to
tractably model many-body systems. But similar models have also been studied in
the stochastic processes and weighted automata literature, with little work on
how these bodies of work relate to each other. We address this gap by showing
how stationary or uniform versions of popular quantum tensor network models have
equivalent representations in the stochastic processes and weighted automata
literature, in the limit of infinitely long sequences. We demonstrate several
equivalence results between models used in these three communities: (i) uniform
variants of matrix product states, Born machines and locally purified states
from the quantum tensor networks literature, (ii) predictive state
representations, hidden Markov models, norm-observable operator models and
hidden quantum Markov models from the stochastic process literature, and (iii)
stochastic weighted automata, probabilistic automata and quadratic automata from
the formal languages literature. Such connections may open the door for results
and methods developed in one area to be applied in another.
---

## The game semantics of game theory

Jules Hedges

Category: symbolic
Keywords: compositional game theory, game semantics, Nash equilibrium, dialectica category, geometry of interaction
Year: 2020

We use a reformulation of compositional game theory to reunite game theory with
game semantics, by viewing an open game as the System and its choice of contexts
as the Environment. Specifically, the system is jointly controlled by n ≥0
noncooperative players, each independently optimising a real-valued payoff. The
goal of the system is to play a Nash equilibrium, and the goal of the
environment is to prevent it. The key to this is the realisation that lenses
(from functional programming) form a dialectica category, which have an existing
game-semantic interpretation. In the second half of this paper, we apply these
ideas to build a compact closed category of 'computable open games' by replacing
the underlying dialectica category with a wave-style geometry of interaction
category, specifically the Int-construction applied to the traced cartesian
category of directed-complete partial orders.
---

## Random Walks: A Review of Algorithms and Applications

Feng Xia, Jiaying Liu, Hansong Nie, Yonghao Fu, Liangtian Wan, Xiangjie Kong

Category: symbolic
Keywords: random walks, quantum walks, algorithm, computational science
Year: 2020

A random walk is known as a random process which describes a path including a
succession of random steps in the mathematical space. It has increasingly been
popular in various disciplines such as mathematics and computer science.
Furthermore, in quantum mechanics, quantum walks can be regarded as quantum
analogues of classical random walks. Classical random walks and quantum walks
can be used to calculate the proximity between nodes and extract the topology in
the network. Various random walk related models can be applied in different
fields, which is of great significance to downstream tasks such as link
prediction, recommendation, computer vision, semi-supervised learning, and
network embedding. In this paper, we aim to provide a comprehensive review of
classical random walks and quantum walks. We first review the knowledge of
classical random walks and quantum walks, including basic concepts and some
typical algorithms. We also compare the algorithms based on quantum walks and
classical random walks from the perspective of time complexity. Then we
introduce their applications in the field of computer science. Finally, we
discuss the open issues from the perspectives of efficiency, main-memory volume,
and computing time of existing algorithms. This study aims to contribute to this
growing area of research by exploring random walks and quantum walks together.
---

## Tensor Network Rewriting Strategies for Satisfiability and Counting

Niel de Beaudrap, Aleks Kissinger, Konstantinos Meichanetzidis

Category: symbolic
Keywords: SAT, #SAT, tensor networks, ZH-calculus, graphical reasoning, complexity theory, CSP, #CSP
Year: 2020

We provide a graphical treatment of SAT and #SAT on equal footing. Instances of
#SAT can be represented as tensor networks in a standard way. These tensor
networks are interpreted by diagrams of the ZH-calculus: a system to reason
about tensors over C in terms of diagrams built from simple generators, in which
computation may be carried out by transformations of diagrams alone. In general,
nodes of ZH diagrams take parameters over C which determine the tensor
coefficients; for the standard representation of #SAT instances, the
coefficients take the value 0 or 1. Then, by choosing the coefficients of a
diagram to range over B, we represent the corresponding instance of SAT. Thus,
by interpreting a diagram either over the boolean semiring or the complex
numbers, we instantiate either the decision or counting version of the problem.
We find that for classes known to be in P, such as 2SAT and #XORSAT, the
existence of appropriate rewrite rules allows for efficient simplification of
the diagram, producing the solution in polynomial time. In contrast, for classes
known to be NP-complete, such as 3SAT, or #P-complete, such as #2SAT, the
corresponding rewrite rules introduce hyperedges to the diagrams, in numbers
which are not easily bounded above by a polynomial. This diagrammatic approach
unifies the diagnosis of the complexity of CSPs and #CSPs and shows promise in
aiding tensor network contraction-based algorithms.
---

## Fonduer: Knowledge Base Construction from Richly Formatted Data

Sen Wu, Luke Hsiao, Xiao Cheng, Braden Hancock, Theodoros Rekatsinas, Philip Levis, Christopher Ré

Category: symbolic
Keywords: knowledge base construction, richly formatted data, multimodality, machine learning, deep learning
Year: 2018

We focus on knowledge base construction (KBC) from richly formatted data. In
contrast to KBC from text or tabular data, KBC from richly formatted data aims
to extract relations conveyed jointly via textual, structural, tabular, and
visual expressions. We introduce Fonduer, a machine-learning-based KBC system
for richly formatted data. Fonduer presents a new data model that accounts for
three challenging characteristics of richly formatted data: (1) prevalent
document-level relations, (2) multimodality, and (3) data variety. Fonduer uses
a new deep-learning model to automatically capture the representation (i.e.,
features) needed to learn how to extract relations from richly formatted data.
Finally, Fonduer provides a new programming model that enables users to convert
domain expertise, based on multiple modalities of information, to meaningful
signals of supervision for training a KBC system. Fonduer-based KBC systems are
in production for a range of use cases, including at a major online retailer. We
compare Fonduer against state-of-the-art KBC approaches in four different
domains. We show that Fonduer achieves an average improvement of 41 F1 points on
the quality of the output knowledge base—and in some cases produces up to 1.87×
the number of correct entries—compared to expert-curated public knowledge bases.
We also conduct a user study to assess the usability of Fonduer's new
programming model. We show that after using Fonduer for only 30 minutes, non-
domain experts are able to design KBC systems that achieve on average 23 F1
points higher quality than traditional machine-learning-based KBC approaches.
---

## What’s the Over/Under? Probabilistic Bounds on Information Leakage

Ian Sweet, José Manuel Calderón Trilla, Chad Scherrer, Michael Hicks, Stephen Magill

Category: symbolic
Keywords: quantitative information flow, information leakage, probabilistic bounds, abstract interpretation, sampling, symbolic execution
Year: 2018

Quantitative information flow (QIF) is concerned with measuring how much of a
secret is leaked to an adversary who observes the result of a computation that
uses it. Prior work has shown that QIF techniques based on abstract
interpretation with probabilistic polyhedra can be used to analyze the worst-
case leakage of a query, on-line, to determine whether that query can be safely
answered. While this approach can provide precise estimates, it does not scale
well. This paper shows how to solve the scalability problem by augmenting the
baseline technique with sampling and symbolic execution. We prove that our
approach never underestimates a query’s leakage (it is sound), and detailed
experimental results show that we can match the precision of the baseline
technique but with orders of magnitude better performance.
---

## World of Bits: An Open-Domain Platform for Web-Based Agents

Tianlin (Tim) Shi, Andrej Karpathy, Linxi (Jim) Fan, Jonathan Hernandez, Percy Liang

Category: symbolic
Keywords: reinforcement learning, web-based agents, World of Bits, open-domain environments, crowdsourcing
Year: 2017

While simulated game environments have greatly accelerated research in
reinforcement learning, existing environments lack the open-domain realism of
tasks in computer vision or natural language processing, which operate on
artifacts created by humans in natural, organic settings. To foster
reinforcement learning research in such settings, we introduce the World of Bits
(WoB), a platform in which agents complete tasks on the Internet by performing
low-level keyboard and mouse actions. The two main challenges are: (i) to curate
a diverse set of natural web-based tasks, and (ii) to ensure that these tasks
have a well-defined reward structure and are reproducible despite the transience
of the web. To tackle this, we develop a methodology in which crowdworkers
create tasks defined by natural language questions and provide demonstrations of
how to answer the question on real websites using keyboard and mouse; HTTP
traffic is cached to create a reproducible offline approximation of the website.
Finally, we show that agents trained via behavioral cloning and reinforcement
learning can complete a range of web-based tasks.
---

## Knowledge Representation in Bicategories of Relations

Evan Patterson

Category: symbolic
Keywords: knowledge representation, relational ologs, category theory, description logic, categorical logic
Year: 2017

We introduce the relational ontology log, or relational olog, a knowledge
representation system based on the category of sets and relations. It is
inspired by Spivak and Kent’s olog, a recent categorical framework for knowledge
representation. Relational ologs interpolate between ologs and description
logic, the dominant formalism for knowledge representation today. In this paper,
we investigate relational ologs both for their own sake and to gain insight into
the relationship between the algebraic and logical approaches to knowledge
representation. On a practical level, we show by example that relational ologs
have a friendly and intuitive—yet fully precise—graphical syntax, derived from
the string diagrams of monoidal categories. We explain several other useful
features of relational ologs not possessed by most description logics, such as a
type system and a rich, flexible notion of instance data. In a more theoretical
vein, we draw on categorical logic to show how relational ologs can be
translated to and from logical theories in a fragment of first-order logic.
Although we make extensive use of categorical language, this paper is designed
to be self-contained and has considerable expository content. The only
prerequisites are knowledge of first-order logic and the rudiments of category
theory.
---

## Formalizing Mathematical Knowledge as a Biform Theory Graph: A Case Study

Jacques Carette, William M. Farmer

Category: symbolic
Keywords: biform theory, theory graph, mathematical knowledge, axiomatic theory, algorithmic theory, cttuqe, Agda, natural number arithmetic
Year: 2017

A biform theory is a combination of an axiomatic theory and an algorithmic
theory that supports the integration of reasoning and computation. These are
ideal for formalizing algorithms that manipulate mathematical expressions. A
theory graph is a network of theories connected by meaning-preserving theory
morphisms that map the formulas of one theory to the formulas of another theory.
Theory graphs are in turn well suited for formalizing mathematical knowledge at
the most convenient level of abstraction using the most convenient vocabulary.
We are interested in the problem of whether a body of mathematical knowledge can
be effectively formalized as a theory graph of biform theories. As a test case,
we look at the graph of theories encoding natural number arithmetic. We used two
different formalisms to do this, which we describe and compare. The first is
realized in cttuqe, a version of Church’s type theory with quotation and
evaluation, and the second is realized in Agda, a dependently typed programming
language.
---

## Planning for Change in a Formal Verification of the Raft Consensus Protocol

Doug Woos, James R. Wilcox, Steve Anton, Zachary Tatlock, Michael D. Ernst, Thomas Anderson

Category: symbolic
Keywords: Formal verification, distributed systems, proof assistants, Coq, Verdi, Raft
Year: 2015

We present the first formal verification of state machine safety for the Raft
consensus protocol, a critical component of many distributed systems. We
connected our proof to previous work to establish an end-to-end guarantee that
our implementation provides linearizable state machine replication. This proof
required iteratively discovering and proving 90 system invariants. Our verified
implementation is extracted to OCaml and runs on real networks. The primary
challenge we faced during the verification process was proof maintenance, since
proving one invariant often required strengthening and updating other parts of
our proof. To address this challenge, we propose a methodology of planning for
change during verification. Our methodology adapts classical information hiding
techniques to the context of proof assistants, factors out common invariant-
strengthening patterns into custom induction principles, proves higher-order
lemmas that show any property proved about a particular component implies
analogous properties about related components, and makes proofs robust to change
using structural tactics. We also discuss how our methodology may be applied to
systems verification more broadly.
---

## In Search of an Understandable Consensus Algorithm (Extended Version)

Diego Ongaro, John Ousterhout

Category: symbolic
Keywords: Raft, consensus algorithm, Paxos, leader election, log replication, safety, cluster membership
Year: 2014

Raft is a consensus algorithm for managing a replicated log. It produces a
result equivalent to (multi-)Paxos, and it is as efficient as Paxos, but its
structure is different from Paxos; this makes Raft more understandable than
Paxos and also provides a better foundation for building practical systems. In
order to enhance understandability, Raft separates the key elements of
consensus, such as leader election, log replication, and safety, and it enforces
a stronger degree of coherency to reduce the number of states that must be
considered. Results from a user study demonstrate that Raft is easier for
students to learn than Paxos. Raft also includes a new mechanism for changing
the cluster membership, which uses overlapping majorities to guarantee safety.
---

## Propositions as Types

Philip Wadler

Category: symbolic
Keywords: Propositions as Types, Curry-Howard Isomorphism, logic, computation, programming languages
Year: 2014

Powerful insights arise from linking two fields of study previously thought
separate. Examples include Descartes’s coordinates, which links geometry to
algebra, Planck’s Quantum Theory, which links particles to waves, and Shannon’s
Information Theory, which links thermodynamics to communication. Such a
synthesis is offered by the principle of Propositions as Types, which links
logic to computation. At first sight it appears to be a simple
coincidence—almost a pun—but it turns out to be remarkably robust, inspiring the
design of automated proof assistants and programming languages, and continuing
to influence the forefronts of computing. Propositions as Types is a notion with
many names and many origins. It is closely related to the BHK Interpretation, a
view of logic developed by the intuitionists Brouwer, Heyting, and Kolmogorov in
the 1930s. It is often referred to as the Curry-Howard Isomorphism, referring to
a correspondence observed by Curry in 1934 and refined by Howard in 1969 (though
not published until 1980, in a Festschrift dedicated to Curry). Others draw
attention to significant contributions from de Bruijn’s Automath and Martin-
Löf’s Type Theory in the 1970s. Many variant names appear in the literature,
including Formulae as Types, Curry-Howard-de Bruijn Correspondence, Brouwer’s
Dictum, and others. Propositions as Types is a notion with depth. It describes a
correspondence between a given logic and a given programming language. At the
surface, it says that for each proposition in the logic there is a corresponding
type in the programming language—and vice versa. Thus we have propositions as
types. It goes deeper, in that for each proof of a given proposition, there is a
program of the corresponding type—and vice versa. Thus we also have proofs as
programs. And it goes deeper still, in that for each way to simplify a proof
there is a corresponding way to evaluate a program—and vice versa. Thus we
further have simplification of proofs as evaluation of programs. Hence, we have
not merely a shallow bijection between propositions and types, but a true
isomorphism preserving the deep structure of proofs and programs, simplification
and evaluation. Propositions as Types is a notion with breadth. It applies to a
range of logics including propositional, predicate, second-order,
intuitionistic, classical, modal, and linear. It underpins the foundations of
functional programming, explaining features including functions, records,
variants, parametric polymorphism, data abstraction, continuations, linear
types, and session types. It has inspired automated proof assistants and
programming languages including Agda, Automath, Coq, Epigram, F#, F⋆, Haskell,
LF, ML, NuPRL, Scala, Singularity, and Trellys. Propositions as Types is a
notion with mystery. Why should it be the case that intuitionistic natural
deduction, as developed by Gentzen in the 1930s, and simply-typed lambda
calculus, as developed by Church around the same time for an unrelated purpose,
should be discovered thirty years later to be essentially identical? And why
should it be the case that the same correspondence arises again and again? The
logician Hindley and the computer scientist Milner independently developed the
same type system, now dubbed Hindley-Milner. The logician Girard and the
computer scientist Reynolds independently developed the same calculus, now
dubbed Girard-Reynolds. Curry-Howard is a double-barrelled name that ensures the
existence of other double-barrelled names. Those of us that design and use
programming languages may often feel they are arbitrary, but Propositions as
Types assures us some aspects of programming are absolute.
---

## Raft Reﬂoated: Do We Have Consensus?

Heidi Howard, Malte Schwarzkopf, Anil Madhavapeddy, Jon Crowcroft

Category: symbolic
Keywords: Raft, Paxos, distributed consensus, simulation, protocol optimization
Year: 2014

The Paxos algorithm is famously difficult to reason about and even more so to
implement, despite having been synonymous with distributed consensus for over a
decade. The recently proposed Raft protocol lays claim to being a new,
understandable consensus algorithm, improving on Paxos without making
compromises in performance or correctness. In this study, we repeat the Raft
authors’ performance analysis. We developed a clean-slate implementation of the
Raft protocol and built an event-driven simulation framework for prototyping it
on experimental topologies. We propose several optimizations to the Raft
protocol and demonstrate their effectiveness under contention. Finally, we
empirically validate the correctness of the Raft protocol invariants and
evaluate Raft’s understandability claims.
---

## Propositions as Types

Philip Wadler

Category: symbolic
Keywords: Propositions as Types, Curry-Howard Isomorphism, logic, computation, type theory, programming languages
Year: 2014

Powerful insights arise from linking two fields of study previously thought
separate. Examples include Descartes’s coordinates, which links geometry to
algebra, Planck’s Quantum Theory, which links particles to waves, and Shannon’s
Information Theory, which links thermodynamics to communication. Such a
synthesis is offered by the principle of Propositions as Types, which links
logic to computation. At first sight it appears to be a simple
coincidence—almost a pun—but it turns out to be remarkably robust, inspiring the
design of automated proof assistants and programming languages, and continuing
to influence the forefronts of computing. Propositions as Types is a notion with
many names and many origins. It is closely related to the BHK Interpretation, a
view of logic developed by the intuitionists Brouwer, Heyting, and Kolmogorov in
the 1930s. It is often referred to as the Curry-Howard Isomorphism, referring to
a correspondence observed by Curry in 1934 and refined by Howard in 1969 (though
not published until 1980, in a Festschrift dedicated to Curry). Others draw
attention to significant contributions from de Bruijn’s Automath and Martin-
Löf’s Type Theory in the 1970s. Many variant names appear in the literature,
including Formulae as Types, Curry-Howard-de Bruijn Correspondence, Brouwer’s
Dictum, and others. Propositions as Types is a notion with depth. It describes a
correspondence between a given logic and a given programming language. At the
surface, it says that for each proposition in the logic there is a corresponding
type in the programming language—and vice versa. Thus we have propositions as
types. It goes deeper, in that for each proof of a given proposition, there is a
program of the corresponding type—and vice versa. Thus we also have proofs as
programs. And it goes deeper still, in that for each way to simplify a proof
there is a corresponding way to evaluate a program—and vice versa. Thus we
further have simplification of proofs as evaluation of programs. Hence, we have
not merely a shallow bijection between propositions and types, but a true
isomorphism preserving the deep structure of proofs and programs, simplification
and evaluation. Propositions as Types is a notion with breadth. It applies to a
range of logics including propositional, predicate, second-order,
intuitionistic, classical, modal, and linear. It underpins the foundations of
functional programming, explaining features including functions, records,
variants, parametric polymorphism, data abstraction, continuations, linear
types, and session types. It has inspired automated proof assistants and
programming languages including Agda, Automath, Coq, Epigram, F#, F⋆, Haskell,
LF, ML, NuPRL, Scala, Singularity, and Trellys. Propositions as Types is a
notion with mystery. Why should it be the case that intuitionistic natural
deduction, as developed by Gentzen in the 1930s, and simply-typed lambda
calculus, as developed by Church around the same time for an unrelated purpose,
should be discovered thirty years later to be essentially identical? And why
should it be the case that the same correspondence arises again and again? The
logician Hindley and the computer scientist Milner independently developed the
same type system, now dubbed Hindley-Milner. The logician Girard and the
computer scientist Reynolds independently developed the same calculus, now
dubbed Girard-Reynolds. Curry-Howard is a double-barrelled name that ensures the
existence of other double-barrelled names. Those of us that design and use
programming languages may often feel they are arbitrary, but Propositions as
Types assures us some aspects of programming are absolute.
---

## Efficient Parallel GPU Algorithms for BDD Manipulation

Miroslav N. Velev, Ping Gao

Category: symbolic
Keywords: Binary Decision Diagrams (BDDs), Boolean Satisfiability, Formal Verification, Graphics Processing Unit (GPU), Parallel Execution
Year: 2014

We present parallel algorithms for Binary Decision Diagram (BDD) manipulation
optimized for efficient execution on Graphics Processing Units (GPUs). Compared
to a sequential CPU-based BDD package with the same capabilities, our GPU
implementation achieves at least 5 orders of magnitude speedup. To the best of
our knowledge, this is the first work on using GPUs to accelerate a BDD package.
---

## Causality, Conditional Independence, and Graphical Separation in Settable Systems

Karim Chalak, Halbert White

Category: symbolic
Keywords: causality, conditional independence, d-separation, Reichenbach principle, settable systems
Year: 2010

We study the connections between conditional independence and causal relations
within the settable systems extension of the Pearl Causal Model. Our analysis
clearly distinguishes between causal notions and probabilistic notions and does
not formally rely on graphical representations. We provide definitions in terms
of functional dependence for direct, indirect, and total causality as well as
for indirect causality via and exclusive of a set of variables. We then provide
necessary and sufficient causal and probabilistic conditions for conditional
dependence among random vectors of interest in structural systems. We state and
prove the conditional Reichenbach principle of common cause, obtaining the
classical Reichenbach principle as a corollary. Finally, we relate our results
to notions of graphical separation such as d-separation and D-separation in the
artificial intelligence and machine learning literature.
---

## Regression by dependence minimization and its application to causal inference in additive noise models

Joris Mooij, Dominik Janzing, Jonas Peters, Bernhard Schölkopf

Category: symbolic
Keywords: causal inference, regression, dependence measures, kernel methods
Year: 2009

Motivated by causal inference problems, we propose a novel method for regression
that minimizes the statistical dependence between regressors and residuals. The
key advantage of this approach to regression is that it does not assume a
particular distribution of the noise, i.e., it is non-parametric with respect to
the noise distribution. We argue that the proposed regression method is well
suited to the task of causal inference in additive noise models. A practical
disadvantage is that the resulting optimization problem is generally non-convex
and can be difficult to solve. Nevertheless, we report good results on one of
the tasks of the NIPS 2008 Causality Challenge, where the goal is to distinguish
causes from effects in pairs of statistically dependent variables. In addition,
we propose an algorithm for efficiently inferring causal models from
observational data for more than two variables. The required number of
regressions and independence tests is quadratic in the number of variables,
which is a significant improvement over the simple method that tests all
possible DAGs.
---

## Typed Tagless Final Interpreters

Oleg Kiselyov

Category: symbolic
Keywords: typed tagless final, interpreters, generic programming, language extensibility, type systems
Year: 2009

The so-called ‘typed tagless final’ approach of Carette et al. has collected and
polished a number of techniques for representing typed higher-order languages in
a typed metalanguage, along with type-preserving interpretation, compilation and
partial evaluation. The approach is an alternative to the traditional, or
‘initial’ encoding of an object language as a (generalized) algebraic data type.
Both approaches permit multiple interpretations of an expression, to evaluate
it, pretty-print, etc. The final encoding represents all and only typed object
terms without resorting to generalized algebraic data types, dependent or other
fancy types. The final encoding lets us add new language forms and
interpretations without breaking the existing terms and interpreters. These
lecture notes introduce the final approach slowly and in detail, highlighting
extensibility, the solution to the expression problem, and the seemingly
impossible pattern-matching. We develop the approach further, to type-safe cast,
run-time-type representation, Dynamics, and type reconstruction. We finish with
telling examples of type-directed partial evaluation and encodings of type-and-
effect systems and linear lambda-calculus.
---

## Physics, Topology, Logic and Computation: A Rosetta Stone

John C. Baez, Mike Stay

Category: symbolic
Keywords: Feynman diagrams, quantum physics, topology, category theory, quantum computation, quantum cryptography
Year: 2009

In physics, Feynman diagrams are used to reason about quantum processes. In the
1980s, it became clear that underlying these diagrams is a powerful analogy
between quantum physics and topology. Namely, a linear operator behaves very
much like a ‘cobordism’: a manifold representing spacetime, going between two
manifolds representing space. This led to a burst of work on topological quantum
field theory and ‘quantum topology’. But this was just the beginning: similar
diagrams can be used to reason about logic, where they represent proofs, and
computation, where they represent programs. With the rise of interest in quantum
cryptography and quantum computation, it became clear that there is an extensive
network of analogies between physics, topology, logic, and computation. In this
expository paper, we make some of these analogies precise using the concept of
‘closed symmetric monoidal category’. We assume no prior knowledge of category
theory, proof theory or computer science.
---

## Physical constraints on hypercomputation

Paul Cockshott, Lewis Mackenzie, Greg Michaelson

Category: symbolic
Keywords: Hyper-computing, Quantum-computing, Computability, Quantum-measurement
Year: 2008

Many attempts to transcend the fundamental limitations to computability implied
by the Halting Problem for Turing Machines depend on the use of forms of
hypercomputation that draw on notions of infinite or continuous, as opposed to
bounded or discrete, computation. Thus, such schemes may include the deployment
of actualized rather than potential infinities of physical resources, or of
physical representations of real numbers to arbitrary precision. Here, we argue
that such bases for hypercomputation are not materially realizable and so cannot
constitute new forms of effective calculability.
---

## Zeno machines and hypercomputation

Petrus H. Potgieter

Category: symbolic
Keywords: Church-Turing Thesis, Zeno machine, Accelerated Turing machine, hypercomputation, halting problem
Year: 2008

This paper reviews the Church-Turing Thesis (or rather, theses) with reference
to their origin and application and considers some models of 'hypercomputation,'
concentrating on perhaps the most straightforward option: Zeno machines (Turing
machines with accelerating clock). The halting problem is briefly discussed in a
general context and the suggestion that it is an inevitable companion of any
reasonable computational model is emphasised. It is suggested that claims to
have 'broken the Turing barrier' could be toned down and that the important and
well-founded role of Turing computability in the mathematical sciences stands
unchallenged.
---

## Model Theory of Ultrafinitism I: Fuzzy Initial Segments of Arithmetic

Mirco A. Mannucci, Rose M. Cherubin

Category: symbolic
Keywords: Ultrafinitism, Model Theory, Arithmetic, Fuzzy Initial Segments, Proof Theory, Semantics
Year: 2008

This article is the first of an intended series of works on the model theory of
Ultrafinitism. It is roughly divided into two parts. The first one addresses
some of the issues related to ultrafinitistic programs, as well as some of the
core ideas proposed thus far. The second part of the paper presents a model of
ultrafinitistic arithmetics based on the notion of fuzzy initial segments of the
standard natural numbers series. We also introduce a proof theory and a
semantics for ultrafinitism through which feasibly consistent theories can be
treated on the same footing as their classically consistent counterparts. We
conclude with a brief sketch of a foundational program, that aims at reproducing
the transfinite within the finite realm.
---

## Model Theory of Ultrafinitism I: Fuzzy Initial Segments of Arithmetic (Preliminary Draft)

Mirco A. Mannucci, Rose M. Cherubin

Category: symbolic
Keywords: ultrafinitism, model theory, fuzzy initial segments, arithmetics, proof theory
Year: 2007

This article is the first of an intended series of works on the model theory of
Ultrafinitism. It is roughly divided into two parts. The first one addresses
some of the issues related to ultrafinitistic programs, as well as some of the
core ideas proposed thus far. The second part of the paper presents a model of
ultrafinitistic arithmetics based on the notion of fuzzy initial segments of the
standard natural numbers series. We also introduce a proof theory and a
semantics for ultrafinitism through which feasibly consistent theories can be
treated on the same footing as their classically consistent counterparts. We
conclude with a brief sketch of a foundational program that aims at reproducing
the transfinite within the finite realm.
---

## Formalising Sylow’s theorems in Coq

Laurence Rideau, Laurent Théry

Category: symbolic
Keywords: Sylow's theorems, Coq, formalization, group theory, proof assistant
Year: 2006

The document discusses the formalization of Sylow's theorems using the Coq proof
assistant. Sylow's theorems are fundamental results in group theory, and
formalizing them in Coq provides a rigorous foundation for their application in
computational contexts.
---

## Light Affine Set Theory: A Naive Set Theory of Polynomial Time

Kazushige Terui

Category: symbolic
Keywords: naive set theory, polynomial time, linear logic, light logic, substructural logics
Year: 2004

In this paper, we consider a naive set theory based on Intuitionistic Light
Affine Logic (ILAL), a simplification of Light Linear Logic (LLL) introduced by
[1], and call it Light Affine Set Theory (LAST). The simplicity of LAST allows
us to rigorously verify its polytime character. In particular, we prove that a
function over {0, 1}* is computable in polynomial time if and only if it is
provably total in LAST.
---

## From max-plus algebra to nonexpansive mappings: a nonlinear theory for discrete event systems

Jeremy Gunawardena

Category: symbolic
Keywords: Cycle time, Discrete event system, Fixed point, Max-plus semiring, Nonexpansive map, Nonlinear eigenvalue, Nonnegative matrix, Topical function
Year: 2003

Discrete event systems provide a useful abstraction for modeling a wide variety
of systems: digital circuits, communication networks, manufacturing plants, etc.
Their dynamics—stability, equilibrium states, cyclical behavior, asymptotic
average delays—are of vital importance to system designers. However, in marked
contrast to continuous dynamical systems, there has been little systematic
mathematical theory that designers can draw upon. In this paper, we survey the
development of such a theory, based on the dynamics of maps which are
nonexpansive in the '∞ norm. This has its origins in linear algebra over the
max-plus semiring but extends to a nonlinear theory that encompasses a variety
of problems arising in other mathematical disciplines. We concentrate on the
mathematical aspects and set out several open problems.
---

## Computational Mechanics: Pattern and Prediction, Structure and Simplicity

Cosma Rohilla Shalizi, James P. Crutchfield

Category: symbolic
Keywords: complexity, computation, entropy, information, pattern, statistical mechanics
Year: 2000

Computational mechanics, an approach to structural complexity, defines a
process’s causal states and gives a procedure for finding them. We show that the
causal-state representation—an 𝜖-machine—is the minimal one consistent with
accurate prediction. We establish several results on 𝜖-machine optimality and
uniqueness and on how 𝜖-machines compare to alternative representations. Further
results relate measures of randomness and structural complexity obtained from
𝜖-machines to those from ergodic and information theories.
---

## Computational Mechanics: Pattern and Prediction, Structure and Simplicity

Cosma Rohilla Shalizi, James P. Crutchfield

Category: symbolic
Keywords: complexity, computation, entropy, information, pattern, statistical mechanics
Year: 2000

Computational mechanics, an approach to structural complexity, defines a
process's causal states and gives a procedure for finding them. We show that the
causal-state representation—an ǫ-machine—is the minimal one consistent with
accurate prediction. We establish several results on ǫ-machine optimality and
uniqueness and on how ǫ-machines compare to alternative representations. Further
results relate measures of randomness and structural complexity obtained from
ǫ-machines to those from ergodic and information theories.
---

## An Introduction to Computational Group Theory

Ákos Seress

Category: symbolic
Keywords: computational group theory, algebra, GAP, Magma, representation theory, permutation groups, matrix groups
Year: 1997

Computational group theory (CGT) is a well-developed branch of computational
algebra. It deals with algebraic structures that are suitable for machine
computations, often using a concise set of generators to define large objects.
The field is supported by specialized software systems like GAP and Magma, which
are particularly effective for group computations. CGT encompasses various
aspects such as permutation groups, matrix groups, and groups defined by
generators and relators, along with representation theory. Specialized
algorithms exist for specific classes of groups, including nilpotent and
solvable groups. This survey introduces key ideas and the capabilities of
current systems in each subarea of CGT.
---

## Towards a proof theory of rewriting: The simply typed 22-calculus

Barnaby P. Hilken

Category: symbolic
Keywords: simply typed λ-calculus, rewriting, confluence, normal forms, proof theory
Year: 1996

This paper describes the simply typed 22-calculus, a language with three levels:
types, terms and rewrites. The types and terms are those of the simply typed
λ-calculus, and the rewrites are expressions denoting sequences of β-reductions
and η-expansions. An equational theory is imposed on the rewrites, based on
2-categorical justifications, and the word problem for this theory is solved by
finding a canonical expression in each equivalence class. The canonical form of
rewrites allows us to prove several properties of the calculus, including a
strong form of confluence and a classification of the long-βη-normal forms in
terms of their rewrites. Finally, we use these properties as the basic
definitions of a theory of categorical rewriting, and find that the expected
relationships between confluence, strong normalisation and normal forms hold.
---

## An information-maximisation approach to blind separation and blind deconvolution

Anthony J. Bell, Terrence J. Sejnowski

Category: symbolic
Keywords: information maximisation, blind separation, blind deconvolution, self-organising learning algorithm, Principal Components Analysis
Year: 1995

We derive a new self-organising learning algorithm which maximises the
information transferred in a network of non-linear units. The algorithm does not
assume any knowledge of the input distributions, and is defined here for the
zero-noise limit. Under these conditions, information maximisation has extra
properties not found in the linear case. The non-linearities in the transfer
function are able to pick up higher-order moments of the input distributions and
perform something akin to true redundancy reduction between units in the output
representation. This enables the network to separate statistically independent
components in the inputs: a higher-order generalisation of Principal Components
Analysis. We apply the network to the source separation (or cocktail party)
problem, successfully separating unknown mixtures of up to ten speakers. We also
show that a variant on the network architecture is able to perform blind
deconvolution (cancellation of unknown echoes and reverberation in a speech
signal). Finally, we derive dependencies of information transfer on time delays.
We suggest that information maximisation provides a unifying framework for
problems in 'blind' signal processing.
---

## Conservative Logic

E Fredkin, T Toffoli

Category: symbolic
Keywords: Conservative logic, Fredkin gate, reversible computing
Year: 1975

Abstract is not available on the first page.
---

## Combinatory Logic in Programming

V. E. Wolfengagen

Category: symbolic
Keywords: combinatory logic, programming, mathematical logic, syntax, semantics
Year: 0

This document provides an introduction to combinatory logic and its applications
in programming. Combinatory logic is a notation to eliminate variables in
mathematical logic, which can be applied to programming languages to simplify
syntax and semantics. The text explores the theoretical foundations of
combinatory logic and its practical implementations in computer programs.