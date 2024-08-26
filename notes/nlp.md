
## Evaluating Byte and Wordpiece Level Models for Massively Multilingual Semantic Parsing

Massimo Nicosia, Francesco Piccinno

Category: nlp
Keywords: semantic parsing, multilingual, ByT5, mT5, cross-lingual transfer, synthetic data augmentation
Year: 2023

Token free approaches have been successfully applied to a series of word and
span level tasks. In this work, we compare a byte-level (ByT5) and a wordpiece
based (mT5) sequence to sequence model on the 51 languages of the MASSIVE
multilingual semantic parsing dataset. We examine multiple experimental
settings: (i) zero-shot, (ii) full gold data and (iii) zero-shot with synthetic
data. By leveraging a state-of-the-art label projection method for machine
translated examples, we are able to reduce the gap in exact match accuracy to
only 5 points with respect to a model trained on gold data from all the
languages. We additionally provide insights on the cross-lingual transfer of
ByT5 and show how the model compares with respect to mT5 across all parameter
sizes.
---

## CANINE: Pre-training an Efficient Tokenization-Free Encoder for Language Representation

Jonathan H. Clark, Dan Garrette, Iulia Turc, John Wieting

Category: nlp
Keywords: tokenization, language representation, character sequences, neural encoder, multilingual
Year: 2022

Pipelined NLP systems have largely been superseded by end-to-end neural
modeling, yet nearly all commonly-used models still require an explicit
tokenization step. While recent tokenization approaches based on data-derived
subword lexicons are less brittle than manually engineered tokenizers, these
techniques are not equally suited to all languages, and the use of any fixed
vocabulary may limit a model’s ability to adapt. In this paper, we present
CANINE, a neural encoder that operates directly on character sequences—without
explicit tokenization or vocabulary—and a pre-training strategy that operates
either directly on characters or optionally uses subwords as a soft inductive
bias. To use its finer-grained input effectively and efficiently, CANINE
combines downsampling, which reduces the input sequence length, with a deep
transformer stack, which encodes context. CANINE outperforms a comparable mBERT
model by 5.7 F1 on TYDI QA, a challenging multilingual benchmark, despite having
fewer model parameters.
---

## RankT5: Fine-Tuning T5 for Text Ranking with Ranking Losses

Honglei Zhuang, Zhen Qin, Rolf Jagerman, Kai Hui, Ji Ma, Jing Lu, Jianmo Ni, Xuanhui Wang, Michael Berdersky

Category: nlp
Keywords: text ranking, T5, sequence-to-sequence models, ranking losses, pretrained language models
Year: 2022

Recently, substantial progress has been made in text ranking based on pretrained
language models such as BERT. However, there are limited studies on how to
leverage more powerful sequence-to-sequence models such as T5. Existing attempts
usually formulate text ranking as classification and rely on postprocessing to
obtain a ranked list. In this paper, we propose RankT5 and study two T5-based
ranking model structures, an encoder-decoder and an encoder-only one, so that
they not only can directly output ranking scores for each query-document pair,
but also can be fine-tuned with “pairwise” or “listwise” ranking losses to
optimize ranking performances. Our experiments show that the proposed models
with ranking losses can achieve substantial ranking performance gains on
different public text ranking datasets. Moreover, when fine-tuned with listwise
ranking losses, the ranking model appears to have better zero-shot ranking
performance on out-of-domain datasets compared to the model fine-tuned with
classification losses.
---

## Efficient Few-Shot Learning Without Prompts

Lewis Tunstall, Nils Reimers, Unso Eun Seo Jo, Luke Bates, Daniel Korat, Moshe Wasserblat, Oren Pereg

Category: nlp
Keywords: Few-shot learning, Sentence Transformers, Fine-tuning, NLP, Multilingual
Year: 2022

Recent few-shot methods, such as parameter-efficient fine-tuning (PEFT) and
pattern exploiting training (PET), have achieved impressive results in label-
scarce settings. However, they are difficult to employ since they are subject to
high variability from manually crafted prompts, and typically require billion-
parameter language models to achieve high accuracy. To address these
shortcomings, we propose SETFIT (Sentence Transformer Fine-tuning), an efficient
and prompt-free framework for few-shot fine-tuning of Sentence Transformers
(ST). SETFIT works by first fine-tuning a pretrained ST on a small number of
text pairs, in a contrastive Siamese manner. The resulting model is then used to
generate rich text embeddings, which are used to train a classification head.
This simple framework requires no prompts or verbalizers, and achieves high
accuracy with orders of magnitude less parameters than existing techniques. Our
experiments show that SETFIT obtains comparable results with PEFT and PET
techniques, while being an order of magnitude faster to train. We also show that
SETFIT can be applied in multilingual settings by simply switching the ST body.
---

## Context Limitations Make Neural Language Models More Human-Like

Tatsuki Kuribayashi, Yohei Oseki, Ana Brassard, Kentaro Inui

Category: nlp
Keywords: natural language processing, neural language models, context access, human-like processing, cognitive plausibility
Year: 2022

Do modern natural language processing (NLP) models exhibit human-like language
processing? How can they be made more human-like? These questions are motivated
by psycholinguistic studies for understanding human language processing as well
as engineering efforts. In this study, we demonstrate the discrepancies in
context access between modern neural language models (LMs) and humans in
incremental sentence processing. Additional context limitation was needed to
make LMs better simulate human reading behavior. Our analyses also showed that
human-LM gaps in memory access are associated with specific syntactic
constructions; incorporating additional syntactic factors into LMs’ context
access could enhance their cognitive plausibility.
---

## Learning to summarize from human feedback

Nisan Stiennon, Long Ouyang, Jeff Wu, Daniel M. Ziegler, Ryan Lowe, Chelsea Voss, Alec Radford, Dario Amodei, Paul Christiano

Category: nlp
Keywords: language models, summarization, human feedback, reinforcement learning, reward model
Year: 2022

As language models become more powerful, training and evaluation are
increasingly bottlenecked by the data and metrics used for a particular task.
For example, summarization models are often trained to predict human reference
summaries and evaluated using ROUGE, but both of these metrics are rough proxies
for what we really care about—summary quality. In this work, we show that it is
possible to significantly improve summary quality by training a model to
optimize for human preferences. We collect a large, high-quality dataset of
human comparisons between summaries, train a model to predict the human-
preferred summary, and use that model as a reward function to fine-tune a
summarization policy using reinforcement learning. We apply our method to a
version of the TL;DR dataset of Reddit posts and find that our models
significantly outperform both human reference summaries and much larger models
fine-tuned with supervised learning alone. Our models also transfer to CNN/DM
news articles, producing summaries nearly as good as the human reference without
any news-specific fine-tuning. We conduct extensive analyses to understand our
human feedback dataset and fine-tuned models. We establish that our reward model
generalizes to new datasets, and that optimizing our reward model results in
better summaries than optimizing ROUGE according to humans. We hope the evidence
from our paper motivates machine learning researchers to pay closer attention to
how their training loss affects the model behavior they actually want.
---

## Exploiting Cloze Questions for Few Shot Text Classification and Natural Language Inference

Timo Schick, Hinrich Schütze

Category: nlp
Keywords: few-shot learning, text classification, natural language inference, semi-supervised learning, language models
Year: 2021

Some NLP tasks can be solved in a fully unsupervised fashion by providing a
pretrained language model with "task descriptions" in natural language (e.g.,
Radford et al., 2019). While this approach underperforms its supervised
counterpart, we show in this work that the two ideas can be combined: We
introduce Pattern-Exploiting Training (PET), a semi-supervised training
procedure that reformulates input examples as cloze-style phrases to help
language models understand a given task. These phrases are then used to assign
soft labels to a large set of unlabeled examples. Finally, standard supervised
training is performed on the resulting training set. For several tasks and
languages, PET outperforms supervised training and strong semi-supervised
approaches in low-resource settings by a large margin.
---

## Few-Shot Text Generation with Pattern-Exploiting Training

Timo Schick, Hinrich Schütze

Category: nlp
Keywords: Few-Shot Learning, Text Generation, Pattern-Exploiting Training, Language Models, NLP
Year: 2021

Providing pretrained language models with simple task descriptions or prompts in
natural language yields impressive few-shot results for a wide range of text
classification tasks when combined with gradient-based learning from examples.
In this paper, we show that the underlying idea can also be applied to text
generation tasks: We adapt Pattern-Exploiting Training (PET), a recently
proposed few-shot approach, for finetuning generative language models on text
generation tasks. On several text summarization and headline generation
datasets, our proposed variant of PET gives consistent improvements over a
strong baseline in few-shot settings.
---

## A Graph-Based Neural Model for End-to-End Frame Semantic Parsing

Zhichao Lin, Yueheng Sun, Meishan Zhang

Category: nlp
Keywords: frame semantic parsing, FrameNet, natural language processing, graph-based model, end-to-end neural model
Year: 2021

Frame semantic parsing is a semantic analysis task based on FrameNet which has
received great attention recently. The task usually involves three subtasks
sequentially: (1) target identification, (2) frame classification, and (3)
semantic role labeling. The three subtasks are closely related while previous
studies model them individually, which ignores their internal connections and
meanwhile induces error propagation problems. In this work, we propose an end-
to-end neural model to tackle the task jointly. Concretely, we exploit a graph-
based method, regarding frame semantic parsing as a graph construction problem.
All predicates and roles are treated as graph nodes, and their relations are
taken as graph edges. Experiment results on two benchmark datasets of frame
semantic parsing show that our method is highly competitive, resulting in better
performance than pipeline models.
---

## Boosting Search Engines with Interactive Agents

Leonard Adolphs, Benjamin Boerschinger, Christian Buck, Michelle Chen Huebscher, Massimiliano Ciaramita, Lasse Espeholt, Thomas Hofmann, Yannic Kilcher

Category: nlp
Keywords: interactive agents, search engines, transformer, reinforcement learning, information retrieval
Year: 2021

Can machines learn to use a search engine as an interactive tool for finding
information? That would have far reaching consequences for making the world’s
knowledge more accessible. This paper presents first steps in designing agents
that learn meta-strategies for contextual query refinements. Our approach uses
machine reading to guide the selection of refinement terms from aggregated
search results. Agents are then empowered with simple but effective search
operators to exert fine-grained and transparent control over queries and search
results. We develop a novel way of generating synthetic search sessions, which
leverages the power of transformer-based generative language models through
(self-)supervised learning. We also present a reinforcement learning agent with
dynamically constrained actions that can learn interactive search strategies
completely from scratch. In both cases, we obtain significant improvements over
one-shot search with a strong information retrieval baseline. Finally, we
provide an in-depth analysis of the learned search policies.
---

## Self-Diagnosis and Self-Debiasing: A Proposal for Reducing Corpus-Based Bias in NLP

Timo Schick, Sahana Udupa, Hinrich Schütze

Category: nlp
Keywords: language models, bias, debiasing, natural language processing, decoding algorithm
Year: 2021

When trained on large, unfiltered crawls from the internet, language models pick
up and reproduce all kinds of undesirable biases that can be found in the data:
they often generate racist, sexist, violent or otherwise toxic language. As
large models often require millions of training examples to achieve good
performance, it is difficult to completely prevent them from being exposed to
such content. In this paper, we investigate whether pretrained language models
at least know when they exhibit some undesirable bias or produce toxic content.
Based on our findings, we propose a decoding algorithm that reduces the
probability of a model producing problematic text given only a textual
description of the undesired behavior. This algorithm does not rely on manually
curated word lists, nor does it require any training data or changes to the
model’s parameters. While our approach does by no means eliminate the issue of
language models generating biased text, we believe it to be an important step in
this direction.
---

## Learning to Reason for Text Generation from Scientific Tables

Nafise Sadat Moosavi, Andreas Rücklé, Dan Roth, Iryna Gurevych

Category: nlp
Keywords: data-to-text generation, scientific tables, reasoning-aware generation, arithmetic reasoning, NLP
Year: 2021

In this paper, we introduce SciGen, a new challenge dataset for the task of
reasoning-aware data-to-text generation consisting of tables from scientific
articles and their corresponding descriptions. Describing scientific tables goes
beyond the surface realization of the table content and requires reasoning over
table values. The unique properties of SciGen are that (1) tables mostly contain
numerical values, and (2) the corresponding descriptions require arithmetic
reasoning. SciGen is therefore the first dataset that assesses the arithmetic
reasoning capabilities of generation models on complex input structures, i.e.,
tables from scientific articles. We study the effectiveness of state-of-the-art
data-to-text generation models on SciGen and evaluate the results using common
metrics as well as human evaluation. Our results and analyses show that (a)
while humans like to reason for describing scientific tables, the ability of
state-of-the-art models is severely limited on this task, (b) while adding more
training data improves the results, it is not the solution for reasoning-aware
text generation, and (c) one of the main bottlenecks for this task is the lack
of proper automatic evaluation metrics. The data, code, and annotations for
human evaluation will be available at https://github.com/UKPLab/SciGen. SciGen
opens new avenues for future research in reasoning-aware text generation and
evaluation.
---

## Exploiting Cloze Questions for Few Shot Text Classification and Natural Language Inference

Timo Schick, Hinrich Schütze

Category: nlp
Keywords: NLP, few-shot learning, semi-supervised learning, language models, Pattern-Exploiting Training
Year: 2021

Some NLP tasks can be solved in a fully unsupervised fashion by providing a
pretrained language model with “task descriptions” in natural language (e.g.,
Radford et al., 2019). While this approach underperforms its supervised
counterpart, we show in this work that the two ideas can be combined: We
introduce Pattern-Exploiting Training (PET), a semi-supervised training
procedure that reformulates input examples as cloze-style phrases to help
language models understand a given task. These phrases are then used to assign
soft labels to a large set of unlabeled examples. Finally, standard supervised
training is performed on the resulting training set. For several tasks and
languages, PET outperforms supervised training and strong semi-supervised
approaches in low-resource settings by a large margin.
---

## High-Precision Extraction of Emerging Concepts from Scientific Literature

Daniel King, Doug Downey, Daniel S. Weld

Category: nlp
Keywords: Concept extraction, scientific literature, citation graph
Year: 2020

Identification of new concepts in scientific literature can help power faceted
search, scientific trend analysis, knowledge-base construction, and more, but
current methods are lacking. Manual identification cannot keep up with the
torrent of new publications, while the precision of existing automatic
techniques is too low for many applications. We present an unsupervised concept
extraction method for scientific literature that achieves much higher precision
than previous work. Our approach relies on a simple but novel intuition: each
scientific concept is likely to be introduced or popularized by a single paper
that is disproportionately cited by subsequent papers mentioning the concept.
From a corpus of computer science papers on arXiv, we find that our method
achieves a Precision@1000 of 99%, compared to 86% for prior work, and a
substantially better precision-yield trade-off across the top 15,000
extractions. To stimulate research in this area, we release our code and data.
---

## Torch-Struct: Deep Structured Prediction Library

Alexander M. Rush

Category: nlp
Keywords: structured prediction, NLP, deep learning, Torch-Struct, auto-differentiation
Year: 2020

The literature on structured prediction for NLP describes a rich collection of
distributions and algorithms over sequences, segmentations, alignments, and
trees; however, these algorithms are difficult to utilize in deep learning
frameworks. We introduce Torch-Struct, a library for structured prediction
designed to take advantage of and integrate with vectorized, auto-
differentiation based frameworks. Torch-Struct includes a broad collection of
probabilistic structures accessed through a simple and flexible distribution-
based API that connects to any deep learning model. The library utilizes
batched, vectorized operations and exploits auto-differentiation to produce
readable, fast, and testable code. Internally, we also include a number of
general-purpose optimizations to provide cross-algorithm efficiency. Experiments
show significant performance gains over fast baselines and case-studies
demonstrate the benefits of the library. Torch-Struct is available at
https://github.com/harvardnlp/pytorch-struct.
---

## Acknowledgement Entity Recognition in CORD-19 Papers

Jian Wu, Pei Wang, Xin Wei, Sarah Michele Rajtmajer, C. Lee Giles, Christopher Griffin

Category: nlp
Keywords: acknowledgement entity recognition, text mining, named entity recognition, scholarly papers, data analysis
Year: 2020

Acknowledgements are ubiquitous in scholarly papers. Existing acknowledgement
entity recognition methods assume all named entities are acknowledged. Here, we
examine the nuances between acknowledged and named entities by analyzing
sentence structure. We develop an acknowledgement extraction system, ACKEXTRACT
based on open-source text mining software and evaluate our method using manually
labeled data. ACKEXTRACT uses the PDF of a scholarly paper as input and outputs
acknowledgement entities. Results show an overall performance of F1 = 0.92. We
built a supplementary database by linking CORD-19 papers with acknowledgement
entities extracted by ACKEXTRACT including persons and organizations and find
that only up to 50–60% of named entities are actually acknowledged. We further
analyze chronological trends of acknowledgement entities in CORD-19 papers. All
codes and labeled data are publicly available at https://github.com/lamps-
lab/ackextract.
---

## Document-Level Definition Detection in Scholarly Documents: Existing Models, Error Analyses, and Future Directions

Dongyeop Kang, Andrew Head, Risham Sidhu, Kyle Lo, Daniel S. Weld, Marti A. Hearst

Category: nlp
Keywords: definition detection, scholarly documents, error analysis, HEDDEx, transformer encoders
Year: 2020

The task of definition detection is important for scholarly papers, because
papers often make use of technical terminology that may be unfamiliar to
readers. Despite prior work on definition detection, current approaches are far
from being accurate enough to use in real-world applications. In this paper, we
first perform in-depth error analysis of the current best performing definition
detection system and discover major causes of errors. Based on this analysis, we
develop a new definition detection system, HEDDEx, that utilizes syntactic
features, transformer encoders, and heuristic filters, and evaluate it on a
standard sentence-level benchmark. Because current benchmarks evaluate randomly
sampled sentences, we propose an alternative evaluation that assesses every
sentence within a document. This allows for evaluating recall in addition to
precision. HEDDEx outperforms the leading system on both the sentence-level and
the document-level tasks, by 12.7 F1 points and 14.4 F1 points, respectively. We
note that performance on the high-recall document-level task is much lower than
in the standard evaluation approach, due to the necessity of incorporation of
document structure as features. We discuss remaining challenges in document-
level definition detection, ideas for improvements, and potential issues for the
development of reading aid applications.
---

## Document-Level Definition Detection in Scholarly Documents: Existing Models, Error Analyses, and Future Directions

Dongyeop Kang, Andrew Head, Risham Sidhu, Kyle Lo, Daniel S. Weld, Marti A. Hearst

Category: nlp
Keywords: definition detection, scholarly documents, error analysis, transformer, natural language processing
Year: 2020

The task of definition detection is important for scholarly papers, because
papers often make use of technical terminology that may be unfamiliar to
readers. Despite prior work on definition detection, current approaches are far
from being accurate enough to use in real-world applications. In this paper, we
first perform in-depth error analysis of the current best performing definition
detection system and discover major causes of errors. Based on this analysis, we
develop a new definition detection system, HEDDEx, that utilizes syntactic
features, transformer encoders, and heuristic filters, and evaluate it on a
standard sentence-level benchmark. Because current benchmarks evaluate randomly
sampled sentences, we propose an alternative evaluation that assesses every
sentence within a document. This allows for evaluating recall in addition to
precision. HEDDEx outperforms the leading system on both the sentence-level and
the document-level tasks, by 12.7 F1 points and 14.4 F1 points, respectively. We
note that performance on the high-recall document-level task is much lower than
in the standard evaluation approach, due to the necessity of incorporation of
document structure as features. We discuss remaining challenges in document-
level definition detection, ideas for improvements, and potential issues for the
development of reading aid applications.
---

## SCIREX: A Challenge Dataset for Document-Level Information Extraction

Sarthak Jain, Madeleine van Zuylen, Hannaneh Hajishirzi, Iz Beltagy

Category: nlp
Keywords: information extraction, document-level IE, scientific articles, N-ary relation identification, neural models
Year: 2020

Extracting information from full documents is an important problem in many
domains, but most previous work focuses on identifying relationships within a
sentence or a paragraph. It is challenging to create a large-scale information
extraction (IE) dataset at the document level since it requires an understanding
of the whole document to annotate entities and their document-level
relationships that usually span beyond sentences or even sections. In this
paper, we introduce SCIREX, a document level IE dataset that encompasses
multiple IE tasks, including salient entity identification and document level
N-ary relation identification from scientific articles. We annotate our dataset
by integrating automatic and human annotations, leveraging existing scientific
knowledge resources. We develop a neural model as a strong baseline that extends
previous state-of-the-art IE models to document-level IE. Analyzing the model
performance shows a significant gap between human performance and current
baselines, inviting the community to use our dataset as a challenge to develop
document-level IE models. Our data and code are publicly available at
https://github.com/allenai/SciREX.
---

## Don’t Stop Pretraining: Adapt Language Models to Domains and Tasks

Suchin Gururangan, Ana Marasović, Swabha Swayamdipta, Kyle Lo, Iz Beltagy, Doug Downey, Noah A. Smith

Category: nlp
Keywords: language models, pretraining, domain adaptation, task adaptation, NLP
Year: 2020

Language models pretrained on text from a wide variety of sources form the
foundation of today’s NLP. In light of the success of these broad-coverage
models, we investigate whether it is still helpful to tailor a pretrained model
to the domain of a target task. We present a study across four domains
(biomedical and computer science publications, news, and reviews) and eight
classification tasks, showing that a second phase of pretraining in-domain
(domain-adaptive pretraining) leads to performance gains, under both high- and
low-resource settings. Moreover, adapting to the task’s unlabeled data (task-
adaptive pretraining) improves performance even after domain-adaptive
pretraining. Finally, we show that adapting to a task corpus augmented using
simple data selection strategies is an effective alternative, especially when
resources for domain-adaptive pretraining might be unavailable. Overall, we
consistently find that multi-phase adaptive pretraining offers large gains in
task performance.
---

## Every Document Owns Its Structure: Inductive Text Classification via Graph Neural Networks

Yufeng Zhang, Xueli Yu, Zeyu Cui, Shu Wu, Zhongzhen Wen, Liang Wang

Category: nlp
Keywords: text classification, Graph Neural Networks, inductive learning, word representations, document embedding
Year: 2020

Text classification is fundamental in natural language processing (NLP), and
Graph Neural Networks (GNN) are recently applied in this task. However, the
existing graph-based works can neither capture the contextual word relationships
within each document nor fulfill the inductive learning of new words. In this
work, to overcome such problems, we propose TextING for inductive text
classification via GNN. We first build individual graphs for each document and
then use GNN to learn the fine-grained word representations based on their local
structures, which can also effectively produce embeddings for unseen words in
the new document. Finally, the word nodes are incorporated as the document
embedding. Extensive experiments on four benchmark datasets show that our method
outperforms state-of-the-art text classification methods.
---

## Automatically Identifying Words That Can Serve as Labels for Few-Shot Text Classification

Timo Schick, Helmut Schmid, Hinrich Schütze

Category: nlp
Keywords: few-shot text classification, language models, automatic label mapping, PET, text classification
Year: 2020

A recent approach for few-shot text classification is to convert textual inputs
to cloze questions that contain some form of task description, process them with
a pretrained language model and map the predicted words to labels. Manually
defining this mapping between words and labels requires both domain expertise
and an understanding of the language model’s abilities. To mitigate this issue,
we devise an approach that automatically finds such a mapping given small
amounts of training data. For a number of tasks, the mapping found by our
approach performs almost as well as hand-crafted label-to-word mappings.
---

## Dense Passage Retrieval for Open-Domain Question Answering

Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, Wen-tau Yih

Category: nlp
Keywords: open-domain question answering, dense passage retrieval, dual-encoder framework, TF-IDF, BM25, embedding
Year: 2020

Open-domain question answering relies on efficient passage retrieval to select
candidate contexts, where traditional sparse vector space models, such as TF-IDF
or BM25, are the de facto method. In this work, we show that retrieval can be
practically implemented using dense representations alone, where embeddings are
learned from a small number of questions and passages by a simple dual-encoder
framework. When evaluated on a wide range of open-domain QA datasets, our dense
retriever outperforms a strong Lucene-BM25 system greatly by 9%-19% absolute in
terms of top-20 passage retrieval accuracy, and helps our end-to-end QA system
establish new state-of-the-art on multiple open-domain QA benchmarks.
---

## Automatically Identifying Words That Can Serve as Labels for Few-Shot Text Classification

Timo Schick, Helmut Schmid, Hinrich Schütze

Category: nlp
Keywords: few-shot text classification, language models, PET, automatic labels, textual inputs
Year: 2020

A recent approach for few-shot text classification is to convert textual inputs
to cloze questions that contain some form of task description, process them with
a pretrained language model and map the predicted words to labels. Manually
defining this mapping between words and labels requires both domain expertise
and an understanding of the language model’s abilities. To mitigate this issue,
we devise an approach that automatically finds such a mapping given small
amounts of training data. For a number of tasks, the mapping found by our
approach performs almost as well as hand-crafted label-to-word mappings.
---

## From Standard Summarization to New Tasks and Beyond: Summarization with Manifold Information

Shen Gao, Xiuying Chen, Zhaochun Ren, Dongyan Zhao, Rui Yan

Category: nlp
Keywords: text summarization, new summarization tasks, real-world applications, manifold information, NLP
Year: 2020

Text summarization is the research area aiming at creating a short and condensed
version of the original document, which conveys the main idea of the document in
a few words. This research topic has started to attract the attention of a large
community of researchers, and it is nowadays counted as one of the most
promising research areas. In general, text summarization algorithms aim at using
a plain text document as input and then output a summary. However, in real-world
applications, most of the data is not in a plain text format. Instead, there is
much manifold information to be summarized, such as the summary for a web page
based on a query in the search engine, extreme long document (e.g., academic
paper), dialog history and so on. In this paper, we focus on the survey of these
new summarization tasks and approaches in the real-world application.
---

## Fact or Fiction: Verifying Scientific Claims

David Wadden, Shanchuan Lin, Kyle Lo, Lucy Lu Wang, Madeleine van Zuylen, Arman Cohan, Hannaneh Hajishirzi

Category: nlp
Keywords: scientific claim verification, SCIFACT, COVID-19, domain adaptation, NLP
Year: 2020

We introduce scientific claim verification, a new task to select abstracts from
the research literature containing evidence that SUPPORTS or REFUTES a given
scientific claim, and to identify rationales justifying each decision. To study
this task, we construct SCIFACT, a dataset of 1.4K expert-written scientific
claims paired with evidence-containing abstracts annotated with labels and
rationales. We develop baseline models for SCIFACT, and demonstrate that simple
domain adaptation techniques substantially improve performance compared to
models trained on Wikipedia or political news. We show that our system is able
to verify claims related to COVID-19 by identifying evidence from the CORD-19
corpus. Our experiments indicate that SCIFACT will provide a challenging testbed
for the development of new systems designed to retrieve and reason over corpora
containing specialized domain knowledge.
---

## S2ORC: The Semantic Scholar Open Research Corpus

Kyle Lo, Lucy Lu Wang, Mark Neumann, Rodney Kinney, Daniel S. Weld

Category: nlp
Keywords: academic papers, corpus, text mining, natural language processing, bibliographic references
Year: 2020

We introduce S2ORC, a large corpus of 81.1M English-language academic papers
spanning many academic disciplines. The corpus consists of rich metadata, paper
abstracts, resolved bibliographic references, as well as structured full text
for 8.1M open access papers. Full text is annotated with automatically-detected
inline mentions of citations, figures, and tables, each linked to their
corresponding paper objects. In S2ORC, we aggregate papers from hundreds of
academic publishers and digital archives into a unified source, and create the
largest publicly-available collection of machine-readable academic text to date.
We hope this resource will facilitate research and development of tools and
tasks for text mining over academic text.
---

## TLDR: Extreme Summarization of Scientific Documents

Isabel Cachola, Kyle Lo, Arman Cohan, Daniel S. Weld

Category: nlp
Keywords: TLDR generation, extreme summarization, scientific papers, SCITLDR dataset, CATTS learning strategy
Year: 2020

We introduce TLDR generation, a new form of extreme summarization, for
scientific papers. TLDR generation involves high source compression and requires
expert background knowledge and understanding of complex domain-specific
language. To facilitate study on this task, we introduce SCITLDR, a new multi-
target dataset of 5.4K TLDRs over 3.2K papers. SCITLDR contains both author-
written and expert-derived TLDRs, where the latter are collected using a novel
annotation protocol that produces high-quality summaries while minimizing
annotation burden. We propose CATTS, a simple yet effective learning strategy
for generating TLDRs that exploits titles as an auxiliary training signal. CATTS
improves upon strong baselines under both automated metrics and human
evaluations. Data and code are publicly available at
https://github.com/allenai/scitldr.
---

## Stolen Probability: A Structural Weakness of Neural Language Models

David Demeter, Gregory Kimmel, Doug Downey

Category: nlp
Keywords: Neural Network Language Models, probability distributions, softmax function, embedding space, stolen probability effect
Year: 2020

Neural Network Language Models (NNLMs) generate probability distributions by
applying a softmax function to a distance metric formed by taking the dot
product of a prediction vector with all word vectors in a high-dimensional
embedding space. The dot-product distance metric forms part of the inductive
bias of NNLMs. Although NNLMs optimize well with this inductive bias, we show
that this results in a sub-optimal ordering of the embedding space that
structurally impoverishes some words at the expense of others when assigning
probability. We present numerical, theoretical and empirical analyses showing
that words on the interior of the convex hull in the embedding space have their
probability bounded by the probabilities of the words on the hull.
---

## SCIREX: A Challenge Dataset for Document-Level Information Extraction

Sarthak Jain, Madeleine van Zuylen, Hannaneh Hajishirzi, Iz Beltagy

Category: nlp
Keywords: information extraction, document-level information extraction, scientific articles, N-ary relation identification, SCIREX
Year: 2020

Extracting information from full documents is an important problem in many
domains, but most previous work focuses on identifying relationships within a
sentence or a paragraph. It is challenging to create a large-scale information
extraction (IE) dataset at the document level since it requires an understanding
of the whole document to annotate entities and their document-level
relationships that usually span beyond sentences or even sections. In this
paper, we introduce SCIREX, a document-level IE dataset that encompasses
multiple IE tasks, including salient entity identification and document-level
N-ary relation identification from scientific articles. We annotate our dataset
by integrating automatic and human annotations, leveraging existing scientific
knowledge resources. We develop a neural model as a strong baseline that extends
previous state-of-the-art IE models to document-level IE. Analyzing the model
performance shows a significant gap between human performance and current
baselines, inviting the community to use our dataset as a challenge to develop
document-level IE models. Our data and code are publicly available at
https://github.com/allenai/SciREX
---

## Tree-Structured Attention with Hierarchical Accumulation

Xuan-Phi Nguyen, Shafiq Joty, Steven C.H. Hoi, Richard Socher

Category: nlp
Keywords: hierarchical structures, constituency trees, natural language processing, Transformers, Tree-LSTM, self-attention, translation tasks, text classification
Year: 2020

Incorporating hierarchical structures like constituency trees has been shown to
be effective for various natural language processing (NLP) tasks. However, it is
evident that state-of-the-art (SOTA) sequence-based models like the Transformer
struggle to encode such structures inherently. On the other hand, dedicated
models like the Tree-LSTM, while explicitly modeling hierarchical structures, do
not perform as efficiently as the Transformer. In this paper, we attempt to
bridge this gap with “Hierarchical Accumulation” to encode parse tree structures
into self-attention at constant time complexity. Our approach outperforms SOTA
methods in four IWSLT translation tasks and the WMT’14 English-German
translation task. It also yields improvements over Transformer and Tree-LSTM on
three text classification tasks. We further demonstrate that using hierarchical
priors can compensate for data shortage, and that our model prefers phrase-level
attentions over token-level attentions.
---

## Every Document Owns Its Structure: Inductive Text Classification via Graph Neural Networks

Zeyu Cui, Zhongzhen Wen, Other authors not listed

Category: nlp
Keywords: text classification, graph neural networks, inductive learning, document structure
Year: 2020

The abstract is not available on the provided page. However, based on the title,
the paper likely discusses a method for inductive text classification using
graph neural networks, emphasizing the structural uniqueness of each document.
---

## Natural Questions: a Benchmark for Question Answering Research

Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur Parikh, Chris Alberti, Danielle Epstein, Illia Polosukhin, Jacob Devlin, Kenton Lee, Kristina Toutanova, Llion Jones, Matthew Kelcey, Ming-Wei Chang, Andrew M. Dai, Jakob Uszkoreit, Quoc Le, Slav Petrov

Category: nlp
Keywords: question answering, natural language understanding, dataset, machine learning, annotation
Year: 2019

We present the Natural Questions corpus, a question answering dataset. Questions
consist of real anonymized, aggregated queries issued to the Google search
engine. An annotator is presented with a question along with a Wikipedia page
from the top 5 search results, and annotates a long answer (typically a
paragraph) and a short answer (one or more entities) if present on the page, or
marks null if no long/short answer is present. The public release consists of
307,373 training examples with single annotations; 7,830 examples with 5-way
annotations for development data; and a further 7,842 examples 5-way annotated
sequestered as test data. We present experiments validating quality of the data.
We also describe analysis of 25-way annotations on 302 examples, giving insights
into human variability on the annotation task. We introduce robust metrics for
the purposes of evaluating question answering systems; demonstrate high human
upper bounds on these metrics; and establish baseline results using competitive
methods drawn from related literature.
---

## BERT Rediscovers the Classical NLP Pipeline

Ian Tenney, Dipanjan Das, Ellie Pavlick

Category: nlp
Keywords: BERT, NLP pipeline, linguistic information, pre-trained text encoders, interpretability
Year: 2019

Pre-trained text encoders have rapidly advanced the state of the art on many NLP
tasks. We focus on one such model, BERT, and aim to quantify where linguistic
information is captured within the network. We find that the model represents
the steps of the traditional NLP pipeline in an interpretable and localizable
way, and that the regions responsible for each step appear in the expected
sequence: POS tagging, parsing, NER, semantic roles, then coreference.
Qualitative analysis reveals that the model can and often does adjust this
pipeline dynamically, revising lower-level decisions on the basis of
disambiguating information from higher-level representations.
---

## Entity, Relation, and Event Extraction with Contextualized Span Representations

David Wadden, Ulme Wennberg, Yi Luan, Hannaneh Hajishirzi

Category: nlp
Keywords: information extraction, named entity recognition, relation extraction, event extraction, contextualized span representations, DYGIE++
Year: 2019

We examine the capabilities of a unified, multi-task framework for three
information extraction tasks: named entity recognition, relation extraction, and
event extraction. Our framework (called DYGIE++) accomplishes all tasks by
enumerating, refining, and scoring text spans designed to capture local (within-
sentence) and global (cross-sentence) context. Our framework achieves state-of-
the-art results across all tasks, on four datasets from a variety of domains. We
perform experiments comparing different techniques to construct span
representations. Contextualized embeddings like BERT perform well at capturing
relationships among entities in the same or adjacent sentences, while dynamic
span graph updates model long-range cross-sentence relationships. For instance,
propagating span representations via predicted coreference links can enable the
model to disambiguate challenging entity mentions.
---

## Extractive Summarization of Long Documents by Combining Global and Local Context

Wen Xiao, Giuseppe Carenini

Category: nlp
Keywords: extractive summarization, neural networks, long documents, scientific papers, context modeling
Year: 2019

In this paper, we propose a novel neural single-document extractive
summarization model for long documents, incorporating both the global context of
the whole document and the local context within the current topic. We evaluate
the model on two datasets of scientific papers, Pubmed and arXiv, where it
outperforms previous work, both extractive and abstractive models, on ROUGE-1,
ROUGE-2 and METEOR scores. We also show that, consistently with our goal, the
benefits of our method become stronger as we apply it to longer documents.
Rather surprisingly, an ablation study indicates that the benefits of our model
seem to come exclusively from modeling the local context, even for the longest
documents.
---

## FAIRSEQ: A Fast, Extensible Toolkit for Sequence Modeling

Myle Ott, Sergey Edunov, Alexei Baevski, Angela Fan, Sam Gross, Nathan Ng, David Grangier, Michael Auli

Category: nlp
Keywords: FAIRSEQ, sequence modeling, PyTorch, translation, summarization, language modeling, text generation, distributed training, mixed-precision training
Year: 2019

FAIRSEQ is an open-source sequence modeling toolkit that allows researchers and
developers to train custom models for translation, summarization, language
modeling, and other text generation tasks. The toolkit is based on PyTorch and
supports distributed training across multiple GPUs and machines. We also support
fast mixed-precision training and inference on modern GPUs.
---

## Neural Modular Control for Embodied Question Answering

Abhishek Das, Georgia Gkioxari, Stefan Lee, Devi Parikh, Dhruv Batra

Category: nlp
Keywords: modular approach, hierarchical policy, imitation learning, reinforcement learning, Embodied Question Answering
Year: 2019

We present a modular approach for learning policies for navigation over long
planning horizons from language input. Our hierarchical policy operates at
multiple timescales, where the higher-level master policy proposes subgoals to
be executed by specialized sub-policies. Our choice of subgoals is compositional
and semantic, i.e. they can be sequentially combined in arbitrary orderings, and
assume human-interpretable descriptions (e.g. ‘exit room’, ‘find kitchen’, ‘find
refrigerator’, etc.). We use imitation learning to warm-start policies at each
level of the hierarchy, dramatically increasing sample efficiency, followed by
reinforcement learning. Independent reinforcement learning at each level of
hierarchy enables sub-policies to adapt to consequences of their actions and
recover from errors. Subsequent joint hierarchical training enables the master
policy to adapt to the sub-policies. On the challenging EQA benchmark in
House3D, requiring navigating diverse realistic indoor environments, our
approach outperforms prior work by a significant margin, both in terms of
navigation and question answering.
---

## Federated Learning of N-gram Language Models

Mingqing Chen, Ananda Theertha Suresh, Rajiv Mathews, Adeline Wong, Cyril Allauzen, Françoise Beaufays, Michael Riley

Category: nlp
Keywords: federated learning, n-gram language models, virtual keyboards, privacy, mobile devices
Year: 2019

We propose algorithms to train production-quality n-gram language models using
federated learning. Federated learning is a distributed computation platform
that can be used to train global models for portable devices such as
smartphones. Federated learning is especially relevant for applications handling
privacy-sensitive data, such as virtual keyboards, because training is performed
without the users’ data ever leaving their devices. While the principles of
federated learning are fairly generic, its methodology assumes that the
underlying models are neural networks. However, virtual keyboards are typically
powered by n-gram language models for latency reasons. We propose to train a
recurrent neural network language model using the decentralized
FederatedAveraging algorithm and to approximate this federated model server-side
with an n-gram model that can be deployed to devices for fast inference. Our
technical contributions include ways of handling large vocabularies, algorithms
to correct capitalization errors in user data, and efficient finite state
transducer algorithms to convert word language models to word-piece language
models and vice versa. The n-gram language models trained with federated
learning are compared to n-grams trained with traditional server-based
algorithms using A/B tests on tens of millions of users of a virtual keyboard.
Results are presented for two languages, American English and Brazilian
Portuguese. This work demonstrates that high-quality n-gram language models can
be trained directly on client mobile devices without sensitive training data
ever leaving the devices.
---

## Learned in Translation: Contextualized Word Vectors

Bryan McCann, James Bradbury, Caiming Xiong, Richard Socher

Category: nlp
Keywords: contextualized word vectors, LSTM encoder, machine translation, NLP tasks, transfer learning
Year: 2018

Computer vision has benefited from initializing multiple deep layers with
weights pretrained on large supervised training sets like ImageNet. Natural
language processing (NLP) typically sees initialization of only the lowest layer
of deep models with pretrained word vectors. In this paper, we use a deep LSTM
encoder from an attentional sequence-to-sequence model trained for machine
translation (MT) to contextualize word vectors. We show that adding these
context vectors (CoVe) improves performance over using only unsupervised word
and character vectors on a wide variety of common NLP tasks: sentiment analysis
(SST, IMDb), question classification (TREC), entailment (SNLI), and question
answering (SQuAD). For fine-grained sentiment analysis and entailment, CoVe
improves performance of our baseline models to the state of the art.
---

## Deep contextualized word representations

Matthew E. Peters, Mark Neumann, Mohit Iyyer, Matt Gardner, Christopher Clark, Kenton Lee, Luke Zettlemoyer

Category: nlp
Keywords: deep contextualized word representation, ELMo, natural language processing, word vectors, bidirectional language model
Year: 2018

We introduce a new type of deep contextualized word representation that models
both (1) complex characteristics of word use (e.g., syntax and semantics), and
(2) how these uses vary across linguistic contexts (i.e., to model polysemy).
Our word vectors are learned functions of the internal states of a deep
bidirectional language model (biLM), which is pretrained on a large text corpus.
We show that these representations can be easily added to existing models and
significantly improve the state of the art across six challenging NLP problems,
including question answering, textual entailment, and sentiment analysis. We
also present an analysis showing that exposing the deep internals of the pre-
trained network is crucial, allowing downstream models to mix different types of
semi-supervision signals.
---

## Efficient and Robust Question Answering from Minimal Context over Documents

Sewon Min, Victor Zhong, Richard Socher, Caiming Xiong

Category: nlp
Keywords: question answering, neural models, minimal context, sentence selection, adversarial robustness
Year: 2018

Neural models for question answering (QA) over documents have achieved
significant performance improvements. Although effective, these models do not
scale to large corpora due to their complex modeling of interactions between the
document and the question. Moreover, recent work has shown that such models are
sensitive to adversarial inputs. In this paper, we study the minimal context
required to answer the question, and find that most questions in existing
datasets can be answered with a small set of sentences. Inspired by this
observation, we propose a simple sentence selector to select the minimal set of
sentences to feed into the QA model. Our overall system achieves significant
reductions in training (up to 15 times) and inference times (up to 13 times),
with accuracy comparable to or better than the state-of-the-art on SQuAD,
NewsQA, TriviaQA, and SQuAD-Open. Furthermore, our experimental results and
analyses show that our approach is more robust to adversarial inputs.
---

## SLING: A framework for frame semantic parsing

Michael Ringgaard, Rahul Gupta, Fernando C. N. Pereira

Category: nlp
Keywords: frame semantic parsing, neural networks, LSTM, transition-based parsing, natural language understanding
Year: 2017

We describe SLING, a framework for parsing natural language into semantic
frames. SLING supports general transition-based, neural-network parsing with
bidirectional LSTM input encoding and a Transition Based Recurrent Unit (TBRU)
for output decoding. The parsing model is trained end-to-end using only the text
tokens as input. The transition system has been designed to output frame graphs
directly without any intervening symbolic representation. The SLING framework
includes an efficient and scalable frame store implementation as well as a
neural network JIT compiler for fast inference during parsing. SLING is
implemented in C++ and it is available for download on GitHub.
---

## A Generalised Quantifier Theory of Natural Language in Categorical Compositional Distributional Semantics with Bialgebras

Jules Hedges, Mehrnoosh Sadrzadeh

Category: nlp
Keywords: categorical compositional distributional semantics, generalised quantifier theory, natural language, compact closed category, bialgebras
Year: 2017

Categorical compositional distributional semantics is a model of natural
language; it combines the statistical vector space models of words with the
compositional models of grammar. We formalise in this model the generalised
quantifier theory of natural language, due to Barwise and Cooper. The underlying
setting is a compact closed category with bialgebras. We start from a generative
grammar formalisation and develop an abstract categorical compositional
semantics for it, then instantiate the abstract setting to sets and relations
and to finite dimensional vector spaces and linear maps. We prove the
equivalence of the relational instantiation to the truth theoretic semantics of
generalised quantifiers. The vector space instantiation formalises the
statistical usages of words and enables us to, for the first time, reason about
quantified phrases and sentences compositionally in distributional semantics.
---

## Character Models with Attention

Yoon Kim, Yacine Jernite, David Sontag, Alexander M. Rush

Category: nlp
Keywords: character-level models, neural machine translation, attention, recurrent neural networks, out-of-vocabulary words
Year: 2016

We demonstrate the ability of character-level neural machine translation models
to achieve competitive performance with word-level models, while providing
robustness to rare and out-of-vocabulary words. This is achieved through the use
of a character-level recurrent neural network with attention, which allows the
model to directly learn character-level representations and alignments. Our
results show that character-level models can perform translation at state-of-
the-art levels on datasets of modest size, and are competitive on larger
datasets when combined with word-level features.
---

## GloVe: Global Vectors for Word Representation

Jeffrey Pennington, Richard Socher, Christopher D. Manning

Category: nlp
Keywords: word representation, semantic vector space, log-bilinear regression, word analogy, named entity recognition
Year: 2014

Recent methods for learning vector space representations of words have succeeded
in capturing fine-grained semantic and syntactic regularities using vector
arithmetic, but the origin of these regularities has remained opaque. We analyze
and make explicit the model properties needed for such regularities to emerge in
word vectors. The result is a new global log-bilinear regression model that
combines the advantages of the two major model families in the literature:
global matrix factorization and local context window methods. Our model
efficiently leverages statistical information by training only on the nonzero
elements in a word-word co-occurrence matrix, rather than on the entire sparse
matrix or on individual context windows in a large corpus. The model produces a
vector space with meaningful substructure, as evidenced by its performance of
75% on a recent word analogy task. It also outperforms related models on
similarity tasks and named entity recognition.
---

## A Convolutional Neural Network for Modelling Sentences

Nal Kalchbrenner, Edward Grefenstette, Phil Blunsom

Category: nlp
Keywords: convolutional neural network, sentence modelling, dynamic k-max pooling, emotion prediction, sentiment prediction
Year: 2014

We introduce a Dynamic Convolutional Neural Network (DCNN) for semantic
modelling of sentences. The network uses dynamic k-max pooling, a global pooling
operation over linear sequences. The network handles input sentences of varying
length and induces a feature graph over the sentence that is capable of
explicitly capturing short and long-range relations. The model is applied to
emotion prediction and sentiment prediction and achieves state-of-the-art
results on multiple benchmarks.
---

## word2vec Explained: Deriving Mikolov et al.’s Negative-Sampling Word-Embedding Method

Yoav Goldberg, Omer Levy

Category: nlp
Keywords: word2vec, word embeddings, negative sampling, skip-gram model, neural networks
Year: 2014

The word2vec software of Tomas Mikolov and colleagues has gained a lot of
traction lately, and provides state-of-the-art word embeddings. The learning
models behind the software are described in two research papers. We found the
description of the models in these papers to be somewhat cryptic and hard to
follow. While the motivations and presentation may be obvious to the neural-
networks language-modeling crowd, we had to struggle quite a bit to figure out
the rationale behind the equations. This note is an attempt to explain equation
(4) (negative sampling) in “Distributed Representations of Words and Phrases and
their Compositionality” by Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg
Corrado, and Jeffrey Dean.
---

## Frame-Semantic Parsing

Dipanjan Das, Desai Chen, André F. T. Martins, Nathan Schneider, Noah A. Smith

Category: nlp
Keywords: frame semantics, frame-semantic parsing, FrameNet, semantic frames, natural language processing
Year: 2014

Frame semantics is a linguistic theory that has been instantiated for English in
the FrameNet lexicon. We solve the problem of frame-semantic parsing using a
two-stage statistical model that takes lexical targets (i.e., content words and
phrases) in their sentential contexts and predicts frame-semantic structures.
Given a target in context, the first stage disambiguates it to a semantic frame.
This model uses latent variables and semi-supervised learning to improve frame
disambiguation for targets unseen at training time. The second stage finds the
target’s locally expressed semantic arguments. At inference time, a fast exact
dual decomposition algorithm collectively predicts all the arguments of a frame
at once in order to respect declaratively stated linguistic constraints,
resulting in qualitatively better structures than naive local predictors. Both
components are feature-based and discriminatively trained on a small set of
annotated frame-semantic parses. On the SemEval 2007 benchmark data set, the
approach, along with a heuristic identifier of frame-evoking targets,
outperforms the prior state of the art by significant margins. Additionally, we
present experiments on the much larger FrameNet 1.5 data set. We have released
our frame-semantic parser as open-source software.
---

## Efficient Estimation of Word Representations in Vector Space

Tomas Mikolov, Kai Chen, Greg Corrado, Jeffrey Dean

Category: nlp
Keywords: word representations, vector space, neural networks, word similarity, NLP
Year: 2013

We propose two novel model architectures for computing continuous vector
representations of words from very large data sets. The quality of these
representations is measured in a word similarity task, and the results are
compared to the previously best performing techniques based on different types
of neural networks. We observe large improvements in accuracy at much lower
computational cost, i.e., it takes less than a day to learn high quality word
vectors from a 1.6 billion words data set. Furthermore, we show that these
vectors provide state-of-the-art performance on our test set for measuring
syntactic and semantic word similarities.
---

## Recurrent Neural Network Based Language Model

Tomáš Mikolov

Category: nlp
Keywords: recurrent neural networks, language model, ASR, MT
Year: 2010

The document describes the use of recurrent neural networks (RNNs) for language
modeling. It provides an overview of the model, ASR results, extensions, MT
results, and comparisons with other models. Key outcomes and future work are
also discussed.
---

## Bibliographic Meta-Data Extraction Using Probabilistic Finite State Transducers

Martin Krämer, Hagen Kaprykowsky, Daniel Keysers, Thomas Breuel

Category: nlp
Keywords: probabilistic finite state transducers, bibliographic meta-data extraction, computational linguistics, Cora dataset, BIBTEX
Year: 2005

We present the application of probabilistic finite state transducers to the task
of bibliographic meta-data extraction from scientific references. By using the
transducer approach, which is often applied successfully in computational
linguistics, we obtain a trainable and modular framework. This results in
simplicity, flexibility, and easy adaptability to changing requirements. An
evaluation on the Cora dataset that serves as a common benchmark for accuracy
measurements yields a word accuracy of 88.5%, a field accuracy of 82.6%, and an
instance accuracy of 42.7%. Based on a comparison to other published results, we
conclude that our system performs second best on the given data set using a
conceptually simple approach and implementation.
---

## A Probabilistic Parsing Method for Sentence Disambiguation

T. Fujisaki, F. Jelinek, J. Côté, E. Black, T. Nishincr

Category: nlp
Keywords: probabilistic parsing, sentence disambiguation, natural language processing, semantic constraints, pragmatic constraints
Year: 1989

Constructing a grammar to parse sentences from a natural language corpus is
challenging due to numerous ambiguities. Pure syntactic analysis can result in
many ambiguous parses. Semantic and pragmatic constraints are essential for
parsing and should be represented formally. The paper discusses the difficulty
of encoding all necessary syntactic, semantic, and pragmatic information for
disambiguation. It proposes a probabilistic parsing method that considers
statistics from past discourse, task domains, and speaker characteristics to aid
in sentence disambiguation.
---

## Unsupervised Deep Learning for NLP



Category: nlp
Keywords: unsupervised learning, deep learning, NLP, natural language processing
Year: 0

This document discusses the application of unsupervised deep learning techniques
to natural language processing (NLP). It explores various models and methods
that can be used to improve understanding and generation of natural language
without the need for labeled data.