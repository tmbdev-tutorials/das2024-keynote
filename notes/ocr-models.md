# A Review On Table Recognition Based On Deep Learning

Shi Jiyuan, Shi chunqi

2023-12-08 [entry](http://arxiv.org/abs/2312.04808v1) [pdf](http://arxiv.org/pdf/2312.04808v1)

Table recognition is using the computer to automatically understand the table,
to detect the position of the table from the document or picture, and to
correctly extract and identify the internal structure and content of the table.
After earlier mainstream approaches based on heuristic rules and machine
learning, the development of deep learning techniques has brought a new paradigm
to this field. This review mainly discusses the table recognition problem from
five aspects. The first part introduces data sets, benchmarks, and commonly used
evaluation indicators. This section selects representative data sets,
benchmarks, and evaluation indicators that are frequently used by researchers.
The second part introduces the table recognition model. This survey introduces
the development of the table recognition model, especially the table recognition
model based on deep learning. It is generally accepted that table recognition is
divided into two stages: table detection and table structure recognition. This
section introduces the models that follow this paradigm (TD and TSR). The third
part is the End-to-End method, this section introduces some scholars' attempts
to use an end-to-end approach to solve the table recognition problem once and
for all and the part are Data-centric methods, such as data augmentation,
aligning benchmarks, and other methods. The fourth part is the data-centric
approach, such as data enhancement, alignment benchmark, and so on. The fifth
part summarizes and compares the experimental data in the field of form
recognition, and analyzes the mainstream and more advantageous methods. Finally,
this paper also discusses the possible development direction and trend of form
processing in the future, to provide some ideas for researchers in the field of
table recognition. (Resource will be released at https://github.com/Wa1den-
jy/Topic-on-Table-Recognition .)


---

Nougat: Neural Optical Understanding for Academic Documents

Lukas Blecher, Guillem Cucurull, Thomas Scialom, Robert Stojnic

2023-08-25 [entry](http://arxiv.org/abs/2308.13418v1) [pdf](http://arxiv.org/pdf/2308.13418v1)

Scientific knowledge is predominantly stored in books and scientific journals,
often in the form of PDFs. However, the PDF format leads to a loss of semantic
information, particularly for mathematical expressions. We propose Nougat
(Neural Optical Understanding for Academic Documents), a Visual Transformer
model that performs an Optical Character Recognition (OCR) task for processing
scientific documents into a markup language, and demonstrate the effectiveness
of our model on a new dataset of scientific documents. The proposed approach
offers a promising solution to enhance the accessibility of scientific knowledge
in the digital age, by bridging the gap between human-readable documents and
machine-readable text. We release the models and code to accelerate future work
on scientific text recognition.

---

DocILE Benchmark for Document Information Localization and Extraction

Štěpán Šimsa, Milan Šulc, Michal Uřičář, Yash Patel, Ahmed Hamdi, Matěj Kocián, Matyáš Skalický, Jiří Matas, Antoine Doucet, Mickaël Coustaty, Dimosthenis Karatzas

2023-02-11 [entry](http://arxiv.org/abs/2302.05658v2) [pdf](http://arxiv.org/pdf/2302.05658v2)

This paper introduces the DocILE benchmark with the largest dataset of business
documents for the tasks of Key Information Localization and Extraction and Line
Item Recognition. It contains 6.7k annotated business documents, 100k
synthetically generated documents, and nearly~1M unlabeled documents for
unsupervised pre-training. The dataset has been built with knowledge of domain-
and task-specific aspects, resulting in the following key features: (i)
annotations in 55 classes, which surpasses the granularity of previously
published key information extraction datasets by a large margin; (ii) Line Item
Recognition represents a highly practical information extraction task, where key
information has to be assigned to items in a table; (iii) documents come from
numerous layouts and the test set includes zero- and few-shot cases as well as
layouts commonly seen in the training set. The benchmark comes with several
baselines, including RoBERTa, LayoutLMv3 and DETR-based Table Transformer;
applied to both tasks of the DocILE benchmark, with results shared in this
paper, offering a quick starting point for future work. The dataset, baselines
and supplementary material are available at https://github.com/rossumai/docile.

---

Unifying Vision, Text, and Layout for Universal Document Processing

Zineng Tang, Ziyi Yang, Guoxin Wang, Yuwei Fang, Yang Liu, Chenguang Zhu, Michael Zeng, Cha Zhang, Mohit Bansal

2022-12-05 [entry](http://arxiv.org/abs/2212.02623v3) [pdf](http://arxiv.org/pdf/2212.02623v3)

We propose Universal Document Processing (UDOP), a foundation Document AI model
which unifies text, image, and layout modalities together with varied task
formats, including document understanding and generation. UDOP leverages the
spatial correlation between textual content and document image to model image,
text, and layout modalities with one uniform representation. With a novel
Vision-Text-Layout Transformer, UDOP unifies pretraining and multi-domain
downstream tasks into a prompt-based sequence generation scheme. UDOP is
pretrained on both large-scale unlabeled document corpora using innovative self-
supervised objectives and diverse labeled data. UDOP also learns to generate
document images from text and layout modalities via masked image reconstruction.
To the best of our knowledge, this is the first time in the field of document AI
that one model simultaneously achieves high-quality neural document editing and
content customization. Our method sets the state-of-the-art on 8 Document AI
tasks, e.g., document understanding and QA, across diverse data domains like
finance reports, academic papers, and websites. UDOP ranks first on the
leaderboard of the Document Understanding Benchmark.

---

Knowing Where and What: Unified Word Block Pretraining for Document Understanding

Song Tao, Zijian Wang, Tiantian Fan, Canjie Luo, Can Huang

2022-07-28 [entry](http://arxiv.org/abs/2207.13979v2) [pdf](http://arxiv.org/pdf/2207.13979v2)

Due to the complex layouts of documents, it is challenging to extract
information for documents. Most previous studies develop multimodal pre-trained
models in a self-supervised way. In this paper, we focus on the embedding
learning of word blocks containing text and layout information, and propose
UTel, a language model with Unified TExt and Layout pre-training. Specifically,
we propose two pre-training tasks: Surrounding Word Prediction (SWP) for the
layout learning, and Contrastive learning of Word Embeddings (CWE) for
identifying different word blocks. Moreover, we replace the commonly used 1D
position embedding with a 1D clipped relative position embedding. In this way,
the joint training of Masked Layout-Language Modeling (MLLM) and two newly
proposed tasks enables the interaction between semantic and spatial features in
a unified way. Additionally, the proposed UTel can process arbitrary-length
sequences by removing the 1D position embedding, while maintaining competitive
performance. Extensive experimental results show UTel learns better joint
representations and achieves superior performance than previous methods on
various downstream tasks, though requiring no image modality. Code is available
at \url{https://github.com/taosong2019/UTel}.

---

TGRNet: A Table Graph Reconstruction Network for Table Structure Recognition

Wenyuan Xue, Baosheng Yu, Wen Wang, Dacheng Tao, Qingyong Li

2021-06-20 [entry](http://arxiv.org/abs/2106.10598v3) [pdf](http://arxiv.org/pdf/2106.10598v3)

A table arranging data in rows and columns is a very effective data structure,
which has been widely used in business and scientific research. Considering
large-scale tabular data in online and offline documents, automatic table
recognition has attracted increasing attention from the document analysis
community. Though human can easily understand the structure of tables, it
remains a challenge for machines to understand that, especially due to a variety
of different table layouts and styles. Existing methods usually model a table as
either the markup sequence or the adjacency matrix between different table
cells, failing to address the importance of the logical location of table cells,
e.g., a cell is located in the first row and the second column of the table. In
this paper, we reformulate the problem of table structure recognition as the
table graph reconstruction, and propose an end-to-end trainable table graph
reconstruction network (TGRNet) for table structure recognition. Specifically,
the proposed method has two main branches, a cell detection branch and a cell
logical location branch, to jointly predict the spatial location and the logical
location of different cells. Experimental results on three popular table
recognition datasets and a new dataset with table graph annotations
(TableGraph-350K) demonstrate the effectiveness of the proposed TGRNet for table
structure recognition. Code and annotations will be made publicly available.

---

Learning to Extract Semantic Structure from Documents Using Multimodal Fully Convolutional Neural Network

Xiao Yang, Ersin Yumer, Paul Asente, Mike Kraley, Daniel Kifer, C. Lee Giles

2017-06-07 [entry](http://arxiv.org/abs/1706.02337v1) [pdf](http://arxiv.org/pdf/1706.02337v1)

We present an end-to-end, multimodal, fully convolutional network for extracting
semantic structures from document images. We consider document semantic
structure extraction as a pixel-wise segmentation task, and propose a unified
model that classifies pixels based not only on their visual appearance, as in
the traditional page segmentation task, but also on the content of underlying
text. Moreover, we propose an efficient synthetic document generation process
that we use to generate pretraining data for our network. Once the network is
trained on a large set of synthetic documents, we fine-tune the network on
unlabeled real documents using a semi-supervised approach. We systematically
study the optimum network architecture and show that both our multimodal
approach and the synthetic data pretraining significantly boost the performance.

---

TableFormer: Table Structure Understanding with Transformers

Ahmed Nassar, Nikolaos Livathinos, Maksym Lysak, Peter Staar

2022-03-02 [entry](http://arxiv.org/abs/2203.01017v2) [pdf](http://arxiv.org/pdf/2203.01017v2)

Tables organize valuable content in a concise and compact representation. This
content is extremely valuable for systems such as search engines, Knowledge
Graph's, etc, since they enhance their predictive capabilities. Unfortunately,
tables come in a large variety of shapes and sizes. Furthermore, they can have
complex column/row-header configurations, multiline rows, different variety of
separation lines, missing entries, etc. As such, the correct identification of
the table-structure from an image is a non-trivial task. In this paper, we
present a new table-structure identification model. The latter improves the
latest end-to-end deep learning model (i.e. encoder-dual-decoder from PubTabNet)
in two significant ways. First, we introduce a new object detection decoder for
table-cells. In this way, we can obtain the content of the table-cells from
programmatic PDF's directly from the PDF source and avoid the training of the
custom OCR decoders. This architectural change leads to more accurate table-
content extraction and allows us to tackle non-english tables. Second, we
replace the LSTM decoders with transformer based decoders. This upgrade improves
significantly the previous state-of-the-art tree-editing-distance-score (TEDS)
from 91% to 98.5% on simple tables and from 88.7% to 95% on complex tables.

---

ViBERTgrid: A Jointly Trained Multi-Modal 2D Document Representation for Key Information Extraction from Documents

Weihong Lin, Qifang Gao, Lei Sun, Zhuoyao Zhong, Kai Hu, Qin Ren, Qiang Huo

2021-05-25 [entry](http://arxiv.org/abs/2105.11672v1) [pdf](http://arxiv.org/pdf/2105.11672v1)

Recent grid-based document representations like BERTgrid allow the simultaneous
encoding of the textual and layout information of a document in a 2D feature map
so that state-of-the-art image segmentation and/or object detection models can
be straightforwardly leveraged to extract key information from documents.
However, such methods have not achieved comparable performance to state-of-the-
art sequence- and graph-based methods such as LayoutLM and PICK yet. In this
paper, we propose a new multi-modal backbone network by concatenating a BERTgrid
to an intermediate layer of a CNN model, where the input of CNN is a document
image and the BERTgrid is a grid of word embeddings, to generate a more powerful
grid-based document representation, named ViBERTgrid. Unlike BERTgrid, the
parameters of BERT and CNN in our multimodal backbone network are trained
jointly. Our experimental results demonstrate that this joint training strategy
improves significantly the representation ability of ViBERTgrid. Consequently,
our ViBERTgrid-based key information extraction approach has achieved state-of-
the-art performance on real-world datasets.

---

PEaCE: A Chemistry-Oriented Dataset for Optical Character Recognition on Scientific Documents

Nan Zhang, Connor Heaton, Sean Timothy Okonsky, Prasenjit Mitra, Hilal Ezgi Toraman

2024-03-23 [entry](http://arxiv.org/abs/2403.15724v1) [pdf](http://arxiv.org/pdf/2403.15724v1)

Optical Character Recognition (OCR) is an established task with the objective of
identifying the text present in an image. While many off-the-shelf OCR models
exist, they are often trained for either scientific (e.g., formulae) or generic
printed English text. Extracting text from chemistry publications requires an
OCR model that is capable in both realms. Nougat, a recent tool, exhibits strong
ability to parse academic documents, but is unable to parse tables in PubMed
articles, which comprises a significant part of the academic community and is
the focus of this work. To mitigate this gap, we present the Printed English and
Chemical Equations (PEaCE) dataset, containing both synthetic and real-world
records, and evaluate the efficacy of transformer-based OCR models when trained
on this resource. Given that real-world records contain artifacts not present in
synthetic records, we propose transformations that mimic such qualities. We
perform a suite of experiments to explore the impact of patch size, multi-domain
training, and our proposed transformations, ultimately finding that models with
a small patch size trained on multiple domains using the proposed
transformations yield the best performance. Our dataset and code is available at
https://github.com/ZN1010/PEaCE.

---

Page Layout Analysis of Text-heavy Historical Documents: a Comparison of Textual and Visual Approaches

Najem-Meyer Sven, Romanello Matteo

2022-12-12 [entry](http://arxiv.org/abs/2212.13924v1) [pdf](http://arxiv.org/pdf/2212.13924v1)

Page layout analysis is a fundamental step in document processing which enables
to segment a page into regions of interest. With highly complex layouts and
mixed scripts, scholarly commentaries are text-heavy documents which remain
challenging for state-of-the-art models. Their layout considerably varies across
editions and their most important regions are mainly defined by semantic rather
than graphical characteristics such as position or appearance. This setting
calls for a comparison between textual, visual and hybrid approaches. We
therefore assess the performances of two transformers (LayoutLMv3 and RoBERTa)
and an objection-detection network (YOLOv5). If results show a clear advantage
in favor of the latter, we also list several caveats to this finding. In
addition to our experiments, we release a dataset of ca. 300 annotated pages
sampled from 19th century commentaries.

---

Chargrid-OCR: End-to-end Trainable Optical Character Recognition for Printed Documents using Instance Segmentation

Christian Reisswig, Anoop R Katti, Marco Spinaci, Johannes Höhne

2019-09-10 [entry](http://arxiv.org/abs/1909.04469v4) [pdf](http://arxiv.org/pdf/1909.04469v4)

We present an end-to-end trainable approach for Optical Character Recognition
(OCR) on printed documents. Specifically, we propose a model that predicts a) a
two-dimensional character grid (\emph{chargrid}) representation of a document
image as a semantic segmentation task and b) character boxes for delineating
character instances as an object detection task. For training the model, we
build two large-scale datasets without resorting to any manual annotation -
synthetic documents with clean labels and real documents with noisy labels. We
demonstrate experimentally that our method, trained on the combination of these
datasets, (i) outperforms previous state-of-the-art approaches in accuracy (ii)
is easily parallelizable on GPU and is, therefore, significantly faster and
(iii) is easy to train and adapt to a new domain.

---

Modelling the semantics of text in complex document layouts using graph transformer networks

Thomas Roland Barillot, Jacob Saks, Polena Lilyanova, Edward Torgas, Yachen Hu, Yuanqing Liu, Varun Balupuri, Paul Gaskell

2022-02-18 [entry](http://arxiv.org/abs/2202.09144v1) [pdf](http://arxiv.org/pdf/2202.09144v1)

Representing structured text from complex documents typically calls for
different machine learning techniques, such as language models for paragraphs
and convolutional neural networks (CNNs) for table extraction, which prohibits
drawing links between text spans from different content types. In this article
we propose a model that approximates the human reading pattern of a document and
outputs a unique semantic representation for every text span irrespective of the
content type they are found in. We base our architecture on a graph
representation of the structured text, and we demonstrate that not only can we
retrieve semantically similar information across documents but also that the
embedding space we generate captures useful semantic information, similar to
language models that work only on text sequences.

---

RegCLR: A Self-Supervised Framework for Tabular Representation Learning in the Wild

Weiyao Wang, Byung-Hak Kim, Varun Ganapathi

2022-11-02 [entry](http://arxiv.org/abs/2211.01165v1) [pdf](http://arxiv.org/pdf/2211.01165v1)

Recent advances in self-supervised learning (SSL) using large models to learn
visual representations from natural images are rapidly closing the gap between
the results produced by fully supervised learning and those produced by SSL on
downstream vision tasks. Inspired by this advancement and primarily motivated by
the emergence of tabular and structured document image applications, we
investigate which self-supervised pretraining objectives, architectures, and
fine-tuning strategies are most effective. To address these questions, we
introduce RegCLR, a new self-supervised framework that combines contrastive and
regularized methods and is compatible with the standard Vision Transformer
architecture. Then, RegCLR is instantiated by integrating masked autoencoders as
a representative example of a contrastive method and enhanced Barlow Twins as a
representative example of a regularized method with configurable input image
augmentations in both branches. Several real-world table recognition scenarios
(e.g., extracting tables from document images), ranging from standard Word and
Latex documents to even more challenging electronic health records (EHR)
computer screen images, have been shown to benefit greatly from the
representations learned from this new framework, with detection average-
precision (AP) improving relatively by 4.8% for Table, 11.8% for Column, and
11.1% for GUI objects over a previous fully supervised baseline on real-world
EHR screen images.

---

QueryForm: A Simple Zero-shot Form Entity Query Framework

Zifeng Wang, Zizhao Zhang, Jacob Devlin, Chen-Yu Lee, Guolong Su, Hao Zhang, Jennifer Dy, Vincent Perot, Tomas Pfister

2022-11-14 [entry](http://arxiv.org/abs/2211.07730v2) [pdf](http://arxiv.org/pdf/2211.07730v2)

Zero-shot transfer learning for document understanding is a crucial yet under-
investigated scenario to help reduce the high cost involved in annotating
document entities. We present a novel query-based framework, QueryForm, that
extracts entity values from form-like documents in a zero-shot fashion.
QueryForm contains a dual prompting mechanism that composes both the document
schema and a specific entity type into a query, which is used to prompt a
Transformer model to perform a single entity extraction task. Furthermore, we
propose to leverage large-scale query-entity pairs generated from form-like
webpages with weak HTML annotations to pre-train QueryForm. By unifying pre-
training and fine-tuning into the same query-based framework, QueryForm enables
models to learn from structured documents containing various entities and
layouts, leading to better generalization to target document types without the
need for target-specific training data. QueryForm sets new state-of-the-art
average F1 score on both the XFUND (+4.6%~10.1%) and the Payment (+3.2%~9.5%)
zero-shot benchmark, with a smaller model size and no additional image input.

---

Position Masking for Improved Layout-Aware Document Understanding

Anik Saha, Catherine Finegan-Dollak, Ashish Verma

2021-09-01 [entry](http://arxiv.org/abs/2109.00442v1) [pdf](http://arxiv.org/pdf/2109.00442v1)

Natural language processing for document scans and PDFs has the potential to
enormously improve the efficiency of business processes. Layout-aware word
embeddings such as LayoutLM have shown promise for classification of and
information extraction from such documents. This paper proposes a new pre-
training task called that can improve performance of layout-aware word
embeddings that incorporate 2-D position embeddings. We compare models pre-
trained with only language masking against models pre-trained with both language
masking and position masking, and we find that position masking improves
performance by over 5% on a form understanding task.

---

TDeLTA: A Light-weight and Robust Table Detection Method based on Learning Text Arrangement

Yang Fan, Xiangping Wu, Qingcai Chen, Heng Li, Yan Huang, Zhixiang Cai, Qitian Wu

2023-12-18 [entry](http://arxiv.org/abs/2312.11043v1) [pdf](http://arxiv.org/pdf/2312.11043v1)

The diversity of tables makes table detection a great challenge, leading to
existing models becoming more tedious and complex. Despite achieving high
performance, they often overfit to the table style in training set, and suffer
from significant performance degradation when encountering out-of-distribution
tables in other domains. To tackle this problem, we start from the essence of
the table, which is a set of text arranged in rows and columns. Based on this,
we propose a novel, light-weighted and robust Table Detection method based on
Learning Text Arrangement, namely TDeLTA. TDeLTA takes the text blocks as input,
and then models the arrangement of them with a sequential encoder and an
attention module. To locate the tables precisely, we design a text-
classification task, classifying the text blocks into 4 categories according to
their semantic roles in the tables. Experiments are conducted on both the text
blocks parsed from PDF and extracted by open-source OCR tools, respectively.
Compared to several state-of-the-art methods, TDeLTA achieves competitive
results with only 3.1M model parameters on the large-scale public datasets.
Moreover, when faced with the cross-domain data under the 0-shot setting, TDeLTA
outperforms baselines by a large margin of nearly 7%, which shows the strong
robustness and transferability of the proposed model.

---

Rethinking Table Recognition using Graph Neural Networks

Shah Rukh Qasim, Hassan Mahmood, Faisal Shafait

2019-05-31 [entry](http://arxiv.org/abs/1905.13391v2) [pdf](http://arxiv.org/pdf/1905.13391v2)

Document structure analysis, such as zone segmentation and table recognition, is
a complex problem in document processing and is an active area of research. The
recent success of deep learning in solving various computer vision and machine
learning problems has not been reflected in document structure analysis since
conventional neural networks are not well suited to the input structure of the
problem. In this paper, we propose an architecture based on graph networks as a
better alternative to standard neural networks for table recognition. We argue
that graph networks are a more natural choice for these problems, and explore
two gradient-based graph neural networks. Our proposed architecture combines the
benefits of convolutional neural networks for visual feature extraction and
graph networks for dealing with the problem structure. We empirically
demonstrate that our method outperforms the baseline by a significant margin. In
addition, we identify the lack of large scale datasets as a major hindrance for
deep learning research for structure analysis and present a new large scale
synthetic dataset for the problem of table recognition. Finally, we open-source
our implementation of dataset generation and the training framework of our graph
networks to promote reproducible research in this direction.

---

RDU: A Region-based Approach to Form-style Document Understanding

Fengbin Zhu, Chao Wang, Wenqiang Lei, Ziyang Liu, Tat Seng Chua

2022-06-14 [entry](http://arxiv.org/abs/2206.06890v1) [pdf](http://arxiv.org/pdf/2206.06890v1)

Key Information Extraction (KIE) is aimed at extracting structured information
(e.g. key-value pairs) from form-style documents (e.g. invoices), which makes an
important step towards intelligent document understanding. Previous approaches
generally tackle KIE by sequence tagging, which faces difficulty to process non-
flatten sequences, especially for table-text mixed documents. These approaches
also suffer from the trouble of pre-defining a fixed set of labels for each type
of documents, as well as the label imbalance issue. In this work, we assume
Optical Character Recognition (OCR) has been applied to input documents, and
reformulate the KIE task as a region prediction problem in the two-dimensional
(2D) space given a target field. Following this new setup, we develop a new KIE
model named Region-based Document Understanding (RDU) that takes as input the
text content and corresponding coordinates of a document, and tries to predict
the result by localizing a bounding-box-like region. Our RDU first applies a
layout-aware BERT equipped with a soft layout attention masking and bias
mechanism to incorporate layout information into the representations. Then, a
list of candidate regions is generated from the representations via a Region
Proposal Module inspired by computer vision models widely applied for object
detection. Finally, a Region Categorization Module and a Region Selection Module
are adopted to judge whether a proposed region is valid and select the one with
the largest probability from all proposed regions respectively. Experiments on
four types of form-style documents show that our proposed method can achieve
impressive results. In addition, our RDU model can be trained with different
document types seamlessly, which is especially helpful over low-resource
documents.

---

Vision-Enhanced Semantic Entity Recognition in Document Images via Visually-Asymmetric Consistency Learning

Hao Wang, Xiahua Chen, Rui Wang, Chenhui Chu

2023-10-23 [entry](http://arxiv.org/abs/2310.14785v1) [pdf](http://arxiv.org/pdf/2310.14785v1)

Extracting meaningful entities belonging to predefined categories from Visually-
rich Form-like Documents (VFDs) is a challenging task. Visual and layout
features such as font, background, color, and bounding box location and size
provide important cues for identifying entities of the same type. However,
existing models commonly train a visual encoder with weak cross-modal
supervision signals, resulting in a limited capacity to capture these non-
textual features and suboptimal performance. In this paper, we propose a novel
\textbf{V}isually-\textbf{A}symmetric co\textbf{N}sisten\textbf{C}y
\textbf{L}earning (\textsc{Vancl}) approach that addresses the above limitation
by enhancing the model's ability to capture fine-grained visual and layout
features through the incorporation of color priors. Experimental results on
benchmark datasets show that our approach substantially outperforms the strong
LayoutLM series baseline, demonstrating the effectiveness of our approach.
Additionally, we investigate the effects of different color schemes on our
approach, providing insights for optimizing model performance. We believe our
work will inspire future research on multimodal information extraction.

---

Tag, Copy or Predict: A Unified Weakly-Supervised Learning Framework for Visual Information Extraction using Sequences

Jiapeng Wang, Tianwei Wang, Guozhi Tang, Lianwen Jin, Weihong Ma, Kai Ding, Yichao Huang

2021-06-20 [entry](http://arxiv.org/abs/2106.10681v1) [pdf](http://arxiv.org/pdf/2106.10681v1)

Visual information extraction (VIE) has attracted increasing attention in recent
years. The existing methods usually first organized optical character
recognition (OCR) results into plain texts and then utilized token-level entity
annotations as supervision to train a sequence tagging model. However, it
expends great annotation costs and may be exposed to label confusion, and the
OCR errors will also significantly affect the final performance. In this paper,
we propose a unified weakly-supervised learning framework called TCPN (Tag, Copy
or Predict Network), which introduces 1) an efficient encoder to simultaneously
model the semantic and layout information in 2D OCR results; 2) a weakly-
supervised training strategy that utilizes only key information sequences as
supervision; and 3) a flexible and switchable decoder which contains two
inference modes: one (Copy or Predict Mode) is to output key information
sequences of different categories by copying a token from the input or
predicting one in each time step, and the other (Tag Mode) is to directly tag
the input sequence in a single forward pass. Our method shows new state-of-the-
art performance on several public benchmarks, which fully proves its
effectiveness.

---

Improving Information Extraction on Business Documents with Specific Pre-Training Tasks

Thibault Douzon, Stefan Duffner, Christophe Garcia, Jérémy Espinas

2023-09-11 [entry](http://arxiv.org/abs/2309.05429v1) [pdf](http://arxiv.org/pdf/2309.05429v1)

Transformer-based Language Models are widely used in Natural Language Processing
related tasks. Thanks to their pre-training, they have been successfully adapted
to Information Extraction in business documents. However, most pre-training
tasks proposed in the literature for business documents are too generic and not
sufficient to learn more complex structures. In this paper, we use LayoutLM, a
language model pre-trained on a collection of business documents, and introduce
two new pre-training tasks that further improve its capacity to extract relevant
information. The first is aimed at better understanding the complex layout of
documents, and the second focuses on numeric values and their order of
magnitude. These tasks force the model to learn better-contextualized
representations of the scanned documents. We further introduce a new post-
processing algorithm to decode BIESO tags in Information Extraction that
performs better with complex entities. Our method significantly improves
extraction performance on both public (from 93.88 to 95.50 F1 score) and private
(from 84.35 to 84.84 F1 score) datasets composed of expense receipts, invoices,
and purchase orders.

---

Information Redundancy and Biases in Public Document Information Extraction Benchmarks

Seif Laatiri, Pirashanth Ratnamogan, Joel Tang, Laurent Lam, William Vanhuffel, Fabien Caspani

2023-04-28 [entry](http://arxiv.org/abs/2304.14936v1) [pdf](http://arxiv.org/pdf/2304.14936v1)

Advances in the Visually-rich Document Understanding (VrDU) field and
particularly the Key-Information Extraction (KIE) task are marked with the
emergence of efficient Transformer-based approaches such as the LayoutLM models.
Despite the good performance of KIE models when fine-tuned on public benchmarks,
they still struggle to generalize on complex real-life use-cases lacking
sufficient document annotations. Our research highlighted that KIE standard
benchmarks such as SROIE and FUNSD contain significant similarity between
training and testing documents and can be adjusted to better evaluate the
generalization of models. In this work, we designed experiments to quantify the
information redundancy in public benchmarks, revealing a 75% template
replication in SROIE official test set and 16% in FUNSD. We also proposed
resampling strategies to provide benchmarks more representative of the
generalization ability of models. We showed that models not suited for document
analysis struggle on the adjusted splits dropping on average 10,5% F1 score on
SROIE and 3.5% on FUNSD compared to multi-modal models dropping only 7,5% F1 on
SROIE and 0.5% F1 on FUNSD.

---

Layout and Task Aware Instruction Prompt for Zero-shot Document Image Question Answering

Wenjin Wang, Yunhao Li, Yixin Ou, Yin Zhang

2023-06-01 [entry](http://arxiv.org/abs/2306.00526v4) [pdf](http://arxiv.org/pdf/2306.00526v4)

Layout-aware pre-trained models has achieved significant progress on document
image question answering. They introduce extra learnable modules into existing
language models to capture layout information within document images from text
bounding box coordinates obtained by OCR tools. However, extra modules
necessitate pre-training on extensive document images. This prevents these
methods from directly utilizing off-the-shelf instruction-tuning language
foundation models, which have recently shown promising potential in zero-shot
learning. Instead, in this paper, we find that instruction-tuning language
models like Claude and ChatGPT can understand layout by spaces and line breaks.
Based on this observation, we propose the LAyout and Task aware Instruction
Prompt (LATIN-Prompt), which consists of layout-aware document content and task-
aware instruction. Specifically, the former uses appropriate spaces and line
breaks to recover the layout information among text segments obtained by OCR
tools, and the latter ensures that generated answers adhere to formatting
requirements. Moreover, we propose the LAyout and Task aware Instruction Tuning
(LATIN-Tuning) to improve the performance of small instruction-tuning models
like Alpaca. Experimental results show that LATIN-Prompt enables zero-shot
performance of Claude and ChatGPT to be comparable to the fine-tuning
performance of SOTAs on document image question answering, and LATIN-Tuning
enhances the zero-shot performance of Alpaca significantly. For example, LATIN-
Prompt improves the performance of Claude and ChatGPT on DocVQA by 263% and 20%
respectively. LATIN-Tuning improves the performance of Alpaca on DocVQA by
87.7%. Quantitative and qualitative analyses demonstrate the effectiveness of
LATIN-Prompt and LATIN-Tuning. We provide the code in supplementary and will
release it to facilitate future research.

---

ICDAR 2021 Competition on Scientific Table Image Recognition to LaTeX

Pratik Kayal, Mrinal Anand, Harsh Desai, Mayank Singh

2021-05-30 [entry](http://arxiv.org/abs/2105.14426v2) [pdf](http://arxiv.org/pdf/2105.14426v2)

Tables present important information concisely in many scientific documents.
Visual features like mathematical symbols, equations, and spanning cells make
structure and content extraction from tables embedded in research documents
difficult. This paper discusses the dataset, tasks, participants' methods, and
results of the ICDAR 2021 Competition on Scientific Table Image Recognition to
LaTeX. Specifically, the task of the competition is to convert a tabular image
to its corresponding LaTeX source code. We proposed two subtasks. In Subtask 1,
we ask the participants to reconstruct the LaTeX structure code from an image.
In Subtask 2, we ask the participants to reconstruct the LaTeX content code from
an image. This report describes the datasets and ground truth specification,
details the performance evaluation metrics used, presents the final results, and
summarizes the participating methods. Submission by team VCGroup got the highest
Exact Match accuracy score of 74% for Subtask 1 and 55% for Subtask 2, beating
previous baselines by 5% and 12%, respectively. Although improvements can still
be made to the recognition capabilities of models, this competition contributes
to the development of fully automated table recognition systems by challenging
practitioners to solve problems under specific constraints and sharing their
approaches; the platform will remain available for post-challenge submissions at
https://competitions.codalab.org/competitions/26979 .

---

FormNet: Structural Encoding beyond Sequential Modeling in Form Document Information Extraction

Chen-Yu Lee, Chun-Liang Li, Timothy Dozat, Vincent Perot, Guolong Su, Nan Hua, Joshua Ainslie, Renshen Wang, Yasuhisa Fujii, Tomas Pfister

2022-03-16 [entry](http://arxiv.org/abs/2203.08411v2) [pdf](http://arxiv.org/pdf/2203.08411v2)

Sequence modeling has demonstrated state-of-the-art performance on natural
language and document understanding tasks. However, it is challenging to
correctly serialize tokens in form-like documents in practice due to their
variety of layout patterns. We propose FormNet, a structure-aware sequence model
to mitigate the suboptimal serialization of forms. First, we design Rich
Attention that leverages the spatial relationship between tokens in a form for
more precise attention score calculation. Second, we construct Super-Tokens for
each word by embedding representations from their neighboring tokens through
graph convolutions. FormNet therefore explicitly recovers local syntactic
information that may have been lost during serialization. In experiments,
FormNet outperforms existing methods with a more compact model size and less
pre-training data, establishing new state-of-the-art performance on CORD, FUNSD
and Payment benchmarks.

---

GroupLink: An End-to-end Multitask Method for Word Grouping and Relation Extraction in Form Understanding

Zilong Wang, Mingjie Zhan, Houxing Ren, Zhaohui Hou, Yuwei Wu, Xingyan Zhang, Ding Liang

2021-05-10 [entry](http://arxiv.org/abs/2105.04650v1) [pdf](http://arxiv.org/pdf/2105.04650v1)

Forms are a common type of document in real life and carry rich information
through textual contents and the organizational structure. To realize automatic
processing of forms, word grouping and relation extraction are two fundamental
and crucial steps after preliminary processing of optical character reader
(OCR). Word grouping is to aggregate words that belong to the same semantic
entity, and relation extraction is to predict the links between semantic
entities. Existing works treat them as two individual tasks, but these two tasks
are correlated and can reinforce each other. The grouping process will refine
the integrated representation of the corresponding entity, and the linking
process will give feedback to the grouping performance. For this purpose, we
acquire multimodal features from both textual data and layout information and
build an end-to-end model through multitask training to combine word grouping
and relation extraction to enhance performance on each task. We validate our
proposed method on a real-world, fully-annotated, noisy-scanned benchmark,
FUNSD, and extensive experiments demonstrate the effectiveness of our method.

---

Document Layout Analysis on BaDLAD Dataset: A Comprehensive MViTv2 Based Approach

Ashrafur Rahman Khan, Asif Azad

2023-08-31 [entry](http://arxiv.org/abs/2308.16571v1) [pdf](http://arxiv.org/pdf/2308.16571v1)

In the rapidly evolving digital era, the analysis of document layouts plays a
pivotal role in automated information extraction and interpretation. In our
work, we have trained MViTv2 transformer model architecture with cascaded mask
R-CNN on BaDLAD dataset to extract text box, paragraphs, images and tables from
a document. After training on 20365 document images for 36 epochs in a 3 phase
cycle, we achieved a training loss of 0.2125 and a mask loss of 0.19. Our work
extends beyond training, delving into the exploration of potential enhancement
avenues. We investigate the impact of rotation and flip augmentation, the
effectiveness of slicing input images pre-inference, the implications of varying
the resolution of the transformer backbone, and the potential of employing a
dual-pass inference to uncover missed text-boxes. Through these explorations, we
observe a spectrum of outcomes, where some modifications result in tangible
performance improvements, while others offer unique insights for future
endeavors.

---

Multi-Type-TD-TSR -- Extracting Tables from Document Images using a Multi-stage Pipeline for Table Detection and Table Structure Recognition: from OCR to Structured Table Representations

Pascal Fischer, Alen Smajic, Alexander Mehler, Giuseppe Abrami

2021-05-23 [entry](http://arxiv.org/abs/2105.11021v1) [pdf](http://arxiv.org/pdf/2105.11021v1)

As global trends are shifting towards data-driven industries, the demand for
automated algorithms that can convert digital images of scanned documents into
machine readable information is rapidly growing. Besides the opportunity of data
digitization for the application of data analytic tools, there is also a massive
improvement towards automation of processes, which previously would require
manual inspection of the documents. Although the introduction of optical
character recognition technologies mostly solved the task of converting human-
readable characters from images into machine-readable characters, the task of
extracting table semantics has been less focused on over the years. The
recognition of tables consists of two main tasks, namely table detection and
table structure recognition. Most prior work on this problem focuses on either
task without offering an end-to-end solution or paying attention to real
application conditions like rotated images or noise artefacts inside the
document image. Recent work shows a clear trend towards deep learning approaches
coupled with the use of transfer learning for the task of table structure
recognition due to the lack of sufficiently large datasets. In this paper we
present a multistage pipeline named Multi-Type-TD-TSR, which offers an end-to-
end solution for the problem of table recognition. It utilizes state-of-the-art
deep learning models for table detection and differentiates between 3 different
types of tables based on the tables' borders. For the table structure
recognition we use a deterministic non-data driven algorithm, which works on all
table types. We additionally present two algorithms. One for unbordered tables
and one for bordered tables, which are the base of the used table structure
recognition algorithm. We evaluate Multi-Type-TD-TSR on the ICDAR 2019 table
structure recognition dataset and achieve a new state-of-the-art.

---

PingAn-VCGroup's Solution for ICDAR 2021 Competition on Scientific Table Image Recognition to Latex

Yelin He, Xianbiao Qi, Jiaquan Ye, Peng Gao, Yihao Chen, Bingcong Li, Xin Tang, Rong Xiao

2021-05-05 [entry](http://arxiv.org/abs/2105.01846v1) [pdf](http://arxiv.org/pdf/2105.01846v1)

This paper presents our solution for the ICDAR 2021 Competition on Scientific
Table Image Recognition to LaTeX. This competition has two sub-tasks: Table
Structure Reconstruction (TSR) and Table Content Reconstruction (TCR). We treat
both sub-tasks as two individual image-to-sequence recognition problems. We
leverage our previously proposed algorithm MASTER \cite{lu2019master}, which is
originally proposed for scene text recognition. We optimize the MASTER model
from several perspectives: network structure, optimizer, normalization method,
pre-trained model, resolution of input image, data augmentation, and model
ensemble. Our method achieves 0.7444 Exact Match and 0.8765 Exact Match @95\% on
the TSR task, and obtains 0.5586 Exact Match and 0.7386 Exact Match 95\% on the
TCR task.

---

ExTTNet: A Deep Learning Algorithm for Extracting Table Texts from Invoice Images

Adem Akdoğan, Murat Kurt

2024-02-03 [entry](http://arxiv.org/abs/2402.02246v1) [pdf](http://arxiv.org/pdf/2402.02246v1)

In this work, product tables in invoices are obtained autonomously via a deep
learning model, which is named as ExTTNet. Firstly, text is obtained from
invoice images using Optical Character Recognition (OCR) techniques. Tesseract
OCR engine [37] is used for this process. Afterwards, the number of existing
features is increased by using feature extraction methods to increase the
accuracy. Labeling process is done according to whether each text obtained as a
result of OCR is a table element or not. In this study, a multilayer artificial
neural network model is used. The training has been carried out with an Nvidia
RTX 3090 graphics card and taken $162$ minutes. As a result of the training, the
F1 score is $0.92$.

---

TLGAN: document Text Localization using Generative Adversarial Nets

Dongyoung Kim, Myungsung Kwak, Eunji Won, Sejung Shin, Jeongyeon Nam

2020-10-22 [entry](http://arxiv.org/abs/2010.11547v1) [pdf](http://arxiv.org/pdf/2010.11547v1)

Text localization from the digital image is the first step for the optical
character recognition task. Conventional image processing based text
localization performs adequately for specific examples. Yet, a general text
localization are only archived by recent deep-learning based modalities. Here we
present document Text Localization Generative Adversarial Nets (TLGAN) which are
deep neural networks to perform the text localization from digital image. TLGAN
is an versatile and easy-train text localization model requiring a small amount
of data. Training only ten labeled receipt images from Robust Reading Challenge
on Scanned Receipts OCR and Information Extraction (SROIE), TLGAN achieved
99.83% precision and 99.64% recall for SROIE test data. Our TLGAN is a practical
text localization solution requiring minimal effort for data labeling and model
training and producing a state-of-art performance.

---

FUNSD: A Dataset for Form Understanding in Noisy Scanned Documents

Guillaume Jaume, Hazim Kemal Ekenel, Jean-Philippe Thiran

2019-05-27 [entry](http://arxiv.org/abs/1905.13538v2) [pdf](http://arxiv.org/pdf/1905.13538v2)

We present a new dataset for form understanding in noisy scanned documents
(FUNSD) that aims at extracting and structuring the textual content of forms.
The dataset comprises 199 real, fully annotated, scanned forms. The documents
are noisy and vary widely in appearance, making form understanding (FoUn) a
challenging task. The proposed dataset can be used for various tasks, including
text detection, optical character recognition, spatial layout analysis, and
entity labeling/linking. To the best of our knowledge, this is the first
publicly available dataset with comprehensive annotations to address FoUn task.
We also present a set of baselines and introduce metrics to evaluate performance
on the FUNSD dataset, which can be downloaded at
https://guillaumejaume.github.io/FUNSD/.

---

Post-OCR Paragraph Recognition by Graph Convolutional Networks

Renshen Wang, Yasuhisa Fujii, Ashok C. Popat

2021-01-29 [entry](http://arxiv.org/abs/2101.12741v6) [pdf](http://arxiv.org/pdf/2101.12741v6)

We propose a new approach for paragraph recognition in document images by
spatial graph convolutional networks (GCN) applied on OCR text boxes. Two steps,
namely line splitting and line clustering, are performed to extract paragraphs
from the lines in OCR results. Each step uses a beta-skeleton graph constructed
from bounding boxes, where the graph edges provide efficient support for graph
convolution operations. With only pure layout input features, the GCN model size
is 3~4 orders of magnitude smaller compared to R-CNN based models, while
achieving comparable or better accuracies on PubLayNet and other datasets.
Furthermore, the GCN models show good generalization from synthetic training
data to real-world images, and good adaptivity for variable document styles.

---

Value Retrieval with Arbitrary Queries for Form-like Documents

Mingfei Gao, Le Xue, Chetan Ramaiah, Chen Xing, Ran Xu, Caiming Xiong

2021-12-15 [entry](http://arxiv.org/abs/2112.07820v2) [pdf](http://arxiv.org/pdf/2112.07820v2)

We propose value retrieval with arbitrary queries for form-like documents to
reduce human effort of processing forms. Unlike previous methods that only
address a fixed set of field items, our method predicts target value for an
arbitrary query based on the understanding of the layout and semantics of a
form. To further boost model performance, we propose a simple document language
modeling (SimpleDLM) strategy to improve document understanding on large-scale
model pre-training. Experimental results show that our method outperforms
previous designs significantly and the SimpleDLM further improves our
performance on value retrieval by around 17% F1 score compared with the state-
of-the-art pre-training method. Code is available at
https://github.com/salesforce/QVR-SimpleDLM.

---

Chargrid: Towards Understanding 2D Documents

Anoop Raveendra Katti, Christian Reisswig, Cordula Guder, Sebastian Brarda, Steffen Bickel, Johannes Höhne, Jean Baptiste Faddoul

2018-09-24 [entry](http://arxiv.org/abs/1809.08799v1) [pdf](http://arxiv.org/pdf/1809.08799v1)

We introduce a novel type of text representation that preserves the 2D layout of
a document. This is achieved by encoding each document page as a two-dimensional
grid of characters. Based on this representation, we present a generic document
understanding pipeline for structured documents. This pipeline makes use of a
fully convolutional encoder-decoder network that predicts a segmentation mask
and bounding boxes. We demonstrate its capabilities on an information extraction
task from invoices and show that it significantly outperforms approaches based
on sequential text or document images.

---

Multimodal Tree Decoder for Table of Contents Extraction in Document Images

Pengfei Hu, Zhenrong Zhang, Jianshu Zhang, Jun Du, Jiajia Wu

2022-12-06 [entry](http://arxiv.org/abs/2212.02896v1) [pdf](http://arxiv.org/pdf/2212.02896v1)

Table of contents (ToC) extraction aims to extract headings of different levels
in documents to better understand the outline of the contents, which can be
widely used for document understanding and information retrieval. Existing works
often use hand-crafted features and predefined rule-based functions to detect
headings and resolve the hierarchical relationship between headings. Both the
benchmark and research based on deep learning are still limited. Accordingly, in
this paper, we first introduce a standard dataset, HierDoc, including image
samples from 650 documents of scientific papers with their content labels. Then
we propose a novel end-to-end model by using the multimodal tree decoder (MTD)
for ToC as a benchmark for HierDoc. The MTD model is mainly composed of three
parts, namely encoder, classifier, and decoder. The encoder fuses the
multimodality features of vision, text, and layout information for each entity
of the document. Then the classifier recognizes and selects the heading
entities. Next, to parse the hierarchical relationship between the heading
entities, a tree-structured decoder is designed. To evaluate the performance,
both the metric of tree-edit-distance similarity (TEDS) and F1-Measure are
adopted. Finally, our MTD approach achieves an average TEDS of 87.2% and an
average F1-Measure of 88.1% on the test set of HierDoc. The code and dataset
will be released at: https://github.com/Pengfei-Hu/MTD.

---

Unveiling Document Structures with YOLOv5 Layout Detection

Herman Sugiharto, Yorissa Silviana, Yani Siti Nurpazrin

2023-09-29 [entry](http://arxiv.org/abs/2309.17033v1) [pdf](http://arxiv.org/pdf/2309.17033v1)

The current digital environment is characterized by the widespread presence of
data, particularly unstructured data, which poses many issues in sectors
including finance, healthcare, and education. Conventional techniques for data
extraction encounter difficulties in dealing with the inherent variety and
complexity of unstructured data, hence requiring the adoption of more efficient
methodologies. This research investigates the utilization of YOLOv5, a cutting-
edge computer vision model, for the purpose of rapidly identifying document
layouts and extracting unstructured data.   The present study establishes a
conceptual framework for delineating the notion of "objects" as they pertain to
documents, incorporating various elements such as paragraphs, tables, photos,
and other constituent parts. The main objective is to create an autonomous
system that can effectively recognize document layouts and extract unstructured
data, hence improving the effectiveness of data extraction.   In the conducted
examination, the YOLOv5 model exhibits notable effectiveness in the task of
document layout identification, attaining a high accuracy rate along with a
precision value of 0.91, a recall value of 0.971, an F1-score of 0.939, and an
area under the receiver operating characteristic curve (AUC-ROC) of 0.975. The
remarkable performance of this system optimizes the process of extracting
textual and tabular data from document images. Its prospective applications are
not limited to document analysis but can encompass unstructured data from
diverse sources, such as audio data.   This study lays the foundation for future
investigations into the wider applicability of YOLOv5 in managing various types
of unstructured data, offering potential for novel applications across multiple
domains.

---

DocStruct: A Multimodal Method to Extract Hierarchy Structure in Document for General Form Understanding

Zilong Wang, Mingjie Zhan, Xuebo Liu, Ding Liang

2020-10-15 [entry](http://arxiv.org/abs/2010.11685v1) [pdf](http://arxiv.org/pdf/2010.11685v1)

Form understanding depends on both textual contents and organizational
structure. Although modern OCR performs well, it is still challenging to realize
general form understanding because forms are commonly used and of various
formats. The table detection and handcrafted features in previous works cannot
apply to all forms because of their requirements on formats. Therefore, we
concentrate on the most elementary components, the key-value pairs, and adopt
multimodal methods to extract features. We consider the form structure as a
tree-like or graph-like hierarchy of text fragments. The parent-child relation
corresponds to the key-value pairs in forms. We utilize the state-of-the-art
models and design targeted extraction modules to extract multimodal features
from semantic contents, layout information, and visual images. A hybrid fusion
method of concatenation and feature shifting is designed to fuse the
heterogeneous features and provide an informative joint representation. We adopt
an asymmetric algorithm and negative sampling in our model as well. We validate
our method on two benchmarks, MedForm and FUNSD, and extensive experiments
demonstrate the effectiveness of our method.

---

OrigamiNet: Weakly-Supervised, Segmentation-Free, One-Step, Full Page Text Recognition by learning to unfold

Mohamed Yousef, Tom E. Bishop

2020-06-12 [entry](http://arxiv.org/abs/2006.07491v1) [pdf](http://arxiv.org/pdf/2006.07491v1)

Text recognition is a major computer vision task with a big set of associated
challenges. One of those traditional challenges is the coupled nature of text
recognition and segmentation. This problem has been progressively solved over
the past decades, going from segmentation based recognition to segmentation free
approaches, which proved more accurate and much cheaper to annotate data for. We
take a step from segmentation-free single line recognition towards segmentation-
free multi-line / full page recognition. We propose a novel and simple neural
network module, termed \textbf{OrigamiNet}, that can augment any CTC-trained,
fully convolutional single line text recognizer, to convert it into a multi-line
version by providing the model with enough spatial capacity to be able to
properly collapse a 2D input signal into 1D without losing information. Such
modified networks can be trained using exactly their same simple original
procedure, and using only \textbf{unsegmented} image and text pairs. We carry
out a set of interpretability experiments that show that our trained models
learn an accurate implicit line segmentation. We achieve state-of-the-art
character error rate on both IAM \& ICDAR 2017 HTR benchmarks for handwriting
recognition, surpassing all other methods in the literature. On IAM we even
surpass single line methods that use accurate localization information during
training. Our code is available online at
\url{https://github.com/IntuitionMachines/OrigamiNet}.

---

A Graphical Approach to Document Layout Analysis

Jilin Wang, Michael Krumdick, Baojia Tong, Hamima Halim, Maxim Sokolov, Vadym Barda, Delphine Vendryes, Chris Tanner

2023-08-03 [entry](http://arxiv.org/abs/2308.02051v1) [pdf](http://arxiv.org/pdf/2308.02051v1)

Document layout analysis (DLA) is the task of detecting the distinct, semantic
content within a document and correctly classifying these items into an
appropriate category (e.g., text, title, figure). DLA pipelines enable users to
convert documents into structured machine-readable formats that can then be used
for many useful downstream tasks. Most existing state-of-the-art (SOTA) DLA
models represent documents as images, discarding the rich metadata available in
electronically generated PDFs. Directly leveraging this metadata, we represent
each PDF page as a structured graph and frame the DLA problem as a graph
segmentation and classification problem. We introduce the Graph-based Layout
Analysis Model (GLAM), a lightweight graph neural network competitive with SOTA
models on two challenging DLA datasets - while being an order of magnitude
smaller than existing models. In particular, the 4-million parameter GLAM model
outperforms the leading 140M+ parameter computer vision-based model on 5 of the
11 classes on the DocLayNet dataset. A simple ensemble of these two models
achieves a new state-of-the-art on DocLayNet, increasing mAP from 76.8 to 80.8.
Overall, GLAM is over 5 times more efficient than SOTA models, making GLAM a
favorable engineering choice for DLA tasks.

---

Transformer-Based UNet with Multi-Headed Cross-Attention Skip Connections to Eliminate Artifacts in Scanned Documents

David Kreuzer, Michael Munz

2023-06-05 [entry](http://arxiv.org/abs/2306.02815v1) [pdf](http://arxiv.org/pdf/2306.02815v1)

The extraction of text in high quality is essential for text-based document
analysis tasks like Document Classification or Named Entity Recognition.
Unfortunately, this is not always ensured, as poor scan quality and the
resulting artifacts lead to errors in the Optical Character Recognition (OCR)
process. Current approaches using Convolutional Neural Networks show promising
results for background removal tasks but fail correcting artifacts like
pixelation or compression errors. For general images, Transformer backbones are
getting integrated more frequently in well-known neural network structures for
denoising tasks. In this work, a modified UNet structure using a Swin
Transformer backbone is presented to remove typical artifacts in scanned
documents. Multi-headed cross-attention skip connections are used to more
selectively learn features in respective levels of abstraction. The performance
of this approach is examined regarding compression errors, pixelation and random
noise. An improvement in text extraction quality with a reduced error rate of up
to 53.9% on the synthetic data is archived. The pretrained base-model can be
easily adapted to new artifacts. The cross-attention skip connections allow to
integrate textual information extracted from the encoder or in form of commands
to more selectively control the models outcome. The latter is shown by means of
an example application.

---

You Actually Look Twice At it (YALTAi): using an object detection approach instead of region segmentation within the Kraken engine

Thibault Clérice

2022-07-19 [entry](http://arxiv.org/abs/2207.11230v2) [pdf](http://arxiv.org/pdf/2207.11230v2)

Layout Analysis (the identification of zones and their classification) is the
first step along line segmentation in Optical Character Recognition and similar
tasks. The ability of identifying main body of text from marginal text or
running titles makes the difference between extracting the work full text of a
digitized book and noisy outputs. We show that most segmenters focus on pixel
classification and that polygonization of this output has not been used as a
target for the latest competition on historical document (ICDAR 2017 and
onwards), despite being the focus in the early 2010s. We propose to shift, for
efficiency, the task from a pixel classification-based polygonization to an
object detection using isothetic rectangles. We compare the output of Kraken and
YOLOv5 in terms of segmentation and show that the later severely outperforms the
first on small datasets (1110 samples and below). We release two datasets for
training and evaluation on historical documents as well as a new package,
YALTAi, which injects YOLOv5 in the segmentation pipeline of Kraken 4.1.

---

End-to-End Information Extraction by Character-Level Embedding and Multi-Stage Attentional U-Net

Tuan-Anh Nguyen Dang, Dat-Thanh Nguyen

2021-06-02 [entry](http://arxiv.org/abs/2106.00952v3) [pdf](http://arxiv.org/pdf/2106.00952v3)

Information extraction from document images has received a lot of attention
recently, due to the need for digitizing a large volume of unstructured
documents such as invoices, receipts, bank transfers, etc. In this paper, we
propose a novel deep learning architecture for end-to-end information extraction
on the 2D character-grid embedding of the document, namely the \textit{Multi-
Stage Attentional U-Net}. To effectively capture the textual and spatial
relations between 2D elements, our model leverages a specialized multi-stage
encoder-decoders design, in conjunction with efficient uses of the self-
attention mechanism and the box convolution. Experimental results on different
datasets show that our model outperforms the baseline U-Net architecture by a
large margin while using 40\% fewer parameters. Moreover, it also significantly
improved the baseline in erroneous OCR and limited training data scenario, thus
becomes practical for real-world applications.

---

Document Domain Randomization for Deep Learning Document Layout Extraction

Meng Ling, Jian Chen, Torsten Möller, Petra Isenberg, Tobias Isenberg, Michael Sedlmair, Robert S. Laramee, Han-Wei Shen, Jian Wu, C. Lee Giles

2021-05-20 [entry](http://arxiv.org/abs/2105.14931v1) [pdf](http://arxiv.org/pdf/2105.14931v1)

We present document domain randomization (DDR), the first successful transfer of
convolutional neural networks (CNNs) trained only on graphically rendered
pseudo-paper pages to real-world document segmentation. DDR renders pseudo-
document pages by modeling randomized textual and non-textual contents of
interest, with user-defined layout and font styles to support joint learning of
fine-grained classes. We demonstrate competitive results using our DDR approach
to extract nine document classes from the benchmark CS-150 and papers published
in two domains, namely annual meetings of Association for Computational
Linguistics (ACL) and IEEE Visualization (VIS). We compare DDR to conditions of
style mismatch, fewer or more noisy samples that are more easily obtained in the
real world. We show that high-fidelity semantic information is not necessary to
label semantic classes but style mismatch between train and test can lower model
accuracy. Using smaller training samples had a slightly detrimental effect.
Finally, network models still achieved high test accuracy when correct labels
are diluted towards confusing labels; this behavior hold across several classes.

---

ICDAR2019 Competition on Scanned Receipt OCR and Information Extraction

Zheng Huang, Kai Chen, Jianhua He, Xiang Bai, Dimosthenis Karatzas, Shjian Lu, C. V. Jawahar

2021-03-18 [entry](http://arxiv.org/abs/2103.10213v1) [pdf](http://arxiv.org/pdf/2103.10213v1)

Scanned receipts OCR and key information extraction (SROIE) represent the
processeses of recognizing text from scanned receipts and extracting key texts
from them and save the extracted tests to structured documents. SROIE plays
critical roles for many document analysis applications and holds great
commercial potentials, but very little research works and advances have been
published in this area. In recognition of the technical challenges, importance
and huge commercial potentials of SROIE, we organized the ICDAR 2019 competition
on SROIE. In this competition, we set up three tasks, namely, Scanned Receipt
Text Localisation (Task 1), Scanned Receipt OCR (Task 2) and Key Information
Extraction from Scanned Receipts (Task 3). A new dataset with 1000 whole scanned
receipt images and annotations is created for the competition. In this report we
will presents the motivation, competition datasets, task definition, evaluation
protocol, submission statistics, performance of submitted methods and results
analysis.

---

DeepErase: Weakly Supervised Ink Artifact Removal in Document Text Images

W. Ronny Huang, Yike Qi, Qianqian Li, Jonathan Degange

2019-10-15 [entry](http://arxiv.org/abs/1910.07070v3) [pdf](http://arxiv.org/pdf/1910.07070v3)

Paper-intensive industries like insurance, law, and government have long
leveraged optical character recognition (OCR) to automatically transcribe hordes
of scanned documents into text strings for downstream processing. Even in 2019,
there are still many scanned documents and mail that come into businesses in
non-digital format. Text to be extracted from real world documents is often
nestled inside rich formatting, such as tabular structures or forms with fill-
in-the-blank boxes or underlines whose ink often touches or even strikes through
the ink of the text itself. Further, the text region could have random ink
smudges or spurious strokes. Such ink artifacts can severely interfere with the
performance of recognition algorithms or other downstream processing tasks. In
this work, we propose DeepErase, a neural-based preprocessor to erase ink
artifacts from text images. We devise a method to programmatically assemble real
text images and real artifacts into realistic-looking "dirty" text images, and
use them to train an artifact segmentation network in a weakly supervised
manner, since pixel-level annotations are automatically obtained during the
assembly process. In addition to high segmentation accuracy, we show that our
cleansed images achieve a significant boost in recognition accuracy by popular
OCR software such as Tesseract 4.0. Finally, we test DeepErase on out-of-
distribution datasets (NIST SDB) of scanned IRS tax return forms and achieve
double-digit improvements in accuracy. All experiments are performed on both
printed and handwritten text. Code for all experiments is available at
https://github.com/yikeqicn/DeepErase

---

VRDSynth: Synthesizing Programs for Multilingual Visually Rich Document Information Extraction

Thanh-Dat Nguyen, Tung Do-Viet, Hung Nguyen-Duy, Tuan-Hai Luu, Hung Le, Bach Le, Patanamon, Thongtanunam

2024-07-09 [entry](http://arxiv.org/abs/2407.06826v1) [pdf](http://arxiv.org/pdf/2407.06826v1)

Businesses need to query visually rich documents (VRDs) like receipts, medical
records, and insurance forms to make decisions. Existing techniques for
extracting entities from VRDs struggle with new layouts or require extensive
pre-training data. We introduce VRDSynth, a program synthesis method to
automatically extract entity relations from multilingual VRDs without pre-
training data. To capture the complexity of VRD domain, we design a domain-
specific language (DSL) to capture spatial and textual relations to describe the
synthesized programs. Along with this, we also derive a new synthesis algorithm
utilizing frequent spatial relations, search space pruning, and a combination of
positive, negative, and exclusive programs to improve coverage.   We evaluate
VRDSynth on the FUNSD and XFUND benchmarks for semantic entity linking,
consisting of 1,592 forms in 8 languages. VRDSynth outperforms state-of-the-art
pre-trained models (LayoutXLM, InfoXLMBase, and XLMRobertaBase) in 5, 6, and 7
out of 8 languages, respectively, improving the F1 score by 42% over LayoutXLM
in English. To test the extensibility of the model, we further improve VRDSynth
with automated table recognition, creating VRDSynth(Table), and compare it with
extended versions of the pre-trained models, InfoXLM(Large) and
XLMRoberta(Large). VRDSynth(Table) outperforms these baselines in 4 out of 8
languages and in average F1 score. VRDSynth also significantly reduces memory
footprint (1M and 380MB vs. 1.48GB and 3GB for LayoutXLM) while maintaining
similar time efficiency.

---

