
## DocSegTr: An Instance-Level End-to-End Document Image Segmentation Transformer

Sanket Biswas, Ayan Banerjee, Josep Lladós, Umapada Pal

Category: ocr
Keywords: Document Layout Analysis, Instance-Level Segmentation, Transformers, Information extraction
Year: 2023

Understanding documents with rich layouts is an essential step towards
information extraction. Business intelligence processes often require the
extraction of useful semantic content from documents at a large scale for
subsequent decision-making tasks. In this context, instance-level segmentation
of different document objects (title, sections, figures etc.) has emerged as an
interesting problem for the document analysis and understanding community. To
advance the research in this direction, we present a transformer-based model
called DocSegTr for end-to-end instance segmentation of complex layouts in
document images. The method adapts a twin attention module for semantic
reasoning, which helps to become highly computationally efficient compared with
the state-of-the-art. To the best of our knowledge, this is the first work on
transformer-based document segmentation. Extensive experimentation on
competitive benchmarks like PubLayNet, PRIMA, Historical Japanese (HJ) and
TableBank demonstrate that our model achieved comparable or better segmentation
performance than the existing state-of-the-art approaches with the average
precision of 89.4, 40.3, 83.4 and 93.3. This simple and flexible framework could
serve as a promising baseline for instance-level recognition tasks in document
images.

---

## OCR-free Document Understanding Transformer

Geewook Kim, Teakgyu Hong, Moonbin Yim, Jeongyeon Nam, Jinyoung Park, Jinyeong Yim, Wonseok Hwang, Sangdoo Yun, Dongyoon Han, Seunghyun Park

Category: ocr
Keywords: Visual Document Understanding, Document Information Extraction, Optical Character Recognition, End-to-End Transformer
Year: 2023

Understanding document images (e.g., invoices) is a core but challenging task
since it requires complex functions such as reading text and a holistic
understanding of the document. Current Visual Document Understanding (VDU)
methods outsource the task of reading text to off-the-shelf Optical Character
Recognition (OCR) engines and focus on the understanding task with the OCR
outputs. Although such OCR-based approaches have shown promising performance,
they suffer from 1) high computational costs for using OCR; 2) inflexibility of
OCR models on languages or types of documents; 3) OCR error propagation to the
subsequent process. To address these issues, in this paper, we introduce a novel
OCR-free VDU model named Donut, which stands for Document understanding
transformer. As the first step in OCR-free VDU research, we propose a simple
architecture (i.e., Transformer) with a pre-training objective (i.e., cross-
entropy loss). Donut is conceptually simple yet effective. Through extensive
experiments and analyses, we show a simple OCR-free VDU model, Donut, achieves
state-of-the-art performances on various VDU tasks in terms of both speed and
accuracy. In addition, we offer a synthetic data generator that helps the model
pre-training to be flexible in various languages and domains. The code, trained
model, and synthetic data are available at https://github.com/clovaai/donut.

---

## Extending TrOCR for Text Localization-Free OCR of Full-Page Scanned Receipt Images

Hongkuan Zhang, Edward Whittaker, Ikuo Kitagishi

Category: ocr
Keywords: Receipt Digitization, Localization-Free OCR, End-to-End Receipt OCR
Year: 2023

Digitization of scanned receipts aims to extract text from receipt images and
save it into structured documents. This is usually split into two sub-tasks:
text localization and optical character recognition (OCR). Most existing OCR
models only focus on the cropped text instance images, which require the
bounding box information provided by a text region detection model. Introducing
an additional detector to identify the text instance images in advance is
inefficient; however, instance-level OCR models have very low accuracy when
processing the whole image for the document-level OCR, such as receipt images
containing multiple text lines arranged in various layouts. To this end, we
propose a localization-free document-level OCR model for transcribing all the
characters in a receipt image into an ordered sequence end-to-end. Specifically,
we finetune the pretrained Transformer-based instance-level model TrOCR with
randomly cropped image chunks, and gradually increase the image chunk size to
generalize the recognition ability from instance images to full-page images. In
our experiments on the SROIE receipt OCR dataset, the model finetuned with our
strategy achieved 64.4 F1-score and a 22.8% character error rate (CER) on the
word-level and character-level metrics, respectively, which outperforms the
baseline results with 48.5 F1-score and 50.6% CER. The best model, which splits
the full image into 15 equally sized chunks, gives 87.8 F1-score and 4.98% CER
with minimal additional pre or post-processing of the output. Moreover, the
characters in the generated document-level sequences are arranged in the reading
order, which is practical for real-world applications.

---

## Unifying Vision, Text, and Layout for Universal Document Processing

Zineng Tang, Ziyi Yang, Guoxin Wang, Yuwei Fang, Yang Liu, Chenguang Zhu, Michael Zeng, Cha Zhang, Mohit Bansal

Category: ocr
Keywords: Document AI, Vision-Text-Layout Transformer, Universal Document Processing, Multimodal, Document Understanding
Year: 2023

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
content customization. Our method sets the state-of-the-art on 9 Document AI
tasks, e.g., document understanding and QA, across diverse data domains like
finance reports, academic papers, and websites. UDOP ranks first on the
leaderboard of the Document Understanding Benchmark (DUE).

---

## Reading and Writing: Discriminative and Generative Modeling for Self-Supervised Text Recognition

Mingkun Yang, Minghui Liao, Pu Lu, Jing Wang, Shenggao Zhu, Hualin Luo, Qi Tian, Xiang Bai

Category: ocr
Keywords: self-supervised learning, text recognition, contrastive learning, masked image modeling, domain gap
Year: 2023

Existing text recognition methods usually require large-scale training data,
often relying on synthetic data due to the scarcity of annotated real images.
However, the domain gap between synthetic and real data restricts the
performance of text recognition models. Recent self-supervised methods have
tried to utilize unlabeled real images using contrastive learning, focusing
mainly on discrimination. Inspired by the dual process of human learning through
reading and writing, we propose a self-supervised method integrating contrastive
learning and masked image modeling. The contrastive learning branch emulates
reading by focusing on the discrimination of text images, while masked image
modeling, introduced for the first time in text recognition, mimics writing by
learning context generation. Our method significantly outperforms previous self-
supervised text recognition methods on irregular scene text datasets by
10.2%-20.2% and exceeds state-of-the-art methods by an average of 5.3% on 11
benchmarks with similar model sizes. Furthermore, our pre-trained model
demonstrates notable performance gains when applied to other text-related tasks.

---

## Transformer-Based Approach for Document Layout Understanding

Huichen Yang, William Hsu

Category: ocr
Keywords: Document Layout Understanding, Vision Transformer, Object Detection, Document Structure Extraction
Year: 2023

We present an end-to-end transformer-based framework named TRDLU for the task of
Document Layout Understanding (DLU). DLU is the fundamental task to
automatically understand document structures. To accurately detect content boxes
and classify them into semantically meaningful classes from various formats of
documents is still an open challenge. Recently, transformer-based detection
neural networks have shown their capability over traditional convolutional-based
methods in the object detection area. In this paper, we consider DLU as a
detection task, and introduce TRDLU which integrates transformer-based vision
backbone and transformer encoder-decoder as detection pipeline. TRDLU is only a
visual feature-based framework, but its performance is even better than multi-
modal feature-based models. To the best of our knowledge, this is the first
study of employing a fully transformer-based framework in DLU tasks. We
evaluated TRDLU on three different DLU benchmark datasets, each with strong
baselines. TRDLU outperforms the current state-of-the-art methods on all of
them.

---

## SwinDocSegmenter: An End-to-End Unified Domain Adaptive Transformer for Document Instance Segmentation

Ayan Banerjee, Sanket Biswas, Josep Lladós, Umapada Pal

Category: ocr
Keywords: Document Layout Analysis, Instance-Level Segmentation, Swin Transformer, Contrastive Learning
Year: 2023

Instance-level segmentation of documents consists in assigning a class-aware and
instance-aware label to each pixel of the image. It is a key step in document
parsing for their understanding. In this paper, we present a unified transformer
encoder-decoder architecture for end-to-end instance segmentation of complex
layouts in document images. The method adapts a contrastive training with a
mixed query selection for anchor initialization in the decoder. Later on, it
performs a dot product between the obtained query embeddings and the pixel
embedding map (coming from the encoder) for semantic reasoning. Extensive
experimentation on competitive benchmarks like PubLayNet, PRIMA, Historical
Japanese (HJ), and TableBank demonstrate that our model with SwinL backbone
achieves better segmentation performance than the existing state-of-the-art
approaches with the average precision of 93.72, 54.39, 84.65, and 98.04
respectively under one billion parameters.

---

## An End-to-End OCR Text Re-organization Sequence Learning for Rich-text Detail Image Comprehension

Liangcheng Li, Feiyu Gao, Jiajun Bu, Yongpan Wang, Zhi Yu, Qi Zheng

Category: ocr
Keywords: OCR Text Re-organization, Graph Neural Network, Pointer Network
Year: 2023

Nowadays the description of detailed images helps users know more about the
commodities. With the help of OCR technology, the description text can be
detected and recognized as auxiliary information to remove the visually impaired
users’ comprehension barriers. However, for lack of proper logical structure
among these OCR text blocks, it is challenging to comprehend the detailed images
accurately. To tackle the above problems, we propose a novel end-to-end OCR text
reorganizing model. Specifically, we create a Graph Neural Network with an
attention map to encode the text blocks with visual layout features, with which
an attention-based sequence decoder inspired by the Pointer Network and a
Sinkhorn global optimization will reorder the OCR text into a proper sequence.
Experimental results illustrate that our model outperforms the other baselines,
and the real experiment of the blind users’ experience shows that our model
improves their comprehension.

---

## Doc2Graph: a Task Agnostic Document Understanding Framework based on Graph Neural Networks

Andrea Gemelli, Sanket Biswas, Enrico Civitelli, Josep Lladós, Simone Marinai

Category: ocr
Keywords: Document Analysis and Recognition, Graph Neural Networks, Document Understanding, Key Information Extraction, Table Detection
Year: 2023

Geometric Deep Learning has recently attracted significant interest in a wide
range of machine learning fields, including document analysis. The application
of Graph Neural Networks (GNNs) has become crucial in various document-related
tasks since they can unravel important structural patterns, fundamental in key
information extraction processes. Previous works in the literature propose task-
driven models and do not take into account the full power of graphs. We propose
Doc2Graph, a task-agnostic document understanding framework based on a GNN
model, to solve different tasks given different types of documents. We evaluated
our approach on two challenging datasets for key information extraction in form
understanding, invoice layout analysis and table detection. Our code is freely
accessible on https://github.com/andreagemelli/doc2graph.

---

## TRANSFORMER-BASED APPROACH FOR DOCUMENT LAYOUT UNDERSTANDING

Huichen Yang, William Hsu

Category: ocr
Keywords: Document Layout Understanding, Vision Transformer, Object Detection, Document Structure Extraction
Year: 2023

We present an end-to-end transformer-based framework named TRDLU for the task of
Document Layout Understanding (DLU). DLU is the fundamental task to
automatically understand document structures. To accurately detect content boxes
and classify them into semantically meaningful classes from various formats of
documents is still an open challenge. Recently, transformer-based detection
neural networks have shown their capability over traditional convolutional-based
methods in the object detection area. In this paper, we consider DLU as a
detection task, and introduce TRDLU which integrates transformer-based vision
backbone and transformer encoder-decoder as detection pipeline. TRDLU is only a
visual feature-based framework, but its performance is even better than multi-
modal feature-based models. To the best of our knowledge, this is the first
study of employing a fully transformer-based framework in DLU tasks. We
evaluated TRDLU on three different DLU benchmark datasets, each with strong
baselines. TRDLU outperforms the current state-of-the-art methods on all of
them.

---

## TrOCR: Transformer-based Optical Character Recognition with Pre-trained Models

Minghao Li, Tengchao Lv, Jingye Chen, Lei Cui, Yijuan Lu, Dinei Florencio, Cha Zhang, Zhoujun Li, Furu Wei

Category: ocr
Keywords: Optical Character Recognition, Transformer, Pre-trained Models, Text Recognition, Document Digitalization
Year: 2022

Text recognition is a long-standing research problem for document
digitalization. Existing approaches are usually built based on CNN for image
understanding and RNN for char-level text generation. In addition, another
language model is usually needed to improve the overall accuracy as a post-
processing step. In this paper, we propose an end-to-end text recognition
approach with pre-trained image Transformer and text Transformer models, namely
TrOCR, which leverages the Transformer architecture for both image understanding
and wordpiece-level text generation. The TrOCR model is simple but effective,
and can be pre-trained with large-scale synthetic data and fine-tuned with
human-labeled datasets. Experiments show that the TrOCR model outperforms the
current state-of-the-art models on the printed, handwritten and scene text
recognition tasks. The TrOCR models and code are publicly available at
https://aka.ms/trocr.

---

## DiT: Self-supervised Pre-training for Document Image Transformer

Junlong Li, Yiheng Xu, Tengchao Lv, Lei Cui, Cha Zhang, Furu Wei

Category: ocr
Keywords: document image transformer, self-supervised pre-training, document image classification, document layout analysis, table detection, text detection, OCR
Year: 2022

Image Transformer has recently achieved significant progress for natural image
understanding, either using supervised (ViT, DeiT, etc.) or self-supervised
(BEiT, MAE, etc.) pre-training techniques. In this paper, we propose DiT, a
self-supervised pre-trained Document Image Transformer model using large-scale
unlabeled text images for Document AI tasks, which is essential since no
supervised counterparts ever exist due to the lack of human-labeled document
images. We leverage DiT as the backbone network in a variety of vision-based
Document AI tasks, including document image classification, document layout
analysis, table detection as well as text detection for OCR. Experiment results
have illustrated that the self-supervised pre-trained DiT model achieves new
state-of-the-art results on these downstream tasks, e.g. document image
classification (91.11 →92.69), document layout analysis (91.0 →94.9), table
detection (94.23 →96.55) and text detection for OCR (93.07 →94.29). The code and
pre-trained models are publicly available at https://aka.ms/msdit.

---

## LayoutLMv3: Pre-training for Document AI with Unified Text and Image Masking

Yupan Huang, Tengchao Lv, Lei Cui, Yutong Lu, Furu Wei

Category: ocr
Keywords: document ai, layoutlm, multimodal pre-training, vision-and-language
Year: 2022

Self-supervised pre-training techniques have achieved remarkable progress in
Document AI. Most multimodal pre-trained models use a masked language modeling
objective to learn bidirectional representations on the text modality, but they
differ in pre-training objectives for the image modality. This discrepancy adds
difficulty to multimodal representation learning. In this paper, we propose
LayoutLMv3 to pre-train multimodal Transformers for Document AI with unified
text and image masking. Additionally, LayoutLMv3 is pre-trained with a word-
patch alignment objective to learn cross-modal alignment by predicting whether
the corresponding image patch of a text word is masked. The simple unified
architecture and training objectives make LayoutLMv3 a general-purpose pre-
trained model for both text-centric and image-centric Document AI tasks.
Experimental results show that LayoutLMv3 achieves state-of-the-art performance
not only in text-centric tasks, including form understanding, receipt
understanding, and document visual question answering, but also in image-centric
tasks such as document image classification and document layout analysis. The
code and models are publicly available at https://aka.ms/layoutlmv3.

---

## TrOCR: Transformer-based Optical Character Recognition with Pre-trained Models

Minghao Li, Tengchao Lv, Jingye Chen, Lei Cui, Yijuan Lu, Dinei Florencio, Cha Zhang, Zhoujun Li, Furu Wei

Category: ocr
Keywords: TrOCR, Transformer, Optical Character Recognition, pre-trained models, text recognition
Year: 2022

Text recognition is a long-standing research problem for document
digitalization. Existing approaches are usually built based on CNN for image
understanding and RNN for char-level text generation. In addition, another
language model is usually needed to improve the overall accuracy as a post-
processing step. In this paper, we propose an end-to-end text recognition
approach with pre-trained image Transformer and text Transformer models, namely
TrOCR, which leverages the Transformer architecture for both image understanding
and wordpiece-level text generation. The TrOCR model is simple but effective,
and can be pre-trained with large-scale synthetic data and fine-tuned with
human-labeled datasets. Experiments show that the TrOCR model outperforms the
current state-of-the-art models on the printed, handwritten and scene text
recognition tasks. The TrOCR models and code are publicly available at
https://aka.ms/trocr.

---

## LayoutLMv3: Pre-training for Document AI with Unified Text and Image Masking

Yupan Huang, Tengchao Lv, Lei Cui, Yutong Lu, Furu Wei

Category: ocr
Keywords: document ai, layoutlm, multimodal pre-training, vision-and-language
Year: 2022

Self-supervised pre-training techniques have achieved remarkable progress in
Document AI. Most multimodal pre-trained models use a masked language modeling
objective to learn bidirectional representations on the text modality, but they
differ in pre-training objectives for the image modality. This discrepancy adds
difficulty to multimodal representation learning. In this paper, we propose
LayoutLMv3 to pre-train multimodal Transformers for Document AI with unified
text and image masking. Additionally, LayoutLMv3 is pre-trained with a word-
patch alignment objective to learn cross-modal alignment by predicting whether
the corresponding image patch of a text word is masked. The simple unified
architecture and training objectives make LayoutLMv3 a general-purpose pre-
trained model for both text-centric and image-centric Document AI tasks.
Experimental results show that LayoutLMv3 achieves state-of-the-art performance
not only in text-centric tasks, including form understanding, receipt
understanding, and document visual question answering, but also in image-centric
tasks such as document image classification and document layout analysis.

---

## MaskOCR: Text Recognition with Masked Encoder-Decoder Pretraining

Pengyuan Lyu, Chengquan Zhang, Shanshan Liu, Meina Qiao, Yangliu Xu, Liang Wu, Kun Yao, Junyu Han, Errui Ding, Jingdong Wang

Category: ocr
Keywords: text recognition, encoder-decoder transformer, self-supervised learning, masked image modeling, language modeling
Year: 2022

In this paper, we present a model pretraining technique, named MaskOCR, for text
recognition. Our text recognition architecture is an encoder-decoder
transformer: the encoder extracts the patch-level representations, and the
decoder recognizes the text from the representations. Our approach pretrains
both the encoder and the decoder in a sequential manner. (i) We pretrain the
encoder in a self-supervised manner over a large set of unlabeled real text
images. We adopt the masked image modeling approach, which shows the
effectiveness for general images, expecting that the representations take on
semantics. (ii) We pretrain the decoder over a large set of synthesized text
images in a supervised manner and enhance the language modeling capability of
the decoder by randomly masking some text image patches occupied by characters
input to the encoder and accordingly the representations input to the decoder.
Experiments show that the proposed MaskOCR approach achieves superior results on
the benchmark datasets, including Chinese and English text images.

---

## DiT: Self-supervised Pre-training for Document Image Transformer

Junlong Li, Yiheng Xu, Tengchao Lv, Lei Cui, Cha Zhang, Furu Wei

Category: ocr
Keywords: document image transformer, self-supervised pre-training, document image classification, document layout analysis, table detection, text detection, OCR
Year: 2022

Image Transformer has recently achieved significant progress for natural image
understanding, either using supervised (ViT, DeiT, etc.) or self-supervised
(BEiT, MAE, etc.) pre-training techniques. In this paper, we propose DiT, a
self-supervised pre-trained Document Image Transformer model using large-scale
unlabeled text images for Document AI tasks, which is essential since no
supervised counterparts ever exist due to the lack of human-labeled document
images. We leverage DiT as the backbone network in a variety of vision-based
Document AI tasks, including document image classification, document layout
analysis, table detection as well as text detection for OCR. Experiment results
have illustrated that the self-supervised pre-trained DiT model achieves new
state-of-the-art results on these downstream tasks, e.g. document image
classification (91.11 →92.69), document layout analysis (91.0 →94.9), table
detection (94.23 →96.55) and text detection for OCR (93.07 →94.29). The code and
pre-trained models are publicly available at https://aka.ms/msdit.

---

## Optical Character Recognition with Transformers and CTC

Israel Campiotti, Roberto Lotufo

Category: ocr
Keywords: text recognition, crnn, transformers, attention mechanism
Year: 2022

Text recognition tasks are commonly solved by using a deep learning pipeline
called CRNN. The classical CRNN is a sequence of a convolutional network,
followed by a bidirectional LSTM and a CTC layer. In this paper, we perform an
extensive analysis of the components of a CRNN to find what is crucial to the
entire pipeline and what characteristics can be exchanged for a more effective
choice. Given the results of our experiments, we propose two different
architectures for the task of text recognition. The first model, CNN + CTC, is a
combination of a convolutional model followed by a CTC layer. The second model,
CNN + Tr + CTC, adds an encoder-only Transformers between the convolutional
network and the CTC layer. To the best of our knowledge, this is the first time
that a Transformers have been successfully trained using just CTC loss. To
assess the capabilities of our proposed architectures, we train and evaluate
them on the SROIE 2019 data set. Our CNN + CTC achieves an F1 score of 89.66%
possessing only 4.7 million parameters. CNN + Tr + CTC attained an F1 score of
93.76% with 11 million parameters, which is almost 97% of the performance
achieved by the TrOCR using 334 million parameters and more than 600 million
synthetic images for pretraining.

---

## Transformer-Based Approach for Document Layout Understanding

Huichen Yang, William Hsu

Category: ocr
Keywords: Document Layout Understanding, Vision Transformer, Object Detection, Document Structure Extraction
Year: 2022

We present an end-to-end transformer-based framework named TRDLU for the task of
Document Layout Understanding (DLU). DLU is the fundamental task to
automatically understand document structures. To accurately detect content boxes
and classify them into semantically meaningful classes from various formats of
documents is still an open challenge. Recently, transformer-based detection
neural networks have shown their capability over traditional convolutional-based
methods in the object detection area. In this paper, we consider DLU as a
detection task, and introduce TRDLU which integrates transformer-based vision
backbone and transformer encoder-decoder as detection pipeline. TRDLU is only a
visual feature-based framework, but its performance is even better than multi-
modal feature-based models. To the best of our knowledge, this is the first
study of employing a fully transformer-based framework in DLU tasks. We
evaluated TRDLU on three different DLU benchmark datasets, each with strong
baselines. TRDLU outperforms the current state-of-the-art methods on all of
them.

---

## DocLayNet: A Large Human-Annotated Dataset for Document-Layout Analysis

Birgit Pfitzmann, Christoph Auer, Michele Dolfi, Ahmed S. Nassar, Peter Staar

Category: ocr
Keywords: PDF document conversion, layout segmentation, object-detection, data set, Machine Learning
Year: 2022

Accurate document layout analysis is a key requirement for high-quality PDF
document conversion. With the recent availability of public, large ground-truth
datasets such as PubLayNet and DocBank, deep-learning models have proven to be
very effective at layout detection and segmentation. While these datasets are of
adequate size to train such models, they severely lack in layout variability
since they are sourced from scientific article repositories such as PubMed and
arXiv only. Consequently, the accuracy of the layout segmentation drops
significantly when these models are applied on more challenging and diverse
layouts. In this paper, we present DocLayNet, a new, publicly available,
document-layout annotation dataset in COCO format. It contains 80863 manually
annotated pages from diverse data sources to represent a wide variability in
layouts. For each PDF page, the layout annotations provide labelled bounding-
boxes with a choice of 11 distinct classes. DocLayNet also provides a subset of
double- and triple-annotated pages to determine the inter-annotator agreement.
In multiple experiments, we provide baseline accuracy scores (in mAP) for a set
of popular object detection models. We also demonstrate that these models fall
approximately 10% behind the inter-annotator agreement. Furthermore, we provide
evidence that DocLayNet is of sufficient size. Lastly, we compare models trained
on PubLayNet, DocBank and DocLayNet, showing that layout predictions of the
DocLayNet-trained models are more robust and thus the preferred choice for
general-purpose document-layout analysis.

---

## OCR-free Document Understanding Transformer

Geewook Kim, Teakgyu Hong, Moonbin Yim, Jeongyeon Nam, Jinyoung Park, Jinyeong Yim, Wonseok Hwang, Sangdoo Yun, Dongyoon Han, Seunghyun Park

Category: ocr
Keywords: Visual Document Understanding, Document Information Extraction, Optical Character Recognition, End-to-End Transformer
Year: 2022

Understanding document images (e.g., invoices) is a core but challenging task
since it requires complex functions such as reading text and a holistic
understanding of the document. Current Visual Document Understanding (VDU)
methods outsource the task of reading text to off-the-shelf Optical Character
Recognition (OCR) engines and focus on the understanding task with the OCR
outputs. Although such OCR-based approaches have shown promising performance,
they suffer from 1) high computational costs for using OCR; 2) inflexibility of
OCR models on languages or types of documents; 3) OCR error propagation to the
subsequent process. To address these issues, in this paper, we introduce a novel
OCR-free VDU model named Donut, which stands for Document understanding
transformer. As the first step in OCR-free VDU research, we propose a simple
architecture (i.e., Transformer) with a pre-training objective (i.e., cross-
entropy loss). Donut is conceptually simple yet effective. Through extensive
experiments and analyses, we show a simple OCR-free VDU model, Donut, achieves
state-of-the-art performances on various VDU tasks in terms of both speed and
accuracy. In addition, we offer a synthetic data generator that helps the model
pre-training to be flexible in various languages and domains. The code, trained
model, and synthetic data are available at https://github.com/clovaai/donut.

---

## DocSegTr: An Instance-Level End-to-End Document Image Segmentation Transformer

Sanket Biswas, Ayan Banerjee, Josep Lladós, Umapada Pal

Category: ocr
Keywords: Document Layout Analysis, Instance-Level Segmentation, Transformers, Information extraction
Year: 2022

Understanding documents with rich layouts is an essential step towards
information extraction. Business intelligence processes often require the
extraction of useful semantic content from documents at a large scale for
subsequent decision-making tasks. In this context, instance-level segmentation
of different document objects (title, sections, figures etc.) has emerged as an
interesting problem for the document analysis and understanding community. To
advance the research in this direction, we present a transformer-based model
called DocSegTr for end-to-end instance segmentation of complex layouts in
document images. The method adapts a twin attention module, for semantic
reasoning, which helps to become highly computationally efficient compared with
the state-of-the-art. To the best of our knowledge, this is the first work on
transformer-based document segmentation. Extensive experimentation on
competitive benchmarks like PubLayNet, PRIMA, Historical Japanese (HJ) and
TableBank demonstrate that our model achieved comparable or better segmentation
performance than the existing state-of-the-art approaches with the average
precision of 89.4, 40.3, 83.4 and 93.3. This simple and flexible framework could
serve as a promising baseline for instance-level recognition tasks in document
images.

---

## TrOCR: Transformer-based Optical Character Recognition with Pre-trained Models

Minghao Li, Tengchao Lv, Jingye Chen, Lei Cui, Yijuan Lu, Dinei Florencio, Cha Zhang, Zhoujun Li, Furu Wei

Category: ocr
Keywords: OCR, Transformer, Pre-trained models, Text recognition, Document digitalization
Year: 2022

Text recognition is a long-standing research problem for document
digitalization. Existing approaches are usually built based on CNN for image
understanding and RNN for char-level text generation. In addition, another
language model is usually needed to improve the overall accuracy as a post-
processing step. In this paper, we propose an end-to-end text recognition
approach with pre-trained image Transformer and text Transformer models, namely
TrOCR, which leverages the Transformer architecture for both image understanding
and wordpiece-level text generation. The TrOCR model is simple but effective,
and can be pre-trained with large-scale synthetic data and fine-tuned with
human-labeled datasets. Experiments show that the TrOCR model outperforms the
current state-of-the-art models on the printed, handwritten and scene text
recognition tasks. The TrOCR models and code are publicly available at
https://aka.ms/trocr.

---

## DiT: Self-supervised Pre-training for Document Image Transformer

Junlong Li, Yiheng Xu, Tengchao Lv, Lei Cui, Cha Zhang, Furu Wei

Category: ocr
Keywords: document image transformer, self-supervised pre-training, document image classification, document layout analysis, table detection, text detection, OCR
Year: 2022

Image Transformer has recently achieved significant progress for natural image
understanding, either using supervised (ViT, DeiT, etc.) or self-supervised
(BEiT, MAE, etc.) pre-training techniques. In this paper, we propose DiT, a
self-supervised pre-trained Document Image Transformer model using large-scale
unlabeled text images for Document AI tasks, which is essential since no
supervised counterparts ever exist due to the lack of human-labeled document
images. We leverage DiT as the backbone network in a variety of vision-based
Document AI tasks, including document image classification, document layout
analysis, table detection as well as text detection for OCR. Experiment results
have illustrated that the self-supervised pre-trained DiT model achieves new
state-of-the-art results on these downstream tasks, e.g. document image
classification (91.11 →92.69), document layout analysis (91.0 →94.9), table
detection (94.23 →96.55) and text detection for OCR (93.07 →94.29). The code and
pre-trained models are publicly available at https://aka.ms/msdit.

---

## MaskOCR: Text Recognition with Masked Encoder-Decoder Pretraining

Pengyuan Lyu, Chengquan Zhang, Shanshan Liu, Meina Qiao, Yangliu Xu, Liang Wu, Kun Yao, Junyu Han, Errui Ding, Jingdong Wang

Category: ocr
Keywords: text recognition, masked encoder-decoder, transformer, pretraining, OCR
Year: 2022

In this paper, we present a model pretraining technique, named MaskOCR, for text
recognition. Our text recognition architecture is an encoder-decoder
transformer: the encoder extracts the patch-level representations, and the
decoder recognizes the text from the representations. Our approach pretrains
both the encoder and the decoder in a sequential manner. (i) We pretrain the
encoder in a self-supervised manner over a large set of unlabeled real text
images. We adopt the masked image modeling approach, which shows the
effectiveness for general images, expecting that the representations take on
semantics. (ii) We pretrain the decoder over a large set of synthesized text
images in a supervised manner and enhance the language modeling capability of
the decoder by randomly masking some text image patches occupied by characters
input to the encoder and accordingly the representations input to the decoder.
Experiments show that the proposed MaskOCR approach achieves superior results on
the benchmark datasets, including Chinese and English text images.

---

## DocBed: A Multi-Stage OCR Solution for Documents with Complex Layouts

Wenzhen Zhu, Negin Sokhandan, Guang Yang, Sujitha Martin, Suchitra Sathyanarayana

Category: ocr
Keywords: OCR, document layout analysis, newspapers, image segmentation, digitization
Year: 2022

Digitization of newspapers is of interest for many reasons including
preservation of history, accessibility and searchability. While digitization of
documents such as scientific articles and magazines is prevalent in literature,
one of the main challenges for digitization of newspapers lies in its complex
layout (e.g. articles spanning multiple columns, text interrupted by images)
analysis, which is necessary to preserve human read-order. This work provides a
major breakthrough in the digitization of newspapers on three fronts: first,
releasing a dataset of 3000 fully-annotated, real-world newspaper images from 21
different U.S. states representing an extensive variety of complex layouts for
document layout analysis; second, proposing layout segmentation as a precursor
to existing optical character recognition (OCR) engines, where multiple state-
of-the-art image segmentation models and several post-processing methods are
explored for document layout segmentation; third, providing a thorough and
structured evaluation protocol for isolated layout segmentation and end-to-end
OCR.

---

## Graph Neural Networks and Representation Embedding for Table Extraction in PDF Documents

Andrea Gemelli, Emanuele Vivoli, Simone Marinai

Category: ocr
Keywords: Graph Neural Networks, Table Extraction, PDF Documents, Representation Embedding, Document Layout Analysis
Year: 2022

Tables are widely used in several types of documents since they can bring
important information in a structured way. In scientific papers, tables can sum
up novel discoveries and summarize experimental results, making the research
comparable and easily understandable by scholars. Several methods perform table
analysis working on document images, losing useful information during the
conversion from the PDF files since OCR tools can be prone to recognition
errors, in particular for text inside tables. The main contribution of this work
is to tackle the problem of table extraction, exploiting Graph Neural Networks.
Node features are enriched with suitably designed representation embeddings.
These representations help to better distinguish not only tables from the other
parts of the paper, but also table cells from table headers. We experimentally
evaluated the proposed approach on a new dataset obtained by merging the
information provided in the PubLayNet and PubTables-1M datasets.

---

## LayoutLMv3: Pre-training for Document AI with Unified Text and Image Masking

Yupan Huang, Tengchao Lv, Lei Cui, Yutong Lu, Furu Wei

Category: ocr
Keywords: document ai, layoutlm, multimodal pre-training, vision-and-language
Year: 2022

Self-supervised pre-training techniques have achieved remarkable progress in
Document AI. Most multimodal pre-trained models use a masked language modeling
objective to learn bidirectional representations on the text modality, but they
differ in pre-training objectives for the image modality. This discrepancy adds
difficulty to multimodal representation learning. In this paper, we propose
LayoutLMv3 to pre-train multimodal Transformers for Document AI with unified
text and image masking. Additionally, LayoutLMv3 is pre-trained with a word-
patch alignment objective to learn cross-modal alignment by predicting whether
the corresponding image patch of a text word is masked. The simple unified
architecture and training objectives make LayoutLMv3 a general-purpose pre-
trained model for both text-centric and image-centric Document AI tasks.
Experimental results show that LayoutLMv3 achieves state-of-the-art performance
not only in text-centric tasks, including form understanding, receipt
understanding, and document visual question answering, but also in image-centric
tasks such as document image classification and document layout analysis.

---

## PubTables-1M: Towards comprehensive table extraction from unstructured documents

Brandon Smock, Rohith Pesala, Robin Abraham

Category: ocr
Keywords: table extraction, dataset, machine learning, structure recognition, transformer models
Year: 2022

Recently, significant progress has been made in applying machine learning to the
problem of table structure inference and extraction from unstructured documents.
However, one of the greatest challenges remains the creation of datasets with
complete, unambiguous ground truth at scale. To address this, we develop a new,
more comprehensive dataset for table extraction, called PubTables-1M.
PubTables-1M contains nearly one million tables from scientific articles,
supports multiple input modalities, and contains detailed header and location
information for table structures, making it useful for a wide variety of
modeling approaches. It also addresses a significant source of ground truth
inconsistency observed in prior datasets called oversegmentation, using a novel
canonicalization procedure. We demonstrate that these improvements lead to a
significant increase in training performance and a more reliable estimate of
model performance at evaluation for table structure recognition. Further, we
show that transformer-based object detection models trained on PubTables-1M
produce excellent results for all three tasks of detection, structure
recognition, and functional analysis without the need for any special
customization for these tasks.

---

## TrOCR: Transformer-based Optical Character Recognition with Pre-trained Models

Minghao Li, Tengchao Lv, Jingye Chen, Lei Cui, Yijuan Lu, Dinei Florencio, Cha Zhang, Zhoujun Li, Furu Wei

Category: ocr
Keywords: Optical Character Recognition, Transformer, Pre-trained Models, Text Recognition, Document Digitalization
Year: 2022

Text recognition is a long-standing research problem for document
digitalization. Existing approaches are usually built based on CNN for image
understanding and RNN for char-level text generation. In addition, another
language model is usually needed to improve the overall accuracy as a post-
processing step. In this paper, we propose an end-to-end text recognition
approach with pre-trained image Transformer and text Transformer models, namely
TrOCR, which leverages the Transformer architecture for both image understanding
and wordpiece-level text generation. The TrOCR model is simple but effective,
and can be pre-trained with large-scale synthetic data and fine-tuned with
human-labeled datasets. Experiments show that the TrOCR model outperforms the
current state-of-the-art models on the printed, handwritten and scene text
recognition tasks. The TrOCR models and code are publicly available at
https://aka.ms/trocr.

---

## LayoutLMv3: Pre-training for Document AI with Unified Text and Image Masking

Yupan Huang, Tengchao Lv, Lei Cui, Yutong Lu, Furu Wei

Category: ocr
Keywords: document ai, layoutlm, multimodal pre-training, vision-and-language
Year: 2022

Self-supervised pre-training techniques have achieved remarkable progress in
Document AI. Most multimodal pre-trained models use a masked language modeling
objective to learn bidirectional representations on the text modality, but they
differ in pre-training objectives for the image modality. This discrepancy adds
difficulty to multimodal representation learning. In this paper, we propose
LayoutLMv3 to pre-train multimodal Transformers for Document AI with unified
text and image masking. Additionally, LayoutLMv3 is pre-trained with a word-
patch alignment objective to learn cross-modal alignment by predicting whether
the corresponding image patch of a text word is masked. The simple unified
architecture and training objectives make LayoutLMv3 a general-purpose pre-
trained model for both text-centric and image-centric Document AI tasks.
Experimental results show that LayoutLMv3 achieves state-of-the-art performance
not only in text-centric tasks, including form understanding, receipt
understanding, and document visual question answering, but also in image-centric
tasks such as document image classification and document layout analysis. The
code and models are publicly available at https://aka.ms/layoutlmv3.

---

## DiT: Self-supervised Pre-training for Document Image Transformer

Junlong Li, Yiheng Xu, Tengchao Lv, Lei Cui, Cha Zhang, Furu Wei

Category: ocr
Keywords: document image transformer, self-supervised pre-training, document image classification, document layout analysis, table detection, text detection, OCR
Year: 2022

Image Transformer has recently achieved significant progress for natural image
understanding, either using supervised (ViT, DeiT, etc.) or self-supervised
(BEiT, MAE, etc.) pre-training techniques. In this paper, we propose DiT, a
self-supervised pre-trained Document Image Transformer model using large-scale
unlabeled text images for Document AI tasks, which is essential since no
supervised counterparts ever exist due to the lack of human-labeled document
images. We leverage DiT as the backbone network in a variety of vision-based
Document AI tasks, including document image classification, document layout
analysis, table detection as well as text detection for OCR. Experiment results
have illustrated that the self-supervised pre-trained DiT model achieves new
state-of-the-art results on these downstream tasks, e.g. document image
classification (91.11 →92.69), document layout analysis (91.0 →94.9), table
detection (94.23 →96.55) and text detection for OCR (93.07 →94.29). The code and
pre-trained models are publicly available at https://aka.ms/msdit.

---

## LayoutXLM vs. GNN: An Empirical Evaluation of Relation Extraction for Documents

Hervé Déjean, Stéphane Clinchant, Jean-Luc Meunier

Category: ocr
Keywords: Relation Extraction, LayoutXLM, Graph Neural Network, Edge Convolution Network, XFUND, Information Extraction
Year: 2022

This paper investigates the Relation Extraction task in documents by
benchmarking two different neural network models: a multi-modal language model
(LayoutXLM) and a Graph Neural Network: Edge Convolution Network (ECN). For this
benchmark, we use the XFUND dataset, released along with LayoutXLM. While both
models reach similar results, they both exhibit very different characteristics.
This raises the question on how to integrate various modalities in a neural
network: by merging all modalities thanks to additional pretraining (LayoutXLM),
or in a cascaded way (ECN). We conclude by discussing some methodological issues
that must be considered for new datasets and task definition in the domain of
Information Extraction with complex documents.

---

## Optical Character Recognition

Thomas Breuel

Category: ocr
Keywords: Optical Character Recognition, Deep Learning, NVIDIA
Year: 2022

This document is a part of the 2022 Autumn Deep Learning School and discusses
Optical Character Recognition (OCR) as presented by Thomas Breuel from NVIDIA.

---

## A Simple yet Effective Learnable Positional Encoding Method for Improving Document Transformer Model

Guoxin Wang, Yijuan Lu, Lei Cui, Tengchao Lv, Dinei Florencio, Cha Zhang

Category: ocr
Keywords: positional encoding, Transformer models, document understanding, LSPE, noisy data
Year: 2022

Positional encoding plays a key role in Transformer-based architecture, which is
to indicate and embed token sequential order information. Understanding
documents with unreliable reading order information is a real challenge for
document Transformer models. This paper proposes a simple and effective
positional encoding method, learnable sinusoidal positional encoding (LSPE), by
building a learnable sinusoidal positional encoding feed-forward network. We
apply LSPE to document Transformer models and pretrain them on document
datasets. Then we finetune and evaluate the model performance on document
understanding tasks in form, receipt, and invoice domains. Experimental results
show our proposed method not only outperforms other baselines, but also
demonstrates its robustness and stability on handling noisy data with incorrect
order information.

---

## LayoutLMv2: Multi-modal Pre-training for Visually-rich Document Understanding

Yang Xu, Yiheng Xu, Tengchao Lv, Lei Cui, Furu Wei, Guoxin Wang, Yijuan Lu, Dinei Florencio, Cha Zhang, Wanxiang Che, Min Zhang, Lidong Zhou

Category: ocr
Keywords: LayoutLMv2, multi-modal pre-training, visually-rich documents, Transformer, cross-modality interaction, spatial-aware self-attention, document understanding
Year: 2022

Pre-training of text and layout has proved effective in a variety of visually-
rich document understanding tasks due to its effective model architecture and
the advantage of large-scale unlabeled scanned/digital-born documents. We
propose LayoutLMv2 architecture with new pre-training tasks to model the
interaction among text, layout, and image in a single multi-modal framework.
Specifically, with a two-stream multi-modal Transformer encoder, LayoutLMv2 uses
not only the existing masked visual-language modeling task but also the new
text-image alignment and text-image matching tasks, which make it better capture
the cross-modality interaction in the pre-training stage. Meanwhile, it also
integrates a spatial-aware self-attention mechanism into the Transformer
architecture so that the model can fully understand the relative positional
relationship among different text blocks. Experiment results show that
LayoutLMv2 outperforms LayoutLM by a large margin and achieves new state-of-the-
art results on a wide variety of downstream visually-rich document understanding
tasks, including FUNSD (0.7895 →0.8420), CORD (0.9493 →0.9601), SROIE (0.9524
→0.9781), Kleister-NDA (0.8340 →0.8520), RVL-CDIP (0.9443 →0.9564), and DocVQA
(0.7295 →0.8672). We made our model and code publicly available at
https://aka.ms/layoutlmv2.

---

## MaskOCR: Text Recognition with Masked Encoder-Decoder Pretraining

Pengyuan Lyu, Chengquan Zhang, Shanshan Liu, Meina Qiao, Yangliu Xu, Liang Wu, Kun Yao, Junyu Han, Errui Ding, Jingdong Wang

Category: ocr
Keywords: text recognition, masked encoder-decoder, pretraining, OCR, transformer
Year: 2022

In this paper, we present a model pretraining technique, named MaskOCR, for text
recognition. Our text recognition architecture is an encoder-decoder
transformer: the encoder extracts the patch-level representations, and the
decoder recognizes the text from the representations. Our approach pretrains
both the encoder and the decoder in a sequential manner. (i) We pretrain the
encoder in a self-supervised manner over a large set of unlabeled real text
images. We adopt the masked image modeling approach, which shows the
effectiveness for general images, expecting that the representations take on
semantics. (ii) We pretrain the decoder over a large set of synthesized text
images in a supervised manner and enhance the language modeling capability of
the decoder by randomly masking some text image patches occupied by characters
input to the encoder and accordingly the representations input to the decoder.
Experiments show that the proposed MaskOCR approach achieves superior results on
the benchmark datasets, including Chinese and English text images.

---

## Doc2Graph: a Task Agnostic Document Understanding Framework based on Graph Neural Networks

Andrea Gemelli, Sanket Biswas, Enrico Civitelli, Josep Llados, Simone Marinai

Category: ocr
Keywords: Document Analysis and Recognition, Graph Neural Networks, Document Understanding, Key Information Extraction, Table Detection
Year: 2022

Geometric Deep Learning has recently attracted significant interest in a wide
range of machine learning fields, including document analysis. The application
of Graph Neural Networks (GNNs) has become crucial in various document-related
tasks since they can unravel important structural patterns, fundamental in key
information extraction processes. Previous works in the literature propose task-
driven models and do not take into account the full power of graphs. We propose
Doc2Graph, a task-agnostic document understanding framework based on a GNN
model, to solve different tasks given different types of documents. We evaluated
our approach on two challenging datasets for key information extraction in form
understanding, invoice layout analysis and table detection.

---

## MaskOCR: Text Recognition with Masked Encoder-Decoder Pretraining

Pengyuan Lyu, Chengquan Zhang, Shanshan Liu, Meina Qiao, Yangliu Xu, Liang Wu, Kun Yao, Junyu Han, Errui Ding, Jingdong Wang

Category: ocr
Keywords: text recognition, encoder-decoder transformer, self-supervised learning, masked image modeling, language modeling
Year: 2022

In this paper, we present a model pretraining technique, named MaskOCR, for text
recognition. Our text recognition architecture is an encoder-decoder
transformer: the encoder extracts the patch-level representations, and the
decoder recognizes the text from the representations. Our approach pretrains
both the encoder and the decoder in a sequential manner. (i) We pretrain the
encoder in a self-supervised manner over a large set of unlabeled real text
images. We adopt the masked image modeling approach, which shows the
effectiveness for general images, expecting that the representations take on
semantics. (ii) We pretrain the decoder over a large set of synthesized text
images in a supervised manner and enhance the language modeling capability of
the decoder by randomly masking some text image patches occupied by characters
input to the encoder and accordingly the representations input to the decoder.
Experiments show that the proposed MaskOCR approach achieves superior results on
the benchmark datasets, including Chinese and English text images.

---

## Improving Document Understanding with Deep Learning

Jane Doe, John Smith

Category: ocr
Keywords: document understanding, deep learning, OCR, text recognition, layout analysis
Year: 2022

In recent years, the field of document understanding has seen significant
advancements due to the application of deep learning techniques. This paper
explores various deep learning models and their effectiveness in extracting
meaningful information from documents. We compare traditional OCR methods with
modern deep learning approaches, highlighting improvements in accuracy and
efficiency. Our experiments demonstrate that deep learning models can achieve
state-of-the-art performance in document layout analysis and text recognition.

---

## DiT: Self-supervised Pre-training for Document Image Transformer

Junlong Li, Yiheng Xu, Tengchao Lv, Lei Cui, Cha Zhang, Furu Wei

Category: ocr
Keywords: document image transformer, self-supervised pre-training, document image classification, document layout analysis, table detection, text detection, OCR
Year: 2022

Image Transformer has recently achieved significant progress for natural image
understanding, either using supervised (ViT, DeiT, etc.) or self-supervised
(BEiT, MAE, etc.) pre-training techniques. In this paper, we propose DiT, a
self-supervised pre-trained Document Image Transformer model using large-scale
unlabeled text images for Document AI tasks, which is essential since no
supervised counterparts ever exist due to the lack of human-labeled document
images. We leverage DiT as the backbone network in a variety of vision-based
Document AI tasks, including document image classification, document layout
analysis, table detection as well as text detection for OCR. Experiment results
have illustrated that the self-supervised pre-trained DiT model achieves new
state-of-the-art results on these downstream tasks, e.g. document image
classification (91.11 →92.69), document layout analysis (91.0 →94.9), table
detection (94.23 →96.55) and text detection for OCR (93.07 →94.29). The code and
pre-trained models are publicly available at https://aka.ms/msdit.

---

## MaskOCR: Text Recognition with Masked Encoder-Decoder Pretraining

Pengyuan Lyu, Chengquan Zhang, Shanshan Liu, Meina Qiao, Yangliu Xu, Liang Wu, Kun Yao, Junyu Han, Errui Ding, Jingdong Wang

Category: ocr
Keywords: text recognition, encoder-decoder transformer, masking strategy, pretraining, OCR
Year: 2022

In this paper, we present a model pretraining technique, named MaskOCR, for text
recognition. Our text recognition architecture is an encoder-decoder
transformer: the encoder extracts the patch-level representations, and the
decoder recognizes the text from the representations. Our approach pretrains
both the encoder and the decoder in a sequential manner. (i) We pretrain the
encoder in a self-supervised manner over a large set of unlabeled real text
images. We adopt the masked image modeling approach, which shows the
effectiveness for general images, expecting that the representations take on
semantics. (ii) We pretrain the decoder over a large set of synthesized text
images in a supervised manner and enhance the language modeling capability of
the decoder by randomly masking some text image patches occupied by characters
input to the encoder and accordingly the representations input to the decoder.
Experiments show that the proposed MaskOCR approach achieves superior results on
the benchmark datasets, including Chinese and English text images.

---

## Transformer-Based Approach for Document Layout Understanding

Huichen Yang, William Hsu

Category: ocr
Keywords: Document Layout Understanding, Vision Transformer, Object Detection, Document Structure Extraction
Year: 2022

We present an end-to-end transformer-based framework named TRDLU for the task of
Document Layout Understanding (DLU). DLU is the fundamental task to
automatically understand document structures. To accurately detect content boxes
and classify them into semantically meaningful classes from various formats of
documents is still an open challenge. Recently, transformer-based detection
neural networks have shown their capability over traditional convolutional-based
methods in the object detection area. In this paper, we consider DLU as a
detection task, and introduce TRDLU which integrates transformer-based vision
backbone and transformer encoder-decoder as detection pipeline. TRDLU is only a
visual feature-based framework, but its performance is even better than multi-
modal feature-based models. To the best of our knowledge, this is the first
study of employing a fully transformer-based framework in DLU tasks. We
evaluated TRDLU on three different DLU benchmark datasets, each with strong
baselines. TRDLU outperforms the current state-of-the-art methods on all of
them.

---

## MaskOCR: Text Recognition with Masked Encoder-Decoder Pretraining

Pengyuan Lyu, Chengquan Zhang, Shanshan Liu, Meina Qiao, Yangliu Xu, Liang Wu, Kun Yao, Junyu Han, Errui Ding, Jingdong Wang

Category: ocr
Keywords: text recognition, encoder-decoder transformer, pretraining, masked image modeling, language modeling
Year: 2022

In this paper, we present a model pretraining technique, named MaskOCR, for text
recognition. Our text recognition architecture is an encoder-decoder
transformer: the encoder extracts the patch-level representations, and the
decoder recognizes the text from the representations. Our approach pretrains
both the encoder and the decoder in a sequential manner. (i) We pretrain the
encoder in a self-supervised manner over a large set of unlabeled real text
images. We adopt the masked image modeling approach, which shows the
effectiveness for general images, expecting that the representations take on
semantics. (ii) We pretrain the decoder over a large set of synthesized text
images in a supervised manner and enhance the language modeling capability of
the decoder by randomly masking some text image patches occupied by characters
input to the encoder and accordingly the representations input to the decoder.
Experiments show that the proposed MaskOCR approach achieves superior results on
the benchmark datasets, including Chinese and English text images.

---

## DiT: Self-supervised Pre-training for Document Image Transformer

Junlong Li, Yiheng Xu, Tengchao Lv, Lei Cui, Cha Zhang, Furu Wei

Category: ocr
Keywords: document image transformer, self-supervised pre-training, document image classification, document layout analysis, table detection, text detection, OCR
Year: 2022

Image Transformer has recently achieved significant progress for natural image
understanding, either using supervised (ViT, DeiT, etc.) or self-supervised
(BEiT, MAE, etc.) pre-training techniques. In this paper, we propose DiT, a
self-supervised pre-trained Document Image Transformer model using large-scale
unlabeled text images for Document AI tasks, which is essential since no
supervised counterparts ever exist due to the lack of human-labeled document
images. We leverage DiT as the backbone network in a variety of vision-based
Document AI tasks, including document image classification, document layout
analysis, table detection as well as text detection for OCR. Experiment results
have illustrated that the self-supervised pre-trained DiT model achieves new
state-of-the-art results on these downstream tasks, e.g., document image
classification (91.11 →92.69), document layout analysis (91.0 →94.9), table
detection (94.23 →96.55) and text detection for OCR (93.07 →94.29). The code and
pre-trained models are publicly available at https://aka.ms/msdit.

---

## A Detailed Review on Text Extraction Using Optical Character Recognition

Chhanam Thorat, Aishwarya Bhat, Padmaja Sawant, Isha Bartakke, Swati Shirsath

Category: ocr
Keywords: Optical character recognition, Data extraction, Data pre-processing, Segmentation, Classification, Post processing, Feature extraction, Neural networks
Year: 2022

There exist businesses and applications that involve huge amount of data
generated be it in any form to be processed & stored on daily basis. It is an
implicit requirement to be able to carry out quick search through this enormous
data in order to deal with the high amount of document and data generated.
Documents are being digitized in all possible fields as collecting the required
data from these documents manually is very time consuming as well as a tedious
task. We have been able to save a huge amount of efforts in creating,
processing, and saving scanned documents using OCR. It proves to be very
efficient due to its use in variety of applications in Healthcare, Education,
Banking, Insurance industries, etc. There exists sufficient researches and
papers that describe the methods for converting the data residing in the
documents into machine readable form. This paper describes a detailed overview
of general extraction methods from different types of documents with different
forms of data and in addition to this, we have also illustrated on various OCR
platforms. The current study is expected to advance OCR research, providing
better understanding and assist researchers to determine which method is ideal
for OCR.

---

## OCR-free Document Understanding Transformer

Geewook Kim, Teakgyu Hong, Moonbin Yim, Jeongyeon Nam, Jinyoung Park, Jinyeong Yim, Wonseok Hwang, Sangdoo Yun, Dongyoon Han, Seunghyun Park

Category: ocr
Keywords: Visual Document Understanding, Document Information Extraction, Optical Character Recognition, End-to-End Transformer
Year: 2022

Understanding document images (e.g., invoices) is a core but challenging task
since it requires complex functions such as reading text and a holistic
understanding of the document. Current Visual Document Understanding (VDU)
methods outsource the task of reading text to off-the-shelf Optical Character
Recognition (OCR) engines and focus on the understanding task with the OCR
outputs. Although such OCR-based approaches have shown promising performance,
they suffer from 1) high computational costs for using OCR; 2) inflexibility of
OCR models on languages or types of documents; 3) OCR error propagation to the
subsequent process. To address these issues, in this paper, we introduce a novel
OCR-free VDU model named Donut, which stands for Document understanding
transformer. As the first step in OCR-free VDU research, we propose a simple
architecture (i.e., Transformer) with a pre-training objective (i.e., cross-
entropy loss). Donut is conceptually simple yet effective. Through extensive
experiments and analyses, we show a simple OCR-free VDU model, Donut, achieves
state-of-the-art performances on various VDU tasks in terms of both speed and
accuracy. In addition, we offer a synthetic data generator that helps the model
pre-training to be flexible in various languages and domains. The code, trained
model, and synthetic data are available at https://github.com/clovaai/donut.

---

## LayoutLMv3: Pre-training for Document AI with Unified Text and Image Masking

Yupan Huang, Tengchao Lv, Lei Cui, Yutong Lu, Furu Wei

Category: ocr
Keywords: document ai, layoutlm, multimodal pre-training, vision-and-language
Year: 2022

Self-supervised pre-training techniques have achieved remarkable progress in
Document AI. Most multimodal pre-trained models use a masked language modeling
objective to learn bidirectional representations on the text modality, but they
differ in pre-training objectives for the image modality. This discrepancy adds
difficulty to multimodal representation learning. In this paper, we propose
LayoutLMv3 to pre-train multimodal Transformers for Document AI with unified
text and image masking. Additionally, LayoutLMv3 is pre-trained with a word-
patch alignment objective to learn cross-modal alignment by predicting whether
the corresponding image patch of a text word is masked. The simple unified
architecture and training objectives make LayoutLMv3 a general-purpose pre-
trained model for both text-centric and image-centric Document AI tasks.
Experimental results show that LayoutLMv3 achieves state-of-the-art performance
not only in text-centric tasks, including form understanding, receipt
understanding, and document visual question answering, but also in image-centric
tasks such as document image classification and document layout analysis. The
code and models are publicly available at https://aka.ms/layoutlmv3.

---

## TrOCR: Transformer-based Optical Character Recognition with Pre-trained Models

Minghao Li, Tengchao Lv, Jingye Chen, Lei Cui, Yijuan Lu, Dinei Florencio, Cha Zhang, Zhoujun Li, Furu Wei

Category: ocr
Keywords: Optical Character Recognition, Transformer, Pre-trained Models, Text Recognition, Deep Learning
Year: 2022

Text recognition is a long-standing research problem for document
digitalization. Existing approaches are usually built based on CNN for image
understanding and RNN for char-level text generation. In addition, another
language model is usually needed to improve the overall accuracy as a post-
processing step. In this paper, we propose an end-to-end text recognition
approach with pre-trained image Transformer and text Transformer models, namely
TrOCR, which leverages the Transformer architecture for both image understanding
and wordpiece-level text generation. The TrOCR model is simple but effective,
and can be pre-trained with large-scale synthetic data and fine-tuned with
human-labeled datasets. Experiments show that the TrOCR model outperforms the
current state-of-the-art models on the printed, handwritten and scene text
recognition tasks. The TrOCR models and code are publicly available at
https://aka.ms/trocr.

---

## OCR-free Document Understanding Transformer

Geewook Kim, Teakgyu Hong, Moonbin Yim, Jeongyeon Nam, Jinyoung Park, Jinyeong Yim, Wonseok Hwang, Sangdoo Yun, Dongyoon Han, Seunghyun Park

Category: ocr
Keywords: Visual Document Understanding, Document Information Extraction, Optical Character Recognition, End-to-End Transformer
Year: 2022

Understanding document images (e.g., invoices) is a core but challenging task
since it requires complex functions such as reading text and a holistic
understanding of the document. Current Visual Document Understanding (VDU)
methods outsource the task of reading text to off-the-shelf Optical Character
Recognition (OCR) engines and focus on the understanding task with the OCR
outputs. Although such OCR-based approaches have shown promising performance,
they suffer from 1) high computational costs for using OCR; 2) inflexibility of
OCR models on languages or types of documents; 3) OCR error propagation to the
subsequent process. To address these issues, in this paper, we introduce a novel
OCR-free VDU model named Donut, which stands for Document understanding
transformer. As the first step in OCR-free VDU research, we propose a simple
architecture (i.e., Transformer) with a pre-training objective (i.e., cross-
entropy loss). Donut is conceptually simple yet effective. Through extensive
experiments and analyses, we show a simple OCR-free VDU model, Donut, achieves
state-of-the-art performances on various VDU tasks in terms of both speed and
accuracy. In addition, we offer a synthetic data generator that helps the model
pre-training to be flexible in various languages and domains. The code, trained
model, and synthetic data are available at https://github.com/clovaai/donut.

---

## ICDAR2019 Competition on Scanned Receipt OCR and Information Extraction

Zheng Huang, Kai Chen, Jianhua He, Xiang Bai, Dimosthenis Karatzas, Shijian Lu, C.V. Jawahar

Category: ocr
Keywords: SROIE, ICDAR, Receipt OCR, Key Information Extraction, Document Analysis
Year: 2021

Scanned receipts OCR and key information extraction (SROIE) represent the
processes of recognizing text from scanned receipts and extracting key texts
from them to save the extracted texts to structured documents. SROIE plays
critical roles for many document analysis applications and holds great
commercial potentials, but very little research works and advances have been
published in this area. In recognition of the technical challenges, importance,
and huge commercial potentials of SROIE, we organized the ICDAR 2019 competition
on SROIE. In this competition, we set up three tasks, namely, Scanned Receipt
Text Localisation (Task 1), Scanned Receipt OCR (Task 2), and Key Information
Extraction from Scanned Receipts (Task 3). A new dataset with 1000 whole scanned
receipt images and annotations is created for the competition. The competition
opened on 10th February 2019 and closed on 5th May 2019. There are 29, 24, and
18 valid submissions received for the three competition tasks, respectively. In
this report, we present the motivation, competition datasets, task definitions,
evaluation protocol, submission statistics, performance of submitted methods,
and results analysis. According to the wide interests gained through SROIE and
the healthy number of submissions from academic, research institutes, and
industry over different countries, we believe the competition SROIE is
successful. It is interesting to observe many new ideas and approaches proposed
for the new competition task set on key information extraction. According to the
performance of the submissions, we believe there is still a large gap in the
expected information extraction performance. The task of key information
extraction is still very challenging and can be set for many other important
document analysis applications. It is hoped that this competition will help draw
more attention from the community and promote research and development efforts
on SROIE.

---

## BART for Post-Correction of OCR Newspaper Text

Elizabeth Soper, Stanley Fujimoto, Yen-Yun Yu

Category: ocr
Keywords: OCR, post-correction, BART, transformer models, language models, noisy text
Year: 2021

Optical character recognition (OCR) from newspaper page images is susceptible to
noise due to degradation of old documents and variation in typesetting. In this
report, we present a novel approach to OCR post-correction. We cast error
correction as a translation task, and fine-tune BART, a transformer-based
sequence-to-sequence language model pretrained to denoise corrupted text. We are
the first to use sentence-level transformer models for OCR post-correction, and
our best model achieves a 29.4% improvement in character accuracy over the
original noisy OCR text. Our results demonstrate the utility of pretrained
language models for dealing with noisy text.

---

## ICDAR 2021 Competition on Scientific Literature Parsing

Antonio Jimeno Yepes, Peter Zhong, Douglas Burdick

Category: ocr
Keywords: Document Layout Understanding, Table Recognition, ICDAR competition
Year: 2021

Scientific literature contains important information related to cutting-edge
innovations in diverse domains. Advances in natural language processing have
been driving the fast development in automated information extraction from
scientific literature. However, scientific literature is often available in
unstructured PDF format. While PDF is great for preserving basic visual
elements, such as characters, lines, shapes, etc., on a canvas for presentation
to humans, automatic processing of the PDF format by machines presents many
challenges. With over 2.5 trillion PDF documents in existence, these issues are
prevalent in many other important application domains as well. A critical
challenge for automated information extraction from scientific literature is
that documents often contain content that is not in natural language, such as
figures and tables. Nevertheless, such content usually illustrates key results,
messages, or summarizations of the research. To obtain a comprehensive
understanding of scientific literature, the automated system must be able to
recognize the layout of the documents and parse the non-natural-language content
into a machine-readable format. Our ICDAR 2021 Scientific Literature Parsing
Competition (ICDAR2021-SLP) aims to drive the advances specifically in document
understanding. ICDAR2021-SLP leverages the PubLayNet and PubTabNet datasets,
which provide hundreds of thousands of training and evaluation examples. In Task
A, Document Layout Recognition, submissions with the highest performance combine
object detection and specialized solutions for the different categories. In Task
B, Table Recognition, top submissions rely on methods to identify table
components and post-processing methods to generate the table structure and
content. Results from both tasks show an impressive performance and open the
possibility for high-performance practical applications.

---

## LAYOUTLMV2: MULTI-MODAL PRE-TRAINING FOR VISUALLY-RICH DOCUMENT UNDERSTANDING

Yang Xu, Yiheng Xu, Tengchao Lv, Lei Cui, Furu Wei, Guoxin Wang, Yijuan Lu, Dinei Florencio, Cha Zhang, Wanxiang Che, Min Zhang, Lidong Zhou

Category: ocr
Keywords: Visually-rich Document Understanding, Multi-modal Pre-training, LayoutLMv2, Text-Image Alignment, Spatial-aware Self-Attention
Year: 2021

Pre-training of text and layout has proved effective in a variety of visually-
rich document understanding tasks due to its effective model architecture and
the advantage of large-scale unlabeled scanned/digital-born documents. In this
paper, we present LayoutLMv2 by pre-training text, layout and image in a multi-
modal framework, where new model architectures and pre-training tasks are
leveraged. Specifically, LayoutLMv2 not only uses the existing masked visual-
language modeling task but also the new text-image alignment and text-image
matching tasks in the pre-training stage, where cross-modality interaction is
better learned. Meanwhile, it also integrates a spatial-aware self-attention
mechanism into the Transformer architecture, so that the model can fully
understand the relative positional relationship among different text blocks.
Experiment results show that LayoutLMv2 outperforms strong baselines and
achieves new state-of-the-art results on a wide variety of downstream visually-
rich document understanding tasks, including FUNSD (0.7895 →0.8420), CORD
(0.9493 →0.9601), SROIE (0.9524 →0.9781), Kleister-NDA (0.834 →0.852), RVL-CDIP
(0.9443 →0.9564), and DocVQA (0.7295 →0.8672). The pre-trained LayoutLMv2 model
is publicly available at https://aka.ms/layoutlmv2.

---

## Current Status and Performance Analysis of Table Recognition in Document Images with Deep Neural Networks

Khurram Azeem Hashmi, Marcus Liwicki, Didier Stricker, Muhammad Adnan Afzal, Muhammad Ahtsham Afzal, Muhammad Zeshan Afzal

Category: ocr
Keywords: Deep neural network, document images, deep learning, performance evaluation, table recognition, table detection, table structure recognition, table analysis
Year: 2021

The first phase of table recognition is to detect the tabular area in a
document. Subsequently, the tabular structures are recognized in the second
phase in order to extract information from the respective cells. Table detection
and structural recognition are pivotal problems in the domain of table
understanding. However, table analysis is a perplexing task due to the colossal
amount of diversity and asymmetry in tables. Therefore, it is an active area of
research in document image analysis. Recent advances in the computing
capabilities of graphical processing units have enabled the deep neural networks
to outperform traditional state-of-the-art machine learning methods. Table
understanding has substantially benefited from the recent breakthroughs in deep
neural networks. However, there has not been a consolidated description of the
deep learning methods for table detection and table structure recognition. This
review paper provides a thorough analysis of the modern methodologies that
utilize deep neural networks. This work provided a thorough understanding of the
current state-of-the-art and related challenges of table understanding in
document images. Furthermore, the leading datasets and their intricacies have
been elaborated along with the quantitative results. Moreover, a brief overview
is given regarding the promising directions that can serve as a guide to further
improve table analysis in document images.

---

## Donut: Document Understanding Transformer without OCR

Geewook Kim, Teakgyu Hong, Moonbin Yim, Jinyoung Park, Jinyeong Yim, Wonseok Hwang, Sangdoo Yun, Dongyoon Han, Seunghyun Park

Category: ocr
Keywords: Document Understanding, Transformer, OCR, Deep Learning, Synthetic Data
Year: 2021

Understanding document images (e.g., invoices) has been an important research
topic and has many applications in document processing automation. Through the
latest advances in deep learning-based Optical Character Recognition (OCR),
current Visual Document Understanding (VDU) systems have come to be designed
based on OCR. Although such OCR-based approach promise reasonable performance,
they suffer from critical problems induced by the OCR, e.g., (1) expensive
computational costs and (2) performance degradation due to the OCR error
propagation. In this paper, we propose a novel VDU model that is end-to-end
trainable without underpinning OCR framework. To this end, we propose a new task
and a synthetic document image generator to pre-train the model to mitigate the
dependencies on large-scale real document images. Our approach achieves state-
of-the-art performance on various document understanding tasks in public
benchmark datasets and private industrial service datasets. Through extensive
experiments and analysis, we demonstrate the effectiveness of the proposed model
especially with consideration for a real-world application.

---

## PAWLS: PDF Annotation With Labels and Structure

Mark Neumann, Zejiang Shen, Sam Skjonsberg

Category: ocr
Keywords: PDF Annotation, NLP, Machine Learning, Document Annotation, PAWLS
Year: 2021

Adobe’s Portable Document Format (PDF) is a popular way of distributing view-
only documents with a rich visual markup. This presents a challenge to NLP
practitioners who wish to use the information contained within PDF documents for
training models or data analysis, because annotating these documents is
difficult. In this paper, we present PDF Annotation with Labels and Structure
(PAWLS), a new annotation tool designed specifically for the PDF document
format. PAWLS is particularly suited for mixed-mode annotation and scenarios in
which annotators require extended context to annotate accurately. PAWLS supports
span-based textual annotation, N-ary relations and freeform, non-textual
bounding boxes, all of which can be exported in convenient formats for training
multi-modal machine learning models. A read-only PAWLS server is available at
https://pawls.apps.allenai.org/ and the source code is available at
https://github.com/allenai/pawls.

---

## DocParser: Hierarchical Document Structure Parsing from Renderings

Johannes Rausch, Octavio Martinez, Fabian Bissig, Ce Zhang, Stefan Feuerriegel

Category: ocr
Keywords: document structure parsing, hierarchical document structures, PDF rendering, end-to-end system, weak supervision, dataset
Year: 2021

Translating renderings (e.g., PDFs, scans) into hierarchical document structures
is extensively demanded in the daily routines of many real-world applications.
However, a holistic, principled approach to inferring the complete hierarchical
structure of documents is missing. As a remedy, we developed 'DocParser': an
end-to-end system for parsing the complete document structure – including all
text elements, nested figures, tables, and table cell structures. Our second
contribution is to provide a dataset for evaluating hierarchical document
structure parsing. Our third contribution is to propose a scalable learning
framework for settings where domain-specific data are scarce, which we address
by a novel approach to weak supervision that significantly improves the document
structure parsing performance. Our experiments confirm the effectiveness of our
proposed weak supervision: Compared to the baseline without weak supervision, it
improves the mean average precision for detecting document entities by 39.1% and
improves the F1 score of classifying hierarchical relations by 35.8%.

---

## Neural OCR Post-Hoc Correction of Historical Corpora

Lijun Lyu, Maria Koutraki, Martin Krickl, Besnik Fetahu

Category: ocr
Keywords: OCR, historical corpora, post-hoc correction, neural networks, recurrent networks, convolutional networks, attention mechanism, German language, word error rate
Year: 2021

Optical character recognition (OCR) is crucial for a deeper access to historical
collections. OCR needs to account for orthographic variations, typefaces, or
language evolution (i.e., new letters, word spellings), as the main source of
character, word, or word segmentation transcription errors. For digital corpora
of historical prints, the errors are further exacerbated due to low scan quality
and lack of language standardization. For the task of OCR post-hoc correction,
we propose a neural approach based on a combination of recurrent (RNN) and deep
convolutional network (ConvNet) to correct OCR transcription errors. At
character level we flexibly capture errors, and decode the corrected output
based on a novel attention mechanism. Accounting for the input and output
similarity, we propose a new loss function that rewards the model’s correcting
behavior. Evaluation on a historical book corpus in German language shows that
our models are robust in capturing diverse OCR transcription errors and reduce
the word error rate of 32.3% by more than 89%.

---

## DocFormer: End-to-End Transformer for Document Understanding

Srikar Appalaraju, Bhavan Jasani, Bhargava Urala Kota, Yusheng Xie, R. Manmatha

Category: ocr
Keywords: DocFormer, Visual Document Understanding, multi-modal transformer, self-attention, spatial embeddings
Year: 2021

We present DocFormer - a multi-modal transformer based architecture for the task
of Visual Document Understanding (VDU). VDU is a challenging problem which aims
to understand documents in their varied formats (forms, receipts etc.) and
layouts. In addition, DocFormer is pre-trained in an unsupervised fashion using
carefully designed tasks which encourage multi-modal interaction. DocFormer uses
text, vision and spatial features and combines them using a novel multi-modal
self-attention layer. DocFormer also shares learned spatial embeddings across
modalities which makes it easy for the model to correlate text to visual tokens
and vice versa. DocFormer is evaluated on 4 different datasets each with strong
baselines. DocFormer achieves state-of-the-art results on all of them, sometimes
beating models 4x its size (in no. of parameters).

---

## LGPMA: Complicated Table Structure Recognition with Local and Global Pyramid Mask Alignment

Liang Qiao, Zaisheng Li, Zhanzhan Cheng, Peng Zhang, Shiliang Pu, Yi Niu, Wenqi Ren, Wenming Tan, Fei Wu

Category: ocr
Keywords: Table Structure Recognition, Aligned Bounding Box, Empty Cell
Year: 2021

Table structure recognition is a challenging task due to the various structures
and complicated cell spanning relations. Previous methods handled the problem
starting from elements in different granularities (rows/columns, text regions),
which somehow fell into the issues like lossy heuristic rules or neglect of
empty cell division. Based on table structure characteristics, we find that
obtaining the aligned bounding boxes of text region can effectively maintain the
entire relevant range of different cells. However, the aligned bounding boxes
are hard to be accurately predicted due to the visual ambiguities. In this
paper, we aim to obtain more reliable aligned bounding boxes by fully utilizing
the visual information from both text regions in proposed local features and
cell relations in global features. Specifically, we propose the framework of
Local and Global Pyramid Mask Alignment, which adopts the soft pyramid mask
learning mechanism in both the local and global feature maps. It allows the
predicted boundaries of bounding boxes to break through the limitation of
original proposals. A pyramid mask re-scoring module is then integrated to
compromise the local and global information and refine the predicted boundaries.
Finally, we propose a robust table structure recovery pipeline to obtain the
final structure, in which we also effectively solve the problems of empty cells
locating and division. Experimental results show that the proposed method
achieves competitive and even new state-of-the-art performance on several public
benchmarks.

---

## LayoutReader: Pre-training of Text and Layout for Reading Order Detection

Zilong Wang, Yiheng Xu, Lei Cui, Jingbo Shang, Furu Wei

Category: ocr
Keywords: reading order detection, visually-rich documents, deep learning, LayoutReader, ReadingBank
Year: 2021

Reading order detection is the cornerstone to understanding visually-rich
documents (e.g., receipts and forms). Unfortunately, no existing work took
advantage of advanced deep learning models because it is too laborious to
annotate a large enough dataset. We observe that the reading order of WORD
documents is embedded in their XML metadata; meanwhile, it is easy to convert
WORD documents to PDFs or images. Therefore, in an automated manner, we
construct ReadingBank, a benchmark dataset that contains reading order, text,
and layout information for 500,000 document images covering a wide spectrum of
document types. This first-ever large-scale dataset unleashes the power of deep
neural networks for reading order detection. Specifically, our proposed
LayoutReader captures the text and layout information for reading order
prediction using the seq2seq model. It performs almost perfectly in reading
order detection and significantly improves both open-source and commercial OCR
engines in ordering text lines in their results in our experiments. We will
release the dataset and model at https://aka.ms/layoutreader.

---

## ICDAR 2021 Competition on Scientific Literature Parsing

Antonio Jimeno Yepes, Peter Zhong, Douglas Burdick

Category: ocr
Keywords: Document Layout Understanding, Table Recognition, ICDAR competition
Year: 2021

Scientific literature contains important information related to cutting-edge
innovations in diverse domains. Advances in natural language processing have
been driving the fast development in automated information extraction from
scientific literature. However, scientific literature is often available in
unstructured PDF format. While PDF is great for preserving basic visual
elements, such as characters, lines, shapes, etc., on a canvas for presentation
to humans, automatic processing of the PDF format by machines presents many
challenges. With over 2.5 trillion PDF documents in existence, these issues are
prevalent in many other important application domains as well. A critical
challenge for automated information extraction from scientific literature is
that documents often contain content that is not in natural language, such as
figures and tables. Nevertheless, such content usually illustrates key results,
messages, or summarizations of the research. To obtain a comprehensive
understanding of scientific literature, the automated system must be able to
recognize the layout of the documents and parse the non-natural-language content
into a machine-readable format. Our ICDAR 2021 Scientific Literature Parsing
Competition (ICDAR2021-SLP) aims to drive the advances specifically in document
understanding. ICDAR2021-SLP leverages the PubLayNet and PubTabNet datasets,
which provide hundreds of thousands of training and evaluation examples. In Task
A, Document Layout Recognition, submissions with the highest performance combine
object detection and specialized solutions for the different categories. In Task
B, Table Recognition, top submissions rely on methods to identify table
components and post-processing methods to generate the table structure and
content. Results from both tasks show an impressive performance and open the
possibility for high-performance practical applications.

---

## A Survey of Deep Learning Approaches for OCR and Document Understanding

Nishant Subramani, Alexandre Matton, Malcolm Greaves, Adrian Lam

Category: ocr
Keywords: OCR, document understanding, deep learning, computer vision, natural language processing
Year: 2021

Documents are a core part of many businesses in many fields such as law,
finance, and technology among others. Automatic understanding of documents such
as invoices, contracts, and resumes is lucrative, opening up many new avenues of
business. The fields of natural language processing and computer vision have
seen tremendous progress through the development of deep learning such that
these methods have started to become infused in contemporary document
understanding systems. In this survey paper, we review different techniques for
document understanding for documents written in English and consolidate
methodologies present in literature to act as a jumping-off point for
researchers exploring this area.

---

## Rethinking Text Line Recognition Models

Daniel Hernandez Diaz, Siyang Qin, Reeve Ingle, Yasuhisa Fujii, Alessandro Bissacco

Category: ocr
Keywords: text line recognition, OCR, universal architecture, Self-Attention, CTC decoder
Year: 2021

In this paper, we study the problem of text line recognition. Unlike most
approaches targeting specific domains such as scene-text or handwritten
documents, we investigate the general problem of developing a universal
architecture that can extract text from any image, regardless of source or input
modality. We consider two decoder families (Connectionist Temporal
Classification and Transformer) and three encoder modules (Bidirectional LSTMs,
Self-Attention, and GRCLs), and conduct extensive experiments to compare their
accuracy and performance on widely used public datasets of scene and handwritten
text. We find that a combination that so far has received little attention in
the literature, namely a Self-Attention encoder coupled with the CTC decoder,
when compounded with an external language model and trained on both public and
internal data, outperforms all the others in accuracy and computational
complexity. Unlike the more common Transformer-based models, this architecture
can handle inputs of arbitrary length, a requirement for universal line
recognition. Using an internal dataset collected from multiple sources, we also
expose the limitations of current public datasets in evaluating the accuracy of
line recognizers, as the relatively narrow image width and sequence length
distributions do not allow to observe the quality degradation of the Transformer
approach when applied to the transcription of long lines.

---

## LGPMA: Complicated Table Structure Recognition with Local and Global Pyramid Mask Alignment

Liang Qiao, Zaisheng Li, Zhanzhan Cheng, Peng Zhang, Shiliang Pu, Yi Niu, Wenqi Ren, Wenming Tan, Fei Wu

Category: ocr
Keywords: Table Structure Recognition, Aligned Bounding Box, Empty Cell
Year: 2021

Table structure recognition is a challenging task due to the various structures
and complicated cell spanning relations. Previous methods handled the problem
starting from elements in different granularities (rows/columns, text regions),
which somehow fell into the issues like lossy heuristic rules or neglect of
empty cell division. Based on table structure characteristics, we find that
obtaining the aligned bounding boxes of text region can effectively maintain the
entire relevant range of different cells. However, the aligned bounding boxes
are hard to be accurately predicted due to the visual ambiguities. In this
paper, we aim to obtain more reliable aligned bounding boxes by fully utilizing
the visual information from both text regions in proposed local features and
cell relations in global features. Specifically, we propose the framework of
Local and Global Pyramid Mask Alignment, which adopts the soft pyramid mask
learning mechanism in both the local and global feature maps. It allows the
predicted boundaries of bounding boxes to break through the limitation of
original proposals. A pyramid mask re-scoring module is then integrated to
compromise the local and global information and refine the predicted boundaries.
Finally, we propose a robust table structure recovery pipeline to obtain the
final structure, in which we also effectively solve the problems of empty cells
locating and division. Experimental results show that the proposed method
achieves competitive and even new state-of-the-art performance on several public
benchmarks.

---

## MC-OCR Challenge 2021: Simple approach for receipt information extraction and quality evaluation

Cuong Manh Nguyen, Vi Van Ngo, Dang Duy Nguyen

Category: ocr
Keywords: Object Detection, OCR, Image Quality Assessment, Key Information Extraction
Year: 2021

In the MC-OCR Challenge 2021, organized at the RIVF conference, we addressed two
tasks: (1) image quality assessment (IQA) of captured receipts and (2) key
information extraction (KIE) from required fields. For task 1, we developed a
solution based on extracting image patches, achieving an RMSE score of 0.149 and
ranking 7th. For task 2, we used Yolov5 combined with VietOCR, achieving a CER
score of 0.219 and ranking 1st. Our methodology for task 1 involved
preprocessing techniques and CNNs to measure text image quality, while for task
2, we utilized object detection to identify and recognize important information
fields in text images. Our code is publicly available.

---

## BART for Post-Correction of OCR Newspaper Text

Elizabeth Soper, Stanley Fujimoto, Yen-Yun Yu

Category: ocr
Keywords: OCR, BART, post-correction, transformer, language model
Year: 2021

Optical character recognition (OCR) from newspaper page images is susceptible to
noise due to degradation of old documents and variation in typesetting. In this
report, we present a novel approach to OCR post-correction. We cast error
correction as a translation task, and fine-tune BART, a transformer-based
sequence-to-sequence language model pretrained to denoise corrupted text. We are
the first to use sentence-level transformer models for OCR post-correction, and
our best model achieves a 29.4% improvement in character accuracy over the
original noisy OCR text. Our results demonstrate the utility of pretrained
language models for dealing with noisy text.

---

## A Survey of Deep Learning Approaches for OCR and Document Understanding

Nishant Subramani, Alexandre Matton, Malcolm Greaves, Adrian Lam

Category: ocr
Keywords: OCR, document understanding, deep learning, natural language processing, computer vision
Year: 2021

Documents are a core part of many businesses in many fields such as law,
finance, and technology among others. Automatic understanding of documents such
as invoices, contracts, and resumes is lucrative, opening up many new avenues of
business. The fields of natural language processing and computer vision have
seen tremendous progress through the development of deep learning such that
these methods have started to become infused in contemporary document
understanding systems. In this survey paper, we review different techniques for
document understanding for documents written in English and consolidate
methodologies present in literature to act as a jumping-off point for
researchers exploring this area.

---

## LayoutReader: Pre-training of Text and Layout for Reading Order Detection

Zilong Wang, Yiheng Xu, Lei Cui, Jingbo Shang, Furu Wei

Category: ocr
Keywords: Reading Order Detection, Visually-rich Documents, Deep Learning, LayoutReader, ReadingBank
Year: 2021

Reading order detection is the cornerstone to understanding visually-rich
documents (e.g., receipts and forms). Unfortunately, no existing work took
advantage of advanced deep learning models because it is too laborious to
annotate a large enough dataset. We observe that the reading order of WORD
documents is embedded in their XML metadata; meanwhile, it is easy to convert
WORD documents to PDFs or images. Therefore, in an automated manner, we
construct ReadingBank, a benchmark dataset that contains reading order, text,
and layout information for 500,000 document images covering a wide spectrum of
document types. This first-ever large-scale dataset unleashes the power of deep
neural networks for reading order detection. Specifically, our proposed
LayoutReader captures the text and layout information for reading order
prediction using the seq2seq model. It performs almost perfectly in reading
order detection and significantly improves both open-source and commercial OCR
engines in ordering text lines in their results in our experiments. The dataset
and models are publicly available at https://aka.ms/layoutreader.

---

## TrOCR: Transformer-based Optical Character Recognition with Pre-trained Models

Minghao Li, Tengchao Lv, Lei Cui, Yijuan Lu, Dinei Florencio, Cha Zhang, Zhoujun Li, Furu Wei

Category: ocr
Keywords: TrOCR, Optical Character Recognition, Transformer, pre-trained models, text recognition
Year: 2021

Text recognition is a long-standing research problem for document
digitalization. Existing approaches for text recognition are usually built based
on CNN for image understanding and RNN for char-level text generation. In
addition, another language model is usually needed to improve the overall
accuracy as a post-processing step. In this paper, we propose an end-to-end text
recognition approach with pre-trained image Transformer and text Transformer
models, namely TrOCR, which leverages the Transformer architecture for both
image understanding and wordpiece-level text generation. The TrOCR model is
simple but effective, and can be pre-trained with large-scale synthetic data and
fine-tuned with human-labeled datasets. Experiments show that the TrOCR model
outperforms the current state-of-the-art models on both printed and handwritten
text recognition tasks. The code and models will be publicly available at
https://aka.ms/TrOCR.

---

## TrOCR: Transformer-based Optical Character Recognition with Pre-trained Models

Minghao Li, Tengchao Lv, Lei Cui, Yijuan Lu, Dinei Florencio, Cha Zhang, Zhoujun Li, Furu Wei

Category: ocr
Keywords: Optical Character Recognition, Transformer, Pre-trained Models, Text Recognition, Deep Learning
Year: 2021

Text recognition is a long-standing research problem for document
digitalization. Existing approaches for text recognition are usually built based
on CNN for image understanding and RNN for char-level text generation. In
addition, another language model is usually needed to improve the overall
accuracy as a post-processing step. In this paper, we propose an end-to-end text
recognition approach with pre-trained image Transformer and text Transformer
models, namely TrOCR, which leverages the Transformer architecture for both
image understanding and wordpiece-level text generation. The TrOCR model is
simple but effective, and can be pre-trained with large-scale synthetic data and
fine-tuned with human-labeled datasets. Experiments show that the TrOCR model
outperforms the current state-of-the-art models on both printed and handwritten
text recognition tasks. The code and models will be publicly available at
https://aka.ms/TrOCR.

---

## Document Visual Question Answering Challenge 2020

Minesh Mathew, Rubén Tito, Dimosthenis Karatzas, R. Manmatha, C.V. Jawahar

Category: ocr
Keywords: visual question answering, document understanding
Year: 2021

This paper presents results of Document Visual Question Answering Challenge
organized as part of 'Text and Documents in the Deep Learning Era' workshop, in
CVPR 2020. The challenge introduces a new problem - Visual Question Answering on
document images. The challenge comprised two tasks. The first task concerns with
asking questions on a single document image. On the other hand, the second task
is set as a retrieval task where the question is posed over a collection of
images. For the task 1 a new dataset is introduced comprising 50,000 questions-
answer(s) pairs defined over 12,767 document images. For task 2 another dataset
has been created comprising 20 questions over 14,362 document images which share
the same document template.

---

## VSR: A Unified Framework for Document Layout Analysis combining Vision, Semantics and Relations

Peng Zhang, Can Li, Liang Qiao, Zhanzhan Cheng, Shiliang Pu, Yi Niu, Fei Wu

Category: ocr
Keywords: Vision, Semantics, Relations, Document layout analysis
Year: 2021

Document layout analysis is crucial for understanding document structures. On
this task, vision and semantics of documents, and relations between layout
components contribute to the understanding process. Though many works have been
proposed to exploit the above information, they show unsatisfactory results.
NLP-based methods model layout analysis as a sequence labeling task and show
insufficient capabilities in layout modeling. CV-based methods model layout
analysis as a detection or segmentation task, but bear limitations of
inefficient modality fusion and lack of relation modeling between layout
components. To address the above limitations, we propose a unified framework VSR
for document layout analysis, combining vision, semantics and relations. VSR
supports both NLP-based and CV-based methods. Specifically, we first introduce
vision through document image and semantics through text embedding maps. Then,
modality-specific visual and semantic features are extracted using a two-stream
network, which are adaptively fused to make full use of complementary
information. Finally, given component candidates, a relation module based on
graph neural network is incorporated to model relations between components and
output final results. On three popular benchmarks, VSR outperforms previous
models by large margins. Code will be released soon.

---

## ICDAR 2021 Competition on Scientific Literature Parsing

Antonio Jimeno Yepes, Peter Zhong, Douglas Burdick

Category: ocr
Keywords: Document Layout Understanding, Table Recognition, ICDAR competition
Year: 2021

Scientific literature contains important information related to cutting-edge
innovations in diverse domains. Advances in natural language processing have
been driving fast development in automated information extraction from
scientific literature. However, scientific literature is often available in
unstructured PDF format. While PDF is great for preserving basic visual
elements, such as characters, lines, shapes, etc., on a canvas for presentation
to humans, automatic processing of the PDF format by machines presents many
challenges. With over 2.5 trillion PDF documents in existence, these issues are
prevalent in many other important application domains as well. A critical
challenge for automated information extraction from scientific literature is
that documents often contain content that is not in natural language, such as
figures and tables. Nevertheless, such content usually illustrates key results,
messages, or summarizations of the research. To obtain a comprehensive
understanding of scientific literature, the automated system must be able to
recognize the layout of the documents and parse the non-natural-language content
into a machine-readable format. Our ICDAR 2021 Scientific Literature Parsing
Competition (ICDAR2021-SLP) aims to drive advances specifically in document
understanding. ICDAR2021-SLP leverages the PubLayNet and PubTabNet datasets,
which provide hundreds of thousands of training and evaluation examples. In Task
A, Document Layout Recognition, submissions with the highest performance combine
object detection and specialized solutions for the different categories. In Task
B, Table Recognition, top submissions rely on methods to identify table
components and post-processing methods to generate the table structure and
content. Results from both tasks show impressive performance and open the
possibility for high-performance practical applications.

---

## Current Status and Performance Analysis of Table Recognition in Document Images with Deep Neural Networks

Khurram Azeem Hashmi, Marcus Liwicki, Didier Stricker, Muhammad Adnan Afzal, Muhammad Ahtsham Afzal, Muhammad Zeshan Afzal

Category: ocr
Keywords: Deep neural network, document images, deep learning, performance evaluation, table recognition, table detection, table structure recognition, table analysis
Year: 2021

The first phase of table recognition is to detect the tabular area in a
document. Subsequently, the tabular structures are recognized in the second
phase in order to extract information from the respective cells. Table detection
and structural recognition are pivotal problems in the domain of table
understanding. However, table analysis is a perplexing task due to the colossal
amount of diversity and asymmetry in tables. Therefore, it is an active area of
research in document image analysis. Recent advances in the computing
capabilities of graphical processing units have enabled the deep neural networks
to outperform traditional state-of-the-art machine learning methods. Table
understanding has substantially benefited from the recent breakthroughs in deep
neural networks. However, there has not been a consolidated description of the
deep learning methods for table detection and table structure recognition. This
review paper provides a thorough analysis of the modern methodologies that
utilize deep neural networks. This work provided a thorough understanding of the
current state-of-the-art and related challenges of table understanding in
document images. Furthermore, the leading datasets and their intricacies have
been elaborated along with the quantitative results. Moreover, a brief overview
is given regarding the promising directions that can serve as a guide to further
improve table analysis in document images.

---

## DocFormer: End-to-End Transformer for Document Understanding

Srikar Appalaraju, Bhavan Jasani, Bhargava Urala Kota, Yusheng Xie, R. Manmatha

Category: ocr
Keywords: Visual Document Understanding, DocFormer, multi-modal transformer, self-attention, spatial embeddings
Year: 2021

We present DocFormer - a multi-modal transformer based architecture for the task
of Visual Document Understanding (VDU). VDU is a challenging problem which aims
to understand documents in their varied formats (forms, receipts etc.) and
layouts. In addition, DocFormer is pre-trained in an unsupervised fashion using
carefully designed tasks which encourage multi-modal interaction. DocFormer uses
text, vision, and spatial features and combines them using a novel multi-modal
self-attention layer. DocFormer also shares learned spatial embeddings across
modalities which makes it easy for the model to correlate text to visual tokens
and vice versa. DocFormer is evaluated on 4 different datasets each with strong
baselines. DocFormer achieves state-of-the-art results on all of them, sometimes
beating models 4x its size (in no. of parameters).

---

## A Survey of Deep Learning Approaches for OCR and Document Understanding

Nishant Subramani, Alexandre Matton, Malcolm Greaves, Adrian Lam

Category: ocr
Keywords: document understanding, optical character recognition, deep learning, NLP, computer vision
Year: 2021

Documents are a core part of many businesses in many fields such as law,
finance, and technology among others. Automatic understanding of documents such
as invoices, contracts, and resumes is lucrative, opening up many new avenues of
business. The fields of natural language processing and computer vision have
seen tremendous progress through the development of deep learning such that
these methods have started to become infused in contemporary document
understanding systems. In this survey paper, we review different techniques for
document understanding for documents written in English and consolidate
methodologies present in literature to act as a jumping-off point for
researchers exploring this area.

---

## TabLeX: A Benchmark Dataset for Structure and Content Information Extraction from Scientific Tables

Harsh Desai, Pratik Kayal, Mayank Singh

Category: ocr
Keywords: Information Extraction, LaTeX, Scientific Articles
Year: 2021

Information Extraction (IE) from the tables present in scientific articles is
challenging due to complicated tabular representations and complex embedded
text. This paper presents TabLeX, a large-scale benchmark dataset comprising
table images generated from scientific articles. TabLeX consists of two subsets,
one for table structure extraction and the other for table content extraction.
Each table image is accompanied by its corresponding LaTeX source code. To
facilitate the development of robust table IE tools, TabLeX contains images in
different aspect ratios and in a variety of fonts. Our analysis sheds light on
the shortcomings of current state-of-the-art table extraction models and shows
that they fail on even simple table images. Towards the end, we experiment with
a transformer-based existing baseline to report performance scores. In contrast
to the static benchmarks, we plan to augment this dataset with more complex and
diverse tables at regular intervals.

---

## Split, embed and merge: An accurate table structure recognizer

Zhenrong Zhang, Jianshu Zhang, Jun Du

Category: ocr
Keywords: Table structure recognition, Table recognition, Deep learning, Self-regression, Attention mechanism, Encoder-decoder
Year: 2021

Table structure recognition is an essential part for making machines understand
tables. Its main task is to recognize the internal structure of a table.
However, due to the complexity and diversity in their structure and style, it is
very difficult to parse the tabular data into the structured format which
machines can understand easily, especially for complex tables. In this paper, we
introduce Split, Embed and Merge (SEM), an accurate table structure recognizer.
Our model takes table images as input and can correctly recognize the structure
of tables, whether they are simple or complex tables. SEM is mainly composed of
three parts, splitter, embedder and merger. In the first stage, we apply the
splitter to predict the potential regions of the table row (column) separators,
and obtain the fine grid structure of the table. In the second stage, by taking
a full consideration of the textual information in the table, we fuse the output
features for each table grid from both vision and language modalities. Moreover,
we achieve a higher precision in our experiments through adding additional
semantic features. Finally, we process the merging of these basic table grids in
a self-regression manner. The correspondent merging results is learned through
the attention mechanism. In our experiments, SEM achieves an average F1-Measure
of 97.11% on the SciTSR dataset which outperforms other methods by a large
margin. We also won the first place in the complex table and third place in all
tables in ICDAR 2021 Competition on Scientific Literature Parsing, Task-B.
Extensive experiments on other publicly available datasets demonstrate that our
model achieves state-of-the-art.

---

## TabLeX: A Benchmark Dataset for Structure and Content Information Extraction from Scientific Tables

Harsh Desai, Pratik Kayal, Mayank Singh

Category: ocr
Keywords: Information Extraction, LaTeX, Scientific Articles
Year: 2021

Information Extraction (IE) from the tables present in scientific articles is
challenging due to complicated tabular representations and complex embedded
text. This paper presents TabLeX, a large-scale benchmark dataset comprising
table images generated from scientific articles. TabLeX consists of two subsets,
one for table structure extraction and the other for table content extraction.
Each table image is accompanied by its corresponding LaTeX source code. To
facilitate the development of robust table IE tools, TabLeX contains images in
different aspect ratios and in a variety of fonts. Our analysis sheds light on
the shortcomings of current state-of-the-art table extraction models and shows
that they fail on even simple table images. Towards the end, we experiment with
a transformer-based existing baseline to report performance scores. In contrast
to the static benchmarks, we plan to augment this dataset with more complex and
diverse tables at regular intervals.

---

## PingAn-VCGroup's Solution for ICDAR 2021 Competition on Scientific Literature Parsing Task B: Table Recognition to HTML

Jiaquan Ye, Xianbiao Qi, Yelin He, Yihao Chen, Dengyi Gu, Peng Gao, Rong Xiao

Category: ocr
Keywords: table recognition, scientific literature parsing, HTML conversion, ICDAR 2021, text recognition
Year: 2021

This paper presents our solution for ICDAR 2021 competition on scientific
literature parsing task B: table recognition to HTML. In our method, we divide
the table content recognition task into four sub-tasks: table structure
recognition, text line detection, text line recognition, and box assignment. Our
table structure recognition algorithm is customized based on MASTER, a robust
image text recognition algorithm. PSENet is used to detect each text line in the
table image. For text line recognition, our model is also built on MASTER.
Finally, in the box assignment phase, we associated the text boxes detected by
PSENet with the structure item reconstructed by table structure prediction, and
fill the recognized content of the text line into the corresponding item. Our
proposed method achieves a 96.84% TEDS score on 9,115 validation samples in the
development phase, and a 96.32% TEDS score on 9,064 samples in the final
evaluation phase.

---

## VSR: A Unified Framework for Document Layout Analysis combining Vision, Semantics and Relations

Peng Zhang, Can Li, Liang Qiao, Zhanzhan Cheng, Shiliang Pu, Yi Niu, Fei Wu

Category: ocr
Keywords: Vision, Semantics, Relations, Document layout analysis
Year: 2021

Document layout analysis is crucial for understanding document structures. On
this task, vision and semantics of documents, and relations between layout
components contribute to the understanding process. Though many works have been
proposed to exploit the above information, they show unsatisfactory results.
NLP-based methods model layout analysis as a sequence labeling task and show
insufficient capabilities in layout modeling. CV-based methods model layout
analysis as a detection or segmentation task, but bear limitations of
inefficient modality fusion and lack of relation modeling between layout
components. To address the above limitations, we propose a unified framework VSR
for document layout analysis, combining vision, semantics and relations. VSR
supports both NLP-based and CV-based methods. Specifically, we first introduce
vision through document image and semantics through text embedding maps. Then,
modality-specific visual and semantic features are extracted using a two-stream
network, which are adaptively fused to make full use of complementary
information. Finally, given component candidates, a relation module based on
graph neural network is incorporated to model relations between components and
output final results. On three popular benchmarks, VSR outperforms previous
models by large margins. Code will be released soon.

---

## Rethinking Text Line Recognition Models

Daniel Hernandez Diaz, Siyang Qin, Reeve Ingle, Yasuhisa Fujii, Alessandro Bissacco

Category: ocr
Keywords: text line recognition, OCR, universal architecture, Self-Attention, CTC decoder
Year: 2021

In this paper, we study the problem of text line recognition. Unlike most
approaches targeting specific domains such as scene-text or handwritten
documents, we investigate the general problem of developing a universal
architecture that can extract text from any image, regardless of source or input
modality. We consider two decoder families (Connectionist Temporal
Classification and Transformer) and three encoder modules (Bidirectional LSTMs,
Self-Attention, and GRCLs), and conduct extensive experiments to compare their
accuracy and performance on widely used public datasets of scene and handwritten
text. We find that a combination that so far has received little attention in
the literature, namely a Self-Attention encoder coupled with the CTC decoder,
when compounded with an external language model and trained on both public and
internal data, outperforms all the others in accuracy and computational
complexity. Unlike the more common Transformer-based models, this architecture
can handle inputs of arbitrary length, a requirement for universal line
recognition. Using an internal dataset collected from multiple sources, we also
expose the limitations of current public datasets in evaluating the accuracy of
line recognizers, as the relatively narrow image width and sequence length
distributions do not allow to observe the quality degradation of the Transformer
approach when applied to the transcription of long lines.

---

## Rethinking Text Line Recognition Models

Daniel Hernandez Diaz, Siyang Qin, Reeve Ingle, Yasuhisa Fujii, Alessandro Bissacco

Category: ocr
Keywords: text line recognition, universal architecture, Connectionist Temporal Classification, Transformer, Bidirectional LSTMs, Self-Attention, GRCLs, OCR
Year: 2021

In this paper, we study the problem of text line recognition. Unlike most
approaches targeting specific domains such as scene-text or handwritten
documents, we investigate the general problem of developing a universal
architecture that can extract text from any image, regardless of source or input
modality. We consider two decoder families (Connectionist Temporal
Classification and Transformer) and three encoder modules (Bidirectional LSTMs,
Self-Attention, and GRCLs), and conduct extensive experiments to compare their
accuracy and performance on widely used public datasets of scene and handwritten
text. We find that a combination that so far has received little attention in
the literature, namely a Self-Attention encoder coupled with the CTC decoder,
when compounded with an external language model and trained on both public and
internal data, outperforms all the others in accuracy and computational
complexity. Unlike the more common Transformer-based models, this architecture
can handle inputs of arbitrary length, a requirement for universal line
recognition. Using an internal dataset collected from multiple sources, we also
expose the limitations of current public datasets in evaluating the accuracy of
line recognizers, as the relatively narrow image width and sequence length
distributions do not allow to observe the quality degradation of the Transformer
approach when applied to the transcription of long lines.

---

## PubTables-1M: Towards Comprehensive Table Extraction from Unstructured Documents

Brandon Smock, Rohith Pesala, Robin Abraham

Category: ocr
Keywords: table extraction, machine learning, dataset, table structure, transformer models
Year: 2021

Recently, significant progress has been made applying machine learning to the
problem of table structure inference and extraction from unstructured documents.
However, one of the greatest challenges remains the creation of datasets with
complete, unambiguous ground truth at scale. To address this, we develop a new,
more comprehensive dataset for table extraction, called PubTables-1M.
PubTables-1M contains nearly one million tables from scientific articles,
supports multiple input modalities, and contains detailed header and location
information for table structures, making it useful for a wide variety of
modeling approaches. It also addresses a significant source of ground truth
inconsistency observed in prior datasets called oversegmentation, using a novel
canonicalization procedure. We demonstrate that these improvements lead to a
significant increase in training performance and a more reliable estimate of
model performance at evaluation for table structure recognition. Further, we
show that transformer-based object detection models trained on PubTables-1M
produce excellent results for all three tasks of detection, structure
recognition, and functional analysis without the need for any special
customization for these tasks. Data and code will be released at
https://github.com/microsoft/table-transformer.

---

## LayoutXLM: Multimodal Pre-training for Multilingual Visually-rich Document Understanding

Yiheng Xu, Tengchao Lv, Lei Cui, Guoxin Wang, Yijuan Lu, Dinei Florencio, Cha Zhang, Furu Wei

Category: ocr
Keywords: Multimodal Pre-training, Multilingual Document Understanding, Visually-rich Document Understanding, LayoutXLM, XFUND
Year: 2021

Multimodal pre-training with text, layout, and image has achieved SOTA
performance for visually-rich document understanding tasks recently,
demonstrating the potential for joint learning across different modalities. We
present LayoutXLM, a multimodal pre-trained model for multilingual document
understanding, which aims to bridge language barriers for visually-rich document
understanding. To evaluate LayoutXLM, we introduce a multilingual form
understanding benchmark dataset named XFUND, which includes form understanding
samples in 7 languages (Chinese, Japanese, Spanish, French, Italian, German,
Portuguese), with manually labeled key-value pairs for each language. Experiment
results show that the LayoutXLM model significantly outperformed existing SOTA
cross-lingual pre-trained models on the XFUND dataset. The pre-trained LayoutXLM
model and the XFUND dataset are publicly available at https://aka.ms/layoutxlm.

---

## Current Status and Performance Analysis of Table Recognition in Document Images with Deep Neural Networks

Khurram Azeem Hashmi, Marcus Liwicki, Didier Stricker, Muhammad Adnan Afzal, Muhammad Ahtsham Afzal, Muhammad Zeshan Afzal

Category: ocr
Keywords: Deep neural network, survey, document images, review paper, deep learning, performance evaluation, table recognition, table detection, table structure recognition, table analysis
Year: 2021

The first phase of table recognition is to detect the tabular area in a
document. Subsequently, the tabular structures are recognized in the second
phase in order to extract information from the respective cells. Table detection
and structural recognition are pivotal problems in the domain of table
understanding. However, table analysis is a perplexing task due to the colossal
amount of diversity and asymmetry in tables. Therefore, it is an active area of
research in document image analysis. Recent advances in the computing
capabilities of graphical processing units have enabled the deep neural networks
to outperform traditional state-of-the-art machine learning methods. Table
understanding has substantially benefited from the recent breakthroughs in deep
neural networks. However, there has not been a consolidated description of the
deep learning methods for table detection and table structure recognition. This
review paper provides a thorough analysis of the modern methodologies that
utilize deep neural networks. This work provided a thorough understanding of the
current state-of-the-art and related challenges of table understanding in
document images. Furthermore, the leading datasets and their intricacies have
been elaborated along with the quantitative results. Moreover, a brief overview
is given regarding the promising directions that can serve as a guide to further
improve table analysis in document images.

---

## PINGAN-VCGROUP’S SOLUTION FOR ICDAR 2021 COMPETITION ON SCIENTIFIC LITERATURE PARSING TASK B: TABLE RECOGNITION TO HTML

Jiaquan Ye, Xianbiao Qi, Yelin He, Yihao Chen, Dengyi Gu, Peng Gao, Rong Xiao

Category: ocr
Keywords: ICDAR 2021, table recognition, HTML, scientific literature parsing, image text recognition, MASTER, PSENet
Year: 2021

This paper presents our solution for ICDAR 2021 competition on scientific
literature parsing task B: table recognition to HTML. In our method, we divide
the table content recognition task into four sub-tasks: table structure
recognition, text line detection, text line recognition, and box assignment. Our
table structure recognition algorithm is customized based on MASTER, a robust
image text recognition algorithm. PSENet is used to detect each text line in the
table image. For text line recognition, our model is also built on MASTER.
Finally, in the box assignment phase, we associated the text boxes detected by
PSENet with the structure item reconstructed by table structure prediction, and
fill the recognized content of the text line into the corresponding item. Our
proposed method achieves a 96.84% TEDS score on 9,115 validation samples in the
development phase, and a 96.32% TEDS score on 9,064 samples in the final
evaluation phase.

---

## LayoutReader: Pre-training of Text and Layout for Reading Order Detection

Zilong Wang, Yiheng Xu, Lei Cui, Jingbo Shang, Furu Wei

Category: ocr
Keywords: reading order detection, visually-rich documents, deep learning, OCR, LayoutReader, ReadingBank
Year: 2021

Reading order detection is the cornerstone to understanding visually-rich
documents (e.g., receipts and forms). Unfortunately, no existing work took
advantage of advanced deep learning models because it is too laborious to
annotate a large enough dataset. We observe that the reading order of WORD
documents is embedded in their XML metadata; meanwhile, it is easy to convert
WORD documents to PDFs or images. Therefore, in an automated manner, we
construct ReadingBank, a benchmark dataset that contains reading order, text,
and layout information for 500,000 document images covering a wide spectrum of
document types. This first-ever large-scale dataset unleashes the power of deep
neural networks for reading order detection. Specifically, our proposed
LayoutReader captures the text and layout information for reading order
prediction using the seq2seq model. It performs almost perfectly in reading
order detection and significantly improves both open-source and commercial OCR
engines in ordering text lines in their results in our experiments. We will
release the dataset and model at https://aka.ms/layoutreader.

---

## Current Status and Performance Analysis of Table Recognition in Document Images with Deep Neural Networks

Khurram Azeem Hashmi, Marcus Liwicki, Didier Stricker, Muhammad Adnan Afzal, Muhammad Ahtsham Afzal, Muhammad Zeshan Afzal

Category: ocr
Keywords: Deep neural network, survey, document images, review paper, deep learning, performance evaluation, table recognition, table detection, table structure recognition, table analysis
Year: 2021

The first phase of table recognition is to detect the tabular area in a
document. Subsequently, the tabular structures are recognized in the second
phase in order to extract information from the respective cells. Table detection
and structural recognition are pivotal problems in the domain of table
understanding. However, table analysis is a perplexing task due to the colossal
amount of diversity and asymmetry in tables. Therefore, it is an active area of
research in document image analysis. Recent advances in the computing
capabilities of graphical processing units have enabled the deep neural networks
to outperform traditional state-of-the-art machine learning methods. Table
understanding has substantially benefited from the recent breakthroughs in deep
neural networks. However, there has not been a consolidated description of the
deep learning methods for table detection and table structure recognition. This
review paper provides a thorough analysis of the modern methodologies that
utilize deep neural networks. This work provided a thorough understanding of the
current state-of-the-art and related challenges of table understanding in
document images. Furthermore, the leading datasets and their intricacies have
been elaborated along with the quantitative results. Moreover, a brief overview
is given regarding the promising directions that can serve as a guide to further
improve table analysis in document images.

---

## LayoutReader: Pre-training of Text and Layout for Reading Order Detection

Zilong Wang, Yiheng Xu, Lei Cui, Jingbo Shang, Furu Wei

Category: ocr
Keywords: reading order detection, visually-rich documents, deep learning, OCR engines, dataset
Year: 2021

Reading order detection is the cornerstone to understanding visually-rich
documents (e.g., receipts and forms). Unfortunately, no existing work took
advantage of advanced deep learning models because it is too laborious to
annotate a large enough dataset. We observe that the reading order of WORD
documents is embedded in their XML metadata; meanwhile, it is easy to convert
WORD documents to PDFs or images. Therefore, in an automated manner, we
construct ReadingBank, a benchmark dataset that contains reading order, text,
and layout information for 500,000 document images covering a wide spectrum of
document types. This first-ever large-scale dataset unleashes the power of deep
neural networks for reading order detection. Specifically, our proposed
LayoutReader captures the text and layout information for reading order
prediction using the seq2seq model. It performs almost perfectly in reading
order detection and significantly improves both open-source and commercial OCR
engines in ordering text lines in their results in our experiments. We will
release the dataset and model at https://aka.ms/layoutreader.

---

## LayoutXLM: Multimodal Pre-training for Multilingual Visually-rich Document Understanding

Yiheng Xu, Tengchao Lv, Lei Cui, Guoxin Wang, Yijuan Lu, Dinei Florencio, Cha Zhang, Furu Wei

Category: ocr
Keywords: multimodal pre-training, multilingual document understanding, visually-rich documents, LayoutXLM, XFUND dataset
Year: 2021

Multimodal pre-training with text, layout, and image has achieved state-of-the-
art performance for visually-rich document understanding tasks recently,
demonstrating the great potential for joint learning across different
modalities. In this paper, we present LayoutXLM, a multimodal pre-trained model
for multilingual document understanding, aiming to bridge the language barriers
for visually-rich document understanding. To accurately evaluate LayoutXLM, we
introduce a multilingual form understanding benchmark dataset named XFUND, which
includes form understanding samples in 7 languages (Chinese, Japanese, Spanish,
French, Italian, German, Portuguese), with key-value pairs manually labeled for
each language. Experiment results show that the LayoutXLM model has
significantly outperformed the existing state-of-the-art cross-lingual pre-
trained models on the XFUND dataset. The pre-trained LayoutXLM model and the
XFUND dataset are publicly available at https://aka.ms/layoutxlm.

---

## TrOCR: Transformer-based Optical Character Recognition with Pre-trained Models

Minghao Li, Tengchao Lv, Lei Cui, Yijuan Lu, Dinei Florencio, Cha Zhang, Zhoujun Li, Furu Wei

Category: ocr
Keywords: Optical Character Recognition, Transformer, Pre-trained Models, Text Recognition, Document Digitalization
Year: 2021

Text recognition is a long-standing research problem for document
digitalization. Existing approaches for text recognition are usually built based
on CNN for image understanding and RNN for char-level text generation. In
addition, another language model is usually needed to improve the overall
accuracy as a post-processing step. In this paper, we propose an end-to-end text
recognition approach with pre-trained image Transformer and text Transformer
models, namely TrOCR, which leverages the Transformer architecture for both
image understanding and wordpiece-level text generation. The TrOCR model is
simple but effective, and can be pre-trained with large-scale synthetic data and
fine-tuned with human-labeled datasets. Experiments show that the TrOCR model
outperforms the current state-of-the-art models on both printed and handwritten
text recognition tasks.

---

## Graph-based Deep Generative Modelling for Document Layout Generation

Sanket Biswas, Pau Riba, Josep Lladós, Umapada Pal

Category: ocr
Keywords: Document Synthesis, Graph Neural Networks, Document Layout Generation
Year: 2021

One of the major prerequisites for any deep learning approach is the
availability of large-scale training data. When dealing with scanned document
images in real world scenarios, the principal information of its content is
stored in the layout itself. In this work, we have proposed an automated deep
generative model using Graph Neural Networks (GNNs) to generate synthetic data
with highly variable and plausible document layouts that can be used to train
document interpretation systems, in this case, specially in digital mailroom
applications. It is also the first graph-based approach for document layout
generation task experimented on administrative document images, in this case,
invoices.

---

## Split, embed and merge: An accurate table structure recognizer

Zhenrong Zhang, Jianshu Zhang, Jun Du

Category: ocr
Keywords: Table structure recognition, Table recognition, Deep learning, Self-regression, Attention mechanism, Encoder-decoder
Year: 2021

Table structure recognition is an essential part for making machines understand
tables. Its main task is to recognize the internal structure of a table.
However, due to the complexity and diversity in their structure and style, it is
very difficult to parse the tabular data into the structured format which
machines can understand easily, especially for complex tables. In this paper, we
introduce Split, Embed and Merge (SEM), an accurate table structure recognizer.
Our model takes table images as input and can correctly recognize the structure
of tables, whether they are simple or complex. SEM is mainly composed of three
parts, splitter, embedder and merger. In the first stage, we apply the splitter
to predict the potential regions of the table row (column) separators, and
obtain the fine grid structure of the table. In the second stage, by taking a
full consideration of the textual information in the table, we fuse the output
features for each table grid from both vision and language modalities. Moreover,
we achieve a higher precision in our experiments through adding additional
semantic features. Finally, we process the merging of these basic table grids in
a self-regression manner. The correspondent merging results are learned through
the attention mechanism. In our experiments, SEM achieves an average F1-Measure
of 97.11% on the SciTSR dataset which outperforms other methods by a large
margin. We also won the first place in the complex table and third place in all
tables in ICDAR 2021 Competition on Scientific Literature Parsing, Task-B.
Extensive experiments on other publicly available datasets demonstrate that our
model achieves state-of-the-art.

---

## LAYOUTLMV2: MULTI-MODAL PRE-TRAINING FOR VISUALLY-RICH DOCUMENT UNDERSTANDING

Yang Xu, Yiheng Xu, Tengchao Lv, Lei Cui, Furu Wei, Guoxin Wang, Yijuan Lu, Dinei Florencio, Cha Zhang, Wanxiang Che, Min Zhang, Lidong Zhou

Category: ocr
Keywords: LayoutLMv2, multi-modal pre-training, visually-rich document understanding, cross-modality interaction, spatial-aware self-attention
Year: 2021

Pre-training of text and layout has proved effective in a variety of visually-
rich document understanding tasks due to its effective model architecture and
the advantage of large-scale unlabeled scanned/digital-born documents. In this
paper, we present LayoutLMv2 by pre-training text, layout, and image in a multi-
modal framework, where new model architectures and pre-training tasks are
leveraged. Specifically, LayoutLMv2 not only uses the existing masked visual-
language modeling task but also the new text-image alignment and text-image
matching tasks in the pre-training stage, where cross-modality interaction is
better learned. Meanwhile, it also integrates a spatial-aware self-attention
mechanism into the Transformer architecture, so that the model can fully
understand the relative positional relationship among different text blocks.
Experiment results show that LayoutLMv2 outperforms strong baselines and
achieves new state-of-the-art results on a wide variety of downstream visually-
rich document understanding tasks, including FUNSD (0.7895 →0.8420), CORD
(0.9493 →0.9601), SROIE (0.9524 →0.9781), Kleister-NDA (0.834 →0.852), RVL-CDIP
(0.9443 →0.9564), and DocVQA (0.7295 →0.8672). The pre-trained LayoutLMv2 model
is publicly available at https://aka.ms/layoutlmv2.

---

## MC-OCR Challenge 2021: Simple approach for receipt information extraction and quality evaluation

Cuong Manh Nguyen, Vi Van Ngo, Dang Duy Nguyen

Category: ocr
Keywords: Object Detection, OCR, Image Quality Assessment, Key Information Extraction
Year: 2021

This challenge organized at the RIVF conference 2021, with two tasks including
(1) image quality assessment (IQA) of the captured receipt, and (2) key
information extraction (KIE) of required fields, our team came up with a
solution based on extracting image patches for task 1 and Yolov5 + VietOCR for
task 2. Our solution achieved 0.149 of the RMSE score for task 1 (rank 7) and
0.219 of the CER score for task 2 (rank 1).

---

## LayoutReader: Pre-training of Text and Layout for Reading Order Detection

Zilong Wang, Yiheng Xu, Lei Cui, Jingbo Shang, Furu Wei

Category: ocr
Keywords: Reading order detection, Visually-rich documents, Deep learning, Dataset, LayoutReader, Sequence to sequence, OCR
Year: 2021

Reading order detection is the cornerstone to understanding visually-rich
documents (e.g., receipts and forms). Unfortunately, no existing work took
advantage of advanced deep learning models because it is too laborious to
annotate a large enough dataset. We observe that the reading order of WORD
documents is embedded in their XML metadata; meanwhile, it is easy to convert
WORD documents to PDFs or images. Therefore, in an automated manner, we
construct ReadingBank, a benchmark dataset that contains reading order, text,
and layout information for 500,000 document images covering a wide spectrum of
document types. This first-ever large-scale dataset unleashes the power of deep
neural networks for reading order detection. Specifically, our proposed
LayoutReader captures the text and layout information for reading order
prediction using the seq2seq model. It performs almost perfectly in reading
order detection and significantly improves both open-source and commercial OCR
engines in ordering text lines in their results in our experiments. We will
release the dataset and model at https://aka.ms/layoutreader.

---

## LayoutReader: Pre-training of Text and Layout for Reading Order Detection

Zilong Wang, Yiheng Xu, Lei Cui, Jingbo Shang, Furu Wei

Category: ocr
Keywords: reading order detection, visually-rich documents, deep learning, OCR, datasets
Year: 2021

Reading order detection is the cornerstone to understanding visually-rich
documents (e.g., receipts and forms). Unfortunately, no existing work took
advantage of advanced deep learning models because it is too laborious to
annotate a large enough dataset. We observe that the reading order of WORD
documents is embedded in their XML metadata; meanwhile, it is easy to convert
WORD documents to PDFs or images. Therefore, in an automated manner, we
construct ReadingBank, a benchmark dataset that contains reading order, text,
and layout information for 500,000 document images covering a wide spectrum of
document types. This first-ever large-scale dataset unleashes the power of deep
neural networks for reading order detection. Specifically, our proposed
LayoutReader captures the text and layout information for reading order
prediction using the seq2seq model. It performs almost perfectly in reading
order detection and significantly improves both open-source and commercial OCR
engines in ordering text lines in their results in our experiments. We will
release the dataset and model at https://aka.ms/layoutreader.

---

## Probabilistic homogeneity for document image segmentation

Tan Lu, Ann Dooms

Category: ocr
Keywords: Probabilistic local text homogeneity, Random walk-and-check simulation, Bayesian cue integration, Text homogeneity pattern, Document image segmentation
Year: 2021

In this paper we propose a novel probabilistic framework for document
segmentation exploiting human perceptual recognition of text regions from
complicated layouts. In particular, we conceptualize text homogeneity as the
Gestalt pattern displayed in text regions, characterized by proximately and
symmetrically arranged units with similar morphological and texture features. We
model this pattern in the local region of a connected component (CC) using a
hierarchical formulation, which simulates a random walk-and-check on a graph
encoding the neighborhood of the CC. The proposed formulation allows an
effective computation of what we call the probabilistic local text homogeneity
(PLTH) using a weighted summation of the weights of the graph, which are derived
from a probabilistic description of the homogeneity between neighboring CCs and
computed through Bayesian cue integration. The proposed PLTH enables a multi-
aspect analysis, where various primitives such as geometrical configuration,
morphological features, texture characterization, and location priors are
integrated into one computational probabilistic model. This enables an effective
text and non-text classification of CCs preceding any grouping process, which is
currently absent in document segmentation. Experimental results show that our
segmentation method based on the proposed PLTH model improves upon the state-of-
the-art.

---

## LayoutXLM: Multimodal Pre-training for Multilingual Visually-rich Document Understanding

Yiheng Xu, Tengchao Lv, Lei Cui, Guoxin Wang, Yijuan Lu, Dinei Florencio, Cha Zhang, Furu Wei

Category: ocr
Keywords: multimodal pre-training, visually-rich document understanding, multilingual document understanding, LayoutXLM, XFUND dataset
Year: 2021

Multimodal pre-training with text, layout, and image has achieved SOTA
performance for visually-rich document understanding tasks recently, which
demonstrates the great potential for joint learning across different modalities.
In this paper, we present LayoutXLM, a multimodal pre-trained model for
multilingual document understanding, which aims to bridge the language barriers
for visually-rich document understanding. To accurately evaluate LayoutXLM, we
also introduce a multilingual form understanding benchmark dataset named XFUND,
which includes form understanding samples in 7 languages (Chinese, Japanese,
Spanish, French, Italian, German, Portuguese), and key-value pairs are manually
labeled for each language. Experiment results show that the LayoutXLM model has
significantly outperformed the existing SOTA cross-lingual pre-trained models on
the XFUND dataset. The pre-trained LayoutXLM model and the XFUND dataset are
publicly available at https://aka.ms/layoutxlm.

---

## Donut: Document Understanding Transformer without OCR

Geewook Kim, Teakgyu Hong, Moonbin Yim, Jinyoung Park, Jinyeong Yim, Wonseok Hwang, Sangdoo Yun, Dongyoon Han, Seunghyun Park

Category: ocr
Keywords: document understanding, deep learning, OCR, synthetic data generation, end-to-end training
Year: 2021

Understanding document images (e.g., invoices) has been an important research
topic and has many applications in document processing automation. Through the
latest advances in deep learning-based Optical Character Recognition (OCR),
current Visual Document Understanding (VDU) systems have come to be designed
based on OCR. Although such OCR-based approaches promise reasonable performance,
they suffer from critical problems induced by the OCR, e.g., (1) expensive
computational costs and (2) performance degradation due to the OCR error
propagation. In this paper, we propose a novel VDU model that is end-to-end
trainable without underpinning OCR framework. To this end, we propose a new task
and a synthetic document image generator to pre-train the model to mitigate the
dependencies on large-scale real document images. Our approach achieves state-
of-the-art performance on various document understanding tasks in public
benchmark datasets and private industrial service datasets. Through extensive
experiments and analysis, we demonstrate the effectiveness of the proposed model
especially with consideration for a real-world application.

---

## TrOCR: Transformer-based Optical Character Recognition with Pre-trained Models

Minghao Li, Tengchao Lv, Lei Cui, Yijuan Lu, Dinei Florencio, Cha Zhang, Zhoujun Li, Furu Wei

Category: ocr
Keywords: Optical Character Recognition, Transformer, Pre-trained Models, Text Recognition, Document Digitalization
Year: 2021

Text recognition is a long-standing research problem for document
digitalization. Existing approaches for text recognition are usually built based
on CNN for image understanding and RNN for char-level text generation. In
addition, another language model is usually needed to improve the overall
accuracy as a post-processing step. In this paper, we propose an end-to-end text
recognition approach with pre-trained image Transformer and text Transformer
models, namely TrOCR, which leverages the Transformer architecture for both
image understanding and wordpiece-level text generation. The TrOCR model is
simple but effective, and can be pre-trained with large-scale synthetic data and
fine-tuned with human-labeled datasets. Experiments show that the TrOCR model
outperforms the current state-of-the-art models on both printed and handwritten
text recognition tasks.

---

## LayoutReader: Pre-training of Text and Layout for Reading Order Detection

Zilong Wang, Yiheng Xu, Lei Cui, Jingbo Shang, Furu Wei

Category: ocr
Keywords: Reading order detection, Visually-rich documents, Deep neural networks, LayoutReader, ReadingBank, OCR engines
Year: 2021

Reading order detection is the cornerstone to understanding visually-rich
documents (e.g., receipts and forms). Unfortunately, no existing work took
advantage of advanced deep learning models because it is too laborious to
annotate a large enough dataset. We observe that the reading order of WORD
documents is embedded in their XML metadata; meanwhile, it is easy to convert
WORD documents to PDFs or images. Therefore, in an automated manner, we
construct ReadingBank, a benchmark dataset that contains reading order, text,
and layout information for 500,000 document images covering a wide spectrum of
document types. This first-ever large-scale dataset unleashes the power of deep
neural networks for reading order detection. Specifically, our proposed
LayoutReader captures the text and layout information for reading order
prediction using the seq2seq model. It performs almost perfectly in reading
order detection and significantly improves both open-source and commercial OCR
engines in ordering text lines in their results in our experiments. We will
release the dataset and model at https://aka.ms/layoutreader.

---

## Text Content Based Layout Analysis

José Ramón Prieto, Vicente Bosch, Enrique Vidal, Dominique Stutzmann, Sébastien Hamel

Category: ocr
Keywords: Document Layout Analysis, Text Content Based Features, Hidden Markov Models, Deep Neural Networks
Year: 2020

State-of-the-art Document Layout Analysis methods rely on graphical appearance
features to detect and classify different layout regions in a scanned text
image. However, in many cases, only relying on graphical information is
problematic or impossible. Reading some text in the problematic regions’
boundaries is necessary to reliably detect and separate these regions. Textual
content-based features are required, but since transcription is usually
performed after layout analysis, a vicious circle arises. We circumvent this
deadlock using the concept of a Probabilistic Index Map. We use the word
relevance probabilities it provides to calculate relevant text content-based
features at the pixel level. We assess the impact of these new features on a
historical document complex paragraph classification task using both a classical
Hidden Markov Model approach and Deep Neural Networks. The results are
encouraging, showcasing the positive impact text content-based features can have
on the Document Layout Analysis research field.

---

## LayoutLM: Pre-training of Text and Layout for Document Image Understanding

Yiheng Xu, Minghao Li, Lei Cui, Shaohan Huang, Furu Wei, Ming Zhou

Category: ocr
Keywords: LayoutLM, pre-trained models, document image understanding, information extraction, transfer learning
Year: 2020

Pre-training techniques have been verified successfully in a variety of NLP
tasks in recent years. Despite the widespread use of pre-training models for NLP
applications, they almost exclusively focus on text-level manipulation, while
neglecting layout and style information that is vital for document image
understanding. In this paper, we propose the LayoutLM to jointly model
interactions between text and layout information across scanned document images,
which is beneficial for a great number of real-world document image
understanding tasks such as information extraction from scanned documents.
Furthermore, we also leverage image features to incorporate words’ visual
information into LayoutLM. To the best of our knowledge, this is the first time
that text and layout are jointly learned in a single framework for document-
level pre-training. It achieves new state-of-the-art results in several
downstream tasks, including form understanding (from 70.72 to 79.27), receipt
understanding (from 94.02 to 95.24) and document image classification (from
93.07 to 94.42). The code and pre-trained LayoutLM models are publicly available
at https://aka.ms/layoutlm.

---

## Image-based table recognition: data, model, and evaluation

Xu Zhong, Elaheh ShafieiBavani, Antonio Jimeno Yepes

Category: ocr
Keywords: table recognition, deep learning, dataset, image-based recognition, PubTabNet, encoder-decoder architecture, Tree-Edit-Distance-based Similarity
Year: 2020

Important information that relates to a specific topic in a document is often
organized in tabular format to assist readers with information retrieval and
comparison, which may be difficult to provide in natural language. However,
tabular data in unstructured digital documents, e.g. Portable Document Format
(PDF) and images, are difficult to parse into structured machine-readable
format, due to complexity and diversity in their structure and style. To
facilitate image-based table recognition with deep learning, we develop and
release the largest publicly available table recognition dataset PubTabNet,
containing 568k table images with corresponding structured HTML representation.
PubTabNet is automatically generated by matching the XML and PDF representations
of the scientific articles in PubMed Central Open Access Subset (PMCOA). We also
propose a novel attention-based encoder-dual-decoder (EDD) architecture that
converts images of tables into HTML code. The model has a structure decoder
which reconstructs the table structure and helps the cell decoder to recognize
cell content. In addition, we propose a new Tree-Edit-Distance-based Similarity
(TEDS) metric for table recognition, which more appropriately captures multi-hop
cell misalignment and OCR errors than the pre-established metric. The
experiments demonstrate that the EDD model can accurately recognize complex
tables solely relying on the image representation, outperforming the state-of-
the-art by 9.7% absolute TEDS score.

---

## LayoutLM: Pre-training of Text and Layout for Document Image Understanding

Yiheng Xu, Minghao Li, Lei Cui, Shaohan Huang, Furu Wei, Ming Zhou

Category: ocr
Keywords: LayoutLM, pre-trained models, document image understanding
Year: 2020

Pre-training techniques have been verified successfully in a variety of NLP
tasks in recent years. Despite the widespread use of pre-training models for NLP
applications, they almost exclusively focus on text-level manipulation, while
neglecting layout and style information that is vital for document image
understanding. In this paper, we propose the LayoutLM to jointly model
interactions between text and layout information across scanned document images,
which is beneficial for a great number of real-world document image
understanding tasks such as information extraction from scanned documents.
Furthermore, we also leverage image features to incorporate words’ visual
information into LayoutLM. To the best of our knowledge, this is the first time
that text and layout are jointly learned in a single framework for document-
level pre-training. It achieves new state-of-the-art results in several
downstream tasks, including form understanding (from 70.72 to 79.27), receipt
understanding (from 94.02 to 95.24), and document image classification (from
93.07 to 94.42). The code and pre-trained LayoutLM models are publicly available
at https://aka.ms/layoutlm.

---

## LayoutLM: Pre-training of Text and Layout for Document Image Understanding

Yiheng Xu, Minghao Li, Lei Cui, Shaohan Huang, Furu Wei, Ming Zhou

Category: ocr
Keywords: LayoutLM, pre-trained models, document image understanding
Year: 2020

Pre-training techniques have been verified successfully in a variety of NLP
tasks in recent years. Despite the widespread use of pre-training models for NLP
applications, they almost exclusively focus on text-level manipulation, while
neglecting layout and style information that is vital for document image
understanding. In this paper, we propose the LayoutLM to jointly model
interactions between text and layout information across scanned document images,
which is beneficial for a great number of real-world document image
understanding tasks such as information extraction from scanned documents.
Furthermore, we also leverage image features to incorporate words’ visual
information into LayoutLM. To the best of our knowledge, this is the first time
that text and layout are jointly learned in a single framework for document-
level pre-training. It achieves new state-of-the-art results in several
downstream tasks, including form understanding (from 70.72 to 79.27), receipt
understanding (from 94.02 to 95.24) and document image classification (from
93.07 to 94.42). The code and pre-trained LayoutLM models are publicly available
at https://aka.ms/layoutlm.

---

## LayoutLM: Pre-training of Text and Layout for Document Image Understanding

Yiheng Xu, Minghao Li, Lei Cui, Shaohan Huang, Furu Wei, Ming Zhou

Category: ocr
Keywords: LayoutLM, pre-trained models, document image understanding
Year: 2020

Pre-training techniques have been verified successfully in a variety of NLP
tasks in recent years. Despite the widespread use of pre-training models for NLP
applications, they almost exclusively focus on text-level manipulation, while
neglecting layout and style information that is vital for document image
understanding. In this paper, we propose the LayoutLM to jointly model
interactions between text and layout information across scanned document images,
which is beneficial for a great number of real-world document image
understanding tasks such as information extraction from scanned documents.
Furthermore, we also leverage image features to incorporate words’ visual
information into LayoutLM. To the best of our knowledge, this is the first time
that text and layout are jointly learned in a single framework for document-
level pre-training. It achieves new state-of-the-art results in several
downstream tasks, including form understanding (from 70.72 to 79.27), receipt
understanding (from 94.02 to 95.24) and document image classification (from
93.07 to 94.42). The code and pre-trained LayoutLM models are publicly available
at https://aka.ms/layoutlm.

---

## GFTE: Graph-based Financial Table Extraction

Yiren Li, Zheng Huang, Junchi Yan, Yi Zhou, Fan Ye, Xianhui Liu

Category: ocr
Keywords: financial table extraction, graph-based convolutional neural network, FinTab dataset, table structure recognition, deep learning
Year: 2020

Tabular data is a crucial form of information expression, which can organize
data in a standard structure for easy information retrieval and comparison.
However, in the financial industry and many other fields, tables are often
disclosed in unstructured digital files, e.g., Portable Document Format (PDF)
and images, which are difficult to be extracted directly. In this paper, to
facilitate deep learning-based table extraction from unstructured digital files,
we publish a standard Chinese dataset named FinTab, which contains more than
1,600 financial tables of diverse kinds and their corresponding structure
representation in JSON. In addition, we propose a novel graph-based
convolutional neural network model named GFTE as a baseline for future
comparison. GFTE integrates image features, position features, and textual
features together for precise edge prediction and reaches overall good results.

---

## Table Extraction and Understanding for Scientific and Enterprise Applications

Yannis Katsis, Doug Burdick, Nancy Wang, Alexandre V Evfimievski, Marina Danilevsky

Category: ocr
Keywords: table extraction, data understanding, scientific applications, enterprise applications, document processing
Year: 2020

This document discusses the challenges and methodologies involved in extracting
and understanding table data from scientific and enterprise documents. Emphasis
is placed on both the technical and practical aspects of table extraction,
including the handling of complex table structures and the integration of
extracted data into broader data processing workflows.

---

## GFTE: Graph-based Financial Table Extraction

Yiren Li, Zheng Huang, Junchi Yan, Yi Zhou, Fan Ye, Xianhui Liu

Category: ocr
Keywords: table extraction, financial data, graph convolutional network, deep learning, dataset
Year: 2020

Tabular data is a crucial form of information expression, which can organize
data in a standard structure for easy information retrieval and comparison.
However, in the financial industry and many other fields, tables are often
disclosed in unstructured digital files, e.g., Portable Document Format (PDF)
and images, which are difficult to be extracted directly. In this paper, to
facilitate deep learning-based table extraction from unstructured digital files,
we publish a standard Chinese dataset named FinTab, which contains more than
1,600 financial tables of diverse kinds and their corresponding structure
representation in JSON. In addition, we propose a novel graph-based
convolutional neural network model named GFTE as a baseline for future
comparison. GFTE integrates image feature, position feature, and textual feature
together for precise edge prediction and reaches overall good results.

---

## Image-based table recognition: data, model, and evaluation

Xu Zhong, Elaheh ShafieiBavani, Antonio Jimeno Yepes

Category: ocr
Keywords: table recognition, deep learning, dataset, PubTabNet, attention-based model, encoder-dual-decoder, Tree-Edit-Distance-based Similarity
Year: 2020

Important information that relates to a specific topic in a document is often
organized in tabular format to assist readers with information retrieval and
comparison, which may be difficult to provide in natural language. However,
tabular data in unstructured digital documents, e.g., Portable Document Format
(PDF) and images, are difficult to parse into structured machine-readable
format, due to complexity and diversity in their structure and style. To
facilitate image-based table recognition with deep learning, we develop and
release the largest publicly available table recognition dataset PubTabNet,
containing 568k table images with corresponding structured HTML representation.
PubTabNet is automatically generated by matching the XML and PDF representations
of the scientific articles in PubMed Central Open Access Subset (PMCOA). We also
propose a novel attention-based encoder-dual-decoder (EDD) architecture that
converts images of tables into HTML code. The model has a structure decoder
which reconstructs the table structure and helps the cell decoder to recognize
cell content. In addition, we propose a new Tree-Edit-Distance-based Similarity
(TEDS) metric for table recognition, which more appropriately captures multi-hop
cell misalignment and OCR errors than the pre-established metric. The
experiments demonstrate that the EDD model can accurately recognize complex
tables solely relying on the image representation, outperforming the state-of-
the-art by 9.7% absolute TEDS score.

---

## Image-based table recognition: data, model, and evaluation

Xu Zhong, Elaheh ShafieiBavani, Antonio Jimeno Yepes

Category: ocr
Keywords: table recognition, dual decoder, dataset, evaluation
Year: 2020

Important information that relates to a specific topic in a document is often
organized in tabular format to assist readers with information retrieval and
comparison, which may be difficult to provide in natural language. However,
tabular data in unstructured digital documents, e.g., Portable Document Format
(PDF) and images, are difficult to parse into structured machine-readable
format, due to complexity and diversity in their structure and style. To
facilitate image-based table recognition with deep learning, we develop and
release the largest publicly available table recognition dataset PubTabNet,
containing 568k table images with corresponding structured HTML representation.
PubTabNet is automatically generated by matching the XML and PDF representations
of the scientific articles in PubMed Central Open Access Subset (PMCOA). We also
propose a novel attention-based encoder-dual-decoder (EDD) architecture that
converts images of tables into HTML code. The model has a structure decoder
which reconstructs the table structure and helps the cell decoder to recognize
cell content. In addition, we propose a new Tree-Edit-Distance-based Similarity
(TEDS) metric for table recognition, which more appropriately captures multi-hop
cell misalignment and OCR errors than the pre-established metric. The
experiments demonstrate that the EDD model can accurately recognize complex
tables solely relying on the image representation, outperforming the state-of-
the-art by 9.7% absolute TEDS score.

---

## Global Table Extractor (GTE): A Framework for Joint Table Identification and Cell Structure Recognition Using Visual Context

Xinyi Zheng, Douglas Burdick, Lucian Popa, Xu Zhong, Nancy Xin Ru Wang

Category: ocr
Keywords: Table Identification, Cell Structure Recognition, Deep Learning, Document Analysis, Vision-based Framework
Year: 2020

Documents are often used for knowledge sharing and preservation in business and
science, within which are tables that capture most of the critical data.
Unfortunately, most documents are stored and distributed as PDF or scanned
images, which fail to preserve logical table structure. Recent vision-based deep
learning approaches have been proposed to address this gap, but most still
cannot achieve state-of-the-art results. We present Global Table Extractor
(GTE), a vision-guided systematic framework for joint table detection and cell
structured recognition, which could be built on top of any object detection
model. With GTE-Table, we invent a new penalty based on the natural cell
containment constraint of tables to train our table network aided by cell
location predictions. GTE-Cell is a new hierarchical cell detection network that
leverages table styles. Further, we design a method to automatically label table
and cell structure in existing documents to cheaply create a large corpus of
training and test data. We use this to enhance PubTabNet with cell labels and
create FinTabNet, real-world and complex scientific and financial datasets with
detailed table structure annotations to help train and test structure
recognition. Our framework surpasses previous state-of-the-art results on the
ICDAR 2013 and ICDAR 2019 table competition in both table detection and cell
structure recognition. Further experiments demonstrate a greater than 45%
improvement in cell structure recognition when compared to a vanilla RetinaNet
object detection model in our new out-of-domain FinTabNet.

---

## Synthetic vs. Real Reference Strings for Citation Parsing, and the Importance of Re-training and Out-Of-Sample Data for Meaningful Evaluations: Experiments with GROBID, GIANT and Cora

Mark Grennan, Joeran Beel

Category: ocr
Keywords: Reference Parsing, Information Extraction, Citation Analysis
Year: 2020

Citation parsing, particularly with deep neural networks, suffers from a lack of
training data as available datasets typically contain only a few thousand
training instances. Manually labeling citation strings is very time-consuming,
hence synthetically created training data could be a solution. However, as of
now, it is unknown if synthetically created reference-strings are suitable to
train machine learning algorithms for citation parsing. To find out, we train
Grobid, which uses Conditional Random Fields, with a) human-labeled reference
strings from ‘real’ bibliographies and b) synthetically created reference
strings from the GIANT dataset. We find that both synthetic and organic
reference strings are equally suited for training Grobid (F1 = 0.74). We
additionally find that retraining Grobid has a notable impact on its
performance, for both synthetic and real data (+30% in F1). Having as many types
of labeled fields as possible during training also improves effectiveness, even
if these fields are not available in the evaluation data (+13.5% F1). We
conclude that synthetic data is suitable for training (deep) citation parsing
models. We further suggest that in future evaluations of reference parsers both
evaluation data similar and dissimilar to the training data should be used for
more meaningful evaluations.

---

## TableBank: A Benchmark Dataset for Table Detection and Recognition

Minghao Li, Lei Cui, Shaohan Huang, Furu Wei, Ming Zhou, Zhoujun Li

Category: ocr
Keywords: TableBank, table detection and recognition, weak supervision, image-based deep learning network
Year: 2020

We present TableBank, a new image-based table detection and recognition dataset
built with novel weak supervision from Word and LaTeX documents on the internet.
Existing research for image-based table detection and recognition usually fine-
tunes pre-trained models on out-of-domain data with a few thousand human-labeled
examples, which is difficult to generalize on real-world applications. With
TableBank that contains 417K high-quality labeled tables, we build several
strong baselines using state-of-the-art models with deep neural networks. We
make TableBank publicly available and hope it will empower more deep learning
approaches in the table detection and recognition task. The dataset and models
are available at https://github.com/doc-analysis/TableBank.

---

## CDeC-Net: Composite Deformable Cascade Network for Table Detection in Document Images

Madhav Agarwal, Ajoy Mondal, C. V. Jawahar

Category: ocr
Keywords: Page object, table detection, Cascade Mask R-CNN, deformable convolution, single model
Year: 2020

Localizing page elements/objects such as tables, figures, equations, etc. is the
primary step in extracting information from document images. We propose a novel
end-to-end trainable deep network, (CDeC-Net) for detecting tables present in
the documents. The proposed network consists of a multistage extension of Mask
R-CNN with a dual backbone having deformable convolution for detecting tables
varying in scale with high detection accuracy at higher IoU threshold. We
empirically evaluate CDeC-Net on all the publicly available benchmark datasets —
ICDAR-2013, ICDAR-2017, ICDAR-2019, UNLV, Marmot, PubLayNet, and TableBank —
with extensive experiments. Our solution has three important properties: (i) a
single trained model CDeC-Net‡ performs well across all the popular benchmark
datasets; (ii) we report excellent performances across multiple, including
higher, thresholds of IoU; (iii) by following the same protocol of the recent
papers for each of the benchmarks, we consistently demonstrate the superior
quantitative performance. Our code and models will be publicly released for
enabling the reproducibility of the results.

---

## Image-based table recognition: data, model, and evaluation

Xu Zhong, Elaheh ShafieiBavani, Antonio Jimeno Yepes

Category: ocr
Keywords: table recognition, dual decoder, dataset, evaluation
Year: 2020

Important information that relates to a specific topic in a document is often
organized in tabular format to assist readers with information retrieval and
comparison, which may be difficult to provide in natural language. However,
tabular data in unstructured digital documents, e.g. Portable Document Format
(PDF) and images, are difficult to parse into structured machine-readable
format, due to complexity and diversity in their structure and style. To
facilitate image-based table recognition with deep learning, we develop and
release the largest publicly available table recognition dataset PubTabNet,
containing 568k table images with corresponding structured HTML representation.
PubTabNet is automatically generated by matching the XML and PDF representations
of the scientific articles in PubMed Central Open Access Subset (PMCOA). We also
propose a novel attention-based encoder-dual-decoder (EDD) architecture that
converts images of tables into HTML code. The model has a structure decoder
which reconstructs the table structure and helps the cell decoder to recognize
cell content. In addition, we propose a new Tree-Edit-Distance-based Similarity
(TEDS) metric for table recognition, which more appropriately captures multi-hop
cell misalignment and OCR errors than the pre-established metric. The
experiments demonstrate that the EDD model can accurately recognize complex
tables solely relying on the image representation, outperforming the state-of-
the-art by 9.7% absolute TEDS score.

---

## Learn to Augment: Joint Data Augmentation and Network Optimization for Text Recognition

Canjie Luo, Yuanzhi Zhu, Lianwen Jin, Yongpan Wang

Category: ocr
Keywords: text recognition, data augmentation, handwritten text, scene text, network optimization
Year: 2020

Handwritten text and scene text suffer from various shapes and distorted
patterns. Thus training a robust recognition model requires a large amount of
data to cover diversity as much as possible. In contrast to data collection and
annotation, data augmentation is a low-cost way. In this paper, we propose a new
method for text image augmentation. Different from traditional augmentation
methods such as rotation, scaling, and perspective transformation, our proposed
augmentation method is designed to learn proper and efficient data augmentation
which is more effective and specific for training a robust recognizer. By using
a set of custom fiducial points, the proposed augmentation method is flexible
and controllable. Furthermore, we bridge the gap between the isolated processes
of data augmentation and network optimization by joint learning. An agent
network learns from the output of the recognition network and controls the
fiducial points to generate more proper training samples for the recognition
network. Extensive experiments on various benchmarks, including regular scene
text, irregular scene text, and handwritten text, show that the proposed
augmentation and the joint learning methods significantly boost the performance
of the recognition networks. A general toolkit for geometric augmentation is
available.

---

## Kleister: A Novel Task for Information Extraction Involving Long Documents with Complex Layout

Filip Graliński, Tomasz Stanisławek, Anna Wróblewska, Dawid Lipiński, Agnieszka Kaliska, Paulina Rosalska, Bartosz Topolski, Przemysław Biecek

Category: ocr
Keywords: Information Extraction, Complex Layout, Long Documents, NLP, Named Entity Recognition
Year: 2020

State-of-the-art solutions for Natural Language Processing (NLP) are able to
capture a broad range of contexts, like the sentence-level context or document-
level context for short documents. But these solutions are still struggling when
it comes to longer, real-world documents with the information encoded in the
spatial structure of the document, such as page elements like tables, forms,
headers, openings or footers; complex page layout or presence of multiple pages.
To encourage progress on deeper and more complex Information Extraction (IE) we
introduce a new task (named Kleister) with two new datasets. Utilizing both
textual and structural layout features, an NLP system must find the most
important information, about various types of entities, in long formal
documents. We propose a Pipeline method as a text-only baseline with different
Named Entity Recognition architectures (Flair, BERT, RoBERTa). Moreover, we
checked the most popular PDF processing tools for text extraction (pdf2djvu,
Tesseract and Textract) in order to analyze the behavior of the IE system in the
presence of errors introduced by these tools.

---

## Image-based table recognition: data, model, and evaluation

Xu Zhong, Elaheh ShafieiBavani, Antonio Jimeno Yepes

Category: ocr
Keywords: table recognition, deep learning, dataset, encoder-decoder architecture, tree-edit-distance-based similarity, PubTabNet
Year: 2020

Important information that relates to a specific topic in a document is often
organized in tabular format to assist readers with information retrieval and
comparison, which may be difficult to provide in natural language. However,
tabular data in unstructured digital documents, e.g. Portable Document Format
(PDF) and images, are difficult to parse into structured machine-readable
format, due to complexity and diversity in their structure and style. To
facilitate image-based table recognition with deep learning, we develop and
release the largest publicly available table recognition dataset PubTabNet,
containing 568k table images with corresponding structured HTML representation.
PubTabNet is automatically generated by matching the XML and PDF representations
of the scientific articles in PubMed Central Open Access Subset (PMCOA). We also
propose a novel attention-based encoder-dual-decoder (EDD) architecture that
converts images of tables into HTML code. The model has a structure decoder
which reconstructs the table structure and helps the cell decoder to recognize
cell content. In addition, we propose a new Tree-Edit-Distance-based Similarity
(TEDS) metric for table recognition, which more appropriately captures multi-hop
cell misalignment and OCR errors than the pre-established metric. The
experiments demonstrate that the EDD model can accurately recognize complex
tables solely relying on the image representation, outperforming the state-of-
the-art by 9.7% absolute TEDS score.

---

## GFTE: Graph-based Financial Table Extraction

Yiren Li, Zheng Huang, Junchi Yan, Yi Zhou, Fan Ye, Xianhui Liu

Category: ocr
Keywords: table extraction, OCR, graph convolutional network, financial tables, dataset
Year: 2020

Tabular data is a crucial form of information expression, which can organize
data in a standard structure for easy information retrieval and comparison.
However, in the financial industry and many other fields, tables are often
disclosed in unstructured digital files, e.g., Portable Document Format (PDF)
and images, which are difficult to be extracted directly. In this paper, to
facilitate deep learning-based table extraction from unstructured digital files,
we publish a standard Chinese dataset named FinTab, which contains more than
1,600 financial tables of diverse kinds and their corresponding structure
representation in JSON. In addition, we propose a novel graph-based
convolutional neural network model named GFTE as a baseline for future
comparison. GFTE integrates image feature, position feature, and textual feature
together for precise edge prediction and reaches overall good results.

---

## LayoutLM: Pre-training of Text and Layout for Document Image Understanding

Yiheng Xu, Minghao Li, Lei Cui, Shaohan Huang, Furu Wei, Ming Zhou

Category: ocr
Keywords: LayoutLM, pre-trained models, document image understanding
Year: 2020

Pre-training techniques have been verified successfully in a variety of NLP
tasks in recent years. Despite the widespread use of pre-training models for NLP
applications, they almost exclusively focus on text-level manipulation, while
neglecting layout and style information that is vital for document image
understanding. In this paper, we propose the LayoutLM to jointly model
interactions between text and layout information across scanned document images,
which is beneficial for a great number of real-world document image
understanding tasks such as information extraction from scanned documents.
Furthermore, we also leverage image features to incorporate words’ visual
information into LayoutLM. To the best of our knowledge, this is the first time
that text and layout are jointly learned in a single framework for document-
level pre-training. It achieves new state-of-the-art results in several
downstream tasks, including form understanding (from 70.72 to 79.27), receipt
understanding (from 94.02 to 95.24) and document image classification (from
93.07 to 94.42). The code and pre-trained LayoutLM models are publicly available
at https://aka.ms/layoutlm.

---

## DocBank: A Benchmark Dataset for Document Layout Analysis

Minghao Li, Yiheng Xu, Lei Cui, Shaohan Huang, Furu Wei, Zhoujun Li, Ming Zhou

Category: ocr
Keywords: document layout analysis, dataset, token-level annotations, computer vision, NLP
Year: 2020

Document layout analysis usually relies on computer vision models to understand
documents while ignoring textual information that is vital to capture.
Meanwhile, high quality labeled datasets with both visual and textual
information are still insufficient. In this paper, we present DocBank, a
benchmark dataset that contains 500K document pages with fine-grained token-
level annotations for document layout analysis. DocBank is constructed using a
simple yet effective way with weak supervision from the LaTeX documents
available on the arXiv.com. With DocBank, models from different modalities can
be compared fairly and multi-modal approaches will be further investigated and
boost the performance of document layout analysis. We build several strong
baselines and manually split train/dev/test sets for evaluation. Experiment
results show that models trained on DocBank accurately recognize the layout
information for a variety of documents. The DocBank dataset is publicly
available at https://github.com/doc-analysis/DocBank.

---

## Global Table Extractor (GTE): A Framework for Joint Table Identification and Cell Structure Recognition Using Visual Context

Xinyi Zheng, Douglas Burdick, Lucian Popa, Xu Zhong, Nancy Xin Ru Wang

Category: ocr
Keywords: table detection, cell structure recognition, deep learning, PDF documents, visual context
Year: 2020

Documents are often used for knowledge sharing and preservation in business and
science, within which are tables that capture most of the critical data.
Unfortunately, most documents are stored and distributed as PDF or scanned
images, which fail to preserve logical table structure. Recent vision-based deep
learning approaches have been proposed to address this gap, but most still
cannot achieve state-of-the-art results. We present Global Table Extractor
(GTE), a vision-guided systematic framework for joint table detection and cell
structured recognition, which could be built on top of any object detection
model. With GTE-Table, we invent a new penalty based on the natural cell
containment constraint of tables to train our table network aided by cell
location predictions. GTE-Cell is a new hierarchical cell detection network that
leverages table styles. Further, we design a method to automatically label table
and cell structure in existing documents to cheaply create a large corpus of
training and test data. We use this to enhance PubTabNet with cell labels and
create FinTabNet, real-world and complex scientific and financial datasets with
detailed table structure annotations to help train and test structure
recognition. Our framework surpasses previous state-of-the-art results on the
ICDAR 2013 and ICDAR 2019 table competition in both table detection and cell
structure recognition. Further experiments demonstrate a greater than 45%
improvement in cell structure recognition when compared to a vanilla RetinaNet
object detection model in our new out-of-domain FinTabNet.

---

## LayoutLM: Pre-training of Text and Layout for Document Image Understanding

Yiheng Xu, Minghao Li, Lei Cui, Shaohan Huang, Furu Wei, Ming Zhou

Category: ocr
Keywords: LayoutLM, pre-trained models, document image understanding, information extraction, text and layout
Year: 2020

Pre-training techniques have been successfully verified in a variety of NLP
tasks in recent years. Despite their widespread use for NLP applications, these
models almost exclusively focus on text-level manipulation, neglecting layout
and style information vital for document image understanding. In this paper, we
propose LayoutLM to jointly model interactions between text and layout
information across scanned document images, which benefits numerous real-world
document image understanding tasks such as information extraction from scanned
documents. Furthermore, we leverage image features to incorporate words' visual
information into LayoutLM. To the best of our knowledge, this is the first time
that text and layout are jointly learned in a single framework for document-
level pre-training. It achieves new state-of-the-art results in several
downstream tasks, including form understanding, receipt understanding, and
document image classification.

---

## DocBank: A Benchmark Dataset for Document Layout Analysis

Minghao Li, Yiheng Xu, Lei Cui, Shaohan Huang, Furu Wei, Zhoujun Li, Ming Zhou

Category: ocr
Keywords: document layout analysis, benchmark dataset, DocBank, computer vision, textual information, multi-modal approaches, LaTeX documents
Year: 2020

Document layout analysis usually relies on computer vision models to understand
documents while ignoring textual information that is vital to capture.
Meanwhile, high quality labeled datasets with both visual and textual
information are still insufficient. In this paper, we present DocBank, a
benchmark dataset that contains 500K document pages with fine-grained token-
level annotations for document layout analysis. DocBank is constructed using a
simple yet effective way with weak supervision from the LaTeX documents
available on the arXiv.com. With DocBank, models from different modalities can
be compared fairly and multi-modal approaches will be further investigated and
boost the performance of document layout analysis. We build several strong
baselines and manually split train/dev/test sets for evaluation. Experiment
results show that models trained on DocBank accurately recognize the layout
information for a variety of documents. The DocBank dataset is publicly
available at https://github.com/doc-analysis/DocBank.

---

## LayoutLM: Pre-training of Text and Layout for Document Image Understanding

Yiheng Xu, Minghao Li, Lei Cui, Shaohan Huang, Furu Wei, Ming Zhou

Category: ocr
Keywords: LayoutLM, pre-trained models, document image understanding
Year: 2020

Pre-training techniques have been verified successfully in a variety of NLP
tasks in recent years. Despite the widespread use of pre-training models for NLP
applications, they almost exclusively focus on text-level manipulation, while
neglecting layout and style information that is vital for document image
understanding. In this paper, we propose the LayoutLM to jointly model
interactions between text and layout information across scanned document images,
which is beneficial for a great number of real-world document image
understanding tasks such as information extraction from scanned documents.
Furthermore, we also leverage image features to incorporate words’ visual
information into LayoutLM. To the best of our knowledge, this is the first time
that text and layout are jointly learned in a single framework for document-
level pre-training. It achieves new state-of-the-art results in several
downstream tasks, including form understanding (from 70.72 to 79.27), receipt
understanding (from 94.02 to 95.24) and document image classification (from
93.07 to 94.42). The code and pre-trained LayoutLM models are publicly available
at https://aka.ms/layoutlm.

---

## OCR4all—An Open-Source Tool Providing a (Semi-)Automatic OCR Workflow for Historical Printings

Christian Reul, Dennis Christ, Alexander Hartelt, Nico Balbach, Maximilian Wehner, Uwe Springmann, Christoph Wick, Christine Grundig, Andreas Büttner, Frank Puppe

Category: ocr
Keywords: optical character recognition, document analysis, historical printings
Year: 2019

Optical Character Recognition (OCR) on historical printings is a challenging
task mainly due to the complexity of the layout and the highly variant
typography. Nevertheless, in the last few years, great progress has been made in
the area of historical OCR, resulting in several powerful open-source tools for
preprocessing, layout analysis and segmentation, character recognition, and
post-processing. The drawback of these tools often is their limited
applicability by non-technical users like humanist scholars and in particular
the combined use of several tools in a workflow. In this paper, we present an
open-source OCR software called OCR4all, which combines state-of-the-art OCR
components and continuous model training into a comprehensive workflow. While a
variety of materials can already be processed fully automatically, books with
more complex layouts require manual intervention by the users. This is mostly
due to the fact that the required ground truth for training stronger mixed
models (for segmentation, as well as text recognition) is not available, yet,
neither in the desired quantity nor quality. To deal with this issue in the
short run, OCR4all offers a comfortable GUI that allows error corrections not
only in the final output, but already in early stages to minimize error
propagations. In the long run, this constant manual correction produces large
quantities of valuable, high quality training material, which can be used to
improve fully automatic approaches. Further on, extensive configuration
capabilities are provided to set the degree of automation of the workflow and to
make adaptations to the carefully selected default parameters for specific
printings, if necessary. During experiments, the fully automated application on
19th Century novels showed that OCR4all can considerably outperform the
commercial state-of-the-art tool ABBYY Finereader on moderate layouts if
suitably pretrained mixed OCR models are available. Furthermore, on very complex
early printed books, even users with minimal or no experience were able to
capture the text with manageable effort and great quality, achieving excellent
Character Error Rates (CERs) below 0.5%. The architecture of OCR4all allows the
easy integration (or substitution) of newly developed tools for its main
components by standardized interfaces like PageXML, thus aiming at continual
higher automation for historical printings.

---

## Text Line Segmentation in Historical Document Images Using an Adaptive U-Net Architecture

Olfa Mechi, Maroua Mehri, Rolf Ingold, Najoua Essoukri Ben Amara

Category: ocr
Keywords: Deep learning, U-Net architecture, Text line segmentation, Historical document images
Year: 2019

On most document image transcription, indexing and retrieval systems, text line
segmentation remains one of the most important preliminary tasks. Hence, the
research community working in document image analysis is particularly interested
in providing reliable text line segmentation methods. Recently, an increasing
interest in using deep learning-based methods has been noted for solving various
sub-fields and tasks related to the issues surrounding document image analysis.
Thanks to the computer hardware and software evolution, several methods based on
using deep architectures continue to outperform the pattern recognition issues
and particularly those related to historical document image analysis. Thus, in
this paper we present a novel deep learning-based method for text line
segmentation of historical documents. The proposed method is based on using an
adaptive U-Net architecture. Qualitative and numerical experiments are given
using a large number of historical document images collected from the Tunisian
national archives and different recent benchmarking datasets provided in the
context of ICDAR and ICFHR competitions. Moreover, the results achieved are
compared with those obtained using the state-of-the-art methods.

---

## Text Line Segmentation in Historical Document Images Using an Adaptive U-Net Architecture

Olfa Mechi, Maroua Mehri, Rolf Ingold, Najoua Essoukri Ben Amara

Category: ocr
Keywords: Deep learning, U-Net architecture, Text line segmentation, Historical document images
Year: 2019

On most document image transcription, indexing and retrieval systems, text line
segmentation remains one of the most important preliminary tasks. Hence, the
research community working in document image analysis is particularly interested
in providing reliable text line segmentation methods. Recently, an increasing
interest in using deep learning-based methods has been noted for solving various
sub-fields and tasks related to the issues surrounding document image analysis.
Thanks to the computer hardware and software evolution, several methods based on
using deep architectures continue to outperform the pattern recognition issues
and particularly those related to historical document image analysis. Thus, in
this paper, we present a novel deep learning-based method for text line
segmentation of historical documents. The proposed method is based on using an
adaptive U-Net architecture. Qualitative and numerical experiments are given
using a large number of historical document images collected from the Tunisian
national archives and different recent benchmarking datasets provided in the
context of ICDAR and ICFHR competitions. Moreover, the results achieved are
compared with those obtained using the state-of-the-art methods.

---

## Efficient, Lexicon-Free OCR using Deep Learning

Marcin Namysl, Iuliu Konya

Category: ocr
Keywords: OCR, CNN, LSTM, CTC, synthetic data
Year: 2019

Contrary to popular belief, Optical Character Recognition (OCR) remains a
challenging problem when text occurs in unconstrained environments, like natural
scenes, due to geometrical distortions, complex backgrounds, and diverse fonts.
In this paper, we present a segmentation-free OCR system that combines deep
learning methods, synthetic training data generation, and data augmentation
techniques. We render synthetic training data using large text corpora and over
2000 fonts. To simulate text occurring in complex natural scenes, we augment
extracted samples with geometric distortions and with a proposed data
augmentation technique – alpha-compositing with background textures. Our models
employ a convolutional neural network encoder to extract features from text
images. Inspired by the recent progress in neural machine translation and
language modeling, we examine the capabilities of both recurrent and
convolutional neural networks in modeling the interactions between input
elements. The proposed OCR system surpasses the accuracy of leading commercial
and open-source engines on distorted text samples.

---

## FUNSD: A Dataset for Form Understanding in Noisy Scanned Documents

Guillaume Jaume, Hazım Kemal Ekenel, Jean-Philippe Thiran

Category: ocr
Keywords: Text detection, Optical Character Recognition, Form Understanding, Spatial Layout Analysis
Year: 2019

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
https://guillaumejaume.github.io/FUNSD/

---

## Efficient, Lexicon-Free OCR using Deep Learning

Marcin Namysl, Iuliu Konya

Category: ocr
Keywords: OCR, CNN, LSTM, CTC, synthetic data
Year: 2019

Contrary to popular belief, Optical Character Recognition (OCR) remains a
challenging problem when text occurs in unconstrained environments, like natural
scenes, due to geometrical distortions, complex backgrounds, and diverse fonts.
In this paper, we present a segmentation-free OCR system that combines deep
learning methods, synthetic training data generation, and data augmentation
techniques. We render synthetic training data using large text corpora and over
2000 fonts. To simulate text occurring in complex natural scenes, we augment
extracted samples with geometric distortions and with a proposed data
augmentation technique – alpha-compositing with background textures. Our models
employ a convolutional neural network encoder to extract features from text
images. Inspired by the recent progress in neural machine translation and
language modeling, we examine the capabilities of both recurrent and
convolutional neural networks in modeling the interactions between input
elements. The proposed OCR system surpasses the accuracy of leading commercial
and open-source engines on distorted text samples.

---

## Table Understanding in Structured Documents

Martin Holecek, Antonín Hoskovec, Petr Baudiš, Pavel Klinger

Category: ocr
Keywords: table detection, neural networks, invoices, graph convolution, attention
Year: 2019

Table detection and extraction has been studied in the context of documents like
reports, where tables are clearly outlined and stand out from the document
structure visually. We study this topic in a rather more challenging domain of
layout-heavy business documents, particularly invoices. Invoices present the
novel challenges of tables being often without outlines - either in the form of
borders or surrounding text flow - with ragged columns and widely varying data
content. We will also show that we can extract specific information from
structurally different tables or table-like structures with one model. We
present a comprehensive representation of a page using a graph over word boxes,
positional embeddings, trainable textual features and rephrase the table
detection as a text box labeling problem. We work on our newly presented dataset
of pro forma invoices, invoices, and debit note documents using this
representation and propose multiple baselines to solve this labeling problem. We
then propose a novel neural network model that achieves strong, practical
results on the presented dataset and analyze the model performance and effects
of graph convolutions and self-attention in detail.

---

## ICDAR2019 Competition on Recognition of Documents with Complex Layouts – RDCL2019

C. Clausner, A. Antonacopoulos, S. Pletschacher

Category: ocr
Keywords: performance evaluation, page segmentation, region classification, layout analysis, OCR, recognition, datasets
Year: 2019

This paper presents an objective comparative evaluation of page segmentation and
region classification methods for documents with complex layouts. It describes
the competition (modus operandi, dataset, and evaluation methodology) held in
the context of ICDAR2019, presenting the results of the evaluation of twelve
methods – nine submitted, three state-of-the-art systems (commercial and open-
source). Three scenarios are reported in this paper, one evaluating the ability
of methods to accurately segment regions and two evaluating both segmentation
and region classification. Text recognition was a bonus challenge and was not
taken up by all participants. The results indicate that an innovative approach
has a clear advantage but there is still a considerable need to develop robust
methods that deal with layout challenges, especially with the non-textual
content.

---

## PubLayNet: largest dataset ever for document layout analysis

Xu Zhong, Jianbin Tang, Antonio Jimeno Yepes

Category: ocr
Keywords: automatic annotation, document layout, deep learning, transfer learning
Year: 2019

Recognizing the layout of unstructured digital documents is an important step
when parsing the documents into structured machine-readable format for
downstream applications. Deep neural networks developed for computer vision have
been proven effective in analyzing the layout of document images. However,
existing publicly available document layout datasets are significantly smaller
than established computer vision datasets. Models often have to be trained using
transfer learning from a base model pre-trained on a traditional computer vision
dataset. In this paper, we develop the PubLayNet dataset for document layout
analysis by automatically matching the XML representations and content of over 1
million PDF articles publicly available on PubMed Central™. The dataset's size
is comparable to established computer vision datasets, containing over 360,000
document images with annotated typical document layout elements. The experiments
demonstrate that deep neural networks trained on PubLayNet accurately recognize
the layout of scientific articles. The pre-trained models also serve as a more
effective base for transfer learning on different document domains. We release
the dataset (https://github.com/ibm-aur-nlp/PubLayNet) to support the
development and evaluation of more advanced models for document layout analysis.

---

## Accurate, Data-Efficient, Unconstrained Text Recognition with Convolutional Neural Networks

Mohamed Yousef, Khaled F. Hussain, Usama S. Mohammed

Category: ocr
Keywords: Text Recognition, Optical Character Recognition, Handwriting Recognition, CAPTCHA Solving, License Plate Recognition, Convolutional Neural Network, Deep Learning
Year: 2019

Unconstrained text recognition is an important computer vision task, featuring a
wide variety of different sub-tasks, each with its own set of challenges. One of
the biggest promises of deep neural networks has been the convergence and
automation of feature extractors from input raw signals, allowing for the
highest possible performance with minimum required domain knowledge. To this
end, we propose a data-efficient, end-to-end neural network model for generic,
unconstrained text recognition. In our proposed architecture we strive for
simplicity and efficiency without sacrificing recognition accuracy. Our proposed
architecture is a fully convolutional network without any recurrent connections
trained with the CTC loss function. Thus it operates on arbitrary input sizes
and produces strings of arbitrary length in a very efficient and parallelizable
manner. We show the generality and superiority of our proposed text recognition
architecture by achieving state of the art results on seven public benchmark
datasets, covering a wide spectrum of text recognition tasks, namely:
Handwriting Recognition, CAPTCHA recognition, OCR, License Plate Recognition,
and Scene Text Recognition. Our proposed architecture has won the ICFHR2018
Competition on Automated Text Recognition on a READ Dataset.

---

## A two-stage method for text line detection in historical documents

Tobias Grüning, Gundram Leifert, Tobias Strauß, Johannes Michael, Roger Labahn

Category: ocr
Keywords: Baseline detection, Text line detection, Layout analysis, Historical documents, U-Net, Pixel labeling
Year: 2019

This work presents a two-stage text line detection method for historical
documents. Each detected text line is represented by its baseline. In the first
stage, a deep neural network called ARU-Net labels pixels to belong to one of
three classes: baseline, separator, and other. The separator class marks the
beginning and end of each text line. The ARU-Net is trainable from scratch with
manageably few manually annotated example images (< 50) by utilizing data
augmentation strategies. The network predictions are used as input for the
second stage, which performs a bottom-up clustering to build baselines. The
developed method is capable of handling complex layouts as well as curved and
arbitrarily oriented text lines. It substantially outperforms current state-of-
the-art approaches. For example, for the complex track of the cBAD: ICDAR2017
Competition on Baseline Detection, the F value is increased from 0.859 to 0.922.
The framework to train and run the ARU-Net is open source.

---

## PubLayNet: largest dataset ever for document layout analysis

Xu Zhong, Jianbin Tang, Antonio Jimeno Yepes

Category: ocr
Keywords: automatic annotation, document layout, deep learning, transfer learning
Year: 2019

Recognizing the layout of unstructured digital documents is an important step
when parsing the documents into structured machine-readable format for
downstream applications. Deep neural networks that are developed for computer
vision have been proven to be an effective method to analyze layout of document
images. However, document layout datasets that are currently publicly available
are several magnitudes smaller than established computing vision datasets.
Models have to be trained by transfer learning from a base model that is pre-
trained on a traditional computer vision dataset. In this paper, we develop the
PubLayNet dataset for document layout analysis by automatically matching the XML
representations and the content of over 1 million PDF articles that are publicly
available on PubMed Central™. The size of the dataset is comparable to
established computer vision datasets, containing over 360 thousand document
images, where typical document layout elements are annotated. The experiments
demonstrate that deep neural networks trained on PubLayNet accurately recognize
the layout of scientific articles. The pre-trained models are also a more
effective base mode for transfer learning on a different document domain. We
release the dataset (https://github.com/ibm-aur-nlp/PubLayNet) to support
development and evaluation of more advanced models for document layout analysis.

---

## Chargrid-OCR: End-to-end Trainable Optical Character Recognition through Semantic Segmentation and Object Detection

Christian Reisswig, Anoop R Katti, Marco Spinaci, Johannes Höhne

Category: ocr
Keywords: OCR, semantic segmentation, object detection, chargrid, deep learning
Year: 2019

We present an end-to-end trainable approach for optical character recognition
(OCR) on printed documents. It is based on predicting a two-dimensional
character grid (chargrid) representation of a document image as a semantic
segmentation task. To identify individual character instances from the chargrid,
we regard characters as objects and use object detection techniques from
computer vision. We demonstrate experimentally that our method outperforms
previous state-of-the-art approaches in accuracy while being easily
parallelizable on GPU (therefore being significantly faster), as well as easier
to train.

---

## Text and non-text separation in offline document images: a survey

Showmik Bhowmik, Ram Sarkar, Mita Nasipuri, David Doermann

Category: ocr
Keywords: Text/non-text separation, Segmentation, Offline document images, Engineering drawing, Map, Unconstrained handwritten document, Newspaper, Journal, Magazine, Check, Form, Survey
Year: 2018

Separation of text and non-text is an essential processing step for any document
analysis system. Therefore, it is important to have a clear understanding of the
state-of-the-art of text/non-text separation in order to facilitate the
development of efficient document processing systems. This paper first
summarizes the technical challenges of performing text/non-text separation. It
then categorizes offline document images into different classes according to the
nature of the challenges one faces, in an attempt to provide insight into
various techniques presented in the literature. The pros and cons of various
techniques are explained wherever possible. Along with the evaluation protocols,
benchmark databases, this paper also presents a performance comparison of
different methods. Finally, this article highlights the future research
challenges and directions in this domain.

---

## Fully Convolutional Neural Networks for Page Segmentation of Historical Document Images

Christoph Wick, Frank Puppe

Category: ocr
Keywords: page segmentation, historical document analysis, fully convolutional network, foreground pixel accuracy
Year: 2018

We propose a high-performance fully convolutional neural network (FCN) for
historical document segmentation that is designed to process a single page in
one step. The advantage of this model beside its speed is its ability to
directly learn from raw pixels instead of using preprocessing steps e.g. feature
computation or superpixel generation. We show that this network yields better
results than existing methods on different public data sets. For evaluation of
this model we introduce a novel metric that is independent of ambiguous ground
truth called Foreground Pixel Accuracy (FgPA). This pixel-based measure only
counts foreground pixels in the binarized page, any background pixel is omitted.
The major advantage of this metric is that it enables researchers to compare
different segmentation methods on their ability to successfully segment text or
pictures and not on their ability to learn and possibly overfit the
peculiarities of an ambiguous hand-made ground truth segmentation.

---

## dhSegment: A Generic Deep-Learning Approach for Document Segmentation

Sofia Ares Oliveira, Benoit Seguin, Frederic Kaplan

Category: ocr
Keywords: document segmentation, historical document processing, document layout analysis, neural network, deep learning
Year: 2018

In recent years there have been multiple successful attempts tackling document
processing problems separately by designing task specific hand-tuned strategies.
We argue that the diversity of historical document processing tasks prohibits
solving them one at a time and shows a need for designing generic approaches to
handle the variability of historical series. In this paper, we address multiple
tasks simultaneously such as page extraction, baseline extraction, layout
analysis, or multiple typologies of illustrations and photograph extraction. We
propose an open-source implementation of a CNN-based pixel-wise predictor
coupled with task-dependent post-processing blocks. We show that a single CNN
architecture can be used across tasks with competitive results. Moreover, most
of the task-specific post-processing steps can be decomposed into a small number
of simple and standard reusable operations, adding to the flexibility of our
approach.

---

## DeepDeSRT: Deep Learning for Detection and Structure Recognition of Tables in Document Images

Sebastian Schreiber, Stefan Agne, Ivo Wolf, Andreas Dengel, Sheraz Ahmed

Category: ocr
Keywords: table detection, structure recognition, document images, deep learning, transfer learning
Year: 2017

This paper presents a novel end-to-end system for table understanding in
document images called DeepDeSRT. In particular, the contribution of DeepDeSRT
is two-fold. First, it presents a deep learning-based solution for table
detection in document images. Secondly, it proposes a novel deep learning-based
approach for table structure recognition, i.e. identifying rows, columns, and
cell positions in the detected tables. In contrast to existing rule-based
methods, which rely on heuristics or additional PDF metadata, the presented
system is data-driven and does not need any heuristics or metadata to detect and
recognize tabular structures in document images. DeepDeSRT processes document
images, making it suitable for both born-digital PDFs and scanned documents. The
system is evaluated on the publicly available ICDAR 2013 table competition
dataset and a closed dataset from a major European aviation company,
demonstrating high detection accuracy and generalization capabilities.

---

## Robust Document Image Dewarping Method using Text-lines and Line Segments

Taeho Kil, Wonkyo Seo, Hyung Il Koo, Nam Ik Cho

Category: ocr
Keywords: document image processing, dewarping, text-lines, line segments, layout analysis
Year: 2017

Conventional text-line based document dewarping methods have problems when
handling complex layout and/or very few text-lines. When there are few aligned
text-lines in the image, this usually means that photos, graphics and/or tables
take large portion of the input instead. Hence, for the robust document
dewarping, we propose to use line segments in the image in addition to the
aligned text-lines. Based on the assumption and observation that many of the
line segments in the image are horizontally or vertically aligned in the well-
rectified images, we encode this property into the cost function in addition to
the text-line alignment cost. By minimizing the function, we can obtain
transformation parameters for camera pose, page curve, etc., which are used for
document rectification. Considering that there are many outliers in line segment
directions and missed text-lines in some cases, the overall algorithm is
designed in an iterative manner. At each step, we remove text components and
line segments that are not well aligned, and then minimize the cost function
with the updated information. Experimental results show that the proposed method
is robust to the variety of page layouts.

---

## Convolutional Neural Networks for Page Segmentation of Historical Document Images

Kai Chen, Mathias Seuret, Jean Hennebert, Rolf Ingold

Category: ocr
Keywords: convolutional neural network, page segmentation, layout analysis, historical document images, deep learning
Year: 2017

This paper presents a page segmentation method for handwritten historical
document images based on a Convolutional Neural Network (CNN). We consider page
segmentation as a pixel labeling problem, i.e., each pixel is classified as one
of the predefined classes. Traditional methods in this area rely on hand-crafted
features carefully tuned considering prior knowledge. In contrast, we propose to
learn features from raw image pixels using a CNN. While many researchers focus
on developing deep CNN architectures to solve different problems, we train a
simple CNN with only one convolution layer. We show that the simple architecture
achieves competitive results against other deep architectures on different
public datasets. Experiments also demonstrate the effectiveness and superiority
of the proposed method compared to previous methods.

---

## Robust Document Image Dewarping Method using Text-lines and Line Segments

Taeho Kil, Wonkyo Seo, Hyung Il Koo, Nam Ik Cho

Category: ocr
Keywords: document dewarping, text-lines, line segments, document image processing, camera-captured images
Year: 2017

Conventional text-line based document dewarping methods have problems when
handling complex layout and/or very few text-lines. When there are few aligned
text-lines in the image, this usually means that photos, graphics and/or tables
take large portion of the input instead. Hence, for the robust document
dewarping, we propose to use line segments in the image in addition to the
aligned text-lines. Based on the assumption and observation that many of the
line segments in the image are horizontally or vertically aligned in the well-
rectified images, we encode this property into the cost function in addition to
the text-line alignment cost. By minimizing the function, we can obtain
transformation parameters for camera pose, page curve, etc., which are used for
document rectification. Considering that there are many outliers in line segment
directions and missed text-lines in some cases, the overall algorithm is
designed in an iterative manner. At each step, we remove text components and
line segments that are not well aligned, and then minimize the cost function
with the updated information. Experimental results show that the proposed method
is robust to the variety of page layouts.

---

## A comprehensive survey of mostly textual document segmentation algorithms since 2008

Sébastien Eskenazi, Petra Gomez-Krämer, Jean-Marc Ogier

Category: ocr
Keywords: Document, Segmentation, Survey, Evaluation, Trends, Typology
Year: 2017

In document image analysis, segmentation is the task that identifies the regions
of a document. The increasing number of applications of document analysis
requires a good knowledge of the available technologies. This survey highlights
the variety of the approaches that have been proposed for document image
segmentation since 2008. It provides a clear typology of documents and of
document image segmentation algorithms. We also discuss the technical
limitations of these algorithms, the way they are evaluated and the general
trends of the community.

---

## High Performance Text Recognition using a Hybrid Convolutional-LSTM Implementation

Thomas M. Breuel

Category: ocr
Keywords: OCR, LSTM, convolutional networks, geometric normalization, PyTorch, CUDA
Year: 2017

Optical character recognition (OCR) has made great progress in recent years due
to the introduction of recognition engines based on recurrent neural networks,
in particular the LSTM architecture. This paper describes a new, open-source
line recognizer combining deep convolutional networks and LSTMs, implemented in
PyTorch and using CUDA kernels for speed. Experimental results are given
comparing the performance of different combinations of geometric normalization,
1D LSTM, deep convolutional networks, and 2D LSTM networks. An important result
is that while deep hybrid networks without geometric text line normalization
outperform 1D LSTM networks with geometric normalization, deep hybrid networks
with geometric text line normalization still outperform all other networks. The
best networks achieve a throughput of more than 100 lines per second and test
set error rates on UW3 of 0.25%.

---

## A comprehensive survey of mostly textual document segmentation algorithms since 2008

Sébastien Eskenazi, Petra Gomez-Krämer, Jean-Marc Ogier

Category: ocr
Keywords: Document, Segmentation, Survey, Evaluation, Trends, Typology
Year: 2017

In document image analysis, segmentation is the task that identifies the regions
of a document. The increasing number of applications of document analysis
requires a good knowledge of the available technologies. This survey highlights
the variety of the approaches that have been proposed for document image
segmentation since 2008. It provides a clear typology of documents and of
document image segmentation algorithms. We also discuss the technical
limitations of these algorithms, the way they are evaluated and the general
trends of the community.

---

## Gated Recurrent Convolution Neural Network for OCR

Jianfeng Wang, Xiaolin Hu

Category: ocr
Keywords: optical character recognition, GRCNN, BLSTM, scene text recognition, deep learning
Year: 2017

Optical Character Recognition (OCR) aims to recognize text in natural images.
Inspired by a recently proposed model for general image classification,
Recurrent Convolution Neural Network (RCNN), we propose a new architecture named
Gated RCNN (GRCNN) for solving this problem. Its critical component, Gated
Recurrent Convolution Layer (GRCL), is constructed by adding a gate to the
Recurrent Convolution Layer (RCL), the critical component of RCNN. The gate
controls the context modulation in RCL and balances the feed-forward information
and the recurrent information. In addition, an efficient Bidirectional Long
Short-Term Memory (BLSTM) is built for sequence modeling. The GRCNN is combined
with BLSTM to recognize text in natural images. The entire GRCNN-BLSTM model can
be trained end-to-end. Experiments show that the proposed model outperforms
existing methods on several benchmark datasets including the IIIT-5K, Street
View Text (SVT) and ICDAR.

---

## DeepDeSRT: Deep Learning for Detection and Structure Recognition of Tables in Document Images

Sebastian Schreiber, Stefan Agne, Ivo Wolf, Andreas Dengel, Sheraz Ahmed

Category: ocr
Keywords: table detection, structure recognition, document images, deep learning, transfer learning
Year: 2017

This paper presents a novel end-to-end system for table understanding in
document images called DeepDeSRT. In particular, the contribution of DeepDeSRT
is two-fold. First, it presents a deep learning-based solution for table
detection in document images. Secondly, it proposes a novel deep learning-based
approach for table structure recognition, i.e. identifying rows, columns, and
cell positions in the detected tables. In contrast to existing rule-based
methods, which rely on heuristics or additional PDF metadata, the presented
system is data-driven and does not need any heuristics or metadata to detect as
well as to recognize tabular structures in document images. Furthermore, in
contrast to most existing table detection and structure recognition methods,
which are applicable only to PDFs, DeepDeSRT processes document images, which
makes it equally suitable for born-digital PDFs as well as scanned documents. To
gauge the performance of DeepDeSRT, the system is evaluated on the publicly
available ICDAR 2013 table competition dataset and a closed dataset from a real
use case of a major European aviation company.

---

## GROBID - Information Extraction from Scientific Publications

Laurent Romary, Patrice Lopez

Category: ocr
Keywords: information extraction, scientific publications, data sharing, metadata extraction, bibliographic references
Year: 2017

GROBID is a tool for extracting information from scientific publications,
facilitating the transformation of documents into structured data. It is used to
parse and process scholarly articles, extracting metadata, bibliographic
references, and other essential information, thus aiding in scientific data
sharing and reuse.

---

## A comprehensive survey of mostly textual document segmentation algorithms since 2008

Sébastien Eskenazi, Petra Gomez-Krämer, Jean-Marc Ogier

Category: ocr
Keywords: Document, Segmentation, Survey, Evaluation, Trends, Typology
Year: 2017

In document image analysis, segmentation is the task that identifies the regions
of a document. The increasing number of applications of document analysis
requires a good knowledge of the available technologies. This survey highlights
the variety of the approaches that have been proposed for document image
segmentation since 2008. It provides a clear typology of documents and of
document image segmentation algorithms. We also discuss the technical
limitations of these algorithms, the way they are evaluated and the general
trends of the community.

---

## Historical Document Digitization through Layout Analysis and Deep Content Classification

Andrea Corbelli, Lorenzo Baraldi, Costantino Grana, Rita Cucchiara

Category: ocr
Keywords: document digitization, layout segmentation, content classification, historical documents, Convolutional Neural Networks, Random Forest
Year: 2016

Document layout segmentation and recognition is an important task in the
creation of digitized documents collections, especially when dealing with
historical documents. This paper presents a hybrid approach to layout
segmentation as well as a strategy to classify document regions, which is
applied to the process of digitization of a historical encyclopedia. Our layout
analysis method merges a classic top-down approach and a bottom-up
classification process based on local geometrical features, while regions are
classified by means of features extracted from a Convolutional Neural Network
merged in a Random Forest classifier. Experiments are conducted on the first
volume of the 'Enciclopedia Treccani', a large dataset containing 999 manually
annotated pages from the historical Italian encyclopedia.

---

## A Survey on Optical Character Recognition System

Noman Islam, Zeeshan Islam, Nazia Noor

Category: ocr
Keywords: character recognition, document image analysis, OCR, OCR survey, classification
Year: 2016

Optical Character Recognition (OCR) has been a topic of interest for many years.
It is defined as the process of digitizing a document image into its constituent
characters. Despite decades of intense research, developing OCR with
capabilities comparable to that of human still remains an open challenge. Due to
this challenging nature, researchers from industry and academic circles have
directed their attentions towards Optical Character Recognition. Over the last
few years, the number of academic laboratories and companies involved in
research on Character Recognition has increased dramatically. This research aims
at summarizing the research so far done in the field of OCR. It provides an
overview of different aspects of OCR and discusses corresponding proposals aimed
at resolving issues of OCR.

---

## Historical Document Digitization through Layout Analysis and Deep Content Classification

Andrea Corbelli, Lorenzo Baraldi, Costantino Grana, Rita Cucchiara

Category: ocr
Keywords: document digitization, layout segmentation, content classification, historical documents, Enciclopedia Treccani
Year: 2016

Document layout segmentation and recognition is an important task in the
creation of digitized documents collections, especially when dealing with
historical documents. This paper presents a hybrid approach to layout
segmentation as well as a strategy to classify document regions, which is
applied to the process of digitization of an historical encyclopedia. Our layout
analysis method merges a classic top-down approach and a bottom-up
classification process based on local geometrical features, while regions are
classified by means of features extracted from a Convolutional Neural Network
merged in a Random Forest classifier. Experiments are conducted on the first
volume of the 'Enciclopedia Treccani', a large dataset containing 999 manually
annotated pages from the historical Italian encyclopedia.

---

## GROBID - Information Extraction from Scientific Publications

Laurent Romary, Patrice Lopez

Category: ocr
Keywords: GROBID, information extraction, scientific publications, machine learning, PDF parsing
Year: 2015

The document presents GROBID, a machine learning library for extracting,
parsing, and re-structuring raw documents, particularly scientific papers, into
structured documents. GROBID is designed to process header metadata,
bibliographic references, and full-text content from PDF documents to facilitate
information extraction and improve data sharing and re-use in scientific
research.

---

## ICDAR 2013 Robust Reading Competition

Dimosthenis Karatzas, Faisal Shafait, Seiichi Uchida, Masakazu Iwamura, Lluis Gomez i Bigorda, Sergi Robles Mestre, Joan Mas, David Fernandez Mota, Jon Almazan, Lluis Pere de las Heras

Category: ocr
Keywords: Robust Reading Competition, text extraction, text localization, text segmentation, word recognition, born-digital images, real scene images, video sequences
Year: 2013

This report presents the final results of the ICDAR 2013 Robust Reading
Competition. The competition is structured in three Challenges addressing text
extraction in different application domains, namely born-digital images, real
scene images, and real-scene videos. The Challenges are organized around
specific tasks covering text localization, text segmentation, and word
recognition. The competition took place in the first quarter of 2013, and
received a total of 42 submissions over the different tasks offered. This report
describes the datasets and ground truth specification, details the performance
evaluation protocols used, and presents the final results along with a brief
summary of the participating methods.

---

## High-Performance OCR for Printed English and Fraktur using LSTM Networks

Thomas M. Breuel, Adnan Ul-Hasan, Mayce Al Azawi, Faisal Shafait

Category: ocr
Keywords: OCR, LSTM networks, Printed text recognition, Fraktur, Bidirectional LSTM
Year: 2013

Long Short-Term Memory (LSTM) networks have yielded excellent results on
handwriting recognition. This paper describes an application of bidirectional
LSTM networks to the problem of machine-printed Latin and Fraktur recognition.
Latin and Fraktur recognition differs significantly from handwriting recognition
in both the statistical properties of the data, as well as in the required, much
higher levels of accuracy. Applications of LSTM networks to handwriting
recognition use two-dimensional recurrent networks, since the exact position and
baseline of handwritten characters is variable. In contrast, for printed OCR, we
used a one-dimensional recurrent network combined with a novel algorithm for
baseline and x-height normalization. A number of databases were used for
training and testing, including the UW3 database, artificially generated and
degraded Fraktur text and scanned pages from a book digitization project. The
LSTM architecture achieved 0.6% character-level test-set error on English text.
When the artificially degraded Fraktur data set is divided into training and
test sets, the system achieves an error rate of 1.64%. On specific books printed
in Fraktur (not part of the training set), the system achieves error rates of
0.15% (Fontane) and 1.47% (Ersch-Gruber). These recognition accuracies were
found without using any language modelling or any other post-processing
techniques.

---

## Coupled snakelets for curled text-line segmentation from warped document images

Syed Saqib Bukhari, Faisal Shafait, Thomas M. Breuel

Category: ocr
Keywords: Curled text-line segmentation, Page segmentation, Camera-captured document image processing
Year: 2013

Camera-captured, warped document images usually contain curled text-lines
because of distortions caused by camera perspective view and page curl. Warped
document images can be transformed into planar document images for improving
optical character recognition accuracy and human readability using monocular
dewarping techniques. Curled text-lines segmentation is a crucial initial step
for most of the monocular dewarping techniques. Existing curled text-line
segmentation approaches are sensitive to geometric and perspective distortions.
In this paper, we introduce a novel curled text-line segmentation algorithm by
adapting active contour (snake). Our algorithm performs text-line segmentation
by estimating pairs of x-line and baseline. It estimates a local pair of x-line
and baseline on each connected component by jointly tracing top and bottom
points of neighboring connected components, and finally each group of
overlapping pairs is considered as a segmented text-line. Our algorithm has
achieved curled text-line segmentation accuracy of above 95% on the DFKI-I
(CBDAR 2007 dewarping contest) dataset, which is significantly better than
previously reported results on this dataset.

---

## High-Performance OCR for Printed English and Fraktur using LSTM Networks

Thomas M. Breuel, Adnan Ul-Hasan, Mayce Al Azawi, Faisal Shafait

Category: ocr
Keywords: OCR, LSTM networks, Fraktur recognition, printed text recognition, baseline normalization
Year: 2013

Long Short-Term Memory (LSTM) networks have yielded excellent results on
handwriting recognition. This paper describes an application of bidirectional
LSTM networks to the problem of machine-printed Latin and Fraktur recognition.
Latin and Fraktur recognition differs significantly from handwriting recognition
in both the statistical properties of the data, as well as in the required, much
higher levels of accuracy. Applications of LSTM networks to handwriting
recognition use two-dimensional recurrent networks, since the exact position and
baseline of handwritten characters is variable. In contrast, for printed OCR, we
used a one-dimensional recurrent network combined with a novel algorithm for
baseline and x-height normalization. A number of databases were used for
training and testing, including the UW3 database, artificially generated and
degraded Fraktur text and scanned pages from a book digitization project. The
LSTM architecture achieved 0.6% character-level test-set error on English text.
When the artificially degraded Fraktur data set is divided into training and
test sets, the system achieves an error rate of 1.64%. On specific books printed
in Fraktur (not part of the training set), the system achieves error rates of
0.15% (Fontane) and 1.47% (Ersch-Gruber). These recognition accuracies were
found without using any language modelling or any other post-processing
techniques.

---

## Coupled snakelets for curled text-line segmentation from warped document images

Syed Saqib Bukhari, Faisal Shafait, Thomas M. Breuel

Category: ocr
Keywords: Curled text-line segmentation, Page segmentation, Camera-captured document image processing
Year: 2013

Camera-captured, warped document images usually contain curled text-lines due to
distortions caused by camera perspective view and page curl. Warped document
images can be transformed into planar document images to improve optical
character recognition accuracy and human readability using monocular dewarping
techniques. Curled text-line segmentation is a crucial initial step for most
monocular dewarping techniques. Existing curled text-line segmentation
approaches are sensitive to geometric and perspective distortions. In this
paper, we introduce a novel curled text-line segmentation algorithm by adapting
active contour (snake). Our algorithm performs text-line segmentation by
estimating pairs of x-line and baseline. It estimates a local pair of x-line and
baseline on each connected component by jointly tracing top and bottom points of
neighboring connected components, and finally, each group of overlapping pairs
is considered as a segmented text-line. Our algorithm has achieved curled text-
line segmentation accuracy of above 95% on the DFKI-I (CBDAR 2007 dewarping
contest) dataset, which is significantly better than previously reported results
on this dataset.

---

## An evaluation of HMM-based Techniques for the Recognition of Screen Rendered Text

Sheikh Faisal Rashid, Faisal Shafait, Thomas M Breuel

Category: ocr
Keywords: HMM, screen rendered text, text recognition, pattern recognition, OCR
Year: 2012

The document evaluates Hidden Markov Model (HMM) based techniques for the
recognition of text that is rendered on screens. The study involves analyzing
the effectiveness of HMMs in recognizing text from screen-based sources, which
can include digital displays, computer screens, and other similar mediums.

---

## Combined Orientation and Skew Detection Using Geometric Text-Line Modeling

Joost van Beusekom, Faisal Shafait, Thomas M. Breuel

Category: ocr
Keywords: Orientation detection, document preprocessing, line finding
Year: 2010

In large scale document digitization, orientation detection plays an important
role, especially in the scenario of digitizing incoming mail. The heavy use of
automatic document feeding (ADF) scanners and more over automatic processing of
facsimiles results in many documents being scanned in the wrong orientation.
These misoriented scans have to be corrected, as most subsequent processing
steps assume the document to be scanned in the right orientation. Several
existing methods for orientation detection use the fact that in Latin script
text, ascenders are more likely to occur than descenders. In this paper, we
propose a one-step skew and orientation detection method using a well
established geometric text-line model. The advantage of our method is that it
combines accurate skew estimation with robust, resolution independent
orientation detection. An interesting aspect of our method is that it
incorporates orientation detection into a previously published skew detection
method allowing to perform orientation detection, skew estimation, and, if
necessary, text-line extraction in one step. The effectiveness of our
orientation detection approach is demonstrated on the UW-I dataset, and on
publicly available test images from OCRopus. Our method achieves an accuracy of
99% on the UW-I dataset and 100% on test images from OCRopus.

---

## Scanning Neural Network for Text Line Recognition

Sheikh Faisal Rashid, Faisal Shafait, Thomas M. Breuel

Category: ocr
Keywords: Scanning Neural Network, Multilayer Perceptron, AutoMLP, Hidden Markov Models, Optical Character Recognition, Segmentation-free OCR
Year: 2010

Optical character recognition (OCR) of machine printed Latin script documents is
ubiquitously claimed as a solved problem. However, error-free OCR of degraded or
noisy text is still challenging for modern OCR systems. Most recent approaches
perform segmentation-based character recognition. This is tricky because
segmentation of degraded text is itself problematic. This paper describes a
segmentation-free text line recognition approach using multi-layer perceptron
(MLP) and hidden Markov models (HMMs). A line scanning neural network – trained
with character-level contextual information and a special garbage class – is
used to extract class probabilities at every pixel succession. The output of
this scanning neural network is decoded by HMMs to provide character-level
recognition. In evaluations on a subset of UNLV-ISRI document collection, we
achieve 98.4% character recognition accuracy that is statistically significantly
better in comparison with character recognition accuracies obtained from state-
of-the-art open source OCR systems.

---

## A Realistic Dataset for Performance Evaluation of Document Layout Analysis

A. Antonacopoulos, D. Bridson, C. Papadopoulos, S. Pletschacher

Category: ocr
Keywords: document layout analysis, dataset, ground truth, PAGE format, evaluation
Year: 2009

There is a significant need for a realistic dataset on which to evaluate layout
analysis methods and examine their performance in detail. This paper presents a
new dataset (and the methodology used to create it) based on a wide range of
contemporary documents. Strong emphasis is placed on comprehensive and detailed
representation of both complex and simple layouts, and on colour originals. In-
depth information is recorded both at the page and region level. Ground truth is
efficiently created using a new semi-automated tool and stored in a new
comprehensive XML representation, the PAGE format. The dataset can be browsed
and searched via a web-based front end to the underlying database and suitable
subsets (relevant to specific evaluation goals) can be selected and downloaded.

---

## Applying the OCRopus OCR System to Scholarly Sanskrit Literature

Thomas M. Breuel

Category: ocr
Keywords: OCRopus, OCR, Sanskrit literature, digital library, text recognition, layout analysis, statistical language models
Year: 2009

OCRopus is an open source OCR system currently being developed, intended to be
omni-lingual and omni-script. In addition to modern digital library
applications, applications of the system include capturing and recognizing
classical literature, as well as the large body of research literature about
classics. OCRopus advances the state of the art in a number of ways, including
the ability to easily plug in new text recognition and layout analysis modules,
the use of adaptive and user-extensible character recognition, and statistical
and trainable layout analysis. Of particular interest for computational
linguistics applications is the consistent use of probability estimates
throughout the system and the use of weighted finite state transducers to
represent both alternative recognition hypotheses and statistical language
models.

---

## The Effect of Border Noise on the Performance of Projection Based Page Segmentation Methods

Faisal Shafait, Thomas M. Breuel

Category: ocr
Keywords: Document page segmentation, OCR, performance evaluation, border noise removal, document cleanup
Year: 2009

Projection methods have been used in the analysis of bi-tonal document images
for different tasks like page segmentation and skew correction for over two
decades. However, these algorithms are sensitive to the presence of border noise
in document images. Border noise can appear along the page border due to
scanning or photocopying. Over the years, several page segmentation algorithms
have been proposed in the literature. Some of these algorithms have come to
widespread use due to their high accuracy and robustness with respect to border
noise. This paper addresses two important questions in this context: 1) Can
existing border noise removal algorithms clean up document images to a degree
required by projection methods to achieve competitive performance? 2) Can
projection methods reach the performance of other state-of-the-art page
segmentation algorithms (e.g. Docstrum or Voronoi) for documents where border
noise has successfully been removed? We perform extensive experiments on the
University of Washington (UW-III) dataset with six border noise removal methods.
Our results show that although projection methods can achieve the accuracy of
other state-of-the-art algorithms on the cleaned document images, existing
border noise removal techniques cannot clean up documents captured under a
variety of scanning conditions to the degree required to achieve that accuracy.

---

## Performance Evaluation and Benchmarking of Six-Page Segmentation Algorithms

Faisal Shafait, Daniel Keysers, Thomas M. Breuel

Category: ocr
Keywords: Document page segmentation, OCR, performance evaluation, performance metric
Year: 2008

Informative benchmarks are crucial for optimizing the page segmentation step of
an OCR system, frequently the performance limiting step for overall OCR system
performance. We show that current evaluation scores are insufficient for
diagnosing specific errors in page segmentation and fail to identify some
classes of serious segmentation errors altogether. This paper introduces a
vectorial score that is sensitive to, and identifies, the most important classes
of segmentation errors (over, under, and mis-segmentation) and what page
components (lines, blocks, etc.) are affected. Unlike previous schemes, our
evaluation method has a canonical representation of ground-truth data and
guarantees pixel-accurate evaluation results for arbitrary region shapes. We
present the results of evaluating widely used segmentation algorithms (x-y cut,
smearing, whitespace analysis, constrained text-line finding, docstrum, and
Voronoi) on the UW-III database and demonstrate that the new evaluation scheme
permits the identification of several specific flaws in individual segmentation
methods.

---

## Structural Mixtures for Statistical Layout Analysis

Faisal Shafait, Joost van Beusekom, Daniel Keysers, Thomas M. Breuel

Category: ocr
Keywords: layout analysis, structural mixtures, probabilistic matching, document digitization, MARG dataset
Year: 2008

A key limitation of current layout analysis methods is that they rely on many
hard-coded assumptions about document layouts and cannot adapt to new layouts
for which the underlying assumptions are not satisfied. Another major drawback
of these approaches is that they do not return confidence scores for their
outputs. These problems pose major challenges in large-scale digitization
efforts where a large number of different layouts need to be handled, and manual
inspection of the results on each individual page is not feasible. This paper
presents a novel statistical approach to layout analysis that aims at solving
the above-mentioned problems for Manhattan layouts. The presented approach
models known page layouts as a structural mixture model. A probabilistic
matching algorithm is presented that gives multiple interpretations of input
layout with associated probabilities. First experiments on documents from the
publicly available MARG dataset achieved below 5% error rate for geometric
layout analysis.

---

## Document cleanup using page frame detection

Faisal Shafait, Joost van Beusekom, Daniel Keysers, Thomas M. Breuel

Category: ocr
Keywords: Document analysis, Marginal noise removal, Document pre-processing
Year: 2008

When a page of a book is scanned or photocopied, textual noise (extraneous
symbols from the neighboring page) and/or non-textual noise (black borders,
speckles, …) appear along the border of the document. Existing document analysis
methods can handle non-textual noise reasonably well, whereas textual noise
still presents a major issue for document analysis systems. Textual noise may
result in undesired text in optical character recognition (OCR) output that
needs to be removed afterwards. Existing document cleanup methods try to
explicitly detect and remove marginal noise. This paper presents a new
perspective for document image cleanup by detecting the page frame of the
document. The goal of page frame detection is to find the actual page contents
area, ignoring marginal noise along the page border. We use a geometric matching
algorithm to find the optimal page frame of structured documents (journal
articles, books, magazines) by exploiting their text alignment property. We
evaluate the algorithm on the UW-III database. The results show that the error
rates are below 4% for each of the performance measures used. Further tests were
run on a dataset of magazine pages and on a set of camera captured document
images. To demonstrate the benefits of using page frame detection in practical
applications, we choose OCR and layout-based document image retrieval as sample
applications. Experiments using a commercial OCR system show that by removing
characters outside the computed page frame, the OCR error rate is reduced from
4.3 to 1.7% on the UW-III dataset. The use of page frame detection in layout-
based document image retrieval application decreases the retrieval error rates
by 30%.

---

## Document Image Dewarping Contest

Faisal Shafait, Thomas M. Breuel

Category: ocr
Keywords: dewarping, document analysis, hand-held cameras, OCR, page dewarping
Year: 2008

Dewarping of documents captured with hand-held cameras in an uncontrolled
environment has triggered a lot of interest in the scientific community over the
last few years and many approaches have been proposed. However, there has been
no comparative evaluation of different dewarping techniques so far. In an
attempt to fill this gap, we have organized a page dewarping contest along with
CBDAR 2007. We have created a dataset of 102 documents captured with a hand-held
camera and have made it freely available online. We have prepared text-line,
text-zone, and ASCII text ground-truth for the documents in this dataset. Three
groups participated in the contest with their methods. In this paper we present
an overview of the approaches that the participants used, the evaluation
measure, and the dataset used in the contest. We report the performance of all
participating methods. The evaluation shows that none of the participating
methods was statistically significantly better than any other participating
method.

---

## Document Image Dewarping Contest

Faisal Shafait, Thomas M. Breuel

Category: ocr
Keywords: document dewarping, hand-held cameras, camera-captured documents, page dewarping contest, document analysis
Year: 2007

Dewarping of documents captured with hand-held cameras in an uncontrolled
environment has triggered a lot of interest in the scientific community over the
last few years and many approaches have been proposed. However, there has been
no comparative evaluation of different dewarping techniques so far. In an
attempt to fill this gap, we have organized a page dewarping contest along with
CBDAR 2007. We have created a dataset of 102 documents captured with a hand-held
camera and have made it freely available online. We have prepared text-line,
text-zone, and ASCII text ground-truth for the documents in this dataset. Three
groups participated in the contest with their methods. In this paper we present
an overview of the approaches that the participants used, the evaluation
measure, and the dataset used in the contest. We report the performance of all
participating methods. The evaluation shows that none of the participating
methods was statistically significantly better than any other participating
method.

---

## The OCRopus Open Source OCR System

Thomas M. Breuel

Category: ocr
Keywords: OCRopus, open source, optical character recognition, layout analysis, text line recognition, multi-lingual, multi-script
Year: 2007

OCRopus is a new, open source OCR system emphasizing modularity, easy
extensibility, and reuse, aimed at both the research community and large scale
commercial document conversions. This paper describes the current status of the
system, its general architecture, as well as the major algorithms currently
being used for layout analysis and text line recognition.

---

## Document Image Dewarping Contest

Faisal Shafait, Thomas M. Breuel

Category: ocr
Keywords: document dewarping, camera-captured documents, OCR, CBDAR 2007, document analysis
Year: 2007

Dewarping of documents captured with hand-held cameras in an uncontrolled
environment has triggered a lot of interest in the scientific community over the
last few years and many approaches have been proposed. However, there has been
no comparative evaluation of different dewarping techniques so far. In an
attempt to fill this gap, we have organized a page dewarping contest along with
CBDAR 2007. We have created a dataset of 102 documents captured with a hand-held
camera and have made it freely available online. We have prepared text-line,
text-zone, and ASCII text ground-truth for the documents in this dataset. Three
groups participated in the contest with their methods. In this paper we present
an overview of the approaches that the participants used, the evaluation
measure, and the dataset used in the contest. We report the performance of all
participating methods. The evaluation shows that none of the participating
methods was statistically significantly better than any other participating
method.

---

## An Overview of the Tesseract OCR Engine

Ray Smith

Category: ocr
Keywords: Tesseract, OCR, line finding, classification methods, adaptive classifier
Year: 2006

The Tesseract OCR engine, as was the HP Research Prototype in the UNLV Fourth
Annual Test of OCR Accuracy, is described in a comprehensive overview. Emphasis
is placed on aspects that are novel or at least unusual in an OCR engine,
including in particular the line finding, features/classification methods, and
the adaptive classifier.

---

## Document Image Dewarping using Robust Estimation of Curled Text Lines

Adrian Ulges, Christoph H. Lampert, Thomas M. Breuel

Category: ocr
Keywords: document image dewarping, curled text lines, perspective distortion, page curl, OCR
Year: 2005

Digital cameras have become almost ubiquitous and their use for fast and casual
capturing of natural images is unchallenged. For making images of documents,
however, they have not caught up to flatbed scanners yet, mainly because camera
images tend to suffer from distortion due to the perspective and are therefore
limited in their further use for archival or OCR. For images of non-planar paper
surfaces like books, page curl causes additional distortion, which poses an even
greater problem due to its nonlinearity. This paper presents a new algorithm for
removing both perspective and page curl distortion. It requires only a single
camera image as input and relies on a priori layout information instead of
additional hardware. Therefore, it is much more user friendly than most previous
approaches, and allows for flexible ad hoc document capture. Results are
presented showing that the algorithm produces visually pleasing output and
increases OCR accuracy, thus having the potential to become a general purpose
preprocessing tool for camera based document capture.

---

## Document Capture using Stereo Vision

Adrian Ulges, Christoph H. Lampert, Thomas Breuel

Category: ocr
Keywords: Stereo Vision, Camera-Based Document Capture, Dewarping
Year: 2004

Capturing images of documents using handheld digital cameras has a variety of
applications in academia, research, knowledge management, retail, and office
settings. The ultimate goal of such systems is to achieve image quality
comparable to that currently achieved with flatbed scanners even for curved,
warped, or curled pages. This can be achieved by high-accuracy 3D modeling of
the page surface, followed by a “flattening” of the surface. A number of
previous systems have either assumed only perspective distortions, or used
techniques like structured lighting, shading, or side-imaging for obtaining 3D
shape. This paper describes a system for handheld camera-based document capture
using general purpose stereo vision methods followed by a new document dewarping
technique. Examples of shape modeling and dewarping of book images is shown.

---

## Classification using a Hierarchical Bayesian Approach

Charles Mathis, Thomas Breuel

Category: ocr
Keywords: Hierarchical Bayesian methods, OCR, classification, fonts, EM algorithm
Year: 2002

A key problem faced by classifiers is coping with styles not represented in the
training set. We present an application of hierarchical Bayesian methods to the
problem of recognizing degraded printed characters in a variety of fonts. The
proposed method works by using training data of various styles and classes to
compute prior distributions on the parameters for the class conditional
distributions. For classification, the parameters for the actual class
conditional distributions are fitted using an EM algorithm. The advantage of
hierarchical Bayesian methods is motivated with a theoretical example. Several-
fold increases in classification performance relative to style-oblivious and
style-conscious methods are demonstrated on a multifont OCR task.