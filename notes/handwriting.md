
## A Review of Various Line Segmentation Techniques Used in Handwritten Character Recognition

Solley Joseph, Jossy George

Category: handwriting
Keywords: Segmentation, OCR, Pattern recognition, Line segmentation, Word segmentation, Character segmentation, Touching lines segmentation, Overlapping lines segmentation
Year: 2023

Segmentation is a very critical stage in the character recognition process as
the performance of any character recognition system depends heavily on the
accuracy of segmentation. Although segmentation is a well-researched area,
segmentation of handwritten text is still difficult owing to several factors
like skewed and overlapping lines, the presence of touching, broken and degraded
characters, and variations in writing styles. Therefore, researchers in this
area are working continuously to develop new techniques for the efficient
segmentation and recognition of characters. In the character recognition
process, segmentation can be implemented at the line, word, and character level.
Text line segmentation is the first step in the text/character recognition
process. The line segmentation methods used in the character recognition of
handwritten documents are presented in this paper. The various levels of
segmentation which include line, word, and character segmentation are discussed
with a focus on line segmentation.

---

## DAN: a Segmentation-free Document Attention Network for Handwritten Document Recognition

Denis Coquenet, Clément Chatelain, Thierry Paquet

Category: handwriting
Keywords: Seq2Seq model, Segmentation-free, Handwritten Text Recognition, Transformer, Layout Analysis
Year: 2023

Unconstrained handwritten text recognition is a challenging computer vision
task. It is traditionally handled by a two-step approach, combining line
segmentation followed by text line recognition. For the first time, we propose
an end-to-end segmentation-free architecture for the task of handwritten
document recognition: the Document Attention Network. In addition to text
recognition, the model is trained to label text parts using begin and end tags
in an XML-like fashion. This model is made up of an FCN encoder for feature
extraction and a stack of transformer decoder layers for a recurrent token-by-
token prediction process. It takes whole text documents as input and
sequentially outputs characters, as well as logical layout tokens. Contrary to
the existing segmentation-based approaches, the model is trained without using
any segmentation label. We achieve competitive results on the READ 2016 dataset
at page level, as well as double-page level with a CER of 3.43% and 3.70%,
respectively. We also provide results for the RIMES 2009 dataset at page level,
reaching 4.54% of CER. We provide all source code and pre-trained model weights
at https://github.com/FactoDeepLearning/DAN.

---

## DAN: A Segmentation-Free Document Attention Network for Handwritten Document Recognition

Denis Coquenet, Clément Chatelain, Thierry Paquet

Category: handwriting
Keywords: Handwritten text recognition, layout analysis, segmentation-free, Seq2Seq model, transformer
Year: 2023

Unconstrained handwritten text recognition is a challenging computer vision
task. It is traditionally handled by a two-step approach, combining line
segmentation followed by text line recognition. For the first time, we propose
an end-to-end segmentation-free architecture for the task of handwritten
document recognition: the Document Attention Network. In addition to text
recognition, the model is trained to label text parts using begin and end tags
in an XML-like fashion. This model is made up of an FCN encoder for feature
extraction and a stack of transformer decoder layers for a recurrent token-by-
token prediction process. It takes whole text documents as input and
sequentially outputs characters, as well as logical layout tokens. Contrary to
the existing segmentation-based approaches, the model is trained without using
any segmentation label. We achieve competitive results on the READ 2016 dataset
at page level, as well as double-page level with a CER of 3.43% and 3.70%,
respectively. We also provide results for the RIMES 2009 dataset at page level,
reaching 4.54% of CER. We provide all source code and pre-trained model weights
at https://github.com/FactoDeepLearning/DAN.

---

## DAN: A Segmentation-Free Document Attention Network for Handwritten Document Recognition

Denis Coquenet, Clément Chatelain, Thierry Paquet

Category: handwriting
Keywords: Handwritten text recognition, layout analysis, segmentation-free, Seq2Seq model, transformer
Year: 2023

Unconstrained handwritten text recognition is a challenging computer vision
task. It is traditionally handled by a two-step approach, combining line
segmentation followed by text line recognition. For the first time, we propose
an end-to-end segmentation-free architecture for the task of handwritten
document recognition: the Document Attention Network. In addition to text
recognition, the model is trained to label text parts using begin and end tags
in an XML-like fashion. This model is made up of an FCN encoder for feature
extraction and a stack of transformer decoder layers for a recurrent token-by-
token prediction process. It takes whole text documents as input and
sequentially outputs characters, as well as logical layout tokens. Contrary to
the existing segmentation-based approaches, the model is trained without using
any segmentation label. We achieve competitive results on the READ 2016 dataset
at page level, as well as double-page level with a CER of 3.43% and 3.70%,
respectively. We also provide results for the RIMES 2009 dataset at page level,
reaching 4.54% of CER. We provide all source code and pre-trained model weights
at https://github.com/FactoDeepLearning/DAN.

---

## Rescoring Sequence-to-Sequence Models for Text Line Recognition with CTC-Prefixes

Christoph Wick, Jochen Zöllner, Tobias Grüning

Category: handwriting
Keywords: Text Line Recognition, Handwritten Text Recognition, Document Analysis, Sequence-To-Sequence, CTC
Year: 2022

In contrast to Connectionist Temporal Classification (CTC) approaches, Sequence-
To-Sequence (S2S) models for Handwritten Text Recognition (HTR) suffer from
errors such as skipped or repeated words which often occur at the end of a
sequence. In this paper, to combine the best of both approaches, we propose to
use the CTC-Prefix-Score during S2S decoding. Hereby, during beam search, paths
that are invalid according to the CTC confidence matrix are penalised. Our
network architecture is composed of a Convolutional Neural Network (CNN) as
visual backbone, bidirectional Long-Short-Term-Memory-Cells (LSTMs) as encoder,
and a decoder which is a Transformer with inserted mutual attention layers. The
CTC confidences are computed on the encoder while the Transformer is only used
for character-wise S2S decoding. We evaluate this setup on three HTR data sets:
IAM, Rimes, and StAZH. On IAM, we achieve a competitive Character Error Rate
(CER) of 2.95% when pretraining our model on synthetic data and including a
character-based language model for contemporary English. Compared to other
state-of-the-art approaches, our model requires about 10-20 times less
parameters.

---

## Transformer-based HTR for Historical Documents

Phillip Benjamin Ströbel, Simon Clematide, Martin Volk, Tobias Hodel

Category: handwriting
Keywords: Handwritten Text Recognition, Transformers, Historical Documents, Digitization, Neural Networks
Year: 2022

Handwritten Text Recognition (HTR) has become a valuable tool to extract text
from scanned documents. The current digitization wave in libraries and archives
does not stop at historical manuscripts. HTR plays an essential role in making
the contents of manuscripts available to researchers and the public. HTR has
undergone significant improvements in recent years, thanks in large part to the
introduction of neural network-based techniques. Platforms like Transkribus have
successfully integrated these approaches in a way that its HTR+ model can
achieve character error rates (CERs) of below 5% with little annotated ground
truth material. However, large manuscript collections pose significant
challenges to libraries and archives, especially because of the variety of
handwriting styles. Transformer-based architectures have proven suitable to
build large language representation models and recently found their way into
image processing, driving the development of image transformers.

---

## Transformer-based HTR for Historical Documents

Phillip Benjamin Ströbel, Simon Clematide, Martin Volk, Tobias Hodel

Category: handwriting
Keywords: Handwritten Text Recognition, HTR, Transformer, Historical Documents, Digitisation, Neural Networks, BEiT, RoBERTa, TrOCR
Year: 2022

Handwritten Text Recognition (HTR) has become a valuable tool to extract text
from scanned documents. The current digitisation wave in libraries and archives
includes historical manuscripts, making HTR essential for making these contents
accessible to researchers and the public. Recent advances in HTR have been
driven by neural network-based techniques, achieving low character error rates
with minimal annotated ground truth. However, large manuscript collections, with
their variety of handwriting styles, pose significant challenges. Transformer-
based architectures, such as TrOCR, which combines the BERT-style vision
transformer BEiT with a RoBERTa language representation model, offer promising
solutions by adapting well to different handwriting styles with little training
data.

---

## Pay attention to what you read: Non-recurrent handwritten text-Line recognition

Lei Kang, Pau Riba, Marçal Rusiñol, Alicia Fornés, Mauricio Villegas

Category: handwriting
Keywords: Handwriting text recognition, Transformers, Self-Attention, Implicit language model
Year: 2022

The advent of recurrent neural networks for handwriting recognition marked an
important milestone reaching impressive recognition accuracies despite the great
variability that we observe across different writing styles. Sequential
architectures are a perfect fit to model text lines, not only because of the
inherent temporal aspect of text, but also to learn probability distributions
over sequences of characters and words. However, using such recurrent paradigms
comes at a cost at training stage, since their sequential pipelines prevent
parallelization. In this work, we introduce a novel method that bypasses any
recurrence during the training process with the use of transformer models. By
using multi-head self-attention layers both at the visual and textual stages, we
are able to tackle character recognition as well as to learn language-related
dependencies of the character sequences to be decoded. Our model is
unconstrained to any predefined vocabulary, being able to recognize out-of-
vocabulary words, i.e. words that do not appear in the training vocabulary. We
significantly advance over prior art and demonstrate that satisfactory
recognition accuracies are yielded even in few-shot learning scenarios.

---

## Transformer-based HTR for Historical Documents

Phillip Benjamin Ströbel, Simon Clematide, Martin Volk, Tobias Hodel

Category: handwriting
Keywords: Handwritten Text Recognition, neural networks, transformers, historical documents, digitisation
Year: 2022

Handwritten Text Recognition (HTR) has become a valuable tool to extract text
from scanned documents. The current digitisation wave in libraries and archives
includes historical manuscripts, making HTR essential for making their contents
available. HTR has improved significantly with neural network-based techniques.
Platforms like Transkribus have integrated these approaches to achieve low
character error rates with minimal annotated ground truth material. Despite
these advancements, large manuscript collections remain challenging due to
handwriting variability. Transformer-based architectures, like BERT, have been
effective for language representation and are adaptable for various tasks.
Recent developments in image processing have led to image transformers. This
paper explores the use of TrOCR, which combines a vision transformer with a
language representation model, for historical document HTR.

---

## Rescoring Sequence-to-Sequence Models for Text Line Recognition with CTC-Preﬁxes

Christoph Wick, Jochen Zöllner, Tobias Grüning

Category: handwriting
Keywords: Text Line Recognition, Handwritten Text Recognition, Document Analysis, Sequence-To-Sequence, CTC
Year: 2022

In contrast to Connectionist Temporal Classification (CTC) approaches, Sequence-
To-Sequence (S2S) models for Handwritten Text Recognition (HTR) suffer from
errors such as skipped or repeated words which often occur at the end of a
sequence. In this paper, to combine the best of both approaches, we propose to
use the CTC-Prefix-Score during S2S decoding. Hereby, during beam search, paths
that are invalid according to the CTC confidence matrix are penalized. Our
network architecture is composed of a Convolutional Neural Network (CNN) as
visual backbone, bidirectional Long-Short-Term-Memory-Cells (LSTMs) as encoder,
and a decoder which is a Transformer with inserted mutual attention layers. The
CTC confidences are computed on the encoder while the Transformer is only used
for character-wise S2S decoding. We evaluate this setup on three HTR data sets:
IAM, Rimes, and StAZH. On IAM, we achieve a competitive Character Error Rate
(CER) of 2.95% when pretraining our model on synthetic data and including a
character-based language model for contemporary English. Compared to other
state-of-the-art approaches, our model requires about 10-20 times less
parameters.

---

## Transformer-based HTR for Historical Documents

Phillip Benjamin Ströbel, Simon Clematide, Martin Volk, Tobias Hodel

Category: handwriting
Keywords: Handwritten Text Recognition, HTR, Transformers, Historical Documents, Digitisation
Year: 2022

Handwritten Text Recognition (HTR) has become a valuable tool to extract text
from scanned documents. The current digitisation wave in libraries and archives
includes historical manuscripts, making HTR essential for making these contents
available to researchers and the public. HTR has significantly improved with
neural network-based techniques, such as those used by platforms like
Transkribus. However, large manuscript collections with varied handwriting
styles present significant challenges. Transformer-based architectures, such as
BERT and its adaptations, have shown promise in adapting to different
handwritings with minimal training data. This paper explores the use of the
TrOCR model, which combines a vision transformer with a language representation
model, for HTR tasks.

---

## Transformer for Handwritten Text Recognition Using Bidirectional Post-decoding

Christoph Wick, Jochen Zöllner, Tobias Grüning

Category: handwriting
Keywords: Handwritten Text Recognition, Transformer, Bidirectional
Year: 2021

Most recently, Transformers – which are recurrent-free neural network
architectures – achieved tremendous performances on various Natural Language
Processing (NLP) tasks. Since Transformers represent a traditional Sequence-To-
Sequence (S2S)-approach they can be used for several different tasks such as
Handwritten Text Recognition (HTR). In this paper, we propose a bidirectional
Transformer architecture for line-based HTR that is composed of a Convolutional
Neural Network (CNN) for feature extraction and a Transformer-based
encoder/decoder, whereby the decoding is performed in reading-order direction
and reversed. A voter combines the two predicted sequences to obtain a single
result. Our network performed worse compared to a traditional Connectionist
Temporal Classification (CTC) approach on the IAM-dataset but reduced the state-
of-the-art of Transformers-based approaches by about 25% without using
additional data. On a significantly larger dataset, the proposed Transformer
significantly outperformed our reference model by about 26%. In an error
analysis, we show that the Transformer is able to learn a strong language model
which explains why a larger training dataset is required to outperform
traditional approaches and discuss why Transformers should be used with caution
for HTR due to several shortcomings such as repetitions in the text.

---

## End-to-end Handwritten Paragraph Text Recognition Using a Vertical Attention Network

Denis Coquenet, Clément Chatelain, Thierry Paquet

Category: handwriting
Keywords: Seq2Seq model, Hybrid attention, Segmentation-free, Paragraph handwriting recognition, Fully Convolutional Network, Encoder-decoder, Optical Character Recognition
Year: 2021

Unconstrained handwritten text recognition remains challenging for computer
vision systems. Paragraph text recognition is traditionally achieved by two
models: the first one for line segmentation and the second one for text line
recognition. We propose a unified end-to-end model using hybrid attention to
tackle this task. This model is designed to iteratively process a paragraph
image line by line. It can be split into three modules. An encoder generates
feature maps from the whole paragraph image. Then, an attention module
recurrently generates a vertical weighted mask enabling focus on the current
text line features. This way, it performs a kind of implicit line segmentation.
For each text line feature, a decoder module recognizes the character sequence
associated, leading to the recognition of a whole paragraph. We achieve state-
of-the-art character error rate at paragraph level on three popular datasets:
1.91% for RIMES, 4.45% for IAM and 3.59% for READ 2016. Our code and trained
model weights are available at
https://github.com/FactoDeepLearning/VerticalAttentionOCR.

---

## Pay Attention to What You Read: Non-recurrent Handwritten Text-Line Recognition

Lei Kang, Pau Riba, Marçal Rusiñol, Alicia Fornés, Mauricio Villegas

Category: handwriting
Keywords: handwritten text recognition, transformer models, non-recurrent, self-attention, out-of-vocabulary
Year: 2020

The advent of recurrent neural networks for handwriting recognition marked an
important milestone reaching impressive recognition accuracies despite the great
variability that we observe across different writing styles. Sequential
architectures are a perfect fit to model text lines, not only because of the
inherent temporal aspect of text, but also to learn probability distributions
over sequences of characters and words. However, using such recurrent paradigms
comes at a cost at training stage, since their sequential pipelines prevent
parallelization. In this work, we introduce a non-recurrent approach to
recognize handwritten text by the use of transformer models. We propose a novel
method that bypasses any recurrence. By using multi-head self-attention layers
both at the visual and textual stages, we are able to tackle character
recognition as well as to learn language-related dependencies of the character
sequences to be decoded. Our model is unconstrained to any predefined
vocabulary, being able to recognize out-of-vocabulary words, i.e. words that do
not appear in the training vocabulary. We significantly advance over prior art
and demonstrate that satisfactory recognition accuracies are yielded even in
few-shot learning scenarios.

---

## Evaluating Sequence-to-Sequence Models for Handwritten Text Recognition

Johannes Michael, Roger Labahn, Tobias Grüning, Jochen Zöllner

Category: handwriting
Keywords: sequence-to-sequence, Seq2Seq, encoder-decoder, attention, handwritten text recognition, HTR
Year: 2019

Encoder-decoder models have become an effective approach for sequence learning
tasks like machine translation, image captioning, and speech recognition, but
have yet to show competitive results for handwritten text recognition. To this
end, we propose an attention-based sequence-to-sequence model. It combines a
convolutional neural network as a generic feature extractor with a recurrent
neural network to encode both the visual information, as well as the temporal
context between characters in the input image, and uses a separate recurrent
neural network to decode the actual character sequence. We make experimental
comparisons between various attention mechanisms and positional encodings, in
order to find an appropriate alignment between the input and output sequence.
The model can be trained end-to-end and the optional integration of a hybrid
loss allows the encoder to retain an interpretable and usable output, if
desired. We achieve competitive results on the IAM and ICFHR2016 READ data sets
compared to the state-of-the-art without the use of a language model, and we
significantly improve over any recent sequence-to-sequence approaches.

---

## Start, Follow, Read: End-to-End Full-Page Handwriting Recognition

Curtis Wigington, Chris Tensmeyer, Brian Davis, William Barrett, Brian Price, Scott Cohen

Category: handwriting
Keywords: Handwriting Recognition, Document Analysis, Historical Document Processing, Text Detection, Text Line Segmentation
Year: 2018

Despite decades of research, offline handwriting recognition (HWR) of degraded
historical documents remains a challenging problem, which if solved could
greatly improve the searchability of online cultural heritage archives. HWR
models are often limited by the accuracy of the preceding steps of text
detection and segmentation. Motivated by this, we present a deep learning model
that jointly learns text detection, segmentation, and recognition using mostly
images without detection or segmentation annotations. Our Start, Follow, Read
(SFR) model is composed of a Region Proposal Network to find the start position
of text lines, a novel line follower network that incrementally follows and
preprocesses lines of (perhaps curved) text into dewarped images suitable for
recognition by a CNN-LSTM network. SFR exceeds the performance of the winner of
the ICDAR2017 handwriting recognition competition, even when not using the
provided competition region annotations.

---

## Fully Convolutional Network with Dilated Convolutions for Handwritten Text Line Segmentation

Guillaume Renton, Yann Soullard, Clément Chatelain, Sébastien Adam, Christopher Kermorvant, Thierry Paquet

Category: handwriting
Keywords: handwritten text line segmentation, fully convolutional networks, dilated convolutions, X-height labeling, text recognition
Year: 2018

We present a learning-based method for handwritten text line segmentation in
document images. Our approach relies on a variant of deep fully convolutional
networks (FCNs) with dilated convolutions. Dilated convolutions allow to never
reduce the input resolution and produce a pixel-level labeling. The FCN is
trained to identify X-height labeling as text line representation, which has
many advantages for text recognition. We show that our approach outperforms the
most popular variants of FCN, based on deconvolution or unpooling layers, on a
public dataset. We also provide results investigating various settings, and we
conclude with a comparison of our model with recent approaches defined as part
of the cBAD international competition, leading us to a 91.3% F-measure.

---

## Start, Follow, Read: End-to-End Full-Page Handwriting Recognition

Curtis Wigington, Chris Tensmeyer, Brian Davis, William Barrett, Brian Price, Scott Cohen

Category: handwriting
Keywords: Handwriting Recognition, Document Analysis, Historical Document Processing, Text Detection, Text Line Segmentation
Year: 2018

Despite decades of research, offline handwriting recognition (HWR) of degraded
historical documents remains a challenging problem, which if solved could
greatly improve the searchability of online cultural heritage archives. HWR
models are often limited by the accuracy of the preceding steps of text
detection and segmentation. Motivated by this, we present a deep learning model
that jointly learns text detection, segmentation, and recognition using mostly
images without detection or segmentation annotations. Our Start, Follow, Read
(SFR) model is composed of a Region Proposal Network to find the start position
of text lines, a novel line follower network that incrementally follows and
preprocesses lines of (perhaps curved) text into dewarped images suitable for
recognition by a CNN-LSTM network. SFR exceeds the performance of the winner of
the ICDAR2017 handwriting recognition competition, even when not using the
provided competition region annotations.

---

## Gated Convolutional Recurrent Neural Networks for Multilingual Handwriting Recognition

Théodore Bluche, Ronaldo Messina

Category: handwriting
Keywords: handwriting recognition, neural networks, convolutional encoder, bidirectional LSTM, multilingual
Year: 2017

In this paper, we propose a new neural network architecture for state-of-the-art
handwriting recognition, alternative to multi-dimensional long short-term memory
(MD-LSTM) recurrent neural networks. The model is based on a convolutional
encoder of the input images, and a bidirectional LSTM decoder predicting
character sequences. In this paradigm, we aim at producing generic, multilingual
and reusable features with the convolutional encoder, leveraging more data for
transfer learning. The architecture is also motivated by the need for a fast
training on GPUs, and the requirement of a fast decoding on CPUs. The main
contribution of this paper lies in the convolutional gates in the encoder,
enabling hierarchical context-sensitive feature extraction. The experiments on a
large benchmark including seven languages show a consistent and significant
improvement of the proposed approach over our previous production systems. We
also report state-of-the-art results on line and paragraph level recognition on
the IAM and Rimes databases.

---

## Gated Convolutional Recurrent Neural Networks for Multilingual Handwriting Recognition

Théodore Bluche, Ronaldo Messina

Category: handwriting
Keywords: Handwriting Recognition, Convolutional Neural Networks, Recurrent Neural Networks, Multilingual, Transfer Learning
Year: 2017

In this paper, we propose a new neural network architecture for state-of-the-art
handwriting recognition, alternative to multi-dimensional long short-term memory
(MD-LSTM) recurrent neural networks. The model is based on a convolutional
encoder of the input images, and a bidirectional LSTM decoder predicting
character sequences. In this paradigm, we aim at producing generic, multilingual
and reusable features with the convolutional encoder, leveraging more data for
transfer learning. The architecture is also motivated by the need for a fast
training on GPUs, and the requirement of a fast decoding on CPUs. The main
contribution of this paper lies in the convolutional gates in the encoder,
enabling hierarchical context-sensitive feature extraction. The experiments on a
large benchmark including seven languages show a consistent and significant
improvement of the proposed approach over our previous production systems. We
also report state-of-the-art results on line and paragraph level recognition on
the IAM and Rimes databases.

---

## Generating Synthetic Data for Text Recognition

Praveen Krishnan, C.V. Jawahar

Category: handwriting
Keywords: synthetic data, handwritten text recognition, data augmentation, deep learning, word spotting
Year: 2016

Generating synthetic images is an art which emulates the natural process of
image generation in a closest possible manner. In this work, we exploit such a
framework for data generation in handwritten domain. We render synthetic data
using open source fonts and incorporate data augmentation schemes. As part of
this work, we release 9M synthetic handwritten word image corpus which could be
useful for training deep network architectures and advancing the performance in
handwritten word spotting and recognition tasks.

---

## Joint Line Segmentation and Transcription for End-to-End Handwritten Paragraph Recognition

Théodore Bluche

Category: handwriting
Keywords: handwriting recognition, MDLSTM-RNN, end-to-end processing, line segmentation, attention weights
Year: 2016

Offline handwriting recognition systems require cropped text line images for
both training and recognition. On the one hand, the annotation of position and
transcript at line level is costly to obtain. On the other hand, automatic line
segmentation algorithms are prone to errors, compromising the subsequent
recognition. In this paper, we propose a modification of the popular and
efficient Multi-Dimensional Long Short-Term Memory Recurrent Neural Networks
(MDLSTM-RNNs) to enable end-to-end processing of handwritten paragraphs. More
particularly, we replace the collapse layer transforming the two-dimensional
representation into a sequence of predictions by a recurrent version which can
select one line at a time. In the proposed model, a neural network performs a
kind of implicit line segmentation by computing attention weights on the image
representation. The experiments on paragraphs of Rimes and IAM databases yield
results that are competitive with those of networks trained at line level, and
constitute a significant step towards end-to-end transcription of full
documents.

---

## Segmentation-free Handwritten Chinese Text Recognition with LSTM-RNN

Ronaldo Messina, Jerome Louradour

Category: handwriting
Keywords: MDLSTM-RNN, Chinese text recognition, segmentation-free, character recognition, CTC
Year: 2015

We present initial results on the use of Multi-Dimensional Long-Short Term
Memory Recurrent Neural Networks (MDLSTM-RNN) in recognizing lines of
handwritten Chinese text without explicit segmentation of the characters. In
fact, most of Chinese text recognizers in the literature perform a pre-
segmentation of text image into characters. This can be a drawback, as explicit
segmentation is an extra step before recognizing the text, and the errors made
at this stage have direct impact on the performance of the whole system. MDLSTM-
RNN is now a state-of-the-art technology that provides the best performance on
languages with Latin and Arabic characters, hence we propose to apply RNN on
Chinese text recognition. Our results on the data from the Task 4 in ICDAR 2013
competition for handwritten Chinese recognition are comparable in performance
with the best reported systems.

---

## Spatially-sparse convolutional neural networks

Benjamin Graham

Category: handwriting
Keywords: online character recognition, convolutional neural network, sparsity, computer vision
Year: 2014

Convolutional neural networks (CNNs) perform well on problems such as
handwriting recognition and image classification. However, the performance of
the networks is often limited by budget and time constraints, particularly when
trying to train deep networks. Motivated by the problem of online handwriting
recognition, we developed a CNN for processing spatially-sparse inputs; a
character drawn with a one-pixel wide pen on a high resolution grid looks like a
sparse matrix. Taking advantage of the sparsity allowed us more efficiently to
train and test large, deep CNNs. On the CASIA-OLHWDB1.1 dataset containing 3755
character classes we get a test error of 3.82%. Although pictures are not
sparse, they can be thought of as sparse by adding padding. Applying a deep
convolutional network using sparsity has resulted in a substantial reduction in
test error on the CIFAR small picture datasets: 6.28% on CIFAR-10 and 24.30% for
CIFAR-100.

---

## Generating Sequences With Recurrent Neural Networks

Alex Graves

Category: handwriting
Keywords: Recurrent Neural Networks, Long Short-term Memory, Sequence Generation, Handwriting Synthesis, Text Prediction
Year: 2014

This paper shows how Long Short-term Memory recurrent neural networks can be
used to generate complex sequences with long-range structure, simply by
predicting one data point at a time. The approach is demonstrated for text
(where the data are discrete) and online handwriting (where the data are real-
valued). It is then extended to handwriting synthesis by allowing the network to
condition its predictions on a text sequence. The resulting system is able to
generate highly realistic cursive handwriting in a wide variety of styles.

---

## TANDEM HMM WITH CONVOLUTIONAL NEURAL NETWORK FOR HANDWRITTEN WORD RECOGNITION

Théodore Bluche, Hermann Ney, Christopher Kermorvant

Category: handwriting
Keywords: Handwriting recognition, Hidden Markov Model, Convolutional Neural Network
Year: 2013

In this paper, we investigate the combination of hidden Markov models and
convolutional neural networks for handwritten word recognition. The
convolutional neural networks have been successfully applied to various computer
vision tasks, including handwritten character recognition. In this work, we show
that they can replace Gaussian mixtures to compute emission probabilities in
hidden Markov models (hybrid combination), or serve as feature extractor for a
standard Gaussian HMM system (tandem combination). The proposed systems
outperform a basic HMM based on either decorrelated pixels or handcrafted
features. We validated the approach on two publicly available databases, and we
report up to 60% (Rimes) and 35% (IAM) relative improvement compared to a
Gaussian HMM based on pixel values. The final systems give comparable results to
recurrent neural networks, which are the best systems since 2009.

---

## Results of the RIMES evaluation campaign for handwritten mail processing

Emmanuèle Grosicki, Matthieu Carré, Jean-Marie Brodin, Edouard Geoffrois

Category: handwriting
Keywords: Evaluation, database, layout analysis, handwriting recognition, writer identification, metric
Year: 2009

This paper presents the results of the second test phase of the RIMES evaluation
campaign. The latter is the first large-scale evaluation campaign intended for
all the key players of the handwritten recognition and document analysis
communities. It proposes various tasks around recognition and indexing of
handwritten letters such as those sent by postal mail or fax by individuals to
companies or administrations. In this second evaluation test, automatic systems
have been evaluated on three themes: layout analysis, handwriting recognition,
and writer identification. The databases used are part of the RIMES database of
5605 real mails completely annotated, as well as secondary databases of isolated
characters and handwritten words (250,000 snippets). The paper reports on
protocols and gives the results obtained in the campaign.

---

## Off-line Handwriting Text Line Segmentation: A Review

Zaidi Razak, Khansa Zulkiflee, Mohd Yamani Idna Idris, Emran Mohd Tamil, Mohd Noorzaily Mohamed Noor, Rosli Salleh, Mohd Yaakob @ Zulkifli Mohd Yusof, Mashkuri Yaacob

Category: handwriting
Keywords: Off-line handwriting recognition, text line segmentation
Year: 2008

Text line segmentation is an essential pre-processing stage for off-line
handwriting recognition in many Optical Character Recognition (OCR) systems. It
is an important step because inaccurately segmented text lines will cause errors
in the recognition stage. Text line segmentation of handwritten documents is
still one of the most complicated problems in developing a reliable OCR. The
nature of handwriting makes the process of text line segmentation very
challenging. Several techniques to segment handwriting text line have been
proposed in the past. This paper seeks to provide a comprehensive review of the
methods of off-line handwriting text line segmentation proposed by researchers.

---

## A Novel Approach to On-Line Handwriting Recognition Based on Bidirectional Long Short-Term Memory Networks

Marcus Liwicki, Alex Graves, Horst Bunke, Jürgen Schmidhuber

Category: handwriting
Keywords: handwriting recognition, bidirectional LSTM, Connectionist Temporal Classification, whiteboard notes, recurrent neural network
Year: 2007

In this paper we introduce a new connectionist approach to on-line handwriting
recognition and address in particular the problem of recognizing handwritten
whiteboard notes. The approach uses a bidirectional recurrent neural network
with long short-term memory blocks. We use a recently introduced objective
function, known as Connectionist Temporal Classification (CTC), that directly
trains the network to label unsegmented sequence data. Our new system achieves a
word recognition rate of 74.0%, compared with 65.4% using a previously developed
HMM-based recognition system.

---

## The IAM-database: an English sentence database for offline handwriting recognition

U.-V. Marti, H. Bunke

Category: handwriting
Keywords: Handwriting recognition, Database, Unconstrained English sentences, Corpus, Linguistic knowledge
Year: 2002

In this paper we describe a database that consists of handwritten English
sentences. It is based on the Lancaster-Oslo/Bergen (LOB) corpus. This corpus is
a collection of texts that comprise about one million word instances. The
database includes 1,066 forms produced by approximately 400 different writers. A
total of 82,227 word instances out of a vocabulary of 10,841 words occur in the
collection. The database consists of full English sentences. It can serve as a
basis for a variety of handwriting recognition tasks. However, it is expected
that the database would be particularly useful for recognition tasks where
linguistic knowledge beyond the lexicon level is used, because this knowledge
can be automatically derived from the underlying corpus. The database also
includes a few image-processing procedures for extracting the handwritten text
from the forms and the segmentation of the text into lines and words.

---

## A System for the Offline Recognition of Handwritten Text

Thomas M. Breuel

Category: handwriting
Keywords: handwritten text recognition, MLP, Viterbi algorithm, language models, census forms
Year: 1995

A new system for the recognition of handwritten text is described. The system
goes from raw binary scanned images of census forms to ASCII transcriptions of
the fields contained within the forms. The first step is to locate and extract
the handwritten input from the forms. Then a large number of character subimages
are extracted and individually classified using a MLP (Multi-Layer Perceptron).
A Viterbi-like algorithm is used to assemble the individual classified character
subimages into optimal interpretations of an input string, taking into account
both the quality of the overall segmentation and the degree to which each
character subimage of the segmentation matches a character model. The system
uses two different statistical language models: one based on a phrase dictionary
and the other based on a simple word grammar. Hypotheses from recognition based
on each language model are integrated using a decision tree classifier. Results
from the application of the system to the recognition of handwritten responses
on U.S. census forms are reported.

---

## Language Modeling for a Real-World Handwriting Recognition Task

Thomas M. Breuel

Category: handwriting
Keywords: handwriting recognition, language modeling, U.S. census forms, real-world handwriting tasks
Year: 1994

The author has developed a system for the recognition of handwritten responses
on 1990 U.S. census forms. This is a typical task for handwriting recognition.
It involves unconstrained, short responses to a set of three questions about the
nature and place of employment of the respondent. A significant fraction of the
responses are of poor writing quality or ambiguous and require the use of
language models for disambiguation, recognition, and/or rejection. The goals of
this paper are to discuss some of the constraints and trade-offs in an existing
real-world handwriting recognition system, to present work already done on
language modeling for the system, and to suggest possible future directions of
research in language modeling for handwriting recognition tasks.

---

## Signature Verification using a "Siamese" Time Delay Neural Network

Jane Bromley, Isabelle Guyon, Yann LeCun, Eduard Sickinger, Roopak Shah

Category: handwriting
Keywords: signature verification, Siamese neural network, pen-input tablet, dynamic verification, feature extraction
Year: 1994

This paper describes an algorithm for verification of signatures written on a
pen-input tablet. The algorithm is based on a novel, artificial neural network,
called a "Siamese" neural network. This network consists of two identical sub-
networks joined at their outputs. During training the two sub-networks extract
features from two signatures, while the joining neuron measures the distance
between the two feature vectors. Verification consists of comparing the
extracted feature vector with a stored feature vector for the signer. Signatures
closer to this stored representation than a chosen threshold are accepted, all
other signatures are rejected as forgeries.

---

## Segmentation of Handprinted Letter Strings using a Dynamic Programming Algorithm

Thomas M. Breuel

Category: handwriting
Keywords: segmentation, handwriting recognition, dynamic programming, CPSC algorithm, Roman alphabets
Year: 1991

Segmentation of handwritten input into individual characters is a crucial step
in many connected handwriting recognition systems. This paper describes a
segmentation algorithm for letters in Roman alphabets, curved pre-stroke cut
(CPSC) segmentation. The CPSC algorithm evaluates a large set of curved cuts
through the image of the input string using dynamic programming and selects a
small 'optimal' subset of cuts for segmentation. It usually generates pixel
accurate segmentations, indistinguishable from characters written in isolation.
At four times oversegmentation, segmentation points are missed with an
undetectable frequency on real-world databases. The CPSC algorithm has been used
as part of a high-performance handwriting recognition system.