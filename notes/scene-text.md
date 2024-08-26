
## Natural Image Text Recognition with a Transformer

Anonymous

Category: scene-text
Keywords: scene text recognition, transformer, self-attention, natural images, benchmark datasets
Year: 2023

Scene text recognition has become an important research area in computer vision.
This paper proposes a new method for recognizing text in natural images using a
transformer-based model. Our approach leverages the self-attention mechanism of
transformers to effectively capture contextual information in images, leading to
improved recognition accuracy. We present experiments on several benchmark
datasets, demonstrating the superiority of our method over existing approaches.
The results indicate that the proposed model achieves state-of-the-art
performance in recognizing text from challenging scenes.
---

## Towards End-to-End Unified Scene Text Detection and Layout Analysis

Shangbang Long, Siyang Qin, Dmitry Panteleev, Alessandro Bissacco, Yasuhisa Fujii, Michalis Raptis

Category: scene-text
Keywords: scene text detection, layout analysis, unified model, hierarchical scene text dataset, text clusters
Year: 2023

Scene text detection and document layout analysis have long been treated as two
separate tasks in different image domains. In this paper, we bring them together
and introduce the task of unified scene text detection and layout analysis. The
first hierarchical scene text dataset is introduced to enable this novel
research task. We also propose a novel method that is able to simultaneously
detect scene text and form text clusters in a unified way. Comprehensive
experiments show that our unified model achieves better performance than
multiple well-designed baseline methods. Additionally, this model achieves
state-of-the-art results on multiple scene text detection datasets without the
need of complex post-processing. Dataset and code: https://github.com/google-
research-datasets/hiertext.
---

## Levenshtein OCR

Cheng Da, Peng Wang, Cong Yao

Category: scene-text
Keywords: Scene text recognition, Transformer, Interpretability
Year: 2022

A novel scene text recognizer based on Vision-Language Transformer (VLT) is
presented. Inspired by Levenshtein Transformer in the area of NLP, the proposed
method (named Levenshtein OCR, and LevOCR for short) explores an alternative way
for automatically transcribing textual content from cropped natural images.
Specifically, we cast the problem of scene text recognition as an iterative
sequence refinement process. The initial prediction sequence produced by a pure
vision model is encoded and fed into a cross-modal transformer to interact and
fuse with the visual features, to progressively approximate the ground truth.
The refinement process is accomplished via two basic character-level operations:
deletion and insertion, which are learned with imitation learning and allow for
parallel decoding, dynamic length change and good interpretability. The
quantitative experiments clearly demonstrate that LevOCR achieves state-of-the-
art performances on standard benchmarks and the qualitative analyses verify the
effectiveness and advantage of the proposed LevOCR algorithm.
---

## Visual-Semantic Transformer for Scene Text Recognition

Xin Tang, Yongquan Lai, Ying Liu, Yuanyuan Fu, Rui Fang

Category: scene-text
Keywords: scene text recognition, transformer, visual-semantic alignment, semantic modeling, text prediction
Year: 2022

Modeling semantic information is helpful for scene text recognition. In this
work, we propose to model semantic and visual information jointly with a Visual-
Semantic Transformer (VST). The VST first explicitly extracts primary semantic
information from visual feature maps with a transformer module and a primary
visual-semantic alignment module. The semantic information is then joined with
the visual feature maps (viewed as a sequence) to form a pseudo multi-domain
sequence combining visual and semantic information, which is subsequently fed
into a transformer-based interaction module to enable learning of interactions
between visual and semantic features. In this way, the visual features can be
enhanced by the semantic information and vice versa. The enhanced version of
visual features are further decoded by a secondary visual-semantic alignment
module which shares weights with the primary one. Finally, the decoded visual
features and the enhanced semantic features are jointly processed by the third
transformer module obtaining the final text prediction. Experiments on seven
public benchmarks including regular/irregular text recognition datasets verifies
the effectiveness of our proposed model, reaching state of the art on four of
the seven benchmarks.
---

## Visual-Semantic Transformer for Scene Text Recognition

Xin Tang, Yongquan Lai, Ying Liu, Yuanyuan Fu, Rui Fang

Category: scene-text
Keywords: Scene Text Recognition, Visual-Semantic Transformer, Semantic Information, Transformer, Visual Features
Year: 2022

Modeling semantic information is helpful for scene text recognition. In this
work, we propose to model semantic and visual information jointly with a Visual-
Semantic Transformer (VST). The VST first explicitly extracts primary semantic
information from visual feature maps with a transformer module and a primary
visual-semantic alignment module. The semantic information is then joined with
the visual feature maps (viewed as a sequence) to form a pseudo multi-domain
sequence combining visual and semantic information, which is subsequently fed
into a transformer-based interaction module to enable learning of interactions
between visual and semantic features. In this way, the visual features can be
enhanced by the semantic information and vice versa. The enhanced version of
visual features are further decoded by a secondary visual-semantic alignment
module which shares weights with the primary one. Finally, the decoded visual
features and the enhanced semantic features are jointly processed by the third
transformer module obtaining the final text prediction. Experiments on seven
public benchmarks including regular/irregular text recognition datasets verify
the effectiveness of our proposed model, reaching state of the art on four of
the seven benchmarks.
---

## Scene text detection and recognition: a survey

Fatemeh Naiemi, Vahid Ghods, Hassan Khalesi

Category: scene-text
Keywords: Scene text localization, Text image detection, End-to-end recognition, Multi-oriented, Convolutional neural network, Text recognition
Year: 2022

Scene text detection and recognition have been given a lot of attention in
recent years and have been used in many vision-based applications. In this
field, there are various types of challenges, including images with wavy text,
images with text rotation and orientation, changing the scale and variety of
text fonts, noisy images, wild background images, which make the detection and
recognition of text from the image more complex and difficult. In this article,
we first presented a comprehensive review of recent advances in text detection
and recognition and described the advantages and disadvantages. The common
datasets were introduced. Then, the recent methods compared together and
analyzed the text detection and recognition systems. According to the recent
decade studies, one of the most important challenges is curved and vertical text
detection in this field. We have expressed approaches for the development of the
detection and recognition system. Also, we have described the methods that are
robust in the detection and recognition of curved and vertical texts. Finally,
we have presented some approaches to develop text detection and recognition
systems as the future work.
---

## Data Augmentation for Scene Text Recognition

Rowel Atienza

Category: scene-text
Keywords: scene text recognition, data augmentation, STR, RandAugment, STR models
Year: 2021

Scene text recognition (STR) is a challenging task in computer vision due to the
large number of possible text appearances in natural scenes. Most STR models
rely on synthetic datasets for training since there are no sufficiently big and
publicly available labelled real datasets. Since STR models are evaluated using
real data, the mismatch between training and testing data distributions results
in poor performance of models, especially on challenging text that are affected
by noise, artifacts, geometry, structure, etc. In this paper, we introduce
STRAug, which is made of 36 image augmentation functions designed for STR. Each
function mimics certain text image properties that can be found in natural
scenes, caused by camera sensors, or induced by signal processing operations but
poorly represented in the training dataset. When applied to strong baseline
models using RandAugment, STRAug significantly increases the overall absolute
accuracy of STR models across regular and irregular test datasets by as much as
2.10% on Rosetta, 1.48% on R2AM, 1.30% on CRNN, 1.35% on RARE, 1.06% on TRBA,
and 0.89% on GCRNN. The diversity and simplicity of API provided by STRAug
functions enable easy replication and validation of existing data augmentation
methods for STR. STRAug is available at https://github.com/roatienza/straug.
---

## Utilizing Resource-Rich Language Datasets for End-to-End Scene Text Recognition in Resource-Poor Languages

Shota Orihashi, Yoshihiro Yamazaki, Naoki Makishima, Mana Ihori, Akihiko Takashima, Tomohiro Tanaka, Ryo Masumura

Category: scene-text
Keywords: scene text recognition, pre-training, Transformer, resource-poor language
Year: 2021

This paper presents a novel training method for end-to-end scene text
recognition. End-to-end scene text recognition offers high recognition accuracy,
especially when using the encoder-decoder model based on Transformer. To train a
highly accurate end-to-end model, we need to prepare a large image-to-text
paired dataset for the target language. However, it is difficult to collect this
data, especially for resource-poor languages. To overcome this difficulty, our
proposed method utilizes well-prepared large datasets in resource-rich languages
such as English, to train the resource-poor encoder-decoder model. Our key idea
is to build a model in which the encoder reflects knowledge of multiple
languages while the decoder specializes in knowledge of just the resource-poor
language. To this end, the proposed method pre-trains the encoder by using a
multilingual dataset that combines the resource-poor language’s dataset and the
resource-rich language’s dataset to learn language-invariant knowledge for scene
text recognition. The proposed method also pre-trains the decoder by using the
resource-poor language’s dataset to make the decoder better suited to the
resource-poor language. Experiments on Japanese scene text recognition using a
small, publicly available dataset demonstrate the effectiveness of the proposed
method.
---

## Text Recognition in the Wild: A Survey

Xiaoxue Chen, Lianwen Jin, Yuanzhi Zhu, Canjie Luo, Tianwei Wang

Category: scene-text
Keywords: Scene text recognition, end-to-end systems, deep learning
Year: 2021

The history of text can be traced back over thousands of years. Rich and precise
semantic information carried by text is important in a wide range of vision-
based application scenarios. Therefore, text recognition in natural scenes has
been an active research topic in computer vision and pattern recognition. In
recent years, with the rise and development of deep learning, numerous methods
have shown promising results in terms of innovation, practicality, and
efficiency. This article aims to (1) summarize the fundamental problems and the
state-of-the-art associated with scene text recognition, (2) introduce new
insights and ideas, (3) provide a comprehensive review of publicly available
resources, and (4) point out directions for future work. In summary, this
literature review attempts to present an entire picture of the field of scene
text recognition. It provides a comprehensive reference for people entering this
field and could be helpful in inspiring future research.
---

## Vision Transformer for Fast and Efficient Scene Text Recognition

Rowel Atienza

Category: scene-text
Keywords: Scene text recognition, Transformer, Data augmentation
Year: 2021

Scene text recognition (STR) enables computers to read text in natural scenes
such as object labels, road signs, and instructions. STR helps machines perform
informed decisions such as what object to pick, which direction to go, and what
is the next step of action. In the body of work on STR, the focus has always
been on recognition accuracy. There is little emphasis placed on speed and
computational efficiency, which are equally important, especially for energy-
constrained mobile machines. In this paper, we propose ViTSTR, an STR with a
simple single-stage model architecture built on a compute and parameter-
efficient vision transformer (ViT). On a comparable strong baseline method such
as TRBA with an accuracy of 84.3%, our small ViTSTR achieves a competitive
accuracy of 82.6% (84.2% with data augmentation) at 2.4× speed up, using only
43.4% of the number of parameters and 42.2% FLOPS. The tiny version of ViTSTR
achieves 80.3% accuracy (82.1% with data augmentation), at 2.5× the speed,
requiring only 10.9% of the number of parameters and 11.9% FLOPS. With data
augmentation, our base ViTSTR outperforms TRBA at 85.2% accuracy (83.7% without
augmentation) at 2.3× the speed but requires 73.2% more parameters and 61.5%
more FLOPS. In terms of trade-offs, nearly all ViTSTR configurations are at or
near the frontiers to maximize accuracy, speed, and computational efficiency all
at the same time.
---

## TRIG: Transformer-Based Text Recognizer with Initial Embedding Guidance

Yue Tao, Zhiwei Jia, Runze Ma, Shugong Xu

Category: scene-text
Keywords: scene text recognition, transformer, self-attention, 1-D split, initial embedding
Year: 2021

Scene text recognition (STR) is an important bridge between images and text,
attracting abundant research attention. While convolutional neural networks
(CNNs) have achieved remarkable progress in this task, most of the existing
works need an extra module (context modeling module) to help CNN to capture
global dependencies to solve the inductive bias and strengthen the relationship
between text features. Recently, the transformer has been proposed as a
promising network for global context modeling by self-attention mechanism, but
one of the main shortcomings, when applied to recognition, is the efficiency. We
propose a 1-D split to address the challenges of complexity and replace the CNN
with the transformer encoder to reduce the need for a context modeling module.
Furthermore, recent methods use a frozen initial embedding to guide the decoder
to decode the features to text, leading to a loss of accuracy. We propose to use
a learnable initial embedding learned from the transformer encoder to make it
adaptive to different input images. Above all, we introduce a novel architecture
for text recognition, named TRansformer-based text recognizer with Initial
embedding Guidance (TRIG), composed of three stages (transformation, feature
extraction, and prediction). Extensive experiments show that our approach can
achieve state-of-the-art on text recognition benchmarks.
---

## Utilizing Resource-Rich Language Datasets for End-to-End Scene Text Recognition in Resource-Poor Languages

Shota Orihashi, Yoshihiro Yamazaki, Naoki Makishima, Mana Ihori, Akihiko Takashima, Tomohiro Tanaka, Ryo Masumura

Category: scene-text
Keywords: scene text recognition, pre-training, Transformer, resource-poor language
Year: 2021

This paper presents a novel training method for end-to-end scene text
recognition. End-to-end scene text recognition offers high recognition accuracy,
especially when using the encoder-decoder model based on Transformer. To train a
highly accurate end-to-end model, we need to prepare a large image-to-text
paired dataset for the target language. However, it is difficult to collect this
data, especially for resource-poor languages. To overcome this difficulty, our
proposed method utilizes well-prepared large datasets in resource-rich languages
such as English, to train the resource-poor encoder-decoder model. Our key idea
is to build a model in which the encoder reflects knowledge of multiple
languages while the decoder specializes in knowledge of just the resource-poor
language. To this end, the proposed method pre-trains the encoder by using a
multilingual dataset that combines the resource-poor language’s dataset and the
resource-rich language’s dataset to learn language-invariant knowledge for scene
text recognition. The proposed method also pre-trains the decoder by using the
resource-poor language’s dataset to make the decoder better suited to the
resource-poor language. Experiments on Japanese scene text recognition using a
small, publicly available dataset demonstrate the effectiveness of the proposed
method.
---

## Vision Transformer for Fast and Efficient Scene Text Recognition

Rowel Atienza

Category: scene-text
Keywords: Scene text recognition, Transformer, Data augmentation
Year: 2021

Scene text recognition (STR) enables computers to read text in natural scenes
such as object labels, road signs, and instructions. STR helps machines perform
informed decisions such as what object to pick, which direction to go, and what
is the next step of action. In the body of work on STR, the focus has always
been on recognition accuracy. There is little emphasis placed on speed and
computational efficiency, which are equally important, especially for energy-
constrained mobile machines. In this paper, we propose ViTSTR, an STR with a
simple single-stage model architecture built on a compute and parameter-
efficient vision transformer (ViT). On a comparable strong baseline method such
as TRBA with an accuracy of 84.3%, our small ViTSTR achieves a competitive
accuracy of 82.6% (84.2% with data augmentation) at 2.4× speed up, using only
43.4% of the number of parameters and 42.2% FLOPS. The tiny version of ViTSTR
achieves 80.3% accuracy (82.1% with data augmentation), at 2.5× the speed,
requiring only 10.9% of the number of parameters and 11.9% FLOPS. With data
augmentation, our base ViTSTR outperforms TRBA at 85.2% accuracy (83.7% without
augmentation) at 2.3× the speed but requires 73.2% more parameters and 61.5%
more FLOPS. In terms of trade-offs, nearly all ViTSTR configurations are at or
near the frontiers to maximize accuracy, speed, and computational efficiency all
at the same time.
---

## TRIG: Transformer-Based Text Recognizer with Initial Embedding Guidance

Yue Tao, Zhiwei Jia, Runze Ma, Shugong Xu

Category: scene-text
Keywords: scene text recognition, transformer, self-attention, 1-D split, initial embedding
Year: 2021

Scene text recognition (STR) is an important bridge between images and text,
attracting abundant research attention. While convolutional neural networks
(CNNs) have achieved remarkable progress in this task, most of the existing
works need an extra module (context modeling module) to help CNN to capture
global dependencies to solve the inductive bias and strengthen the relationship
between text features. Recently, the transformer has been proposed as a
promising network for global context modeling by self-attention mechanism, but
one of the main shortcomings, when applied to recognition, is the efficiency. We
propose a 1-D split to address the challenges of complexity and replace the CNN
with the transformer encoder to reduce the need for a context modeling module.
Furthermore, recent methods use a frozen initial embedding to guide the decoder
to decode the features to text, leading to a loss of accuracy. We propose to use
a learnable initial embedding learned from the transformer encoder to make it
adaptive to different input images. Above all, we introduce a novel architecture
for text recognition, named TRansformer-based text recognizer with Initial
embedding Guidance (TRIG), composed of three stages (transformation, feature
extraction, and prediction). Extensive experiments show that our approach can
achieve state-of-the-art on text recognition benchmarks.
---

## MASTER: Multi-Aspect Non-local Network for Scene Text Recognition

Ning Lu, Wenwen Yu, Xianbiao Qi, Yihao Chen, Ping Gong, Rong Xiao, Xiang Bai

Category: scene-text
Keywords: scene text recognition, attention mechanism, RNN, self-attention, efficiency
Year: 2021

Attention-based scene text recognizers have gained huge success, which leverages
a more compact intermediate representation to learn 1d- or 2d- attention by a
RNN-based encoder-decoder architecture. However, such methods suffer from
attention-drift problem because high similarity among encoded features leads to
attention confusion under the RNN-based local attention mechanism. Moreover,
RNN-based methods have low efficiency due to poor parallelization. To overcome
these problems, we propose the MASTER, a self-attention based scene text
recognizer that not only encodes the input-output attention but also learns
self-attention.
---

## Scene Text Detection and Recognition: The Deep Learning Era

Shangbang Long, Xin He, Cong Yao

Category: scene-text
Keywords: Scene text, Optical character recognition, Detection, Recognition, Deep learning, Survey
Year: 2021

With the rise and development of deep learning, computer vision has been
tremendously transformed and reshaped. As an important research area in computer
vision, scene text detection and recognition has been inevitably influenced by
this wave of revolution, consequentially entering the era of deep learning. In
recent years, the community has witnessed substantial advancements in mindset,
methodology and performance. This survey is aimed at summarizing and analyzing
the major changes and significant progresses of scene text detection and
recognition in the deep learning era. Through this article, we devote to: (1)
introduce new insights and ideas; (2) highlight recent techniques and
benchmarks; (3) look ahead into future trends. Specifically, we will emphasize
the dramatic differences brought by deep learning and remaining grand
challenges. We expect that this review paper would serve as a reference book for
researchers in this field. Related resources are also collected in our Github
repository.
---

## MASTER: Multi-Aspect Non-local Network for Scene Text Recognition

Ning Lu, Wenwen Yu, Xianbiao Qi, Yihao Chen, Ping Gong, Rong Xiao, Xiang Bai

Category: scene-text
Keywords: scene text recognition, attention mechanism, RNN, self-attention, non-local network
Year: 2021

Attention-based scene text recognizers have gained huge success, which leverages
a more compact intermediate representation to learn 1d- or 2d- attention by a
RNN-based encoder-decoder architecture. However, such methods suffer from
attention-drift problem because high similarity among encoded features leads to
attention confusion under the RNN-based local attention mechanism. Moreover,
RNN-based methods have low efficiency due to poor parallelization. To overcome
these problems, we propose the MASTER, a self-attention based scene text
recognizer that (1) not only encodes the input-output attention but also learns
self-attention which enables non-local operations.
---

## TRIG: Transformer-Based Text Recognizer with Initial Embedding Guidance

Yue Tao, Zhiwei Jia, Runze Ma, Shugong Xu

Category: scene-text
Keywords: scene text recognition, transformer, self-attention, 1-D split, initial embedding
Year: 2021

Scene text recognition (STR) is an important bridge between images and text,
attracting abundant research attention. While convolutional neural networks
(CNNs) have achieved remarkable progress in this task, most of the existing
works need an extra module (context modeling module) to help CNN to capture
global dependencies to solve the inductive bias and strengthen the relationship
between text features. Recently, the transformer has been proposed as a
promising network for global context modeling by self-attention mechanism, but
one of the main shortcomings, when applied to recognition, is the efficiency. We
propose a 1-D split to address the challenges of complexity and replace the CNN
with the transformer encoder to reduce the need for a context modeling module.
Furthermore, recent methods use a frozen initial embedding to guide the decoder
to decode the features to text, leading to a loss of accuracy. We propose to use
a learnable initial embedding learned from the transformer encoder to make it
adaptive to different input images. Above all, we introduce a novel architecture
for text recognition, named TRansformer-based text recognizer with Initial
embedding Guidance (TRIG), composed of three stages (transformation, feature
extraction, and prediction). Extensive experiments show that our approach can
achieve state-of-the-art on text recognition benchmarks.
---

## Data Augmentation for Scene Text Recognition

Rowel Atienza

Category: scene-text
Keywords: scene text recognition, data augmentation, STR, STRAug, computer vision
Year: 2020

Scene text recognition (STR) is a challenging task in computer vision due to the
large number of possible text appearances in natural scenes. Most STR models
rely on synthetic datasets for training since there are no sufficiently big and
publicly available labelled real datasets. Since STR models are evaluated using
real data, the mismatch between training and testing data distributions results
in poor performance of models, especially on challenging text that is affected
by noise, artifacts, geometry, structure, etc. In this paper, we introduce
STRAug, which is made of 36 image augmentation functions designed for STR. Each
function mimics certain text image properties that can be found in natural
scenes, caused by camera sensors, or induced by signal processing operations but
poorly represented in the training dataset. When applied to strong baseline
models using RandAugment, STRAug significantly increases the overall absolute
accuracy of STR models across regular and irregular test datasets by as much as
2.10% on Rosetta, 1.48% on R2AM, 1.30% on CRNN, 1.35% on RARE, 1.06% on TRBA,
and 0.89% on GCRNN. The diversity and simplicity of the API provided by STRAug
functions enable easy replication and validation of existing data augmentation
methods for STR. STRAug is available at https://github.com/roatienza/straug.
---

## TextSnake: A Flexible Representation for Detecting Text of Arbitrary Shapes

Shangbang Long, Jiaqiang Ruan, Wenjie Zhang, Xin He, Wenhao Wu, Cong Yao

Category: scene-text
Keywords: Scene Text Detection, Deep Neural Network, Curved Text
Year: 2020

Driven by deep neural networks and large scale datasets, scene text detection
methods have progressed substantially over the past years, continuously
refreshing the performance records on various standard benchmarks. However,
limited by the representations (axis-aligned rectangles, rotated rectangles or
quadrangles) adopted to describe text, existing methods may fall short when
dealing with much more free-form text instances, such as curved text, which are
actually very common in real-world scenarios. To tackle this problem, we propose
a more flexible representation for scene text, termed as TextSnake, which is
able to effectively represent text instances in horizontal, oriented and curved
forms. In TextSnake, a text instance is described as a sequence of ordered,
overlapping disks centered at symmetric axes, each of which is associated with
potentially variable radius and orientation. Such geometry attributes are
estimated via a Fully Convolutional Network (FCN) model. In experiments, the
text detector based on TextSnake achieves state-of-the-art or comparable
performance on Total-Text and SCUT-CTW1500, the two newly published benchmarks
with special emphasis on curved text in natural images, as well as the widely-
used datasets ICDAR 2015 and MSRA-TD500. Specifically, TextSnake outperforms the
baseline on Total-Text by more than 40% in F-measure.
---

## Towards Accurate Scene Text Recognition with Semantic Reasoning Networks

Deli Yu, Xuan Li, Chengquan Zhang, Tao Liu, Junyu Han, Jingtuo Liu, Errui Ding

Category: scene-text
Keywords: scene text recognition, semantic reasoning, global semantic context, RNN, GSRM
Year: 2020

Scene text image contains two levels of contents: visual texture and semantic
information. Although the previous scene text recognition methods have made
great progress over the past few years, the research on mining semantic
information to assist text recognition attracts less attention, only RNN-like
structures are explored to implicitly model semantic information. However, we
observe that RNN based methods have some obvious shortcomings, such as time-
dependent decoding manner and one-way serial transmission of semantic context,
which greatly limit the help of semantic information and the computation
efficiency. To mitigate these limitations, we propose a novel end-to-end
trainable framework named semantic reasoning network (SRN) for accurate scene
text recognition, where a global semantic reasoning module (GSRM) is introduced
to capture global semantic context through multi-way parallel transmission. The
state-of-the-art results on 7 public benchmarks, including regular text,
irregular text and non-Latin long text, verify the effectiveness and robustness
of the proposed method. In addition, the speed of SRN has significant advantages
over the RNN based methods, demonstrating its value in practical use.
---

## Towards Accurate Scene Text Recognition with Semantic Reasoning Networks

Deli Yu, Xuan Li, Chengquan Zhang, Tao Liu, Junyu Han, Jingtuo Liu, Errui Ding

Category: scene-text
Keywords: scene text recognition, semantic reasoning, global semantic reasoning module, text recognition, RNN
Year: 2020

Scene text image contains two levels of contents: visual texture and semantic
information. Although the previous scene text recognition methods have made
great progress over the past few years, the research on mining semantic
information to assist text recognition attracts less attention, only RNN-like
structures are explored to implicitly model semantic information. However, we
observe that RNN-based methods have some obvious shortcomings, such as time-
dependent decoding manner and one-way serial transmission of semantic context,
which greatly limit the help of semantic information and the computation
efficiency. To mitigate these limitations, we propose a novel end-to-end
trainable framework named semantic reasoning network (SRN) for accurate scene
text recognition, where a global semantic reasoning module (GSRM) is introduced
to capture global semantic context through multi-way parallel transmission. The
state-of-the-art results on 7 public benchmarks, including regular text,
irregular text and non-Latin long text, verify the effectiveness and robustness
of the proposed method. In addition, the speed of SRN has significant advantages
over the RNN-based methods, demonstrating its value in practical use.
---

## Real-time Scene Text Detection with Differentiable Binarization

Minghui Liao, Zhaoyi Wan, Cong Yao, Kai Chen, Xiang Bai

Category: scene-text
Keywords: scene text detection, differentiable binarization, segmentation, real-time detection, MSRA-TD500
Year: 2020

Recently, segmentation-based methods are quite popular in scene text detection,
as the segmentation results can more accurately describe scene text of various
shapes such as curve text. However, the post-processing of binarization is
essential for segmentation-based detection, which converts probability maps
produced by a segmentation method into bounding boxes/regions of text. In this
paper, we propose a module named Differentiable Binarization (DB), which can
perform the binarization process in a segmentation network. Optimized along with
a DB module, a segmentation network can adaptively set the thresholds for
binarization, which not only simplifies the post-processing but also enhances
the performance of text detection. Based on a simple segmentation network, we
validate the performance improvements of DB on five benchmark datasets, which
consistently achieves state-of-the-art results, in terms of both detection
accuracy and speed. In particular, with a light-weight backbone, the performance
improvements by DB are significant so that we can look for an ideal tradeoff
between detection accuracy and efficiency. Specifically, with a backbone of
ResNet-18, our detector achieves an F-measure of 82.8, running at 62 FPS, on the
MSRA-TD500 dataset.
---

## Towards Accurate Scene Text Recognition with Semantic Reasoning Networks

Deli Yu, Xuan Li, Chengquan Zhang, Tao Liu, Junyu Han, Jingtuo Liu, Errui Ding

Category: scene-text
Keywords: scene text recognition, semantic reasoning networks, global semantic context, multi-way parallel transmission, RNN limitations
Year: 2020

Scene text image contains two levels of contents: visual texture and semantic
information. Although the previous scene text recognition methods have made
great progress over the past few years, the research on mining semantic
information to assist text recognition attracts less attention, only RNN-like
structures are explored to implicitly model semantic information. However, we
observe that RNN based methods have some obvious shortcomings, such as time-
dependent decoding manner and one-way serial transmission of semantic context,
which greatly limit the help of semantic information and the computation
efficiency. To mitigate these limitations, we propose a novel end-to-end
trainable framework named semantic reasoning network (SRN) for accurate scene
text recognition, where a global semantic reasoning module (GSRM) is introduced
to capture global semantic context through multi-way parallel transmission. The
state-of-the-art results on 7 public benchmarks, including regular text,
irregular text and non-Latin long text, verify the effectiveness and robustness
of the proposed method. In addition, the speed of SRN has significant advantages
over the RNN based methods, demonstrating its value in practical use.
---

## What Is Wrong With Scene Text Recognition Model Comparisons? Dataset and Model Analysis

Jeonghun Baek, Geewook Kim, Junyeop Lee, Sungrae Park, Dongyoon Han, Sangdoo Yun, Seong Joon Oh, Hwalsuk Lee

Category: scene-text
Keywords: Scene Text Recognition, Dataset Analysis, Model Comparison, Module-wise Evaluation, Deep Learning
Year: 2019

Many new proposals for scene text recognition (STR) models have been introduced
in recent years. While each claims to have pushed the boundary of the
technology, a holistic and fair comparison has been largely missing in the field
due to the inconsistent choices of training and evaluation datasets. This paper
addresses this difficulty with three major contributions. First, we examine the
inconsistencies of training and evaluation datasets, and the performance gap
results from inconsistencies. Second, we introduce a unified four-stage STR
framework that most existing STR models fit into. Using this framework allows
for the extensive evaluation of previously proposed STR modules and the
discovery of previously unexplored module combinations. Third, we analyze the
module-wise contributions to performance in terms of accuracy, speed, and memory
demand, under one consistent set of training and evaluation datasets. Such
analyses clean up the hindrance on the current comparisons to understand the
performance gain of the existing modules.
---

## What Is Wrong With Scene Text Recognition Model Comparisons? Dataset and Model Analysis

Jeonghun Baek, Geewook Kim, Junyeop Lee, Sungrae Park, Dongyoon Han, Sangdoo Yun, Seong Joon Oh, Hwalsuk Lee

Category: scene-text
Keywords: scene text recognition, dataset analysis, model evaluation, STR framework, performance comparison
Year: 2019

Many new proposals for scene text recognition (STR) models have been introduced
in recent years. While each claims to have pushed the boundary of the
technology, a holistic and fair comparison has been largely missing in the field
due to the inconsistent choices of training and evaluation datasets. This paper
addresses this difficulty with three major contributions. First, we examine the
inconsistencies of training and evaluation datasets, and the performance gap
results from these inconsistencies. Second, we introduce a unified four-stage
STR framework that most existing STR models fit into. Using this framework
allows for the extensive evaluation of previously proposed STR modules and the
discovery of previously unexplored module combinations. Third, we analyze the
module-wise contributions to performance in terms of accuracy, speed, and memory
demand, under one consistent set of training and evaluation datasets. Such
analyses clean up the hindrance on the current comparisons to understand the
performance gain of the existing modules. Our code is publicly available.
---

## What Is Wrong With Scene Text Recognition Model Comparisons? Dataset and Model Analysis

Jeonghun Baek, Geewook Kim, Junyeop Lee, Sungrae Park, Dongyoon Han, Sangdoo Yun, Seong Joon Oh, Hwalsuk Lee

Category: scene-text
Keywords: Scene Text Recognition, STR, Dataset Analysis, Model Comparison, Deep Learning
Year: 2019

Many new proposals for scene text recognition (STR) models have been introduced
in recent years. While each claims to have pushed the boundary of the
technology, a holistic and fair comparison has been largely missing in the field
due to the inconsistent choices of training and evaluation datasets. This paper
addresses this difficulty with three major contributions. First, we examine the
inconsistencies of training and evaluation datasets, and the performance gap
results from inconsistencies. Second, we introduce a unified four-stage STR
framework that most existing STR models fit into. Using this framework allows
for the extensive evaluation of previously proposed STR modules and the
discovery of previously unexplored module combinations. Third, we analyze the
module-wise contributions to performance in terms of accuracy, speed, and memory
demand, under one consistent set of training and evaluation datasets. Such
analyses clean up the hindrance on the current comparisons to understand the
performance gain of the existing modules.
---

## Scene Text Recognition from Two-Dimensional Perspective

Minghui Liao, Jian Zhang, Zhaoyi Wan, Fengming Xie, Jiajun Liang, Pengyuan Lyu, Cong Yao, Xiang Bai

Category: scene-text
Keywords: scene text recognition, two-dimensional perspective, Character Attention Fully Convolutional Network, semantic segmentation, attention mechanism
Year: 2018

Inspired by speech recognition, recent state-of-the-art algorithms mostly
consider scene text recognition as a sequence prediction problem. Though
achieving excellent performance, these methods usually neglect an important fact
that text in images are actually distributed in two-dimensional space. It is a
nature quite different from that of speech, which is essentially a one-
dimensional signal. In principle, directly compressing features of text into a
one-dimensional form may lose useful information and introduce extra noise. In
this paper, we approach scene text recognition from a two-dimensional
perspective. A simple yet effective model, called Character Attention Fully
Convolutional Network (CA-FCN), is devised for recognizing the text of arbitrary
shapes. Scene text recognition is realized with a semantic segmentation network,
where an attention mechanism for characters is adopted. Combined with a word
formation module, CA-FCN can simultaneously recognize the script and predict the
position of each character. Experiments demonstrate that the proposed algorithm
outperforms previous methods on both regular and irregular text datasets.
Moreover, it is proven to be more robust to imprecise localizations in the text
detection phase, which are very common in practice.
---

## Scene Text Recognition with Sliding Convolutional Character Models

Fei Yin, Yi-Chao Wu, Xu-Yao Zhang, Cheng-Lin Liu

Category: scene-text
Keywords: scene text recognition, convolutional neural networks, character models, Connectionist Temporal Classification, pattern recognition
Year: 2017

Scene text recognition has attracted great interests from the computer vision
and pattern recognition community in recent years. State-of-the-art methods use
convolutional neural networks (CNNs), recurrent neural networks with long short-
term memory (RNN-LSTM) or the combination of them. In this paper, we investigate
the intrinsic characteristics of text recognition, and inspired by human
cognition mechanisms in reading texts, we propose a scene text recognition
method with character models on convolutional feature map. The method
simultaneously detects and recognizes characters by sliding the text line image
with character models, which are learned end-to-end on text line images labeled
with text transcripts. The character classifier outputs on the sliding windows
are normalized and decoded with Connectionist Temporal Classification (CTC)
based algorithm. Compared to previous methods, our method has a number of
appealing properties: (1) It avoids the difficulty of character segmentation
which hinders the performance of segmentation-based recognition methods; (2) The
model can be trained simply and efficiently because it avoids gradient
vanishing/exploding in training RNN-LSTM based models; (3) It bases on character
models trained free of lexicon, and can recognize unknown words. (4) The
recognition process is highly parallel and enables fast recognition. Our
experiments on several challenging English and Chinese benchmarks, including the
IIIT-5K, SVT, ICDAR03/13 and TRW15 datasets, demonstrate that the proposed
method yields superior or comparable performance to state-of-the-art methods
while the model size is relatively small.
---

## Robust Scene Text Recognition with Automatic Rectification

Baoguang Shi, Xinggang Wang, Pengyuan Lyu, Cong Yao, Xiang Bai

Category: scene-text
Keywords: scene text recognition, deep neural networks, spatial transformer network, sequence recognition, irregular text, automatic rectification
Year: 2016

Recognizing text in natural images is a challenging task with many unsolved
problems. Different from those in documents, words in natural images often
possess irregular shapes, which are caused by perspective distortion, curved
character placement, etc. We propose RARE (Robust text recognizer with Automatic
Rectification), a recognition model that is robust to irregular text. RARE is a
specially-designed deep neural network, which consists of a Spatial Transformer
Network (STN) and a Sequence Recognition Network (SRN). In testing, an image is
firstly rectified via a predicted Thin-Plate-Spline (TPS) transformation, into a
more 'readable' image for the following SRN, which recognizes text through a
sequence recognition approach. We show that the model is able to recognize
several types of irregular text, including perspective text and curved text.
RARE is end-to-end trainable, requiring only images and associated text labels,
making it convenient to train and deploy the model in practical systems. State-
of-the-art or highly-competitive performance achieved on several benchmarks well
demonstrates the effectiveness of the proposed model.
---

## Reading Scene Text in Deep Convolutional Sequences

Pan He, Weilin Huang, Yu Qiao, Chen Change Loy, Xiaoou Tang

Category: scene-text
Keywords: scene text recognition, deep learning, convolutional neural networks, recurrent neural networks, sequence labelling
Year: 2015

We develop a Deep-Text Recurrent Network (DTRN) that regards scene text reading
as a sequence labelling problem. We leverage recent advances of deep
convolutional neural networks to generate an ordered high-level sequence from a
whole word image, avoiding the difficult character segmentation problem. Then a
deep recurrent model, building on long short-term memory (LSTM), is developed to
robustly recognize the generated CNN sequences, departing from most existing
approaches recognising each character independently. Our model has a number of
appealing properties in comparison to existing scene text recognition methods:
(i) It can recognise highly ambiguous words by leveraging meaningful context
information, allowing it to work reliably without either pre- or post-
processing; (ii) the deep CNN feature is robust to various image distortions;
(iii) it retains the explicit order information in word image, which is
essential to discriminate word strings; (iv) the model does not depend on pre-
defined dictionary, and it can process unknown words and arbitrary strings. It
achieves impressive results on several benchmarks, advancing the state-of-the-
art substantially.
---

## An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to Scene Text Recognition

Baoguang Shi, Xiang Bai, Cong Yao

Category: scene-text
Keywords: image-based sequence recognition, scene text recognition, end-to-end training, neural network, variable-length sequences
Year: 2015

Image-based sequence recognition has been a longstanding research topic in
computer vision. In this paper, we investigate the problem of scene text
recognition, which is among the most important and challenging tasks in image-
based sequence recognition. A novel neural network architecture, which
integrates feature extraction, sequence modeling and transcription into a
unified framework, is proposed. Compared with previous systems for scene text
recognition, the proposed architecture possesses four distinctive properties:
(1) It is end-to-end trainable, in contrast to most of the existing algorithms
whose components are separately trained and tuned. (2) It naturally handles
sequences in arbitrary lengths, involving no character segmentation or
horizontal scale normalization. (3) It is not confined to any predefined lexicon
and achieves remarkable performances in both lexicon-free and lexicon-based
scene text recognition tasks. (4) It generates an effective yet much smaller
model, which is more practical for real-world application scenarios. The
experiments on standard benchmarks, including the IIIT-5K, Street View Text and
ICDAR datasets, demonstrate the superiority of the proposed algorithm over the
prior arts. Moreover, the proposed algorithm performs well in the task of image-
based music score recognition, which evidently verifies the generality of it.
---

## ICDAR 2015 Competition on Robust Reading

Dimosthenis Karatzas, Lluis Gomez-Bigorda, Anguelos Nicolaou, Suman Ghosh, Andrew Bagdanov, Masakazu Iwamura, Jiri Matas, Lukas Neumann, Vijay Ramaseshan Chandrasekhar, Shijian Lu, Faisal Shafait, Seiichi Uchida, Ernest Valveny

Category: scene-text
Keywords: Robust Reading, Scene Text, ICDAR 2015, Text Localization, Word Recognition
Year: 2015

Results of the ICDAR 2015 Robust Reading Competition are presented. A new
Challenge 4 on Incidental Scene Text has been added to the Challenges on Born-
Digital Images, Focused Scene Images, and Video Text. Challenge 4 is run on a
newly acquired dataset of 1,670 images evaluating Text Localization, Word
Recognition, and End-to-End pipelines. In addition, the dataset for Challenge 3
on Video Text has been substantially updated with more video sequences and more
accurate ground truth data. Finally, tasks assessing End-to-End system
performance have been introduced to all Challenges. The competition took place
in the first quarter of 2015, and received a total of 44 submissions. Only the
tasks newly introduced in 2015 are reported on. The datasets, the ground truth
specification, and the evaluation protocols are presented together with the
results and a brief summary of the participating methods.
---

## Multi-digit Number Recognition from Street View Imagery using Deep Convolutional Neural Networks

Ian J. Goodfellow, Yaroslav Bulatov, Julian Ibarz, Sacha Arnoud, Vinay Shet

Category: scene-text
Keywords: multi-digit number recognition, Street View imagery, deep convolutional neural networks, DistBelief, SVHN dataset, reCAPTCHA
Year: 2014

Recognizing arbitrary multi-character text in unconstrained natural photographs
is a hard problem. In this paper, we address an equally hard sub-problem in this
domain viz. recognizing arbitrary multi-digit numbers from Street View imagery.
Traditional approaches to solve this problem typically separate out the
localization, segmentation, and recognition steps. In this paper, we propose a
unified approach that integrates these three steps via the use of a deep
convolutional neural network that operates directly on the image pixels. We
employ the DistBelief implementation of deep neural networks in order to train
large, distributed neural networks on high-quality images. We find that the
performance of this approach increases with the depth of the convolutional
network, with the best performance occurring in the deepest architecture we
trained, with eleven hidden layers. We evaluate this approach on the publicly
available SVHN dataset and achieve over 96% accuracy in recognizing complete
street numbers. We show that on a per-digit recognition task, we improve upon
the state-of-the-art, achieving 97.84% accuracy. We also evaluate this approach
on an even more challenging dataset generated from Street View imagery
containing several tens of millions of street number annotations and achieve
over 90% accuracy. To further explore the applicability of the proposed system
to broader text recognition tasks, we apply it to transcribing synthetic
distorted text from a popular CAPTCHA service, reCAPTCHA. With the proposed
approach, we report a 99.8% accuracy on transcribing the hardest category of
reCAPTCHA puzzles. Our evaluations on both tasks, the street number recognition
as well as reCAPTCHA puzzle transcription, indicate that at specific operating
thresholds, the performance of the proposed system is comparable to, and in some
cases exceeds, that of human operators.
---

## Reading Text in the Wild with Convolutional Neural Networks

Max Jaderberg, Karen Simonyan, Andrea Vedaldi, Andrew Zisserman

Category: scene-text
Keywords: text spotting, convolutional neural networks, natural scene images, image retrieval, synthetic text generation
Year: 2014

In this work we present an end-to-end system for text spotting – localising and
recognising text in natural scene images – and text-based image retrieval. This
system is based on a region proposal mechanism for detection and deep
convolutional neural networks for recognition. Our pipeline uses a novel
combination of complementary proposal generation techniques to ensure high
recall, and a fast subsequent filtering stage for improving precision. For the
recognition and ranking of proposals, we train very large convolutional neural
networks to perform word recognition on the whole proposal region at the same
time, departing from the character classifier based systems of the past. These
networks are trained solely on data produced by a synthetic text generation
engine, requiring no human labelled data. Analysing the stages of our pipeline,
we show state-of-the-art performance throughout. We perform rigorous experiments
across a number of standard end-to-end text spotting benchmarks and text-based
image retrieval datasets, showing a large improvement over all previous methods.
Finally, we demonstrate a real-world application of our text spotting system to
allow thousands of hours of news footage to be instantly searchable via a text
query.
---

## Scene Text Recognition using Higher Order Language Priors

Anand Mishra, Karteek Alahari, C.V. Jawahar

Category: scene-text
Keywords: scene text recognition, language priors, natural images, text recognition, machine vision
Year: 2013

This paper presents a novel approach to scene text recognition by incorporating
higher-order language priors. The methodology leverages advanced linguistic
models to enhance the accuracy of text recognition in natural scenes, providing
improvements over traditional methods that rely primarily on visual appearance.
The proposed system demonstrates significant performance gains, especially in
challenging scenarios with complex backgrounds and varied text orientations.
---

## End-to-End Text Recognition with Convolutional Neural Networks

Tao Wang, David J. Wu, Adam Coates, Andrew Y. Ng

Category: scene-text
Keywords: end-to-end text recognition, convolutional neural networks, scene text recognition, unsupervised feature learning, lexicon-driven system
Year: 2012

Full end-to-end text recognition in natural images is a challenging problem that
has received much attention recently. Traditional systems in this area have
relied on elaborate models incorporating carefully hand-engineered features or
large amounts of prior knowledge. In this paper, we take a different route and
combine the representational power of large, multilayer neural networks together
with recent developments in unsupervised feature learning, which allows us to
use a common framework to train highly-accurate text detector and character
recognizer modules. Then, using only simple off-the-shelf methods, we integrate
these two modules into a full end-to-end, lexicon-driven, scene text recognition
system that achieves state-of-the-art performance on standard benchmarks, namely
Street View Text and ICDAR 2003.
---

## End-to-End Scene Text Recognition

Kai Wang, Boris Babenko, Serge Belongie

Category: scene-text
Keywords: scene text recognition, word detection, OCR, object recognition, text detection
Year: 2011

This paper focuses on the problem of word detection and recognition in natural
images. The problem is significantly more challenging than reading text in
scanned documents, and has only recently gained attention from the computer
vision community. Sub-components of the problem, such as text detection and
cropped image word recognition, have been studied in isolation. However, what is
unclear is how these recent approaches contribute to solving the end-to-end
problem of word recognition. We fill this gap by constructing and evaluating two
systems. The first, representing the de facto state-of-the-art, is a two-stage
pipeline consisting of text detection followed by a leading OCR engine. The
second is a system rooted in generic object recognition, an extension of our
previous work. We show that the latter approach achieves superior performance.
While scene text recognition has generally been treated with highly domain-
specific methods, our results demonstrate the suitability of applying generic
computer vision methods. Adopting this approach opens the door for real-world
scene text recognition to benefit from the rapid advances that have been taking
place in object recognition.