
CRAFT Text Detector for High-Resolution Images

https://github.com/YongWookHa/craft-text-detector (Relevance: 1.0)

Keywords: CRAFT, text detection, OCR, high-resolution, PyTorch, document analysis, preprocessing, image processing, machine learning
Stages: preprocessing, segmentation
Architecture: convnet

The CRAFT (Character Region Awareness for Text detection) text detector is a
tool designed to enhance text detection capabilities in high-resolution images,
particularly useful for document analysis. Originally developed for scene text
detection with an input image size of 384x384, this repository allows users to
modify the input size to better suit high-resolution tasks. To utilize this
tool, users need to create a custom data loader that returns images, character
bounding boxes, words, and image file names. Through the provided settings file,
users can specify paths for training and validation datasets. A preprocessing
script is used to prepare the data by generating character and affinity
heatmaps, which speeds up the training process as these computations are only
done once. The training can then be conducted using the main script, and
progress can be monitored using TensorBoard. This project utilizes PyTorch and
PyTorch Lightning frameworks to facilitate model training and deployment.

---



Kraken OCR System

https://github.com/mittagessen/kraken (Relevance: 1.0)

Keywords: OCR, non-Latin scripts, layout analysis, Kraken, multi-script recognition, machine learning, historical texts, open-source, eScriptorium
Stages: layout-analysis, reading-order, text-recognition, output-representation
Architecture: 

Kraken is a comprehensive Optical Character Recognition (OCR) system designed to
process historical and non-Latin script materials efficiently. It offers
features such as fully trainable layout analysis, reading order, and character
recognition, supporting right-to-left, BiDi, and top-to-bottom scripts. The
software produces outputs in formats like ALTO, PageXML, abbyyXML, and hOCR, and
provides word bounding boxes and character cuts for enhanced accuracy. Kraken
supports multi-script recognition and offers a public repository for model
files. It is built for Linux and Mac OS X environments, and can be installed via
PyPi or conda. The system is closely linked with the eScriptorium project,
providing a user-friendly interface for data annotation, model training, and
inference. Kraken is developed at the École Pratique des Hautes Études,
Université PSL, with funding from the European Union and the French National
Research Agency.

---



CRNN for Sequence Recognition

https://github.com/ahmedmazari-dhatim/CRNN-for-sequence-recognition- (Relevance: 1.0)

Keywords: CRNN, sequence recognition, PyTorch, text recognition, neural network, image processing, machine learning, open source, MIT License
Stages: text-recognition
Architecture: crnn

The CRNN (Convolutional Recurrent Neural Network) for sequence recognition is a
software implementation in PyTorch designed to recognize text in images. It
provides a demo program (`demo.py`) that utilizes a pre-trained model to
interpret text from sample images. Users can download the pre-trained model from
external sources such as Baidu Netdisk or Dropbox, place it in the specified
directory, and run the demo to see text recognition in action. The project also
supports training new models with user-specific datasets, allowing flexibility
in handling variable text lengths by organizing images accordingly. Dependencies
for the project include 'warp_ctc_pytorch' and 'lmdb'. The project is open-
source and licensed under the MIT License.

---



OCR Evaluation Tools with UTF-8 Support

https://github.com/eddieantonio/isri-ocr-evaluation-tools (Relevance: 1.0)

Keywords: OCR, UTF-8, evaluation tools, ISRI, Optical Character Recognition, Linux, macOS, unicode, open source
Stages: evaluation
Architecture: 

The ocreval project is a modern port of the ISRI Analytic Tools for OCR
Evaluation, updated to include UTF-8 support and other improvements. It consists
of a suite of 17 tools designed for measuring and experimenting with the
performance of OCR (Optical Character Recognition) outputs. The project aims to
provide comprehensive evaluation tools that are suitable for contemporary OCR
applications, supporting a wide array of character encodings. Building the tools
requires dependencies like utf8proc, and the project provides installation
guidance for various operating systems including macOS, Ubuntu/Debian, and other
Linux distributions. The tools can be installed using package managers like
Homebrew on macOS or by building from source. The ocreval project is maintained
under the Apache-2.0 license and is developed by Eddie Antonio Santos, with
contributions from the National Research Council Canada. It is built primarily
in C, with some components in Python, Roff, Makefile, and Shell scripting.

---



SBB Textline Detection Tool

https://github.com/qurator-spk/sbb_textline_detector (Relevance: 1.0)

Keywords: textline detection, document images, OCR, border detection, layout detection, heuristic methods, machine learning, document segmentation, historical documents
Stages: segmentation, layout-analysis, preprocessing
Architecture: convnet

The SBB Textline Detection project is a tool designed to perform border, region,
and textline detection from document images. The extracted textlines can be fed
to an OCR model for text recognition. The process involves four stages: border
detection, layout detection, textline detection, and heuristic methods. Border
detection uses a binary pixelwise-segmentation model, trained on a dataset of
2,000 documents, to identify the printed frame's border. Layout detection
identifies text regions and other document elements like separators and images
using a similar model. Textline detection classifies pixels into textlines
through binary segmentation, with models trained on both single and multi-column
documents. Heuristic methods improve model predictions by cropping images to
borders, scaling images for better detection, and applying deskewing. Although
this tool is no longer actively maintained and has been superseded by the more
advanced 'eynollah' tool, it remains a useful resource for historical document
analysis. The tool requires pre-trained models for operation and offers both a
standard command-line interface and integration with OCR-D.

---



Caffe_OCR: OCR Algorithm Research Project

https://github.com/senlinuc/caffe_ocr (Relevance: 1.0)

Keywords: OCR, CNN, BLSTM, CTC, Caffe, DenseNet, Inception, ResNet, LSTM
Stages: text-recognition, training-data-generation
Architecture: convnet, lstm

The Caffe_OCR project is an experimental research initiative focused on
mainstream Optical Character Recognition (OCR) algorithms. It implements a
recognition architecture combining Convolutional Neural Networks (CNN),
Bidirectional Long Short-Term Memory (BLSTM), and Connectionist Temporal
Classification (CTC). The project conducts extensive experiments in data
preparation, network design, and parameter tuning, adapting and modifying
components such as LSTM, Warp-CTC, and multi-label configurations. It supports
network structures based on Inception, ResNet, and DenseNet, and is primarily
developed for the Windows platform, with modifications available for Linux. Key
contributions include multi-label support in the data layer, modified LSTM for
variable input lengths, and simplified network structures by removing sequence
indicators. The project demonstrates high accuracy on both English and Chinese
datasets, using various network designs and configurations. Future improvements
aim to increase data volume, enhance network complexity, and incorporate
advanced techniques like Attention and Spatial Transformer Networks (STN).

---



TableBank: Table Detection Dataset

https://github.com/doc-analysis/TableBank (Relevance: 1.0)

Keywords: TableBank, dataset, table detection, recognition, Word, Latex, weak supervision, Detectron2, OpenNMT
Stages: segmentation, layout-analysis, dataset, evaluation-results
Architecture: convnet

TableBank is a comprehensive dataset for table detection and recognition,
offering 417,234 high-quality labeled tables sourced from Word and Latex
documents. It utilizes a novel weak supervision method to create a large-scale
dataset, surpassing traditional human-labeled datasets. The dataset supports
table detection, which locates tables in document images, and table structure
recognition, which identifies the layout of rows and columns, especially in non-
digital formats like scanned images. TableBank is designed for research purposes
and is available for download on platforms like HuggingFace. It includes
benchmark results using state-of-the-art models like Faster R-CNN for table
detection and OpenNMT for table structure recognition. The dataset also provides
detailed statistics on the number of tables and images across different document
formats and splits for training, validation, and testing. TableBank aims to
advance large-scale table analysis tasks by providing a diverse set of labeled
data from various domains such as business and academic documents.

---



Calamari OCR Engine

https://github.com/ChWick/calamari (Relevance: 1.0)

Keywords: OCR, Calamari, Python, Deep Learning, OCRopy, Kraken, TensorFlow, Open Source, Automation
Stages: text-recognition
Architecture: convnet

Calamari is an Optical Character Recognition (OCR) engine developed using Python
3, based on OCRopy and Kraken. It is designed to be both user-friendly from the
command line and highly modular for integration and customization through other
Python scripts. The project is open source and distributed under the Apache-2.0
license. Calamari is particularly used for line-based ATR (Automated Text
Recognition) and leverages TensorFlow for high-performance deep learning
capabilities. The repository provides access to pretrained models, and users can
install the package via PyPI. Comprehensive documentation is available to guide
users on installation, command-line usage, and API customization. The project
has garnered significant attention, with over 1,000 stars and 210 forks on
GitHub.

---



HierText: Hierarchical Text Annotation Dataset

https://github.com/google-research-datasets/hiertext (Relevance: 1.0)

Keywords: HierText, OCR, dataset, text detection, layout analysis, annotations, Open Images, Unified Detector, CC-BY-SA-4.0
Stages: segmentation, layout-analysis, text-recognition, dataset, evaluation
Architecture: 

The HierText dataset is a comprehensive collection of approximately 12,000
images from the Open Images dataset v6, designed to aid in the development of
robust OCR models and layout analysis. It features hierarchical annotations at
the word, line, and paragraph levels, focusing on natural scenes and documents.
The dataset includes over 1.2 million words, with lines and paragraphs logically
connected and spatially aligned. HierText serves as a benchmark for tasks such
as text detection and recognition at various levels. Additionally, it includes a
novel method called the Unified Detector, which integrates text detection and
layout analysis, with available code and pretrained models. The dataset is split
into training, validation, and test sets, and supports several evaluation tasks,
including word detection, line detection, and end-to-end recognition. HierText
is released under the CC-BY-SA-4.0 license and is a valuable resource for
researchers seeking to improve OCR technology and conduct layout analysis.

---



MMOCR: Text Detection and Recognition Toolbox

https://github.com/open-mmlab/mmocr (Relevance: 1.0)

Keywords: text detection, text recognition, OCR, PyTorch, OpenMMLab, MMDetection, key information extraction, modular design, open-source
Stages: segmentation, text-recognition, evaluation
Architecture: convnet, transformer

MMOCR is an open-source toolbox designed for text detection, recognition, and
key information extraction, developed by OpenMMLab. It leverages PyTorch and
integrates with MMDetection to offer a comprehensive pipeline for OCR tasks. The
toolbox provides support for multiple state-of-the-art models in text detection,
recognition, and key information extraction. It features a modular design,
allowing customization of optimizers, data preprocessors, and model components
such as backbones and heads. MMOCR includes utilities for performance
assessment, visualization, and data conversion. The latest release, v1.0.0,
includes support for new datasets and updated documentation. Installation
requires PyTorch, MMEngine, MMCV, and MMDetection, with detailed instructions
available in the documentation. The project is licensed under Apache 2.0 and is
part of the broader OpenMMLab ecosystem.

---



SBB Textline Detection Tool

https://github.com/qurator-spk/sbb_textline_detection (Relevance: 1.0)

Keywords: textline detection, OCR, document images, pixelwise segmentation, layout detection, border detection, heuristic methods, historical documents, open-source
Stages: preprocessing, segmentation, layout-analysis, text-recognition, output-representation
Architecture: 

The SBB Textline Detection tool is designed to identify and extract textlines
from document images, primarily to prepare them for OCR (Optical Character
Recognition) processing. The tool operates through a series of stages including
border detection, layout detection, textline detection, and heuristic methods.
It uses pixelwise segmentation models trained on diverse datasets to accurately
detect borders and text regions, focusing on historical documents. Heuristic
methods improve detection accuracy by cropping images, scaling for text region
detection, and deskewing regions to separate textlines from background pixels.
The software is no longer actively maintained and has been superseded by the
'eynollah' tool, which offers enhanced performance and functionality. The tool
can be installed via pip and requires pretrained models available for download.
It supports a command-line interface and integration with OCR-D, allowing it to
process images and output results in the PAGE-XML format. The project is open-
source, licensed under Apache-2.0, and primarily developed in Python.

---



UniTable: Unified Table Recognition Model

https://github.com/poloclub/unitable (Relevance: 1.0)

Keywords: UniTable, table recognition, self-supervised pretraining, Transformers, machine learning, data extraction, AI model, structure recognition, NeurIPS
Stages: layout-analysis, table-recognition, text-recognition, evaluation-results
Architecture: transformer, convnet

UniTable is a novel framework aimed at improving table structure recognition by
unifying the training paradigm, training objective, and model architecture. The
project addresses the challenge of parsing tables, which contain complex human-
created data structures, through a unified approach that combines pixel-level
inputs and self-supervised pretraining. UniTable supports three primary tasks in
table recognition: extracting table structure, cell content, and cell bounding
boxes, using a task-agnostic language modeling objective. Demonstrating state-
of-the-art performance on major table recognition datasets, UniTable promotes
reproducible research and transparency by providing a comprehensive Jupyter
Notebook for inference and fine-tuning. The project leverages technologies such
as Transformers and convolutional neural networks to enhance accuracy and
efficiency in table structure recognition. UniTable's development and deployment
include the availability of model weights and resources on platforms like
HuggingFace, catering to researchers and developers looking to implement
advanced table recognition solutions.

---



PSENet: Pytorch Text Detection Implementation

https://github.com/whai362/PSENet (Relevance: 1.0)

Keywords: PSENet, text detection, Pytorch, deep learning, ICDAR 2015, Total-Text, CTW1500, shape detection, ResNet50
Stages: segmentation
Architecture: convnet

PSENet (Progressive Scale Expansion Network) is a state-of-the-art deep learning
model for detecting text in images, particularly beneficial for handling
irregular and curved texts. Developed by W. Wang and colleagues, it offers
robust shape detection. The official Pytorch implementation of PSENet provides
tools for training, testing, and evaluating text detection on datasets like
ICDAR 2015, Total-Text, and CTW1500. It features high precision and recall rates
across various configurations, employing a backbone of ResNet50. The
implementation supports Python 3.6+, Pytorch 1.1.0, and other dependencies
detailed in its repository. Users can train the model using provided
configuration files and evaluate its performance using benchmark scripts. The
repository includes scripts for installing dependencies, compiling code, and
executing training and testing processes. Additionally, PSENet is incorporated
in the MMOCR project and has been adapted to PaddlePaddle. The work is licensed
under the Apache 2.0 license and maintained by IMAGINE Lab at Nanjing
University.

---



Eynollah: Document Layout Analysis Tool

https://github.com/qurator-spk/eynollah (Relevance: 1.0)

Keywords: Document Layout Analysis, Deep Learning, Segmentation, OCR-D, Python, Tensorflow, Image Processing, Historical Documents, Open Source
Stages: preprocessing, segmentation, layout-analysis, output-representation
Architecture: convnet

Eynollah is an advanced tool for document layout analysis that utilizes deep
learning and heuristic methods to segment and analyze document structures. It
supports up to 10 segmentation classes, including text regions, headers, images,
and tables. The tool offers various image optimization operations like
binarization, deskewing, and dewarping, and can output results in PAGE-XML
format. Eynollah is designed to work with Python 3.8-3.11 and Tensorflow
2.12-2.15 on Linux, with optional GPU support via the CUDA toolkit. Users can
install Eynollah via PyPI or clone its GitHub repository. It also functions as
an OCR-D processor, facilitating integration into OCR workflows. Pre-trained
models are available for download, and users can train their models using
related tools. The project is focused on enhancing quality for historical
documents, although processing times can be slow. The development team welcomes
contributions to improve efficiency. Eynollah is released under the Apache-2.0
license.

---



OCR-D Tesserocr Integration

https://github.com/OCR-D/ocrd_tesserocr (Relevance: 1.0)

Keywords: OCR-D, Tesserocr, Tesseract, Image Processing, Text Recognition, Layout Analysis, Python API, Open Source, Document Segmentation
Stages: preprocessing, segmentation, layout-analysis, text-recognition, output-representation
Architecture: 

The 'ocrd_tesserocr' project provides OCR-D compliant workspace processors
utilizing Tesseract's functionality through the tesserocr Python API wrapper.
This project facilitates various OCR processes, including image preprocessing,
layout analysis, script identification, font style recognition, and text
recognition. It supports workflows that involve configurable steps for cropping,
binarization, deskewing, region and line segmentation, and text recognition,
adhering to the PAGE hierarchy for data representation. Installation can be
performed via Docker, PyPI, or directly from the source code, depending on the
availability of Tesseract on the user's system. The module supports dynamic
model selection based on segment conditions and offers a variety of OCR-D
processors to handle different segmentation and recognition tasks. The project
is licensed under the MIT license and is actively maintained with numerous
contributors.

---



CRNN for Text Recognition in TensorFlow 2

https://github.com/FLming/CRNN.tf2 (Relevance: 1.0)

Keywords: CRNN, TensorFlow 2, text recognition, neural network, Keras, OCR, deep learning, Tensorboard, open-source
Stages: text-recognition, evaluation
Architecture: crnn

The CRNN.tf2 project is a re-implementation of the Convolutional Recurrent
Neural Network (CRNN) using TensorFlow 2 for end-to-end text recognition tasks.
This project aims to provide a simple and efficient method for recognizing text
using a combination of convolutional and recurrent neural networks. The
implementation leverages TensorFlow 2's Keras API for model building, tf.data
for data pipeline construction, and model.fit for training. The repository
includes tools for training, evaluating, and deploying the model, and supports
integration with TensorFlow features such as Tensorboard and TensorFlow
Profiler. The model can be trained using datasets such as MJSynth and ICDAR, and
is capable of recognizing alphanumeric characters. Pre-trained models and
example scripts are provided for demonstrating the text recognition
capabilities. The project is licensed under the MIT license, promoting open-
source collaboration.

---



PAGE-XML: Document Image Content Format

https://github.com/PRImA-Research-Lab/PAGE-XML (Relevance: 1.0)

Keywords: PAGE-XML, document image, XML schema, layout analysis, dewarping, text content, document processing, PRImA Research Lab
Stages: layout-analysis, output-representation, evaluation
Architecture: 

PAGE-XML is a collection of XML formats designed for representing and processing
document image page content. This project, maintained by PRImA Research Lab,
facilitates the encoding of document layout information, such as text regions,
reading order, and text content, using a standardized XML schema. It also
supports layout analysis evaluation and document image dewarping, making it a
comprehensive tool for document analysis tasks. The formats are defined by an
XML schema officially hosted on primaresearch.org, ensuring consistency and
reliability. PAGE-XML is widely used for its ability to handle complex document
structures, supporting various applications in document digitization and
processing. The project is open source, licensed under the Apache-2.0 License,
and offers resources such as documentation to aid in its implementation.

---



Old Books Dataset for OCR Studies

https://github.com/PedroBarcha/old-books-dataset (Relevance: 1.0)

Keywords: OCR, dataset, old books, groundtruth, binarization, text recognition, scanned pages
Stages: preprocessing, text-recognition, dataset
Architecture: 

The 'Old Books Dataset' is a collection of scanned pages from old books, curated
for use in Optical Character Recognition (OCR) studies. The dataset includes
pages in various resolutions such as 300dpi, 500dpi, and 1000dpi, and offers
both noised and denoised versions using different binarization methods. The
groundtruth data for these pages is derived from Project Gutenberg ebooks, while
the original pages are sourced from PDFs available through the Internet Archive.
Some of the books included are 'Betrayed Armenia' by Diana Agabeg Apcar, 'The
Boy Apprenticed to an Enchanter' by Padraic Colum, and 'The Lusitania's Last
Voyage' by Charles E. Lauriat. This dataset serves as a valuable resource for
researchers and developers focusing on text recognition and historical document
analysis.

---



OCRopus Document Analysis Tools

https://github.com/tmbdev/ocropy (Relevance: 1.0)

Keywords: OCRopus, document analysis, OCR, Python tools, text recognition, CLSTM, layout analysis, open source, deep learning
Stages: preprocessing, layout-analysis, text-recognition
Architecture: lstm

OCRopus is a suite of Python-based tools designed for document analysis and
optical character recognition (OCR). It is not a standalone OCR system but
rather a collection of programs that require some preprocessing and potentially
new model training to apply them effectively to documents. The package includes
a variety of scripts for ground truth editing, error rate measurement, and
confusion matrix determination. Installation can be performed using system-wide
dependencies, Python virtual environments, or Conda. The tool comprises several
key functions including text binarization, page layout analysis, and text line
recognition, supporting model training for improved accuracy. OCRopus is
equipped to handle various document types and is continually evolving to
incorporate deep learning techniques for layout analysis and text recognition.
The project also offers CLSTM, a C++ based alternative for text line
recognition, providing faster processing with minimal dependencies. Users can
contribute by developing new command line tools and enhancing text/image
segmentation and detection capabilities.

---



CRNN TensorFlow OCR Implementation

https://github.com/Belval/CRNN (Relevance: 1.0)

Keywords: CRNN, TensorFlow, OCR, Convolutional Neural Network, Recurrent Neural Network, Text Recognition, Python, Machine Learning, Archived Repository
Stages: text-recognition, training-data-generation
Architecture: crnn

The CRNN project is a TensorFlow implementation of a Convolutional Recurrent
Neural Network (CRNN) used for Optical Character Recognition (OCR). This
archived repository provides a Python-based framework to recognize text from
images using a CRNN model. It was originally inspired by another project
available at https://github.com/bgshih/crnn. The implementation allows users to
generate training data using a TextRecognitionDataGenerator and train a CRNN
model using TensorFlow, with instructions to specify character sets and use
pretrained models for testing. Although functional, the project admits its
solution for OCR as being suboptimal and encourages users to explore other
implementations. The repository includes code, issues, pull requests, and
insights sections, but is now in a read-only state since its archiving on June
11, 2020.

---



AON: Text Recognition with Tensorflow

https://github.com/huizhang0110/AON (Relevance: 1.0)

Keywords: AON, Tensorflow, text recognition, CVPR 2018, arbitrarily-oriented, attention-based decoder, machine learning, feature extraction
Stages: text-recognition
Architecture: attentional, decoder-encoder

The AON project on GitHub implements the paper 'AON: Towards Arbitrarily-
Oriented Text Recognition', which was presented at CVPR 2018. This project
utilizes Tensorflow to create a system capable of recognizing text in various
orientations. The core technology involves extracting feature sequences from
four different directions and integrating them using an attention-based decoder
to produce a sequence of characters. This approach is particularly useful for
recognizing text that is not aligned horizontally, making it suitable for
diverse real-world applications. The repository contains the necessary code and
resources, including Python scripts for training, testing, and evaluating the
model. The project is open-source and licensed under the MIT License,
encouraging further development and collaboration.

---



OCR Project Using PyTorch

https://github.com/courao/ocr.pytorch (Relevance: 1.0)

Keywords: OCR, PyTorch, Text Detection, Text Recognition, CTPN, CRNN, Open Source, Image Processing, Deep Learning
Stages: segmentation, text-recognition
Architecture: convnet, rnn, crnn

The ocr.pytorch project is an open-source implementation of Optical Character
Recognition (OCR) using PyTorch. This project focuses on text detection and
recognition, leveraging two key models: CTPN (Connectionist Text Proposal
Network) for detection and CRNN (Convolutional Recurrent Neural Network) for
recognition. The project supports a variety of detection and recognition methods
and aims to expand its capabilities. It requires Python 3.5+, PyTorch 0.4.1+,
torchvision 0.2.1, OpenCV 3.4.0.14, and NumPy 1.14.3. Pre-trained models can be
downloaded and used for testing images, with results saved in a specified
directory. Training instructions and code are also provided to help users train
their models. The project is licensed under the MIT License, encouraging wide
usage and collaboration.

---



ARU-Net for Historical Document Analysis

https://github.com/TobiasGruening/ARU-Net (Relevance: 1.0)

Keywords: ARU-Net, historical documents, pixel labeling, text detection, U-Net, Tensorflow, layout analysis, document analysis, image segmentation
Stages: layout-analysis, segmentation, preprocessing, text-recognition
Architecture: unet

ARU-Net is a project focused on the layout analysis of historical documents
using neural pixel labeling. The repository provides Tensorflow code for a two-
stage method specifically designed for text line detection in historical
documents. ARU-Net extends the U-Net architecture to enhance its capabilities in
pixel labeling tasks, such as baseline detection. The repository offers features
such as a trained Tensorflow graph for inference, a full workflow for training
custom models, and various data augmentation strategies to minimize training
data needs. ARU-Net can be applied to multiple pixel labeling tasks, including
binarization and page segmentation. The project is compatible with Python 2.7
and Tensorflow versions above 1.0, and it comes with a demo for inference on
sample images from the cBad test set. Despite being archived, the repository
remains a valuable resource for researchers and developers interested in
document image analysis, particularly those dealing with historical documents.

---



CRAFT Text Detector

https://github.com/fcakyon/craft-text-detector (Relevance: 1.0)

Keywords: CRAFT, text detection, PyTorch, OCR, deep learning, computer vision, cross-platform, Python
Stages: segmentation, text-recognition
Architecture: convnet

The CRAFT Text Detector is a PyTorch implementation designed to detect text
areas through character-region awareness and affinity between characters. It
provides a packaged, easy-to-use, and cross-platform solution for text detection
tasks. By analyzing each character region and their relationships, the tool
calculates bounding boxes for text regions. Users can install the package via
pip and utilize it with simple Python scripts. It supports both basic and
advanced usage, allowing users to perform text detection and export results. The
repository includes features like model loading, prediction, and exporting
detected text regions and additional results like heatmaps. The project is
licensed under the MIT License and has been archived as a read-only repository
since December 2022.

---



DAVAR-Lab OCR Toolbox

https://github.com/hikopensource/davar-lab-ocr (Relevance: 1.0)

Keywords: OCR, text detection, text recognition, document understanding, open-source, DAVAR Lab, Hikvision, mmdetection, machine learning
Stages: segmentation, text-recognition, layout-analysis, evaluation
Architecture: convnet, lstm, attentional, crnn

DAVAR-Lab OCR is an open-source Optical Character Recognition (OCR) toolbox
developed by Hikvision Research Institute's DAVAR Lab. The project provides
implementations of various OCR algorithms and modules, including text detection,
text recognition, text spotting, video text spotting, document understanding,
and more. Notable algorithms featured in the toolbox include EAST, MASK RCNN,
Attention, CRNN, ACE, SPIN, RF-Learning, and others. The toolbox is built on top
of open-source frameworks such as mmdetection and mmcv, ensuring compatibility
with other tools in the open-mmlab ecosystem. The repository is designed for
researchers and developers interested in using or contributing to OCR
technologies. It includes installation instructions, environment setup
guidelines, and a FAQ for troubleshooting. The software is released under the
Apache 2.0 license, and users are encouraged to cite the project in their
academic work if it aids their research.

---



CRAFT: Text Detection with PyTorch

https://github.com/clovaai/CRAFT-pytorch (Relevance: 1.0)

Keywords: CRAFT, text detection, PyTorch, OCR, Clova AI, character region, affinity, bounding boxes, pretrained model
Stages: segmentation, text-recognition
Architecture: convnet

The CRAFT-pytorch project is the official PyTorch implementation of the
Character Region Awareness for Text Detection (CRAFT) model developed by Clova
AI Research at NAVER Corp. CRAFT effectively detects text areas in images by
identifying individual character regions and the affinities between them. This
approach allows the model to create bounding boxes around text by finding
minimum bounding rectangles on binary maps after applying thresholds to
character regions and affinity scores. Although the repository does not include
training code due to intellectual property reasons, it provides a pretrained
model that can be used for testing and inference. The model supports various
datasets, including SynthText, IC13, IC15, IC17, and CTW1500, and can be used
for English and multi-language text detection. Users can run the pretrained
model by following instructions and using specific command-line arguments to
customize the inference process. The project has been released under the MIT
license and has gained significant attention, with over 3,000 stars and more
than 860 forks on GitHub.

---



Attention-based OCR using TensorFlow

https://github.com/emedvedev/attention-ocr (Relevance: 1.0)

Keywords: OCR, TensorFlow, text recognition, CNN, seq2seq, visual attention, machine learning, Python package, Google Cloud ML
Stages: text-recognition, training-data-generation
Architecture: convnet, lstm, decoder-encoder

The Attention-based OCR project provides a TensorFlow model designed for text
recognition in images, utilizing a combination of convolutional neural networks
(CNN) and sequence-to-sequence (seq2seq) models with visual attention
mechanisms. The model processes images by first applying a sliding CNN, followed
by a long short-term memory (LSTM) network, and then uses an attention-based
decoder for the final output. This project, available as a Python package, is
compatible with Google Cloud ML Engine and includes tools for creating TFRecords
datasets and exporting trained models as SavedModels or frozen graphs. The
repository offers a comprehensive guide to installation, usage, dataset
creation, model training, testing, visualization, and exporting. The model can
also be served via TensorFlow Serving to enable REST API integration. While the
current implementation supports TensorFlow 1.x, a future upgrade to TensorFlow 2
is planned. The project acknowledges contributions from Qi Guo and Yuntian Deng,
and it is released under the MIT license.

---



Text Renderer for OCR Training

https://github.com/Sanster/text_renderer (Relevance: 1.0)

Keywords: OCR, text images, deep learning, CRNN, synthetic data, image generation, GPU, Python, text effects
Stages: training-data-generation
Architecture: crnn

Text Renderer is a tool for generating text images tailored for training deep
learning Optical Character Recognition (OCR) models like CRNN. It supports both
Latin and non-Latin scripts, making it versatile for various language datasets.
The application allows users to produce synthetic text images by configuring
text effects such as perspective transformation, random cropping, and character
spacing. Users can run the tool on Ubuntu systems with Python and can further
enhance performance by leveraging GPU capabilities after compiling OpenCV with
CUDA. The tool offers a strict mode to handle font support issues for non-Latin
languages, ensuring all characters are represented accurately. Developers can
use the tool in debug mode to visualize detailed transformations and bounding
boxes. Text Renderer is open-source under the MIT license and is accompanied by
scripts to check font compatibility with character sets. The project is actively
maintained with ongoing updates to improve functionality and user experience.

---



CNN+LSTM+CTC TensorFlow OCR

https://github.com/watsonyanghx/CNN_LSTM_CTC_Tensorflow (Relevance: 1.0)

Keywords: OCR, TensorFlow, CNN, LSTM, CTC, Image Processing, Deep Learning, Character Recognition, Machine Learning
Stages: text-recognition
Architecture: convnet, lstm

The project 'CNN_LSTM_CTC_Tensorflow' is an implementation of Optical Character
Recognition (OCR) using a combination of Convolutional Neural Networks (CNN),
Long Short-Term Memory networks (LSTM), and Connectionist Temporal
Classification (CTC) within the TensorFlow framework. This approach processes
images to recognize characters of variable lengths, making it versatile for
different OCR applications. The CNN component extracts features from images,
which are then processed by the LSTM to predict character sequences, with CTC
used to interpret these sequences. The project demonstrated a high accuracy of
99.75% on a test dataset after training with 100,000 images. Users can modify
network architecture parameters and run the model on their data following the
provided instructions. The project is licensed under the MIT License, making it
open for community contributions and use.

---



docTR: Document Text Recognition Library

https://github.com/mindee/doctr (Relevance: 1.0)

Keywords: OCR, text recognition, deep learning, TensorFlow, PyTorch, document analysis, docTR, KIE predictor, image processing
Stages: segmentation, text-recognition, layout-analysis, preprocessing
Architecture: convnet, transformer

docTR (Document Text Recognition) is an OCR library designed to make text
recognition seamless and accessible using deep learning. It leverages TensorFlow
2 and PyTorch to provide efficient text detection and recognition capabilities.
The library supports end-to-end OCR with a two-stage approach: text detection
and text recognition. Users can choose different architectures for these tasks
and integrate the library into existing workflows. docTR is designed to handle
various document formats, including PDFs and images, and offers options to
manage rotated documents. It includes a KIE predictor for detecting multiple
classes in documents, enhancing flexibility beyond traditional OCR. Installation
is supported via pip for both TensorFlow and PyTorch, with additional options
for visualization and HTML modules. A demo app and Docker containers are
available for easy deployment, along with an example script for document
analysis. The library is open-source, distributed under the Apache 2.0 License,
and contributions are welcome.

---



Table Transformer for Document Table Extraction

https://github.com/microsoft/table-transformer (Relevance: 1.0)

Keywords: Table extraction, Deep learning, PDF, Images, PubTables-1M, GriTS, Object detection, Table structure, Functional analysis
Stages: segmentation, layout-analysis, table-recognition, evaluation, dataset
Architecture: transformer

Table Transformer (TATR) is a deep learning model designed for extracting tables
from unstructured documents such as PDFs and images. It leverages object
detection techniques to identify tables and their structures within these
documents. The project includes the official code for the PubTables-1M dataset,
which provides a comprehensive resource for training and evaluating models on
tasks such as table detection, table structure recognition, and functional
analysis. PubTables-1M features a vast collection of annotated document pages,
tables, and bounding boxes in both image and PDF coordinates. TATR also includes
the GriTS evaluation metric for assessing the accuracy of table structure
recognition. Pre-trained models are available, mainly trained on PubTables-1M
and FinTabNet datasets. The repository offers training scripts, evaluation
metrics, and an inference pipeline for users who wish to apply TATR to their
datasets. The project is open source and welcomes contributions under the MIT
license.

---



Keras-OCR: Text Detection and Recognition

https://github.com/faustomorales/keras-ocr (Relevance: 1.0)

Keywords: OCR, Keras, CRNN, CRAFT, text detection, text recognition, machine learning, deep learning, computer vision
Stages: segmentation, text-recognition
Architecture: convnet, crnn

Keras-OCR is a comprehensive tool for optical character recognition (OCR) that
combines the CRAFT text detection model with the Keras CRNN recognition model.
Designed to provide a flexible, high-level API, it allows users to train and
deploy a complete text detection and recognition pipeline. The library supports
Python 3.6+ and TensorFlow 2.0+. Installation can be done via PyPi or directly
from the GitHub repository. Keras-OCR simplifies the process of setting up an
OCR system by automatically downloading pretrained weights for both the detector
and recognizer. The tool can process batches of images, returning predictions in
the form of word and bounding box tuples. It offers a valuable comparison
against cloud-based OCR services, showcasing competitive precision and recall
metrics while operating with relatively low latency, especially when run on a
GPU. Keras-OCR is particularly suitable for academic and commercial applications
that require efficient and customizable OCR solutions. The library is open-
source and distributed under the MIT license, encouraging contributions and
modifications from the community.

---



TensorFlow OCR with Attention

https://github.com/pannous/tensorflow-ocr (Relevance: 1.0)

Keywords: OCR, TensorFlow, Attention, Image Processing, Text Recognition, Machine Learning, AI, Python, Open Source
Stages: text-recognition, segmentation, training-data-generation, evaluation
Architecture: attentional

The 'tensorflow-ocr' project leverages TensorFlow to perform Optical Character
Recognition (OCR) using attention mechanisms. The project includes a fully
functional OCR system, which can be used to detect text in images. It offers a
straightforward setup process and a variety of tools for text detection and
recognition. The repository includes scripts for training models, such as
'train_letters.py', which generates and trains on various font types, and
'train.py' for a comprehensive training experience. The project also integrates
the EAST text detector for real-world image processing. Users can evaluate the
system by using 'mouse_prediction.py' to detect text under the mouse pointer.
The codebase is entirely written in Python and is open-source, allowing for
community contributions and customization.

---



DeepLayout: Page Layout Analysis

https://github.com/leonlulu/DeepLayout (Relevance: 1.0)

Keywords: Deep learning, Page layout analysis, Semantic segmentation, Python, TensorFlow, DeepLab, Image processing, Layout detection, Machine learning
Stages: layout-analysis, segmentation
Architecture: convnet

DeepLayout is a Python-based tool for performing page layout analysis using deep
learning techniques. The tool segments page images into different regions and
classifies them using a semantic segmentation model, specifically DeepLab_v2.
The primary objective is to predict a pixel-wise probability map and apply post-
processing to generate bounding boxes with corresponding labels and confidence
scores. It supports only Python2 and requires TensorFlow, along with other
libraries like Cython, numpy, and scikit-image. The tool comes with pre-trained
models for two datasets: CNKI and POD, allowing it to distinguish between text
and non-text regions, and further classify figures, tables, equations, and text.
The project provides functionality for running the code via command line or
importing as a module in Python, with outputs saved in JSON format. The
repository includes detailed instructions for setup, usage, and output
interpretation.

---



CNN+LSTM OCR with TensorFlow

https://github.com/weinman/cnn_lstm_ctc_ocr (Relevance: 1.0)

Keywords: OCR, TensorFlow, CNN, LSTM, CTC, text recognition, deep learning
Stages: text-recognition, dataset, training-data-generation
Architecture: convnet, lstm

This project showcases a TensorFlow-based implementation of a CNN+LSTM
architecture trained with Connectionist Temporal Classification (CTC) loss for
Optical Character Recognition (OCR). The model is an adaptation of Shi et al.'s
CRNN architecture, utilizing CNN features as input for a bidirectional stacked
LSTM, achieving a lower word error rate than CRNN on the MJSynth dataset. The
architecture includes enhancements such as deeper early convolutions, batch
normalization, and reduced convolutional parameters, improving character
recognition capabilities. Training involves using the MJSynth dataset with pre-
trained model checkpoints available for download. The project supports dynamic
training data and features a closed vocabulary recognition mode. It is designed
for Python 2.7 and requires TensorFlow version 1.10 or higher. The codebase
includes scripts for training, validation, and testing, with configurations
available through command-line options. The work is detailed in the ICDAR 2019
paper by Weinman et al., with acknowledgments to the National Science Foundation
for support.

---



Attention-OCR: Visual Attention Based OCR

https://github.com/da03/Attention-OCR (Relevance: 1.0)

Keywords: OCR, visual attention, CNN, LSTM, TensorFlow, Keras, text recognition, deep learning, machine learning
Stages: text-recognition, evaluation, dataset
Architecture: convnet, lstm, attentional

Attention-OCR is an Optical Character Recognition (OCR) system utilizing a
visual attention mechanism to improve text recognition from images. Developed by
Qi Guo and Yuntian Deng, this project combines Convolutional Neural Networks
(CNNs) and Long Short-Term Memory (LSTM) networks to process images resized to a
height of 32 pixels while preserving their aspect ratios. The model uses a
sliding CNN to extract features from the images, followed by an LSTM network,
with an attention model serving as the decoder to produce the final text
outputs. The framework is built primarily on TensorFlow, with Keras used for the
CNN component, and optionally incorporates a Python package for calculating edit
distances during evaluation. The repository provides a comprehensive guide for
training the model using the Synth 90k dataset, as well as testing and
visualizing the results using standard datasets like ICDAR03 and SVT. The
project is licensed under the MIT License, making it accessible for further
development and deployment.

---



Texify: Math OCR to LaTeX Converter

https://github.com/VikParuchuri/texify (Relevance: 0.9)

Keywords: OCR, LaTeX, Markdown, MathJax, Deep Learning, Texify, Image Conversion, Math OCR
Stages: text-recognition, math-recognition, output-representation, preprocessing
Architecture: 

Texify is an advanced Optical Character Recognition (OCR) model specifically
designed to convert images and PDFs containing mathematical expressions into
markdown and LaTeX formats. This tool is capable of running on CPU, GPU, or MPS
systems and is particularly useful for rendering mathematical equations via
MathJax. It can handle both block equations and inline equations mixed with
text. Compared to similar tools like pix2tex and nougat, Texify offers a broader
range of functionalities by supporting a diverse set of image types and is
trained on a wider dataset. It provides an interactive application for selecting
and converting equations and supports various outputs, including KaTeX-
compatible rendering. Texify is open-source and available for commercial use
under the CC BY-SA 4.0 license, integrating advancements from projects like
im2latex and Donut. Despite its capabilities, Texify has limitations regarding
image cropping and resolution dependencies. Its training involved 96 DPI images,
which may affect performance on images outside this specification.

---



dhSegment: Historical Document Processing Framework

https://github.com/dhlab-epfl/dhSegment (Relevance: 0.9)

Keywords: historical documents, document processing, deep learning, segmentation, tensorflow, dhSegment, EPFL, open source
Stages: segmentation, layout-analysis, text-recognition
Architecture: convnet

dhSegment is a versatile tool designed for processing historical documents.
Developed by Benoit Seguin and Sofia Ares Oliveira at EPFL's DHLAB, it employs a
generic deep-learning approach to segment regions and extract content from
various document types. The framework is built using Python, primarily utilizing
TensorFlow, and is tailored to handle the unique challenges presented by
historical data. Users can explore its capabilities through a demo that allows
training and application of the tool using a script. Comprehensive documentation
is available on ReadTheDocs, providing installation instructions and usage
examples. The project has been recognized in academic settings, with its
methodology detailed in a paper presented at the 16th International Conference
on Frontiers in Handwriting Recognition in 2018. The software is distributed
under the GPL-3.0 license, encouraging open-source collaboration and adaptation.

---



Deep Learning for Page Layout Segmentation

https://github.com/watersink/ocrsegment (Relevance: 0.9)

Keywords: deep learning, page segmentation, OCR, machine learning, TensorFlow, Python, MDLSTM, neural networks, document analysis
Stages: segmentation, layout-analysis, preprocessing, evaluation
Architecture: lstm

The OCR Segmentation project is a deep learning model designed for page layout
analysis and segmentation. This project, housed on GitHub under the repository
'watersink/ocrsegment,' utilizes TensorFlow 1.8 and is implemented in Python 3.
The model is aimed at efficiently segmenting page layouts, a crucial task in
optical character recognition (OCR) systems. It uses a dataset called
'uw3-framed-lines-degraded-000' and involves several processes including data
preprocessing, training, and testing using Python scripts such as
'data_pre_process.py,' 'train_test.py,' and 'segmentation.py.' The methodology
draws on multi-dimensional recurrent neural networks (MDLSTM), as referenced in
several academic papers. The repository has garnered interest with 100 stars and
27 forks, indicating a growing community of developers and researchers
interested in improving OCR systems. The project does not have any packaged
releases yet, but the code is available for use and contribution by the open-
source community. It is especially useful for developers focused on machine
learning and computer vision tasks related to document digitization.

---



Lasciva Roma: OCR Models for Latin Texts

https://github.com/lascivaroma/lexical (Relevance: 0.9)

Keywords: OCR, Latin, Kraken, Lexical resources, Digitization, 19th-century texts, Machine-readable, Open-source, Creative Commons
Stages: text-recognition, dataset, evaluation
Architecture: 

Lasciva Roma is a project focused on creating and evaluating OCR (Optical
Character Recognition) models tailored for 19th-century Latin lexical resources.
The repository provides training and evaluation data used for developing OCR
models with the Kraken OCR system. It includes transcriptions of historical
texts, such as 'Manuel de Synonymie Latine' by Ludwig von Doederlein, published
in 1873. The project aims to facilitate the digitization and accessibility of
Latin texts by providing ground truth data and models that can be used to
convert scanned images of texts into machine-readable formats. The repository is
licensed under the Creative Commons Attribution 4.0 International License,
encouraging collaboration and further development by the community.
Contributions can be made by submitting pull requests or contacting the project
maintainer.

---



Text Spotting Transformers Project

https://github.com/mlpc-ucsd/TESTR (Relevance: 0.9)

Keywords: Text Spotting, Transformers, CVPR 2022, Machine Learning, Image Processing, PyTorch, Detectron2, Computer Vision
Stages: segmentation, text-recognition, evaluation
Architecture: transformer

The TESTR (Text Spotting Transformers) project is an official implementation of
a paper presented at CVPR 2022 by Xiang Zhang, Yongwen Su, Subarna Tripathi, and
Zhuowen Tu. This project focuses on leveraging transformer-based architecture
for text spotting tasks, which involves detecting and recognizing text in
images. The repository provides detailed instructions on setting up the required
environment, including dependencies such as CUDA 11.3, Python 3.8, PyTorch
1.10.1, and Detectron2. It also offers guidance on preparing necessary datasets
like TotalText, CTW1500, and ICDAR2015. Users can visualize network predictions,
train models from scratch or finetune them using pretrained weights, and
evaluate model performance using various configurations. The repository includes
pretrained models with performance metrics on different datasets, showcasing the
effectiveness of the transformer-based approach for text detection and
recognition. The project is released under the Apache License 2.0 and
acknowledges contributions from AdelaiDet and Deformable-DETR for providing
standardized frameworks and implementations.

---



Awesome SynthText Repository

https://github.com/TianzhongSong/awesome-SynthText (Relevance: 0.9)

Keywords: synthetic data, text recognition, text location, OCR datasets, SynthText, curated list, machine learning, text detection
Stages: training-data-generation, dataset
Architecture: 

The "awesome-SynthText" repository, curated by Tianzhong Song, offers a
comprehensive list of synthetic data resources primarily focused on text
location, recognition, and OCR datasets. It includes links to various projects
and datasets like SynthText, SynthText Chinese version, and CurvedSynthText for
text location. For text recognition, it features resources such as Chinese OCR
synthetic data and TextRecognitionDataGenerator. Additionally, the repository
provides links to numerous OCR datasets including ICDAR competitions, SynthText
in the Wild, and the MSRA Text Detection Database. This collection is valuable
for developers and researchers working on text recognition and location projects
by providing access to a variety of synthetic datasets and tools.

---



OCR4all: Open-Source OCR Tool

https://github.com/OCR4all/OCR4all (Relevance: 0.9)

Keywords: OCR, open-source, historical texts, automation, web application, OCR4all, document analysis
Stages: layout-analysis, text-recognition, evaluation
Architecture: crnn

OCR4all is an open-source project providing Optical Character Recognition (OCR)
services through web applications. It is designed to allow users, especially
those without technical expertise, to perform OCR on a wide variety of
historical texts, including the earliest printed books. The tool focuses on a
semi-automatic workflow that balances manual interaction with automation to
achieve high-quality results. OCR4all is part of a collaborative effort with the
OCR-D project, aiming to enhance mass full-text recognition of historical
materials. The project is built using technologies like Docker, Maven, Spring,
Materialize, and jQuery, and includes components like OCRopus, Calamari, and
LAREX for document analysis and layout recognition. The project is actively
developed with frequent updates and has received funding from initiatives like
the DFG-funded OCR-D and the BMBF Project “Kallimachos.”

---



OCRopus3: PyTorch OCR System

https://github.com/NVlabs/ocropus3 (Relevance: 0.9)

Keywords: OCR, PyTorch, OCRopus3, NVlabs, archived, optical character recognition, repository, submodules, installation
Stages: text-recognition
Architecture: 

OCRopus3 is a repository that serves as a collection of submodules for an
optical character recognition (OCR) system based on PyTorch. Developed by
NVlabs, this project aimed to provide a comprehensive solution for OCR tasks by
integrating various components into a cohesive system. The repository was
archived in February 2021, indicating that it is no longer actively maintained.
It is intended to be replaced by OCRopus4, which is a rewritten version
utilizing PyTorch 1.7. Users interested in the latest developments should refer
to the OCRopus4 repository. To set up OCRopus3, users could clone the repository
and run the install script. Despite its archival status, the repository garnered
a moderate level of interest, with 142 stars and 37 forks on GitHub.

---



PytorchOCR: OCR Library with Pytorch

https://github.com/WenmuZhou/PytorchOCR (Relevance: 0.9)

Keywords: Pytorch, OCR, Text Detection, Text Recognition, Model Conversion, PaddleOCR, Deep Learning, ONNX, Inference
Stages: segmentation, text-recognition, evaluation
Architecture: convnet

PytorchOCR is an open-source Optical Character Recognition (OCR) library based
on the Pytorch framework. It is designed to support a variety of text detection
and recognition algorithms commonly used in OCR tasks. This repository
facilitates the conversion of models from PaddleOCR to PytorchOCR, providing
users with a toolkit for integrating OCR functionalities into their
applications. With PytorchOCR, developers can perform end-to-end inference,
including detection, recognition, and classification of text. The library
supports training, evaluation, and inference processes and allows for model
export to the ONNX format for further deployment. The project includes detailed
configurations for aligning multiple OCR models, ensuring compatibility between
the Paddle and Pytorch frameworks. PytorchOCR is a versatile library suitable
for developers looking to implement or experiment with OCR solutions
efficiently.

---



TableNet: End-to-end Table Detection

https://github.com/jainammm/TableNet (Relevance: 0.9)

Keywords: TableNet, Deep Learning, Table Detection, Tabular Data Extraction, Scanned Document Images, Semantic Segmentation, Tesseract OCR
Stages: preprocessing, segmentation, table-recognition
Architecture: encoder-decoder

TableNet is an unofficial implementation of a deep learning model designed for
detecting tables and extracting tabular data from scanned document images.
Originally proposed by a team at TCS Research, the architecture aims to
facilitate the extraction of tabular information captured through mobile phones
or cameras. The model employs an encoder-decoder architecture based on Long et
al.'s semantic segmentation model, and uses Tesseract OCR for preprocessing
images. The solution accurately detects the regions of tables in images and
extracts information from rows and columns. Users are directed to download the
Marmot Dataset for training and testing, and follow a Jupyter Notebook for
execution. The project requires a system with a powerful GPU to achieve accurate
results on high-resolution images.

---



Textline Segmentation Using FCN

https://github.com/beratkurar/textline-segmentation-using-fcn (Relevance: 0.9)

Keywords: textline segmentation, FCN, deep learning, Theano, Python, image processing, open-source, machine learning, GitHub
Stages: preprocessing, segmentation
Architecture: convnet

This project, 'textline_segmentation_using_fcn', is a repository hosted on
GitHub that focuses on text line segmentation using a Fully Convolutional
Network (FCN). The main objective is to segment text lines in challenging
datasets, leveraging deep learning techniques with a focus on the FCN
architecture. The project provides a full pipeline for preprocessing, training,
and postprocessing text line images, including the use of Theano for GPU-
accelerated computations. Users are guided through setting up their environment
with necessary installations such as upgrading Keras and Theano, and installing
OpenCV and pygpu for handling GPU tasks. The repository includes Python scripts
for preprocessing data, training the model, making predictions, and
postprocessing the results. The project is open-source under the MIT license,
allowing for broad use and collaboration. Although the repository lacks a
detailed description or documentation, the provided scripts and installation
instructions form a comprehensive guide to utilizing the FCN approach for text
line segmentation.

---



CRNN Torch Implementation

https://github.com/yisongbetter/crnn (Relevance: 0.9)

Keywords: CRNN, Torch, OCR, image recognition, CNN, RNN, CTC, text recognition, deep learning
Stages: text-recognition
Architecture: convnet, rnn

This project provides a Torch implementation of the Convolutional Recurrent
Neural Network (CRNN), designed for image-based sequence recognition tasks such
as scene text recognition and Optical Character Recognition (OCR). The CRNN
combines Convolutional Neural Networks (CNN), Recurrent Neural Networks (RNN),
and Connectionist Temporal Classification (CTC) loss to effectively recognize
sequences in images. The project includes instructions for building and running
demos, utilizing pretrained models, and training new models on custom datasets.
The software is primarily tested on Ubuntu 14.04 with CUDA-enabled GPUs and
requires Torch7, fblualib, and LMDB. It also includes a Docker setup for
simplified deployment. The implementation is open-source under the MIT license.

---



CRNN TensorFlow Implementation

https://github.com/AimeeKing/crnn-tensorflow (Relevance: 0.9)

Keywords: CRNN, TensorFlow, neural network, OCR, text recognition, machine learning, deep learning, AI, pre-trained model
Stages: text-recognition, training-data-generation
Architecture: crnn

This project implements the Convolutional Recurrent Neural Network (CRNN) using
TensorFlow. CRNNs are effective for sequence prediction tasks, such as optical
character recognition (OCR), where they combine the strengths of convolutional
layers for feature extraction and recurrent layers for sequence prediction. The
repository includes a demo script, 'demo_mjsyth.py', which demonstrates text
recognition on an input image using a pre-trained model. Users can also train a
new model by converting their datasets into TFRecords. The original CRNN
implementation can be found on GitHub under the repository 'crnn'.

---



OCR with Densenet for Text Recognition

https://github.com/yinchangchang/ocr_densenet (Relevance: 0.9)

Keywords: OCR, DenseNet, text recognition, deep learning, PyTorch, image processing, AI competition, F1 score, data imbalance
Stages: preprocessing, text-recognition, evaluation
Architecture: convnet

The 'ocr_densenet' project is a deep learning model designed to recognize text
from images using the DenseNet architecture. It was developed as part of the
2018 AI Practice Competition at Xi'an Jiaotong University, where it achieved
first place. The model takes an input image of size 64x512 and outputs a
probability distribution over 2159 characters for each 8x8 grid in the image. It
employs a unique loss balancing technique to handle data imbalance and uses
hard-mining for training. The model demonstrated high performance with an F1
score of 0.9911 on training data and 0.9582 on validation data. The project is
implemented in Python using PyTorch and requires Ubuntu 16.04, Python 2.7, and
CUDA 9.0 for execution. The repository includes scripts for data preprocessing,
training, and testing, with specific instructions for dataset handling and model
evaluation.

---



Nidaba: Scalable OCR Pipeline

https://github.com/OpenPhilology/nidaba (Relevance: 0.9)

Keywords: OCR, pipeline, automation, scalability, Tesseract, Kraken, Ocropy, Celery, digitization
Stages: preprocessing, segmentation, text-recognition
Architecture: 

Nidaba is a comprehensive, expandable, and scalable OCR (Optical Character
Recognition) pipeline designed to automate the conversion of raw images into
digitized texts. It serves as the central controller for the OGL OCR pipeline,
facilitating seamless processing through various OCR engines. Key
functionalities include grayscale conversion, binarization using adaptive
thresholding techniques like Sauvola and Otsu, deskewing, dewarping, and
integration with OCR engines such as Tesseract, Kraken, and Ocropy.
Additionally, Nidaba supports page segmentation and post-processing utilities
like spell-checking and ground truth comparison. Designed for scalability,
Nidaba employs a common storage medium on network-attached storage and the
Celery distributed task queue, allowing efficient operation across multi-machine
clusters. It can be easily installed from PyPi, and its development version is
available through its GitHub repository. Users can configure the system with
YAML files and initiate jobs via the Celery daemon, tracking job statuses and
retrieving outputs through command-line utilities. Nidaba is licensed under
GPL-2.0, promoting open-source collaboration and innovation.

---



TensorFlow OCR Implementation

https://github.com/BowieHsu/tensorflow_ocr (Relevance: 0.9)

Keywords: OCR, TensorFlow, Deep Learning, Text Recognition, Image Processing, Machine Learning, Python, Dice Coefficient, SoftMax
Stages: text-recognition, evaluation, dataset
Architecture: convnet

This project, hosted on GitHub by BowieHsu, is an implementation of Optical
Character Recognition (OCR) using TensorFlow version 1.4. The repository
includes code for setting up and running OCR models, with a focus on using
TensorFlow for processing and recognizing text in images. Key components of the
project involve tests for various functions such as Dice Coefficient, SoftMax,
and Focal Loss, which are critical for evaluating the performance of the OCR
models. The project structure includes datasets, network configurations, and
tools necessary for training and testing the OCR models. It provides scripts for
both training and testing the models, leveraging Python as the primary
programming language, with some shell scripts for automation. The repository is
open for collaboration and has a few forks and stars, indicating community
interest and participation.

---



Calamari OCR Engine

https://github.com/Calamari-OCR/calamari (Relevance: 0.9)

Keywords: OCR, Calamari, Python, Optical Character Recognition, OCRopy, Kraken, Open Source, Machine Learning, Text Recognition
Stages: text-recognition
Architecture: 

Calamari is an OCR (Optical Character Recognition) engine built on top of OCRopy
and Kraken using Python 3. It is designed to be user-friendly from the command
line while also offering modularity for integration and customization through
Python scripts. Calamari facilitates text recognition tasks and provides
flexibility for researchers and developers to adapt its functionalities to
specific needs. The project is open-source and licensed under the Apache-2.0
license. It offers pretrained models to enhance its functionality, which are
available for download. The tool is equipped with a command-line interface and a
comprehensive API, making it accessible for both individual and enterprise use.
Documentation and further instructions for installation are available online.
Calamari is actively maintained, with contributions from multiple developers,
and it is widely used in the field of digital humanities and other OCR-related
applications.

---



TableTrainNet: Document Table Recognition

https://github.com/mawanda-jun/TableTrainNet (Relevance: 0.9)

Keywords: Table recognition, Neural networks, OCR, Tensorflow, Document analysis, Image processing, Dataset preparation, Object detection, Machine learning
Stages: segmentation, layout-analysis, evaluation
Architecture: convnet

TableTrainNet is a project designed to recognize tables within documents using
neural networks. The project leverages Tensorflow's pre-trained object detection
models and is aimed at providing an intelligent OCR solution specifically for
table detection. It utilizes datasets from the ICDAR 2017 POD Competition and
supports potential integration with other datasets like the UNLV and Marmot
datasets. The project pipeline involves converting images to grayscale,
preparing datasets for Tensorflow, and training the neural network. Once
trained, the model can be tested using an inference pipeline that generates
results and logs for evaluation. Necessary libraries include Python 3,
Tensorflow, Pillow, OpenCV, and Pandas. The project is open source under the MIT
license.

---



DeepLearning-OCR on GitHub

https://github.com/vinayakkailas/Deeplearning-OCR (Relevance: 0.9)

Keywords: DeepLearning, OCR, GitHub, Text Recognition, Machine Learning, Optical Character Recognition, Document Processing, AI, Code Repository
Stages: text-recognition
Architecture: 

The 'Deeplearning-OCR' project on GitHub is designed to provide Optical
Character Recognition (OCR) capabilities using deep learning techniques. The
project aims to assist in the automatic recognition and conversion of different
types of documents and images into machine-encoded text. This is useful for
applications that require reading and processing large volumes of textual data
from various formats. The repository includes code, documentation, and resources
that developers can use to integrate OCR functionalities into their own
applications. The project leverages modern deep learning frameworks to enhance
the accuracy and efficiency of text recognition tasks.

---



CRNN for Image Sequence Recognition

https://github.com/bgshih/crnn (Relevance: 0.9)

Keywords: CRNN, OCR, image recognition, sequence recognition, CNN, RNN, CTC loss, image-based text, Torch7
Stages: text-recognition
Architecture: convnet, rnn

The Convolutional Recurrent Neural Network (CRNN) is designed for image-based
sequence recognition tasks such as optical character recognition (OCR) and scene
text recognition. It combines Convolutional Neural Networks (CNNs), Recurrent
Neural Networks (RNNs), and Connectionist Temporal Classification (CTC) loss to
effectively recognize sequences from images. The project includes a demo for
recognizing text in images using a pretrained model and provides instructions
for building, running, and training with CRNN using Docker and Torch7. The
software is primarily tested on Ubuntu 14.04 and requires CUDA-enabled GPUs.
Additionally, it offers a PyTorch port and an implementation combining CTPN and
CRNN for end-to-end text detection and recognition.

---



TextRecognitionDataGenerator Overview

https://github.com/Belval/TextRecognitionDataGenerator (Relevance: 0.9)

Keywords: OCR, synthetic data, text recognition, image generation, non-Latin text, Python module, Docker, text distortion, handwritten text
Stages: training-data-generation
Architecture: 

TextRecognitionDataGenerator is a tool for generating synthetic data to train
OCR (Optical Character Recognition) software. It is designed to create text
image samples, supporting both Latin and non-Latin scripts. Users can install it
via a PyPI package or use a Docker image for easier deployment. The tool
provides various customization options, such as text stroke width, color, word
splitting, font selection, and output masks. It can generate images with
different text orientations, distortions, and backgrounds, including handwritten
text simulation. Moreover, it allows for the inclusion of new languages by
adding corresponding fonts and dictionary files. The tool is useful for creating
training datasets for OCR applications, and offers both CLI and Python module
interfaces for integration into workflows. It is open-source and licensed under
the MIT license.

---



FOTS: Fast Oriented Text Spotting

https://github.com/Masao-Taketani/FOTS_OCR (Relevance: 0.9)

Keywords: OCR, Text Spotting, TensorFlow, Deep Learning, Computer Vision, Image Recognition, Scene Text Recognition, FOTS, Oriented Text
Stages: segmentation, text-recognition, training-data-generation, evaluation
Architecture: convnet

The FOTS_OCR project is a TensorFlow implementation of FOTS (Fast Oriented Text
Spotting), a unified network designed for efficient text detection and
recognition in images. The project focuses on spotting oriented text using deep
learning techniques, primarily leveraging TensorFlow. It supports pre-training
using the SynthText dataset and fine-tuning with datasets like ICDAR 2015, 2017
MLT, and 2013, utilizing models such as ResNet-50 for feature extraction. The
repository provides scripts and instructions for pre-training, fine-tuning, and
testing the models. The implementation is tested on various TensorFlow versions,
with plans to support TensorFlow 2.x in the future. The project is open-source
under the GPL-3.0 license, and the code is primarily written in C++ and Python.
It is suitable for researchers and developers interested in OCR, computer
vision, and deep learning applications.

---



LSTM CTC OCR with TensorFlow

https://github.com/ilovin/lstm_ctc_ocr (Relevance: 0.9)

Keywords: LSTM, CTC, OCR, TensorFlow, Machine Learning, Image Processing, Deep Learning, warpCTC, Python
Stages: text-recognition
Architecture: lstm

The 'lstm_ctc_ocr' project utilizes Long Short-Term Memory networks (LSTM) and
Connectionist Temporal Classification (CTC) for Optical Character Recognition
(OCR) tasks using TensorFlow. The project is designed to handle the challenges
of variable-length sequence data in OCR, allowing for the recognition of
characters in images with varying widths by padding them to uniform sizes. The
repository provides scripts for training the model on custom datasets, using
Python 3 and TensorFlow 1.0.1, along with dependencies like the 'captcha'
package and warpCTC TensorFlow bindings. The project is structured into
different branches, including a beta version that generates data on-the-fly and
manages multi-width images. The accuracy of this approach can exceed 95%,
demonstrating its effectiveness in OCR applications. Users can adjust various
parameters such as learning rate and image height to fine-tune the model for
specific use cases.

---



PaddleOCR: Multilingual OCR Toolkit

https://github.com/PaddlePaddle/PaddleOCR (Relevance: 0.9)

Keywords: PaddleOCR, OCR, multilingual, PaddlePaddle, deep learning, data annotation, model deployment, open source, AI
Stages: preprocessing, segmentation, layout-analysis, text-recognition, training-data-generation, dataset
Architecture: convnet, lstm, transformer

PaddleOCR is an advanced, multilingual OCR toolkit developed using the
PaddlePaddle deep learning framework. It supports the recognition of over 80
languages and provides tools for data annotation and synthesis. PaddleOCR is
designed to be a practical, ultra-lightweight system that can be deployed on
various platforms including servers, mobile devices, embedded systems, and IoT
devices. The project offers a comprehensive suite of features, including support
for the latest OCR algorithms, a full pipeline from data production to model
training, compression, and deployment. Notably, PaddleOCR includes industry-
level models such as PP-OCR, PP-Structure, and PP-ChatOCRv2, aiming to
streamline OCR processes across different applications. The toolkit is supported
by a vibrant community and offers extensive documentation to facilitate ease of
use. It is licensed under the Apache-2.0 license, ensuring open-source access
and contribution.

---



FOTS: Fast Oriented Text Spotting

https://github.com/xieyufei1993/FOTS (Relevance: 0.9)

Keywords: FOTS, text spotting, PyTorch, text detection, ICDAR, unified network, oriented text, machine learning
Stages: segmentation, text-recognition, evaluation
Architecture: convnet

FOTS (Fast Oriented Text Spotting) is a PyTorch-based implementation focused on
detecting text in images using a unified network. This project re-implements the
detection part of the original FOTS paper, which provides a method for spotting
and recognizing text in arbitrary orientations efficiently. The implementation
supports models trained on datasets like ICDAR 2015 and 2017. Users can train
the model by specifying a dataset path containing images and ground truth text
files, and test it using provided scripts. The repository includes various
scripts and configuration files necessary to facilitate training and evaluation.
It is primarily written in C++ and Python, with a small portion in Makefile, and
has gained moderate community attention with 173 stars and 41 forks.

---



OCR-D Segment Layout Analysis

https://github.com/OCR-D/ocrd_segment (Relevance: 0.9)

Keywords: OCR-D, page segmentation, layout analysis, OCR, image processing, open-source, Python
Stages: segmentation, layout-analysis, evaluation
Architecture: 

OCR-D/ocrd_segment is a repository that offers OCR-D compliant processors for
layout analysis and evaluation. It is designed to handle various tasks related
to page segmentation in optical character recognition (OCR) processes. The tools
within this project allow for exporting segmented images, importing layout
segmentations from different formats, and post-processing or repairing layout
segmentations. Additionally, it supports comparing different layout
segmentations. The repository emphasizes modularity and compliance with OCR-D
specifications, facilitating integration into larger OCR workflows. Installation
is straightforward, requiring a Python environment, and the project is open-
source, licensed under the MIT license.

---



SBB Binarization Tool

https://github.com/qurator-spk/sbb_binarization (Relevance: 0.9)

Keywords: Document Binarization, OCR, Image Processing, Python, TensorFlow, Open Source, Qurator
Stages: preprocessing
Architecture: 

The SBB Binarization project is an open-source tool designed for document image
binarization. It provides a method to convert images of documents into binary
images, which are crucial for tasks like Optical Character Recognition (OCR).
The tool supports Python versions 3.7 to 3.10 and TensorFlow versions up to
2.11.1. Users can install it via PyPI or by cloning the repository from GitHub.
The project offers pre-trained models available in different formats, which can
be downloaded for use. The tool outputs binary images in TIFF or PNG formats and
offers an OCR-D interface for integration with OCR workflows. The project is
available under the Apache-2.0 license and has received contributions from
multiple developers. It is part of the Qurator initiative, which focuses on
improving digital document processing.

---



Image Table OCR: Convert Images to CSV

https://github.com/eihli/image-table-ocr (Relevance: 0.9)

Keywords: OCR, image processing, PDF extraction, CSV conversion, table detection, Python, Tesseract, Poppler, ImageMagick
Stages: segmentation, layout-analysis, text-recognition, table-recognition
Architecture: 

Image Table OCR is a Python package designed to extract tabular data from images
and PDFs, converting it into a CSV format. It utilizes various modules to detect
tables within images, extract and sequence the cells, and apply Optical
Character Recognition (OCR) to read the text. The package is compatible with
tools like Poppler, Tesseract, and ImageMagick, which are essential for image
processing and text extraction. Image Table OCR supports a full workflow from
PDF to CSV by chaining its modular components, allowing users to automate the
conversion of complex table images into usable data formats. This tool can be
particularly useful for digitizing printed documents or analyzing scanned
reports, where manual data entry would be time-consuming. The project is open-
source and available under the MIT license, encouraging contributions and
modifications.

---



DewarpNet: Document Unwarping Network

https://github.com/cvlab-stonybrook/DewarpNet (Relevance: 0.9)

Keywords: DewarpNet, document unwarping, 3D regression, deep learning, ICCV 2019, OCR accuracy, doc3D dataset, image processing, open-source
Stages: preprocessing, evaluation, dataset
Architecture: convnet

DewarpNet is a project that provides code for implementing a single-image
document unwarping solution using stacked 3D and 2D regression networks.
Developed as part of a paper presented at ICCV 2019, it utilizes deep learning
techniques to correct distortions in scanned or photographed documents. The
repository contains training scripts, evaluation tools, and pre-trained models
for users to reproduce results from the paper or apply the solution to other
datasets. The project includes a dataset called doc3D, which can be downloaded
for experimentation. DewarpNet's approach improves document readability and OCR
accuracy by unwarping images to their original layout. The project also
highlights the impact of different Matlab versions on SSIM scores and provides
OCR evaluation metrics for model performance assessment. Users can access more
resources, such as a demo and the project's webpage, for further exploration.
The code is open-source under the MIT license, encouraging contributions and
usage in related research.

---



Ocrodeg: Document Image Degradation Library

https://github.com/NVlabs/ocrodeg (Relevance: 0.9)

Keywords: document image degradation, data augmentation, OCR, handwriting recognition, image transformation, Python library, random distortion, NVlabs
Stages: preprocessing, training-data-generation
Architecture: 

Ocrodeg is a Python library developed by NVlabs for simulating document image
degradation, primarily aimed at enhancing data augmentation processes for
handwriting recognition and Optical Character Recognition (OCR) applications.
The library provides a variety of degradation techniques, such as page rotation,
random geometric transformations, random distortions, ruled surface distortions,
and imaging artifacts like blur, thresholding, and noise. These techniques allow
users to simulate real-world document imperfections, making it easier to train
more robust OCR models. Ocrodeg enables users to apply transformations like
random rotation, scaling, translation, and anisotropic distortion to document
images, offering a realistic depiction of document handling and processing
errors. Furthermore, it provides utilities to model imaging artifacts such as
blurring, noise, ink spread, and fibrous noise, which are common in scanned or
photocopied documents. These features make Ocrodeg a valuable tool for
researchers and developers working on OCR systems, allowing them to create more
resilient models by training on a diverse set of augmented data.

---



ocr_attention: Text Recognition with Attention

https://github.com/marvis/ocr_attention (Relevance: 0.9)

Keywords: OCR, text recognition, attention mechanism, neural networks, image processing, GitHub project, pattern recognition
Stages: text-recognition
Architecture: transformer, attentional

The 'ocr_attention' project on GitHub focuses on implementing a text recognition
system utilizing neural network-based attention mechanisms. This approach
enhances the accuracy and efficiency of Optical Character Recognition (OCR)
systems by allowing the model to focus on relevant parts of the input image when
decoding text. The system can be applied to various applications requiring text
recognition from images, such as document digitization, license plate
recognition, and more. The project is hosted on GitHub, providing code and
resources for developers interested in integrating or improving text recognition
capabilities in their applications.

---



CRNN-TensorFlow Implementation

https://github.com/caihaoye16/crnn (Relevance: 0.9)

Keywords: CRNN, TensorFlow, text recognition, neural network, image processing, sequence prediction, demo, pre-trained model
Stages: text-recognition
Architecture: crnn

This project provides an implementation of a Convolutional Recurrent Neural
Network (CRNN) using TensorFlow. CRNNs are particularly effective for sequence
prediction tasks that involve spatial dependencies, such as reading text from
images. The project includes a demo script that demonstrates how to use a pre-
trained CRNN model to recognize text content from an example image. Users are
advised to download a pre-trained model and place it in the specified directory
for the demo to function. Additionally, instructions are provided for converting
datasets to TFRecords format for training new models. The original CRNN software
can be found on GitHub, linked in the project description.

---



Document Image Binarization with Auto-Encoder

https://github.com/ajgallego/document-image-binarization (Relevance: 0.9)

Keywords: document binarization, auto-encoder, image processing, machine learning, pattern recognition, OCR, DIBCO, training, Python
Stages: preprocessing, training-data-generation, dataset
Architecture: vae

The 'Document Image Binarization' project utilizes a selectional auto-encoder
approach to enhance the binarization of document images. This method involves
processing images to distinguish text from the background, which is crucial for
improving the readability of degraded documents and optimizing OCR processes.
The repository provides scripts for both binarization and training of models
using various datasets, such as those from the DIBCO competitions. The
`binarize.py` script allows users to process images with pre-trained models,
while `train.py` facilitates the training of new models with configurable
parameters. The models are trained to enhance document clarity by learning from
labeled datasets, and the project is tied to a publication in 'Pattern
Recognition' journal, providing a comprehensive framework for document image
processing.

---



CRNN-TF: Image Sequence Recognition

https://github.com/shoaibahmed/CRNN-TF (Relevance: 0.9)

Keywords: CRNN, TensorFlow, CNN, RNN, CTC loss, image recognition, sequence recognition, Inception ResNet v2
Stages: text-recognition
Architecture: convnet, rnn

CRNN-TF is a project that implements a Convolutional Recurrent Neural Network
(CRNN) in TensorFlow for the purpose of image-based sequence recognition. The
system integrates Convolutional Neural Networks (CNN) and Recurrent Neural
Networks (RNN) along with Connectionist Temporal Classification (CTC) loss to
transcribe images into sequences. It leverages a pretrained Inception ResNet v2
model for feature extraction. These features are then processed by the RNN to
execute the sequence recognition task. Although the implementation of the system
is currently incomplete, it represents a combination of advanced neural network
architectures aimed at improving the accuracy and efficiency of sequence
recognition in images.

---



Awesome Deep Text Detection Recognition

https://github.com/hwalsuklee/awesome-deep-text-detection-recognition (Relevance: 0.9)

Keywords: text detection, text recognition, deep learning, optical character recognition, OCR, machine learning, AI, data resources, research papers
Stages: text-recognition, segmentation, dataset
Architecture: 

The 'Awesome Deep Text Detection Recognition' project is a curated list of
resources focusing on text detection and recognition using deep learning
methods, particularly optical character recognition (OCR). This repository
provides a comprehensive collection of papers, tools, and datasets aimed at
advancing research and development in the field of text detection and
recognition. It includes resources on various topics such as text detection,
text recognition, end-to-end text recognition, and additional related works. The
project categorizes papers by publication date and provides links to related
resources, including code repositories and datasets. With contributions from the
OCR team at Clova AI powered by NAVER-LINE, the repository is regularly updated
to reflect the latest advancements and trends in AI conferences.

---



Chinese Textline Recognition with CRNN

https://github.com/qiaohan/crnn-train-tf (Relevance: 0.9)

Keywords: CRNN, text recognition, Chinese characters, CNN pretraining, RNN, LSTM, TensorFlow, OCR, deep learning
Stages: text-recognition
Architecture: convnet, lstm, crnn

The CRNN-Train-TF project is a comprehensive toolkit designed for the training
and testing of Convolutional Recurrent Neural Network (CRNN) models,
specifically aimed at recognizing Chinese textlines containing over 4000
characters. The repository offers scripts for various stages of the CRNN
pipeline, including CNN pretraining, CRNN initialization, and testing both with
and without Recurrent Neural Network (RNN) layers, such as a one-layer
bidirectional LSTM. The setup begins with pretraining a CNN, exporting its
weights to initialize the CRNN, and performing training and testing tasks using
files formatted with image filenames and character indices. The project also
includes a character dictionary file, 'word_dict.txt', which lists recognizable
characters including Chinese characters, numbers, English letters, and symbols.
Instructions for exporting CRNN weights for deployment on C++ server
implementations with cuDNN or TensorFlow Servering are provided, along with
Python scripts for training and testing CRNN models with and without RNNs.

---



TensorFlow CRNN for Text Recognition

https://github.com/solivr/tf-crnn (Relevance: 0.9)

Keywords: TensorFlow, CRNN, text recognition, OCR, deep learning, machine learning, image processing, neural networks, open source
Stages: text-recognition
Architecture: convnet, rnn, crnn

The tf-crnn project provides an implementation of a Convolutional Recurrent
Neural Network (CRNN) for image-based sequence recognition tasks, such as scene
text recognition and Optical Character Recognition (OCR). Based on TensorFlow
2.0, it utilizes `tf.keras` and `tf.data` modules to construct the model and
manage input data. The project facilitates training on datasets like the IAM
handwriting database, requiring users to have accounts and credentials for data
access. Installation of the project is simplified through the use of a provided
`environment.yml` file, enabling users to set up the necessary environment with
dependencies like `tensorflow-gpu`, CUDA, and cuDNN. The repository includes
scripts for data preparation and model training, alongside detailed
documentation to guide users through the setup and execution processes. The
project is licensed under GPL-3.0 and has a substantial community with 291 stars
and 98 forks on GitHub.

---



Chinese OCR with CTPN and DenseNet

https://github.com/YCG09/chinese_ocr (Relevance: 0.8)

Keywords: Chinese OCR, CTPN, DenseNet, Tensorflow, Keras, CTC, Text Recognition, Machine Learning, Open Source
Stages: segmentation, text-recognition, training-data-generation
Architecture: convnet

The project 'chinese_ocr' is an end-to-end solution for Chinese Optical
Character Recognition (OCR) implemented using Tensorflow and Keras. It leverages
a combination of CTPN (Connectionist Text Proposal Network) for text detection
and DenseNet with CTC (Connectionist Temporal Classification) for text
recognition. The OCR system is designed to handle variable-length Chinese
character recognition. The training data consists of approximately 3.64 million
images generated from a Chinese corpus, including newspapers and classical
Chinese texts, with randomly applied variations in font, size, grayscale, blur,
perspective, and stretching. The recognition model supports a diverse set of
characters, including Chinese characters, English letters, numbers, and
punctuation marks, with a total of 5990 characters. The training process
includes detailed steps for data preparation, training execution, and
performance evaluation. The project also provides a demo script to test the OCR
capabilities, saving results to a specified directory. The repository is open-
source and released under the Apache-2.0 license, inviting collaboration and
further development.

---



End-to-End TextSpotter Project

https://github.com/tonghe90/textspotter (Relevance: 0.8)

Keywords: TextSpotter, text detection, text recognition, Caffe, CVPR 2018, end-to-end, neural networks, image processing, open source
Stages: segmentation, text-recognition
Architecture: attentional

The TextSpotter project is an end-to-end text detection and recognition system
designed to identify text in images using explicit alignment and attention
mechanisms. Originally presented in a CVPR 2018 paper, this system employs Caffe
for its neural network framework. Users can install the system by cloning the
repository and setting up Caffe with specific configurations. For testing, a
pre-trained model is available for download, and users can execute test scripts
on sample images. The project includes features like multi-scale input support,
configurable hyperparameters, and a 90K-lexicon dictionary for text recognition.
Although the training code is partially provided, users must create custom
layers, such as an IOU loss layer, due to intellectual property restrictions.
The project is licensed under the GNU General Public License for non-commercial
use, with commercial inquiries directed to Chunhua Shen. Key topics include text
detection, text recognition, and end-to-end neural networks.

---



GRCNN for OCR Implementation

https://github.com/Jianfeng1991/GRCNN-for-OCR (Relevance: 0.8)

Keywords: GRCNN, OCR, Neural Network, CRNN, Character Recognition, Machine Learning, Deep Learning, Open Source, MIT License
Stages: text-recognition, evaluation
Architecture: crnn

This project provides the implementation of the Gated Recurrent Convolution
Neural Network (GRCNN) for Optical Character Recognition (OCR), as detailed in
the paper 'Gated Recurrent Convolution Neural Network for OCR.' The GRCNN is
built upon the CRNN architecture and employs a gated recurrent convolutional
approach to improve character recognition accuracy. Key requirements for
building the project include Ubuntu 14.04, CUDA 7.5, and CUDNN 5. The repository
offers pre-trained models and the necessary datasets for inference, along with
instructions for training new models using custom datasets. The visualization
tools within the project demonstrate the dynamic receptive fields in GRCNN,
showcasing the model's ability to differentiate between characters effectively.
The project is licensed under the MIT license and is available for public use
and contribution.

---



Deskew: Image Skew Correction Library

https://github.com/sbrunner/deskew (Relevance: 0.8)

Keywords: deskew, scanned documents, image processing, skew correction, Python library, command-line tool, scikit-image, OpenCV, debugging images
Stages: preprocessing
Architecture: 

Deskew is a Python library designed to correct skew in scanned documents,
ensuring text is horizontally and vertically aligned. The library detects the
skew angle of an image, which ranges between -45 and 45 degrees, and rotates the
image to negate this skew. Users can extend this range to -90 to 90 degrees by
setting a specific parameter. Deskew can be installed via pip and used both as a
command-line tool and within Python scripts, supporting integration with
libraries like scikit-image and OpenCV. Debugging tools are available to fine-
tune skew detection parameters, making it versatile for various skew detection
needs.

---



Text Detection and Translation Tool

https://github.com/s3nh/pytorch-text-recognition (Relevance: 0.8)

Keywords: text detection, translation, CRAFT, CRNN, neural networks, PyTorch, image processing, OCR, deep learning
Stages: segmentation, text-recognition
Architecture: crnn

The 'text-detector' project is a tool designed to detect and translate text from
images using machine learning techniques. It employs two primary neural network
architectures: CRAFT (Character Region Awareness for Text Detection) to identify
text regions and CRNN (Convolutional Recurrent Neural Network) to recognize the
text within those regions. The solution is implemented in Python and utilizes
libraries such as PyTorch and OpenCV for image processing and neural network
operations. Despite being a demonstration or 'toy example,' the system can
identify and translate individual words from images, showcasing the practical
use of neural networks in text recognition and translation applications. The
project is deployable on platforms like Heroku, although performance limitations
exist due to memory constraints on such platforms. Installation is supported on
both Windows and Linux, with specific dependencies outlined for each operating
system. The project encourages contributions and feedback from users to enhance
its capabilities and performance.

---



Tesseract OCR Engine Overview

https://github.com/tesseract-ocr/tesseract (Relevance: 0.8)

Keywords: Tesseract, OCR, Open Source, Machine Learning, LSTM, Image Recognition, Text Extraction, Command Line, C++ API
Stages: text-recognition
Architecture: lstm

Tesseract is an open-source Optical Character Recognition (OCR) engine developed
initially by Hewlett-Packard and later open-sourced by Google. This powerful
tool is capable of recognizing over 100 languages and supports various input and
output formats including PNG, JPEG, TIFF, PDF, and more. Tesseract 4 introduced
a neural network-based OCR engine focused on line recognition while maintaining
support for the legacy engine from version 3. It is primarily used via a
command-line interface and can be integrated into applications using its C or
C++ API. Tesseract does not include a graphical user interface but can be
trained to recognize new languages. The project is maintained by Zdenko Podobny
with contributions from a large community of developers. The software is
licensed under the Apache License, Version 2.0, and relies on the Leptonica
library for handling image files.

---



CRAFT Text Detection Remade

https://github.com/autonise/CRAFT-Remade (Relevance: 0.8)

Keywords: CRAFT, text detection, OCR, pytorch, weak-supervision, pre-trained models, computer vision, image processing, machine learning
Stages: segmentation, evaluation
Architecture: convnet

CRAFT-Remade is an implementation of the Character Region Awareness for Text
Detection (CRAFT) model, designed to detect text in images. The project focuses
on reproducing weak-supervision training as outlined in the original CRAFT
paper, allowing users to generate character bounding boxes on various popular
datasets. Users can leverage pre-trained models and a command-line interface to
synthesize results on custom images. The repository provides instructions for
setting up the environment using Conda or pip, and outlines steps for running
the model on custom images. Further, it details processes for training the model
from scratch, including strong and weak supervision training. Pre-trained models
for both supervision types are available, with links for downloading. The
project uses datasets such as ICDAR 2013, ICDAR 2015, ICDAR 2017, Total Text,
and MS COCO for evaluation. The implementation is primarily in Python and offers
insights into text detection using the CRAFT model, making it a valuable
resource for optical character recognition (OCR) and computer vision tasks.

---



Gold Standard OCR Data Repository

https://github.com/OpenITI/OCR_GS_Data (Relevance: 0.8)

Keywords: OCR, Gold Standard Data, Arabic, Persian, Turkish, OpenITI, Creative Commons, Data Repository, Optical Character Recognition
Stages: dataset
Architecture: 

The OCR_GS_Data repository, part of the OpenITI GitHub organization, provides a
collection of double-checked gold standard datasets for the training and testing
of Optical Character Recognition (OCR) engines. This repository includes data
for multiple languages such as Persian, Arabic, Ottoman Turkish, and Azeri
Turkish. The data is curated and validated to ensure accuracy, making it a
valuable resource for developers and researchers working on OCR technologies.
The repository is released under a Creative Commons Attribution-NonCommercial-
ShareAlike 4.0 International (CC BY-NC-SA 4.0) license, allowing for non-
commercial use and adaptation with proper attribution.

---



CLEval: Character-Level Evaluation Tool

https://github.com/clovaai/CLEval (Relevance: 0.8)

Keywords: CLEval, text detection, OCR, evaluation, character-level, text recognition, datasets, annotation types, TorchMetric
Stages: evaluation, evaluation-results
Architecture: 

CLEval is a tool designed for character-level evaluation in text detection and
recognition tasks. It provides a fine-grained assessment through a process that
matches instances and conducts evaluations at the character level. The tool
supports various annotation types including LTRB, QUAD, and POLY, and is
compatible with datasets like ICDAR 2013, ICDAR 2015, and TotalText. CLEval
offers a command-line interface for detection and end-to-end evaluations and
supports integration with TorchMetric for use in machine learning workflows. The
latest updates have enhanced its speed and added support for scale-wise
evaluation. The tool is available under the MIT license, and users can install
it via pip or build it from source. CLEval is aimed at improving the fairness of
evaluations in the optical character recognition (OCR) community, and the
developers encourage feedback and contributions.

---



shinTB: Text Detection with TensorFlow

https://github.com/shinjayne/shinTB (Relevance: 0.8)

Keywords: Textboxes, text detection, TensorFlow, deep learning, image processing, machine learning, Python, SVT dataset, tensorflow package
Stages: segmentation, text-recognition
Architecture: convnet

shinTB is a Python package designed for image text detection using the Textboxes
model, implemented with TensorFlow. It leverages the capabilities of the
Textboxes model, which is detailed in the corresponding paper, to detect text in
images. The package includes various components such as 'shintb' for core
functionalities, 'svt1' for accessing the Street View Text dataset, and
configuration files for setting up and training the model. Users can clone the
repository, configure the model using 'config.py', and utilize the 'shintb'
package to build and train their own Textboxes detection models. Training and
testing functionalities are provided through a comprehensive API that includes
instances for drawing graphs, controlling default boxes, and loading data. After
training, users can detect text in images using the 'Runner.image()' method. The
package requires Python 3.5.3, alongside dependencies like NumPy, TensorFlow,
and OpenCV.

---



Character Region Awareness for Text Detection

https://github.com/guruL/Character-Region-Awareness-for-Text-Detection- (Relevance: 0.8)

Keywords: text detection, character region, image processing, computer vision, SynthText, machine learning, Python, GitHub
Stages: segmentation, training-data-generation
Architecture: convnet

Character Region Awareness for Text Detection is a project hosted on GitHub,
aimed at developing a method for detecting text in images using character region
awareness. The repository provides a framework for training text detection
models on synthesized text data using a Python-based approach. The primary
script for training utilizes SynthText data, and although the project mentions
that training speed is limited by hardware constraints, initial results on
synthetic data are promising. The project is open-source and allows
contributions, with a to-do list that includes validation code and weakly-
supervised learning enhancements. The project has gained interest as evidenced
by its stars, forks, and watchers on GitHub.

---



TextSnake: Detecting Arbitrary Shaped Text

https://github.com/princewang1994/TextSnake.pytorch (Relevance: 0.8)

Keywords: TextSnake, PyTorch, text detection, arbitrary shapes, ECCV 2018, Megvii, TotalText, SynthText, curved text
Stages: segmentation, text-recognition
Architecture: convnet

TextSnake is a PyTorch implementation based on the ECCV 2018 paper titled
'TextSnake: A Flexible Representation for Detecting Text of Arbitrary Shapes'.
Developed by Megvii, this project provides a robust framework for accurately
identifying text in images, even when the text is curved or distorted. Unlike
traditional methods that struggle with non-linear text, TextSnake effectively
captures geometric properties such as location, scale, and curvature. The
implementation supports training and inference on popular datasets like
TotalText and SynthText. It is designed to work efficiently on systems with
NVIDIA GPUs and is compatible with Python 3.6 and the latest PyTorch versions.
The project also includes pre-trained models for improved performance and offers
a detailed guide for setting up and running the system. TextSnake's flexible
representation and comprehensive toolkit make it a valuable resource for
developers and researchers working on text detection in complex visual
environments.

---



Arabic OCR System

https://github.com/HusseinYoussef/Arabic-OCR (Relevance: 0.8)

Keywords: OCR, Arabic language, text recognition, machine learning, image processing, character segmentation, computer vision, neural networks, open source
Stages: text-recognition, segmentation, dataset
Architecture: 

The Arabic OCR project is an optical character recognition system designed to
convert images of typed Arabic text into machine-encoded text. It focuses
exclusively on Arabic script, supporting the 29 Arabic letters and the ligature
لا, but does not account for numbers or special symbols. The system processes
images by segmenting lines, words, and characters, and generates corresponding
text files. It achieves an average accuracy of 95% with a processing time of
approximately 16 seconds per image. Users can test the system by comparing the
output with ground truth data. The project includes a dataset of 1000 images for
training and is implemented entirely in Python. The code is open-source and
available under the MIT license.

---



Docstrum Algorithm for Document Segmentation

https://github.com/chulwoopack/docstrum (Relevance: 0.8)

Keywords: Docstrum, document segmentation, image processing, page layout analysis, historical documents
Stages: segmentation, layout-analysis, preprocessing
Architecture: 

The Docstrum project focuses on developing an algorithm for segmenting document
images into meaningful components, specifically targeting historical machine-
printed and handwritten documents. The algorithm, based on the work of Lawrence
O'Gorman, aims to analyze page layouts by processing document images through
several steps including pre-processing, clustering, spacing, orientation
estimation, and structural block determination. The process involves techniques
such as bilateral filtering, Otsu's thresholding, morphological operations, and
nearest-neighbor clustering to accurately determine text lines and structural
blocks. The code is primarily written in Python and Jupyter Notebook, utilizing
libraries like numpy and OpenCV for image processing tasks. The project builds
upon existing work by Chadoliver, with the source code available for public use
and contributions. While the project is still in development, with certain
evaluation and post-processing tasks marked as 'to be determined', it offers a
robust foundation for document image analysis using the Docstrum approach.

---



Skew Correction Tool

https://github.com/prajwalmylar/skew_correction (Relevance: 0.8)

Keywords: skew correction, image processing, computer vision, GitHub, OCR, image alignment, scanned documents
Stages: preprocessing
Architecture: 

This project, hosted on GitHub, is focused on providing a tool for correcting
skew in images. Skew correction is an important preprocessing step in image
processing and computer vision, often required for improving the accuracy of
subsequent analysis such as optical character recognition (OCR). The tool aims
to automatically detect and correct the skew angle of an image, thereby aligning
it properly. This can be particularly useful in applications dealing with
scanned documents or photos that need to be perfectly aligned for further
processing.

---



CRNN in PyTorch

https://github.com/meijieru/crnn.pytorch (Relevance: 0.8)

Keywords: CRNN, PyTorch, text recognition, convolutional network, recurrent network, machine learning, deep learning, neural networks, scene text
Stages: text-recognition
Architecture: crnn

The CRNN (Convolutional Recurrent Neural Network) project in PyTorch, developed
by meijieru, is designed to recognize text from images using a combination of
convolutional and recurrent neural network layers. It is inspired by the
original CRNN implementation by bgshih. The repository provides essential
scripts and resources for running demos, training new models, and utilizing
pretrained models. Users can run a demo using the provided 'demo.py' script
after downloading a pretrained model, which processes an example image to
recognize its text content. To train a new model, users need to prepare a
dataset and can use 'train.py' to start the training process. Dependencies for
this project include 'warp_ctc_pytorch' and 'lmdb'. The project is open-source
under the MIT license, and it has garnered significant attention with over 2.4k
stars and 658 forks on GitHub.

---



Textboxes TensorFlow Implementation

https://github.com/shinjayne/textboxes (Relevance: 0.8)

Keywords: Textboxes, TensorFlow, OCR, Python, SSD, MIT license, Text detection, OpenCV
Stages: segmentation, text-recognition
Architecture: convnet

The Textboxes project is a Python implementation using TensorFlow, focusing on
object character recognition (OCR) tasks. It is based on the architecture of
Single Shot MultiBox Detector (SSD) but tailored for text detection. The
repository includes various scripts for model training, data loading, and
matching boxes computation. It supports image processing functions like input
and output data pre- and post-processing. The project depends on TensorFlow r1.0
and OpenCV2, and is shared under the MIT license. The implementation aims to
provide robust text detection capabilities by incorporating additional default
boxes compared to standard SSD implementations.

---



CRNN with Tensorflow Support

https://github.com/chengzhang/CRNN (Relevance: 0.8)

Keywords: CRNN, Tensorflow, STN, multi-GPU, neural network, sequential data, convolutional, recurrent
Stages: text-recognition
Architecture: crnn

The CRNN (Convolutional Recurrent Neural Network) project is developed using
Tensorflow and supports multi-GPU environments. It optionally integrates a
Spatial Transformer Network (STN) to enhance its capabilities. The repository
includes various Python scripts for handling datasets, model configuration,
training, and multi-GPU support. The project is suitable for tasks that require
the combination of convolutional and recurrent neural networks for sequential
data processing. The STN component is adapted from the Tensorflow models
repository, showcasing its versatile application in spatial transformations. The
repository is actively maintained with contributions open for testing and
improvement.

---



TextProposals: Text Detection Algorithm

https://github.com/lluisgomez/TextProposals (Relevance: 0.8)

Keywords: TextProposals, text detection, word spotting, text extraction, image processing, machine learning, OpenCV, Caffe, object proposals
Stages: segmentation, text-recognition, evaluation, evaluation-results
Architecture: 

TextProposals is a project that implements the methods described in the research
papers 'TextProposals: a Text-specific Selective Search Algorithm for Word
Spotting in the Wild' and 'Object Proposals for Text Extraction in the Wild' by
Lluis Gomez and Dimosthenis Karatzas. These papers, published in 2016 and 2015,
present algorithms for detecting and proposing text regions in images,
particularly useful for word spotting and text extraction in challenging
environments. The implementation is capable of reproducing the results from the
SVT, ICDAR2013, and ICDAR2015 datasets. It uses a combination of OpenCV, Caffe,
and tinyXML for its functionality, and includes third-party code for clustering
and binomial coefficient approximations. The project supports end-to-end
evaluation using the DictNet_VGG model, which is necessary for running certain
evaluations. The methods are designed to produce bounding boxes around detected
text areas, ranked by confidence scores. The project offers tools for generating
proposals for text in images and provides scripts for evaluation of results
using MATLAB. If used in academic work, users are encouraged to cite the related
papers.

---



Ground Truth Files from BHL

https://github.com/impactcentre/groundtruth-bhl (Relevance: 0.8)

Keywords: ground truth, Biodiversity Heritage Library, digitisation, OCR, XML, IMPACT Centre, Creative Commons, text recognition, historical documents
Stages: dataset, evaluation
Architecture: 

The 'groundtruth-bhl' repository is a collection of ground truth files sourced
from the Biodiversity Heritage Library (BHL), provided by the IMPACT Centre of
Competence in Digitisation. These files are intended to support research and
development in the field of document digitisation, specifically for improving
the accuracy of text recognition systems. The repository includes various XML
files representing digitised historical documents, which can be used for
training and evaluating optical character recognition (OCR) systems. The dataset
is shared under the Creative Commons Attribution (CC-BY) license, allowing for
wide usage in academic and commercial projects. More details about the
collaboration and the project can be found on the associated website.

---



ALTO XML Documentation Overview

https://github.com/altoxml/documentation (Relevance: 0.8)

Keywords: ALTO XML, documentation, text digitization, schema, Creative Commons, METAe project, GitHub
Stages: output-representation
Architecture: 

The ALTO XML documentation repository offers a comprehensive guide on the ALTO
(Analyzed Layout and Text Object) XML format, which is used for describing the
layout and content of textual materials. This format is particularly useful in
digitizing and managing large-scale text documents. The repository includes
generic documentation, a summary of version changes, samples illustrating new
features and use cases, and software tools for creating and editing ALTO files.
Additionally, it lists organizations utilizing ALTO. The documentation is
organized into subfolders based on specific schema versions, and the latest
schema along with previous versions can be accessed within the repository. ALTO
XML is available under the Creative Commons Attribution-ShareAlike 4.0
International license. Originally developed during the EU-funded METAe project,
the format has contributions from various institutions including the University
of Graz and the University of Innsbruck. The repository aims to support
developers and organizations in effectively implementing the ALTO XML format in
their digital text processing workflows.

---



OCR Conversion Tools

https://github.com/cneud/ocr-conversion-scripts (Relevance: 0.8)

Keywords: OCR, conversion, ABBYY, ALTO, hOCR, PAGE XML, TEI XML
Stages: output-representation
Architecture: 

The 'OCR Conversion' project is a collection of scripts and stylesheets designed
for converting various Optical Character Recognition (OCR) formats. This
repository provides tools to facilitate the transformation of OCR data into
different standardized formats such as ABBYY FineReader XML, ALTO, hOCR, PAGE,
and TEI. It includes specific converters like ABBYY to hOCR, ALTO to TEI, and
hOCR to ALTO, among others. These tools are essential for digital libraries,
archives, and researchers who need to convert scanned document data into usable
and standardized formats for further processing or analysis. The repository
links to other related projects and tools, allowing users to expand
functionality and incorporate additional formats as needed.

---



ABBYY Cloud OCR SDK Overview

https://github.com/abbyysdk/ocrsdk.com (Relevance: 0.8)

Keywords: OCR, ABBYY, API, Text Recognition, Document Conversion, ICR, Data Extraction, Programming Languages, Web API
Stages: text-recognition, output-representation
Architecture: 

ABBYY Cloud OCR SDK is a web-based API service designed to facilitate optical
character recognition (OCR) and document conversion tasks. It supports a wide
range of programming languages, including C#, Java, Python, and more, making it
accessible to developers across various platforms. The SDK offers comprehensive
text recognition features such as full-page and zonal OCR for over 200
languages, ICR for hand-printed text, and barcode recognition. Additionally, it
enables document conversion into formats like searchable PDFs, Microsoft Word,
and Excel. The SDK's capabilities also extend to data extraction from business
cards and IDs. ABBYY Cloud OCR SDK provides two API versions, V1 with XML
response format and V2 with JSON response format, although the repository
samples currently support V1 only. Users can export recognized texts into
multiple file types, including TXT, RTF, DOCX, and PDF/A-1b. To use the service,
you need to register on the ABBYY Cloud OCR SDK website, download the sample
code, and implement applications using the provided examples and API
documentation.

---



PageXML C++ Library with Python Wrapper

https://github.com/omni-us/pagexml (Relevance: 0.8)

Keywords: PageXML, C++, Python wrapper, document processing, annotation, open-source, MIT license, Docker, text extraction
Stages: layout-analysis, output-representation
Architecture: 

The PageXML project, hosted on GitHub by omni-us, provides a library written in
C++ with a Python wrapper for processing Page XML files. Page XML is typically
used for representing structured document data, often in applications dealing
with document layout and content analysis. The project includes two main
components: the C++ library and a SWIG-based Python wrapper called py-pagexml,
which facilitates integration with Python environments. Additionally, it offers
the TextFeatExtractor library and its corresponding Python wrapper, py-textfeat,
for extracting textual features from documents. The project is open-source,
licensed under the MIT License, and supports cross-platform usage with Docker
integration. It includes comprehensive online documentation for both py-pagexml
and py-textfeat to assist users in implementation and deployment. The library is
maintained actively with several releases, the latest being in July 2021.

---



Dinglehopper OCR Evaluation Tool

https://github.com/qurator-spk/dinglehopper (Relevance: 0.8)

Keywords: OCR, evaluation, ALTO, PAGE XML, metrics, batch processing, error rate, Unicode, OCR-D
Stages: evaluation, output-representation
Architecture: 

Dinglehopper is a tool designed to evaluate Optical Character Recognition (OCR)
systems by comparing OCR outputs against ground truth documents. It supports
various document formats including ALTO and PAGE XML, and plain text files. The
tool generates reports detailing character and word differences, and computes
metrics such as Character Error Rate (CER) and Word Error Rate (WER).
Dinglehopper facilitates batch processing by aggregating and summarizing
multiple reports. It can be used as a command-line interface or as part of an
OCR-D workflow. The tool is designed for flexibility, offering options to enable
or disable metrics, and to specify text extraction levels. Installation is
straightforward using pip, and the tool supports Unicode, making it versatile
for different languages and scripts. Dinglehopper is suitable for use as a UI
tool, in automated evaluations, or as a library in larger systems.

---



Tobler-Lommatzsch OCR Project

https://github.com/PonteIneptique/toebler-ocr (Relevance: 0.8)

Keywords: OCR, Altfranzösisches Wörterbuch, Old French, Kraken, transcription, open-source, machine learning, digital humanities, Creative Commons
Stages: text-recognition, dataset, evaluation
Architecture: 

The 'Tobler-Lommatzsch: Altfranzösisches Wörterbuch' repository provides
training and evaluation data for the optical character recognition (OCR) of the
Altfranzösisches Wörterbuch, a dictionary of Old French. The project includes
scripts to facilitate the transcription of the dictionary into double column
HTML formats. It also features OCR models that have been trained and tested
using the provided data for the Kraken OCR system. This open-source project is
licensed under the Creative Commons Attribution 4.0 International License,
encouraging contributions from the community. Interested contributors can clone
the repository and submit pull requests or contact the maintainer via email. The
repository aims to aid in the digital transcription and accessibility of
historical French texts.

---



Caroline Minuscule OCR Ground Truth

https://github.com/rescribe/carolineminuscule-groundtruth (Relevance: 0.8)

Keywords: OCR, Caroline Minuscule, ground truth, manuscripts, OCRopus, historical text, transcription, medieval, public domain
Stages: dataset, output-representation
Architecture: 

The Caroline Minuscule OCR ground truth repository is an extensive collection of
manually curated data designed to support the development and training of OCR
(Optical Character Recognition) models, specifically targeting the Caroline
Minuscule script. This repository is part of the broader OCRopus project, which
aims to enhance OCR capabilities for historical manuscripts. The repository
contains images and corresponding ground truth text files, organized into
subdirectories for individual manuscripts and pages. Each page is further
segmented into lines with associated `.png` images and `.gt.txt` files suitable
for OCRopus and potentially other OCR engines. The repository also includes
full-color page images and ALTO XML format representations, which can be adapted
for other OCR tools like Kraken. The ground truth transcription follows specific
protocols that balance the peculiarities of medieval script and the limitations
of modern OCR software, including decisions on punctuation, capitalization,
abbreviations, spacing, and numeral representation. The repository is licensed
under Public Domain or Apache License 2.0, and it uses freely redistributable
manuscript images.

---



OcrOcIS: Optical Character Recognition

https://github.com/kaumanns/ocrocis (Relevance: 0.8)

Keywords: OCR, machine learning, digitization, text recognition, AI, document processing, GitHub, collaboration, automation
Stages: text-recognition
Architecture: 

OcrOcIS is a project focused on Optical Character Recognition (OCR), which is a
technology used to convert different types of documents, such as scanned paper
documents, PDFs, or images captured by a digital camera, into editable and
searchable data. The project aims to improve the accuracy and efficiency of OCR
processes, making it easier to digitize and process large volumes of text. This
can be particularly useful in industries like healthcare, finance, and
manufacturing where large amounts of paper-based information need to be
digitized and analyzed. The project leverages machine learning and artificial
intelligence to enhance the capabilities of OCR systems, potentially integrating
with platforms like GitHub for collaborative development and deployment of OCR
solutions.

---



Text Detection with Script ID

https://github.com/isi-vista/textDetectionWithScriptID (Relevance: 0.8)

Keywords: Text Detection, Script Identification, Convolutional Neural Network, Deep Learning, Keras, TensorFlow, Scene Text, Document Text
Stages: segmentation, text-recognition
Architecture: convnet

The 'textDetectionWithScriptID' project is a fully convolutional neural network
designed for text detection in both scene and document images, with support for
script identification. The repository hosts two main deep neural networks: the
pixel-level Text Detection Classification Network (TDCN) and the page-level
Script ID Classification Network (SICN). TDCN specializes in classifying text at
the word level in scene images and the line level in document images,
identifying pixels as non-text, border, or text. SICN classifies images into
scripts like Latin, Hebrew, Cyrillic, Arabic, and Chinese, among others. The
models were trained using the Keras library with TensorFlow as the backend. The
project provides various tools, including a simple and a lazy decoder for text
detection, and a command-line tool for more comprehensive usage. Dependencies
include Keras, TensorFlow, OpenCV-Python, and Skimage. The project is intended
for academic and non-commercial use, as outlined in its licensing terms, with
options for commercial licensing. It is based on the ICCV17 paper 'Self-
organized Text Detection with Minimal Post-processing via Border Learning' by Wu
and Natarajan.

---



Ocular: Historical OCR System

https://github.com/ndnlp/ocular (Relevance: 0.8)

Keywords: OCR, historical documents, unsupervised learning, multilingual, orthographic variation, code-switching, transcription, diplomatic form, normalized form
Stages: text-recognition
Architecture: 

Ocular is a cutting-edge optical character recognition (OCR) system designed for
historical documents. It features unsupervised learning to handle unknown fonts
and can process multilingual documents with code-switching at the word level.
Ocular is particularly adept at dealing with noisy historical texts that may
have inconsistent inking, spacing, and alignment. It can transcribe text into
both literal and normalized forms, learning orthographic variations such as
archaic spellings. The system is built to learn from document images and a text
corpus without the need for supervised training, making it highly flexible for
various historical document transcription tasks. Ocular has been documented in
several academic publications and its development has been supported by grants
focused on enhancing early-modern multilingual OCR capabilities.

---



Chamanti OCR: Multilingual OCR Framework

https://github.com/rakeshvar/chamanti_ocr (Relevance: 0.8)

Keywords: OCR, CRNN, CTC, TensorFlow, Keras, Multilingual, Telugu, Scripts, Neural Networks
Stages: text-recognition
Architecture: crnn

Chamanti OCR is a cutting-edge Optical Character Recognition (OCR) framework
designed to work with multiple languages, especially those with complex scripts.
The project primarily focuses on developing a robust OCR system that does not
rely on segmentation algorithms at the glyph level, making it particularly
suitable for highly agglutinative scripts like Arabic and Devanagari. Initially,
the framework is being developed for the Telugu language. Chamanti OCR utilizes
Convolutional Recurrent Neural Networks (CRNN) integrated with the Connectionist
Temporal Classification (CTC) loss function, leveraging TensorFlow 2.0 and
Keras. The project's core components include model building, training, and
utilities for processing OCR outputs. The setup requires TensorFlow and Lekhaka,
a package for generating complex text. Chamanti OCR aims to provide an advanced,
flexible, and language-agnostic OCR solution.

---



Polish Digital Libraries Ground Truth

https://github.com/impactcentre/groundtruth-pol (Relevance: 0.8)

Keywords: ground truth, Polish libraries, PSNC, digitization, CC-BY license, historical documents, OCR, digital humanities, cultural heritage
Stages: dataset
Architecture: 

The 'groundtruth-pol' repository on GitHub hosts ground truth files from various
Polish Digital Libraries, managed by PSNC (Poznań Supercomputing and Networking
Center). These files are part of the IMPACT project, which aims to improve and
facilitate access to historical documents through digitization. The ground truth
data is released under a Creative Commons Attribution (CC-BY) license, allowing
for wide usage and distribution with appropriate credit. This dataset is
valuable for various applications, including OCR (Optical Character Recognition)
training, historical research, and digital humanities projects. The repository
includes a variety of texts, each categorized into specific folders, reflecting
a diverse range of historical documents from Polish collections. This initiative
supports the enhancement of digital resources and the preservation of cultural
heritage by making these documents more accessible and usable in digital
formats.

---



Text Detection CTPN in TensorFlow

https://github.com/eragonruan/text-detection-ctpn (Relevance: 0.8)

Keywords: CTPN, text detection, TensorFlow, OCR, ID card, Python, machine learning, image processing, open source
Stages: segmentation, text-recognition
Architecture: convnet

The 'text-detection-ctpn' project by eragonruan is a text detection model
primarily based on the Connectionist Text Proposal Network (CTPN) implemented in
TensorFlow. The project aims to detect text in images, using ID card detection
as an example, though it is applicable to most horizontal scene text detection
tasks. The original CTPN paper is referenced, and the project is based on its
implementation in Caffe. Key components include a demo for inference using a
pre-trained model, details for setting up the environment, and instructions for
training the model with custom data. The project supports both horizontal and
oriented text detection modes, and uses Python, Cython, and CUDA for
implementation. Pre-trained models, sample training data, and detailed setup
instructions are provided to facilitate ease of use. The project is licensed
under the MIT license and has gained a significant community following with over
3,400 stars and 1,300 forks on GitHub.

---



ocrevalUAtion OCR Evaluation Tool

https://github.com/impactcentre/ocrevalUAtion (Relevance: 0.8)

Keywords: OCR, evaluation, tool, text comparison, University of Alicante, open-source, Apache-2.0
Stages: evaluation, output-representation
Architecture: 

ocrevalUAtion is a versatile tool developed by the University of Alicante for
evaluating the accuracy of Optical Character Recognition (OCR) systems. It
compares two text files: a reference file (ground-truth) and the OCR engine's
output. The tool offers customizable options, such as ignoring case, diacritics,
punctuation, stop-words, and defining character equivalences. Users can operate
it via a graphical user interface or command line. Supported input formats
include plain text, FineReader 10 XML, PAGE XML, ALTO XML, and hOCR HTML. The
tool generates reports with statistical metrics like Character Error Rate (CER)
and Word Error Rate (WER), and highlights differences between texts. It is
available for download on GitHub, and additional usage instructions can be found
in its wiki. The project is open-source under the Apache-2.0 license.

---



Geometric Augmentation for Text Images

https://github.com/Canjie-Luo/Scene-Text-Image-Transformer (Relevance: 0.8)

Keywords: text augmentation, image processing, geometric transformations, text recognition, CVPR 2020, augmentation tool, OpenCV, Python, robustness
Stages: preprocessing, text-recognition
Architecture: 

The 'Text Image Augmentation' project provides a geometric augmentation tool
specifically designed for text images, as described in the CVPR 2020 paper
'Learn to Augment: Joint Data Augmentation and Network Optimization for Text
Recognition'. This tool aims to enhance the robustness and reduce overfitting of
text recognition models by applying various geometric transformations, such as
distortion, stretch, and perspective changes to text images. The toolkit is
highly customizable, allowing users to adapt it to their specific needs. It has
been utilized in various research contexts, including the AAAI 2020 paper
'Decoupled Attention Network for Text Recognition' and the ICDAR 2019
competition, where it contributed to a winning model. The repository provides
installation instructions, requirements, and demo scripts to facilitate easy
integration into existing workflows. The augmentation process is efficient,
taking less than 3ms per image on a 2.0GHz CPU, and offers potential for speed
improvements through multi-process batch sampling. The tool significantly
improves text recognition accuracy, as demonstrated by experiments showing
increased performance across multiple datasets. It is available for academic
research under the MIT license.

---



Mask TextSpotter: Text Detection and Recognition

https://github.com/lvpengyuan/masktextspotter.caffe2 (Relevance: 0.8)

Keywords: Text Detection, Neural Network, Caffe2, Arbitrary Shapes, End-to-End Training, Computer Vision, Text Recognition, ICDAR Datasets
Stages: segmentation, text-recognition, evaluation
Architecture: convnet, rnn

Mask TextSpotter is an end-to-end trainable neural network designed for spotting
text with arbitrary shapes. This project is an official implementation that
utilizes the Caffe2 framework and is specifically tailored for tasks involving
text detection and recognition in diverse geometric configurations. The system
is capable of handling intricate text shapes, making it suitable for various
applications in computer vision and document analysis. The implementation
requires an NVIDIA GPU and Linux environment, with dependencies on Python2 and
Caffe2, among other standard Python packages. Users can install the system by
following the instructions provided for Caffe2, ensuring that the Detectron
module is included. The project supports testing and training on datasets like
ICDAR2013 and ICDAR2015, with instructions provided for dataset placement and
model training. The repository includes detailed steps for setting up the
environment, downloading necessary models, and running tests or training new
models. The project contributes to the field by providing a tool that enhances
text spotting capabilities in complex scenarios.

---



hOCR Tools for OCR Manipulation

https://github.com/tmbdev/hocr-tools (Relevance: 0.8)

Keywords: hOCR, OCR, HTML, text extraction, document processing, Python, PDF creation, command-line tools
Stages: segmentation, layout-analysis, text-recognition, output-representation, evaluation
Architecture: 

The hOCR Tools project provides a suite of command-line utilities designed for
manipulating and evaluating the hOCR format, which is used to represent multi-
lingual OCR results by embedding them into HTML. These tools facilitate various
operations on hOCR files, such as combining multiple pages into a single
document, splitting files into individual pages, calculating word frequencies,
and creating searchable PDFs from image and hOCR pairs. Additionally, the tools
offer functionality for evaluating OCR output against ground truth data,
assessing segmentation errors, and extracting text and image data from hOCR
files. The project is implemented in Python and can be installed system-wide
using pip or from source. It supports running in isolated environments via
virtualenv. The hOCR format is notable for its ability to integrate OCR data
seamlessly with HTML, allowing for extensive manipulation while retaining the
text and layout information. This makes hOCR Tools particularly useful for
researchers and developers working with OCR technologies and digital document
processing.

---



OCR Open Dataset Repository

https://github.com/xylcbd/ocr-open-dataset (Relevance: 0.8)

Keywords: OCR, datasets, printed text, handwritten text, open source, GitHub, Apache-2.0, machine learning, text recognition
Stages: dataset
Architecture: 

The 'ocr-open-dataset' repository hosted on GitHub is a comprehensive collection
of open datasets focused on Optical Character Recognition (OCR). This repository
categorizes datasets into printed, handwritten, and mixed printed and
handwritten text, providing a valuable resource for researchers and developers
working on OCR technologies. Notable datasets include COCO-Text, mnist, and The
Street View Text Dataset, among others. Each dataset entry includes the name,
year of release, and a link for access. The repository is publicly accessible
under the Apache-2.0 license and has garnered attention with over 100 stars and
13 forks. The repository aims to support advancements in OCR by providing a
centralized listing of datasets that can be used for training and testing OCR
models.

---



ALTO XML Schema Repository

https://github.com/altoxml/schema (Relevance: 0.8)

Keywords: ALTO XML, schema, OCR, optical character recognition, version 4.4, GitHub, open source
Stages: output-representation
Architecture: 

The ALTO XML Schema repository on GitHub contains the latest and previous
versions of the ALTO schema, which is used for defining the structure of OCR
(Optical Character Recognition) output. The repository provides access to both
draft and final released versions of the schema. The latest official version
available is 4.4, which includes updates such as the addition of language and
rotation attributes at the PageType level, and adaptations to the PointsType
documentation and xLink attributes. The schema is primarily sourced from the
Library of Congress's website, with an alternate source available via a raw
GitHub link. The repository also hosts discussions and issues related to ongoing
changes and improvements to the ALTO standard, providing a centralized location
for collaboration and tracking of updates. Documentation for the schema is
available in a separate documentation repository linked from the main project
page.

---



Awesome OCR Resources

https://github.com/soumendra/awesome-ocr (Relevance: 0.8)

Keywords: OCR, Tesseract, Open Source, Document Management, Deep Learning, APIs, Text Recognition, GitHub, Synthetic Data
Stages: text-recognition, training-data-generation
Architecture: 

The 'awesome-ocr' repository curated by the user 'soumendra' is a comprehensive
collection of open-source tools, libraries, and resources related to Optical
Character Recognition (OCR). This repository assembles various popular OCR
projects like Tesseract and its JavaScript and React Native wrappers, as well as
applications for document management and text extraction from different formats.
It also highlights APIs, both paid and open-source, that facilitate OCR
functionalities. Additionally, the repository provides a lit review of notable
research papers, synthetic data generators for OCR, and deep learning
implementations for text recognition. It serves as a valuable resource for
developers and researchers interested in OCR technology, offering links to
projects on GitHub, blog posts, and Q&A discussions that explore practical
applications and techniques to improve OCR accuracy.

---



Text Image Deskewing Tool

https://github.com/dehaisea/text_deskewing (Relevance: 0.8)

Keywords: text recognition, image processing, deskewing, OCR, Python, rotation, text images, alignment, open source
Stages: preprocessing
Architecture: 

The 'text_deskewing' project aims to enhance the detection and recognition of
text in images by automatically rotating images that are not properly aligned.
This tool is especially useful for preprocessing text images to improve the
accuracy of optical character recognition (OCR) systems. The repository provides
a Python-based implementation, where users can run 'rotation_demo.py' to see a
sample rotation and 'rotation.py' to process multiple images in batch mode. The
project is open source, hosted on GitHub, and written entirely in Python.

---



Document Image Skew Estimation

https://github.com/phamquiluan/jdeskew (Relevance: 0.8)

Keywords: document, image, skew, estimation, correction, Fourier, projection, Python, Docker
Stages: preprocessing
Architecture: 

The project 'jdeskew' provides a method for estimating and correcting the skew
in document images. It introduces an Adaptive Radial Projection on the Fourier
Magnitude Spectrum technique, which aims to determine the angle of skew in a
document image accurately. The tool can be integrated within Python environments
or utilized through Docker and Cog for more versatile use. It is particularly
designed to enhance the accuracy of text recognition systems by first ensuring
that the input images are properly aligned. Performance comparisons demonstrate
its superior accuracy and efficiency against other methods in the DISE 2021
dataset, showing significant improvements in Correct Estimation rate and Worst
Error metrics. The project is open-source, licensed under the MIT License, and
its implementation can be accessed and contributed to on GitHub. The repository
includes resources such as installation instructions, usage examples, and
reproducibility codes, making it accessible for developers and researchers
interested in document image processing.

---



OCR Test Data for Early Printed Books

https://github.com/chreul/OCR_Testdata_EarlyPrintedBooks (Relevance: 0.8)

Keywords: OCR, early printed books, OCRopus, historical texts, Latin books, character error rate, ground truth, model training, open source
Stages: text-recognition, training-data-generation, dataset, evaluation, evaluation-results
Architecture: 

The OCR_Testdata_EarlyPrintedBooks project provides a comprehensive set of test
lines from various early printed books, alongside individual and mixed OCRopus
models. The repository includes a mixed model trained on twelve Latin books
printed between 1471 and 1686, focusing primarily on works from before 1600.
This model, trained on over 8,600 lines, achieved a Character Error Rate (CER)
of 2.92% after 98,000 training steps. The project also offers seven early
printed books for evaluation, each with 150 lines of ground truth (GT) data and
an individual OCR model. These resources can be used for improving OCR accuracy
on historical texts, utilizing techniques like cross-fold training and transfer
learning. The data is licensed under a Creative Commons Attribution-
NonCommercial-ShareAlike 4.0 International License, and users are encouraged to
cite relevant publications when using the data.

---



LAREX: Layout Analysis Tool

https://github.com/OCR4all/LAREX (Relevance: 0.8)

Keywords: LAREX, layout analysis, region extraction, OCR, open-source, early printed books, PAGE XML, text segmentation, OCR4all
Stages: layout-analysis, segmentation, output-representation
Architecture: 

LAREX is a semi-automatic open-source tool designed for layout analysis and
region extraction from early printed books. Using a rule-based connected
components approach, it allows for fast and easily understandable processing,
with options for intuitive manual correction. It integrates with existing OCR
workflows through the PAGE XML format, offering an efficient and flexible
solution for segmenting pages of historical texts. LAREX can be installed via
Docker, Linux, Windows, or macOS setups, and is configured to handle images and
books locally or via web interfaces. The tool is part of the OCR4all project and
supports integration with broader OCR applications. It aims to streamline the
digitization process of early printed materials, preserving textual cultural
heritage.

---



TextBoxes-TensorFlow Re-Implementation

https://github.com/gxd1994/TextBoxes-TensorFlow (Relevance: 0.8)

Keywords: TextBoxes, TensorFlow, text detection, deep learning, image processing, machine learning, data processing, Python, Jupyter Notebook
Stages: segmentation, text-recognition
Architecture: convnet

TextBoxes-TensorFlow is a project that re-implements the TextBoxes model using
TensorFlow. The project is inspired by the TensorFlow slim project and
incorporates elements from the SSD-TensorFlow project. It aims to provide a
flexible and modularized framework for text detection in images. The project
involves various phases, including data processing and training. The data
processing phase is designed to handle datasets and transform them into a
suitable format for training. The training phase is customizable with options
for setting training directories, dataset directories, and other hyperparameters
like learning rate and batch size. The authors propose to improve visualization
and image processing aspects and plan to extend the training to other datasets,
fine-tune models, and automate dataset downloading. The project is authored by
Daitao Xing and Jin Huang.

---



Ocular: Historical OCR System

https://github.com/tberg12/ocular (Relevance: 0.8)

Keywords: OCR, historical documents, unsupervised learning, multilingual, code-switching, orthographic variation, digital humanities
Stages: text-recognition
Architecture: 

Ocular is an advanced Optical Character Recognition (OCR) system designed for
the transcription of historical documents. It employs unsupervised learning to
decipher unknown fonts using only document images and a text corpus, making it
capable of handling noisy documents with inconsistent inking and spacing. Ocular
supports multilingual documents, including those with code-switching, and can
learn orthographic variations such as archaic spellings. The system provides
transcriptions in both literal and normalized forms. Ocular has been detailed in
several publications, showcasing its ability to process early-modern
multilingual texts. The project is supported by the National Endowment for the
Humanities, under a Digital Humanities Implementation Grant.

---



VistaOCR: ISI's OCR Software

https://github.com/isi-vista/VistaOCR (Relevance: 0.8)

Keywords: OCR, machine-print, handwriting, CNN, LSTM, ISI, segmentation-free, neural networks, VistaOCR
Stages: text-recognition, segmentation
Architecture: convnet, lstm

VistaOCR is an Optical Character Recognition (OCR) software developed by ISI,
designed to recognize and process machine-printed and handwritten text data. The
project leverages advanced neural network architectures, including Convolutional
Neural Networks (CNNs) and Long Short-Term Memory networks (LSTMs), to perform
segmentation-free OCR. VistaOCR aims to efficiently increase resolution in
neural OCR models, making it suitable for various languages, including English,
French, and Arabic. The software has been discussed in several publications,
such as those presented at the IEEE Workshop on Arabic Script Analysis and
Recognition and the International Conference on Document Analysis and
Recognition. While pre-trained models and performance metrics are expected to be
made available soon, VistaOCR serves as a robust tool for researchers and
developers interested in state-of-the-art OCR technology.

---



EasyOCR: Multi-language OCR Library

https://github.com/JaidedAI/EasyOCR (Relevance: 0.8)

Keywords: OCR, EasyOCR, multi-language, text recognition, deep learning, JaidedAI, Python, open-source, optical character recognition
Stages: text-recognition
Architecture: convnet, lstm

EasyOCR is a ready-to-use Optical Character Recognition (OCR) library developed
by JaidedAI. It supports over 80 languages and includes scripts for widely used
writing systems such as Latin, Chinese, Arabic, Devanagari, and Cyrillic. Built
on deep learning frameworks, EasyOCR is designed for easy integration, offering
features like multi-language text recognition and compatibility with both CPU
and GPU execution. The library is continually updated, with recent improvements
adding Apple Silicon support and new text detection models. EasyOCR is available
for installation via pip and can be integrated into various applications using
Python. The project is open-source, allowing contributions and improvements from
the community, and is licensed under the Apache-2.0 license.

---



Awesome OCR Resources Collection

https://github.com/ZumingHuang/awesome-ocr-resources (Relevance: 0.8)

Keywords: OCR, Optical Character Recognition, Resources, Papers, Datasets, Text Recognition, Text Detection, Computer Vision, Deep Learning
Stages: dataset
Architecture: 

The 'Awesome OCR Resources' repository curated by ZumingHuang is a comprehensive
collection of resources related to Optical Character Recognition (OCR). This
repository aggregates various papers, datasets, and references essential for
researchers and developers working in the field of OCR. The resources are
systematically categorized by year, topic, and publication venues, covering a
wide range of OCR applications such as text detection, text recognition, and
document image understanding. Additionally, it includes links to related
projects and curated lists from other contributors in the field. The repository
aims to serve as a valuable reference for OCR development and research, offering
a structured overview of the most significant advancements and datasets
available in this domain.

---



InsightOCR: MXNet OCR Implementation

https://github.com/deepinsight/insightocr (Relevance: 0.8)

Keywords: OCR, MXNet, text recognition, text detection, CRNN, machine learning, image processing, deep learning, MIT License
Stages: text-recognition, segmentation, dataset
Architecture: crnn, lstm

InsightOCR is a project that provides an Optical Character Recognition (OCR)
implementation using the MXNet deep learning framework. The project focuses on
both text recognition and text detection, aiming to enhance the accuracy and
efficiency of processing textual data. It showcases impressive text recognition
accuracy on various datasets including Chinese datasets and the VGG_Text
dataset. For instance, the project reports a 99.73% accuracy using the SE-
ResNet34 model on a Chinese dataset. Additionally, different network
architectures such as SimpleNet, SE-ResNet50-PReLU, and SE-ResNeXt101-PReLU are
evaluated on their performance metrics like LSTM usage, pooling strategies, and
grayscale image processing. InsightOCR is licensed under the MIT license,
encouraging open-source contributions and further development. The repository
consists of key components like the CRNN (Convolutional Recurrent Neural
Network) folder, data generators, and training scripts for CRNN models,
indicating a comprehensive approach to OCR tasks. With a focus on Python, the
project invites collaboration and exploration in the field of machine learning-
driven text recognition.

---



PRLib: OCR Image Enhancement Library

https://github.com/leha-bot/PRLib (Relevance: 0.8)

Keywords: OCR, image processing, binarization, deskew, denoise, thinning, blur detection, deblur, OpenCV
Stages: preprocessing
Architecture: 

PRLib, or Pre-Recognize Library, is designed to enhance image quality for better
Optical Character Recognition (OCR) performance. The library includes a suite of
image processing algorithms that prepare images for recognition by improving
their quality. Key features include various binarization techniques such as
Global Otsu, Sauvola, and Niblack, as well as deskewing, denoising using Non-
local Means Denoising, and thinning through Zhang-Suen and Guo-Hall methods.
Additionally, PRLib offers blur detection and deblurring capabilities, white
balance adjustments, border detection, removal of perspective warp, and image
cropping. The library is dependent on OpenCV and Leptonica for its operations.
To build the library, users need to create a build directory, run cmake, and
then make. PRLib is open-source and available under the MIT license.

---



P2PaLA: Document Layout Analysis Toolkit

https://github.com/lquirosd/P2PaLA (Relevance: 0.8)

Keywords: document layout analysis, neural networks, PyTorch, image segmentation, handwritten text recognition, open-source, deprecated, P2PaLA, toolkit
Stages: segmentation, layout-analysis, output-representation
Architecture: convnet

P2PaLA (Page to PAGE Layout Analysis) is a toolkit designed for document layout
analysis using neural networks, notably compatible with PyTorch. The tool is
aimed at efficiently processing the layout of documents, particularly for tasks
involving image segmentation and handwritten text recognition. The toolkit is
built to support Linux and uses Python (versions 2.7 and 3.6 recommended) along
with other dependencies like Numpy, PyTorch, and OpenCV. P2PaLA leverages neural
networks to analyze document layouts and provides functionalities such as
baseline detection, training visualization through TensorBoard, and output
generation in XML-PAGE format. Although P2PaLA is deprecated, it includes
various examples and pre-trained models for users to experiment with. The
project originally aimed to advance the capabilities of document analysis by
automating layout tasks, which can be particularly beneficial for digitizing and
organizing large volumes of documents. Users interested in trying out its
features can access a demo and utilize the provided tools for visualization and
editing. The project is open-source under the GNU GPL-3.0 license.

---



STN-OCR: Text Detection and Recognition

https://github.com/Bartzi/stn-ocr (Relevance: 0.8)

Keywords: text recognition, text detection, neural network, deep learning, MXNet, OCR, STN-OCR, convolutional neural networks, FSNS
Stages: segmentation, text-recognition
Architecture: convnet

STN-OCR is a project that provides the source code for the paper 'STN-OCR: A
single Neural Network for Text Detection and Text Recognition'. This project
focuses on developing a neural network capable of both detecting and recognizing
text from images, simplifying the process into a single network architecture.
The repository includes code for training models on various datasets, such as
SVHN for house number recognition and FSNS for text recognition. The code is
structured to facilitate experimentation with different datasets and
configurations, and it employs MXNet as the deep learning framework.
Additionally, it provides scripts for downloading and preprocessing datasets, as
well as training and evaluation scripts for model performance assessment. The
project is licensed under the GPL-3.0 license and has a citation available for
academic use.

---



TensorFlow PSENet Reimplementation

https://github.com/liuheng92/tensorflow_PSENet (Relevance: 0.8)

Keywords: TensorFlow, PSENet, Text Detection, OCR, Machine Learning, Shape Robustness, Deep Learning, Open Source, Python
Stages: segmentation, text-recognition
Architecture: convnet

This project is a TensorFlow-based reimplementation of PSENet (Progressive Scale
Expansion Network), which is designed for robust text detection. PSENet aims to
improve text detection by addressing shape robustness through a progressive
scale expansion strategy. The repository provides code and instructions for
training, testing, and evaluating the model using datasets like ICDAR 2015 and
ICDAR2017 MLT. It supports TensorFlow versions above 1.0 and can be run on both
Python 2 and 3. The implementation includes tools for polygon shrinking using
the pyclipper module, and the ability to modify data formats for input. Users
can download pre-trained models for reference and further optimization. The
project includes detailed instructions for setting up the environment, training
the model, and conducting evaluations. It also recognizes contributions and
feedback from users to improve the codebase. The project is open-source and
licensed under the MIT license.

---



Deep Learning for Document Dewarping

https://github.com/thomasjhuang/deep-learning-for-document-dewarping (Relevance: 0.8)

Keywords: GAN, document dewarping, pix2pix, deep learning, image processing, Python, PyTorch, AI, OCR
Stages: preprocessing
Architecture: gan

The project 'Deep Learning for Document Dewarping' utilizes a generative
adversarial network (GAN) model, specifically pix2pixHD, to correct warping in
document images. The goal is to transform distorted document images to a flat,
readable state by training the GAN model on pairs of warped and unwarped
document images. This approach is inspired by related work such as DocUNet and
other deep learning models for image dewarping. The project requires a setup
with Python, PyTorch, and an NVIDIA GPU, among other dependencies. Users can
train the model using their own datasets by following specific file structure
guidelines and training commands provided in the documentation. The project
supports multi-GPU training and can leverage Automatic Mixed Precision for
faster execution. The repository offers scripts for both training and testing
the model, along with detailed instructions for setting up the environment and
prerequisites.

---



STKM: Self-attention Text Knowledge Mining

https://github.com/CVI-SZU/STKM (Relevance: 0.8)

Keywords: STKM, Text Detection, Self-attention, PyTorch, Computer Vision, CVPR 2021
Stages: segmentation, text-recognition
Architecture: transformer

STKM, or Self-attention based Text Knowledge Mining, is a tool designed for text
detection. It employs self-attention mechanisms to enhance the process of
extracting text knowledge from visual data. The project is implemented using
PyTorch, a popular machine learning library, and is available under the
Apache-2.0 license. The repository includes code for training and evaluating
models, as well as transferring and replacing model components. Pre-trained
models are accessible via Baidu Yun and Google Drive for ease of use. STKM is a
significant contribution to the field of computer vision, particularly in text
detection applications, and was presented at CVPR 2021. Researchers and
developers can integrate STKM into their projects to improve text detection
tasks, and they are encouraged to cite the work if it proves beneficial in their
research.

---



Tesseract-Recognize OCR Tool

https://github.com/mauvilsa/tesseract-recognize (Relevance: 0.8)

Keywords: OCR, Tesseract, layout analysis, text recognition, Page XML, Docker, command line, API, document recognition
Stages: layout-analysis, text-recognition, output-representation
Architecture: 

Tesseract-Recognize is a tool designed for layout analysis and text recognition
using the Tesseract OCR engine. It processes images to output results in the
Page XML format. This tool can be compiled from source on Ubuntu systems or used
via Docker for simpler deployment. It supports English by default, but
additional languages can be enabled by providing tessdata files. Tesseract-
Recognize can be used through a command-line interface or a REST API, making it
versatile for integration in various workflows. The results can be viewed or
edited with compatible tools, enhancing document recognition and layout analysis
capabilities.

---



doc2text: OCR for Poorly Scanned PDFs

https://github.com/jlsutherland/doc2text (Relevance: 0.8)

Keywords: OCR, PDF, text extraction, Python, scanning errors, OpenCV, Tesseract, doc2text, image processing
Stages: preprocessing, text-recognition
Architecture: 

doc2text is a Python module designed to enhance the quality of text extracted
from poorly scanned PDFs. The tool focuses on detecting text blocks and
correcting common scanning errors, which often degrade the results of optical
character recognition (OCR). Developed to assist researchers in obtaining high-
quality text from suboptimal scans, doc2text addresses issues such as misaligned
text, low resolution, and unwanted artifacts. The project is currently in an
early alpha stage, supporting Ubuntu 16.04 LTS, and is open to contributions and
feedback. It uses Python libraries like OpenCV, Tesseract, and PythonMagick to
process and improve the readability of scanned documents. Users can install the
package via pip and use it to read, process, and extract text from files in
formats including PDF, PNG, JPG, BMP, and TIFF. The module is licensed under the
MIT License, emphasizing its open-source nature.

---



OCR-DETECTION-CTPN for Image Text Detection

https://github.com/Li-Ming-Fan/OCR-DETECTION-CTPN (Relevance: 0.8)

Keywords: OCR, CTPN, Text Detection, CNN, LSTM, TensorFlow, Image Processing, Machine Learning, Natural Images
Stages: segmentation, text-recognition
Architecture: convnet, lstm

OCR-DETECTION-CTPN is a project utilizing a Convolutional Neural Network (CNN)
combined with Long Short-Term Memory (LSTM) networks, specifically designed for
detecting text in images. The project is implemented using TensorFlow and
follows the methodology described in the paper 'Detecting Text in Natural Image
with Connectionist Text Proposal Network' by Zhi Tian et al. The workflow
involves normalizing images, generating validation and training data, and
training the model using provided scripts. The process results in detecting text
regions in natural images, which are stored in specified directories. This
implementation is useful for tasks requiring robust text detection in various
image datasets.

---



CLSTM: C++ LSTM for OCR

https://github.com/tmbdev/clstm (Relevance: 0.8)

Keywords: CLSTM, C++, LSTM, OCR, Eigen, neural networks, Docker, open source, text recognition
Stages: text-recognition
Architecture: lstm

CLSTM is a small C++ implementation of Long Short-Term Memory (LSTM) networks
specifically designed for Optical Character Recognition (OCR) tasks. The library
leverages the Eigen library for numerical computations and is intended for users
who need a lightweight solution with minimal dependencies. Initially developed
when LSTM implementations were scarce, CLSTM remains a viable choice for text
line recognition despite the availability of more comprehensive libraries. It
supports installation via Docker for ease of use on different operating systems,
particularly Windows, and can also be built from source with some prerequisites
like scons, swig, and Eigen. CLSTM includes Python bindings, though they are
noted to be currently broken, and offers a variety of command-line drivers for
training and applying models for OCR and text transformations. The project is
maintained in a state where it primarily serves users who need a minimalistic
solution, with its code base being open source under the Apache-2.0 license.

---



Book Content Segmentation and Dewarping

https://github.com/RaymondMcGuire/BOOK-CONTENT-SEGMENTATION-AND-DEWARPING (Relevance: 0.8)

Keywords: book segmentation, dewarping, FCN, image processing, Python, machine learning, VGG model, dataset, training
Stages: preprocessing, segmentation, layout-analysis, dataset
Architecture: convnet

The project 'Book Content Segmentation and Dewarping' leverages Fully
Convolutional Networks (FCN) to segment images of book pages into three parts:
left page, right page, and background. This segmentation aids in dewarping the
pages, making them easier to read and process. The dataset, created by Lin
YangBin, consists of 500 images with labeled book pages. The model is trained
using a GTX 1070 8GB GPU for approximately 8 hours, achieving a low loss value
of 0.01. Users are required to download a pre-trained VGG model and prepare the
dataset for training and testing. The project is open for further improvements,
including data augmentation and refined dewarping algorithms. The repository
includes Python scripts for data processing and model training, as well as
visual results of the loss over training iterations.

---



Page Dewarping with Cubic Sheet Model

https://github.com/mzucker/page_dewarp (Relevance: 0.8)

Keywords: page dewarping, cubic sheet model, image processing, Python, OpenCV, scipy, PIL, Pillow
Stages: preprocessing
Architecture: 

The 'page_dewarp' project, developed by mzucker, offers a method for dewarping
scanned text pages using a 'cubic sheet' model. This process involves correcting
distortions in images of pages, making them easier to read and process. The
project is implemented in Python and requires libraries such as scipy, OpenCV
3.0 or greater, and the Image module from PIL or Pillow. The tool can be used by
running 'page_dewarp.py' with one or more image files as arguments. The project
is open source and distributed under the MIT license. More detailed information
and a full write-up on the methodology can be found on the developer's website.

---



ExtractTable: Python Library for Table Extraction

https://github.com/ExtractTable/ExtractTable-py (Relevance: 0.8)

Keywords: Python, table extraction, OCR, image processing, PDF, data extraction, API, ExtractTable
Stages: segmentation, layout-analysis, text-recognition
Architecture: 

ExtractTable-py is a Python library designed to extract tabular data from images
and scanned PDFs. It provides an easy-to-use API for developers to process files
without focusing on intricate details like table areas or column coordinates.
The library requires an API Key for authorization and offers free trial credits.
Installation is simple using pip, and the library supports various output
formats such as data frames. ExtractTable is useful for extracting data from
documents like bank statements, medical records, invoices, and tax forms. The
library allows users to handle both image and PDF inputs, offering high accuracy
in character recognition and table layout detection. Users can also make
corrections, such as splitting merged rows or fixing date formats.
ExtractTable's comprehensive API provides detailed output structures, including
table coordinates and character confidence levels. The library is licensed under
Apache License 2.0.

---



Document Image Dewarping with Line Segments

https://github.com/xellows1305/Document-Image-Dewarping (Relevance: 0.8)

Keywords: document image dewarping, text-line, line segments, rectification, image processing, outlier removal, camera pose, page curve
Stages: preprocessing, evaluation
Architecture: 

The 'Document Image Dewarping' project addresses the challenges of dewarping
document images with complex layouts or limited text lines. Traditional methods
often struggle with images dominated by non-textual elements like photos or
tables. To overcome this, the proposed method incorporates line segments along
with text lines, leveraging the characteristics of line segments that remain
straight under transformation. This approach formulates a cost function
combining text line and line segment properties, which is minimized to determine
transformation parameters for accurate document rectification. The algorithm
iteratively refines the rectification by removing outliers—misaligned text lines
and line segments. Evaluations on datasets like CBDAR 2007 and custom datasets
with varied layouts demonstrate the method's robustness and versatility in
handling different document and curved surfaces.

---



COCO-Text API Overview

https://github.com/andreasveit/coco-text (Relevance: 0.8)

Keywords: COCO-Text, API, text detection, OCR, dataset, annotations, Python
Stages: segmentation, text-recognition, dataset, evaluation
Architecture: 

The COCO-Text API is a Python library designed to facilitate the handling,
parsing, and visualization of text annotations within the COCO-Text dataset.
This dataset is part of the broader Microsoft COCO project, which is widely
recognized for its comprehensive collection of images with object and caption
annotations. COCO-Text specifically focuses on text detection and recognition
tasks and is a pivotal resource for research in optical character recognition
(OCR) and related fields. The API allows users to load and manipulate the
dataset's annotations, offering tools to visualize them effectively. To utilize
this API, users need to download the associated MSCOCO images and the text
annotations from the COCO-Text website. The API's repository contains a variety
of scripts and a demo notebook to demonstrate its capabilities. It is a valuable
tool for researchers and developers working on text detection and recognition
challenges.

---



Image Deskewing with MLPs and LSTMs

https://github.com/sauravbiswasiupr/deskewing (Relevance: 0.8)

Keywords: deskewing, MLP, LSTM, image processing, machine learning, Python, Linear Least Squares, open source
Stages: preprocessing
Architecture: mlp, lstm

The 'deskewing' repository provides a solution for correcting skewed images
using machine learning techniques. It utilizes Multi-Layer Perceptrons (MLPs),
Long Short-Term Memory (LSTM) networks, and Linear Least Squares (LLS)
transformations to effectively deskew images. The repository includes scripts
for running a series of experiments to test the deskewing capabilities of these
algorithms. The LSTM approach is noted for its effectiveness in handling images
with varying lengths of skew. The code is primarily written in Python, with some
shell scripts for batch processing. This project is an open-source effort and
has received a moderate level of community interest, as indicated by its stars
and forks on GitHub.

---



Keras-CTPN: Text Detection in Images

https://github.com/yizt/keras-ctpn (Relevance: 0.8)

Keywords: CTPN, Keras, Text detection, OCR, Deep learning, Image processing, ResNet50, ICDAR, Machine learning
Stages: segmentation, text-recognition, evaluation
Architecture: convnet, rnn

The Keras-CTPN project is an implementation of the Connectionist Text Proposal
Network (CTPN) using Keras. It is designed to detect text in natural images,
supporting tasks such as optical character recognition (OCR). The project is
built upon the architecture of ResNet50 and employs a method for refining side
edges to enhance text detection accuracy. It has been trained and evaluated on
datasets like ICDAR2015 and ICDAR2017. The model's performance includes a recall
rate of 37.07%, a precision of 42.94%, and an Hmean of 39.79% on a subset of the
ICDAR2015 dataset. The project provides scripts for text detection and
evaluation, and instructions for training with pre-trained models. Key features
include the use of a bi-directional GRU and data augmentation techniques like
horizontal flipping and random cropping. The repository includes code for
training, prediction, and evaluation, making it a comprehensive tool for text
detection in images.

---



Geometric Augmentation for Text Images

https://github.com/Canjie-Luo/Text-Image-Augmentation (Relevance: 0.8)

Keywords: geometric augmentation, text recognition, image transformation, CVPR 2020, OpenCV, Python, C++, data augmentation, image processing
Stages: preprocessing, text-recognition, training-data-generation
Architecture: 

The 'Text Image Augmentation' project provides a general geometric augmentation
tool designed to enhance text images, primarily focusing on improving text
recognition models. This tool is based on the methodology described in the CVPR
2020 paper 'Learn to Augment: Joint Data Augmentation and Network Optimization
for Text Recognition.' The augmentation techniques are intended to prevent
overfitting and enhance the robustness of text recognizers by applying geometric
transformations such as distortion, stretch, and perspective adjustments. These
transformations can significantly improve the accuracy of text recognition
models, as demonstrated with datasets like IIIT5K, IC13, and IC15. The tool is
implemented in C++ and Python, utilizing libraries such as OpenCV and Boost, and
is designed to be efficient, taking less than 3 milliseconds to transform a
single image using a 2.0GHz CPU. The project is open-source and licensed under
the MIT License, making it accessible for academic research and application
customization. It also includes a demo and installation guide to help users
implement the tool in their projects.

---



DIVA Layout Analysis Evaluator

https://github.com/DIVA-DIA/DIVA_Layout_Analysis_Evaluator (Relevance: 0.8)

Keywords: layout analysis, ICDAR, document images, evaluation tool, medieval manuscripts, ground truth, visualization, Java, DIVA
Stages: layout-analysis, evaluation, evaluation-results
Architecture: 

The DIVA Layout Analysis Evaluator is a tool designed for evaluating layout
analysis in document images, specifically for competitions such as the ICDAR
2017 competition on challenging medieval manuscripts. It is a Java-based
application that helps in assessing the accuracy of layout analysis algorithms
by comparing predicted layouts with ground truth images. The tool provides both
numerical results (such as Intersection over Union, precision, recall, F1 score)
and visual feedback, which includes a human-friendly visualization of the
results. The evaluator supports using original images to help in understanding
errors in prediction by overlaying results on the original document image. It
provides detailed color-coded feedback to highlight correctly and incorrectly
predicted regions. The tool is also used in the ICDAR 2019 Historical Document
Reading Challenge on Chinese family records. The ground truth format involves a
pixel-label image with class information encoded in the blue channel. This
evaluator serves as a crucial component for researchers and developers working
on document layout analysis, allowing them to fine-tune and validate their
algorithms.

---



SAR: Baseline for Text Recognition

https://github.com/wangpengnorman/SAR-Strong-Baseline-for-Text-Recognition (Relevance: 0.8)

Keywords: text recognition, irregular text, SAR model, Torch, synthetic data, real datasets, CUDA
Stages: text-recognition, training-data-generation
Architecture: transformer, attentional

The project 'SAR: A Simple and Strong Baseline for Irregular Text Recognition'
provides a robust framework for recognizing irregular text in images, developed
by Hui Li, Peng Wang, Chunhua Shen, and Guyu Zhang. The implementation is based
on the 'Show, Attend and Read' model and is coded using Torch, with
compatibility for CUDA-enabled GPUs. This model is particularly designed to
handle the challenges posed by irregular text recognition in various real-world
scenarios. The repository includes scripts for running the model on new images
or directories, as well as instructions for training the model using both
synthetic and real datasets. The project highlights the use of datasets like
Syn90k, SynthText, and others for synthetic data, and IIIT5K, SVT, ICDAR, among
others, for real data. The pretrained model can be downloaded from an external
link and used directly for text recognition tasks. The code is intended for
academic use, with provisions for commercial licensing through contact with the
authors. The project has gained attention with 173 stars and 40 forks on GitHub.

---



Rethinking Text Segmentation

https://github.com/SHI-Labs/Rethinking-TextSegmentation (Relevance: 0.7)

Keywords: text segmentation, natural language processing, computer vision, GitHub, collaboration, innovation, methodology, accuracy, efficiency, open source
Stages: segmentation
Architecture: 

The project focuses on innovative approaches to text segmentation, a critical
task in natural language processing and computer vision. It emphasizes re-
evaluating traditional methods and exploring new techniques to improve
segmentation accuracy and efficiency. The project is hosted on GitHub,
suggesting it may be open for collaboration and further development by the
community.

---



LAREX: Layout Analysis Tool

https://github.com/chreul/LAREX (Relevance: 0.7)

Keywords: LAREX, layout analysis, OCR, open-source, early printed books, region extraction, PAGE XML, manual correction, Docker
Stages: layout-analysis, segmentation, output-representation
Architecture: 

LAREX is a semi-automatic open-source tool designed for layout analysis and
region extraction on early printed books. It employs a rule-based approach for
connected components, offering fast and comprehensible results with options for
manual corrections. LAREX integrates into existing OCR workflows using the PAGE
XML format, making it a flexible tool for segmenting pages of historical texts.
The tool is available for various platforms, including Docker, Linux, Windows,
and macOS, and supports customization through configuration settings like book
paths, save modes, and UI elements. Users can add their own books for processing
and access detailed usage guides for efficient operation.

---



Ochre: OCR Post-Correction Toolbox

https://github.com/KBNLresearch/ochre (Relevance: 0.7)

Keywords: OCR post-correction, toolbox, CWL workflows, character-based models, error analysis, Jupyter Notebooks, data preprocessing, performance assessment, open-source
Stages: preprocessing, language-modeling, evaluation
Architecture: 

Ochre is an experimental toolbox designed for the post-correction of Optical
Character Recognition (OCR) text. It provides workflows for preprocessing OCR
datasets, training character-based models, executing post-correction, and
assessing the performance of these corrections. The toolbox includes predefined
workflows using Common Workflow Language (CWL), but also allows users to create
custom workflows through Jupyter Notebooks. Ochre supports various datasets,
including the VU DNC corpus, ICDAR 2017 shared task data, and others. It
emphasizes aligning OCR text with gold standard texts to facilitate effective
training and correction. Users can perform OCR error analysis by mapping words
between OCR and gold standard texts to understand error types. The toolbox
employs external tools like ocrevalUAtion for evaluating OCR performance. Ochre
is open-source and licensed under the Apache-2.0 license, making it accessible
for further development and integration into different OCR workflows.

---



Document Image Binarization with FCA Tools

https://github.com/zp-j/binarizewolfjolion (Relevance: 0.7)

Keywords: Document Binarization, Formal Concept Analysis, Niblack, Sauvola, Wolf, Image Processing, Python, Linux, Image Analysis
Stages: preprocessing
Architecture: 

This project focuses on document image binarization using various algorithms
within the field of document image analysis. The project utilizes Formal Concept
Analysis (FCA) to measure and analyze the agreement and disagreement between
different automated interpretation algorithms. It specifically employs four
binarization algorithms: a simple binarization method, Niblack, Sauvola et al.,
and Wolf et al. The repository contains both a Python implementation for a
simple binarization algorithm and an enhanced contrast maximization version for
more complex methods, including Niblack and Sauvola. The project aims to
categorize semantic overlaps and differences in algorithmic interpretations
using FCA. The project is particularly intended for execution on Linux systems,
and it uses Java for executing Galicia and Lattice Miner, Python for scripting,
and gcc/g++ for compiling the algorithms. The repository provides a structured
approach to executing the binarization process step-by-step and includes
resources for further exploration of FCA concepts.

---



CRNN TensorFlow Implementation

https://github.com/wcy940418/CRNN-end-to-end (Relevance: 0.7)

Keywords: CRNN, TensorFlow, Python, deep learning, sequence prediction, CNN, RNN, machine learning, open source
Stages: text-recognition
Architecture: crnn

The CRNN-end-to-end project is a Python implementation of a Convolutional
Recurrent Neural Network (CRNN) using TensorFlow. It aims to provide an end-to-
end solution for sequence prediction problems by combining Convolutional Neural
Networks (CNNs) with Recurrent Neural Networks (RNNs). The implementation relies
on several dependencies, including TensorFlow r1.0, lmdb library, OpenCV2, and
Baidu's WarpCTC. Despite its development, the project is currently non-
functional, and reimplementation has been terminated. The repository includes
various updates and files such as utility for handling checkpoints, string
conversion, model testing scripts, dataset handlers, and training scripts. It is
licensed under the MIT license and has received a few stars and forks from the
GitHub community.

---



Typeface Corpus for OCR Training

https://github.com/jbest/typeface-corpus (Relevance: 0.7)

Keywords: OCR, typeface, Tesseract, OCRopus, digital humanities, natural history, text extraction
Stages: training-data-generation
Architecture: 

The 'typeface-corpus' repository is a collection of typeface samples aimed at
enhancing OCR (Optical Character Recognition) activities in the natural history
collections and digital humanities communities. These communities require
efficient text extraction from documents and images, which often feature diverse
typefaces. By standardizing a corpus of typeface samples, this repository
supports the improvement of text output quality from OCR engines like Tesseract
and OCRopus. The repository is publicly accessible on GitHub, offering resources
such as submission procedures for contributing typefaces, and is maintained to
ensure that high-quality text extraction can be achieved through improved
recognition of various typefaces.

---



Archiscribe Corpus: 19th Century Fraktur OCR

https://github.com/jbaiter/archiscribe-corpus (Relevance: 0.7)

Keywords: OCR, fraktur, 19th-century, German prints, historical data, dataset, training data, archiscribe, transcription
Stages: dataset, evaluation
Architecture: 

The Archiscribe Corpus is a repository designed to provide a comprehensive
optical character recognition (OCR) ground truth dataset specifically for 19th-
century German prints. The corpus includes 4,255 transcribed lines from 112
works published over a span of 73 years. The primary focus of this project is to
offer diverse and accurate training data for OCR systems that deal with
historical fraktur script, which was commonly used in German texts of that era.
The repository is managed under a CC-BY-4.0 license, allowing for collaborative
and open-source development. The corpus is hosted on the archiscribe.jbaiter.de
platform and aims to be a valuable resource for evaluating and improving OCR
technologies in the context of historical document digitization. The dataset
covers a wide range of subjects and publication years, providing a broad
spectrum of historical data for research and development in OCR and related
fields.

---



PRImA Page Viewer

https://github.com/PRImA-Research-Lab/prima-page-viewer (Relevance: 0.7)

Keywords: PRImA, Page Viewer, Java, PAGE XML, ALTO XML, FineReader XML, HOCR, layout analysis, OCR
Stages: output-representation
Architecture: 

The PRImA Page Viewer is a Java-based software tool designed to view PAGE XML
files, which include both layout and text content data. It also provides support
for additional XML formats such as ALTO XML, FineReader XML, and HOCR. This
viewer is intended to facilitate the analysis and manipulation of document
layout data, making it a valuable resource for researchers and developers
working with document digitization and OCR (Optical Character Recognition)
technologies. The project is open-source and licensed under the Apache-2.0
license, allowing for broad use and adaptation. With a small but dedicated user
base indicated by its GitHub stars and forks, the PRImA Page Viewer represents a
specialized tool for managing complex document formats.

---



Docker-Ocropy: OCR in Docker

https://github.com/kba/docker-ocropy (Relevance: 0.7)

Keywords: OCR, Ocropus, Docker, Text Recognition, Image Processing, Containerization, Shell Script, Model Training, Pre-trained Models
Stages: preprocessing, segmentation, text-recognition
Architecture: 

Docker-Ocropy is a project that encapsulates the Ocropy optical character
recognition (OCR) system within a Docker container. This allows users to easily
install and run Ocropy without dealing with complex dependencies and setup
procedures. The repository provides a convenient shell script, `run-ocropy`, to
execute various OCR commands such as dewarping, segmenting, and predicting text
in images. Users can also run Ocropy directly using Docker commands. The project
includes pre-trained models, 'en-default' and 'fraktur', for immediate use, and
provides instructions for training new models using a script `ocrotrain.sh`.
This setup is beneficial for those looking to leverage Ocropy's capabilities for
text recognition tasks while ensuring a consistent and isolated execution
environment.

---



Unpaper: Scanned Paper Post-Processing Tool

https://github.com/Flameeyes/unpaper (Relevance: 0.7)

Keywords: Unpaper, scanned paper, post-processing, deskewing, image cleaning, PDF enhancement, ffmpeg, Meson build
Stages: preprocessing
Architecture: 

Unpaper is a software tool designed to enhance scanned sheets of paper,
particularly for book pages that have been digitized from photocopies. Its
primary function is to improve the readability of scanned pages, especially when
converting them to PDF format. Unpaper works by cleaning scanned images,
removing unwanted dark edges, and correcting misalignments by automatically
straightening pages through a process known as deskewing. However, it is
recommended to manually review the results and adjust parameters as needed,
since automatic processing might not always be perfect. Unpaper requires ffmpeg
for file input and output and uses the Meson build system for compilation, with
additional dependencies on pytest and pillow for testing.

---



OCR-IDCard GitHub Repository

https://github.com/littleredhat1997/OCR-IDCard (Relevance: 0.7)

Keywords: OCR, ID Card, GitHub, Image Processing, Machine Learning, Text Extraction, Automation, GitHub Repository, Data Processing
Stages: text-recognition, segmentation
Architecture: 

The OCR-IDCard project on GitHub is focused on developing an Optical Character
Recognition (OCR) system specifically for ID cards. The repository likely
contains code and resources for extracting and processing text from scanned
images of identification cards. The project aims to facilitate the automatic
reading of important information such as names, dates, and identification
numbers from ID cards, using machine learning and image processing techniques.
This project is valuable for applications in sectors such as security, finance,
and administration where quick and accurate data extraction from IDs is crucial.

---



hOCR Format Specification

https://github.com/kba/hocr-spec (Relevance: 0.7)

Keywords: hOCR, OCR, specification, workflow, format, Thomas Breuel, GitHub, open source, documentation
Stages: output-representation
Architecture: 

The hOCR project outlines the specifications for the hOCR format, an embedded
OCR workflow and output format initially developed by Thomas Breuel. The
repository provides detailed documentation for different versions of the hOCR
specification, including versions 1.0, 1.1, and 1.2. The project aims to make
these specifications more accessible and easier to maintain. It encourages
participation through GitHub issues and pull requests, and provides resources
for building the specification using tools like GNU make, bikeshed, and Docker.
The hOCR format is designed to be semantically backwards-compatible across its
versions, ensuring continuity and ease of use for developers and contributors.
The project also includes Chinese translations and aims to harmonize styles and
add cross-references to other specifications, making it a comprehensive resource
for developers working with OCR technologies.

---



CTPN: Text Detection in Images

https://github.com/qingswu/CTPN (Relevance: 0.7)

Keywords: CTPN, text detection, CUDA 8.0, Caffe, GPU, CUDNN, Python, open-source, image processing
Stages: segmentation
Architecture: convnet

The CTPN project provides a CUDA 8.0 compatible implementation of the
Connectionist Text Proposal Network for detecting text in natural images. This
project is an adaptation of the original CTPN by Tianzhi et al., designed to
work with the Caffe deep learning framework. It includes code adjustments to
accommodate the latest version of Caffe and provides a trained model for text-
line detection. Utilizing a GPU is recommended for optimal performance, with
CUDNN support further enhancing efficiency. The software dependencies include
Python 2.7, Cython, and other libraries required by Caffe. Users can clone the
repository, compile the necessary files, and run the demo script to perform text
detection. The project is released under the MIT License, allowing for open-
source use and modification.

---



Ground Truth Files from Spanish Libraries

https://github.com/impactcentre/groundtruth-spa (Relevance: 0.7)

Keywords: ground truth, Spanish libraries, BNE, BVC, digital libraries, IMPACT initiative, historical texts, CC-BY-NC-SA, digitization
Stages: dataset
Architecture: 

The repository 'groundtruth-spa' offers ground truth files derived from Spanish
Digital Libraries, specifically BNE (Biblioteca Nacional de España) and BVC
(Biblioteca Virtual Cervantes). These files are intended to provide accurate
digital representations of various Spanish literary works and are released under
a Creative Commons license (CC-BY-NC-SA). The repository includes a variety of
historically significant texts such as 'AjedrezRuyLopez', 'CartaSorJuana',
'ComentariosIncaGarcilaso', 'DiccionarioAutoridades', and several others. This
project is part of the IMPACT initiative, which focuses on improving the
accessibility and usability of historical texts through digital means. The
repository is public and open for contributions, although it currently has no
stars and only a few forks. This collection of ground truth files can be
particularly valuable for researchers and developers working on projects related
to digitization, text recognition, and cultural heritage preservation.

---



ALTO XML Python Tools

https://github.com/cneud/alto-tools (Relevance: 0.7)

Keywords: ALTO XML, Python tools, OCR, text extraction, digital library
Stages: layout-analysis, output-representation
Architecture: 

The ALTO Tools project provides a set of Python tools designed to perform
various operations on ALTO XML files. ALTO (Analyzed Layout and Text Object) XML
is a standard for representing the layout and content of digitized text
documents, commonly used in digital libraries and optical character recognition
(OCR) processes. The tools allow users to extract text content, mean OCR word
confidence scores, bounding box coordinates of illustrations and graphical
elements, and various statistical information such as the number of text lines,
words, and glyphs. The project is licensed under the Apache-2.0 license and is
available for installation via PyPI or by cloning the repository. The tools are
executed through a command-line interface, where users provide the path to an
ALTO XML file or directory and specify options for the desired operations. The
output is directed to the standard output, enabling easy integration into larger
workflows.

---



CRNN with Chinese Text Recognition

https://github.com/wulivicte/crnn (Relevance: 0.7)

Keywords: CRNN, Chinese recognition, OCR, PyTorch, neural networks, image processing, text recognition, machine learning, deep learning
Stages: text-recognition
Architecture: crnn

This project is an implementation of a Convolutional Recurrent Neural Network
(CRNN) with added functionality for recognizing Chinese characters. It is a fork
from the original CRNN implementation on GitHub (meijieru/crnn.pytorch). The
project is built using Python and PyTorch and provides detailed instructions on
setting up the environment, including dependencies like CUDA, OpenCV, and
wrap_ctc for handling connectionist temporal classification. The repository
includes scripts for training the CRNN model with a dataset of 21 classes of
Chinese and English text, as well as for predicting text from images. Users can
customize the model by adjusting parameters such as image dimensions, LSTM
hidden layers, and the alphabet of recognized characters. The project is open-
source under the MIT license and is designed to facilitate the recognition of
both English and Chinese text in images, making it suitable for multilingual
optical character recognition applications.

---



TreeStructure Table Extraction Tool

https://github.com/HazyResearch/TreeStructure (Relevance: 0.7)

Keywords: Table Extraction, Fonduer, PDF, Hierarchical Structure, Machine Learning, Open Source, Python, Data Mining, Document Processing
Stages: segmentation, layout-analysis, evaluation
Architecture: 

The TreeStructure project, developed by HazyResearch, is an extension of Fonduer
designed for information extraction from richly formatted data, particularly
tables in PDF documents. A significant challenge addressed by this project is
the maintenance of the hierarchical structure of document content during the PDF
to HTML conversion process, which existing tools often fail at, especially in
preserving table cell structures. The project involves building a custom module
to replace non-open-source solutions like Adobe Acrobat. The tool leverages a
machine learning approach to extract tables from PDF files, maintaining the
document's structural integrity using a tree data structure. Users can set up
necessary environment variables, utilize a command-line interface for batch
processing, and evaluate extraction accuracy using precision, recall, and F1
score metrics at the character level. The tool is built using Python and Jupyter
Notebook, and it incorporates open-source resources like the table-extraction
tool from GitHub.

---



OCR Table Extraction Project

https://github.com/cseas/ocr-table (Relevance: 0.7)

Keywords: OCR, PDF, table extraction, Python, Tesseract, ImageMagick, Poppler-utils, open-source, MIT license
Stages: text-recognition, table-recognition
Architecture: 

The 'ocr-table' project is designed to extract tables from scanned image PDFs
using Optical Character Recognition (OCR) technology. It utilizes Python and
Shell scripts to automate the extraction process, leveraging Tesseract OCR and
additional tools like ImageMagick and Poppler-utils for handling PDF files. The
workflow involves placing PDF files in a designated folder, running a script to
perform OCR, and retrieving the resulting text files with extracted table data.
An alternate method is provided for cases where the primary approach is
ineffective. The project is open-source and licensed under the MIT license,
encouraging collaboration and further development.

---



Alyn: Image Skew Detection and Correction

https://github.com/kakul/Alyn (Relevance: 0.7)

Keywords: image processing, skew detection, deskew, text, Canny Edge Detection, Hough Transform, Python, automation, image correction
Stages: preprocessing
Architecture: 

Alyn is a Python-based tool designed to detect and correct skew in images
containing text. This project utilizes techniques such as Canny Edge Detection
and Hough Transform to analyze images, determine skew angles, and adjust the
orientation to rectify any skew. The tool offers both command-line scripts and a
Python API for skew detection and image deskewing. Users can batch process
multiple images, display processed results, and save outputs to files. Alyn
requires several dependencies, including numpy, matplotlib, scipy, and scikit-
image, and is easily installed via pip. The application is suitable for
automating the correction of skewed text images, which is particularly useful in
document processing and digitization tasks.

---



Unproject Text Using Ellipses

https://github.com/mzucker/unproject_text (Relevance: 0.7)

Keywords: perspective recovery, text correction, ellipses, Python, OpenCV, image processing, distorted text, skewed text, mzucker
Stages: preprocessing
Architecture: 

The 'unproject_text' project by mzucker is designed to recover the perspective
of skewed text in images using transformed ellipses. This approach is
particularly useful in situations where text within an image is distorted due to
an angle or curvature, often encountered in scanned documents or photos taken at
angles. The technique leverages Python with libraries such as NumPy, SciPy,
OpenCV, and matplotlib to perform the perspective correction. The core script,
'unproject_text.py', processes an image and adjusts the text perspective by
identifying and transforming ellipses in the image, which helps in rectifying
the skewed text. The project includes several example images and scripts to
demonstrate the functionality. Interested users can run the provided Python
scripts on sample images to see the perspective recovery in action. The project
is open-source and available on GitHub, encouraging contributions and forking by
interested developers.

---



SegLink Text Detection Algorithm

https://github.com/dengdan/seglink (Relevance: 0.7)

Keywords: SegLink, text detection, oriented text, image processing, TensorFlow, OpenCV, natural images, machine learning, deep learning
Stages: segmentation
Architecture: convnet

This project is a re-implementation of the SegLink algorithm, which is designed
to detect oriented text in natural images by linking segments. The original
algorithm was detailed in the paper 'Detecting Oriented Text in Natural Images
by Linking Segments' by Baoguang Shi, Xiang Bai, and Serge Belongie. The
implementation requires TensorFlow and OpenCV, and it utilizes datasets such as
SynthText and ICDAR2015. The project addresses challenges like slow convergence
compared to the original implementation, attributed to factors such as batch
size limitations and learning rate differences. Two models, SegLink-384 and
SegLink-512, have been trained with specific configurations and are available
for download. The repository provides scripts for testing custom images and
visualizing detection results. The project is open-source and licensed under
GPL-3.0.

---



OpenCV Document Layout Analysis

https://github.com/Pay20Y/Layout_Analysis (Relevance: 0.7)

Keywords: RLSA, X-Y Cut, OpenCV, C++, Document Analysis, Layout Analysis, GitHub, Segmentation
Stages: segmentation, layout-analysis
Architecture: 

The Layout_Analysis project on GitHub, developed by user Pay20Y, is an
implementation of two document layout analysis techniques: RLSA (Run-Length
Smoothing Algorithm) and X-Y Cut. These techniques are coded in C++ and utilize
the OpenCV library, requiring version 3.0 or higher. The project aims to
facilitate document layout analysis by providing tools to segment and analyze
document structures. Users can compile the provided C++ files using the g++
compiler, linking necessary OpenCV libraries. The project references significant
works in the domain, including Wong et al.'s document analysis system and Ha et
al.'s recursive X-Y cut method. While the repository has limited community
activity, it provides a foundational implementation for users interested in
document analysis using OpenCV.

---



OpenArabic OCR Gold Standard Data

https://github.com/OpenArabic/OCR_GS_Data (Relevance: 0.7)

Keywords: OCR, Arabic, Data, Training, Testing, Gold Standard, Optical Character Recognition, Open Source, GitHub
Stages: dataset, evaluation
Architecture: 

The OpenArabic/OCR_GS_Data repository provides a comprehensive dataset
specifically curated for training and testing Optical Character Recognition
(OCR) engines. It features double-checked gold standard data aimed at enhancing
the accuracy and reliability of OCR systems, particularly in processing Arabic
scripts. This repository serves as a valuable resource for developers and
researchers working on OCR technologies, offering a robust foundation for
developing algorithms that require precise text recognition capabilities in
Arabic. The repository includes essential scripts for evaluation, extraction,
and testing, facilitating the seamless integration and assessment of OCR
engines. With its focus on high-quality data verification, it ensures that OCR
solutions are built on a solid and reliable dataset, potentially improving the
performance of OCR systems across various applications.

---



OCR File Format Converter

https://github.com/UB-Mannheim/ocr-fileformat (Relevance: 0.7)

Keywords: OCR, file conversion, validation, hOCR, ALTO, PAGE, FineReader, Docker, CLI, API
Stages: output-representation
Architecture: 

The OCR File Format Converter project by UB-Mannheim is designed to validate and
transform various OCR file formats, including hOCR, ALTO, PAGE, and FineReader.
The project provides tools for converting and validating OCR outputs, which can
be accessed via command line scripts, a web interface, or integrated into other
applications through an API. Users can leverage Docker for easy deployment of
the command line scripts and web interface, or install the system-wide app. The
project supports various transformations and validations, offering a wide range
of formats for input and output, and is freely available under the MIT license.
The project is actively maintained with contributions from the community, and it
aims to facilitate seamless transitions between different OCR formats for both
individual and enterprise-level applications.

---



ABBYY FineReader OCR for Senate Reports

https://github.com/dannguyen/abbyy-finereader-ocr-senate (Relevance: 0.7)

Keywords: ABBYY FineReader, OCR, Senate financial disclosures, data extraction, tabular data, PDF, scanned forms, Tesseract, data processing
Stages: text-recognition, table-recognition, evaluation-results
Architecture: 

This project evaluates the performance and accuracy of using ABBYY FineReader's
OCR technology to extract tabular data from U.S. Senators' personal financial
disclosure forms, which are often submitted as scanned images or PDFs. Despite
the Senate's electronic filing system, many reports are still submitted on
paper, necessitating the use of OCR to digitize the data. ABBYY FineReader is
recognized for its capability to convert scanned forms into usable digital
formats, including Excel spreadsheets, although challenges remain in achieving
perfect OCR accuracy and preserving complex tabular structures. The project
compares ABBYY FineReader with open-source OCR tools like Tesseract,
highlighting FineReader's superior ability in recognizing tabular data. The
writeup provides insights into how to apply FineReader in a semi-automated
fashion for batch processing, emphasizing that while OCR technology can
facilitate data extraction, additional systems are required for comprehensive
data processing and integration.

---



HOCR Specification Python Parser

https://github.com/athento/hocr-parser (Relevance: 0.7)

Keywords: HOCR, Python, Parser, OCR, Open-source, Apache-2.0, Text extraction, Data processing
Stages: output-representation, text-recognition
Architecture: 

The 'hocr-parser' is a Python library designed to parse and interpret HOCR (HTML
for OCR) specifications. HOCR is a file format commonly used to represent the
output of Optical Character Recognition (OCR) systems. This parser enables
developers to process and extract structured data from HOCR files, facilitating
the integration of OCR outputs into various applications. The library is open-
source, licensed under the Apache-2.0 license, and available for community
contributions and use. It provides a foundational toolset for developers working
with OCR data, allowing for efficient parsing and manipulation of text
recognized from scanned images or documents.

---



OCRmyPDF: OCR for Scanned PDFs

https://github.com/jbarlow83/OCRmyPDF (Relevance: 0.7)

Keywords: OCR, PDF, Tesseract, command-line, searchable PDFs, PDF/A, image processing, open source, automation
Stages: preprocessing, text-recognition
Architecture: 

OCRmyPDF is a command-line tool designed to add an OCR (Optical Character
Recognition) text layer to scanned PDF files, making them searchable and
allowing text to be copied. It utilizes the Tesseract OCR engine, supporting
over 100 languages, and can perform tasks like deskewing, optimizing image
quality, and generating PDF/A files. The tool is capable of processing large
files efficiently by distributing work across multiple CPU cores. OCRmyPDF
supports various operating systems, including Linux, Windows, macOS, and
FreeBSD, and offers installation through package managers like apt, dnf, and
brew. The software is licensed under MPL-2.0, enabling integration with other
projects. It has been tested on millions of PDFs and is widely used in various
industries for automating OCR processes.

---



Chinese OCR with TensorFlow and PyTorch

https://github.com/xiaofengShi/CHINESE-OCR (Relevance: 0.7)

Keywords: Chinese OCR, TensorFlow, PyTorch, CTPN, CRNN, Text Detection, OCR, Keras, CTC
Stages: segmentation, text-recognition
Architecture: crnn, lstm

The CHINESE-OCR project is designed to detect and recognize text from natural
scenes, particularly focusing on Chinese characters. It uses TensorFlow for text
detection and both Keras and PyTorch for implementing the CTPN (Connectionist
Text Proposal Network) and CRNN (Convolutional Recurrent Neural Network) models
for OCR. The project features a comprehensive approach to text detection,
including image classification for text orientation and region detection using
CTPN, followed by end-to-end OCR with CRNN. It implements GRU/LSTM networks
combined with CTC (Connectionist Temporal Classification) for recognizing
variable-length text sequences. This open-source project provides scripts for
setting up the environment, training custom models, and utilizing pre-trained
models for text recognition tasks. Users can adapt the models for their own
datasets, with example training scripts included for both CTPN and CRNN. The
project's codebase supports both GPU and CPU environments. The repository also
offers links to various datasets and pre-trained models, along with detailed
descriptions of the algorithms and training procedures involved.

---



Form Segmentation Project Overview

https://github.com/doxakis/form-segmentation (Relevance: 0.7)

Keywords: text extraction, forms, OCR, handwriting recognition, ICR, noise removal, shape detection, open-source, document processing
Stages: preprocessing, segmentation, text-recognition
Architecture: 

The Form Segmentation project aims to develop an algorithm capable of extracting
text data from forms and scanned pages, including both printed and handwritten
text, without prior knowledge of the document's layout. This project targets
automatic text extraction to facilitate further processing by business logic
systems, neural networks, or classifiers. Key challenges include identifying
text-containing regions, removing black borders, recognizing hand-drawn
characters through Intelligent Character Recognition (ICR), and eliminating
noise from scanned documents. The project also focuses on shape detection, edge
detection for perspective correction, and the application of OCR, particularly
with Tesseract for printed text. Performance optimization is a consideration,
with possibilities for scaling through additional computational resources. The
ultimate goal is to efficiently identify document types and involved identities
with minimal errors. The project is open-source under the MIT license and is
implemented primarily in Jupyter Notebook.

---



PyTextractor: Python OCR with Tesseract

https://github.com/danwald/pytextractor (Relevance: 0.7)

Keywords: OCR, Python, Tesseract, OpenCV, EAST detector, text extraction, image processing, PyTextractor, MIT license
Stages: segmentation, text-recognition
Architecture: 

PyTextractor is a Python-based Optical Character Recognition (OCR) tool that
utilizes Tesseract and the EAST text detector from OpenCV to extract text or
numbers from images. It provides a command-line interface, 'text_detector',
which allows users to specify input image paths and customize detection
parameters such as confidence level, image dimensions, and the minimum number of
detected boxes. PyTextractor supports text extraction by leveraging the power of
pytesseract, offering flexibility for detecting numbers only or displaying
bounding boxes. Installation of PyTextractor is straightforward, requiring a
Tesseract installation via Homebrew and package installation via pipx.
Additionally, PyTextractor can be used programmatically as a library, allowing
for more advanced integration into Python applications. The project is open-
source under the MIT license, with its repository hosted on GitHub, where users
and contributors can find the source code, report issues, and contribute to its
development.

---



CUTIE: Convolutional Text Information Extractor

https://github.com/vsymbol/CUTIE (Relevance: 0.7)

Keywords: CUTIE, TensorFlow, text extraction, deep learning, document analysis, OCR, NER, slot filling, information extraction
Stages: text-recognition, layout-analysis
Architecture: convnet

CUTIE, which stands for Convolutional Universal Text Information Extractor, is a
deep learning model implemented in TensorFlow. It is designed to extract key
information from documents by leveraging a 2-dimensional approach to Named
Entity Recognition (NER) and slot filling. Before utilizing CUTIE, it is
necessary to preprocess documents using Optical Character Recognition (OCR) to
detect and extract text. CUTIE's architecture enables it to effectively identify
and categorize essential details from various document types, such as receipts,
by analyzing structured text data. The project demonstrates significant
performance on a dataset comprising 4,484 receipts across different categories,
outperforming other methods like CloudScan and BERT in key information
extraction tasks. Installation involves setting up the environment with
necessary dependencies and preparing data for training and evaluation using
provided scripts. CUTIE is particularly efficient when document rows and columns
are well-structured, and it provides detailed results with a focus on accuracy
in identifying key information classes.

---



TransDETR: End-to-End Video Text Spotting

https://github.com/weijiawu/TransDETR (Relevance: 0.7)

Keywords: TransDETR, Video Text Spotting, Transformer, Text Detection, Text Tracking, Bilingual Dataset, Deep Learning, Computer Vision, ICDAR2015
Stages: text-recognition, dataset
Architecture: transformer

TransDETR is a novel framework for end-to-end video text spotting using
Transformer-based sequence modeling. The framework addresses the task of
detecting, tracking, and recognizing text in video sequences. Unlike traditional
methods relying on Intersection over Union (IoU) or appearance similarity across
frames, TransDETR treats video text spotting as a direct temporal modeling
challenge. The project introduces a benchmark dataset called BOVText, which is
designed to support bilingual text spotting in videos. TransDETR shows
competitive performance on standard datasets like ICDAR2015, achieving high
metrics in MOTA and MOTP for video text tracking and spotting tasks. The
framework supports features such as Non-Maximum Suppression (NMS) and optimized
post-processing techniques. The project is built on existing technologies like
Deformable DETR and MOTR, and it requires specific dependencies such as PyTorch
and CUDA for setup. TransDETR's performance and capabilities make it a valuable
tool for researchers and developers interested in video text recognition.

---



TEI-OCR XML Schema Project

https://github.com/OpenPhilology/tei-ocr (Relevance: 0.7)

Keywords: TEI, OCR, XML schema, metadata, text encoding, grapheme, Unicode, customization
Stages: output-representation
Architecture: 

The TEI-OCR project provides a TEI (Text Encoding Initiative) customization for
handling metadata generated from optical character recognition (OCR) processes.
This project facilitates advanced features such as process identification,
grapheme level encoding, and support for glyphs without Unicode code points. It
also allows for variations on elements like lines, segments, and graphemes. The
project offers a schema and documentation to guide users in leveraging these
capabilities in their OCR workflows. Released under the CC0-1.0 license, TEI-OCR
aims to enhance the processing and encoding of OCR-generated layout and content
information.

---



Python Receipt Parser with OCR

https://github.com/ReceiptManager/receipt-parser (Relevance: 0.7)

Keywords: Python, OCR, receipt parsing, Tesseract, supermarket, Docker, imagemagick, open source
Stages: preprocessing, text-recognition
Architecture: 

The 'receipt-parser-legacy' is a Python-based application designed to parse
supermarket receipts using Optical Character Recognition (OCR) with Tesseract.
By extracting key information such as the shop name, date, and total amount from
scanned receipts, the tool facilitates record-keeping and data management. It
can be used as a standalone script or integrated into iOS and Android
applications. The project requires the 'imagemagick' library for processing
images and includes a Dockerfile to simplify deployment. The project originated
from a hackathon idea and has been featured on the trivago tech blog. It is
shared under the Apache-2.0 license and is actively maintained with
contributions from a diverse group of developers. Users can run the parser on
sample images or their own by using Docker commands, making it versatile and
easy to use for various receipt parsing needs.

---



GROBID: Scholarly Document Information Extraction

https://github.com/kermitt2/grobid (Relevance: 0.7)

Keywords: GROBID, information extraction, machine learning, scholarly documents, PDF parsing, XML/TEI, bibliographic data, deep learning, open source
Stages: layout-analysis, output-representation
Architecture: transformer

GROBID (GeneRation Of BIbliographic Data) is a machine learning software
designed to extract, parse, and restructure information from raw scholarly
documents, especially PDFs, into structured XML/TEI formats. Initially developed
as a hobby in 2008, GROBID has evolved into an open-source tool used by
institutions like ResearchGate and Semantic Scholar. It offers functionalities
such as header extraction, reference parsing, citation context recognition,
full-text structuring, and bibliographical reference consolidation. The software
uses deep learning models, particularly from the DeLFT library, for tasks
ranging from parsing names and affiliations to identifying funders and
copyrights. GROBID supports high scalability and speed, making it suitable for
large-scale processing of scientific literature. It is available with a
comprehensive web service API, Docker images, and clients in Python, Java, and
Node.js for parallel batch processing. GROBID is distributed under the Apache
2.0 license and aims to operate efficiently on commodity hardware with good
parallelization capabilities.

---



cvOCR: OCR System for Resumes

https://github.com/Halfish/cvOCR (Relevance: 0.7)

Keywords: OCR, resume, cv, OpenCV, image preprocessing, text recognition, ipin.com
Stages: preprocessing, text-recognition
Architecture: 

cvOCR is an Optical Character Recognition (OCR) system designed to accurately
recognize and extract text from resumes and CVs. The project emphasizes the
importance of preprocessing steps such as image rotation correction and noise
reduction, which are primarily handled using OpenCV to enhance OCR accuracy.
Developed for ipin.com, cvOCR leverages a combination of C++, Python, HTML, and
Shell scripting to perform its operations. It is hosted on GitHub, where it has
received contributions and engagement from the open-source community.

---



LAREX: Layout Analysis Tool

https://github.com/chreul/larex (Relevance: 0.7)

Keywords: LAREX, OCR, layout analysis, region extraction, open-source, early printed books, PAGE XML, digital humanities
Stages: segmentation, layout-analysis, output-representation
Architecture: 

LAREX is a semi-automatic open-source tool designed for layout analysis and
region extraction on early printed books. It employs a rule-based connected
components approach to efficiently segment pages, allowing users to perform
manual corrections intuitively. LAREX integrates with existing OCR workflows
using the PAGE XML format. The tool is available under the MIT license and is
supported on multiple platforms including Docker, Linux, Windows, and macOS. It
offers features for segmenting, editing, and processing text lines in printed
books, making it a valuable resource for digital humanities and archival
research.

---



Invoice Scanner React Native

https://github.com/burhanuday/invoice-scanner-react-native (Relevance: 0.7)

Keywords: React Native, Invoice Scanner, OCR, ML Kit, OpenCV, Image Processing, AIDL 2020, Flask Server, Android SDK
Stages: preprocessing, text-recognition
Architecture: 

The 'Invoice Scanner React Native' project is a solution developed by team
Codesquad for the AIDL 2020 competition. It utilizes Google ML Kit for Optical
Character Recognition (OCR) and OpenCV for image processing to perform
operations on images of bills or invoices. The primary tasks include edge
detection, cropping, flattening, image enhancement, and compression. The project
is built using React Native and requires specific installations like the React
Native CLI and the latest Android SDK. The solution involves extracting text
from processed images and generating a confidence score for the accuracy of the
text extraction. The project also includes a Flask server for handling specific
backend operations. Modified open-source libraries, such as 'react-native-
document-scanner' and 'react-native-perspective-image-cropper', are used to meet
the project's requirements. The repository provides comprehensive steps to set
up and run the application on both Android and iOS platforms, including
instructions for building the application and generating the APK file.

---



hOCRTools: hOCR Processing Utilities

https://github.com/ONB-RD/hOCRTools (Relevance: 0.7)

Keywords: hOCR, OCR, ALTO, XML, transformation, XSLT, Apache-2.0, open-source, utilities
Stages: output-representation
Architecture: 

hOCRTools is a collection of utilities designed to process and handle hOCR
documents, which are used for representing OCR (Optical Character Recognition)
data. One of the key functionalities provided by hOCRTools is the transformation
of hOCR documents into the ALTO format, a standard for representing OCR results
in XML. This transformation attempts to classify elements such as illustrations
and graphical elements within the document. The tool is designed to be used from
the command line, requiring a properly configured system catalog.xml to avoid
repeated DTD requests from the W3C site. The transformation script, named
hOCR2ALTO, is located in the xsl directory of the repository. The project is
open-source and distributed under the Apache-2.0 license. Despite its niche
application, hOCRTools has garnered a small following, indicated by its six
stars on GitHub.

---



CRNN with Attention for Chinese OCR

https://github.com/wushilian/CRNN_Attention_OCR_Chinese (Relevance: 0.7)

Keywords: OCR, CRNN, Attention, Chinese recognition, TensorFlow, BiLSTM, GRU, Image processing, OpenCV
Stages: text-recognition
Architecture: crnn, attentional

The 'CRNN_Attention_OCR_Chinese' project is a GitHub repository that implements
an OCR system using Convolutional Recurrent Neural Networks (CRNN) enhanced with
attention mechanisms, specifically optimized for recognizing Chinese characters.
The architecture combines a CRNN, which integrates a CNN with a BiLSTM encoder
and a GRU decoder, both using 256 hidden units, to effectively process and
recognize sequences of characters in image data. Users can input images into a
designated directory, with filenames formatted to include labels, and train the
model using the provided scripts. The project requires TensorFlow version 1.2 or
higher and OpenCV. It provides a practical tool for developers and researchers
interested in OCR applications, particularly for Chinese text recognition, and
facilitates further experimentation and customization through its open-source
codebase.

---



OCR++: Scholarly Article Information Extractor

https://github.com/mayank4490/OCR-plus-plus (Relevance: 0.7)

Keywords: OCR, scholarly articles, information extraction, Linux, Django, Python
Stages: 
Architecture: 

OCR++ is a framework designed to extract information from scholarly articles.
The tool is optimized for Linux-based systems and requires specific dependencies
such as NLTK and Django. Users can run OCR++ with or without a server. In server
mode, users can host the tool locally using Python's manage.py to start a server
on their machine. In non-server mode, users rename the target PDF to 'input.pdf'
and place it in a specific directory before running a provided script to execute
the OCR process. The project involves multiple programming languages, including
CSS, HTML, Python, JavaScript, and Shell, and it is structured to allow users to
easily deploy and manage document processing tasks through a web interface. The
installation process involves cloning the repository, setting up directories,
and ensuring all dependencies are installed. Despite being robust and versatile,
the project is in its initial stages with limited community engagement, as
indicated by its stars and forks on GitHub.

---



Tesseract.js: JavaScript OCR Library

https://github.com/naptha/tesseract.js (Relevance: 0.7)

Keywords: JavaScript, OCR, Tesseract, WebAssembly, Image Recognition, Node.js, Open Source, Text Extraction, Multilingual
Stages: text-recognition
Architecture: 

Tesseract.js is a powerful JavaScript library designed for optical character
recognition (OCR) in over 100 languages. Built on a webassembly port of the
Tesseract OCR Engine, it allows users to extract text from images in the browser
or on the server using Node.js. The library is easy to use, with functionalities
accessible via simple JavaScript imports or through a script tag for browser
applications. Tesseract.js supports image recognition and real-time video
recognition, making it versatile for various applications. The library is
continuously updated to improve performance and reduce memory usage, with
significant changes in its recent versions, including smaller file sizes and
compatibility upgrades. It supports multiple installation methods, including CDN
and npm, and is equipped with comprehensive documentation to aid developers in
implementation. Community projects and examples demonstrate its practical uses,
such as document scanning applications and PDF-to-text conversion tools.
Tesseract.js is open-source under the Apache-2.0 license, encouraging
contributions from a diverse group of developers.

---



OCR Ground Truth Tools

https://github.com/UB-Mannheim/ocr-gt-tools (Relevance: 0.6)

Keywords: OCR, ground truth, transcription, hOCR, web interface, Tesseract, Ocropy, Perl, JavaScript
Stages: training-data-generation, output-representation
Architecture: 

The ocr-gt-tools project is a web interface designed for creating ground truth
datasets used in evaluating and training Optical Character Recognition (OCR)
systems. Developed by UB-Mannheim, this tool allows users to edit hOCR files,
which are typically generated by OCR frameworks like Tesseract or Ocropy. The
platform supports ergonomic line-by-line transcription of scanned text, enabling
users to edit transcriptions, comment on lines or pages, and apply standardized
error tags. It includes features such as zooming, filtering visible elements,
and selecting multiple lines for tagging. The interface is written primarily in
HTML and JavaScript for the frontend, with Perl used for server-side scripting.
The program facilitates efficient OCR correction by allowing users to drag and
drop images into the interface, which then processes the images in the
background using a Perl script. The software is distributed under the AGPL-3.0
license, and contributions from the community are encouraged, especially in the
form of bug reports or feature suggestions.

---



Deep Chinese OCR Project

https://github.com/JinpengLI/deep_ocr (Relevance: 0.6)

Keywords: OCR, Chinese character recognition, deep learning, Tesseract alternative, Caffe, Docker, identity card recognition, open-source, JinpengLI
Stages: text-recognition
Architecture: 

The Deep OCR project by JinpengLI is an open-source initiative aimed at
developing a superior Optical Character Recognition (OCR) system for Chinese
characters, outperforming Tesseract. This project leverages deep learning
techniques to enhance the accuracy and efficiency of character recognition. The
repository includes scripts, particularly the reco_chars.py, which utilize
Caffe, a deep learning framework, for improved recognition results. It provides
installation instructions for Ubuntu, including setting up a virtual environment
and using Docker for easy deployment. The project also includes a demo for
identity card recognition, although this feature is still in development and
requires additional semantic models for improved stability. Users can train
their own character recognition models using provided data and scripts.

---



OCR for DjVu Files

https://github.com/jwilk/ocrodjvu (Relevance: 0.6)

Keywords: OCR, DjVu, text recognition, ocrodjvu, GPL-2.0
Stages: text-recognition
Architecture: 

Ocrodjvu is a software tool designed to facilitate Optical Character Recognition
(OCR) for DjVu files. It serves as a wrapper for various OCR engines, allowing
users to perform text recognition on DjVu documents. This tool supports multiple
OCR engines, including OCRopus, Cuneiform, Ocrad, GOCR, and Tesseract, and
requires Python 2.7 and DjVuLibre for operation. It can process pages in DjVu
files and output recognized text using commands like 'djvused'. The project was
developed with support from the Polish Ministry of Science and Higher Education
and is distributed under the GPL-2.0 license. The repository is archived and
read-only as of October 2022, indicating it is no longer actively maintained.

---



Symmetry-Based Text Line Detection

https://github.com/stupidZZ/Symmetry_Text_Line_Detection (Relevance: 0.6)

Keywords: text detection, symmetry, algorithm, OpenCV, parallel processing, Windows, ICDAR dataset, Visual Studio, CVPR
Stages: segmentation
Architecture: 

The Symmetry-Based Text Line Detection project provides a source code
implementation of a text detection algorithm that leverages symmetry for
identifying lines of text in images. It utilizes approximate calculation and
parallel processing technologies to enhance the algorithm's speed. The
algorithm's performance is close to its predecessor showcased at CVPR'15,
yielding slight differences in probability output when applied to the ICDAR
dataset. The code is developed for Windows x64 using Visual Studio 2012, with
dependencies on OpenCV 2.4.10 and VL-feat. Users must configure the 'config.txt'
file for parameters, which include dataset paths, mode flags, image scales, and
training sample criteria. The project supports both training and testing modes,
with specific instructions for dataset preparation and annotation. The
repository offers a comprehensive guide for installation and execution, along
with an example dataset to help users get started.

---



GNU Ocrad Docker Container

https://github.com/kba/ocrad-docker (Relevance: 0.6)

Keywords: GNU Ocrad, Docker, OCR, Feature extraction, Text recognition, Image processing, GPL-3.0, Layout analysis, Containerization
Stages: layout-analysis, text-recognition
Architecture: 

The 'GNU Ocrad in a docker container' project provides a Dockerized version of
GNU Ocrad, an Optical Character Recognition (OCR) program. Ocrad is based on a
feature extraction method and can process images in pbm, pgm, and ppm formats,
outputting text in byte or UTF-8 formats. This containerized application
includes a layout analyzer to separate text blocks and columns on printed pages,
offering various options like scaling, charset, and layout analysis to optimize
OCR performance. The project aims to simplify the deployment and usage of GNU
Ocrad by encapsulating it in a Docker container, making it accessible and easy
to use across different environments. The working directory for the container is
set to '/data', where users can mount files to be processed. The project is
licensed under GPL-3.0 and can be found on Docker Hub under the repository
'kbai/ocrad'.

---



Awesome OCR Projects

https://github.com/kba/awesome-ocr (Relevance: 0.6)

Keywords: OCR, Optical Character Recognition, Tesseract, EasyOCR, OCR engines, OCR libraries, OCR datasets, OCR tools, Open Source
Stages: 
Architecture: 

The 'awesome-ocr' repository is a curated list of resources related to Optical
Character Recognition (OCR). It serves as a comprehensive directory of OCR
software tools, libraries, datasets, and literature. The repository includes OCR
engines such as Tesseract and EasyOCR, older OCR engines, various OCR file
formats, and tools for OCR preprocessing and evaluation. There are also
resources for OCR as a service and numerous OCR libraries available in different
programming languages like Python, Java, and Swift. Additionally, the repository
lists OCR training tools and datasets with ground truth for model training and
evaluation. Contributions and feedback from the community are encouraged to
enhance the repository's breadth and depth.

---



OCR File Format Converter

https://github.com/UB-Mannheim/ocr-transform (Relevance: 0.6)

Keywords: OCR, file conversion, validation, hOCR, ALTO, PAGE, FineReader, open source, Docker
Stages: output-representation
Architecture: 

The OCR File Format project by UB-Mannheim provides tools to validate and
transform various OCR file formats including hOCR, ALTO, PAGE, and FineReader.
It supports both command-line and graphical user interfaces, as well as API
access for integration into other applications. The project is available as a
Docker container and can be installed system-wide. Key functionalities include
converting between different OCR formats and validating files against format
schemas. This open-source project is licensed under the MIT License, and it aims
to facilitate the handling of OCR data by providing flexible and robust
transformation and validation tools.

---



ScanTailor Advanced

https://github.com/4lex4/scantailor-advanced (Relevance: 0.6)

Keywords: image-processing, scanned-documents, book-scanning, binarization, digitalization, post-processing, ScanTailor, PDF, DjVu
Stages: preprocessing
Architecture: 

ScanTailor Advanced is an enhanced version of the ScanTailor software,
integrating features from the ScanTailor Featured and ScanTailor Enhanced
versions while introducing new functionalities and fixes. It serves as an
interactive post-processing tool for scanned pages, enabling operations such as
page splitting, deskewing, and content selection. The software is designed to
produce pages ready for printing or assembly into PDF or DjVu files, excluding
scanning, OCR, and document assembly. Key features include auto margins, page
detection, multi-threading support, light and dark color schemes, adaptive
binarization, and various improvements in image processing capabilities. It
offers full control over output settings, multi-column thumbnails, and new zone
interaction modes, enhancing user efficiency and flexibility. ScanTailor
Advanced is open-source, licensed under GPL-3.0, and supports multi-threaded
batch processing, making it suitable for managing large-scale digitization
projects.

---



CRAFT Text Detection Reimplementation

https://github.com/backtime92/CRAFT-Reimplementation (Relevance: 0.5)

Keywords: CRAFT, text detection, PyTorch, deep learning, computer vision, scene text, machine learning, optical character recognition
Stages: segmentation
Architecture: convnet

The CRAFT-Reimplementation project is an open-source initiative that aims to
provide a robust implementation of the CRAFT (Character Region Awareness for
Text Detection) algorithm using PyTorch. The project is based on the original
research by Baek et al., which was published in 2019. This reimplementation
focuses on detecting text in natural scenes, providing high accuracy by
leveraging deep learning techniques. The repository includes code for training
and evaluating text detection models, along with pre-trained models for quick
deployment. The implementation is designed to work with CRAFT's architecture,
allowing users to detect text with high precision using convolutional neural
networks. The project targets researchers and developers interested in computer
vision and optical character recognition, offering them a comprehensive solution
for text detection tasks. The repository provides detailed instructions for
setting up the environment, training models, and performing evaluations, making
it a valuable resource for machine learning practitioners.

---



DetectText: Stroke Width Transform

https://github.com/aperrau/DetectText (Relevance: 0.5)

Keywords: text detection, stroke width transform, OpenCV, C++, image processing
Stages: segmentation
Architecture: 

DetectText is a project focused on text detection using the stroke width
transform (SWT) method. This technique is implemented in C++ and relies on
OpenCV and Boost libraries for its functionality. The project provides a way to
identify text in images, distinguishing text that is either darker or lighter
than the background. The source code is available for compilation with
instructions using both g++ and CMake. The algorithm's details are rooted in a
research project at Cornell University. DetectText is licensed under GPL-3.0,
and the repository has garnered a significant community interest with over 300
stars and 150 forks on GitHub.

---



XY-Cut Tree Document Segmentation

https://github.com/kavishgambhir/xy-cut-tree (Relevance: 0.5)

Keywords: document segmentation, xy-cut algorithm, tree structure, layout similarity, OpenCV, C++, image retrieval, GitHub
Stages: segmentation, layout-analysis
Architecture: 

The XY-Cut Tree project is a tool for segmenting documents using the recursive
xy-cut algorithm. This technique divides a document into segments stored in a
tree structure to facilitate layout similarity matching against a dataset. The
project is implemented in C++ and requires the OpenCV library for document
manipulation. The main functionality is provided by the 'xy_cut.cpp' script,
which processes documents, stores result blocks as binaries, and saves segmented
images. The project serves as a term project from January to July 2018, focusing
on document image retrieval based on layout. It is open-source and hosted on
GitHub, with no current forks or releases.

---



Curve Text Detector Repository

https://github.com/Yuliang-Liu/Curve-Text-Detector (Relevance: 0.5)

Keywords: curve text detection, SCUT-CTW1500, deep learning, object detection, scene text, document analysis, dataset, annotation, evaluation
Stages: segmentation, dataset, evaluation, evaluation-results
Architecture: 

The Curve Text Detector repository, created by Yuliang Liu, offers resources for
detecting and recognizing curved text in images. It includes train and test
code, datasets, detection and recognition annotations, evaluation scripts,
annotation tools, and a ranking system. The repository features the SCUT-CTW1500
dataset, which provides images and annotations for both training and testing
purposes, focusing on multi-oriented and curved text detection. The project is
no longer actively maintained, but it remains accessible for academic research.
Evaluation scripts for original detection and end-to-end evaluation are
provided, with examples for the Total-Text dataset. The repository primarily
uses Jupyter Notebook, C++, and Python, and it is open for academic use only.

---



Ancient Greek OCR Evaluation Tools

https://github.com/ryanfb/ancientgreekocr-ocr-evaluation-tools (Relevance: 0.5)

Keywords: OCR, evaluation, tools, accuracy, ancient Greek, Tesseract, open source, GitHub
Stages: evaluation
Architecture: 

The 'ancientgreekocr-ocr-evaluation-tools' project provides a suite of 19 tools
designed to measure the performance and quality of Optical Character Recognition
(OCR) outputs, particularly focusing on ancient Greek texts. These tools enable
users to test and evaluate OCR accuracy through various metrics and formats. A
notable feature is the 'tessaccsummary' script, which facilitates testing the
accuracy of Tesseract training files by comparing OCR output with ground truth
text data. The tools require a specific directory structure for input files and
support processing through the 'ocrevalutf8' wrapper script to handle UTF-8
encoded text. The project is hosted on GitHub and is open source, with
contributions welcome from the community.

---



SSTDNet: Single Shot Text Detector

https://github.com/HotaekHan/SSTDNet (Relevance: 0.5)

Keywords: SSTDNet, Single Shot Text Detector, Regional Attention, ICCV 2017, PyTorch, Object Detection, Text Detection, High-resolution Images, Machine Learning
Stages: segmentation
Architecture: convnet

SSTDNet implements the 'Single Shot Text Detector with Regional Attention', a
method spotlighted at ICCV 2017. This project, developed using PyTorch, is
designed for general object detection, with future updates planned to support
oriented text detection. The repository provides a comprehensive solution for
detecting objects in high-resolution images. Users need to organize their
datasets with images and corresponding text files that specify object positions
and labels. The labels are encoded into integers using a provided Excel file.
Key implementation features include dataset reading customizations, image size
configuration for training, and parameter settings such as learning rate and
batch size. Although primarily for general object detection, the project aims to
evolve for specific text detection tasks.

---



MathOCR: Scientific Document Recognition

https://github.com/chungkwong/MathOCR (Relevance: 0.5)

Keywords: OCR, Java, scientific documents, mathematical expressions, image processing, character recognition, open-source, LaTeX, HTML+MathML
Stages: preprocessing, layout-analysis, text-recognition
Architecture: 

MathOCR is a scientific document recognition system written in Java that focuses
on recognizing mathematical expressions. The system performs image
preprocessing, layout analysis, and character recognition without relying on
external OCR software. MathOCR can process various image formats and outputs in
formats like LaTeX and HTML+MathML. It features unique algorithms for structural
analysis and is designed to handle complex mathematical symbols. The project is
open-source under the GNU Affero General Public License and is currently in a
pre-alpha stage, meaning its accuracy might not be sufficient for practical
purposes. Java 8 or above is required to run MathOCR, and it can be used through
a prebuilt JAR or by building from the source code. The system includes a GUI
for ease of use and supports modularity for layout, segmentation, and output
format plugins.

---



Chinese OCR Training Program

https://github.com/hehongyu1995/chinese-ocr-train (Relevance: 0.5)

Keywords: Chinese OCR, CRNN, Keras, Training, GitHub, Optical Character Recognition, Deep Learning, Machine Learning
Stages: text-recognition, training-data-generation
Architecture: crnn

The 'chinese-ocr-train' project provides a training program for implementing
CRNN (Convolutional Recurrent Neural Network) using Keras in the Chinese OCR
(Optical Character Recognition) project. The training script, 'train.py', is
adapted from Keras' image OCR examples, and is designed to be integrated into
the existing Chinese OCR project. The project includes a script, 'text_gen.py',
which generates ground truth data, and encourages modification of paths and
hyperparameters to fit specific needs. The project is hosted on GitHub and
welcomes collaboration and discussion through the author's provided contact
information.

---



Nashi: Transcribe Scanned Pages with PageXML

https://github.com/andbue/nashi (Relevance: 0.5)

Keywords: OCR, PageXML, transcription, scanned pages, JavaScript, Python, Flask, LAREX, Kraken
Stages: layout-analysis, segmentation, output-representation
Architecture: 

Nashi is a tool designed to transcribe scanned pages using PageXML, supporting
both left-to-right (ltr) and right-to-left (rtl) languages. It is developed
using JavaScript and offers a complete web application architecture written in
Python/Flask. The application facilitates the import and export of scanned pages
to and from LAREX for semi-automatic layout analysis and performs line
segmentation with the help of Kraken. Users can interact with an interface that
highlights lines based on their status: red for lines without existing text,
blue for lines containing OCR data, and green for transcribed lines. The
platform supports various keyboard shortcuts to enhance user interaction, such
as navigation between text inputs, saving edits, and toggling zoom modes.
Additionally, a server setup guide is provided, which includes instructions for
installing dependencies like Redis and LAREX, configuring the application, and
setting up a database. Future development plans for Nashi include advanced text
editing capabilities, sorting of lines, and integration with external OCR
services.

---



SynthText: Synthetic Text Image Generation

https://github.com/ankush-me/SynthText (Relevance: 0.5)

Keywords: SynthText, synthetic images, text localization, CVPR 2016, Python, image generation, scene-text
Stages: training-data-generation
Architecture: 

SynthText is a project that provides code for generating synthetic text images,
as detailed in the paper 'Synthetic Data for Text Localisation in Natural
Images' by Ankush Gupta, Andrea Vedaldi, and Andrew Zisserman, presented at CVPR
2016. The tool is designed to create synthetic scene-text images for training
computer vision models in text localization tasks. It works by overlaying text
on natural images, leveraging pre-trained models for depth and segmentation.
Users can generate samples using a variety of fonts and text sources, and the
project supports both Python 2 and 3. The repository includes scripts for
visualizing results, using pre-processed background images, and adapting the
text generation for non-Latin scripts. Additionally, it offers a pre-generated
dataset of approximately 800,000 images. The core dependencies for running
SynthText include pygame, OpenCV, PIL, numpy, matplotlib, h5py, and scipy. The
project is open-source and distributed under the Apache-2.0 license.

---



Awesome OCR GitHub Repository

https://github.com/perfectspr/awesome-ocr (Relevance: 0.5)

Keywords: OCR, Optical Character Recognition, text extraction, GitHub, repository, libraries, software, resources, projects
Stages: 
Architecture: 

The Awesome OCR GitHub repository is a curated list of resources and tools
related to Optical Character Recognition (OCR). It compiles a comprehensive
collection of libraries, software, and projects that facilitate text recognition
and extraction from images. This repository serves as a valuable resource for
developers and researchers interested in OCR technology, providing access to
various implementations and solutions for text recognition tasks across
different platforms and languages.

---



Document Layout Analysis with Python-OpenCV

https://github.com/rbaguila/document-layout-analysis (Relevance: 0.5)

Keywords: document analysis, layout analysis, Python, OpenCV, text processing, computer vision, character boxing, paragraph detection
Stages: layout-analysis, segmentation
Architecture: 

This project, 'document-layout-analysis,' is a simple document layout analysis
tool developed using Python and OpenCV. It aims to process and analyze the
layout of documents by categorizing and boxing different elements such as
individual characters, words, lines, and paragraphs, as well as paragraphs with
margins. The repository contains the necessary scripts and files to execute the
analysis, including images used for testing the layout analysis process. To run
the application, users must execute 'python main.py' and ensure an 'output'
folder is created for storing results. The project serves as an introductory
implementation for document layout analysis, demonstrating basic computer vision
techniques applied to textual data.

---



Python Tesseract OCR Wrapper

https://github.com/madmaze/pytesseract (Relevance: 0.5)

Keywords: Python, OCR, Tesseract, Image Processing, Text Recognition, Pytesseract, Google, Pillow, Leptonica
Stages: text-recognition, output-representation
Architecture: 

Pytesseract is a Python wrapper for Google's Tesseract OCR engine, designed to
recognize and read text embedded in images. It supports multiple image formats
like JPEG, PNG, GIF, BMP, and TIFF, using the Pillow and Leptonica imaging
libraries. Pytesseract can be used as a stand-alone invocation script that
prints recognized text directly. Key features include support for multiple
languages, bounding box estimates, detailed data analysis with confidence
levels, and orientation and script detection. It also generates outputs in
various formats, such as searchable PDFs, HOCR, and ALTO XML. The package
requires Python 3.6+, the Python Imaging Library (PIL) or its fork Pillow, and
the Google Tesseract OCR engine. Installation is available via pip or conda, and
the package is Apache-2.0 licensed.

---



Deskew: Command Line Deskewing Tool

https://github.com/galfar/deskew (Relevance: 0.5)

Keywords: deskew, command line, image processing, Hough transform, scanned documents, text alignment, Pascal, open source, cross-platform
Stages: preprocessing
Architecture: 

Deskew is a command line tool designed to correct the skew of scanned text
documents, making the text lines horizontal. It leverages the Hough transform to
detect text lines within images and outputs a rotated image with corrected
orientation. The tool supports a variety of input and output image formats,
including BMP, JPG, PNG, and TIFF, among others. Deskew provides a GUI frontend
for users who prefer a graphical interface, available for Windows, Linux, and
macOS platforms. It supports multiple platforms with pre-compiled binaries for
Win64, Win32, Linux x86_64, macOS x86_64, and Linux ARM architectures. Deskew is
written in Object Pascal, requiring Free Pascal or Delphi for recompilation. The
project is open-source and licensed under the MPL-2.0 license. Users can specify
various options such as output image file, maximum skew angle, background color,
and output pixel format during execution. Additionally, it features advanced
options for resampling filters, content rectangle skew detection, and output
compression specifications. The tool is suitable for developers and users
needing efficient text document image correction.

---



CRNN in MXNet for Chinese OCR

https://github.com/novioleo/crnn.mxnet (Relevance: 0.5)

Keywords: CRNN, MXNet, OCR, Chinese characters, captcha, Android support, Docker, Python, Machine Learning
Stages: text-recognition, training-data-generation
Architecture: crnn

The project 'crnn.mxnet' focuses on implementing a Convolutional Recurrent
Neural Network (CRNN) using the MXNet framework, specifically aimed at Optical
Character Recognition (OCR) with support for Chinese characters. It includes
support for Android and provides a training script for captcha datasets, such as
those from Taobao. The project requires MXNet version 0.11.0 or higher and
utilizes Python Imaging Library (PIL) for generating training data. Users can
customize training with various parameters like character set, batch size,
sequence length, and image dimensions. The project also offers Docker support
for creating portable environments, though GPU training via Docker requires
additional configuration. The repository provides a Python script for
prediction, and users needing help can raise issues for support.

---



Scan Tailor: Interactive Post-Processing Tool

https://github.com/scantailor/scantailor (Relevance: 0.5)

Keywords: Scan Tailor, post-processing, scanned documents, open-source, page splitting, deskewing, C++, Qt, GPL
Stages: preprocessing
Architecture: 

Scan Tailor is an interactive post-processing application designed for enhancing
scanned documents. It offers features such as page splitting, deskewing, and
border management, enabling users to prepare scanned pages for printing or
conversion into digital formats like PDF and DJVU. Developed in C++ with Qt,
Scan Tailor is open-source software released under the GPL v3 license. Initially
developed by Joseph Artsimovich, it reached production quality by 2010 and was
later maintained by Nate Craun. Although the project is no longer actively
maintained, its applications are widely used by individuals and institutions for
processing scanned books, notably visible in collections on Google Books and the
Internet Archive. Installation guides, user manuals, and additional resources
are available in the project's wiki for those interested in using or
contributing to Scan Tailor.

---



go-ocr: Text Extraction Tool

https://github.com/maxim2266/go-ocr (Relevance: 0.5)

Keywords: OCR, text extraction, scanned documents, PDF, DJVU, filters, tesseract, go-ocr, archived
Stages: text-recognition
Architecture: 

The 'go-ocr' project is a tool designed for extracting plain text from scanned
documents in PDF or DJVU formats using Optical Character Recognition (OCR). It
leverages 'pdfimages' or 'ddjvu' for image extraction and 'tesseract' for OCR
processing. Users can define custom post-processing filters to handle OCR
artifacts and irregularities. The tool offers command-line options for
specifying page ranges, languages, output files, and filter definitions. 'Go-
ocr' is intended to streamline the OCR process by integrating image extraction
and text filtering in a single pipeline, reducing manual text correction
efforts. The project is archived and has been superseded by a newer project,
which offers more granular control over text recognition processes. It is
licensed under the BSD-3-Clause license and was primarily tested on Linux
systems.

---



Awesome OCR Resource List

https://github.com/wanghaisheng/awesome-ocr (Relevance: 0.5)

Keywords: OCR, Optical Character Recognition, Tesseract, PaddleOCR, AI papers, Deep Learning, Text Recognition, OCR tools, Commercial OCR
Stages: 
Architecture: 

The 'Awesome OCR' repository, curated by Wang Haisheng, is a comprehensive
collection of resources related to Optical Character Recognition (OCR)
technologies. It includes a variety of libraries, applications, and tools for
OCR development and implementation. The repository features popular OCR engines
like Tesseract, PaddleOCR, and commercial solutions from companies such as ABBYY
and Cloudmersive. It also highlights numerous scripts and tools for AI-related
paper collection and monitoring OCR developments. Additionally, the resource
list provides links to numerous academic papers, projects, and commercial
products that leverage OCR technologies. It also includes practical guides and
blogs for implementing OCR solutions, especially for languages and complex
scenarios such as scene text recognition and CAPTCHA solving. The repository is
updated with daily OCR paper tracking and offers a platform for sharing and
modifying keywords for research purposes.

---



OCRFeeder: OCR Suite by GNOME

https://github.com/GNOME/ocrfeeder (Relevance: 0.5)

Keywords: OCR, GNOME, Document Analysis, Open Source, Python, Optical Character Recognition, GPL-3.0, Command-line interface, Graphical interface
Stages: text-recognition, layout-analysis
Architecture: 

OCRFeeder is a comprehensive Optical Character Recognition (OCR) application
developed by GNOME. It serves as a document analysis and recognition suite,
providing both graphical and command-line interfaces. The program allows users
to convert scanned documents into editable text. It was authored by Joaquim
Rocha and includes contributions from Igalia, S.L. and Søren Roug from the
European Environment Agency. OCRFeeder is free software, available under the GNU
General Public License version 3, ensuring it can be modified and redistributed
under the same terms. The project is primarily written in Python and is hosted
as a read-only mirror on GitHub, with its main repository on GitLab. The
software does not provide any warranty, and its development involves multiple
contributors.

---



Paperwork: Personal Document Manager

https://github.com/openpaperwork/paperwork (Relevance: 0.5)

Keywords: document management, Linux, Windows, OCR, PDF, Python, scanner, GTK, open source
Stages: 
Architecture: 

Paperwork is a personal document management system for Linux and Windows,
designed to help users organize and manage digital documents efficiently.
Originally hosted on GitHub, the project has been moved to Gnome's GitLab. The
software supports various functionalities such as OCR (Optical Character
Recognition), document indexing, and integration with scanners, making it a
versatile tool for handling PDF and other document formats. Written primarily in
Python, Paperwork leverages technologies like GTK for its graphical user
interface. The project has garnered interest from the open-source community, as
evidenced by its 2.4k stars and 149 forks on GitHub. Despite being archived on
GitHub, development continues on GitLab, ensuring ongoing updates and community
contributions.

---



pdf2pdfocr: Open Source PDF OCR Tool

https://github.com/LeoFCardoso/pdf2pdfocr (Relevance: 0.5)

Keywords: OCR, PDF, Open Source, Searchable PDF, Tesseract, Docker, Linux, macOS, Python
Stages: 
Architecture: 

pdf2pdfocr is a free and open-source tool designed to perform Optical Character
Recognition (OCR) on PDF files or supported image formats, enabling the addition
of a text layer to the original file. This process transforms the file into a
searchable PDF. The tool exclusively uses open-source resources and provides
both command-line and graphical user interface (GUI) options for ease of use.
Installation is straightforward on Linux and macOS, with detailed instructions
provided, including optional components for enhanced performance. The project
supports Docker, allowing the application to run within a container. For Windows
users, manual installation of dependencies is necessary. The project is licensed
under the Apache-2.0 license and welcomes donations to support further
development.

---


