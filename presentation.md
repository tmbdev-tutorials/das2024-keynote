---
marp: true
theme: default
paginate: true
footer: 'OCR and LLMs Presentation'

headingDivider: 1
---

<!--
Notes:
-- llms for training quality evaluation, post-correction
-- vlms for image categorization, segmentation via backprop etc
-->

# OCR and LLMs

# Overview

- LLMs/VLMs have replaced most NLP/vision models. OCR?
- Objectives and uses of document analysis in this new world.
- Low/high accuracy worlds.
- Using low accuracy to bootstrap high accuracy.

# REVOLUTION IN MACHINE LEARNING / AI

# LLMs / VLMs Have been Eating Machine Learning

- tasks that used to require extensive, specialized training...
- ... are now handled by foundation models
- ... or with minimal fine tuning

# Foundation Models

- self-supervised training on vast amounts of unlabeled data (images, text)
- multitask learning
- task specifications in natural language
- generalization to new tasks and classes through natural language

# Example: Zero-Shot Document Classification

```Python
prompt = """
### Instructions

You are given the text of the first page of a PDF document. Please extract the title,
author, year, and abstract. Then assign a category to the document
from the following list of categories:

- ocr: text recognition, layout analysis, page segmentation
- handwriting: handwriting recognition, handwriting synthesis, etc.
- scene-text: text recognition in natural images and scenes
... more categories ...
- other: anything else

You must return only a JSON format dictionary with fields of
title, author, abstract, year, and category. Your output
will be parsed by machine.
"""

classifier = OpenAIClient(prompt)
result = classifier.json_query(text)
```

# Zero/Few Shot with LLMs

- Named Entity Recognition (NER)
- Document Categorization
- Sentiment Analysis
- Text Summarization
- Machine Translation
- ...

# Zero/Few Shot with VLMs

- Object Recognition/Classification
- Object Detection
- Scene Understanding
- Action Recognition
- ...

# Require Specialized Custom Models

- Photometric Stereo
- Gaze Estimation
- Anomaly Detection
- 3D Pose Estimation for Articulated Objects
- ...

# FOUNDATION MODELS IN OCR

# Traditional OCR

- high accuracy scanned-to-text conversion
- $<0.5%$ character error
- high quality reading order, layout
- ideally, recover markup (LaTeX, etc.)

# "OCR-Free" Transformers

- text localization (receipts, etc)
- page segmentation and reading order (PubLayNet, PubTables-1M)
- visual question answering (VQA, DocVQA)
- key information extraction (KIE on SROIE)
- no widely used end-to-end OCR benchmarks

# Transformer-Based "Traditional" OCR

Some transformer-based systems capable of converting full page
scans to text with bounding boxes:

- UDOT (CER 2.56%, IOU 91.62%)
- Nougat (CER 25.5%)
- Kosmos 2.5 (CER 9.2%, IOU 82.1%)

(Other systems run "on top of" traditional OCR.)

# NVIDIA OCR Efforts and Foundation Models

Ambitious all-in-one effort:

- VLMs that handle vision, scenes, and documents
- document capabilities:
    - high accuracy image-to-text for books, articles
    - outputs logical and physical markup (headers, footnotes, etc.)
    - handles math and other special content
- massive training and data management effort due to generality of model

# FUTURE OF OCR AND LLMS

# OCR Future

Three different possible scenarios:

1. all in one models: vision+documents $\rightarrow$ text/structure
2. traditional OCR + transformer for logical layout
3. all-in-one transformer with post-processing/correction by specific models

# What do we actually want?

- high quality/accuracy conversions of traditional documents
- high accuracy information extraction
- conversion as input to LLM training
- conversion as input to LLM inference

# LLMs and Noise

- traditional NLP was fragile: grammar, ambiguities, etc.
- LLMs are robust to noise in training data and questions
- robustness to OCR errors can be enhanced by augmenting training data with OCR errors
- redundancy of training data likely partially responsible (facts are represented many times)

# LLMs and Noise (2)

- OCR is also used for LLM input, e.g.:
    - data extraction from financial documents
    - question answering from biomed papers
- high accuracy (e.g., correct numbers) much more important
- LLMs may still have some robustness to layout errors
- LLM-related semantic errors may dominate

# OCR for LLM training and inference

- LLMs deal well with 

# Required vs Actual Error Rates

# Language Model Hallucinations / Corrections

# What are the Use Cases?

- used to be search and retrieval, reformatting
- these days: training, RAG, LLM analysis, information extraction, question answering
- conversion still matters though

# Data Sources for OCR Training and Applications

# Do We Still Need Scanned Book Conversions?

# Paperless Is Here

# The PDF / Word / XML / Microformat / Microdata Mess

- PDF has no standard semantic labeling
- semantic data in PDF via embedded XML
- LaTeX can't produce tagged PDF

# Data Sources as LLM Inputs?

# PyMuPDF + ChatGPT

# OCRBench Tasks

# What does Knowledge Look Like?

# LLM Book Derived Knowledge vs Actual Knowledge

# Wikidata Claims

# Efficiency of Natural Language

# RDF Triples

# Logical Inference in LLMs?

# Attempts: Integrate Knowledge Graphs into LLMs

# Classical Problem: Term Resolution

# Classical Problem: Disambiguation

# Classical Problem: Ontology Mapping

