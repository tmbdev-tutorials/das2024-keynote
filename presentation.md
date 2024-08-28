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
- retrieval, summarization, etc.
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
  - VQA performance may be limited by LLM performance
  - that's why so-so "OCR-free" CER may be acceptable

# Where do we stand?

- traditional OCR _may_ be adequate for both training and inference of LLMs
- "OCR-free" VQA _may_ have advantages because it can refer to visual content
- transformer based OCR system have not yet clearly demonstrated superior performance
- very high accuracy OCR (image $\rightarrow$ markup) still needed

<!--

# HTML Microdata / Microformats

```html
  <div itemscope itemtype="http://xbrl.org/TotalRevenue">
      <span>Total Revenue: </span>
      <span itemprop="amount" content="1300000000">$1.3 bn</span>
      <meta itemprop="currency" content="USD">
  </div>
```

- seamless embedding of metadata in HTML
- easy to implement, easy to make consistent
- common standards (XBRL)
- example of high quality semantic embedding
- also a good _target_ format for OCR systems (hOCR)

# The PDF Disaster

PDF has been disastrous from the point of view of moving to a paperless society:

- originated (via PS) as a printer page language
- most documents have no reading order, layout information
- huge variety of actual PDF file contents for the same appearance
- almost no semantic information standards
- the few that exist simply embed XML along with the PDF
- not even LaTeX generates tagged PDF by default
- Word is not much better, but at least has reading order

-->


# MATERIALS, USERS AND MARKETS

# Example: Internet Archive (<10 users)

- large collection of scanned books
- both pre-1924 and post-1924
- can't easily get digital data even when available
- requires very low error rates, high quality markup
- requires high quality image-based OCR

# Example: Biomedical Researcher (> 100000000 users)

- collection of papers
- most native digital PDF (post 2000), some scanned
- wants to use LLMs to help with categorization, retrieval
- does not require high accuracy, just enough to do retrieval
- mix of approaches:
  - typical OCR for scanned docs (Tesseract is likely sufficient)
  - pdf2text tools for PDFs (occasional reading order problems acceptable)
  - also served by OCR-free VQA (but may be more expensive/slow)

# Example: Financial Services Provider (< 1000 users)

- large, steady influx of native digital PDFs
- typical, specialized layouts and contents
- approaches:
  - image-based OCR _undesirable_ because of potential for character errors
  - requires high throughput, low computational cost
  - use text from native digital PDF
  - combine with Transformer-based information extraction
  - fine tuned models for high domain specific performance
  - "OCR-free VQA" currently very far from meeting performance requirements

# Example: Companies Training Foundation Models (< 100 users)

- large collection of scanned materials (hopefully licensed)
- need large amounts of training data for LLMs
- LLM training is robust to character/layout errors
- information is redundant across many books
- approaches:
  - use an existing off-the-shelf OCR system
  - train a transformer-based OCR system
  - directly train a VQA model

  # Users, Use Cases, and Systems

- many use cases adequately taken care of by existing systems
  - low CER/WER, moderate IOU layout errors
- still needed, in development
  - very high accuracy end-to-end OCR
  - text from native digital, layout/semantics from transformers

# LLMS AND KNOWLEDGE

# MMLU Question Design and Types
- **Subject Selection**: Questions cover a wide range of subjects (humanities, social sciences, hard sciences, professional fields) with varying difficulty levels (Elementary, High School, College, Professional) 
- **Source Collection**: Manually collected from freely available sources, including standardized test practice questions and educational resources 
- **Question Format**: Designed as multiple-choice questions to facilitate assessment, each with one correct answer and several distractors 
- **Knowledge-Based Questions**: Approximately **60%** of questions require recall of factual information.
- **Inference and Reasoning Questions**: About **40%** of questions require applying knowledge, making connections, or solving problems 

# MMLU Examples of Multiple-Choice Questions
## **Mathematics Example**:
If 4 daps = 7 yaps, and 5 yaps = 3 baps, how many daps equal 42 baps? 
(A) 28 (B) 21 (C) 40 (D) 30
(**Answer**: (C) 40)
## **Biology Example**:
What is the powerhouse of the cell? 
(A) Nucleus (B) Mitochondria (C) Ribosome (D) Endoplasmic Reticulum
(**Answer**: (B) Mitochondria)

# MMLU Model Performance on the Benchmark
## **Human Performance**:
- Unspecialized humans: **34.5%** accuracy.
- Expert-level (95th percentile) for US Medical Licensing Exams: **87%** accuracy.
## **Language Model Performance**:
- Largest GPT-3 model (175 billion parameters): Improved by almost **20 percentage points** over smaller models.
- Estimated expert-level accuracy across subjects: **89.8%** 

# Types of Tasks involved in LLM Answers

- knowledge of facts ("Lincoln was president")
- knowledge of meta-facts ("this fact is true according to...")
- ability to reason ("Lincoln was a US citizen because he was president.")
  - reasoning can be imitated with factual knowledge
- ability to recall verbating ("please quote ...")

# Scaling Laws for Neural Language Models

![Scaling Laws](Figures/scaling-laws.png)

Kaplan et al. 2021

# Source of these Scaling Laws: Simple Model

Simple Model:
- sequence of training samples
- collection of facts $F_i$ to be learned
- occurrence of facts is determined by exponential distributions with parameters $\lambda_i$
- the $\lambda_i$ follow a power law / Zipf's law distribution
- we assume that a fact is known once it is seen

Prediction:
- what percentage of facts have been seen after $x$ samples?

# Simple Model Scaling Laws

![h:500px](Figures/order-distribution-exponential.png){height=50%}

# Note on Scaling Laws

Observations:
- it takes many times the number of facts for the model to learn most
- the time it takes to learn is highly dependent on the exponent in the power law

Conclusions:
- massive scanning combined with ever larger models is probably not efficient

# Machine Learning Approaches

- use umbrella sampling
  - sample with an emphasis on documents containing rare facts
  - how do we know which documents contain rare facts and which are redundant?
  - we can simply ask: "hey, LLM, do you already know most of what's in this book?"
- use boosting
  - resample the dataset based on prediction errors
  - combine using boosting algorithm
  - needs to be extended to LLMs though

# What do Facts Look Like?

```YAML
wikidata_item:
  item_id: Q42  # Unique identifier
  labels: "Douglas Adams"  # Main name
  descriptions: "English writer and humorist"  # Short description
  aliases: ["Douglas Noël Adams"]  # Alternative names
  sitelinks: ["https://en.wikipedia.org/wiki/Douglas_Adams"]  # Wikipedia link

  statements:
    - property: Height
      property_id: P2048
      value: "185 cm"  # Simple property example
    
    - property: Educated at
      property_id: P69
      value: {item: "St John's College, Cambridge", item_id: Q691283}  # Statement pointing to another Q item
      qualifiers:
        - {qualifier: Start date, qualifier_id: P580, value: 1971}
        - {qualifier: End date, qualifier_id: P582, value: 1974}
      references:
        - {reference_property: Reference URL, reference_property_id: P854, value: "https://source.link"}  # Reference URL
```

# Pre-training Data Augmentation

How it works:  
- Large-scale knowledge graphs like Wikidata or Freebase provide structured triples (e.g., "Paris is the capital of France").
- These triples are converted into natural language statements or incorporated directly into the pre-training corpus.
- By exposing the LLM to these factual statements during training, the model learns to associate entities and their relationships, enhancing its knowledge of world facts.

Interaction between LLM and Knowledge Graph:  
- The knowledge graph provides structured, factual data that the LLM integrates during training.
- This leads to the model acquiring a richer understanding of relationships between entities (e.g., people, places, concepts), which it can then apply during inference.

Petroni, F., et al. (2019). "Language Models as Knowledge Bases?" https://arxiv.org/abs/1909.01066

# Knowledge Injection

How it works:  
- Knowledge injection involves incorporating structured data from knowledge graphs (e.g., Cyc, ConceptNet) into the training process of the LLM. This is done in multiple ways:
- Knowledge-Enhanced Data as Augmented Text: Facts from the knowledge graph are converted into natural language and added to training data. Example: "Paris, the capital of France, is a beautiful city."
- Entity Linking and Fact Tagging: Entities in the training data are linked to their corresponding nodes in the knowledge graph and tagged with relevant facts.
- Graph Embeddings as Input Features: Knowledge graph embeddings are fed into the model alongside text embeddings to inject structured knowledge.

Interaction between LLM and Knowledge Graph:  
- The LLM incorporates graph-based facts during training via enriched text data or additional features like graph embeddings. This helps the model generate text that reflects known facts and relationships.

Bosselut, A., et al. (2019). "COMET: Commonsense Transformers for Automatic Knowledge Graph Construction." https://arxiv.org/abs/1906.05317

# Post-processing for Query Refinement

How it works:  
- After the LLM generates a response, the output is checked against a knowledge graph to ensure factual accuracy. If discrepancies are found, the response is refined by incorporating correct information from the knowledge graph.

Interaction between LLM and Knowledge Graph:  
- The knowledge graph acts as a post-generation filter, ensuring that the model's responses are consistent with known facts. This validation step corrects or refines generated text based on real-time graph data.

Logan IV, R. L., et al. (2019). "Barack’s Wife Hillary: Using Knowledge Graphs for Fact-Aware Language Modeling." https://arxiv.org/abs/1906.07241

# Graph-based Reasoning

How it works:  
- The LLM queries the knowledge graph to retrieve relevant facts or perform multi-hop reasoning, allowing it to infer new information based on known relationships in the graph.

Interaction between LLM and Knowledge Graph:  
- The LLM interacts with the knowledge graph in real-time, querying it to assist with reasoning tasks that require understanding complex relationships between entities. This provides logical deductions beyond the model’s static training data.

Xiong, W., et al. (2020). "Answering Complex Open-Domain Questions with Multi-Hop Dense Retrieval." https://arxiv.org/abs/2009.12756

# Embedding Alignment

How it works:  
- Knowledge graph embeddings (vector representations of entities and relationships) are aligned with LLM embeddings during training. This alignment allows the LLM to better understand and integrate structured knowledge from graphs with natural language.

Interaction between LLM and Knowledge Graph:  
- The LLM uses the aligned embeddings to bridge the gap between structured knowledge and unstructured text. This improves performance on tasks like entity linking and relationship extraction.

Wang, X., et al. (2021). "KEPLER: A Unified Model for Knowledge Embedding and Pre-trained Language Representation." https://arxiv.org/abs/1911.06136

# Contextual Retrieval

How it works:  
- During text generation, the LLM retrieves relevant facts from a knowledge graph based on the current context (e.g., a question or prompt). This real-time retrieval ensures that the generated text is accurate and contextually relevant.

Interaction between LLM and Knowledge Graph:  
- The LLM queries the knowledge graph in real-time to retrieve information that enhances the generated response. The knowledge graph serves as a dynamic source of factual data that the model incorporates into its output.

Guu, K., et al. (2020). "REALM: Retrieval-Augmented Language Model Pre-Training." https://arxiv.org/abs/2002.08909

# SPARQL Query Generation for Knowledge Retrieval

How it works:  
- The LLM generates SPARQL queries to retrieve specific information from a knowledge graph. For example, a question like "What is the capital of France?" would lead the LLM to generate a SPARQL query such as:
  """
  SELECT ?capital WHERE {
    wd:Q142 wdt:P36 ?capital .
  }
  """
  This query retrieves the capital of France from the knowledge graph.

Interaction between LLM and Knowledge Graph:  
- The LLM generates SPARQL queries based on natural language input and retrieves information from the knowledge graph in real-time. This method leverages the structured nature of the knowledge graph for precise information retrieval during inference.

Sun, Z., et al. (2020). "Reasoning over Entity-Action-Relation Graphs for Open-Domain Question Answering." https://arxiv.org/abs/2012.15315




# Utilizing Knowledge more Efficiently



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

