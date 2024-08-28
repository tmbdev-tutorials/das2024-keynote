# Title Slide
- LLMs for OCR Post-Correction
- Martijn Veninga
- University of Twente, Faculty of Electrical Engineering, Mathematics and Computer Science
- July, 2024

# Introduction
- Examination of Large Language Models (LLMs) for Optical Character Recognition (OCR) post-correction
- OCR post-correction involves correcting mistakes in text identified from images or documents
- Improving OCR accuracy enhances downstream tasks like text summarization and named entity recognition

# Background/Related Work
- State-of-the-art OCR models are efficient but still produce errors
- Errors impact downstream natural language processing tasks
- Traditional post-processing techniques: voting schemes, n-gram models, crowd-sourced corrections

# Contributions
- Fine-tuned ByT5 LLM outperforms state-of-the-art methods for OCR correction
- Preprocessing techniques like lowercasing and removing strange characters improve model performance
- Identified optimal context length for best performance

# Objective
- Investigate the potential of LLMs in correcting OCR errors
- Evaluate the effectiveness of preprocessing techniques
- Compare fine-tuned character-level LLMs with generative LLMs like Llama

# Methodology Overview
- Use of fine-tuned ByT5 LLM for OCR post-correction
- Comparison with baseline state-of-the-art methods
- Evaluation of preprocessing techniques

# Datasets
- ICDAR 2019 post-OCR text correction dataset: old documents in English, Dutch, and German
- Custom dataset from modern Dutch documents: created by degrading PDFs and OCRing them

# Model Details
- ByT5 models: small (300 million parameters) and base (580 million parameters)
- Generative LLM Llama 7B: tested with zero-shot and few-shot learning
- Baseline model: ensemble of sequence-to-sequence models

# Experiments
- Evaluation of different context lengths: 10, 25, 50, and 100 characters
- Testing preprocessing techniques: lowercasing and removing strange characters
- Comparing performance on custom and ICDAR datasets

# Results
- ByT5 base model with 50-character context length and preprocessing achieved 56% CER reduction
- Baseline method achieved 48% CER reduction on custom dataset with strange characters removed
- ByT5 models showed higher recall but lower precision compared to baseline

# Performance Comparisons with Prior Work
- ByT5 models outperformed baseline on custom dataset but not on ICDAR dataset
- Llama model struggled with hallucination and did not improve OCR results

# Discussion
- Preprocessing enhances model performance but varies by technique and model
- Larger ByT5 models might show further improvements
- Limitation: lack of publicly available datasets for modern documents

# Conclusion
- ByT5 models are effective for OCR post-correction on modern documents
- Preprocessing techniques like lowercasing and removing strange characters are beneficial
- Future work: testing larger models, fine-tuning Llama, and organizing new competitions for modern documents

# References
- Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"
- Vaswani et al., "Attention is all you need"
- Rigaud et al., "ICDAR 2019 competition on post-OCR text correction"
- Xue et al., "Byt5: Towards a token-free future with pre-trained byte-to-byte models"
- Touvron et al., "LLaMA: Open and Efficient Foundation Language Models"

# Acknowledgements
- Thanks to the University of Twente and the collaborating tech company for resources and support

# Q&A
- Invitation for Questions
