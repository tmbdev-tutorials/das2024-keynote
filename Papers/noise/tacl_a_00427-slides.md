# Lexically Aware Semi-Supervised Learning for OCR Post-Correction

- Shruti Rijhwani, Daisy Rosenblum, Antonios Anastasopoulos, Graham Neubig
- Carnegie Mellon University, University of British Columbia, George Mason University
- Date of Presentation

# Introduction

- Problem: OCR systems often fail on less-resourced languages
- Motivation: Improve OCR results for endangered languages
- Scope: Propose a semi-supervised learning method using self-training and lexically aware decoding

# Background/Related Work

- Previous Work: Neural post-correction methods for OCR
- Limitations: Reliance on manually curated data
- Key Concepts: OCR, neural sequence-to-sequence models, self-training

# Contributions

- Semi-supervised learning method for OCR post-correction
- Lexically aware decoding using weighted finite-state automata (WFSA)
- Demonstrated effectiveness on four endangered languages

# Objective

- Main Goal: Utilize unannotated raw images to improve OCR post-correction
- Hypothesis: Combining self-training and lexically aware decoding enhances performance

# Methodology Overview

- Approach: Iteratively train a model on its own outputs (self-training)
- Data: Manually transcribed and unannotated raw images
- Model: Neural sequence-to-sequence model with WFSA for decoding

# Datasets

- Ainu: 816 transcribed, 7646 unannotated lines
- Griko: 807 transcribed, 3084 unannotated sentences
- Yakkha: 159 transcribed, no unannotated lines
- Kwak’wala: 262 transcribed, 2255 unannotated lines

# Model Details

- Architecture: Attention-based LSTM encoder-decoder
- Key Components: Character embeddings, bidirectional LSTM, attention mechanism
- Training: Supervised with manual transcriptions, pseudo-training with self-training outputs

# Self-Training

- Steps: Predict on unannotated data, create pseudo-annotated dataset, retrain model
- Fine-tuning: Retrain on gold-transcribed data after pseudo-training
- Iterations: Repeat until performance stabilizes

# Lexically Aware Decoding

- Strategy: Use a count-based language model to enforce vocabulary consistency
- WFSA: Represent language model for efficient decoding
- Combination: Joint decoding with LSTM and WFSA

# Experiments

- Setup: 10-fold cross-validation, character and word error rates (CER, WER)
- Baselines: First-pass OCR systems (Google Vision, Ocular), BASE model

# Results

- Performance: Significant reduction in error rates with semi-supervised learning
- Best Model: Combination of self-training and lexically aware decoding
- Error Reductions: 15%–29% over state-of-the-art methods

# Performance Comparisons with Prior Work

- Baselines: First-pass OCR systems' error rates compared
- Improvements: Semi-supervised learning outperforms baseline methods consistently
- Error Reductions: Higher improvements for languages with higher initial error rates

# Discussion

- Insights: Lexically aware decoding complements self-training
- Interpretation: Reinforces correct predictions, mitigates noise
- Limitations: Occasional introduction of new errors

# Conclusion

- Summary: Semi-supervised learning method enhances OCR post-correction
- Impact: Reduces error rates significantly for endangered languages
- Future Work: Incorporate additional linguistic resources, explore morphological analysis

# References

- Dong and Smith, 2018
- Rijhwani et al., 2020
- Zhu and Goldberg, 2009
- He et al., 2020

# Acknowledgements

- Supported by Bloomberg Data Science Ph.D. Fellowship, National Endowment for the Humanities, National Science Foundation, National Research Council Indigenous Language Technology grant, Government of Canada Social Sciences and Humanities Research Council Insight Development grant

# Q&A

- Invitation for Questions
