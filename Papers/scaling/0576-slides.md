# Case-Based Reasoning with Language Models for Classification of Logical Fallacies

- Zhivar Sourati, Filip Ilievski, Hˆong- ˆAn Sandlin, Alain Mermoud
- University of Southern California, armasuisse Science and Technology
- Presented at IJCAI-23

# Introduction

- Misinformation and propaganda spread rapidly on the Web.
- Detecting logical fallacies in arguments is crucial.
- Current language models lack robustness for complex reasoning tasks.
- Propose a Case-Based Reasoning (CBR) method for logical fallacy classification.

# Background/Related Work

- Previous work has focused on binary misinformation detection.
- Logical Fallacy Detection aims to classify arguments into specific fallacy types.
- Informal fallacies are challenging for both humans and machines.
- Current NLP models struggle with tasks requiring complex reasoning.

# Contributions

- First CBR method for logical fallacy classification.
- Four enriched case representations: Counterarguments, Goals, Explanations, Structure.
- Extensive experiments on in-domain and out-of-domain settings.
- Code and data made available for future research.

# Objective

- Improve the ability of language models to classify logical fallacies.
- Use reasoning over examples to enhance model performance.
- Investigate the impact of different case representations on CBR performance.

# Methodology Overview

- CBR method involves retrieving similar cases, adapting them, and classifying new cases.
- Use language models as retrievers and adapters.
- Enrich case representations with additional information.

# Datasets

- LOGIC dataset: 13 logical fallacy types from common topics.
- LOGIC Climate dataset: Logical fallacies on the climate change topic.
- Augment LOGIC dataset to balance fallacy types.

# Model Details

## Retriever

- Finds k similar cases from a case database.
- Uses cosine similarity to measure similarity between cases.

## Adapter

- Prioritizes relevant information from similar cases.
- Uses multi-headed attention to fuse information from retrieved cases.

## Classifier

- Fully connected neural layer.
- Uses cross-entropy loss for classification.

# Experiments

- Evaluate CBR against Transformer LM baselines.
- Use weighted precision, recall, and F1-score for evaluation.
- Analyze performance with different case representations and database sizes.

# Results

- CBR consistently outperforms vanilla LMs.
- Best performance achieved with counterarguments as case representation.
- CBR models are effective with a small number of retrieved cases.
- Low sensitivity to the size of the case database.

# Performance Comparisons with Prior Work

- CBR outperforms Codex in few-shot setting.
- Significant improvement over frequency-based baseline.
- Enhanced generalization to out-of-domain data.

# Visualizations

((Figure showing the three stages of the CBR pipeline))

# Discussion

- CBR provides indirect benefits through high-level information and symbolic abstractions.
- Retrieving similar cases helps fill knowledge gaps for LMs.
- Case representation plays a crucial role in CBR performance.

# Conclusion

- CBR method enhances logical fallacy classification in language models.
- Effective with enriched case representations, especially counterarguments.
- Future work should explore CBR on other complex NLP tasks and further investigate case similarity.

# References

- Jin et al., 2022: Logical fallacy classification using large LMs.
- Aamodt and Plaza, 1994: Foundational work on Case-Based Reasoning.
- Vaswani et al., 2017: Attention is all you need (Transformer architectures).

# Acknowledgements

- Supported by armasuisse Science and Technology, NSF, and DARPA MCS program.

# Q&A

- Invitation for Questions
