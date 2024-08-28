# Impact of Co-occurrence on Factual Knowledge of Large Language Models

- Cheongwoong Kang, Jaesik Choi
- KAIST
- Date of Presentation

# Introduction

- Large language models (LLMs) often make factually incorrect responses
- Hypothesis: Reliance on simple co-occurrence statistics causes factual errors
- Scope: Analyze co-occurrence bias and suggest mitigation strategies

# Background/Related Work

- LLMs like GPT-3 have shown high capacity to recall factual knowledge
- Previous studies indicate LLMs often rely on spurious features
- This work focuses on the impact of co-occurrence statistics on factual knowledge

# Contributions

- Demonstrate LLMs' vulnerability to co-occurrence bias
- Show that scaling model sizes or finetuning does not resolve this bias
- Suggest finetuning on a debiased dataset as a mitigation strategy

# Objective

- Investigate the effects of co-occurrence statistics on factual knowledge in LLMs
- Hypothesize that co-occurrence bias leads to incorrect factual responses

# Methodology Overview

- Use the LAMA dataset to probe factual knowledge
- Analyze correlation between co-occurrence statistics and performance
- Count co-occurrences of word pairs in pre-training corpora

# Datasets

- LAMA-TREx dataset with 41 relations
- Facts represented as subject-relation-object triples
- Data split into training and test sets with a ratio of 70:30

# Model Details

- Tested models: GPT-Neo (125M, 1.3B, 2.7B) and GPT-J (6B)
- Pre-trained on the Pile dataset, which consists of 800GB of text
- Finetuning with a learning rate of 2e-5 and batch size of 128

# Experiments

- Evaluate factual knowledge probing using hits@1 and MRR metrics
- Test the impact of co-occurrence statistics on factual knowledge
- Correlate factual knowledge accuracy with co-occurrence counts

# Results

- Factual knowledge accuracy correlates with subject-object co-occurrence
- LLMs struggle to recall rare facts despite scaling up model sizes or finetuning
- Debiased finetuning helps recall rare facts but is not effective for unseen facts

# Performance Comparisons with Prior Work

- Previous work shows correlation between term frequency and model behavior
- This study provides in-depth analysis of co-occurrence bias in LLMs
- Scaling up model sizes does not resolve co-occurrence bias

# Ablation Studies

- Finetuned models can memorize seen facts except for the smallest model
- Co-occurrence statistics significantly influence factual knowledge recall
- Debiased finetuning improves recall of rare facts but not generalizable

# Visualizations

- ((Figure showing the framework of correlation analysis))
- ((Figures showing hits@1 and MRR results for different model sizes and settings))

# Discussion

- Co-occurrence bias remains despite improvements in model sizes and finetuning
- Heavy reliance on co-occurrence may lead to hallucinations and biased responses
- Further research needed to develop sophisticated debiasing algorithms

# Conclusion

- Factual knowledge accuracy correlates with subject-object co-occurrence
- Scaling model sizes or finetuning does not resolve co-occurrence bias
- Future work should focus on mitigating co-occurrence bias for reliable LLMs

# References

- Devlin et al., 2019; Brown et al., 2020; Raffel et al., 2020
- Petroni et al., 2019; Jiang et al., 2020; Roberts et al., 2020
- Elazar et al., 2021; Cao et al., 2021

# Acknowledgements

- Supported by Institute of Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korea government(MSIT)

# Q&A

- Invitation for Questions
