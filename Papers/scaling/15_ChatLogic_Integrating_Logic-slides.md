# ChatLogic: Integrating Logic Programming with Large Language Models for Multi-Step Reasoning

- Authors: Zhongsheng Wang, Jiamou Liu, Qiming Bao, Hongfei Rong, Jingfeng Zhang
- School of Computer Science, University of Auckland
- Contact: zwan516@aucklanduni.ac.nz, jiamou.liu@auckland.ac.nz, {qbao775, hron635}@aucklanduni.ac.nz, jingfeng.zhang@auckland.ac.nz

# Introduction

- Problem: LLMs struggle with multi-step logic reasoning
- Motivation: Enhance LLMs' capabilities in prolonged interactions
- Scope: Introduces ChatLogic framework integrating logical reasoning

# Background/Related Work

- LLMs like ChatGPT, GPT-4 excel in generative tasks but fail in multi-step reasoning
- Token limitations in LLMs affect performance in multi-turn conversations
- Previous methods like external memory augmentation have their own challenges

# Contributions

- ChatLogic framework augments LLMs with logical reasoning engine
- Introduces 'Mix-shot Chain of Thought' technique
- Improves multi-step reasoning capabilities with minimal resource consumption

# Objective

- Enhance LLMs' inferential capabilities
- Improve translation of natural language into logical symbols using pyDatalog

# Methodology Overview

- Combines LLMs with pyDatalog for translating queries into logic programs
- Utilizes symbolic memory to boost multi-step reasoning
- Framework phases: input processing, semantic correction, syntax correction, local execution

# Datasets

- PARARULE-Plus: 400,000 samples, two scenarios (People and Animal)
- CONCEPTRULES V1 and V2: Multi-step reasoning datasets with depths up to 3

# Model Details

- Uses Mix-shot CoT approach combining zero-shot and one-shot learning
- Enhances LLMs' ability to handle multi-step reasoning tasks
- Framework includes semantic and syntax correction modules

# Experiments

- Experimental setup: Comparison with baseline LLMs
- Evaluation metrics: Accuracy and code executability
- Models tested: ChatGPT, GPT-4, Llama 2-7B

# Results

- ChatLogic+LLMs significantly outperform baseline LLMs in accuracy
- Notable improvement in multi-step reasoning capabilities
- Enhanced code executability with semantic and syntax corrections

# Performance Comparisons with Prior Work

- ChatLogic shows higher accuracy compared to baseline and Zero-shot CoT
- Demonstrates versatility in enhancing different LLMs
- Superior performance in complex reasoning tasks

# Ablation Studies

- Separate assessment of semantic and syntax correction modules
- Gradual increase in code execution success rates
- Validation of the effectiveness of the approach

# Visualizations

- ((Figure showing comparison of inference processes between ChatGPT and ChatLogic))
- ((Table showing accuracy comparison on datasets))

# Discussion

- Key insights: Logical symbolic operations significantly enhance reasoning
- Limitations: Current datasets may not represent real-world complexities fully
- Future work: Develop adaptable optimization components for broader scenarios

# Conclusion

- Summary: ChatLogic framework effectively enhances LLMs' multi-step reasoning
- Impact: Improves accuracy and reliability in reasoning tasks
- Future directions: Expand optimization modules and tackle real-world complexities

# References

- Brown et al. 2020. "Language models are few-shot learners"
- Wang et al. 2022. "Self-consistency improves chain of thought reasoning in language models"
- OpenAI. 2023. "GPT-4 Technical Report"
- Zhong, Lei, and Chen. 2022. "Training language models with memory augmentation"

# Acknowledgements

- Thanks to HouGarden Company for financial support

# Q&A

- Invitation for questions
