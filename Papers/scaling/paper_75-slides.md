# Beyond Chinchilla-Optimal: Accounting for Inference in Language Model Scaling Laws
- Nikhil Sardana, Jonathan Frankle
- MosaicML
- Presented at NeurIPS 2023

# Introduction
- Large language models (LLMs) have significant training and inference costs
- Training costs driven by model size and data volume
- Inference costs influenced by model size and query volume
- Goal: Minimize the total cost for producing and serving high-quality models

# Background/Related Work
- Previous scaling laws estimate model quality based on parameters and training data
- Hoffmann et al.'s Chinchilla scaling laws are influential
- Chinchilla scaling laws focus on training costs, neglect inference costs
- LLaMA models trained on more data than Chinchilla's optimal to reduce inference costs

# Contributions
- Modify Chinchilla scaling laws to include inference costs
- Calculate optimal parameter count and data size for given model quality and inference demand
- Analysis in terms of compute budget and real-world costs
- Findings suggest smaller and longer-trained models for high inference demand

# Objective
- Minimize computational costs for a given model quality and inference demand
- Use pre-training cross-entropy loss as a quality proxy
- Use floating-point operations (FLOPs) as a computational cost unit

# Methodology Overview
- Follow methodology from Hoffmann et al. (Chinchilla paper)
- Model pre-training loss in terms of parameters and training tokens
- Use constants from the Chinchilla paper for analysis

# Computational Optimality
- Objective: Minimize sum of training and inference FLOPs
- Use standard approximation: 6N FLOPs per training token, 2N per inference token
- Analytical solution for optimization problem without inference
- Computational solution using Newton root-finding method with inference

# Results
- Smaller models trained on more data are optimal for high inference demand
- Example: For 1.5B inference requests, a 16B parameter model on 3.35T tokens is cost-optimal
- Figure 1: Ratios of total FLOPs, model parameters, and pre-training tokens for optimal vs. Chinchilla models ((Figure showing ratios))

# Estimating Real-World Cost Optimality
- Real-world costs differ between training and inference
- Inference hardware utilization lower than training
- Quantization can reduce inference costs
- New objective includes hardware utilization and cost per FLOP

# Results (Real-World Cost Optimality)
- Real-world cost analysis shows further cost reductions
- Example: For 1.5B inference requests, training a 16B model on 3.35T tokens saves 17%
- Figure 2: Ratios of total cost, model parameters, and pre-training tokens for cost-optimal vs. Chinchilla models ((Figure showing ratios))

# Discussion
- Modified scaling laws account for inference costs
- Smaller, longer-trained models are more cost-effective for high inference demand
- Assumptions about scaling laws need further experimental validation

# Conclusion
- Modified Chinchilla scaling laws for computational and real-world costs
- Optimal model configurations depend on inference demand
- Further work needed to validate formulas and scaling laws in extreme ranges

# References
- Hoffmann et al. (2022). Training compute-optimal large language models.
- Touvron et al. (2023). Llama: Open and efficient foundation language models.
- De Vries (2023). Go smol or go home.

# Acknowledgements
- Thanks to Sasha Doubov, Daya Khudia, Mihir Patel, and Linden Li for their feedback and discussions.

# Q&A
- Invitation for questions
