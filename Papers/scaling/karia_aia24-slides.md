# Title Slide
- Can LLMs translate SATisfactorily? Assessing LLMs in Generating and Interpreting Formal Specifications
- Rushang Karia, Daksh Dobhal, Daniel Bramblett, Pulkit Verma, and Siddharth Srivastava
- School of Computing and Augmented Intelligence
- Arizona State University
- Date of Presentation: AAAI 2024 Spring Symposium

# Introduction
- Problem: Converting natural language descriptions to formal specifications
- Motivation: Reduce design costs by automating translation tasks
- Scope: Assess LLMs' capabilities in translating natural language to formal specifications and vice versa

# Background/Related Work
- Previous work: LLMs generating formal syntax such as source code
- Limitations: Often handcrafted and part of LLMs' training sets
- Key Concepts: Boolean satisfiability (SAT), formal specifications

# Contributions
- Generation of SAT-based datasets inspired by real-world system specifications
- Handsfree approach using two LLM copies and SAT solvers for bidirectional assessment
- Empirical evaluation showing limitations of current SOTA LLMs in translating formal specifications

# Objective
- Assess LLMs' translation capabilities between natural language and formal specifications
- Evaluate accuracy without human intervention using SAT solvers

# Methodology Overview
- Approach: Use two copies of an LLM and off-the-shelf SAT solvers
- Data: Generated SAT formulae datasets
- Model: LLMs perform translation tasks, assessed via SAT solvers

# Datasets
- Description: High-quality datasets generated using SAT formulae generators
- Structure: Formulae categorized by complexity (k, m)-CNF
- Generation: Randomly generated formulae using Gsat grammar

# Model Details
- Model Architecture: Utilizes LLMs for both encoding (NL to SAT) and decoding (SAT to NL)
- Training Procedure: Iterative prompt modification to achieve high accuracy

# Experiments
- Setup: Used GPT-4, GPT-3.5-turbo, Mistral-7B-Instruct, and Gemini Pro
- Evaluation Metrics: Accuracy of NL↔SAT translations on generated datasets

# Results
- Performance Metrics: Accuracy of various LLMs on NL↔SAT tasks
- Key Findings: Performance degrades as formula complexity increases
- ((Figure showing accuracies of various SOTA LLMs on NL↔SAT tasks))

# Discussion
- Key Insights: Current SOTA LLMs struggle with increasing formula complexity
- Interpretation: Errors in translation tasks such as handling parentheses and negating propositions
- Limitations: Performance drop even in structured CNF formulae

# Conclusion
- Summary: Effective assessment methodology for LLMs' translation capabilities
- Impact: Shows current limitations in deploying LLMs for formal syntax translation
- Future Work: Investigate approaches to improve translation performance

# References
- Biere et al. 2021. Handbook of Satisfiability
- de Moura, L. M.; and Bjørner, N. S. 2008. Z3: An Efficient SMT Solver
- Fan et al. 2023. NPHardEval: Dynamic Benchmark on Reasoning Ability of Large Language Models
- Shi et al. 2022. Natural Language to Code Translation with Execution
- Xue et al. 2021. mT5: A Massively Multilingual Pre-trained Text-to-Text Transformer

# Acknowledgements
- Supported by Arizona State University's GPSA Jumpstart Research Grant

# Q&A
- Invitation for Questions
