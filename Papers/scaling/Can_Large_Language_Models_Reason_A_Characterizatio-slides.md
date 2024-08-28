# Can Large Language Models Reason? A Characterization via 3-SAT

- Rishi Hazra, Gabriele Venturato, Pedro Zuidberg Dos Martires, Luc De Raedt
- Centre for Applied Autonomous Sensor Systems (AASS), Örebro University, Örebro, Sweden
- Department of Computer Science, KU Leuven, Leuven, Belgium
- Date of Presentation: [Insert Date]

# Introduction
- Investigates whether Large Language Models (LLMs) can reason
- Focus on logical reasoning using 3-SAT, an NP-complete problem
- Empirical analysis of LLMs' reasoning capabilities via phase transitions in 3-SAT

# Background/Related Work
- LLMs are claimed to have advanced reasoning abilities
- Commonsense vs. logical reasoning
- Current benchmarks may be contaminated by training data
- LLMs often exploit statistical features or shortcuts, bypassing true reasoning

# Contributions
- Provide a computational theory perspective on LLM reasoning
- Use 3-SAT phase transitions to characterize reasoning abilities
- Show that LLMs struggle with true reasoning required for 3-SAT problems

# Objective
- Determine if LLMs can solve 3-SAT problems
- Assess reasoning capabilities via empirical analysis on 3-SAT phase transitions

# Methodology Overview
- Study LLMs' performance on 3-SAT problems
- Examine phase transitions in 3-SAT to classify problem difficulty
- Use GPT-4 Turbo as the reference LLM

# Datasets
- Generated 3-SAT formulas with varying α (m/n)
- Three regions: under-constrained (easy), constrained (hard), over-constrained (easy)
- Dataset includes both satisfiable and unsatisfiable instances

# Model Details
- SAT-Menu: Reframe 3-SAT as a natural language menu-selection problem
- SAT-CNF: Provide 3-SAT formulas in Conjunctive Normal Form
- Two problem variants: SAT Decision and SAT Search

# Experiments
- Evaluate GPT-4's accuracy on SAT Search and SAT Decision
- Analyze performance across different α values and satisfiability ratios

# Results
- GPT-4 performs well in easy regions but poorly in the hard region
- SAT Search is more challenging than SAT Decision
- Performance correlates with the satisfiability ratio

# Performance Comparisons with Prior Work
- Compare GPT-4 with other state-of-the-art LLMs
- GPT-4 outperforms others in detecting unsatisfiable problems
- LLM-Modulo frameworks enhance performance

# Discussion
- LLMs exploit statistical features in easy regions, struggle with true reasoning in hard regions
- GPT-4's performance suggests limitations in current LLM architectures for reasoning tasks
- Need for better heuristics and integration with symbolic solvers

# Conclusion
- LLMs show apparent reasoning abilities but struggle with true reasoning tasks
- Integration with symbolic solvers can enhance performance
- Future work: Improve LLM architectures and explore more complex reasoning tasks

# References
- Davis, E., Marcus, G.: Commonsense reasoning and commonsense knowledge in artificial intelligence. Commun. ACM 58(9), 92–103 (2015)
- Russell, S., Norvig, P.: Artificial Intelligence: A Modern Approach, 3rd edn. (2010)
- Wei, J., Tay, Y., Bommasani, R., et al.: Emergent abilities of large language models. Transactions on Machine Learning Research (2022)
- Schaeffer, R., Miranda, B., Koyejo, S.: Are emergent abilities of large language models a mirage? In: Thirty-seventh Conference on Neural Information Processing Systems (2023)
- Bender, E.M., Gebru, T., McMillan-Major, A., Shmitchell, S.: On the dangers of stochastic parrots: Can language models be too big? In: Proceedings of the 2021 ACM Conference on Fairness, Accountability, and Transparency, pp. 610–623 (2021)

# Acknowledgements
- Thanks to Holger Hoos, Heikki Mannila, Paolo Frasconi, Pascal Van Hentenryck, Ross King, Giuseppe Marra, Pieter Delobelle, and Hendrik Blockeel for valuable feedback
- Supported by Wallenberg AI Autonomous Systems and Software Program (WASP), EU H2020 ICT48 project “TAILOR”, and KU Leuven Research Fund

# Q&A
- Invitation for Questions
