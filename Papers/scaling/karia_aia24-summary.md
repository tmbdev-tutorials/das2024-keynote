# ONE SENTENCE SUMMARY:
The study evaluates Large Language Models (LLMs) for translating between natural language and formal specifications, revealing current limitations in their accuracy and practical utility.

# MAIN POINTS:
1. Stakeholders describe system requirements in natural language, converted to formal syntax by domain experts.
2. LLMs' translation capabilities between natural language and formal specifications are assessed.
3. Current evaluations often use hand-crafted problems likely included in LLMs' training sets.
4. A new approach uses two LLM copies with a SAT solver for automatic translation assessment.
5. Boolean satisfiability (SAT) formulae are used to generate datasets for evaluation.
6. The study empirically measures translation accuracy of LLMs in both directions.
7. SOTA LLMs currently fail to solve simple formal specifications adequately.
8. Evaluation includes GPT-4, GPT-3.5-turbo, Mistral-7B-Instruct, and Gemini Pro.
9. SAT→NL errors often involve incorrect parenthesis order; NL→SAT errors include hallucinations.
10. Results show significant performance degradation with increasing formula size.

# TAKEAWAYS:
1. LLMs struggle with accurate translation between natural language and formal specifications.
2. Current methods for evaluating LLMs' translation capabilities are insufficient.
3. The new approach provides a scalable, handsfree assessment method.
4. LLMs need significant improvement before being useful in complex system design tasks.
5. Future work will focus on enhancing LLM performance in formal translation tasks.
