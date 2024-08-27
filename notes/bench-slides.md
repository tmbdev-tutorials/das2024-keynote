# LLM Benchmarks (BigBench)

| Task Type                              | Task Type                                |
| -------------------------------------- | ---------------------------------------- |
| Standard NLP Tasks                     | Problem Solving and Planning             |
| Commonsense Reasoning                  | Language Games                           |
| Creativity and Innovation              | Meta-Learning and Adaptation             |
| Mathematics and Logic                  | Miscellaneous Tasks                      |
| Social and Ethical Understanding       |                                          |
| World Knowledge                        |                                          |

# More LLM Benchmarks

| Task Type                             | Task Type                                |
| ------------------------------------- | ---------------------------------------- |
| Fine-Grained Linguistic Acceptability | Code Generation and Evaluation           |
| Natural Language Inference (NLI)      | Pronoun Disambiguation                   |
| Truthfulness and Misinformation       | Word Sense Disambiguation                |
| Specialized Academic Knowledge        | Reading Comprehension with External Knowledge |

# Task Types in LLM Benchmarks

| Task Type                               | Task Type                                 |
| --------------------------------------- | ----------------------------------------- |
| Multiple-Choice Question Answering      | Text Completion                           |
| Fill-in-the-Blank (Cloze)               | Sequence Labeling (e.g., Named Entity Recognition) |
| Free Text Generation                    | Dialog and Conversational Response Generation |
| Classification (Binary or Multi-Class)  | Logical and Symbolic Reasoning            |
| Ranking and Pairwise Comparison         |                                           |
| Regression (Numerical Prediction)       |                                           |

# MMLU (Massive Multitask Language Understanding) Overview

- **Task Type**: Multiple-Choice Question Answering
- **Subjects Covered**: 57 subjects across STEM, humanities, social sciences, medicine, business, and more
- **Question Development**: Manually curated from high school, college, and professional-level exams and textbooks
- **Focus**: Mixed evaluation of both factual knowledge and reasoning abilities
- **Scale**: 14,000 questions designed to test domain-specific understanding and application

# MMLU Benchmark Results Overview

### Commercial Models
- **GPT-4**: Top score of **86.4%** (5-shot setting)
- **Claude 3 Opus**: Scored **84.6%**, strong performance
- **Gemini 1.0 Pro**: Scored **70.0%**

### Open-Source Models
- **LLaMA 2 (70B)**: Scored **69.5%**
- **Mixtral (8x22B)**: Scored **77.8%**
- **Falcon-40B-Instruct**: Leading open model with **54.1%**

# Important Hugging Face Leaderboards for LLMs

- **Open LLM Leaderboard**: Ranks and evaluates open-source LLMs and chatbots across multiple benchmarks.
- **MMLU-Pro**: An advanced version of MMLU with more difficult multiple-choice questions, testing sophisticated reasoning.
- **BIG-Bench Hard**: A challenging set of tasks from BIG-bench, focusing on complex problem-solving like boolean logic and sarcasm detection.
- **GPQA**: PhD-level questions in biology, physics, and chemistry, testing expert-level knowledge.
- **MuSR**: Multi-step reasoning tasks, requiring models to solve complex problems like murder mysteries and logical puzzles.
- **MATH lvl 5**: Focuses on the hardest level of multi-step math problems, testing advanced mathematical reasoning.
- **IFEval**: Evaluates models based on their ability to follow detailed instructions, such as producing specific text formats.

# Slide 1: Leaderboards for Document AI (Part 1)

## 1. OCR (Text Localization)
- **Leaderboard**: [SROIE Task 1](https://rrc.cvc.uab.es/?ch=13&com=evaluation&task=1)
- **Dataset**: SROIE (Scanned Receipt OCR).
- **Task**: Text detection, recognition, and key information extraction from scanned receipts.

## 2. Document Layout Analysis
- **Leaderboard**: [PubLayNet](https://github.com/ibm-aur-nlp/PubLayNet)
- **Dataset**: PubLayNet (1M+ document pages).
- **Task**: Segment document components (e.g., text, tables) using object detection.

## 3. Table Detection and Extraction
- **Leaderboard**: [Table Transformer on PubTables-1M](https://arxiv.org/abs/2110.00061)
- **Dataset**: PubTables-1M (1M annotated tables).
- **Task**: Detect table boundaries and recognize table structure (rows, columns, cells).

# Slide 2: Leaderboards for Document AI (Part 2)

## 4. Visual Question Answering (VQA)
- **Leaderboard**: [VQA v2.0](https://visualqa.org/roe_2022.html)
- **Dataset**: VQA v2.0 (250K+ image-question pairs).
- **Task**: Answer questions about images; evaluated by accuracy.

## 5. Document Visual Question Answering (DocVQA)
- **Leaderboard**: [DocVQA](https://rrc.cvc.uab.es/?ch=17&com=evaluation&task=1)
- **Dataset**: DocVQA (document images with questions).
- **Task**: Answer questions based on document images; evaluated by ANLS.

## 6. Key Information Extraction (KIE)
- **Leaderboard**: [KIE on SROIE](https://rrc.cvc.uab.es/?ch=13&com=evaluation&task=3)
- **Dataset**: SROIE (scanned documents).
- **Task**: Extract key-value pairs from receipts; evaluated by F1-score.

# Knowledge Base and RDF-Related Leaderboards

- **QALD**: Convert natural language to SPARQL for RDF datasets (20+ models).
- **OGBL-Wikikg2**: Link prediction on Wikidata subset (10-15 models).
- **KGC (FB15k-237)**: Predict missing triples in Freebase (20-30 models).
- **LC-QuAD**: Translate complex questions to SPARQL over DBpedia (15-20 models).
- **WN18RR**: Predict missing triples in WordNet knowledge graph (20-25 models).
- **SimpleQuestions**: Answer single-hop factoid questions over Freebase (various models).
- **WebQuestions**: Answer natural language questions over Freebase, multi-hop reasoning (various models).
- **ComplexWebQuestions**: Answer complex multi-hop questions over Freebase (various models).

