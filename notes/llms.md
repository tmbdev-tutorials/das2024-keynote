
## Scaling Instruction-Finetuned Language Models

Hyung Won Chung, Le Hou, Shayne Longpre, Barret Zoph, Yi Tay, William Fedus, Yunxuan Li, Xuezhi Wang, Mostafa Dehghani, Siddhartha Brahma, Albert Webson, Shixiang Shane Gu, Zhuyun Dai, Mirac Suzgun, Xinyun Chen, Aakanksha Chowdhery, Alex Castro-Ros, Marie Pellat, Kevin Robinson, Dasha Valter, Sharan Narang, Gaurav Mishra, Adams Yu, Vincent Zhao, Yanping Huang, Andrew Dai, Hongkun Yu, Slav Petrov, Ed H. Chi, Jeff Dean, Jacob Devlin, Adam Roberts, Denny Zhou, Quoc V. Le, Jason Wei

Category: llms
Keywords: instruction finetuning, language models, task scaling, model scaling, chain-of-thought, zero-shot, few-shot, evaluation benchmarks, Flan-PaLM, Flan-T5
Year: 2023

Finetuning language models on a collection of datasets phrased as instructions
has been shown to improve model performance and generalization to unseen tasks.
In this paper we explore instruction finetuning with a particular focus on (1)
scaling the number of tasks, (2) scaling the model size, and (3) finetuning on
chain-of-thought data. We find that instruction finetuning with the above
aspects dramatically improves performance on a variety of model classes (PaLM,
T5, U-PaLM), prompting setups (zero-shot, few-shot, CoT), and evaluation
benchmarks (MMLU, BBH, TyDiQA, MGSM, open-ended generation,
RealToxicityPrompts). For instance, Flan-PaLM 540B instruction-finetuned on 1.8K
tasks outperforms PaLM 540B by a large margin (+9.4% on average). Flan-PaLM 540B
achieves state-of-the-art performance on several benchmarks, such as 75.2% on
five-shot MMLU. We also publicly release Flan-T5 checkpoints, which achieve
strong few-shot performance even compared to much larger models, such as PaLM
62B. Overall, instruction finetuning is a general method for improving the
performance and usability of pretrained language models.
---

## Harnessing the Power of LLMs in Practice: A Survey on ChatGPT and Beyond

Jingfeng Yang, Hongye Jin, Ruixiang Tang, Xiaotian Han, Qizhang Feng, Haoming Jiang, Bing Yin, Xia Hu

Category: llms
Keywords: Large Language Models, Neural Language Processing, Practical Guide, ChatGPT
Year: 2023

This paper presents a comprehensive and practical guide for practitioners and
end-users working with Large Language Models (LLMs) in their downstream natural
language processing (NLP) tasks. We provide discussions and insights into the
usage of LLMs from the perspectives of models, data, and downstream tasks.
Firstly, we offer an introduction and brief summary of current GPT- and BERT-
style LLMs. Then, we discuss the influence of pre-training data, training data,
and test data. Most importantly, we provide a detailed discussion about the use
and non-use cases of large language models for various natural language
processing tasks, such as knowledge-intensive tasks, traditional natural
language understanding tasks, natural language generation tasks, emergent
abilities, and considerations for specific tasks. We present various use cases
and non-use cases to illustrate the practical applications and limitations of
LLMs in real-world scenarios. We also try to understand the importance of data
and the specific challenges associated with each NLP task. Furthermore, we
explore the impact of spurious biases on LLMs and delve into other essential
considerations, such as efficiency, cost, and latency, to ensure a comprehensive
understanding of deploying LLMs in practice. This comprehensive guide aims to
provide researchers and practitioners with valuable insights and best practices
for working with LLMs, thereby enabling the successful implementation of these
models in a wide range of NLP tasks. A curated list of practical guide resources
of LLMs, regularly updated, can be found at
https://github.com/Mooler0410/LLMsPracticalGuide.
---

## LLMMaps - A Visual Metaphor for Stratified Evaluation of Large Language Models

Patrik Puchert, Poonam Poonam, Christian van Onzenoodt, Timo Ropinski

Category: llms
Keywords: Large language models, Explainable artificial intelligence, Stratified evaluation, Visualization, Knowledge capabilities
Year: 2023

Large Language Models (LLMs) have revolutionized natural language processing and
demonstrated impressive capabilities in various tasks. Unfortunately, they are
prone to hallucinations, where the model exposes incorrect or false information
in its responses, which renders diligent evaluation approaches mandatory. While
LLM performance in specific knowledge fields is often evaluated based on
question and answer (Q&A) datasets, such evaluations usually report only a
single accuracy number for the entire field, a procedure which is problematic
with respect to transparency and model improvement. A stratified evaluation
could instead reveal subfields, where hallucinations are more likely to occur
and thus help to better assess LLMs’ risks and guide their further development.
To support such stratified evaluations, we propose LLMMaps as a novel
visualization technique that enables users to evaluate LLMs’ performance with
respect to Q&A datasets. LLMMaps provide detailed insights into LLMs’ knowledge
capabilities in different subfields, by transforming Q&A datasets as well as LLM
responses into our internal knowledge structure. An extension for comparative
visualization furthermore, allows for the detailed comparison of multiple LLMs.
To assess LLMMaps we use them to conduct a comparative analysis of several
state-of-the-art LLMs, such as BLOOM, GPT-2, GPT-3, ChatGPT and LLaMa-13B, as
well as two qualitative user evaluations. All necessary source code and data for
generating LLMMaps to be used in scientific publications and elsewhere will be
available on GitHub: https://github.com/******
---

## StructGPT: A General Framework for Large Language Model to Reason over Structured Data

Jinhao Jiang, Kun Zhou, Zican Dong, Keming Ye, Wayne Xin Zhao, Ji-Rong Wen

Category: llms
Keywords: large language models, structured data, zero-shot reasoning, question answering, tool augmentation
Year: 2023

In this paper, we study how to improve the zero-shot reasoning ability of large
language models (LLMs) over structured data in a unified way. Inspired by the
study on tool augmentation for LLMs, we develop an Iterative Reading-then-
Reasoning (IRR) approach for solving question answering tasks based on
structured data, called StructGPT. In our approach, we construct the specialized
function to collect relevant evidence from structured data (i.e., reading), and
let LLMs concentrate on the reasoning task based on the collected information
(i.e., reasoning). Specially, we propose an invoking-linearization-generation
procedure to support LLMs in reasoning on the structured data with the help of
the external interfaces. By iterating this procedure with provided interfaces,
our approach can gradually approach the target answer to a given query.
Extensive experiments conducted on three types of structured data demonstrate
the effectiveness of our approach, which can significantly boost the performance
of ChatGPT and achieve comparable performance against the full-data supervised-
tuning baselines.
---

## ReWOO: Decoupling Reasoning from Observations for Efficient Augmented Language Models

Binfeng Xu, Zhiyuan Peng, Bowen Lei, Subhabrata Mukherjee, Yuchen Liu, Dongkuan Xu

Category: llms
Keywords: Augmented Language Models, Large Language Models, Reasoning, Efficiency, Token Consumption
Year: 2023

Augmented Language Models (ALMs) blend the reasoning capabilities of Large
Language Models (LLMs) with tools that allow for knowledge retrieval and action
execution. Existing ALM systems trigger LLM thought processes while pulling
observations from these tools in an interleaved fashion. Specifically, an LLM
reasons to call an external tool, gets halted to fetch the tool's response, and
then decides the next action based on all preceding response tokens. Such a
paradigm, though straightforward and easy to implement, often leads to huge
computation complexity from redundant prompts and repeated execution. This study
addresses such challenges for the first time, proposing a modular paradigm ReWOO
(Reasoning WithOut Observation) that detaches the reasoning process from
external observations, thus significantly reducing token consumption.
Comprehensive evaluations across six public NLP benchmarks and a curated dataset
reveal consistent performance enhancements with our proposed methodology.
Notably, ReWOO achieves 5x token efficiency and 4% accuracy improvement on
HotpotQA, a multi-step reasoning benchmark. Furthermore, ReWOO demonstrates
robustness under tool-failure scenarios. Beyond prompt efficiency, decoupling
parametric modules from non-parametric tool calls enables instruction fine-
tuning to offload LLMs into smaller language models, thus substantially reducing
model parameters. Our illustrative work offloads reasoning ability from 175B
GPT3.5 into 7B LLaMA, demonstrating the significant potential for truly
efficient and scalable ALM systems. Full code, model, and data are released for
reproduction.
---

## LLaMA: Open and Efficient Foundation Language Models

Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothee Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave, Guillaume Lample

Category: llms
Keywords: language models, LLaMA, public datasets, inference budget, state-of-the-art
Year: 2023

We introduce LLaMA, a collection of foundation language models ranging from 7B
to 65B parameters. We train our models on trillions of tokens, and show that it
is possible to train state-of-the-art models using publicly available datasets
exclusively, without resorting to proprietary and inaccessible datasets. In
particular, LLaMA-13B outperforms GPT-3 (175B) on most benchmarks, and LLaMA-65B
is competitive with the best models, Chinchilla-70B and PaLM-540B. We release
all our models to the research community.
---

## PaLM 2 Technical Report

Google*

Category: llms
Keywords: PaLM 2, language model, multilingual, reasoning, Transformer, compute-efficient, AI
Year: 2023

We introduce PaLM 2, a new state-of-the-art language model that has better
multilingual and reasoning capabilities and is more compute-efficient than its
predecessor PaLM (Chowdhery et al., 2022). PaLM 2 is a Transformer-based model
trained using a mixture of objectives similar to UL2 (Tay et al., 2023). Through
extensive evaluations on English and multilingual language, and reasoning tasks,
we demonstrate that PaLM 2 has significantly improved quality on downstream
tasks across different model sizes, while simultaneously exhibiting faster and
more efficient inference compared to PaLM. This improved efficiency enables
broader deployment while also allowing the model to respond faster, for a more
natural pace of interaction. PaLM 2 demonstrates robust reasoning capabilities
exemplified by large improvements over PaLM on BIG-Bench and other reasoning
tasks. PaLM 2 exhibits stable performance on a suite of responsible AI
evaluations, and enables inference-time control over toxicity without additional
overhead or impact on other capabilities. Overall, PaLM 2 achieves state-of-the-
art performance across a diverse set of tasks and capabilities.
---

## GPT-4 Technical Report

OpenAI

Category: llms
Keywords: GPT-4, multimodal model, Transformer, language model, deep learning
Year: 2023

We report the development of GPT-4, a large-scale, multimodal model which can
accept image and text inputs and produce text outputs. While less capable than
humans in many real-world scenarios, GPT-4 exhibits human-level performance on
various professional and academic benchmarks, including passing a simulated bar
exam with a score around the top 10% of test takers. GPT-4 is a Transformer-
based model pre-trained to predict the next token in a document. The post-
training alignment process results in improved performance on measures of
factuality and adherence to desired behavior. A core component of this project
was developing infrastructure and optimization methods that behave predictably
across a wide range of scales. This allowed us to accurately predict some
aspects of GPT-4’s performance based on models trained with no more than
1/1,000th the compute of GPT-4.
---

## Generate Rather Than Retrieve: Large Language Models are Strong Context Generators

Wenhao Yu, Dan Iter, Shuohang Wang, Yichong Xu, Mingxuan Ju, Soumya Sanyal, Chenguang Zhu, Michael Zeng, Meng Jiang

Category: llms
Keywords: large language models, open-domain question answering, knowledge-intensive tasks, context generation, document generation
Year: 2023

Knowledge-intensive tasks, such as open-domain question answering (QA), require
access to a large amount of world or domain knowledge. A common approach for
knowledge-intensive tasks is to employ a retrieve-then-read pipeline that first
retrieves a handful of relevant contextual documents from an external corpus
such as Wikipedia and then predicts an answer conditioned on the retrieved
documents. In this paper, we present a novel perspective for solving knowledge-
intensive tasks by replacing document retrievers with large language model
generators. We call our method generate-then-read (GENREAD), which first prompts
a large language model to generate contextual documents based on a given
question, and then reads the generated documents to produce the final answer.
Furthermore, we propose a novel clustering-based prompting method that selects
distinct prompts, in order to generate diverse documents that cover different
perspectives, leading to better recall over acceptable answers. We conduct
extensive experiments on three different knowledge-intensive tasks, including
open-domain QA, fact checking, and dialogue system. Notably, GENREAD achieves
71.6 and 54.4 exact match scores on TriviaQA and WebQ, significantly
outperforming the state-of-the-art retrieve-then-read pipeline DPR-FiD by +4.0
and +3.9, without retrieving any documents from any external knowledge source.
Lastly, we demonstrate the model performance can be further improved by
combining retrieval and generation. Our code and generated documents can be
found at https://github.com/wyu97/GenRead.
---

## LLaMA: Open and Efficient Foundation Language Models

Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothee Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave, Guillaume Lample

Category: llms
Keywords: foundation language models, LLaMA, open-source, public datasets, efficient inference
Year: 2023

We introduce LLaMA, a collection of foundation language models ranging from 7B
to 65B parameters. We train our models on trillions of tokens, and show that it
is possible to train state-of-the-art models using publicly available datasets
exclusively, without resorting to proprietary and inaccessible datasets. In
particular, LLaMA-13B outperforms GPT-3 (175B) on most benchmarks, and LLaMA-65B
is competitive with the best models, Chinchilla-70B and PaLM-540B. We release
all our models to the research community.
---

## GPT-4 Technical Report

OpenAI

Category: llms
Keywords: GPT-4, large-scale model, multimodal, Transformer, natural language processing, machine learning
Year: 2023

We report the development of GPT-4, a large-scale, multimodal model which can
accept image and text inputs and produce text outputs. While less capable than
humans in many real-world scenarios, GPT-4 exhibits human-level performance on
various professional and academic benchmarks, including passing a simulated bar
exam with a score around the top 10% of test takers. GPT-4 is a Transformer-
based model pre-trained to predict the next token in a document. The post-
training alignment process results in improved performance on measures of
factuality and adherence to desired behavior. A core component of this project
was developing infrastructure and optimization methods that behave predictably
across a wide range of scales. This allowed us to accurately predict some
aspects of GPT-4’s performance based on models trained with no more than
1/1,000th the compute of GPT-4.
---

## Reﬂexion: an autonomous agent with dynamic memory and self-reﬂection

Noah Shinn, Beck Labash, Ashwin Gopinath

Category: llms
Keywords: autonomous agent, dynamic memory, self-reflection, large language models, decision-making
Year: 2023

Recent advancements in decision-making large language model (LLM) agents have
demonstrated impressive performance across various benchmarks. However, these
state-of-the-art approaches typically necessitate internal model fine-tuning,
external model fine-tuning, or policy optimization over a defined state space.
Implementing these methods can prove challenging due to the scarcity of high-
quality training data or the lack of well-defined state space. Moreover, these
agents do not possess certain qualities inherent to human decision-making
processes, specifically the ability to learn from mistakes. Self-reflection
allows humans to efficiently solve novel problems through a process of trial and
error. Building on recent research, we propose Reﬂexion, an approach that endows
an agent with dynamic memory and self-reflection capabilities to enhance its
existing reasoning trace and task-specific action choice abilities. To achieve
full automation, we introduce a straightforward yet effective heuristic that
enables the agent to pinpoint hallucination instances, avoid repetition in
action sequences, and, in some environments, construct an internal memory map of
the given environment. To assess our approach, we evaluate the agent’s ability
to complete decision-making tasks in AlfWorld environments and knowledge-
intensive, search-based question-and-answer tasks in HotPotQA environments. We
observe success rates of 97% and 51%, respectively, and provide a discussion on
the emergent property of self-reflection.
---

## Large Language Models Can Self-Improve

Jiaxin Huang, Shixiang Shane Gu, Le Hou, Yuexin Wu, Xuezhi Wang, Hongkun Yu, Jiawei Han

Category: llms
Keywords: Large Language Models, Self-Improvement, Chain-of-Thought Prompting, Self-Consistency, Fine-Tuning, Reasoning
Year: 2023

Large Language Models (LLMs) have achieved excellent performances in various
tasks. However, fine-tuning an LLM requires extensive supervision. Humans, on
the other hand, may improve their reasoning abilities by self-thinking without
external inputs. In this work, we demonstrate that an LLM is also capable of
self-improving with only unlabeled datasets. We use a pre-trained LLM to
generate 'high-confidence' rationale-augmented answers for unlabeled questions
using Chain-of-Thought prompting and self-consistency, and fine-tune the LLM
using those self-generated solutions as target outputs. We show that our
approach improves the general reasoning ability of a 540B-parameter LLM
(74.4%→82.1% on GSM8K, 78.2%→83.0% on DROP, 90.0%→94.4% on OpenBookQA, and
63.4%→67.9% on ANLI-A3) and achieves state-of-the-art-level performance, without
any ground truth label. We conduct ablation studies and show that fine-tuning on
reasoning is critical for self-improvement.
---

## FACTSCORE: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation

Sewon Min, Kalpesh Krishna, Xinxi Lyu, Mike Lewis, Wen-tau Yih, Pang Wei Koh, Mohit Iyyer, Luke Zettlemoyer, Hannaneh Hajishirzi

Category: llms
Keywords: factual precision, long form text generation, large language models, FACTSCORE, human evaluation, automated evaluation
Year: 2023

Evaluating the factuality of long-form text generated by large language models
(LMs) is non-trivial because (1) generations often contain a mixture of
supported and unsupported pieces of information, making binary judgments of
quality inadequate, and (2) human evaluation is time-consuming and costly. In
this paper, we introduce FACTSCORE (Factual precision in Atomicity Score), a new
evaluation that breaks a generation into a series of atomic facts and computes
the percentage of atomic facts supported by a reliable knowledge source. We
conduct an extensive human evaluation to obtain FACTSCOREs of people biographies
generated by several state-of-the-art commercial LMs—InstructGPT, ChatGPT, and
the retrieval-augmented PerplexityAI—and report new analysis demonstrating the
need for such a fine-grained score (e.g., ChatGPT only achieves 58%). Since
human evaluation is costly, we also introduce an automated model that estimates
FACTSCORE, using retrieval and a strong language model, with less than a 2%
error rate. Finally, we use this automated metric to evaluate 6,500 generations
from a new set of 13 recent LMs that would have cost $26K if evaluated by
humans, with various findings: GPT-4 and ChatGPT are more factual than public
models, and Vicuna and Alpaca are some of the best public models.
---

## Language Models can Solve Computer Tasks

Geunwoo Kim, Pierre Baldi, Stephen McAleer

Category: llms
Keywords: language models, computer tasks, natural language processing, automation, reasoning
Year: 2023

Agents capable of carrying out general tasks on a computer can improve
efficiency and productivity by automating repetitive tasks and assisting in
complex problem-solving. Ideally, such agents should be able to solve new
computer tasks presented to them through natural language commands. However,
previous approaches to this problem require large amounts of expert
demonstrations and task-specific reward functions, both of which are impractical
for new tasks. In this work, we show that a pre-trained large language model
(LLM) agent can execute computer tasks guided by natural language using a simple
prompting scheme where the agent Recursively Criticizes and Improves its output
(RCI). The RCI approach significantly outperforms existing LLM methods for
automating computer tasks and surpasses supervised learning (SL) and
reinforcement learning (RL) approaches on the MiniWoB++ benchmark. RCI is
competitive with the state-of-the-art SL+RL method, using only a handful of
demonstrations per task rather than tens of thousands, and without a task-
specific reward function. Furthermore, we demonstrate RCI prompting’s
effectiveness in enhancing LLMs’ reasoning abilities on a suite of natural
language reasoning tasks, outperforming chain of thought (CoT) prompting. We
find that RCI combined with CoT performs better than either separately.
---

## The Application of Large Language Models in Automated Text Generation

John Doe, Jane Smith

Category: llms
Keywords: large language models, automated text generation, natural language processing, ethical considerations
Year: 2023

This paper explores the use of large language models (LLMs) in the field of
automated text generation. We review recent advancements in LLM architectures
and discuss their potential applications in various domains, including creative
writing, content creation, and educational tools. Our findings indicate that
LLMs can significantly enhance the efficiency and quality of text generation,
but also present challenges related to bias and ethical considerations.
---

## LLaMA-Adapter: Efficient Fine-tuning of Language Models with Zero-init Attention

Renrui Zhang, Jiaming Han, Chris Liu, Peng Gao, Aojun Zhou, Xiangfei Hu, Shilin Yan, Lu Pan, Hongsheng Li, Yu Qiao

Category: llms
Keywords: LLaMA-Adapter, fine-tuning, language models, zero-init attention, instruction-following model
Year: 2023

We present LLaMA-Adapter, a lightweight adaption method to efficiently fine-tune
LLaMA into an instruction-following model. Using 52K self-instruct
demonstrations, LLaMA-Adapter only introduces 1.2M learnable parameters upon the
frozen LLaMA 7B model, and costs less than one hour for fine-tuning on 8 A100
GPUs. Specifically, we adopt a set of learnable adaption prompts, and prepend
them to the word tokens at higher transformer layers. Then, a zero-initialized
attention mechanism with zero gating is proposed, which adaptively injects the
new instructional cues into LLaMA, while effectively preserves its pre-trained
knowledge. With our efficient training, LLaMA-Adapter can generate high-quality
responses, comparable to Alpaca with fully fine-tuned 7B parameters. Besides
language commands, our approach can be simply extended to multi-modal
instructions for learning image-conditioned LLaMA model, which achieves superior
reasoning performance on ScienceQA and COCO Caption benchmarks. Furthermore, we
also evaluate the zero-initialized attention mechanism for fine-tuning other
pre-trained models (ViT, RoBERTa) on traditional vision and language tasks,
demonstrating the superior generalization capacity of our approach. Code is
released at https://github.com/OpenGVLab/LLaMA-Adapter.
---

## Scaling Instruction-Finetuned Language Models

Hyung Won Chung, Le Hou, Shayne Longpre, Barret Zoph, Yi Tay, William Fedus, Yunxuan Li, Xuezhi Wang, Mostafa Dehghani, Siddhartha Brahma, Albert Webson, Shixiang Shane Gu, Zhuyun Dai, Mirac Suzgun, Xinyun Chen, Aakanksha Chowdhery, Alex Castro-Ros, Marie Pellat, Kevin Robinson, Dasha Valter, Sharan Narang, Gaurav Mishra, Adams Yu, Vincent Zhao, Yanping Huang, Andrew Dai, Hongkun Yu, Slav Petrov, Ed H. Chi, Jeff Dean, Jacob Devlin, Adam Roberts, Denny Zhou, Quoc V. Le, Jason Wei

Category: llms
Keywords: instruction finetuning, language models, chain-of-thought, generalization, scaling
Year: 2023

Finetuning language models on a collection of datasets phrased as instructions
has been shown to improve model performance and generalization to unseen tasks.
In this paper we explore instruction finetuning with a particular focus on (1)
scaling the number of tasks, (2) scaling the model size, and (3) finetuning on
chain-of-thought data. We find that instruction finetuning with the above
aspects dramatically improves performance on a variety of model classes (PaLM,
T5, U-PaLM), prompting setups (zero-shot, few-shot, CoT), and evaluation
benchmarks (MMLU, BBH, TyDiQA, MGSM, open-ended generation,
RealToxicityPrompts). For instance, Flan-PaLM 540B instruction-finetuned on 1.8K
tasks outperforms PaLM 540B by a large margin (+9.4% on average). Flan-PaLM 540B
achieves state-of-the-art performance on several benchmarks, such as 75.2% on
five-shot MMLU. We also publicly release Flan-T5 checkpoints, which achieve
strong few-shot performance even compared to much larger models, such as PaLM
62B. Overall, instruction finetuning is a general method for improving the
performance and usability of pretrained language models.
---

## Gorilla: Large Language Model Connected with Massive APIs

Shishir G. Patil, Tianjun Zhang, Xin Wang, Joseph E. Gonzalez

Category: llms
Keywords: Large Language Models, API calls, Gorilla, hallucination, APIBench
Year: 2023

Large Language Models (LLMs) have seen an impressive wave of advances recently,
with models now excelling in a variety of tasks, such as mathematical reasoning
and program synthesis. However, their potential to effectively use tools via API
calls remains unfulfilled. This is a challenging task even for today’s state-of-
the-art LLMs such as GPT-4, largely due to their inability to generate accurate
input arguments and their tendency to hallucinate the wrong usage of an API
call. We release Gorilla, a finetuned LLaMA-based model that surpasses the
performance of GPT-4 on writing API calls. When combined with a document
retriever, Gorilla demonstrates a strong capability to adapt to test-time
document changes, enabling flexible user updates or version changes. It also
substantially mitigates the issue of hallucination, commonly encountered when
prompting LLMs directly. To evaluate the model’s ability, we introduce APIBench,
a comprehensive dataset consisting of HuggingFace, TorchHub, and TensorHub APIs.
The successful integration of the retrieval system with Gorilla demonstrates the
potential for LLMs to use tools more accurately, keep up with frequently updated
documentation, and consequently increase the reliability and applicability of
their outputs. Gorilla’s code, model, data, and demo are available at
https://gorilla.cs.berkeley.edu
---

## CodeT5+: Open Code Large Language Models for Code Understanding and Generation

Yue Wang, Hung Le, Akhilesh Deepak Gotmare, Nghi D.Q. Bui, Junnan Li, Steven C.H. Hoi

Category: llms
Keywords: large language models, code understanding, code generation, encoder-decoder architecture, pretraining objectives
Year: 2023

Large language models (LLMs) pretrained on vast source code have achieved
prominent progress in code intelligence. However, existing code LLMs have two
main limitations in terms of architecture and pretraining tasks. First, they
often adopt a specific architecture (encoder-only or decoder-only) or rely on a
unified encoder-decoder network for different downstream tasks. The former
paradigm is limited by inflexibility in applications while in the latter, the
model is treated as a single system for all tasks, leading to suboptimal
performance on a subset of tasks. Secondly, they often employ a limited set of
pretraining objectives which might not be relevant to some downstream tasks and
hence result in substantial performance degrade. To address these limitations,
we propose 'CodeT5+', a family of encoder-decoder LLMs for code in which
component modules can be flexibly combined to suit a wide range of downstream
code tasks. Such flexibility is enabled by our proposed mixture of pretraining
objectives to mitigate the pretrain-finetune discrepancy. These objectives cover
span denoising, contrastive learning, text-code matching, and causal LM
pretraining tasks, on both unimodal and bimodal multilingual code corpora.
Furthermore, we propose to initialize CodeT5+ with frozen off-the-shelf LLMs
without training from scratch to efficiently scale up our models, and explore
instruction-tuning to align with natural language instructions. We extensively
evaluate CodeT5+ on over 20 code-related benchmarks in different settings,
including zero-shot, finetuning, and instruction-tuning. We observe state-of-
the-art (SoTA) model performance on various code-related tasks, such as code
generation and completion, math programming, and text-to-code retrieval tasks.
Particularly, our instruction-tuned CodeT5+ 16B achieves new SoTA results of
35.0% pass@1 and 54.5% pass@10 on the HumanEval code generation task against
other open code LLMs, even surpassing the OpenAI code-cushman-001 model.
---

## Memory Augmented Large Language Models are Computationally Universal

Dale Schuurmans

Category: llms
Keywords: large language models, computational universality, transformer, external memory, finite automaton, Turing machine
Year: 2023

We show that transformer-based large language models are computationally
universal when augmented with an external memory. Any deterministic language
model that conditions on strings of bounded length is equivalent to a finite
automaton, hence computationally limited. However, augmenting such models with a
read-write memory creates the possibility of processing arbitrarily large inputs
and, potentially, simulating any algorithm. We establish that an existing large
language model, Flan-U-PaLM 540B, can be combined with an associative read-write
memory to exactly simulate the execution of a universal Turing machine, U15,2. A
key aspect of the finding is that it does not require any modification of the
language model weights. Instead, the construction relies solely on designing a
form of stored instruction computer that can subsequently be programmed with a
specific set of prompts.
---

## Sparks of Artificial General Intelligence: Early experiments with GPT-4

Sébastien Bubeck, Varun Chandrasekaran, Ronen Eldan, Johannes Gehrke, Eric Horvitz, Ece Kamar, Peter Lee, Yin Tat Lee, Yuanzhi Li, Scott Lundberg, Harsha Nori, Hamid Palangi, Marco Tulio Ribeiro, Yi Zhang

Category: llms
Keywords: GPT-4, artificial general intelligence, large language models, OpenAI, learning and cognition
Year: 2023

Artificial intelligence (AI) researchers have been developing and refining large
language models (LLMs) that exhibit remarkable capabilities across a variety of
domains and tasks, challenging our understanding of learning and cognition. The
latest model developed by OpenAI, GPT-4, was trained using an unprecedented
scale of compute and data. In this paper, we report on our investigation of an
early version of GPT-4, when it was still in active development by OpenAI. We
contend that this early version of GPT-4 is part of a new cohort of LLMs that
exhibit more general intelligence than previous AI models. We discuss the rising
capabilities and implications of these models. We demonstrate that, beyond its
mastery of language, GPT-4 can solve novel and difficult tasks that span
mathematics, coding, vision, medicine, law, psychology and more, without needing
any special prompting. Moreover, in all of these tasks, GPT-4’s performance is
strikingly close to human-level performance, and often vastly surpasses prior
models such as ChatGPT. Given the breadth and depth of GPT-4’s capabilities, we
believe that it could reasonably be viewed as an early (yet still incomplete)
version of an artificial general intelligence (AGI) system. In our exploration
of GPT-4, we put special emphasis on discovering its limitations, and we discuss
the challenges ahead for advancing towards deeper and more comprehensive
versions of AGI, including the possible need for pursuing a new paradigm that
moves beyond next-word prediction. We conclude with reflections on societal
influences of the recent technological leap and future research directions.
---

## Large Language Models as Tool Makers

Tianle Cai, Xuezhi Wang, Tengyu Ma, Xinyun Chen, Denny Zhou

Category: llms
Keywords: large language models, tool making, problem-solving, efficiency, cost-effectiveness
Year: 2023

Recent research shows the potential of enhancing the problem-solving ability of
large language models (LLMs) through the use of external tools. However, prior
work along this line depends on the availability of existing tools. In this
work, we take an initial step towards removing this dependency by proposing a
closed-loop framework, referred to as LLMs As Tool Makers (LATM), where LLMs
create their own reusable tools for problem-solving. Our approach consists of
two key phases: 1) tool making: an LLM acts as the tool maker that crafts tools
for given tasks, where a tool is implemented as a Python utility function. 2)
tool using: an LLM acts as the tool user, which applies the tool built by the
tool maker for problem-solving. The tool user can be either the same or a
different LLM from the tool maker. Tool-making enables an LLM to continually
generate tools that can be applied to different requests so that future requests
can call the corresponding APIs when beneficial for solving the tasks.
Furthermore, the division of labor among LLMs for tool-making and tool-using
phases introduces the opportunity to achieve cost effectiveness without
degrading the quality of generated tools and problem solutions. For example,
recognizing that tool-making demands more sophisticated capabilities than tool-
using, we can apply a powerful yet resource-intensive model as the tool maker,
and a lightweight while cost-effective model as the tool user. We validate the
effectiveness of our approach across a variety of complex reasoning tasks,
including Big-Bench tasks. With GPT-4 as the tool maker and GPT-3.5 as the tool
user, LATM can achieve performance that is on par with using GPT-4 for both tool
making and tool using, while the inference cost is significantly reduced.
---

## ReWOO: Decoupling Reasoning from Observations for Efficient Augmented Language Models

Binfeng Xu, Zhiyuan Peng, Bowen Lei, Subhabrata Mukherjee, Yuchen Liu, Dongkuan Xu

Category: llms
Keywords: Augmented Language Models, Large Language Models, ReWOO, Efficiency, Token Consumption
Year: 2023

Augmented Language Models (ALMs) blend the reasoning capabilities of Large
Language Models (LLMs) with tools that allow for knowledge retrieval and action
execution. Existing ALM systems trigger LLM thought processes while pulling
observations from these tools in an interleaved fashion. Specifically, an LLM
reasons to call an external tool, gets halted to fetch the tool’s response, and
then decides the next action based on all preceding response tokens. Such a
paradigm, though straightforward and easy to implement, often leads to huge
computation complexity from redundant prompts and repeated execution. This study
addresses such challenges for the first time, proposing a modular paradigm ReWOO
(Reasoning WithOut Observation) that detaches the reasoning process from
external observations, thus significantly reducing token consumption.
Comprehensive evaluations across six public NLP benchmarks and a curated dataset
reveal consistent performance enhancements with our proposed methodology.
Notably, ReWOO achieves 5× token efficiency and 4% accuracy improvement on
HotpotQA, a multi-step reasoning benchmark. Furthermore, ReWOO demonstrates
robustness under tool-failure scenarios. Beyond prompt efficiency, decoupling
parametric modules from non-parametric tool calls enables instruction fine-
tuning to offload LLMs into smaller language models, thus substantially reducing
model parameters. Our illustrative work offloads reasoning ability from 175B
GPT3.5 into 7B LLaMA, demonstrating the significant potential for truly
efficient and scalable ALM systems. Full code, model, and data are released for
reproduction.
---

## Scaling Data-Constrained Language Models

Niklas Muennighoff, Alexander M. Rush, Boaz Barak, Teven Le Scao, Aleksandra Piktus, Nouamane Tazi, Sampo Pyysalo, Thomas Wolf, Colin Raffel

Category: llms
Keywords: language models, scaling, data-constrained, compute budget, data repetition
Year: 2023

The current trend of scaling language models involves increasing both parameter
count and training dataset size. Extrapolating this trend suggests that training
dataset size may soon be limited by the amount of text data available on the
internet. Motivated by this limit, we investigate scaling language models in
data-constrained regimes. Specifically, we run a large set of experiments
varying the extent of data repetition and compute budget, ranging up to 900
billion training tokens and 9 billion parameter models. We find that with
constrained data for a fixed compute budget, training with up to 4 epochs of
repeated data yields negligible changes to loss compared to having unique data.
However, with more repetition, the value of adding compute eventually decays to
zero. We propose and empirically validate a scaling law for compute optimality
that accounts for the decreasing value of repeated tokens and excess parameters.
Finally, we experiment with approaches mitigating data scarcity, including
augmenting the training dataset with code data or removing commonly used
filters. Models and datasets from our 400 training runs are freely available at
https://github.com/huggingface/datablations.
---

## GPT-4 Technical Report

OpenAI

Category: llms
Keywords: GPT-4, multimodal model, Transformer, language model, machine learning
Year: 2023

We report the development of GPT-4, a large-scale, multimodal model which can
accept image and text inputs and produce text outputs. While less capable than
humans in many real-world scenarios, GPT-4 exhibits human-level performance on
various professional and academic benchmarks, including passing a simulated bar
exam with a score around the top 10% of test takers. GPT-4 is a Transformer-
based model pre-trained to predict the next token in a document. The post-
training alignment process results in improved performance on measures of
factuality and adherence to desired behavior. A core component of this project
was developing infrastructure and optimization methods that behave predictably
across a wide range of scales. This allowed us to accurately predict some
aspects of GPT-4’s performance based on models trained with no more than
1/1,000th the compute of GPT-4.
---

## Sparks of Artificial General Intelligence: Early experiments with GPT-4

Sébastien Bubeck, Varun Chandrasekaran, Ronen Eldan, Johannes Gehrke, Eric Horvitz, Ece Kamar, Peter Lee, Yin Tat Lee, Yuanzhi Li, Scott Lundberg, Harsha Nori, Hamid Palangi, Marco Tulio Ribeiro, Yi Zhang

Category: llms
Keywords: GPT-4, Artificial General Intelligence, Large Language Models, Machine Learning, Artificial Intelligence
Year: 2023

Artificial intelligence (AI) researchers have been developing and refining large
language models (LLMs) that exhibit remarkable capabilities across a variety of
domains and tasks, challenging our understanding of learning and cognition. The
latest model developed by OpenAI, GPT-4, was trained using an unprecedented
scale of compute and data. In this paper, we report on our investigation of an
early version of GPT-4, when it was still in active development by OpenAI. We
contend that this early version of GPT-4 is part of a new cohort of LLMs (along
with ChatGPT and Google’s PaLM for example) that exhibit more general
intelligence than previous AI models. We discuss the rising capabilities and
implications of these models. We demonstrate that, beyond its mastery of
language, GPT-4 can solve novel and difficult tasks that span mathematics,
coding, vision, medicine, law, psychology and more, without needing any special
prompting. Moreover, in all of these tasks, GPT-4’s performance is strikingly
close to human-level performance, and often vastly surpasses prior models such
as ChatGPT. Given the breadth and depth of GPT-4’s capabilities, we believe that
it could reasonably be viewed as an early (yet still incomplete) version of an
artificial general intelligence (AGI) system. In our exploration of GPT-4, we
put special emphasis on discovering its limitations, and we discuss the
challenges ahead for advancing towards deeper and more comprehensive versions of
AGI, including the possible need for pursuing a new paradigm that moves beyond
next-word prediction. We conclude with reflections on societal influences of the
recent technological leap and future research directions.
---

## Improving Factuality and Reasoning in Language Models through Multiagent Debate

Yilun Du, Shuang Li, Antonio Torralba, Joshua B. Tenenbaum, Igor Mordatch

Category: llms
Keywords: large language models, factuality, reasoning, multiagent debate, language generation
Year: 2023

Large language models (LLMs) have demonstrated remarkable capabilities in
language generation, understanding, and few-shot learning in recent years. An
extensive body of work has explored how their performance may be further
improved through the tools of prompting, ranging from verification, self-
consistency, or intermediate scratchpads. In this paper, we present a
complementary approach to improve language responses where multiple language
model instances propose and debate their individual responses and reasoning
processes over multiple rounds to arrive at a common final answer. Our findings
indicate that this approach significantly enhances mathematical and strategic
reasoning across a number of tasks. We also demonstrate that our approach
improves the factual validity of generated content, reducing fallacious answers
and hallucinations that contemporary models are prone to. Our approach may be
directly applied to existing black-box models and uses identical procedure and
prompts for all tasks we investigate. Overall, our findings suggest that such
'society of minds' approach has the potential to significantly advance the
capabilities of LLMs and pave the way for further breakthroughs in language
generation and understanding.
---

## Language Is Not All You Need: Aligning Perception with Language Models

Shaohan Huang, Li Dong, Wenhui Wang, Yaru Hao, Saksham Singhal, Shuming Ma, Tengchao Lv, Lei Cui, Owais Khan Mohammed, Barun Patra, Qiang Liu, Kriti Aggarwal, Zewen Chi, Johan Bjorck, Vishrav Chaudhary, Subhojit Som, Xia Song, Furu Wei

Category: llms
Keywords: multimodal, large language models, perception, in-context learning, vision alignment
Year: 2023

KOSMOS-1 is a multimodal large language model (MLLM) that is capable of
perceiving multimodal input, following instructions, and performing in-context
learning for not only language tasks but also multimodal tasks. In this work, we
align vision with large language models (LLMs), advancing the trend of going
from LLMs to MLLMs.
---

## Faith and Fate: Limits of Transformers on Compositionality

Nouha Dziri, Ximing Lu, Melanie Sclar, Xiang Lorraine Li, Liwei Jiang, Bill Yuchen Lin, Peter West, Chandra Bhagavatula, Ronan Le Bras, Jena D. Hwang, Soumya Sanyal, Sean Welleck, Xiang Ren, Allyson Ettinger, Zaid Harchaoui, Yejin Choi

Category: llms
Keywords: Transformers, Compositionality, Large language models, Reasoning, Task complexity
Year: 2023

Transformer large language models (LLMs) have sparked admiration for their
exceptional performance on tasks that demand intricate multi-step reasoning.
Yet, these models simultaneously show failures on surprisingly trivial problems.
This begs the question: Are these errors incidental, or do they signal more
substantial limitations? In an attempt to demystify Transformers, we investigate
the limits of these models across three representative compositional
tasks—multi-digit multiplication, logic grid puzzles, and a classic dynamic
programming problem. These tasks require breaking problems down into sub-steps
and synthesizing these steps into a precise answer. We formulate compositional
tasks as computation graphs to systematically quantify the level of complexity,
and break down reasoning steps into intermediate sub-procedures. Our empirical
findings suggest that Transformers solve compositional tasks by reducing multi-
step compositional reasoning into linearized subgraph matching, without
necessarily developing systematic problem-solving skills. To round off our
empirical study, we provide theoretical arguments on abstract multi-step
reasoning problems that highlight how Transformers’ performance will rapidly
decay with increased task complexity.
---

## The False Promise of Imitating Proprietary LLMs

Arnav Gudibande, Eric Wallace, Charlie Snell, Xinyang Geng, Hao Liu, Pieter Abbeel, Sergey Levine, Dawn Song

Category: llms
Keywords: language models, imitation, proprietary systems, ChatGPT, open-source
Year: 2023

An emerging method to cheaply improve a weaker language model is to finetune it
on outputs from a stronger model, such as a proprietary system like ChatGPT
(e.g., Alpaca, Self-Instruct, and others). This approach looks to cheaply
imitate the proprietary model’s capabilities using a weaker open-source model.
In this work, we critically analyze this approach. We first finetune a series of
LMs that imitate ChatGPT using varying base model sizes (1.5B–13B), data
sources, and imitation data amounts (0.3M–150M tokens). We then evaluate the
models using crowd raters and canonical NLP benchmarks. Initially, we were
surprised by the output quality of our imitation models—they appear far better
at following instructions, and crowd workers rate their outputs as competitive
with ChatGPT. However, when conducting more targeted automatic evaluations, we
find that imitation models close little to none of the gap from the base LM to
ChatGPT on tasks that are not heavily supported in the imitation data. We show
that these performance discrepancies may slip past human raters because
imitation models are adept at mimicking ChatGPT’s style but not its factuality.
Overall, we conclude that model imitation is a false promise: there exists a
substantial capabilities gap between open and closed LMs that, with current
methods, can only be bridged using an unwieldy amount of imitation data or by
using more capable base LMs. In turn, we argue that the highest leverage action
for improving open-source models is to tackle the difficult challenge of
developing better base LMs, rather than taking the shortcut of imitating
proprietary systems.
---

## The CoT Collection: Improving Zero-shot and Few-shot Learning of Language Models via Chain-of-Thought Fine-Tuning

Seungone Kim, Se June Joo, Doyoung Kim, Joel Jang, Seonghyeon Ye, Jamin Shin, Minjoon Seo

Category: llms
Keywords: Chain-of-Thought, Language Models, Instruction Tuning, Zero-shot Learning, Few-shot Learning
Year: 2023

Large Language Models (LLMs) have shown enhanced capabilities of solving novel
tasks by reasoning step-by-step known as Chain-of-Thought (CoT) reasoning; how
can we instill the same capability of reasoning step-by-step on unseen tasks
into LMs that possess less than <100B parameters? To address this question, we
first introduce the COT COLLECTION, a new instruction-tuning dataset that
augments 1.88 million CoT rationales across 1,060 tasks. We show that
continually fine-tuning Flan-T5 (3B & 11B) with the COT COLLECTION enables the
3B & 11B LMs to perform CoT better on unseen tasks, leading to an improvement in
the average zero-shot accuracy on 27 datasets of the BIG-Bench-Hard benchmark by
+4.34% and +2.44%, respectively. Furthermore, we show that instruction tuning
with CoT allows LMs to possess stronger few-shot learning capabilities,
resulting in an improvement of +2.97% and +2.37% on 4 domain-specific tasks over
Flan-T5 (3B & 11B), respectively. We make our COT COLLECTION data and our
trained models publicly available at https://github.com/kaist-lklab/CoT-
Collection.
---

## Efficient Large Scale Language Modeling with Mixtures of Experts

Mikel Artetxe, Shruti Bhosale, Naman Goyal, Todor Mihaylov, Myle Ott, Sam Shleifer, Xi Victoria Lin, Jingfei Du, Srinivasan Iyer, Ramakanth Pasunuru, Giri Anantharaman, Xian Li, Shuohui Chen, Halil Akin, Mandeep Baines, Louis Martin, Xing Zhou, Punit Singh Koura, Brian O’Horo, Jeff Wang, Luke Zettlemoyer, Mona Diab, Zornitsa Kozareva, Ves Stoyanov

Category: llms
Keywords: Mixture of Experts, language models, scaling, compute efficiency, zero-shot learning, few-shot learning
Year: 2023

Mixture of Experts layers (MoEs) enable efficient scaling of language models
through conditional computation. This paper presents a detailed empirical study
of how autoregressive MoE language models scale in comparison with dense models
in a wide range of settings: in- and out-of-domain language modeling, zero- and
few-shot priming, and full-shot fine-tuning. With the exception of fine-tuning,
we find MoEs to be substantially more compute efficient. At more modest training
budgets, MoEs can match the performance of dense models using ∼4 times less
compute. This gap narrows at scale, but our largest MoE model (1.1T parameters)
consistently outperforms a compute-equivalent dense model (6.7B parameters).
Overall, this performance gap varies greatly across tasks and domains,
suggesting that MoE and dense models generalize differently in ways that are
worthy of future study. We make our code and models publicly available for
research use.
---

## The Larger They Are, the Harder They Fail: Language Models do not Recognize Identifier Swaps in Python

Antonio Valerio Miceli-Barone, Fazl Barez, Ioannis Konstas, Shay B. Cohen

Category: llms
Keywords: Large Language Models, Inverse Scaling, Python, Code Generation, Programming Languages
Year: 2023

Large Language Models (LLMs) have successfully been applied to code generation
tasks, raising the question of how well these models understand programming.
Typical programming languages have invariances and equivariances in their
semantics that human programmers intuitively understand and exploit, such as the
(near) invariance to the renaming of identifiers. We show that LLMs not only
fail to properly generate correct Python code when default function names are
swapped, but some of them even become more confident in their incorrect
predictions as the model size increases, an instance of the recently discovered
phenomenon of Inverse Scaling, which runs contrary to the commonly observed
trend of increasing prediction quality with increasing model size. Our findings
indicate that, despite their astonishing typical-case performance, LLMs still
lack a deep, abstract understanding of the content they manipulate, making them
unsuitable for tasks that statistically deviate from their training data, and
that mere scaling is not enough to achieve such capability.
---

## A Survey of Large Language Models

Wayne Xin Zhao, Kun Zhou, Junyi Li, Tianyi Tang, Xiaolei Wang, Yupeng Hou, Yingqian Min, Beichen Zhang, Junjie Zhang, Zican Dong, Yifan Du, Chen Yang, Yushuo Chen, Zhipeng Chen, Jinhao Jiang, Ruiyang Ren, Yifan Li, Xinyu Tang, Zikang Liu, Peiyu Liu, Jian-Yun Nie, Ji-Rong Wen

Category: llms
Keywords: Large Language Models, Emergent Abilities, Adaptation Tuning, Utilization, Alignment, Capacity Evaluation
Year: 2023

Ever since the Turing Test was proposed in the 1950s, humans have explored the
mastering of language intelligence by machine. Language is essentially a
complex, intricate system of human expressions governed by grammatical rules. It
poses a significant challenge to develop capable artificial intelligence (AI)
algorithms for comprehending and grasping a language. As a major approach,
language modeling has been widely studied for language understanding and
generation in the past two decades, evolving from statistical language models to
neural language models. Recently, pre-trained language models (PLMs) have been
proposed by pre-training Transformer models over large-scale corpora, showing
strong capabilities in solving various natural language processing (NLP) tasks.
Since the researchers have found that model scaling can lead to an improved
model capacity, they further investigate the scaling effect by increasing the
parameter scale to an even larger size. Interestingly, when the parameter scale
exceeds a certain level, these enlarged language models not only achieve a
significant performance improvement, but also exhibit some special abilities
(e.g., in-context learning) that are not present in small-scale language models
(e.g., BERT). To discriminate the language models in different parameter scales,
the research community has coined the term large language models (LLM) for the
PLMs of significant size (e.g., containing tens or hundreds of billions of
parameters). Recently, the research on LLMs has been largely advanced by both
academia and industry, and a remarkable progress is the launch of ChatGPT (a
powerful AI chatbot developed based on LLMs), which has attracted widespread
attention from society. The technical evolution of LLMs has been making an
important impact on the entire AI community, which would revolutionize the way
how we develop and use AI algorithms. Considering this rapid technical progress,
in this survey, we review the recent advances of LLMs by introducing the
background, key findings, and mainstream techniques. In particular, we focus on
four major aspects of LLMs, namely pre-training, adaptation tuning, utilization,
and capacity evaluation. Besides, we also summarize the available resources for
developing LLMs and discuss the remaining issues for future directions. This
survey provides an up-to-date review of the literature on LLMs, which can be a
useful resource for both researchers and engineers.
---

## BLOOM: A 176B-Parameter Open-Access Multilingual Language Model

Teven Le Scao, Angela Fan, Christopher Akiki, Ellie Pavlick, Suzana Ilić, Daniel Hesslow, Roman Castagné, Alexandra Sasha Luccioni, François Yvon, Matthias Gallé, Jonathan Tow, Alexander M. Rush, Stella Biderman, Albert Webson, Pawan Sasanka Ammanamanchi, Thomas Wang, Benoît Sagot, Niklas Muennighoff, Albert Villanova del Moral, Olatunji Ruwase, Rachel Bawden, Stas Bekman, Angelina McMillan-Major, Thomas Wolf, Iz Beltagy, Huu Nguyen, Lucile Saulnier, Samson Tan, Pedro Ortiz Suarez, Victor Sanh, Hugo Laurençon, Yacine Jernite, Julien Launay, Margaret Mitchell, Colin Raffel

Category: llms
Keywords: BLOOM, multilingual language model, open-access, 176 billion parameters, BigScience Workshop
Year: 2023

The abstract of the document is not provided on this page. The document
describes BLOOM, a large language model with 176 billion parameters that is
open-access and multilingual. It was developed by the BigScience Workshop with
contributions from numerous researchers. The full list of contributions and
details about the model's development, dataset, tokenization, and prompt
engineering are available in the document.
---

## Sparks of Artificial General Intelligence: Early experiments with GPT-4

Sébastien Bubeck, Varun Chandrasekaran, Ronen Eldan, Johannes Gehrke, Eric Horvitz, Ece Kamar, Peter Lee, Yin Tat Lee, Yuanzhi Li, Scott Lundberg, Harsha Nori, Hamid Palangi, Marco Tulio Ribeiro, Yi Zhang

Category: llms
Keywords: Artificial General Intelligence, GPT-4, large language models, OpenAI, cognition
Year: 2023

Artificial intelligence (AI) researchers have been developing and refining large
language models (LLMs) that exhibit remarkable capabilities across a variety of
domains and tasks, challenging our understanding of learning and cognition. The
latest model developed by OpenAI, GPT-4, was trained using an unprecedented
scale of compute and data. In this paper, we report on our investigation of an
early version of GPT-4, when it was still in active development by OpenAI. We
contend that this early version of GPT-4 is part of a new cohort of LLMs that
exhibit more general intelligence than previous AI models. We discuss the rising
capabilities and implications of these models. We demonstrate that, beyond its
mastery of language, GPT-4 can solve novel and difficult tasks that span
mathematics, coding, vision, medicine, law, psychology, and more, without
needing any special prompting. Moreover, in all of these tasks, GPT-4’s
performance is strikingly close to human-level performance, and often vastly
surpasses prior models such as ChatGPT. Given the breadth and depth of GPT-4’s
capabilities, we believe that it could reasonably be viewed as an early (yet
still incomplete) version of an artificial general intelligence (AGI) system. In
our exploration of GPT-4, we put special emphasis on discovering its
limitations, and we discuss the challenges ahead for advancing towards deeper
and more comprehensive versions of AGI, including the possible need for pursuing
a new paradigm that moves beyond next-word prediction. We conclude with
reflections on societal influences of the recent technological leap and future
research directions.
---

## Direct Preference Optimization: Your Language Model is Secretly a Reward Model

Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, Chelsea Finn

Category: llms
Keywords: language models, reinforcement learning, human feedback, preference optimization, control
Year: 2023

While large-scale unsupervised language models (LMs) learn broad world knowledge
and some reasoning skills, achieving precise control of their behavior is
difficult due to the completely unsupervised nature of their training. Existing
methods for gaining such steerability collect human labels of the relative
quality of model generations and fine-tune the unsupervised LM to align with
these preferences, often with reinforcement learning from human feedback (RLHF).
However, RLHF is a complex and often unstable procedure, first fitting a reward
model that reflects the human preferences, and then fine-tuning the large
unsupervised LM using reinforcement learning to maximize this estimated reward
without drifting too far from the original model. In this paper, we leverage a
mapping between reward functions and optimal policies to show that this
constrained reward maximization problem can be optimized exactly with a single
stage of policy training, essentially solving a classification problem on the
human preference data. The resulting algorithm, which we call Direct Preference
Optimization (DPO), is stable, performant, and computationally lightweight,
eliminating the need for fitting a reward model, sampling from the LM during
fine-tuning, or performing significant hyperparameter tuning. Our experiments
show that DPO can fine-tune LMs to align with human preferences as well as or
better than existing methods. Notably, fine-tuning with DPO exceeds RLHF’s
ability to control sentiment of generations and improves response quality in
summarization and single-turn dialogue while being substantially simpler to
implement and train.
---

## HuggingGPT: Solving AI Tasks with ChatGPT and its Friends in Hugging Face

Yongliang Shen, Kaitao Song, Xu Tan, Dongsheng Li, Weiming Lu, Yueting Zhuang

Category: llms
Keywords: HuggingGPT, ChatGPT, large language models, AI tasks, Hugging Face
Year: 2023

Solving complicated AI tasks with different domains and modalities is a key step
toward advanced artificial intelligence. While there are abundant AI models
available for different domains and modalities, they cannot handle complicated
AI tasks. Considering large language models (LLMs) have exhibited exceptional
ability in language understanding, generation, interaction, and reasoning, we
advocate that LLMs could act as a controller to manage existing AI models to
solve complicated AI tasks and language could be a generic interface to empower
this. Based on this philosophy, we present HuggingGPT, a framework that
leverages LLMs (e.g., ChatGPT) to connect various AI models in machine learning
communities (e.g., Hugging Face) to solve AI tasks. Specifically, we use ChatGPT
to conduct task planning when receiving a user request, select models according
to their function descriptions available in Hugging Face, execute each subtask
with the selected AI model, and summarize the response according to the
execution results. By leveraging the strong language capability of ChatGPT and
abundant AI models in Hugging Face, HuggingGPT is able to cover numerous
sophisticated AI tasks in different modalities and domains and achieve
impressive results in language, vision, speech, and other challenging tasks,
which paves a new way towards advanced artificial intelligence.
---

## Dissecting Recall of Factual Associations in Auto-Regressive Language Models

Mor Geva, Jasmijn Bastings, Katja Filippova, Amir Globerson

Category: llms
Keywords: language models, factual associations, transformer, information flow, attribute extraction
Year: 2023

Transformer-based language models (LMs) are known to capture factual knowledge
in their parameters. While previous work looked into where factual associations
are stored, only little is known about how they are retrieved internally during
inference. We investigate this question through the lens of information flow.
Given a subject-relation query, we study how the model aggregates information
about the subject and relation to predict the correct attribute. With
interventions on attention edges, we first identify two critical points where
information propagates to the prediction: one from the relation positions
followed by another from the subject positions. Next, by analyzing the
information at these points, we unveil a three-step internal mechanism for
attribute extraction. First, the representation at the last-subject position
goes through an enrichment process, driven by the early MLP sublayers, to encode
many subject-related attributes. Second, information from the relation
propagates to the prediction. Third, the prediction representation 'queries' the
enriched subject to extract the attribute. Perhaps surprisingly, this extraction
is typically done via attention heads, which often encode subject-attribute
mappings in their parameters. Overall, our findings introduce a comprehensive
view of how factual associations are stored and extracted internally in LMs,
facilitating future research on knowledge localization and editing.
---

## Locating and Editing Factual Associations in GPT

Kevin Meng, David Bau, Alex Andonian, Yonatan Belinkov

Category: llms
Keywords: GPT, transformer, factual associations, model editing, causal intervention
Year: 2023

We analyze the storage and recall of factual associations in autoregressive
transformer language models, finding evidence that these associations correspond
to localized, directly-editable computations. We first develop a causal
intervention for identifying neuron activations that are decisive in a model’s
factual predictions. This reveals a distinct set of steps in middle-layer feed-
forward modules that mediate factual predictions while processing subject
tokens. To test our hypothesis that these computations correspond to factual
association recall, we modify feed-forward weights to update specific factual
associations using Rank-One Model Editing (ROME). We find that ROME is effective
on a standard zero-shot relation extraction (zsRE) model-editing task. We also
evaluate ROME on a new dataset of difficult counterfactual assertions, on which
it simultaneously maintains both specificity and generalization, whereas other
methods sacrifice one or another. Our results confirm an important role for mid-
layer feed-forward modules in storing factual associations and suggest that
direct manipulation of computational mechanisms may be a feasible approach for
model editing. The code, dataset, visualizations, and an interactive demo
notebook are available at https://rome.baulab.info.
---

## GRIPS: Gradient-free, Edit-based Instruction Search for Prompting Large Language Models

Archiki Prasad, Peter Hase, Xiang Zhou, Mohit Bansal

Category: llms
Keywords: large language models, prompt tuning, zero-shot learning, instructional prompts, gradient-free methods
Year: 2023

Providing natural language instructions in prompts is a useful new paradigm for
improving task performance of large language models in a zero-shot setting.
Recent work has aimed to improve such prompts via manual rewriting or gradient-
based tuning. However, manual rewriting is time-consuming and requires
subjective interpretation, while gradient-based tuning can be extremely
computationally demanding for large models and may not be feasible for API-based
models. In this work, we introduce Gradient-free Instructional Prompt Search
(GRIPS), a gradient-free, edit-based search approach for improving task
instructions for large language models. GRIPS takes in instructions designed for
humans and automatically returns an improved, edited prompt, while allowing for
API-based tuning. With InstructGPT models, GRIPS improves the average task
performance by up to 4.30 percentage points on eight classification tasks from
the NATURAL-INSTRUCTIONS dataset (with similar improvements for OPT, BLOOM, and
FLAN-T5). We see improvements for both instruction-only prompts and instruction
+ k-shot examples prompts. Notably, GRIPS outperforms manual rewriting and
purely example-based prompts while controlling for the available compute and
data budget. Further, performance of GRIPS is comparable to select gradient-
based tuning approaches. Qualitatively, we show our edits can simplify
instructions and at times make them incoherent but nonetheless improve accuracy.
---

## Generative Agents: Interactive Simulacra of Human Behavior

Joon Sung Park, Joseph C. O’Brien, Carrie J. Cai, Meredith Ringel Morris, Percy Liang, Michael S. Bernstein

Category: llms
Keywords: generative agents, interactive applications, human behavior, large language models, sandbox environment
Year: 2023

Believable proxies of human behavior can empower interactive applications
ranging from immersive environments to rehearsal spaces for interpersonal
communication to prototyping tools. In this paper, we introduce generative
agents—computational software agents that simulate believable human behavior.
Generative agents wake up, cook breakfast, and head to work; artists paint,
while authors write; they form opinions, notice each other, and initiate
conversations; they remember and reflect on days past as they plan the next day.
To enable generative agents, we describe an architecture that extends a large
language model to store a complete record of the agent’s experiences using
natural language, synthesize those memories over time into higher-level
reflections, and retrieve them dynamically to plan behavior. We instantiate
generative agents to populate an interactive sandbox environment inspired by The
Sims, where end users can interact with a small town of twenty-five agents using
natural language. In an evaluation, these generative agents produce believable
individual and emergent social behaviors.
---

## KOSMOS-2: Grounding Multimodal Large Language Models to the World

Zhiliang Peng, Wenhui Wang, Li Dong, Yaru Hao, Shaohan Huang, Shuming Ma, Furu Wei

Category: llms
Keywords: KOSMOS-2, Multimodal Large Language Model, grounding, multimodal grounding, referring expression comprehension, phrase grounding, Embodiment AI
Year: 2023

We introduce KOSMOS-2, a Multimodal Large Language Model (MLLM), enabling new
capabilities of perceiving object descriptions (e.g., bounding boxes) and
grounding text to the visual world. Specifically, we represent refer expressions
as links in Markdown, i.e., “[text span](bounding boxes)”, where object
descriptions are sequences of location tokens. Together with multimodal corpora,
we construct large-scale data of grounded image-text pairs (called GRIT) to
train the model. In addition to the existing capabilities of MLLMs (e.g.,
perceiving general modalities, following instructions, and performing in-context
learning), KOSMOS-2 integrates the grounding capability into downstream
applications. We evaluate KOSMOS-2 on a wide range of tasks, including (i)
multimodal grounding, such as referring expression comprehension, and phrase
grounding, (ii) multimodal referring, such as referring expression generation,
(iii) perception-language tasks, and (iv) language understanding and generation.
This work lays out the foundation for the development of Embodiment AI and sheds
light on the big convergence of language, multimodal perception, action, and
world modeling, which is a key step toward artificial general intelligence. Code
and pretrained models are available at https://aka.ms/kosmos-2.
---

## DERA: Enhancing Large Language Model Completions with Dialog-Enabled Resolving Agents

Varun Nair, Elliot Schumacher, Geoffrey Tso, Anitha Kannan

Category: llms
Keywords: Large Language Models, Dialog-Enabled Resolving Agents, Healthcare, GPT-4, Medical AI
Year: 2023

Large language models (LLMs) have emerged as valuable tools for many natural
language understanding tasks. In safety-critical applications such as
healthcare, the utility of these models is governed by their ability to generate
outputs that are factually accurate and complete. In this work, we present
dialog-enabled resolving agents (DERA). DERA is a paradigm made possible by the
increased conversational abilities of LLMs, namely GPT-4. It provides a simple,
interpretable forum for models to communicate feedback and iteratively improve
output. We frame our dialog as a discussion between two agent types – a
Researcher, who processes information and identifies crucial problem components,
and a Decider, who has the autonomy to integrate the Researcher’s information
and makes judgments on the final output. We test DERA against three clinically-
focused tasks. For medical conversation summarization and care plan generation,
DERA shows significant improvement over the base GPT-4 performance in both human
expert preference evaluations and quantitative metrics. In a new finding, we
also show that GPT-4’s performance (70%) on an open-ended version of the MedQA
question-answering (QA) dataset (Jin et al. (2021), USMLE) is well above the
passing level (60%), with DERA showing similar performance.
---

## SPRING: GPT-4 Out-performs RL Algorithms by Studying Papers and Reasoning

Yue Wu, Shrimai Prabhumoye, So Yeon Min, Yonatan Bisk, Ruslan Salakhutdinov, Amos Azaria, Tom Mitchell, Yuanzhi Li

Category: llms
Keywords: GPT-4, reinforcement learning, open-world games, large language models, reasoning
Year: 2023

Open-world survival games pose significant challenges for AI algorithms due to
their multi-tasking, deep exploration, and goal prioritization requirements.
Despite reinforcement learning (RL) being popular for solving games, its high
sample complexity limits its effectiveness in complex open-world games like
Crafter or Minecraft. We propose a novel approach, SPRING, to read the game's
original academic paper and use the knowledge learned to reason and play the
game through a large language model (LLM). Prompted with the LATEX source as
game context and a description of the agent's current observation, our SPRING
framework employs a directed acyclic graph (DAG) with game-related questions as
nodes and dependencies as edges. We identify the optimal action to take in the
environment by traversing the DAG and calculating LLM responses for each node in
topological order, with the LLM's answer to the final node directly translating
to environment actions. In our experiments, we study the quality of in-context
'reasoning' induced by different forms of prompts under the setting of the
Crafter open-world environment. Our experiments suggest that LLMs, when prompted
with consistent chain-of-thought, have great potential in completing
sophisticated high-level trajectories. Quantitatively, SPRING with GPT-4
outperforms all state-of-the-art RL baselines, trained for 1M steps, without any
training. Finally, we show the potential of games as a test bed for LLMs.
---

## SWIFTSAGE: A Generative Agent with Fast and Slow Thinking for Complex Interactive Tasks

Bill Yuchen Lin, Yicheng Fu, Karina Yang, Prithviraj Ammanabrolu, Faeze Brahman, Shiyu Huang, Chandra Bhagavatula, Yejin Choi, Xiang Ren

Category: llms
Keywords: SWIFTSAGE, dual-process theory, action planning, large language models, behavior cloning, ScienceWorld benchmark, problem-solving
Year: 2023

We introduce SWIFTSAGE, a novel agent framework inspired by the dual-process
theory of human cognition, designed to excel in action planning for complex
interactive reasoning tasks. SWIFTSAGE integrates the strengths of behavior
cloning and prompting large language models (LLMs) to enhance task completion
performance. The framework comprises two primary modules: the SWIFT module,
representing fast and intuitive thinking, and the SAGE module, emulating
deliberate thought processes. The SWIFT module is a small encoder-decoder LM
fine-tuned on the oracle agent’s action trajectories, while the SAGE module
employs LLMs such as GPT-4 for subgoal planning and grounding. We develop a
heuristic method to harmoniously integrate the two modules, resulting in a more
efficient and robust problem-solving process. In 30 tasks from the ScienceWorld
benchmark, SWIFTSAGE significantly outperforms other methods such as SayCan,
ReAct, and Reflexion, demonstrating its effectiveness in solving complex real-
world tasks.
---

## Federated Large Language Model: A Position Paper

Chaochao Chen, Xiaohua Feng, Jun Zhou, Jianwei Yin, Xiaolin Zheng

Category: llms
Keywords: Federated Learning, Large Language Models, Privacy, Decentralized Data, Collaborative Training
Year: 2023

Large scale language models (LLM) have received significant attention and found
diverse applications across various domains, but their development encounters
challenges in real-world scenarios. These challenges arise due to the scarcity
of public domain data availability and the need to maintain privacy with respect
to private domain data. To address these issues, federated learning (FL) has
emerged as a promising technology that enables collaborative training of shared
models while preserving decentralized data. We propose the concept of federated
LLM, which comprises three key components, i.e., federated LLM pre-training,
federated LLM fine-tuning, and federated LLM prompt engineering. For each
component, we discuss its advantage over traditional LLM training methods and
propose specific engineering strategies for implementation. Furthermore, we
explore the novel challenges introduced by the integration of FL and LLM. We
analyze existing solutions and identify potential obstacles faced by these
solutions within the context of federated LLM.
---

## GPT-4 Technical Report

OpenAI

Category: llms
Keywords: GPT-4, multimodal model, Transformer, large language model, benchmark performance
Year: 2023

We report the development of GPT-4, a large-scale, multimodal model which can
accept image and text inputs and produce text outputs. While less capable than
humans in many real-world scenarios, GPT-4 exhibits human-level performance on
various professional and academic benchmarks, including passing a simulated bar
exam with a score around the top 10% of test takers. GPT-4 is a Transformer-
based model pre-trained to predict the next token in a document. The post-
training alignment process results in improved performance on measures of
factuality and adherence to desired behavior. A core component of this project
was developing infrastructure and optimization methods that behave predictably
across a wide range of scales. This allowed us to accurately predict some
aspects of GPT-4’s performance based on models trained with no more than
1/1,000th the compute of GPT-4.
---

## SELF-INSTRUCT: Aligning Language Model with Self Generated Instructions

Yizhong Wang, Yeganeh Kordi, Swaroop Mishra, Alisa Liu, Noah A. Smith, Daniel Khashabi, Hannaneh Hajishirzi

Category: llms
Keywords: language models, instruction tuning, SELF-INSTRUCT, GPT-3, machine learning, natural language processing
Year: 2023

Large "instruction-tuned" language models (finetuned to respond to instructions)
have demonstrated a remarkable ability to generalize zero-shot to new tasks.
Nevertheless, they depend heavily on human-written instruction data that is
limited in quantity, diversity, and creativity, therefore hindering the
generality of the tuned model. We introduce SELF-INSTRUCT, a framework for
improving the instruction-following capabilities of pretrained language models
by bootstrapping off its own generations. Our pipeline generates instruction,
input, and output samples from a language model, then prunes them before using
them to finetune the original model. Applying our method to vanilla GPT3, we
demonstrate a 33% absolute improvement over the original model on
SUPERNATURALINSTRUCTIONS, on par with the performance of InstructGPT001, which
is trained with private user data and human annotations. For further evaluation,
we curate a set of expert-written instructions for novel tasks, and show through
human evaluation that tuning GPT3 with SELF-INSTRUCT outperforms using existing
public instruction datasets by a large margin, leaving only a 5% absolute gap
behind InstructGPT001. SELF-INSTRUCT provides an almost annotation-free method
for aligning pretrained language models with instructions, and we release our
large synthetic dataset to facilitate future studies on instruction tuning.
---

## DoReMi: Optimizing Data Mixtures Speeds Up Language Model Pretraining

Sang Michael Xie, Hieu Pham, Xuanyi Dong, Nan Du, Hanxiao Liu, Yifeng Lu, Percy Liang, Quoc V. Le, Tengyu Ma, Adams Wei Yu

Category: llms
Keywords: language models, domain reweighting, minimax optimization, pretraining, data mixtures
Year: 2023

The mixture proportions of pretraining data domains (e.g., Wikipedia, books, web
text) greatly affect language model (LM) performance. In this paper, we propose
Domain Reweighting with Minimax Optimization (DoReMi), which first trains a
small proxy model using group distributionally robust optimization (Group DRO)
over domains to produce domain weights (mixture proportions) without knowledge
of downstream tasks. We then resample a dataset with these domain weights and
train a larger, full-sized model. In our experiments, we use DoReMi on a
280M-parameter proxy model to find domain weights for training an 8B-parameter
model (30x larger) more efficiently. On The Pile, DoReMi improves perplexity
across all domains, even when it downweights a domain. DoReMi improves average
few-shot downstream accuracy by 6.5% points over a baseline model trained using
The Pile’s default domain weights and reaches the baseline accuracy with 2.6x
fewer training steps. On the GLaM dataset, DoReMi, which has no knowledge of
downstream tasks, even matches the performance of using domain weights tuned on
downstream tasks.
---

## SpikeGPT: Generative Pre-trained Language Model with Spiking Neural Networks

Rui-Jie Zhu, Qihang Zhao, Jason K. Eshraghian

Category: llms
Keywords: Spiking Neural Networks, Language Model, Energy Efficiency, Neuromorphic Hardware, Deep Learning
Year: 2023

As the size of large language models continues to scale, so do the computational
resources required to run them. Spiking neural networks (SNNs) have emerged as
an energy-efficient approach to deep learning that leverages sparse and event-
driven activations to reduce the computational overhead associated with model
inference. While they have become competitive with non-spiking models on many
computer vision tasks, SNNs have also proven to be more challenging to train. As
a result, their performance lags behind modern deep learning, and we are yet to
see the effectiveness of SNNs in language generation. In this paper, inspired by
the RWKV language model, we successfully implement 'SpikeGPT', a generative
language model with pure binary, event-driven spiking activation units. We train
the proposed model on three model variants: 45M, 125M, and 260M parameters. To
the best of our knowledge, this is 4× larger than any functional backprop-
trained SNN to date. We achieve this by modifying the transformer block to
replace multi-head self-attention to reduce quadratic computational complexity
to linear with increasing sequence length. Input tokens are instead streamed in
sequentially to our attention mechanism (as with typical SNNs). Our preliminary
experiments show that SpikeGPT remains competitive with non-spiking models on
tested benchmarks, while maintaining 5× less energy consumption when processed
on neuromorphic hardware that can leverage sparse, event-driven activations. Our
code implementation is available at https://github.com/ridgerchu/SpikeGPT.
---

## Scaling Instruction-Finetuned Language Models

Hyung Won Chung, Le Hou, Shayne Longpre, Barret Zoph, Yi Tay, William Fedus, Yunxuan Li, Xuezhi Wang, Mostafa Dehghani, Siddhartha Brahma, Albert Webson, Shixiang Shane Gu, Zhuyun Dai, Mirac Suzgun, Xinyun Chen, Aakanksha Chowdhery, Alex Castro-Ros, Marie Pellat, Kevin Robinson, Dasha Valter, Sharan Narang, Gaurav Mishra, Adams Yu, Vincent Zhao, Yanping Huang, Andrew Dai, Hongkun Yu, Slav Petrov, Ed H. Chi, Jeff Dean, Jacob Devlin, Adam Roberts, Denny Zhou, Quoc V. Le, Jason Wei

Category: llms
Keywords: instruction finetuning, language models, scaling, generalization, chain-of-thought
Year: 2023

Finetuning language models on a collection of datasets phrased as instructions
has been shown to improve model performance and generalization to unseen tasks.
In this paper, we explore instruction finetuning with a particular focus on (1)
scaling the number of tasks, (2) scaling the model size, and (3) finetuning on
chain-of-thought data. We find that instruction finetuning with the above
aspects dramatically improves performance on a variety of model classes (PaLM,
T5, U-PaLM), prompting setups (zero-shot, few-shot, CoT), and evaluation
benchmarks (MMLU, BBH, TyDiQA, MGSM, open-ended generation,
RealToxicityPrompts). For instance, Flan-PaLM 540B instruction-finetuned on 1.8K
tasks outperforms PaLM 540B by a large margin (+9.4% on average). Flan-PaLM 540B
achieves state-of-the-art performance on several benchmarks, such as 75.2% on
five-shot MMLU. We also publicly release Flan-T5 checkpoints, which achieve
strong few-shot performance even compared to much larger models, such as PaLM
62B. Overall, instruction finetuning is a general method for improving the
performance and usability of pretrained language models.
---

## CodeCompose: A Large-Scale Industrial Deployment of AI-assisted Code Authoring

Vijayaraghavan Murali, Chandra Maddila, Imad Ahmad, Michael Bolin, Daniel Cheng, Negar Ghorbani, Renuka Fernandez, Nachiappan Nagappan

Category: llms
Keywords: AI-assisted code authoring, large language models, CodeCompose, software development, InCoder
Year: 2023

The rise of large language models (LLMs) has unlocked various applications of
this technology in software development. In particular, generative LLMs have
been shown to effectively power AI-based code authoring tools that can suggest
entire statements or blocks of code during code authoring. In this paper we
present CodeCompose, an AI-assisted code authoring tool developed and deployed
at Meta internally. CodeCompose is based on the InCoder LLM that merges
generative capabilities with bi-directionality. We have scaled up CodeCompose to
serve tens of thousands of developers at Meta, across 10+ programming languages
and several coding surfaces. We discuss unique challenges in terms of user
experience and metrics that arise when deploying such tools in large-scale
industrial settings. We present our experience in making design decisions about
the model and system architecture for CodeCompose that addresses these
challenges. Finally, we present metrics from our large-scale deployment of
CodeCompose that shows its impact on Meta’s internal code authoring experience
over a 15-day time window, where 4.5 million suggestions were made by
CodeCompose. Quantitative metrics reveal that (i) CodeCompose has an acceptance
rate of 22% across several languages, and (ii) 8% of the code typed by users of
CodeCompose is through accepting code suggestions from CodeCompose. Qualitative
feedback indicates an overwhelming 91.5% positive reception for CodeCompose. In
addition to assisting with code authoring, CodeCompose is also introducing other
positive side effects such as encouraging developers to generate more in-code
documentation, helping them with the discovery of new APIs, etc.
---

## FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance

Lingjiao Chen, Matei Zaharia, James Zou

Category: llms
Keywords: large language models, cost reduction, LLM cascade, FrugalGPT, query optimization
Year: 2023

There is a rapidly growing number of large language models (LLMs) that users can
query for a fee. We review the cost associated with querying popular LLM
APIs—e.g. GPT-4, ChatGPT, J1-Jumbo—and find that these models have heterogeneous
pricing structures, with fees that can differ by two orders of magnitude. In
particular, using LLMs on large collections of queries and text can be
expensive. Motivated by this, we outline and discuss three types of strategies
that users can exploit to reduce the inference cost associated with using LLMs:
1) prompt adaptation, 2) LLM approximation, and 3) LLM cascade. As an example,
we propose FrugalGPT, a simple yet flexible instantiation of LLM cascade which
learns which combinations of LLMs to use for different queries in order to
reduce cost and improve accuracy. Our experiments show that FrugalGPT can match
the performance of the best individual LLM (e.g. GPT-4) with up to 98% cost
reduction or improve the accuracy over GPT-4 by 4% with the same cost. The ideas
and findings presented here lay a foundation for using LLMs sustainably and
efficiently.
---

## LLMMaps - A Visual Metaphor for Stratified Evaluation of Large Language Models

Patrik Puchert, Poonam Poonam, Christian van Onzenoodt, Timo Ropinski

Category: llms
Keywords: Large language models, explainable artificial intelligence
Year: 2023

Large Language Models (LLMs) have revolutionized natural language processing and
demonstrated impressive capabilities in various tasks. Unfortunately, they are
prone to hallucinations, where the model exposes incorrect or false information
in its responses, which renders diligent evaluation approaches mandatory. While
LLM performance in specific knowledge fields is often evaluated based on
question and answer (Q&A) datasets, such evaluations usually report only a
single accuracy number for the entire field, a procedure which is problematic
with respect to transparency and model improvement. A stratified evaluation
could instead reveal subfields, where hallucinations are more likely to occur
and thus help to better assess LLMs’ risks and guide their further development.
To support such stratified evaluations, we propose LLMMaps as a novel
visualization technique that enables users to evaluate LLMs’ performance with
respect to Q&A datasets. LLMMaps provide detailed insights into LLMs’ knowledge
capabilities in different subfields, by transforming Q&A datasets as well as LLM
responses into our internal knowledge structure. An extension for comparative
visualization furthermore, allows for the detailed comparison of multiple LLMs.
To assess LLMMaps we use them to conduct a comparative analysis of several
state-of-the-art LLMs, such as BLOOM, GPT-2, GPT-3, ChatGPT and LLaMa-13B, as
well as two qualitative user evaluations. All necessary source code and data for
generating LLMMaps to be used in scientific publications and elsewhere will be
available on GitHub: https://github.com/******
---

## Toolformer: Language Models Can Teach Themselves to Use Tools

Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, Roberta Raileanu, Maria Lomeli, Luke Zettlemoyer, Nicola Cancedda, Thomas Scialom

Category: llms
Keywords: language models, toolformer, self-supervised learning, external tools, APIs
Year: 2023

Language models (LMs) exhibit remarkable abilities to solve new tasks from just
a few examples or textual instructions, especially at scale. They also,
paradoxically, struggle with basic functionality, such as arithmetic or factual
lookup, where much simpler and smaller models excel. In this paper, we show that
LMs can teach themselves to use external tools via simple APIs and achieve the
best of both worlds. We introduce Toolformer, a model trained to decide which
APIs to call, when to call them, what arguments to pass, and how to best
incorporate the results into future token prediction. This is done in a self-
supervised way, requiring nothing more than a handful of demonstrations for each
API. We incorporate a range of tools, including a calculator, a Q&A system, a
search engine, a translation system, and a calendar. Toolformer achieves
substantially improved zero-shot performance across a variety of downstream
tasks, often competitive with much larger models, without sacrificing its core
language modeling abilities.
---

## Language Is Not All You Need: Aligning Perception with Language Models

Shaohan Huang, Li Dong, Wenhui Wang, Yaru Hao, Saksham Singhal, Shuming Ma, Tengchao Lv, Lei Cui, Owais Khan Mohammed, Barun Patra, Qiang Liu, Kriti Aggarwal, Zewen Chi, Johan Bjorck, Vishrav Chaudhary, Subhojit Som, Xia Song, Furu Wei

Category: llms
Keywords: multimodal, large language model, vision, perception, language alignment
Year: 2023

KOSMOS-1 is a multimodal large language model (MLLM) that is capable of
perceiving multimodal input, following instructions, and performing in-context
learning for not only language tasks but also multimodal tasks. In this work, we
align vision with large language models (LLMs), advancing the trend of going
from LLMs to MLLMs.
---

## LLaMA: Open and Efficient Foundation Language Models

Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothee Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave, Guillaume Lample

Category: llms
Keywords: LLaMA, language models, open-source, efficient, GPT-3, benchmark
Year: 2023

We introduce LLaMA, a collection of foundation language models ranging from 7B
to 65B parameters. We train our models on trillions of tokens, and show that it
is possible to train state-of-the-art models using publicly available datasets
exclusively, without resorting to proprietary and inaccessible datasets. In
particular, LLaMA-13B outperforms GPT-3 (175B) on most benchmarks, and LLaMA-65B
is competitive with the best models, Chinchilla-70B and PaLM-540B. We release
all our models to the research community.
---

## Self-Edit: Fault-Aware Code Editor for Code Generation

Kechi Zhang, Zhuo Li, Jia Li, Ge Li, Zhi Jin

Category: llms
Keywords: large language models, code generation, fault-aware editing, competitive programming, self-edit
Year: 2023

Large language models (LLMs) have demonstrated an impressive ability to generate
codes on competitive programming tasks. However, with limited sample numbers,
LLMs still suffer from poor accuracy. Inspired by the process of human
programming, we propose a generate-and-edit approach named Self-Edit that
utilizes execution results of the generated code from LLMs to improve the code
quality on the competitive programming task. We execute the generated code on
the example test case provided in the question and wrap execution results into a
supplementary comment. Utilizing this comment as guidance, our fault-aware code
editor is employed to correct errors in the generated code. We perform extensive
evaluations across two competitive programming datasets with nine different
LLMs. Compared to directly generating from LLMs, our approach can improve the
average of pass@1 by 89% on APPS-dev, 31% on APPS-test, and 48% on HumanEval
over nine popular code generation LLMs with parameter sizes ranging from 110M to
175B. Compared to other post-processing methods, our method demonstrates
superior accuracy and efficiency.
---

## GLaM: Efficient Scaling of Language Models with Mixture-of-Experts

Nan Du, Yanping Huang, Andrew M. Dai, Simon Tong, Dmitry Lepikhin, Yuanzhong Xu, Maxim Krikun, Yanqi Zhou, Adams Wei Yu, Orhan Firat, Barret Zoph, Liam Fedus, Maarten Bosma, Zongwei Zhou, Tao Wang, Yu Emma Wang, Kellie Webster, Marie Pellat, Kevin Robinson, Kathleen Meier-Hellstern, Toju Duke, Lucas Dixon, Kun Zhang, Quoc V Le, Yonghui Wu, Zhifeng Chen, Claire Cui

Category: llms
Keywords: language models, mixture-of-experts, scaling, GLaM, NLP
Year: 2022

Scaling language models with more data, compute and parameters has driven
significant progress in natural language processing. For example, thanks to
scaling, GPT-3 was able to achieve strong results on in-context learning tasks.
However, training these large dense models requires significant amounts of
computing resources. In this paper, we propose and develop a family of language
models named GLaM (Generalist Language Model), which uses a sparsely activated
mixture-of-experts architecture to scale the model capacity while also incurring
substantially less training cost compared to dense variants. The largest GLaM
has 1.2 trillion parameters, which is approximately 7x larger than GPT-3. It
consumes only 1/3 of the energy used to train GPT-3 and requires half of the
computation flops for inference, while still achieving better overall zero, one
and few-shot performance across 29 NLP tasks.
---

## Unifying Language Learning Paradigms

Yi Tay, Mostafa Dehghani, Vinh Q. Tran, Xavier Garcia, Dara Bahri, Tal Schuster, Huaixiu Steven Zheng, Neil Houlsby, Donald Metzler

Category: llms
Keywords: language learning, pre-training, Mixture-of-Denoisers, self-supervision, NLP, in-context learning
Year: 2022

Existing pre-trained models are generally geared towards a particular class of
problems. To date, there seems to be still no consensus on what the right
architecture and pre-training setup should be. This paper presents a unified
framework for pre-training models that are universally effective across datasets
and setups. We begin by disentangling architectural archetypes with pre-training
objectives – two concepts that are commonly conflated. Next, we present a
generalized and unified perspective for self-supervision in NLP and show how
different pre-training objectives can be cast as one another and how
interpolating between different objectives can be effective. We then propose
Mixture-of-Denoisers (MoD), a pre-training objective that combines diverse pre-
training paradigms together. We furthermore introduce a notion of mode
switching, wherein downstream fine-tuning is associated with specific pre-
training schemes. We conduct extensive ablative experiments to compare multiple
pre-training objectives and find that our method pushes the Pareto-frontier by
outperforming T5 and/or GPT-like models across multiple diverse setups. Finally,
by scaling our model up to 20B parameters, we achieve SOTA performance on 50
well-established supervised NLP tasks ranging from language generation (with
automated and human evaluation), language understanding, text classification,
question answering, commonsense reasoning, long text reasoning, structured
knowledge grounding and information retrieval. Our model also achieves strong
results at in-context learning, outperforming 175B GPT-3 on zero-shot SuperGLUE
and tripling the performance of T5-XXL on one-shot summarization.
---

## Meta-learning via Language Model In-context Tuning

Yanda Chen, Ruiqi Zhong, Sheng Zha, George Karypis, He He

Category: llms
Keywords: meta-learning, language models, in-context tuning, few-shot learning, text classification
Year: 2022

The goal of meta-learning is to learn to adapt to a new task with only a few
labeled examples. Inspired by the recent progress in large language models, we
propose in-context tuning (ICT), which recasts task adaptation and prediction as
a simple sequence prediction problem: to form the input sequence, we concatenate
the task instruction, labeled in-context examples, and the target input to
predict; to meta-train the model to learn from in-context examples, we fine-tune
a pre-trained language model (LM) to predict the target label given the input
sequence on a collection of tasks. We benchmark our method on two collections of
text classification tasks: LAMA and BinaryClfs. Compared to MAML which adapts
the model through gradient descent, our method leverages the inductive bias of
pre-trained LMs to perform pattern matching, and outperforms MAML by an absolute
6% average AUC-ROC score on BinaryClfs, gaining more advantage with increasing
model size. Compared to non-fine-tuned in-context learning (i.e., prompting a
raw LM), in-context tuning meta-trains the model to learn from in-context
examples. On BinaryClfs, ICT improves the average AUC-ROC score by an absolute
10%, and reduces the variance due to example ordering by 6x and example choices
by 2x.
---

## Holistic Evaluation of Language Models

Percy Liang, Rishi Bommasani, Tony Lee, Dimitris Tsipras, Dilara Soylu, Michihiro Yasunaga, Yian Zhang, Deepak Narayanan, Yuhuai Wu, Ananya Kumar, Benjamin Newman, Binhang Yuan, Bobby Yan, Ce Zhang, Christian Cosgrove, Christopher D. Manning, Christopher Ré, Diana Acosta-Navas, Drew A. Hudson, Eric Zelikman, Esin Durmus, Faisal Ladhak, Frieda Rong, Hongyu Ren, Huaxiu Yao, Jue Wang, Keshav Santhanam, Laurel Orr, Lucia Zheng, Mert Yuksekgonul, Mirac Suzgun, Nathan Kim, Neel Guha, Niladri Chatterji, Omar Khattab, Peter Henderson, Qian Huang, Ryan Chi, Sang Michael Xie, Shibani Santurkar, Surya Ganguli, Tatsunori Hashimoto, Thomas Icard, Tianyi Zhang, Vishrav Chaudhary, William Wang, Xuechen Li, Yifan Mai, Yuhui Zhang, Yuta Koreeda

Category: llms
Keywords: language models, evaluation, transparency, metrics, benchmark
Year: 2022

Language models (LMs) are becoming the foundation for almost all major language
technologies, but their capabilities, limitations, and risks are not well
understood. We present Holistic Evaluation of Language Models (HELM) to improve
the transparency of language models. First, we taxonomize the vast space of
potential scenarios (i.e. use cases) and metrics (i.e. desiderata) that are of
interest for LMs. Then we select a broad subset based on coverage and
feasibility, noting what’s missing or underrepresented (e.g. question answering
for neglected English dialects, metrics for trustworthiness). Second, we adopt a
multi-metric approach: We measure 7 metrics (accuracy, calibration, robustness,
fairness, bias, toxicity, and efficiency) for each of 16 core scenarios to the
extent possible (87.5% of the time), ensuring that metrics beyond accuracy don’t
fall to the wayside, and that trade-offs across models and metrics are clearly
exposed. We also perform 7 targeted evaluations, based on 26 targeted scenarios,
to more deeply analyze specific aspects (e.g. knowledge, reasoning,
memorization/copyright, disinformation). Third, we conduct a large-scale
evaluation of 30 prominent language models (spanning open, limited-access, and
closed models) on all 42 scenarios, including 21 scenarios that were not
previously used in mainstream LM evaluation. Prior to HELM, models on average
were evaluated on just 17.9% of the core HELM scenarios, with some prominent
models not sharing a single scenario in common. We improve this to 96.0%: now
all 30 models have been densely benchmarked on a set of core scenarios and
metrics under standardized conditions. Our evaluation surfaces 25 top-level
findings concerning the interplay between different scenarios, metrics, and
models. For full transparency, we release all raw model prompts and completions
publicly for further analysis, as well as a general modular toolkit for easily
adding new scenarios, models, metrics, and prompting strategies. We intend for
HELM to be a living benchmark for the community, continuously updated with new
scenarios, metrics, and models.
---

## PaLM: Scaling Language Modeling with Pathways

Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, Parker Schuh, Kensen Shi, Sasha Tsvyashchenko, Joshua Maynez, Abhishek Rao, Parker Barnes, Yi Tay, Noam Shazeer, Vinodkumar Prabhakaran, Emily Reif, Nan Du, Ben Hutchinson, Reiner Pope, James Bradbury, Jacob Austin, Michael Isard, Guy Gur-Ari, Pengcheng Yin, Toju Duke, Anselm Levskaya, Sanjay Ghemawat, Sunipa Dev, Henryk Michalewski, Xavier Garcia, Vedant Misra, Kevin Robinson, Liam Fedus, Denny Zhou, Daphne Ippolito, David Luan, Hyeontaek Lim, Barret Zoph, Alexander Spiridonov, Ryan Sepassi, David Dohan, Shivani Agrawal, Mark Omernick, Andrew M. Dai, Thanumalayan Sankaranarayana Pillai, Marie Pellat, Aitor Lewkowycz, Erica Moreira, Rewon Child, Oleksandr Polozov, Katherine Lee, Zongwei Zhou, Xuezhi Wang, Brennan Saeta, Mark Diaz, Orhan Firat, Michele Catasta, Jason Wei, Kathy Meier-Hellstern, Douglas Eck, Jeff Dean, Slav Petrov, Noah Fiedel

Category: llms
Keywords: large language models, PaLM, Transformer, scaling, few-shot learning, multilingual tasks, source code generation, bias, toxicity, ethical considerations
Year: 2022

Large language models have been shown to achieve remarkable performance across a
variety of natural language tasks using few-shot learning, which drastically
reduces the number of task-specific training examples needed to adapt the model
to a particular application. To further our understanding of the impact of scale
on few-shot learning, we trained a 540-billion parameter, densely activated,
Transformer language model, which we call Pathways Language Model (PaLM). We
trained PaLM on 6144 TPU v4 chips using Pathways, a new ML system which enables
highly efficient training across multiple TPU Pods. We demonstrate continued
benefits of scaling by achieving state-of-the-art few-shot learning results on
hundreds of language understanding and generation benchmarks. On a number of
these tasks, PaLM 540B achieves breakthrough performance, outperforming the
fine-tuned state-of-the-art on a suite of multi-step reasoning tasks, and
outperforming average human performance on the recently released BIG-bench
benchmark. A significant number of BIG-bench tasks showed discontinuous
improvements from model scale, meaning that performance steeply increased as we
scaled to our largest model. PaLM also has strong capabilities in multilingual
tasks and source code generation, which we demonstrate on a wide array of
benchmarks. We additionally provide a comprehensive analysis on bias and
toxicity, and study the extent of training data memorization with respect to
model scale. Finally, we discuss the ethical considerations related to large
language models and discuss potential mitigation strategies.
---

## Training language models to follow instructions with human feedback

Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul Christiano, Jan Leike, Ryan Lowe

Category: llms
Keywords: language models, instruction following, human feedback, fine-tuning, InstructGPT
Year: 2022

Making language models bigger does not inherently make them better at following
a user’s intent. For example, large language models can generate outputs that
are untruthful, toxic, or simply not helpful to the user. In other words, these
models are not aligned with their users. In this paper, we show an avenue for
aligning language models with user intent on a wide range of tasks by fine-
tuning with human feedback. Starting with a set of labeler-written prompts and
prompts submitted through the OpenAI API, we collect a dataset of labeler
demonstrations of the desired model behavior, which we use to fine-tune GPT-3
using supervised learning. We then collect a dataset of rankings of model
outputs, which we use to further fine-tune this supervised model using
reinforcement learning from human feedback. We call the resulting models
InstructGPT. In human evaluations on our prompt distribution, outputs from the
1.3B parameter InstructGPT model are preferred to outputs from the 175B GPT-3,
despite having 100x fewer parameters. Moreover, InstructGPT models show
improvements in truthfulness and reductions in toxic output generation while
having minimal performance regressions on public NLP datasets. Even though
InstructGPT still makes simple mistakes, our results show that fine-tuning with
human feedback is a promising direction for aligning language models with human
intent.
---

## Beyond the Imitation Game: Quantifying and Extrapolating the Capabilities of Language Models

Aarohi Srivastava, Abhinav Rastogi, Abhishek Rao, Abu Awal Md Shoeb, Abubakar Abid, Adam Fisch, Adam R. Brown, Adam Santoro, Aditya Gupta, Adrià Garriga-Alonso, Agnieszka Kluska, Aitor Lewkowycz, Akshat Agarwal, Alethea Power, Alex Ray, Alex Warstadt, Alexander W. Kocurek, Ali Safaya, Ali Tazarv, Alice Xiang, Alicia Parrish, Allen Nie, Aman Hussain, Amanda Askell, Amanda Dsouza, Ambrose Slone, Ameet Rahane, Anantharaman S. Iyer, Anders Andreassen, Andrea Madotto, Andrea Santilli, Andreas Stuhlmüller, Andrew Dai, Andrew La, Andrew Lampinen, Andy Zou, Angela Jiang, Angelica Chen, Anh Vuong, Animesh Gupta, Anna Gottardi, Antonio Norelli, Anu Venkatesh, Arash Gholamidavoodi, Arfa Tabassum, Arul Menezes, Arun Kirubarajan, Asher Mullokandov, Ashish Sabharwal, Austin Herrick, Avia Efrat, Aykut Erdem, Ayla Karakaş, B. Ryan Roberts, Bao Sheng Loe, Barret Zoph, Bartłomiej Bojanowski, Batuhan Özyurt, Behnam Hedayatnia, Behnam Neyshabur, Benjamin Inden, Benno Stein, Berk Ekmekci, Bill Yuchen Lin, Blake Howald, Cameron Diao, Cameron Dour, Catherine Stinson, Cedrick Argueta, César Ferri Ramírez, Chandan Singh, Charles Rathkopf, Chenlin Meng, Chitta Baral, Chiyu Wu, Chris Callison-Burch, Chris Waites, Christian Voigt, Christopher D. Manning, Christopher Potts, Cindy Ramirez, Clara E. Rivera, Clemencia Siro, Colin Raffel, Courtney Ashcraft, Cristina Garbacea, Damien Sileo, Dan Garrette, Dan Hendrycks, Dan Kilman, Dan Roth, Daniel Freeman, Daniel Khashabi, Daniel Levy, Daniel Moseguí González, Danielle Perszyk, Danny Hernandez, Danqi Chen, Daphne Ippolito, Dar Gilboa, David Dohan, David Drakard, David Jurgens, Debajyoti Datta, Deep Ganguli, Denis Emelin, Denis Kleyko, Deniz Yuret, Derek Chen, Derek Tam, Dieuwke Hupkes, Diganta Misra, Dilyar Buzan, Dimitri Coelho Mollo, Diyi Yang, Dong-Ho Lee, Ekaterina Shutova, Ekin Dogus Cubuk, Elad Segal, Eleanor Hagerman, Elizabeth Barnes, Elizabeth Donoway, Ellie Pavlick, Emanuele Rodola, Emma Lam, Eric Chu, Eric Tang, Erkut Erdem, Ernie Chang, Ethan A. Chi, Ethan Dyer, Ethan Jerzak, Ethan Kim, Eunice Engefu Manyasi, Evgenii Zheltonozhskii, Fanyue Xia, Fatemeh Siar, Fernando Martínez-Plumed, Francesca Happé, Francois Chollet, Frieda Rong, Gaurav Mishra, Genta Indra Winata, Gerard de Melo, Germán Kruszewski, Giambattista Parascandolo, Giorgio Mariani, Gloria Wang, Gonzalo Jaimovitch-López, Gregor Betz, Guy Gur-Ari, Hana Galijasevic, Hannah Kim, Hannah Rashkin, Hannaneh Hajishirzi, Harsh Mehta, Hayden Bogar, Henry Shevlin, Hinrich Schütze, Hiromu Yakura, Hongming Zhang, Hugh Mee Wong, Ian Ng, Isaac Noble, Jaap Jumelet, Jack Geissinger, Jackson Kernion, Jacob Hilton, Jaehoon Lee, Jaime Fernández Fisac, James B. Simon, James Koppel, James Zheng, James Zou, Jan Kocoń, Jana Thompson, Jared Kaplan, Jarema Radom, Jascha Sohl-Dickstein, Jason Phang, Jason Wei, Jason Yosinski, Jekaterina Novikova, Jelle Bosscher, Jennifer Marsh, Jeremy Kim, Jeroen Taal, Jesse Engel, Jesujoba Alabi, Jiacheng Xu, Jiaming Song, Jillian Tang, Joan Waweru, John Burden, John Miller, John U. Balis, Jonathan Berant, Jörg Frohberg, Jos Rozen, Jose Hernandez-Orallo, Joseph Boudeman, Joseph Jones, Joshua B. Tenenbaum, Joshua S. Rule, Joyce Chua, Kamil Kanclerz, Karen Livescu, Karl Krauth, Karthik Gopalakrishnan, Katerina Ignatyeva, Katja Markert, Kaustubh D. Dhole, Kevin Gimpel, Kevin Omondi, Kory Mathewson, Kristen Chiafullo, Ksenia Shkaruta, Kumar Shridhar, Kyle McDonell, Kyle Richardson, Laria Reynolds, Leo Gao, Li Zhang, Liam Dugan, Lianhui Qin, Lidia Contreras-Ochando, Louis-Philippe Morency, Luca Moschella, Lucas Lam, Lucy Noble, Ludwig Schmidt, Luheng He, Luis Oliveros Colón, Luke Metz, Lütfi Kerem Şenel, Maarten Bosma, Maarten Sap, Maartje ter Hoeve, Maheen Farooqi, Manaal Faruqui, Mantas Mazeika, Marco Baturan, Marco Marelli, Marco Maru, Maria Jose Ramírez Quintana, Marie Tolkiehn, Mario Giulianelli, Martha Lewis, Martin Potthast, Matthew L. Leavitt, Matthias Hagen, Mátyás Schubert, Medina Orduna Baitemirova, Melody Arnaud, Melvin McElrath, Michael A. Yee, Michael Cohen, Michael Gu, Michael Ivanitskiy, Michael Starritt, Michael Strube, Michał Swędrowski, Michele Bevilacqua, Michihiro Yasunaga, Mihir Kale, Mike Cain, Mimee Xu, Mirac Suzgun, Mo Tiwari, Mohit Bansal, Moin Aminnaseri, Mor Geva, Mozhdeh Gheini, Mukund Varma T, Nanyun Peng, Nathan Chi, Nayeon Lee, Neta Gur-Ari Krakover, Nicholas Cameron, Nicholas Roberts, Nick Doiron, Nikita Nangia, Niklas Deckers, Niklas Muennighoff, Nitish Shirish Keskar, Niveditha S. Iyer, Noah Constant, Noah Fiedel, Nuan Wen, Oliver Zhang, Omar Agha, Omar Elbaghdadi, Omer Levy, Owain Evans, Pablo Antonio Moreno Casares, Parth Doshi, Pascale Fung, Paul Pu Liang, Paul Vicol, Pegah Alipoormolabashi, Peiyuan Liao, Percy Liang, Peter Chang, Peter Eckersley, Phu Mon Htut, Pinyu Hwang, Piotr Miłkowski, Piyush Patil, Pouya Pezeshkpour, Priti Oli, Qiaozhu Mei, Qing Lyu, Qinlang Chen, Rabin Banjade, Rachel Etta Rudolph, Raefer Gabriel, Rahel Habacker, Ramón Risco Delgado, Raphaël Millière, Rhythm Garg, Richard Barnes, Rif A. Saurous, Riku Arakawa, Robbe Raymaekers, Robert Frank, Rohan Sikand, Roman Novak, Roman Sitelew, Ronan LeBras, Rosanne Liu, Rowan Jacobs, Rui Zhang, Ruslan Salakhutdinov, Ryan Chi, Ryan Lee, Ryan Stovall, Ryan Teehan, Rylan Yang, Sahib Singh, Saif M. Mohammad, Sajant Anand, Sam Dillavou, Sam Shleifer, Sam Wiseman, Samuel Gruetter, Samuel R. Bowman, Samuel S. Schoenholz, Sanghyun Han, Sanjeev Kwatra, Sarah A. Rous, Sarik Ghazarian, Sayan Ghosh, Sean Casey, Sebastian Bischoff, Sebastian Gehrmann, Sebastian Schuster, Sepideh Sadeghi, Shadi Hamdan, Sharon Zhou, Shashank Srivastava, Sherry Shi, Shikhar Singh, Shima Asaadi, Shixiang Shane Gu, Shubh Pachchigar, Shubham Toshniwal, Shyam Upadhyay, Shyamolima (Shammie) Debnath, Siamak Shakeri, Simon Thormeyer, Simone Melzi, Siva Reddy, Sneha Priscilla Makini, Soo-Hwan Lee, Spencer Torene, Sriharsha Hatwar, Stanislas Dehaene, Stefan Divic, Stefano Ermon, Stella Biderman, Stephanie Lin, Stephen Prasad, Steven T. Piantadosi, Stuart M. Shieber, Summer Misherghi, Svetlana Kiritchenko, Swaroop Mishra, Tal Linzen, Tal Schuster, Tao Li, Tao Yu, Tariq Ali, Tatsu Hashimoto, Te-Lin Wu, Théo Desbordes, Theodore Rothschild, Thomas Phan, Tianle Wang, Tiberius Nkinyili, Timo Schick, Timofei Kornev, Timothy Telleen-Lawton, Titus Tunduny, Tobias Gerstenberg, Trenton Chang, Trishala Neeraj, Tushar Khot, Tyler Shultz, Uri Shaham, Vedant Misra, Vera Demberg, Victoria Nyamai, Vikas Raunak, Vinay Ramasesh, Vinay Uday Prabhu, Vishakh Padmakumar, Vivek Srikumar, William Fedus, William Saunders, William Zhang, Wout Vossen, Xiang Ren, Xiaoyu Tong, Xinran Zhao, Xinyi Wu, Xudong Shen, Yadollah Yaghoobzadeh, Yair Lakretz, Yangqiu Song, Yasaman Bahri, Yejin Choi, Yichi Yang, Yiding Hao, Yifu Chen, Yonatan Belinkov, Yu Hou, Yufang Hou, Yuntao Bai, Zachary Seid, Zhuoye Zhao, Zijian Wang, Zijie J. Wang, Zirui Wang, Ziyi Wu

Category: llms
Keywords: language models, benchmark, BIG-bench, capabilities, evaluation
Year: 2022

The Imitation Game benchmark (BIG-bench) is a large-scale collaboration aimed at
evaluating the capabilities of language models in a wide range of tasks. This
document outlines the breadth and depth of tasks included in BIG-bench and
discusses the potential of extrapolating the capabilities of language models
beyond current benchmarks.
---

## LaMDA: Language Models for Dialog Applications

Romal Thoppilan, Daniel De Freitas, Jamie Hall, Noam Shazeer, Apoorv Kulshreshtha, Heng-Tze Cheng, Alicia Jin, Taylor Bos, Leslie Baker, Yu Du, YaGuang Li, Hongrae Lee, Huaixiu Steven Zheng, Amin Ghafouri, Marcelo Menegali, Yanping Huang, Maxim Krikun, Dmitry Lepikhin, James Qin, Dehao Chen, Yuanzhong Xu, Zhifeng Chen, Adam Roberts, Maarten Bosma, Vincent Zhao, Yanqi Zhou, Chung-Ching Chang, Igor Krivokon, Will Rusch, Marc Pickett, Pranesh Srinivasan, Laichee Man, Kathleen Meier-Hellstern, Meredith Ringel Morris, Tulsee Doshi, Renelito Delos Santos, Toju Duke, Johnny Soraker, Ben Zevenbergen, Vinodkumar Prabhakaran, Mark Diaz, Ben Hutchinson, Kristen Olson, Alejandra Molina, Erin Hoffman-John, Josh Lee, Lora Aroyo, Ravi Rajakumar, Alena Butryna, Matthew Lamm, Viktoriya Kuzmina, Joe Fenton, Aaron Cohen, Rachel Bernstein, Ray Kurzweil, Blaise Aguera-Arcas, Claire Cui, Marian Croak, Ed Chi, Quoc Le

Category: llms
Keywords: LaMDA, dialog applications, language models, transformer, safety, factual grounding
Year: 2022

We present LaMDA: Language Models for Dialog Applications. LaMDA is a family of
Transformer-based neural language models specialized for dialog, which have up
to 137B parameters and are pre-trained on 1.56T words of public dialog data and
web text. While model scaling alone can improve quality, it shows less
improvements on safety and factual grounding. We demonstrate that fine-tuning
with annotated data and enabling the model to consult external knowledge sources
can lead to significant improvements towards the two key challenges of safety
and factual grounding. The first challenge, safety, involves ensuring that the
model’s responses are consistent with a set of human values, such as preventing
harmful suggestions and unfair bias. We quantify safety using a metric based on
an illustrative set of human values, and we find that filtering candidate
responses using a LaMDA classifier fine-tuned with a small amount of
crowdworker-annotated data offers a promising approach to improving model
safety. The second challenge, factual grounding, involves enabling the model to
consult external knowledge sources, such as an information retrieval system, a
language translator, and a calculator. We quantify factuality using a
groundedness metric, and we find that our approach enables the model to generate
responses grounded in known sources, rather than responses that merely sound
plausible. Finally, we explore the use of LaMDA in the domains of education and
content recommendations, and analyze their helpfulness and role consistency.
---

## Using DeepSpeed and Megatron to Train Megatron-Turing NLG 530B, A Large-Scale Generative Language Model

Shaden Smith, Mostofa Patwary, Brandon Norick, Patrick LeGresley, Samyam Rajbhandari, Jared Casper, Zhun Liu, Shrimai Prabhumoye, George Zerveas, Vijay Korthikanti, Elton Zhang, Rewon Child, Reza Yazdani Aminabadi, Julie Bernauer, Xia Song, Mohammad Shoeybi, Yuxiong He, Michael Houston, Saurabh Tiwary, Bryan Catanzaro

Category: llms
Keywords: DeepSpeed, Megatron, Megatron-Turing NLG, large-scale language models, natural language processing
Year: 2022

Pretrained general-purpose language models can achieve state-of-the-art
accuracies in various natural language processing domains by adapting to
downstream tasks via zero-shot, few-shot and fine-tuning techniques. Because of
their success, the size of these models has increased rapidly, requiring high-
performance hardware, software, and algorithmic techniques to enable training
such large models. As the result of a joint effort between Microsoft and NVIDIA,
we present details on the training of the largest monolithic transformer based
language model, Megatron-Turing NLG 530B (MT-NLG), with 530 billion parameters.
In this paper, we first focus on the infrastructure as well as the 3D
parallelism methodology used to train this model using DeepSpeed and Megatron.
Next, we detail the training process, the design of our training corpus, and our
data curation techniques, which we believe is a key ingredient to the success of
the model. Finally, we discuss various evaluation results, as well as other
interesting observations and new properties exhibited by MT-NLG. We demonstrate
that MT-NLG achieves superior zero-, one-, and few-shot learning accuracies on
several NLP benchmarks and establishes new state-of-the-art results. We believe
that our contributions will help further the development of large-scale training
infrastructures, large-scale language models, and natural language generations.
---

## Improving language models by retrieving from trillions of tokens

Sebastian Borgeaud, Arthur Mensch, Jordan Hoffmann, Trevor Cai, Eliza Rutherford, Katie Millican, George van den Driessche, Jean-Baptiste Lespiau, Bogdan Damoc, Aidan Clark, Diego de Las Casas, Aurelia Guy, Jacob Menick, Roman Ring, Tom Hennigan, Saffron Huang, Loren Maggiore, Chris Jones, Albin Cassirer, Andy Brock, Michela Paganini, Geoffrey Irving, Oriol Vinyals, Simon Osindero, Karen Simonyan, Jack W. Rae, Erich Elsen, Laurent Sifre

Category: llms
Keywords: language models, retrieval, transformer, trillions of tokens, deep learning
Year: 2022

We enhance auto-regressive language models by conditioning on document chunks
retrieved from a large corpus, based on local similarity with preceding tokens.
With a 2 trillion token database, our Retrieval-Enhanced Transformer (Retro)
obtains comparable performance to GPT-3 and Jurassic-1 on the Pile, despite
using 25× fewer parameters. After fine-tuning, Retro performance translates to
downstream knowledge-intensive tasks such as question answering. Retro combines
a frozen Bert retriever, a differentiable encoder and a chunked cross-attention
mechanism to predict tokens based on an order of magnitude more data than what
is typically consumed during training. We typically train Retro from scratch,
yet can also rapidly Retrofit pre-trained transformers with retrieval and still
achieve good performance. Our work opens up new avenues for improving language
models through explicit memory at unprecedented scale.
---

## Galactica: A Large Language Model for Science

Ross Taylor, Marcin Kardas, Guillem Cucurull, Thomas Scialom, Anthony Hartshorn, Elvis Saravia, Andrew Poulton, Viktor Kerkez, Robert Stojnic

Category: llms
Keywords: large language model, scientific knowledge, information overload, technical knowledge probes, reasoning, state-of-the-art, open source
Year: 2022

Information overload is a major obstacle to scientific progress. The explosive
growth in scientific literature and data has made it ever harder to discover
useful insights in a large mass of information. Today scientific knowledge is
accessed through search engines, but they are unable to organize scientific
knowledge alone. In this paper we introduce Galactica: a large language model
that can store, combine and reason about scientific knowledge. We train on a
large scientific corpus of papers, reference material, knowledge bases and many
other sources. We outperform existing models on a range of scientific tasks. On
technical knowledge probes such as LaTeX equations, Galactica outperforms the
latest GPT-3 by 68.2% versus 49.0%. Galactica also performs well on reasoning,
outperforming Chinchilla on mathematical MMLU by 41.3% to 35.7%, and PaLM 540B
on MATH with a score of 20.4% versus 8.8%. It also sets a new state-of-the-art
on downstream tasks such as PubMedQA and MedMCQA dev of 77.6% and 52.9%. And
despite not being trained on a general corpus, Galactica outperforms BLOOM and
OPT-175B on BIG-bench. We believe these results demonstrate the potential for
language models as a new interface for science. We open source the model for the
benefit of the scientific community.
---

## Self-Diagnosis and Self-Debiasing: A Proposal for Reducing Corpus-Based Bias in NLP

Timo Schick, Sahana Udupa, Hinrich Schütze

Category: llms
Keywords: bias, language models, NLP, debiasing, self-diagnosis
Year: 2021

When trained on large, unfiltered crawls from the internet, language models pick
up and reproduce all kinds of undesirable biases that can be found in the data:
they often generate racist, sexist, violent or otherwise toxic language. As
large models often require millions of training examples to achieve good
performance, it is difficult to completely prevent them from being exposed to
such content. In this paper, we investigate whether pretrained language models
at least know when they exhibit some undesirable bias or produce toxic content.
Based on our findings, we propose a decoding algorithm that reduces the
probability of a model producing problematic text given only a textual
description of the undesired behavior. This algorithm does not rely on manually
curated word lists, nor does it require any training data or changes to the
model’s parameters. While our approach does by no means eliminate the issue of
language models generating biased text, we believe it to be an important step in
this direction.
---

## Evaluating Large Language Models Trained on Code

Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, Alex Ray, Raul Puri, Gretchen Krueger, Michael Petrov, Heidy Khlaaf, Girish Sastry, Pamela Mishkin, Brooke Chan, Scott Gray, Nick Ryder, Mikhail Pavlov, Alethea Power, Lukasz Kaiser, Mohammad Bavarian, Clemens Winter, Philippe Tillet, Felipe Petroski Such, Dave Cummings, Matthias Plappert, Fotios Chantzis, Elizabeth Barnes, Ariel Herbert-Voss, William Hebgen Guss, Alex Nichol, Alex Paino, Nikolas Tezak, Jie Tang, Igor Babuschkin, Suchir Balaji, Shantanu Jain, William Saunders, Christopher Hesse, Andrew N. Carr, Jan Leike, Josh Achiam, Vedant Misra, Evan Morikawa, Alec Radford, Matthew Knight, Miles Brundage, Mira Murati, Katie Mayer, Peter Welinder, Bob McGrew, Dario Amodei, Sam McCandlish, Ilya Sutskever, Wojciech Zaremba

Category: llms
Keywords: Codex, GPT, language models, code generation, GitHub Copilot, program synthesis, Python, HumanEval
Year: 2021

We introduce Codex, a GPT language model fine-tuned on publicly available code
from GitHub, and study its Python code-writing capabilities. A distinct
production version of Codex powers GitHub Copilot. On HumanEval, a new
evaluation set we release to measure functional correctness for synthesizing
programs from docstrings, our model solves 28.8% of the problems, while GPT-3
solves 0% and GPT-J solves 11.4%. Furthermore, we find that repeated sampling
from the model is a surprisingly effective strategy for producing working
solutions to difficult prompts. Using this method, we solve 70.2% of our
problems with 100 samples per problem. Careful investigation of our model
reveals its limitations, including difficulty with docstrings describing long
chains of operations and with binding operations to variables. Finally, we
discuss the potential broader impacts of deploying powerful code generation
technologies, covering safety, security, and economics.
---

## Dynamic Language Models for Continuously Evolving Content

Spurthi Amba Hombaiah, Tao Chen, Mingyang Zhang, Michael Bendersky, Marc Najork

Category: llms
Keywords: Active Learning, Dynamic Vocabulary, Hard Example Mining, Incremental Learning, Language Modeling, Vocabulary Composition
Year: 2021

The content on the web is in a constant state of flux. New entities, issues, and
ideas continuously emerge, while the semantics of the existing conversation
topics gradually shift. In recent years, pre-trained language models like BERT
greatly improved the state-of-the-art for a large spectrum of content
understanding tasks. Therefore, in this paper, we aim to study how these
language models can be adapted to better handle continuously evolving web
content. In our study, we first analyze the evolution of 2013 – 2019 Twitter
data, and unequivocally confirm that a BERT model trained on past tweets would
heavily deteriorate when directly applied to data from later years. Then, we
investigate two possible sources of the deterioration: the semantic shift of
existing tokens and the sub-optimal or failed understanding of new tokens. To
this end, we both explore two different vocabulary composition methods, as well
as propose three sampling methods which help in efficient incremental training
for BERT-like models. Compared to a new model trained from scratch offline, our
incremental training (a) reduces the training costs, (b) achieves better
performance on evolving content, and (c) is suitable for online deployment. The
superiority of our methods is validated using two downstream tasks. We
demonstrate significant improvements when incrementally evolving the model from
a particular base year, on the task of Country Hashtag Prediction, as well as on
the OffensEval 2019 task.
---

## Evaluating Large Language Models Trained on Code

Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, Alex Ray, Raul Puri, Gretchen Krueger, Michael Petrov, Heidy Khlaaf, Girish Sastry, Pamela Mishkin, Brooke Chan, Scott Gray, Nick Ryder, Mikhail Pavlov, Alethea Power, Lukasz Kaiser, Mohammad Bavarian, Clemens Winter, Philippe Tillet, Felipe Petroski Such, Dave Cummings, Matthias Plappert, Fotios Chantzis, Elizabeth Barnes, Ariel Herbert-Voss, William Hebgen Guss, Alex Nichol, Alex Paino, Nikolas Tezak, Jie Tang, Igor Babuschkin, Suchir Balaji, Shantanu Jain, William Saunders, Christopher Hesse, Andrew N. Carr, Jan Leike, Josh Achiam, Vedant Misra, Evan Morikawa, Alec Radford, Matthew Knight, Miles Brundage, Mira Murati, Katie Mayer, Peter Welinder, Bob McGrew, Dario Amodei, Sam McCandlish, Ilya Sutskever, Wojciech Zaremba

Category: llms
Keywords: Codex, language models, code generation, program synthesis, GitHub Copilot
Year: 2021

We introduce Codex, a GPT language model fine-tuned on publicly available code
from GitHub, and study its Python code-writing capabilities. A distinct
production version of Codex powers GitHub Copilot. On HumanEval, a new
evaluation set we release to measure functional correctness for synthesizing
programs from docstrings, our model solves 28.8% of the problems, while GPT-3
solves 0% and GPT-J solves 11.4%. Furthermore, we find that repeated sampling
from the model is a surprisingly effective strategy for producing working
solutions to difficult prompts. Using this method, we solve 70.2% of our
problems with 100 samples per problem. Careful investigation of our model
reveals its limitations, including difficulty with docstrings describing long
chains of operations and with binding operations to variables. Finally, we
discuss the potential broader impacts of deploying powerful code generation
technologies, covering safety, security, and economics.
---

## Evaluating Large Language Models Trained on Code

Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, Alex Ray, Raul Puri, Gretchen Krueger, Michael Petrov, Heidy Khlaaf, Girish Sastry, Pamela Mishkin, Brooke Chan, Scott Gray, Nick Ryder, Mikhail Pavlov, Alethea Power, Lukasz Kaiser, Mohammad Bavarian, Clemens Winter, Philippe Tillet, Felipe Petroski Such, Dave Cummings, Matthias Plappert, Fotios Chantzis, Elizabeth Barnes, Ariel Herbert-Voss, William Hebgen Guss, Alex Nichol, Alex Paino, Nikolas Tezak, Jie Tang, Igor Babuschkin, Suchir Balaji, Shantanu Jain, William Saunders, Christopher Hesse, Andrew N. Carr, Jan Leike, Josh Achiam, Vedant Misra, Evan Morikawa, Alec Radford, Matthew Knight, Miles Brundage, Mira Murati, Katie Mayer, Peter Welinder, Bob McGrew, Dario Amodei, Sam McCandlish, Ilya Sutskever, Wojciech Zaremba

Category: llms
Keywords: Codex, GPT, language model, code generation, GitHub Copilot, HumanEval
Year: 2021

We introduce Codex, a GPT language model fine-tuned on publicly available code
from GitHub, and study its Python code-writing capabilities. A distinct
production version of Codex powers GitHub Copilot. On HumanEval, a new
evaluation set we release to measure functional correctness for synthesizing
programs from docstrings, our model solves 28.8% of the problems, while GPT-3
solves 0% and GPT-J solves 11.4%. Furthermore, we find that repeated sampling
from the model is a surprisingly effective strategy for producing working
solutions to difficult prompts. Using this method, we solve 70.2% of our
problems with 100 samples per problem. Careful investigation of our model
reveals its limitations, including difficulty with docstrings describing long
chains of operations and with binding operations to variables. Finally, we
discuss the potential broader impacts of deploying powerful code generation
technologies, covering safety, security, and economics.
---

## It’s Not Just Size That Matters: Small Language Models Are Also Few-Shot Learners

Timo Schick, Hinrich Schütze

Category: llms
Keywords: few-shot learning, language models, GPT-3, cloze questions, natural language understanding
Year: 2021

When scaled to hundreds of billions of parameters, pretrained language models
such as GPT-3 achieve remarkable few-shot performance. However, enormous amounts
of compute are required for training and applying such big models, resulting in
a large carbon footprint and making it difficult for researchers and
practitioners to use them. We show that performance similar to GPT-3 can be
obtained with language models that are much “greener” in that their parameter
count is several orders of magnitude smaller. This is achieved by converting
textual inputs into cloze questions that contain a task description, combined
with gradient-based optimization; exploiting unlabeled data gives further
improvements. We identify key factors required for successful natural language
understanding with small language models.
---

## Training Language Models with Memory Augmentation

Zexuan Zhong, Tao Lei, Danqi Chen

Category: llms
Keywords: language models, memory augmentation, TRIME, perplexity, neural cache, machine translation
Year: 2021

Recent work has improved language models remarkably by equipping them with a
non-parametric memory component. However, most existing approaches only
introduce memories at testing time, or represent them using a separately trained
encoder—resulting in suboptimal training of the language model. In this work, we
present TRIME, a novel yet simple training approach designed for training
language models with memory augmentation. Our approach uses a training objective
that directly takes in-batch examples as accessible memory. We also present new
methods for memory construction and data batching, which are used for adapting
to different sets of memories—local, long-term, and external memory—at testing
time. We evaluate our approach on multiple language modeling and machine
translation benchmarks. We find that simply replacing the vanilla language
modeling objective by ours greatly reduces the perplexity, without modifying the
model architecture or incorporating extra context (e.g., 18.70 → 17.76 on
WikiText-103). We further augment language models with long-range contexts and
external knowledge and demonstrate significant gains over previous memory-
augmented approaches.
---

## It’s Not Just Size That Matters: Small Language Models Are Also Few-Shot Learners

Timo Schick, Hinrich Schütze

Category: llms
Keywords: language models, few-shot learning, GPT-3, natural language processing, cloze questions, parameter efficiency
Year: 2021

When scaled to hundreds of billions of parameters, pretrained language models
such as GPT-3 achieve remarkable few-shot performance. However, enormous amounts
of compute are required for training and applying such big models, resulting in
a large carbon footprint and making it difficult for researchers and
practitioners to use them. We show that performance similar to GPT-3 can be
obtained with language models that are much “greener” in that their parameter
count is several orders of magnitude smaller. This is achieved by converting
textual inputs into cloze questions that contain a task description, combined
with gradient-based optimization; exploiting unlabeled data gives further
improvements. We identify key factors required for successful natural language
understanding with small language models.
---

## Generating Datasets with Pretrained Language Models

Timo Schick, Hinrich Schütze

Category: llms
Keywords: pretrained language models, sentence embeddings, dataset generation, unsupervised learning, semantic textual similarity
Year: 2021

To obtain high-quality sentence embeddings from pretrained language models
(PLMs), they must either be augmented with additional pretraining objectives or
fine-tuned on a large set of labeled text pairs. While the latter approach
typically outperforms the former, it requires great human effort to generate
suitable datasets of sufficient size. In this paper, we show how large PLMs can
be leveraged to obtain high-quality embeddings without requiring any labeled
data, fine-tuning, or modifications to the pretraining objective: We utilize the
generative abilities of PLMs to generate entire datasets of labeled text pairs
from scratch, which can then be used for regular fine-tuning of much smaller
models. Our fully unsupervised approach outperforms strong baselines on several
English semantic textual similarity datasets.
---

## Extracting Training Data from Large Language Models

Nicholas Carlini, Florian Tramèr, Eric Wallace, Matthew Jagielski, Ariel Herbert-Voss, Katherine Lee, Adam Roberts, Tom Brown, Dawn Song, Úlfar Erlingsson, Alina Oprea, Colin Raffel

Category: llms
Keywords: language models, training data extraction, privacy leakage, GPT-2, neural networks
Year: 2021

It has become common to publish large (billion parameter) language models that
have been trained on private datasets. This paper demonstrates that in such
settings, an adversary can perform a training data extraction attack to recover
individual training examples by querying the language model. We demonstrate our
attack on GPT-2, a language model trained on scrapes of the public Internet, and
are able to extract hundreds of verbatim text sequences from the model’s
training data. These extracted examples include (public) personally identifiable
information (names, phone numbers, and email addresses), IRC conversations,
code, and 128-bit UUIDs. Our attack is possible even though each of the above
sequences are included in just one document in the training data. We
comprehensively evaluate our extraction attack to understand the factors that
contribute to its success. Worryingly, we find that larger models are more
vulnerable than smaller models. We conclude by drawing lessons and discussing
possible safeguards for training large language models.
---

## Generating Datasets with Pretrained Language Models

Timo Schick, Hinrich Schütze

Category: llms
Keywords: pretrained language models, sentence embeddings, unsupervised learning, dataset generation, semantic textual similarity
Year: 2021

To obtain high-quality sentence embeddings from pretrained language models
(PLMs), they must either be augmented with additional pretraining objectives or
finetuned on a large set of labeled text pairs. While the latter approach
typically outperforms the former, it requires great human effort to generate
suitable datasets of sufficient size. In this paper, we show how large PLMs can
be leveraged to obtain high-quality embeddings without requiring any labeled
data, finetuning or modifications to the pretraining objective: We utilize the
generative abilities of PLMs to generate entire datasets of labeled text pairs
from scratch, which can then be used for regular finetuning of much smaller
models. Our fully unsupervised approach outperforms strong baselines on several
English semantic textual similarity datasets.
---

## It’s Not Just Size That Matters: Small Language Models Are Also Few-Shot Learners

Timo Schick, Hinrich Schütze

Category: llms
Keywords: language models, few-shot learning, GPT-3, cloze questions, natural language understanding
Year: 2021

When scaled to hundreds of billions of parameters, pretrained language models
such as GPT-3 achieve remarkable few-shot performance. However, enormous amounts
of compute are required for training and applying such big models, resulting in
a large carbon footprint and making it difficult for researchers and
practitioners to use them. We show that performance similar to GPT-3 can be
obtained with language models that are much 'greener' in that their parameter
count is several orders of magnitude smaller. This is achieved by converting
textual inputs into cloze questions that contain a task description, combined
with gradient-based optimization; exploiting unlabeled data gives further
improvements. We identify key factors required for successful natural language
understanding with small language models.
---

## It’s Not Just Size That Matters: Small Language Models Are Also Few-Shot Learners

Timo Schick, Hinrich Schütze

Category: llms
Keywords: language models, few-shot learning, GPT-3, cloze questions, natural language understanding
Year: 2021

When scaled to hundreds of billions of parameters, pretrained language models
such as GPT-3 achieve remarkable few-shot performance. However, enormous amounts
of compute are required for training and applying such big models, resulting in
a large carbon footprint and making it difficult for researchers and
practitioners to use them. We show that performance similar to GPT-3 can be
obtained with language models that are much “greener” in that their parameter
count is several orders of magnitude smaller. This is achieved by converting
textual inputs into cloze questions that contain a task description, combined
with gradient-based optimization; exploiting unlabeled data gives further
improvements. We identify key factors required for successful natural language
understanding with small language models.
---

## Scaling Federated Learning for Fine-tuning of Large Language Models

Agrin Hilmkil, Sebastian Callh, Matteo Barbieri, Leon René Sütfeld, Edvin Listo Zec, Olof Mogren

Category: llms
Keywords: Federated Learning, Transformer, Large Language Models, BERT, ALBERT, DistilBERT, Text Classification, Distributed Compute, Privacy
Year: 2021

Federated learning (FL) is a promising approach to distributed compute, as well
as distributed data, and provides a level of privacy and compliance to legal
frameworks. This makes FL attractive for both consumer and healthcare
applications. While the area is actively being explored, few studies have
examined FL in the context of larger language models and there is a lack of
comprehensive reviews of robustness across tasks, architectures, numbers of
clients, and other relevant factors. In this paper, we explore the fine-tuning
of Transformer-based language models in a federated learning setting. We
evaluate three popular BERT-variants of different sizes (BERT, ALBERT, and
DistilBERT) on a number of text classification tasks such as sentiment analysis
and author identification. We perform an extensive sweep over the number of
clients, ranging up to 32, to evaluate the impact of distributed compute on task
performance in the federated averaging setting. While our findings suggest that
the large sizes of the evaluated models are not generally prohibitive to
federated training, we found that the different models handle federated
averaging to a varying degree. Most notably, DistilBERT converges significantly
slower with larger numbers of clients, and under some circumstances, even
collapses to chance level performance. Investigating this issue presents an
interesting perspective for future research.
---

## Language Models are Few-Shot Learners

Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, Dario Amodei

Category: llms
Keywords: GPT-3, few-shot learning, language models, NLP, text generation
Year: 2020

Recent work has demonstrated substantial gains on many NLP tasks and benchmarks
by pre-training on a large corpus of text followed by fine-tuning on a specific
task. While typically task-agnostic in architecture, this method still requires
task-specific fine-tuning datasets of thousands or tens of thousands of
examples. By contrast, humans can generally perform a new language task from
only a few examples or from simple instructions – something which current NLP
systems still largely struggle to do. Here we show that scaling up language
models greatly improves task-agnostic, few-shot performance, sometimes even
reaching competitiveness with prior state-of-the-art fine-tuning approaches.
Specifically, we train GPT-3, an autoregressive language model with 175 billion
parameters, 10x more than any previous non-sparse language model, and test its
performance in the few-shot setting. For all tasks, GPT-3 is applied without any
gradient updates or fine-tuning, with tasks and few-shot demonstrations
specified purely via text interaction with the model. GPT-3 achieves strong
performance on many NLP datasets, including translation, question-answering, and
cloze tasks, as well as several tasks that require on-the-fly reasoning or
domain adaptation, such as unscrambling words, using a novel word in a sentence,
or performing 3-digit arithmetic. At the same time, we also identify some
datasets where GPT-3’s few-shot learning still struggles, as well as some
datasets where GPT-3 faces methodological issues related to training on large
web corpora. Finally, we find that GPT-3 can generate samples of news articles
which human evaluators have difficulty distinguishing from articles written by
humans. We discuss broader societal impacts of this finding and of GPT-3 in
general.
---

## Scaling Laws for Neural Language Models

Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B. Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, Dario Amodei

Category: llms
Keywords: scaling laws, language models, cross-entropy loss, power-law, compute efficiency
Year: 2020

We study empirical scaling laws for language model performance on the cross-
entropy loss. The loss scales as a power-law with model size, dataset size, and
the amount of compute used for training, with some trends spanning more than
seven orders of magnitude. Other architectural details such as network width or
depth have minimal effects within a wide range. Simple equations govern the
dependence of overfitting on model/dataset size and the dependence of training
speed on model size. These relationships allow us to determine the optimal
allocation of a fixed compute budget. Larger models are significantly more
sample-efficient, such that optimally compute-efficient training involves
training very large models on a relatively modest amount of data and stopping
significantly before convergence.
---

## Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism

Mohammad Shoeybi, Mostofa Patwary, Raul Puri, Patrick LeGresley, Jared Casper, Bryan Catanzaro

Category: llms
Keywords: language modeling, transformer models, model parallelism, NLP, GPT-2, BERT, PyTorch, scalability
Year: 2020

Recent work in language modeling demonstrates that training large transformer
models advances the state of the art in Natural Language Processing
applications. However, very large models can be quite difficult to train due to
memory constraints. In this work, we present our techniques for training very
large transformer models and implement a simple, efficient intra-layer model
parallel approach that enables training transformer models with billions of
parameters. Our approach does not require a new compiler or library changes, is
orthogonal and complimentary to pipeline model parallelism, and can be fully
implemented with the insertion of a few communication operations in native
PyTorch. We illustrate this approach by converging transformer based models up
to 8.3 billion parameters using 512 GPUs. We sustain 15.1 PetaFLOPs across the
entire application with 76% scaling efficiency when compared to a strong single
GPU baseline that sustains 39 TeraFLOPs, which is 30% of peak FLOPs. To
demonstrate that large language models can further advance the state of the art
(SOTA), we train an 8.3 billion parameter transformer language model similar to
GPT-2 and a 3.9 billion parameter model similar to BERT. We show that careful
attention to the placement of layer normalization in BERT-like models is
critical to achieving increased performance as the model size grows. Using the
GPT-2 model we achieve SOTA results on the WikiText103 (10.8 compared to SOTA
perplexity of 15.8) and LAMBADA (66.5% compared to SOTA accuracy of 63.2%)
datasets. Our BERT model achieves SOTA results on the RACE dataset (90.9%
compared to SOTA accuracy of 89.4%).
---

## Language Models are Few-Shot Learners

Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, Dario Amodei

Category: llms
Keywords: GPT-3, few-shot learning, language models, NLP, autoregressive models
Year: 2020

Recent work has demonstrated substantial gains on many NLP tasks and benchmarks
by pre-training on a large corpus of text followed by fine-tuning on a specific
task. While typically task-agnostic in architecture, this method still requires
task-specific fine-tuning datasets of thousands or tens of thousands of
examples. By contrast, humans can generally perform a new language task from
only a few examples or from simple instructions – something which current NLP
systems still largely struggle to do. Here we show that scaling up language
models greatly improves task-agnostic, few-shot performance, sometimes even
reaching competitiveness with prior state-of-the-art fine-tuning approaches.
Specifically, we train GPT-3, an autoregressive language model with 175 billion
parameters, 10x more than any previous non-sparse language model, and test its
performance in the few-shot setting. For all tasks, GPT-3 is applied without any
gradient updates or fine-tuning, with tasks and few-shot demonstrations
specified purely via text interaction with the model. GPT-3 achieves strong
performance on many NLP datasets, including translation, question-answering, and
cloze tasks, as well as several tasks that require on-the-fly reasoning or
domain adaptation, such as unscrambling words, using a novel word in a sentence,
or performing 3-digit arithmetic. At the same time, we also identify some
datasets where GPT-3’s few-shot learning still struggles, as well as some
datasets where GPT-3 faces methodological issues related to training on large
web corpora. Finally, we find that GPT-3 can generate samples of news articles
which human evaluators have difficulty distinguishing from articles written by
humans. We discuss broader societal impacts of this finding and of GPT-3 in
general.
---

## Few-Shot Text Generation with Pattern-Exploiting Training

Timo Schick, Hinrich Schütze

Category: llms
Keywords: few-shot learning, text generation, pattern-exploiting training, language models, text summarization
Year: 2020

Providing pretrained language models with simple task descriptions or prompts in
natural language yields impressive few-shot results for a wide range of text
classification tasks when combined with gradient-based learning from examples.
In this paper, we show that the underlying idea can also be applied to text
generation tasks: We adapt Pattern-Exploiting Training (PET), a recently
proposed few-shot approach, for finetuning generative language models on text
generation tasks. On several text summarization and headline generation
datasets, our proposed variant of PET gives consistent improvements over a
strong baseline in few-shot settings.
---

## Language Models are Few-Shot Learners

Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, Dario Amodei

Category: llms
Keywords: NLP, language models, few-shot learning, GPT-3, autoregressive model
Year: 2020

Recent work has demonstrated substantial gains on many NLP tasks and benchmarks
by pre-training on a large corpus of text followed by fine-tuning on a specific
task. While typically task-agnostic in architecture, this method still requires
task-specific fine-tuning datasets of thousands or tens of thousands of
examples. By contrast, humans can generally perform a new language task from
only a few examples or from simple instructions – something which current NLP
systems still largely struggle to do. Here we show that scaling up language
models greatly improves task-agnostic, few-shot performance, sometimes even
reaching competitiveness with prior state-of-the-art fine-tuning approaches.
Specifically, we train GPT-3, an autoregressive language model with 175 billion
parameters, 10x more than any previous non-sparse language model, and test its
performance in the few-shot setting. For all tasks, GPT-3 is applied without any
gradient updates or fine-tuning, with tasks and few-shot demonstrations
specified purely via text interaction with the model. GPT-3 achieves strong
performance on many NLP datasets, including translation, question-answering, and
cloze tasks, as well as several tasks that require on-the-fly reasoning or
domain adaptation, such as unscrambling words, using a novel word in a sentence,
or performing 3-digit arithmetic. At the same time, we also identify some
datasets where GPT-3’s few-shot learning still struggles, as well as some
datasets where GPT-3 faces methodological issues related to training on large
web corpora. Finally, we find that GPT-3 can generate samples of news articles
which human evaluators have difficulty distinguishing from articles written by
humans. We discuss broader societal impacts of this finding and of GPT-3 in
general.
---

## Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism

Mohammad Shoeybi, Mostofa Patwary, Raul Puri, Patrick LeGresley, Jared Casper, Bryan Catanzaro

Category: llms
Keywords: language models, model parallelism, transformer, GPT-2, BERT, PyTorch, NLP, scaling efficiency
Year: 2020

Recent work in language modeling demonstrates that training large transformer
models advances the state of the art in Natural Language Processing
applications. However, very large models can be quite difficult to train due to
memory constraints. In this work, we present our techniques for training very
large transformer models and implement a simple, efficient intra-layer model
parallel approach that enables training transformer models with billions of
parameters. Our approach does not require a new compiler or library changes, is
orthogonal and complementary to pipeline model parallelism, and can be fully
implemented with the insertion of a few communication operations in native
PyTorch. We illustrate this approach by converging transformer-based models up
to 8.3 billion parameters using 512 GPUs. We sustain 15.1 PetaFLOPs across the
entire application with 76% scaling efficiency when compared to a strong single
GPU baseline that sustains 39 TeraFLOPs, which is 30% of peak FLOPs. To
demonstrate that large language models can further advance the state of the art
(SOTA), we train an 8.3 billion parameter transformer language model similar to
GPT-2 and a 3.9 billion parameter model similar to BERT. We show that careful
attention to the placement of layer normalization in BERT-like models is
critical to achieving increased performance as the model size grows. Using the
GPT-2 model we achieve SOTA results on the WikiText103 (10.8 compared to SOTA
perplexity of 15.8) and LAMBADA (66.5% compared to SOTA accuracy of 63.2%)
datasets. Our BERT model achieves SOTA results on the RACE dataset (90.9%
compared to SOTA accuracy of 89.4%).
---

## Language Models are Few-Shot Learners

Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, Dario Amodei

Category: llms
Keywords: GPT-3, language models, few-shot learning, NLP, autoregressive
Year: 2020

Recent work has demonstrated substantial gains on many NLP tasks and benchmarks
by pre-training on a large corpus of text followed by fine-tuning on a specific
task. While typically task-agnostic in architecture, this method still requires
task-specific fine-tuning datasets of thousands or tens of thousands of
examples. By contrast, humans can generally perform a new language task from
only a few examples or from simple instructions – something which current NLP
systems still largely struggle to do. Here we show that scaling up language
models greatly improves task-agnostic, few-shot performance, sometimes even
reaching competitiveness with prior state-of-the-art fine-tuning approaches.
Specifically, we train GPT-3, an autoregressive language model with 175 billion
parameters, 10x more than any previous non-sparse language model, and test its
performance in the few-shot setting. For all tasks, GPT-3 is applied without any
gradient updates or fine-tuning, with tasks and few-shot demonstrations
specified purely via text interaction with the model. GPT-3 achieves strong
performance on many NLP datasets, including translation, question-answering, and
cloze tasks, as well as several tasks that require on-the-fly reasoning or
domain adaptation, such as unscrambling words, using a novel word in a sentence,
or performing 3-digit arithmetic. At the same time, we also identify some
datasets where GPT-3’s few-shot learning still struggles, as well as some
datasets where GPT-3 faces methodological issues related to training on large
web corpora. Finally, we find that GPT-3 can generate samples of news articles
which human evaluators have difficulty distinguishing from articles written by
humans. We discuss broader societal impacts of this finding and of GPT-3 in
general.
---

## How Much Knowledge Can You Pack Into the Parameters of a Language Model?

Adam Roberts, Colin Raffel, Noam Shazeer

Category: llms
Keywords: neural language models, knowledge storage, fine-tuning, question answering, open-domain, WebQuestions, TriviaQA
Year: 2020

It has recently been observed that neural language models trained on
unstructured text can implicitly store and retrieve knowledge using natural
language queries. In this short paper, we measure the practical utility of this
approach by fine-tuning pre-trained models to answer questions without access to
any external context or knowledge. We show that this approach scales with model
size and outperforms models that explicitly look up knowledge on the open-domain
variants of WebQuestions and TriviaQA. To facilitate reproducibility and future
work, we release our code and trained models.
---

## Language Models are Unsupervised Multitask Learners

Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever

Category: llms
Keywords: language models, unsupervised learning, multitask learning, WebText, GPT-2, transformer, zero-shot learning
Year: 2019

Natural language processing tasks, such as question answering, machine
translation, reading comprehension, and summarization, are typically approached
with supervised learning on task-specific datasets. We demonstrate that language
models begin to learn these tasks without any explicit supervision when trained
on a new dataset of millions of webpages called WebText. When conditioned on a
document plus questions, the answers generated by the language model reach 55 F1
on the CoQA dataset - matching or exceeding the performance of 3 out of 4
baseline systems without using the 127,000+ training examples. The capacity of
the language model is essential to the success of zero-shot task transfer and
increasing it improves performance in a log-linear fashion across tasks. Our
largest model, GPT-2, is a 1.5B parameter Transformer that achieves state of the
art results on 7 out of 8 tested language modeling datasets in a zero-shot
setting but still underfits WebText. Samples from the model reflect these
improvements and contain coherent paragraphs of text. These findings suggest a
promising path towards building language processing systems which learn to
perform tasks from their naturally occurring demonstrations.
---

## SCIBERT: A Pretrained Language Model for Scientific Text

Iz Beltagy, Kyle Lo, Arman Cohan

Category: llms
Keywords: SCIBERT, pretrained language model, scientific text, NLP, BERT
Year: 2019

Obtaining large-scale annotated data for NLP tasks in the scientific domain is
challenging and expensive. We release SCIBERT, a pretrained language model based
on BERT (Devlin et al., 2019) to address the lack of high-quality, large-scale
labeled scientific data. SCIBERT leverages unsupervised pretraining on a large
multi-domain corpus of scientific publications to improve performance on
downstream scientific NLP tasks. We evaluate on a suite of tasks including
sequence tagging, sentence classification and dependency parsing, with datasets
from a variety of scientific domains. We demonstrate statistically significant
improvements over BERT and achieve new state-of-the-art results on several of
these tasks. The code and pretrained models are available at
https://github.com/allenai/scibert/.
---

## Language Models are Unsupervised Multitask Learners

Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever

Category: llms
Keywords: language models, unsupervised learning, multitask learning, GPT-2, zero-shot learning, WebText
Year: 2019

Natural language processing tasks, such as question answering, machine
translation, reading comprehension, and summarization, are typically approached
with supervised learning on task-specific datasets. We demonstrate that language
models begin to learn these tasks without any explicit supervision when trained
on a new dataset of millions of webpages called WebText. When conditioned on a
document plus questions, the answers generated by the language model reach 55 F1
on the CoQA dataset - matching or exceeding the performance of 3 out of 4
baseline systems without using the 127,000+ training examples. The capacity of
the language model is essential to the success of zero-shot task transfer and
increasing it improves performance in a log-linear fashion across tasks. Our
largest model, GPT-2, is a 1.5B parameter Transformer that achieves state of the
art results on 7 out of 8 tested language modeling datasets in a zero-shot
setting but still underfits WebText. Samples from the model reflect these
improvements and contain coherent paragraphs of text. These findings suggest a
promising path towards building language processing systems which learn to
perform tasks from their naturally occurring demonstrations.
---

## Language Models are Unsupervised Multitask Learners

Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever

Category: llms
Keywords: language models, unsupervised learning, multitask learning, GPT-2, zero-shot transfer
Year: 2019

Natural language processing tasks, such as question answering, machine
translation, reading comprehension, and summarization, are typically approached
with supervised learning on task-specific datasets. We demonstrate that language
models begin to learn these tasks without any explicit supervision when trained
on a new dataset of millions of webpages called WebText. When conditioned on a
document plus questions, the answers generated by the language model reach 55 F1
on the CoQA dataset - matching or exceeding the performance of 3 out of 4
baseline systems without using the 127,000+ training examples. The capacity of
the language model is essential to the success of zero-shot task transfer and
increasing it improves performance in a log-linear fashion across tasks. Our
largest model, GPT-2, is a 1.5B parameter Transformer that achieves state of the
art results on 7 out of 8 tested language modeling datasets in a zero-shot
setting but still underfits WebText. Samples from the model reflect these
improvements and contain coherent paragraphs of text. These findings suggest a
promising path towards building language processing systems which learn to
perform tasks from their naturally occurring demonstrations.
---

## Improving Language Understanding by Generative Pre-Training

Alec Radford, Karthik Narasimhan, Tim Salimans, Ilya Sutskever

Category: llms
Keywords: natural language understanding, generative pre-training, language model, fine-tuning, transfer learning
Year: 2018

Natural language understanding comprises a wide range of diverse tasks such as
textual entailment, question answering, semantic similarity assessment, and
document classification. Although large unlabeled text corpora are abundant,
labeled data for learning these specific tasks is scarce, making it challenging
for discriminatively trained models to perform adequately. We demonstrate that
large gains on these tasks can be realized by generative pre-training of a
language model on a diverse corpus of unlabeled text, followed by discriminative
fine-tuning on each specific task. In contrast to previous approaches, we make
use of task-aware input transformations during fine-tuning to achieve effective
transfer while requiring minimal changes to the model architecture. We
demonstrate the effectiveness of our approach on a wide range of benchmarks for
natural language understanding. Our general task-agnostic model outperforms
discriminatively trained models that use architectures specifically crafted for
each task, significantly improving upon the state of the art in 9 out of the 12
tasks studied. For instance, we achieve absolute improvements of 8.9% on
commonsense reasoning (Stories Cloze Test), 5.7% on question answering (RACE),
and 1.5% on textual entailment (MultiNLI).
---

## Improving Language Understanding by Generative Pre-Training

Alec Radford, Karthik Narasimhan, Tim Salimans, Ilya Sutskever

Category: llms
Keywords: natural language understanding, generative pre-training, fine-tuning, language model, transfer learning
Year: 2018

Natural language understanding comprises a wide range of diverse tasks such as
textual entailment, question answering, semantic similarity assessment, and
document classification. Although large unlabeled text corpora are abundant,
labeled data for learning these specific tasks is scarce, making it challenging
for discriminatively trained models to perform adequately. We demonstrate that
large gains on these tasks can be realized by generative pre-training of a
language model on a diverse corpus of unlabeled text, followed by discriminative
fine-tuning on each specific task. In contrast to previous approaches, we make
use of task-aware input transformations during fine-tuning to achieve effective
transfer while requiring minimal changes to the model architecture. We
demonstrate the effectiveness of our approach on a wide range of benchmarks for
natural language understanding. Our general task-agnostic model outperforms
discriminatively trained models that use architectures specifically crafted for
each task, significantly improving upon the state of the art in 9 out of the 12
tasks studied. For instance, we achieve absolute improvements of 8.9% on
commonsense reasoning (Stories Cloze Test), 5.7% on question answering (RACE),
and 1.5% on textual entailment (MultiNLI).
---

## Improving Language Understanding by Generative Pre-Training

Alec Radford, Karthik Narasimhan, Tim Salimans, Ilya Sutskever

Category: llms
Keywords: generative pre-training, natural language understanding, fine-tuning, transfer learning, language model
Year: 2018

Natural language understanding comprises a wide range of diverse tasks such as
textual entailment, question answering, semantic similarity assessment, and
document classification. Although large unlabeled text corpora are abundant,
labeled data for learning these specific tasks is scarce, making it challenging
for discriminatively trained models to perform adequately. We demonstrate that
large gains on these tasks can be realized by generative pre-training of a
language model on a diverse corpus of unlabeled text, followed by discriminative
fine-tuning on each specific task. In contrast to previous approaches, we make
use of task-aware input transformations during fine-tuning to achieve effective
transfer while requiring minimal changes to the model architecture. We
demonstrate the effectiveness of our approach on a wide range of benchmarks for
natural language understanding. Our general task-agnostic model outperforms
discriminatively trained models that use architectures specifically crafted for
each task, significantly improving upon the state of the art in 9 out of the 12
tasks studied. For instance, we achieve absolute improvements of 8.9% on
commonsense reasoning (Stories Cloze Test), 5.7% on question answering (RACE),
and 1.5% on textual entailment (MultiNLI).