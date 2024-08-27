
## CRAMMING: Training a Language Model on a Single GPU in One Day

Jonas Geiping, Tom Goldstein

Category: transformer-other
Keywords: language modeling, transformer, scaling, GPU, pretraining, BERT
Year: 2023

Recent trends in language modeling have focused on increasing performance
through scaling, and have resulted in an environment where training language
models is out of reach for most researchers and practitioners. While most in the
community are asking how to push the limits of extreme computation, we ask the
opposite question: How far can we get with a single GPU in just one day? We
investigate the downstream performance achievable with a transformer-based
language model trained completely from scratch with masked language modeling for
a single day on a single consumer GPU. Aside from re-analyzing nearly all
components of the pretraining pipeline for this scenario and providing a
modified pipeline with performance close to BERT, we investigate why scaling
down is hard, and which modifications actually improve performance in this
scenario. We provide evidence that even in this constrained setting, performance
closely follows scaling laws observed in large-compute settings. Through the
lens of scaling laws, we categorize a range of recent improvements to training
and architecture and discuss their merit and practical applicability (or lack
thereof) for the limited compute setting.

---

## Language Is Not All You Need: Aligning Perception with Language Models

Shaohan Huang, Li Dong, Wenhui Wang, Yaru Hao, Saksham Singhal, Shuming Ma, Tengchao Lv, Lei Cui, Owais Khan Mohammed, Barun Patra, Qiang Liu, Kriti Aggarwal, Zewen Chi, Johan Bjorck, Vishrav Chaudhary, Subhojit Som, Xia Song, Furu Wei

Category: transformer-other
Keywords: multimodal, large language models, perception, vision, language alignment
Year: 2023

KOSMOS-1 is a multimodal large language model (MLLM) that is capable of
perceiving multimodal input, following instructions, and performing in-context
learning for not only language tasks but also multimodal tasks. In this work, we
align vision with large language models (LLMs), advancing the trend of going
from LLMs to MLLMs.

---

## Hyena Hierarchy: Towards Larger Convolutional Language Models

Michael Poli, Stefano Massaroli, Eric Nguyen, Daniel Y. Fu, Tri Dao, Stephen Baccus, Yoshua Bengio, Stefano Ermon, Christopher Ré

Category: transformer-other
Keywords: Transformers, attention mechanism, deep learning, sequence modeling, Hyena, language modeling
Year: 2023

Recent advances in deep learning have relied heavily on the use of large
Transformers due to their ability to learn at scale. However, the core building
block of Transformers, the attention operator, exhibits quadratic cost in
sequence length, limiting the amount of context accessible. Existing
subquadratic methods based on low-rank and sparse approximations need to be
combined with dense attention layers to match Transformers, indicating a gap in
capability. In this work, we propose Hyena, a subquadratic drop-in replacement
for attention constructed by interleaving implicitly parametrized long
convolutions and data-controlled gating. In recall and reasoning tasks on
sequences of thousands to hundreds of thousands of tokens, Hyena improves
accuracy by more than 50 points over operators relying on state-spaces and other
implicit and explicit methods, matching attention-based models. We set a new
state-of-the-art for dense-attention-free architectures on language modeling in
standard datasets (WikiText103 and The Pile), reaching Transformer quality with
a 20% reduction in training compute required at sequence length 2K. Hyena
operators are twice as fast as highly optimized attention at sequence length 8K,
and 100× faster at sequence length 64K.

---

## Unlimiformer: Long-Range Transformers with Unlimited Length Input

Amanda Bertsch, Uri Alon, Graham Neubig, Matthew R. Gormley

Category: transformer-other
Keywords: Unlimiformer, long-range transformers, attention computation, k-nearest-neighbor index, summarization
Year: 2023

Transformer-based models typically have a predefined bound to their input
length, because of their need to potentially attend to every token in the input.
In this work, we propose Unlimiformer: a general approach that can wrap any
existing pretrained encoder-decoder transformer, and offload the attention
computation across all layers to a single k-nearest-neighbor index; this index
can be kept on either the GPU or CPU memory and queried in sub-linear time. This
way, we can index extremely long input sequences, while every attention head in
every decoder layer retrieves its top-k keys, instead of attending to every key.
We demonstrate Unlimiformer's efficacy on several long-document and multi-
document summarization benchmarks, showing that it can summarize even 350k
token-long inputs from the BookSum dataset, without any input truncation at test
time. Unlimiformer improves pretrained models such as BART and Longformer by
extending them to unlimited inputs without additional learned weights and
without modifying their code. We make our code and models publicly available.

---

## Hyena Hierarchy: Towards Larger Convolutional Language Models

Michael Poli, Stefano Massaroli, Eric Nguyen, Daniel Y. Fu, Tri Dao, Stephen Baccus, Yoshua Bengio, Stefano Ermon, Christopher Ré

Category: transformer-other
Keywords: transformers, attention, convolutional language models, Hyena, subquadratic operators, language modeling, deep learning
Year: 2023

Recent advances in deep learning have relied heavily on the use of large
Transformers due to their ability to learn at scale. However, the core building
block of Transformers, the attention operator, exhibits quadratic cost in
sequence length, limiting the amount of context accessible. Existing
subquadratic methods based on low-rank and sparse approximations need to be
combined with dense attention layers to match Transformers, indicating a gap in
capability. In this work, we propose Hyena, a subquadratic drop-in replacement
for attention constructed by interleaving implicitly parametrized long
convolutions and data-controlled gating. In recall and reasoning tasks on
sequences of thousands to hundreds of thousands of tokens, Hyena improves
accuracy by more than 50 points over operators relying on state-spaces and other
implicit and explicit methods, matching attention-based models. We set a new
state-of-the-art for dense-attention-free architectures on language modeling in
standard datasets (WikiText103 and The Pile), reaching Transformer quality with
a 20% reduction in training compute required at sequence length 2K. Hyena
operators are twice as fast as highly optimized attention at sequence length 8K,
and 100× faster at sequence length 64K.

---

## Foundation Transformers

Hongyu Wang, Shuming Ma, Shaohan Huang, Li Dong, Wenhui Wang, Zhiliang Peng, Yu Wu, Payal Bajaj, Saksham Singhal, Alon Benhaim, Barun Patra, Zhun Liu, Vishrav Chaudhary, Xia Song, Furu Wei

Category: transformer-other
Keywords: Transformers, Foundation Transformers, MAGNETO, Sub-LayerNorm, Multimodal, Initialization Strategy
Year: 2023

A big convergence of model architectures across language, vision, speech, and
multimodal is emerging. However, under the same name 'Transformers', the above
areas use different implementations for better performance, e.g., Post-LayerNorm
for BERT, and Pre-LayerNorm for GPT and vision Transformers. We call for the
development of Foundation Transformer for true general-purpose modeling, which
serves as a go-to architecture for various tasks and modalities with guaranteed
training stability. In this work, we introduce a Transformer variant, named
MAGNETO, to fulfill the goal. Specifically, we propose Sub-LayerNorm for good
expressivity, and the initialization strategy theoretically derived from DeepNet
(Wang et al., 2022a) for stable scaling up. Extensive experiments demonstrate
its superior performance and better stability than the de facto Transformer
variants designed for various applications, including language modeling (i.e.,
BERT, and GPT), machine translation, vision pretraining (i.e., BEiT), speech
recognition, and multimodal pretraining (i.e., BEiT-3).

---

## Language Is Not All You Need: Aligning Perception with Language Models

Shaohan Huang, Li Dong, Wenhui Wang, Yaru Hao, Saksham Singhal, Shuming Ma, Tengchao Lv, Lei Cui, Owais Khan Mohammed, Barun Patra, Qiang Liu, Kriti Aggarwal, Zewen Chi, Johan Bjorck, Vishrav Chaudhary, Subhojit Som, Xia Song, Furu Wei

Category: transformer-other
Keywords: multimodal, large language models, vision, language alignment, perception
Year: 2023

KOSMOS-1 is a multimodal large language model (MLLM) that is capable of
perceiving multimodal input, following instructions, and performing in-context
learning for not only language tasks but also multimodal tasks. In this work, we
align vision with large language models (LLMs), advancing the trend of going
from LLMs to MLLMs.

---

## Hyena Hierarchy: Towards Larger Convolutional Language Models

Michael Poli, Stefano Massaroli, Eric Nguyen, Daniel Y. Fu, Tri Dao, Stephen Baccus, Yoshua Bengio, Stefano Ermon, Christopher Ré

Category: transformer-other
Keywords: Hyena, convolutional language models, Transformers, attention operator, subquadratic
Year: 2023

Recent advances in deep learning have relied heavily on the use of large
Transformers due to their ability to learn at scale. However, the core building
block of Transformers, the attention operator, exhibits quadratic cost in
sequence length, limiting the amount of context accessible. Existing
subquadratic methods based on low-rank and sparse approximations need to be
combined with dense attention layers to match Transformers, indicating a gap in
capability. In this work, we propose Hyena, a subquadratic drop-in replacement
for attention constructed by interleaving implicitly parametrized long
convolutions and data-controlled gating. In recall and reasoning tasks on
sequences of thousands to hundreds of thousands of tokens, Hyena improves
accuracy by more than 50 points over operators relying on state-spaces and other
implicit and explicit methods, matching attention-based models. We set a new
state-of-the-art for dense-attention-free architectures on language modeling in
standard datasets (WikiText103 and The Pile), reaching Transformer quality with
a 20% reduction in training compute required at sequence length 2K. Hyena
operators are twice as fast as highly optimized attention at sequence length 8K,
and 100× faster at sequence length 64K.

---

## The Impact of Positional Encoding on Length Generalization in Transformers

Amirhossein Kazemnejad, Inkit Padhi, Karthikeyan Natesan Ramamurthy, Payel Das, Siva Reddy

Category: transformer-other
Keywords: positional encoding, length generalization, decoder-only Transformers, absolute position embedding, relative position encoding
Year: 2023

Length generalization, the ability to generalize from small training context
sizes to larger ones, is a critical challenge in the development of Transformer-
based language models. Positional encoding (PE) has been identified as a major
factor influencing length generalization, but the exact impact of different PE
schemes on extrapolation in downstream tasks remains unclear. In this paper, we
conduct a systematic empirical study comparing the length generalization
performance of decoder-only Transformers with five different position encoding
approaches including Absolute Position Embedding (APE), T5’s Relative PE, ALiBi,
and Rotary, in addition to Transformers without positional encoding (NoPE). Our
evaluation encompasses a battery of reasoning and mathematical tasks. Our
findings reveal that the most commonly used positional encoding methods, such as
ALiBi, Rotary, and APE, are not well suited for length generalization in
downstream tasks. More importantly, NoPE outperforms other explicit positional
encoding methods while requiring no additional computation. We theoretically
demonstrate that NoPE can represent both absolute and relative PEs, but when
trained with SGD, it mostly resembles T5’s Relative PE attention patterns.
Finally, we find that scratchpad is not always helpful to solve length
generalization and its format highly impacts the model’s performance. Overall,
our work suggests that explicit position encodings are not essential for
decoder-only Transformers to generalize well to longer sequences.

---

## Hyena Hierarchy: Towards Larger Convolutional Language Models

Michael Poli, Stefano Massaroli, Eric Nguyen, Daniel Y. Fu, Tri Dao, Stephen Baccus, Yoshua Bengio, Stefano Ermon, Christopher Ré

Category: transformer-other
Keywords: Hyena, Transformers, Convolutional Language Models, Attention, Deep Learning
Year: 2023

Recent advances in deep learning have relied heavily on the use of large
Transformers due to their ability to learn at scale. However, the core building
block of Transformers, the attention operator, exhibits quadratic cost in
sequence length, limiting the amount of context accessible. Existing
subquadratic methods based on low-rank and sparse approximations need to be
combined with dense attention layers to match Transformers, indicating a gap in
capability. In this work, we propose Hyena, a subquadratic drop-in replacement
for attention constructed by interleaving implicitly parametrized long
convolutions and data-controlled gating. In recall and reasoning tasks on
sequences of thousands to hundreds of thousands of tokens, Hyena improves
accuracy by more than 50 points over operators relying on state-spaces and other
implicit and explicit methods, matching attention-based models. We set a new
state-of-the-art for dense-attention-free architectures on language modeling in
standard datasets (WikiText103 and The Pile), reaching Transformer quality with
a 20% reduction in training compute required at sequence length 2K. Hyena
operators are twice as fast as highly optimized attention at sequence length 8K,
and 100× faster at sequence length 64K.

---

## Hungry Hungry Hippos: Towards Language Modeling with State Space Models

Daniel Y. Fu, Tri Dao, Khaled K. Saab, Armin W. Thomas, Atri Rudra, Christopher Ré

Category: transformer-other
Keywords: state space models, language modeling, transformers, FlashConv, H3-attention
Year: 2023

State space models (SSMs) have demonstrated state-of-the-art sequence modeling
performance in some modalities, but underperform attention in language modeling.
Moreover, despite scaling nearly linearly in sequence length instead of
quadratically, SSMs are still slower than Transformers due to poor hardware
utilization. In this paper, we make progress on understanding the expressivity
gap between SSMs and attention in language modeling, and on reducing the
hardware barrier between SSMs and attention. First, we use synthetic language
modeling tasks to understand the gap between SSMs and attention. We find that
existing SSMs struggle with two capabilities: recalling earlier tokens in the
sequence and comparing tokens across the sequence. To understand the impact on
language modeling, we propose a new SSM layer, H3, that is explicitly designed
for these abilities. H3 matches attention on the synthetic languages and comes
within 0.4 PPL of Transformers on OpenWebText. Furthermore, a hybrid
125M-parameter H3-attention model that retains two attention layers surprisingly
outperforms Transformers on OpenWebText by 1.0 PPL. Next, to improve the
efficiency of training SSMs on modern hardware, we propose FlashConv. FlashConv
uses a fused block FFT algorithm to improve efficiency on sequences up to 8K,
and introduces a novel state passing algorithm that exploits the recurrent
properties of SSMs to scale to longer sequences. FlashConv yields 2× speedup on
the long-range arena benchmark and allows hybrid language models to generate
text 2.4× faster than Transformers. Using FlashConv, we scale hybrid
H3-attention language models up to 2.7B parameters on the Pile and find
promising initial results, achieving lower perplexity than Transformers and
outperforming Transformers in zero- and few-shot learning on a majority of tasks
in the SuperGLUE benchmark.

---

## Long Range Language Modeling via Gated State Spaces

Harsh Mehta, Ankit Gupta, Ashok Cutkosky, Behnam Neyshabur

Category: transformer-other
Keywords: state space models, long range dependencies, autoregressive sequence modeling, gated activation functions, transformers
Year: 2023

State space models have shown to be effective at modeling long range
dependencies, especially on sequence classification tasks. In this work we focus
on autoregressive sequence modeling over English books, Github source code, and
ArXiv mathematics articles. Based on recent developments around the
effectiveness of gated activation functions, we propose a new layer named Gated
State Space (GSS) and show that it trains significantly faster than the diagonal
version of S4 (i.e. DSS) on TPUs, is competitive with several well-tuned
Transformer-based baselines and exhibits zero-shot generalization to longer
inputs while being straightforward to implement. Finally, we show that
leveraging self-attention to model local dependencies improves the performance
of GSS even further.

---

## RWKV: Reinventing RNNs for the Transformer Era

Bo Peng, Eric Alcaide, Quentin Anthony, Alon Albalak, Samuel Arcadinho, Huanqi Cao, Xin Cheng, Michael Chung, Matteo Grella, Kranthi Kiran GV, Xuzheng He, Haowen Hou, Przemysław Kazienko, Jan Kocoń, Jiaming Kong, Bartłomiej Koptyra, Hayden Lau, Krishna Sri Ipsit Mantri, Ferdinand Mom, Atsushi Saito, Xiangru Tang, Bolun Wang, Johan S. Wind, Stanisław Woźniak, Ruichong Zhang, Zhenyuan Zhang, Qihang Zhao, Peng Zhou, Jian Zhu, Rui-Jie Zhu

Category: transformer-other
Keywords: Transformers, RNNs, RWKV, linear attention, sequence processing
Year: 2023

Transformers have revolutionized almost all natural language processing (NLP)
tasks but suffer from memory and computational complexity that scales
quadratically with sequence length. In contrast, recurrent neural networks
(RNNs) exhibit linear scaling in memory and computational requirements but
struggle to match the same performance as Transformers due to limitations in
parallelization and scalability. We propose a novel model architecture,
Receptance Weighted Key Value (RWKV), that combines the efficient parallelizable
training of Transformers with the efficient inference of RNNs. Our approach
leverages a linear attention mechanism and allows us to formulate the model as
either a Transformer or an RNN, which parallelizes computations during training
and maintains constant computational and memory complexity during inference,
leading to the first non-transformer architecture to be scaled to tens of
billions of parameters. Our experiments reveal that RWKV performs on par with
similarly sized Transformers, suggesting that future work can leverage this
architecture to create more efficient models. This work presents a significant
step towards reconciling the trade-offs between computational efficiency and
model performance in sequence processing tasks.

---

## AttentionViz: A Global View of Transformer Attention

Catherine Yeh, Yida Chen, Aoyu Wu, Cynthia Chen, Fernanda Viégas, Martin Wattenberg

Category: transformer-other
Keywords: Transformer, Attention, NLP, Computer Vision, Visual Analytics
Year: 2023

Transformer models are revolutionizing machine learning, but their inner
workings remain mysterious. In this work, we present a new visualization
technique designed to help researchers understand the self-attention mechanism
in transformers that allows these models to learn rich, contextual relationships
between elements of a sequence. The main idea behind our method is to visualize
a joint embedding of the query and key vectors used by transformer models to
compute attention. Unlike previous attention visualization techniques, our
approach enables the analysis of global patterns across multiple input
sequences. We create an interactive visualization tool, AttentionViz, based on
these joint query-key embeddings, and use it to study attention mechanisms in
both language and vision transformers. We demonstrate the utility of our
approach in improving model understanding and offering new insights about query-
key interactions through several application scenarios and expert feedback.

---

## Faith and Fate: Limits of Transformers on Compositionality

Nouha Dziri, Ximing Lu, Melanie Sclar, Xiang Lorraine Li, Liwei Jiang, Bill Yuchen Lin, Peter West, Chandra Bhagavatula, Ronan Le Bras, Jena D. Hwang, Soumya Sanyal, Sean Welleck, Xiang Ren, Allyson Ettinger, Zaid Harchaoui, Yejin Choi

Category: transformer-other
Keywords: Transformers, compositionality, large language models, multi-step reasoning, computation graphs
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

## BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models

Junnan Li, Dongxu Li, Silvio Savarese, Steven Hoi

Category: transformer-other
Keywords: vision-language pre-training, BLIP-2, Querying Transformer, image encoders, large language models
Year: 2023

The cost of vision-and-language pre-training has become increasingly prohibitive
due to end-to-end training of large-scale models. This paper proposes BLIP-2, a
generic and efficient pre-training strategy that bootstraps vision-language pre-
training from off-the-shelf frozen pre-trained image encoders and frozen large
language models. BLIP-2 bridges the modality gap with a lightweight Querying
Transformer, which is pre-trained in two stages. The first stage bootstraps
vision-language representation learning from a frozen image encoder. The second
stage bootstraps vision-to-language generative learning from a frozen language
model. BLIP-2 achieves state-of-the-art performance on various vision-language
tasks, despite having significantly fewer trainable parameters than existing
methods. For example, our model outperforms Flamingo80B by 8.7% on zero-shot
VQAv2 with 54x fewer trainable parameters. We also demonstrate the model’s
emerging capabilities of zero-shot image-to-text generation that can follow
natural language instructions.

---

## PaLM-E: An Embodied Multimodal Language Model

Danny Driess, Fei Xia, Mehdi S. M. Sajjadi, Corey Lynch, Aakanksha Chowdhery, Brian Ichter, Ayzaan Wahid, Jonathan Tompson, Quan Vuong, Tianhe Yu, Wenlong Huang, Yevgen Chebotar, Pierre Sermanet, Daniel Duckworth, Sergey Levine, Vincent Vanhoucke, Karol Hausman, Marc Toussaint, Klaus Greff, Andy Zeng, Igor Mordatch, Pete Florence

Category: transformer-other
Keywords: embodied language models, multimodal, robotics, language model, visual-language
Year: 2023

Large language models have been demonstrated to perform complex tasks. However,
enabling general inference in the real world, e.g. for robotics problems, raises
the challenge of grounding. We propose embodied language models to directly
incorporate real-world continuous sensor modalities into language models and
thereby establish the link between words and percepts. Input to our embodied
language model are multi-modal sentences that interleave visual, continuous
state estimation, and textual input encodings. We train these encodings end-to-
end, in conjunction with a pretrained large language model, for multiple
embodied tasks including sequential robotic manipulation planning, visual
question answering, and captioning. Our evaluations show that PaLM-E, a single
large embodied multimodal model, can address a variety of embodied reasoning
tasks, from a variety of observation modalities, on multiple embodiments, and
further, exhibits positive transfer: the model benefits from diverse joint
training across internet-scale language, vision, and visual-language domains.
Our largest model, PaLM-E-562B with 562B parameters, in addition to being
trained on robotics tasks, is a visual-language generalist with state-of-the-art
performance on OK-VQA, and retains generalist language capabilities with
increasing scale.

---

## TORCHSCALE: Transformers at Scale

Shuming Ma, Hongyu Wang, Shaohan Huang, Wenhui Wang, Zewen Chi, Li Dong, Alon Benhaim, Barun Patra, Vishrav Chaudhary, Xia Song, Furu Wei

Category: transformer-other
Keywords: Transformers, Scaling, Modeling Techniques, Language Modeling, Neural Machine Translation
Year: 2022

Large Transformers have achieved state-of-the-art performance across many tasks.
Most open-source libraries on scaling Transformers focus on improving training
or inference with better parallelization. In this work, we present TORCHSCALE,
an open-source toolkit that allows researchers and developers to scale up
Transformers efficiently and effectively. TORCHSCALE has the implementation of
several modeling techniques, which can improve modeling generality and
capability, as well as training stability and efficiency. Experimental results
on language modeling and neural machine translation demonstrate that TORCHSCALE
can successfully scale Transformers to different sizes without tears. The
library is available at https://aka.ms/torchscale.

---

## FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness

Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Ré

Category: transformer-other
Keywords: FlashAttention, Transformers, self-attention, IO-awareness, memory efficiency
Year: 2022

Transformers are slow and memory-hungry on long sequences, since the time and
memory complexity of self-attention are quadratic in sequence length.
Approximate attention methods have attempted to address this problem by trading
off model quality to reduce the compute complexity, but often do not achieve
wall-clock speedup. We argue that a missing principle is making attention
algorithms IO-aware—accounting for reads and writes between levels of GPU
memory. We propose FlashAttention, an IO-aware exact attention algorithm that
uses tiling to reduce the number of memory reads/writes between GPU high
bandwidth memory (HBM) and GPU on-chip SRAM. We analyze the IO complexity of
FlashAttention, showing that it requires fewer HBM accesses than standard
attention, and is optimal for a range of SRAM sizes. We also extend
FlashAttention to block-sparse attention, yielding an approximate attention
algorithm that is faster than any existing approximate attention method.
FlashAttention trains Transformers faster than existing baselines: 15% end-to-
end wall-clock speedup on BERT-large (seq. length 512) compared to the MLPerf
1.1 training speed record, 3× speedup on GPT-2 (seq. length 1K), and 2.4×
speedup on long-range arena (seq. length 1K-4K). FlashAttention and block-sparse
FlashAttention enable longer context in Transformers, yielding higher quality
models (0.7 better perplexity on GPT-2 and 6.4 points of lift on long-document
classification) and entirely new capabilities: the first Transformers to achieve
better-than-chance performance on the Path-X challenge (seq. length 16K, 61.4%
accuracy) and Path-256 (seq. length 64K, 63.1% accuracy).

---

## LongT5: Efficient Text-To-Text Transformer for Long Sequences

Mandy Guo, Joshua Ainslie, David Uthus, Santiago Ontañón, Jianmo Ni, Yun-Hsuan Sung, Yinfei Yang

Category: transformer-other
Keywords: LongT5, Transformer, Long sequences, Attention mechanism, Summarization
Year: 2022

Recent work has shown that either (1) increasing the input length or (2)
increasing model size can improve the performance of Transformer-based neural
models. In this paper, we present LongT5, a new model that explores the effects
of scaling both the input length and model size at the same time. Specifically,
we integrate attention ideas from long-input transformers (ETC), and adopt
pretraining strategies from summarization pretraining (PEGASUS) into the
scalable T5 architecture. The result is a new attention mechanism we call
Transient Global (TGlobal), which mimics ETC’s local/global attention mechanism,
but without requiring additional side-inputs. We are able to achieve state-of-
the-art results on several summarization and question answering tasks, as well
as outperform the original T5 models on these tasks. We have open sourced our
architecture and training code, as well as our pre-trained model checkpoints.

---

## OCR-free Document Understanding Transformer

Geewook Kim, Teakgyu Hong, Moonbin Yim, Jeongyeon Nam, Jinyoung Park, Jinyeong Yim, Wonseok Hwang, Sangdoo Yun, Dongyoon Han, Seunghyun Park

Category: transformer-other
Keywords: Visual Document Understanding, Document Information Extraction, Optical Character Recognition, End-to-End Transformer
Year: 2022

Understanding document images (e.g., invoices) is a core but challenging task
since it requires complex functions such as reading text and a holistic
understanding of the document. Current Visual Document Understanding (VDU)
methods outsource the task of reading text to off-the-shelf Optical Character
Recognition (OCR) engines and focus on the understanding task with the OCR
outputs. Although such OCR-based approaches have shown promising performance,
they suffer from 1) high computational costs for using OCR; 2) inflexibility of
OCR models on languages or types of documents; 3) OCR error propagation to the
subsequent process. To address these issues, in this paper, we introduce a novel
OCR-free VDU model named Donut, which stands for Document understanding
transformer. As the first step in OCR-free VDU research, we propose a simple
architecture (i.e., Transformer) with a pre-training objective (i.e., cross-
entropy loss). Donut is conceptually simple yet effective. Through extensive
experiments and analyses, we show a simple OCR-free VDU model, Donut, achieves
state-of-the-art performances on various VDU tasks in terms of both speed and
accuracy. In addition, we offer a synthetic data generator that helps the model
pre-training to be flexible in various languages and domains. The code, trained
model, and synthetic data are available at https://github.com/clovaai/donut.

---

## On the Learning of Non-Autoregressive Transformers

Fei Huang, Tianhua Tao, Hao Zhou, Lei Li, Minlie Huang

Category: transformer-other
Keywords: Non-autoregressive Transformer, text generation, decoding latency, conditional total correlation, proxy distribution
Year: 2022

Non-autoregressive Transformer (NAT) is a family of text generation models,
which aims to reduce the decoding latency by predicting the whole sentences in
parallel. However, such latency reduction sacrifices the ability to capture
left-to-right dependencies, thereby making NAT learning very challenging. In
this paper, we present theoretical and empirical analyses to reveal the
challenges of NAT learning and propose a unified perspective to understand
existing successes. First, we show that simply training NAT by maximizing the
likelihood can lead to an approximation of marginal distributions but drops all
dependencies between tokens, where the dropped information can be measured by
the dataset’s conditional total correlation. Second, we formalize many previous
objectives in a unified framework and show that their success can be concluded
as maximizing the likelihood on a proxy distribution, leading to a reduced
information loss. Empirical studies show that our perspective can explain the
phenomena in NAT learning and guide the design of new training methods.

---

## FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness

Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Ré

Category: transformer-other
Keywords: Transformers, FlashAttention, memory efficiency, IO-awareness, block-sparse attention
Year: 2022

Transformers are slow and memory-hungry on long sequences, since the time and
memory complexity of self-attention are quadratic in sequence length.
Approximate attention methods have attempted to address this problem by trading
off model quality to reduce the compute complexity but often do not achieve
wall-clock speedup. We argue that a missing principle is making attention
algorithms IO-aware—accounting for reads and writes between levels of GPU
memory. We propose FlashAttention, an IO-aware exact attention algorithm that
uses tiling to reduce the number of memory reads/writes between GPU high
bandwidth memory (HBM) and GPU on-chip SRAM. We analyze the IO complexity of
FlashAttention, showing that it requires fewer HBM accesses than standard
attention, and is optimal for a range of SRAM sizes. We also extend
FlashAttention to block-sparse attention, yielding an approximate attention
algorithm that is faster than any existing approximate attention method.
FlashAttention trains Transformers faster than existing baselines: 15% end-to-
end wall-clock speedup on BERT-large (seq. length 512) compared to the MLPerf
1.1 training speed record, 3× speedup on GPT-2 (seq. length 1K), and 2.4×
speedup on long-range arena (seq. length 1K-4K). FlashAttention and block-sparse
FlashAttention enable longer context in Transformers, yielding higher quality
models (0.7 better perplexity on GPT-2 and 6.4 points of lift on long-document
classification) and entirely new capabilities: the first Transformers to achieve
better-than-chance performance on the Path-X challenge (seq. length 16K, 61.4%
accuracy) and Path-256 (seq. length 64K, 63.1% accuracy).

---

## Memorizing Transformers

Yuhuai Wu, Markus N. Rabe, DeLesley Hutchins, Christian Szegedy

Category: transformer-other
Keywords: language models, transformers, memorization, approximate kNN, non-differentiable memory
Year: 2022

Language models typically need to be trained or finetuned in order to acquire
new knowledge, which involves updating their weights. We instead envision
language models that can simply read and memorize new data at inference time,
thus acquiring new knowledge immediately. In this work, we extend language
models with the ability to memorize the internal representations of past inputs.
We demonstrate that an approximate kNN lookup into a non-differentiable memory
of recent (key, value) pairs improves language modeling across various
benchmarks and tasks, including generic webtext (C4), math papers (arXiv), books
(PG-19), code (Github), as well as formal theorems (Isabelle). We show that the
performance steadily improves when we increase the size of memory up to 262K
tokens. On benchmarks including code and mathematics, we find that the model is
capable of making use of newly defined functions and theorems during test time.

---

## Block-Recurrent Transformers

DeLesley Hutchins, Imanol Schlag, Yuhuai Wu, Ethan Dyer, Behnam Neyshabur

Category: transformer-other
Keywords: Block-Recurrent Transformer, transformer, recurrent cell, self-attention, cross-attention, LSTM-style gates, language modeling, long sequences
Year: 2022

We introduce the Block-Recurrent Transformer, which applies a transformer layer
in a recurrent fashion along a sequence, and has linear complexity with respect
to sequence length. Our recurrent cell operates on blocks of tokens rather than
single tokens during training, and leverages parallel computation within a block
in order to make efficient use of accelerator hardware. The cell itself is
strikingly simple. It is merely a transformer layer: it uses self-attention and
cross-attention to efficiently compute a recurrent function over a large set of
state vectors and tokens. Our design was inspired in part by LSTM cells, and it
uses LSTM-style gates, but it scales the typical LSTM cell up by several orders
of magnitude. Our implementation of recurrence has the same cost in both
computation time and parameter count as a conventional transformer layer, but
offers dramatically improved perplexity in language modeling tasks over very
long sequences. Our model out-performs a long-range Transformer XL baseline by a
wide margin, while running twice as fast. We demonstrate its effectiveness on
PG19 (books), arXiv papers, and GitHub source code. Our code has been released
as open source.

---

## VLMO: Unified Vision-Language Pre-Training with Mixture-of-Modality-Experts

Hangbo Bao, Wenhui Wang, Li Dong, Qiang Liu, Owais Khan Mohammed, Kriti Aggarwal, Subhojit Som, Furu Wei

Category: transformer-other
Keywords: Vision-Language Model, Mixture-of-Modality-Experts, Transformer, Pre-training, Image-Text Retrieval, Vision-Language Classification
Year: 2022

We present a unified Vision-Language pretrained Model (VLMO) that jointly learns
a dual encoder and a fusion encoder with a modular Transformer network.
Specifically, we introduce Mixture-of-Modality-Experts (MOME) Transformer, where
each block contains a pool of modality-specific experts and a shared self-
attention layer. Because of the modeling flexibility of MOME, pretrained VLMO
can be fine-tuned as a fusion encoder for vision-language classification tasks,
or used as a dual encoder for efficient image-text retrieval. Moreover, we
propose a stagewise pre-training strategy, which effectively leverages large-
scale image-only and text-only data besides image-text pairs. Experimental
results show that VLMO achieves state-of-the-art results on various vision-
language tasks, including VQA, NLVR2 and image-text retrieval. The code and
pretrained models are available at https://aka.ms/vlmo.

---

## RoFormer: Enhanced Transformer with Rotary Position Embedding

Jianlin Su, Yu Lu, Shengfeng Pan, Ahmed Murtadha, Bo Wen, Yunfeng Liu

Category: transformer-other
Keywords: Pre-trained Language Models, Position Information Encoding, Pre-training, Natural Language Processing
Year: 2022

Position encoding recently has shown effective in the transformer architecture.
It enables valuable supervision for dependency modeling between elements at
different positions of the sequence. In this paper, we first investigate various
methods to integrate positional information into the learning process of
transformer-based language models. Then, we propose a novel method named Rotary
Position Embedding (RoPE) to effectively leverage the positional information.
Specifically, the proposed RoPE encodes the absolute position with a rotation
matrix and meanwhile incorporates the explicit relative position dependency in
self-attention formulation. Notably, RoPE enables valuable properties, including
the flexibility of sequence length, decaying inter-token dependency with
increasing relative distances, and the capability of equipping the linear self-
attention with relative position encoding. Finally, we evaluate the enhanced
transformer with rotary position embedding, also called RoFormer, on various
long text classification benchmark datasets. Our experiments show that it
consistently overcomes its alternatives. Furthermore, we provide a theoretical
analysis to explain some experimental results. RoFormer is already integrated
into Huggingface: https://huggingface.co/docs/transformers/model_doc/roformer.

---

## Pure Transformers are Powerful Graph Learners

Jinwoo Kim, Tien Dat Nguyen, Seonwoo Min, Sungjun Cho, Moontae Lee, Honglak Lee, Seunghoon Hong

Category: transformer-other
Keywords: Transformers, Graph learning, Graph Neural Networks, Tokenized Graph Transformer, Self-attention
Year: 2022

We show that standard Transformers without graph-specific modifications can lead
to promising results in graph learning both in theory and practice. Given a
graph, we simply treat all nodes and edges as independent tokens, augment them
with token embeddings, and feed them to a Transformer. With an appropriate
choice of token embeddings, we prove that this approach is theoretically at
least as expressive as an invariant graph network (2-IGN) composed of
equivariant linear layers, which is already more expressive than all message-
passing Graph Neural Networks (GNN). When trained on a large-scale graph dataset
(PCQM4Mv2), our method coined Tokenized Graph Transformer (TokenGT) achieves
significantly better results compared to GNN baselines and competitive results
compared to Transformer variants with sophisticated graph-specific inductive
bias. Our implementation is available at https://github.com/jw9730/tokengt.

---

## ByT5: Towards a Token-Free Future with Pre-trained Byte-to-Byte Models

Linting Xue, Aditya Barua, Noah Constant, Rami Al-Rfou, Sharan Narang, Mihir Kale, Adam Roberts, Colin Raffel

Category: transformer-other
Keywords: token-free models, byte-level models, Transformer architecture, NLP, pre-trained models
Year: 2022

Most widely used pre-trained language models operate on sequences of tokens
corresponding to word or subword units. By comparison, token-free models that
operate directly on raw text (bytes or characters) have many benefits: They can
process text in any language out of the box, they are more robust to noise, and
they minimize technical debt by removing complex and error-prone text
preprocessing pipelines. Because byte or character sequences are longer than
token sequences, past work on token-free models has often introduced new model
architectures designed to amortize the cost of operating directly on raw text.
In this paper, we show that a standard Transformer architecture can be used with
minimal modifications to process byte sequences. We characterize the trade-offs
in terms of parameter count, training FLOPs, and inference speed, and show that
byte-level models are competitive with their token-level counterparts. We also
demonstrate that byte-level models are significantly more robust to noise and
perform better on tasks that are sensitive to spelling and pronunciation. As
part of our contribution, we release a new set of pre-trained byte-level
Transformer models based on the T5 architecture, as well as all code and data
used in our experiments.

---

## Block-Recurrent Transformers

DeLesley Hutchins, Imanol Schlag, Yuhuai Wu, Ethan Dyer, Behnam Neyshabur

Category: transformer-other
Keywords: Block-Recurrent Transformer, sequence processing, linear complexity, LSTM, language modeling
Year: 2022

We introduce the Block-Recurrent Transformer, which applies a transformer layer
in a recurrent fashion along a sequence, and has linear complexity with respect
to sequence length. Our recurrent cell operates on blocks of tokens rather than
single tokens during training, and leverages parallel computation within a block
in order to make efficient use of accelerator hardware. The cell itself is
strikingly simple. It is merely a transformer layer: it uses self-attention and
cross-attention to efficiently compute a recurrent function over a large set of
state vectors and tokens. Our design was inspired in part by LSTM cells, and it
uses LSTM-style gates, but it scales the typical LSTM cell up by several orders
of magnitude. Our implementation of recurrence has the same cost in both
computation time and parameter count as a conventional transformer layer, but
offers dramatically improved perplexity in language modeling tasks over very
long sequences. Our model out-performs a long-range Transformer XL baseline by a
wide margin, while running twice as fast. We demonstrate its effectiveness on
PG19 (books), arXiv papers, and GitHub source code. Our code has been released
as open source.

---

## Downstream Transformer Generation of Question-Answer Pairs with Preprocessing and Postprocessing Pipelines

Cheng Zhang, Hao Zhang, Jie Wang

Category: transformer-other
Keywords: transformers, question-answer pairs, T5 model, preprocessing, postprocessing, SQuAD dataset, Gaokao-EN dataset
Year: 2022

We present a system called TP3 to perform a downstream task of transformers on
generating question-answer pairs (QAPs) from a given article. TP3 first
finetunes pretrained transformers on QAP datasets, then uses a preprocessing
pipeline to select appropriate answers, feeds the relevant sentences and the
answer to the finetuned transformer to generate candidate QAPs, and finally uses
a postprocessing pipeline to filter inadequate QAPs. In particular, using
pretrained T5 models as transformers and the SQuAD dataset as the finetuning
dataset, we show that TP3 generates a satisfactory number of QAPs with high
qualities on the Gaokao-EN dataset.

---

## EXT5: Towards Extreme Multi-Task Scaling for Transfer Learning

Vamsi Aribandi, Yi Tay, Tal Schuster, Jinfeng Rao, Huaixiu Steven Zheng, Sanket Vaibhav Mehta, Honglei Zhuang, Vinh Q. Tran, Dara Bahri, Jianmo Ni, Jai Gupta, Kai Hui, Sebastian Ruder, Donald Metzler

Category: transformer-other
Keywords: multi-task learning, transfer learning, natural language processing, EXT5, EXMIX
Year: 2022

Despite the recent success of multi-task learning and transfer learning for
natural language processing (NLP), few works have systematically studied the
effect of scaling up the number of tasks during pre-training. Towards this goal,
this paper introduces EXMIX (Extreme Mixture): a massive collection of 107
supervised NLP tasks across diverse domains and task-families. Using EXMIX, we
study the effect of multi-task pre-training at the largest scale to date, and
analyze co-training transfer amongst common families of tasks. Through this
analysis, we show that manually curating an ideal set of tasks for multi-task
pre-training is not straightforward, and that multi-task scaling can vastly
improve models on its own. Finally, we propose EXT5: a model pre-trained using a
multi-task objective of self-supervised span denoising and supervised EXMIX. Via
extensive experiments, we show that EXT5 outperforms strong T5 baselines on
SuperGLUE, GEM, Rainbow, Closed-Book QA tasks, and several tasks outside of
EXMIX. EXT5 also significantly improves sample efficiency while pre-training.

---

## ByT5: Towards a Token-Free Future with Pre-trained Byte-to-Byte Models

Linting Xue, Aditya Barua, Noah Constant, Rami Al-Rfou, Sharan Narang, Mihir Kale, Adam Roberts, Colin Raffel

Category: transformer-other
Keywords: token-free models, Transformer, byte-level models, NLP, pre-trained models
Year: 2022

Most widely used pre-trained language models operate on sequences of tokens
corresponding to word or subword units. By comparison, token-free models that
operate directly on raw text (bytes or characters) have many benefits: They can
process text in any language out of the box, they are more robust to noise, and
they minimize technical debt by removing complex and error-prone text
preprocessing pipelines. Because byte or character sequences are longer than
token sequences, past work on token-free models has often introduced new model
architectures designed to amortize the cost of operating directly on raw text.
In this paper, we show that a standard Transformer architecture can be used with
minimal modifications to process byte sequences. We characterize the trade-offs
in terms of parameter count, training FLOPs, and inference speed, and show that
byte-level models are competitive with their token-level counterparts. We also
demonstrate that byte-level models are significantly more robust to noise and
perform better on tasks that are sensitive to spelling and pronunciation. As
part of our contribution, we release a new set of pre-trained byte-level
Transformer models based on the T5 architecture, as well as all code and data
used in our experiments.

---

## CHARFORMER: Fast Character Transformers via Gradient-Based Subword Tokenization

Yi Tay, Vinh Q. Tran, Sebastian Ruder, Jai Gupta, Hyung Won Chung, Dara Bahri, Zhen Qin, Simon Baumgartner, Cong Yu, Donald Metzler

Category: transformer-other
Keywords: subword tokenization, gradient-based, CHARFORMER, deep Transformer, language processing
Year: 2022

State-of-the-art models in natural language processing rely on separate rigid
subword tokenization algorithms, which limit their generalization ability and
adaptation to new settings. In this paper, we propose a new model inductive bias
that learns a subword tokenization end-to-end as part of the model. To this end,
we introduce a soft gradient-based subword tokenization module (GBST) that
automatically learns latent subword representations from characters in a data-
driven fashion. Concretely, GBST enumerates candidate subword blocks and learns
to score them in a position-wise fashion using a block scoring network. We
additionally introduce CHARFORMER, a deep Transformer model that integrates GBST
and operates on the byte level. Via extensive experiments on English GLUE,
multilingual, and noisy text datasets, we show that CHARFORMER outperforms a
series of competitive byte-level baselines while generally performing on par and
sometimes outperforming subword-based models. Additionally, CHARFORMER is fast,
improving the speed of both vanilla byte-level and subword-level Transformers by
28-100% while maintaining competitive quality. We believe this work paves the
way for highly performant token-free models that are trained completely end-to-
end.

---

## Language Models are General-Purpose Interfaces

Yaru Hao, Haoyu Song, Li Dong, Shaohan Huang, Zewen Chi, Wenhui Wang, Shuming Ma, Furu Wei

Category: transformer-other
Keywords: language models, foundation models, general-purpose interface, semi-causal language modeling, multimodal
Year: 2022

Foundation models have received much attention due to their effectiveness across
a broad range of downstream applications. Though there is a big convergence in
terms of architecture, most pretrained models are typically still developed for
specific tasks or modalities. In this work, we propose to use language models as
a general-purpose interface to various foundation models. A collection of
pretrained encoders perceive diverse modalities (such as vision, and language),
and they dock with a language model that plays the role of a universal task
layer. We propose a semi-causal language modeling objective to jointly pretrain
the interface and the modular encoders. We subsume the advantages and
capabilities from both causal and non-causal modeling, thereby combining the
best of two worlds. Specifically, the proposed method not only inherits the
capabilities of in-context learning and open-ended generation from causal
language modeling, but also is conducive to finetuning because of the
bidirectional encoders. More importantly, our approach seamlessly unlocks the
combinations of the above capabilities, e.g., enabling in-context learning or
instruction following with finetuned encoders. Experimental results across
various language-only and vision-language benchmarks show that our model
outperforms or is competitive with specialized models on finetuning, zero-shot
generalization, and few-shot learning.

---

## Pure Transformers are Powerful Graph Learners

Jinwoo Kim, Tien Dat Nguyen, Seonwoo Min, Sungjun Cho, Moontae Lee, Honglak Lee, Seunghoon Hong

Category: transformer-other
Keywords: Transformer, graph learning, Tokenized Graph Transformer, graph neural networks, self-attention
Year: 2022

We show that standard Transformers without graph-specific modifications can lead
to promising results in graph learning both in theory and practice. Given a
graph, we simply treat all nodes and edges as independent tokens, augment them
with token embeddings, and feed them to a Transformer. With an appropriate
choice of token embeddings, we prove that this approach is theoretically at
least as expressive as an invariant graph network (2-IGN) composed of
equivariant linear layers, which is already more expressive than all message-
passing Graph Neural Networks (GNN). When trained on a large-scale graph dataset
(PCQM4Mv2), our method coined Tokenized Graph Transformer (TokenGT) achieves
significantly better results compared to GNN baselines and competitive results
compared to Transformer variants with sophisticated graph-specific inductive
bias.

---

## Perceiver IO: A General Architecture for Structured Inputs & Outputs

Andrew Jaegle, Sebastian Borgeaud, Jean-Baptiste Alayrac, Carl Doersch, Catalin Ionescu, David Ding, Skanda Koppula, Daniel Zoran, Andrew Brock, Evan Shelhamer, Olivier Hénaff, Matthew M. Botvinick, Andrew Zisserman, Oriol Vinyals, João Carreira

Category: transformer-other
Keywords: Perceiver IO, structured inputs and outputs, machine learning, general-purpose architecture, multi-modal reasoning
Year: 2022

A central goal of machine learning is the development of systems that can solve
many problems in as many data domains as possible. Current architectures,
however, cannot be applied beyond a small set of stereotyped settings, as they
bake in domain & task assumptions or scale poorly to large inputs or outputs. In
this work, we propose Perceiver IO, a general-purpose architecture that handles
data from arbitrary settings while scaling linearly with the size of inputs and
outputs. Our model augments the Perceiver with a flexible querying mechanism
that enables outputs of various sizes and semantics, doing away with the need
for task-specific architecture engineering. The same architecture achieves
strong results on tasks spanning natural language and visual understanding,
multi-task and multi-modal reasoning, and StarCraft II. As highlights, Perceiver
IO outperforms a Transformer-based BERT baseline on the GLUE language benchmark
despite removing input tokenization and achieves state-of-the-art performance on
Sintel optical flow estimation with no explicit mechanisms for multiscale
correspondence.

---

## Memorizing Transformers

Yuhuai Wu, Markus N. Rabe, DeLesley Hutchins, Christian Szegedy

Category: transformer-other
Keywords: language models, transformers, k-nearest-neighbor, memory, attention, information retrieval
Year: 2022

Language models typically need to be trained or finetuned in order to acquire
new knowledge, which involves updating their weights. We instead envision
language models that can simply read and memorize new data at inference time,
thus acquiring new knowledge immediately. In this work, we extend language
models with the ability to memorize the internal representations of past inputs.
We demonstrate that an approximate kNN lookup into a non-differentiable memory
of recent (key, value) pairs improves language modeling across various
benchmarks and tasks, including generic webtext (C4), math papers (arXiv), books
(PG-19), code (Github), as well as formal theorems (Isabelle). We show that the
performance steadily improves when we increase the size of memory up to 262K
tokens. On benchmarks including code and mathematics, we find that the model is
capable of making use of newly defined functions and theorems during test time.

---

## Perceiver IO: A General Architecture for Structured Inputs & Outputs

Andrew Jaegle, Sebastian Borgeaud, Jean-Baptiste Alayrac, Carl Doersch, Catalin Ionescu, David Ding, Skanda Koppula, Daniel Zoran, Andrew Brock, Evan Shelhamer, Olivier Hénaff, Matthew M. Botvinick, Andrew Zisserman, Oriol Vinyals, João Carreira

Category: transformer-other
Keywords: Perceiver IO, general architecture, structured inputs, structured outputs, multi-modal reasoning, machine learning
Year: 2022

A central goal of machine learning is the development of systems that can solve
many problems in as many data domains as possible. Current architectures,
however, cannot be applied beyond a small set of stereotyped settings, as they
bake in domain & task assumptions or scale poorly to large inputs or outputs. In
this work, we propose Perceiver IO, a general-purpose architecture that handles
data from arbitrary settings while scaling linearly with the size of inputs and
outputs. Our model augments the Perceiver with a flexible querying mechanism
that enables outputs of various sizes and semantics, doing away with the need
for task-specific architecture engineering. The same architecture achieves
strong results on tasks spanning natural language and visual understanding,
multi-task and multi-modal reasoning, and StarCraft II. As highlights, Perceiver
IO outperforms a Transformer-based BERT baseline on the GLUE language benchmark
despite removing input tokenization and achieves state-of-the-art performance on
Sintel optical flow estimation with no explicit mechanisms for multiscale
correspondence.

---

## Predicting the Future with Transformer-Based Models

Jane Doe, John Smith

Category: transformer-other
Keywords: transformer, time series, prediction, self-attention, temporal convolutions
Year: 2022

This paper explores the application of transformer-based models for time series
prediction. We present a novel architecture that combines self-attention
mechanisms with temporal convolutions to improve forecasting accuracy. Our
extensive experiments demonstrate the superiority of our model compared to
traditional methods in various datasets, showcasing its potential for real-world
applications.

---

## Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity

William Fedus, Barret Zoph, Noam Shazeer

Category: transformer-other
Keywords: mixture-of-experts, natural language processing, sparsity, large-scale machine learning, distributed computing
Year: 2022

In deep learning, models typically reuse the same parameters for all inputs.
Mixture of Experts (MoE) models defy this and instead select different
parameters for each incoming example. The result is a sparsely-activated
model—with an outrageous number of parameters—but a constant computational cost.
However, despite several notable successes of MoE, widespread adoption has been
hindered by complexity, communication costs, and training instability. We
address these with the introduction of the Switch Transformer. We simplify the
MoE routing algorithm and design intuitive improved models with reduced
communication and computational costs. Our proposed training techniques mitigate
the instabilities, and we show large sparse models may be trained, for the first
time, with lower precision (bfloat16) formats. We design models based off
T5-Base and T5-Large (Raffel et al., 2019) to obtain up to 7x increases in pre-
training speed with the same computational resources. These improvements extend
into multilingual settings where we measure gains over the mT5-Base version
across all 101 languages. Finally, we advance the current scale of language
models by pre-training up to trillion parameter models on the 'Colossal Clean
Crawled Corpus', and achieve a 4x speedup over the T5-XXL model.

---

## EncT5: A Framework for Fine-tuning T5 as Non-autoregressive Models

Frederick Liu, Terry Huang, Shihang Lyu, Siamak Shakeri, Hongkun Yu, Jing Li

Category: transformer-other
Keywords: EncT5, T5, encoder-decoder, non-autoregressive models, fine-tuning, classification, structured prediction
Year: 2022

Pre-trained encoder-decoder transformer architectures have become increasingly
popular with the advent of T5 models. T5 has also become more favorable over
other architectures like BERT due to the amount of data that it is pre-trained
on, increased scale of model parameter sizes, and easy applicability to a
diverse set of tasks due to the generative nature of the model. While being able
to generalize to a wide variety of tasks, it is not clear that encoder-decoder
architectures are the most efficient for fine-tuning tasks that don’t require
auto-regressive decoding. In this work, we study fine-tuning pre-trained
encoder-decoder models for tasks such as classification, multi-label
classification, and structured prediction. We propose EncT5, a framework for
these problems, and illustrate instantiations for these tasks. Our experiment
results show that EncT5 has advantages over T5 such as efficiency and usability,
and outperforms BERT when evaluated on publicly available pre-trained
checkpoints.

---

## FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness

Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Ré

Category: transformer-other
Keywords: Transformers, IO-awareness, FlashAttention, self-attention, memory efficiency
Year: 2022

Transformers are slow and memory-hungry on long sequences, since the time and
memory complexity of self-attention are quadratic in sequence length.
Approximate attention methods have attempted to address this problem by trading
off model quality to reduce the compute complexity, but often do not achieve
wall-clock speedup. We argue that a missing principle is making attention
algorithms IO-aware—accounting for reads and writes between levels of GPU
memory. We propose FlashAttention, an IO-aware exact attention algorithm that
uses tiling to reduce the number of memory reads/writes between GPU high
bandwidth memory (HBM) and GPU on-chip SRAM. We analyze the IO complexity of
FlashAttention, showing that it requires fewer HBM accesses than standard
attention, and is optimal for a range of SRAM sizes. We also extend
FlashAttention to block-sparse attention, yielding an approximate attention
algorithm that is faster than any existing approximate attention method.
FlashAttention trains Transformers faster than existing baselines: 15% end-to-
end wall-clock speedup on BERT-large (seq. length 512) compared to the MLPerf
1.1 training speed record, 3× speedup on GPT-2 (seq. length 1K), and 2.4×
speedup on long-range arena (seq. length 1K-4K). FlashAttention and block-sparse
FlashAttention enable longer context in Transformers, yielding higher quality
models (0.7 better perplexity on GPT-2 and 6.4 points of lift on long-document
classification) and entirely new capabilities: the first Transformers to achieve
better-than-chance performance on the Path-X challenge (seq. length 16K, 61.4%
accuracy) and Path-256 (seq. length 64K, 63.1% accuracy).

---

## Challenges in large scale training of Giant Transformers on Google TPU machines

Sameer Kumar

Category: transformer-other
Keywords: Giant Transformers, Google TPU, large-scale training, machine learning, optimization
Year: 2021

The document discusses the challenges encountered while training large-scale
Giant Transformers on Google TPU machines. It explores the technical
difficulties, optimization strategies, and the performance considerations that
are critical when scaling up transformer models on specialized hardware like
TPUs. The focus is on addressing these challenges to achieve efficient and
effective training processes.

---

## OCR-free Document Understanding Transformer

Geewook Kim, Teakgyu Hong, Moonbin Yim, JeongYeon Nam, Jinyoung Park, Jinyeong Yim, Wonseok Hwang, Sangdoo Yun, Dongyoon Han, Seunghyun Park

Category: transformer-other
Keywords: Visual Document Understanding, Document Information Extraction, Optical Character Recognition, End-to-End Transformer
Year: 2021

Understanding document images (e.g., invoices) is a core but challenging task
since it requires complex functions such as reading text and a holistic
understanding of the document. Current Visual Document Understanding (VDU)
methods outsource the task of reading text to off-the-shelf Optical Character
Recognition (OCR) engines and focus on the understanding task with the OCR
outputs. Although such OCR-based approaches have shown promising performance,
they suffer from 1) high computational costs for using OCR; 2) inflexibility of
OCR models on languages or types of documents; 3) OCR error propagation to the
subsequent process. To address these issues, in this paper, we introduce a novel
OCR-free VDU model named Donut, which stands for Document understanding
transformer. As the first step in OCR-free VDU research, we propose a simple
architecture (i.e., Transformer) with a pre-training objective (i.e., cross-
entropy loss). Donut is conceptually simple yet effective. Through extensive
experiments and analyses, we show a simple OCR-free VDU model, Donut, achieves
state-of-the-art performances on various VDU tasks in terms of both speed and
accuracy. In addition, we offer a synthetic data generator that helps the model
pre-training to be flexible in various languages and domains. The code, trained
model, and synthetic data are available at https://github.com/clovaai/donut.

---

## Emerging Properties in Self-Supervised Vision Transformers

Mathilde Caron, Hugo Touvron, Ishan Misra, Hervé Jegou, Julien Mairal, Piotr Bojanowski, Armand Joulin

Category: transformer-other
Keywords: Vision Transformer, self-supervised learning, semantic segmentation, k-NN classifiers, DINO
Year: 2021

In this paper, we question if self-supervised learning provides new properties
to Vision Transformer (ViT) that stand out compared to convolutional networks
(convnets). Beyond the fact that adapting self-supervised methods to this
architecture works particularly well, we make the following observations: first,
self-supervised ViT features contain explicit information about the semantic
segmentation of an image, which does not emerge as clearly with supervised ViTs,
nor with convnets. Second, these features are also excellent k-NN classifiers,
reaching 78.3% top-1 on ImageNet with a small ViT. Our study also underlines the
importance of momentum encoder, multi-crop training, and the use of small
patches with ViTs. We implement our findings into a simple self-supervised
method, called DINO, which we interpret as a form of self-distillation with no
labels. We show the synergy between DINO and ViTs by achieving 80.1% top-1 on
ImageNet in linear evaluation with ViT-Base.

---

## Emerging Properties in Self-Supervised Vision Transformers

Mathilde Caron, Hugo Touvron, Ishan Misra, Hervé Jegou, Julien Mairal, Piotr Bojanowski, Armand Joulin

Category: transformer-other
Keywords: Vision Transformer, self-supervised learning, semantic segmentation, k-NN classifier, DINO
Year: 2021

In this paper, we question if self-supervised learning provides new properties
to Vision Transformer (ViT) that stand out compared to convolutional networks
(convnets). Beyond the fact that adapting self-supervised methods to this
architecture works particularly well, we make the following observations: first,
self-supervised ViT features contain explicit information about the semantic
segmentation of an image, which does not emerge as clearly with supervised ViTs,
nor with convnets. Second, these features are also excellent k-NN classifiers,
reaching 78.3% top-1 on ImageNet with a small ViT. Our study also underlines the
importance of momentum encoder, multi-crop training, and the use of small
patches with ViTs. We implement our findings into a simple self-supervised
method, called DINO, which we interpret as a form of self-distillation with no
labels. We show the synergy between DINO and ViTs by achieving 80.1% top-1 on
ImageNet in linear evaluation with ViT-Base.

---

## Do Long-Range Language Models Actually Use Long-Range Context?

Simeng Sun, Kalpesh Krishna, Andrew Mattarella-Micke, Mohit Iyyer

Category: transformer-other
Keywords: language models, long-range context, self-attention, transformers, PG-19
Year: 2021

Language models are generally trained on short, truncated input sequences, which
limits their ability to use discourse-level information present in long-range
context to improve their predictions. Recent efforts to improve the efficiency
of self-attention have led to a proliferation of long-range Transformer language
models, which can process much longer sequences than models of the past.
However, the ways in which such models take advantage of the long-range context
remain unclear. In this paper, we perform a fine-grained analysis of two long-
range Transformer language models (including the Routing Transformer, which
achieves state-of-the-art perplexity on the PG-19 long-sequence LM benchmark
dataset) that accept input sequences of up to 8K tokens. Our results reveal that
providing long-range context (i.e., beyond the previous 2K tokens) to these
models only improves their predictions on a small set of tokens (e.g., those
that can be copied from the distant context) and does not help at all for
sentence-level prediction tasks. Finally, we discover that PG-19 contains a
variety of different document types and domains, and that long-range context
helps most for literary novels (as opposed to textbooks or magazines).

---

## A Generalization of Transformer Networks to Graphs

Vijay Prakash Dwivedi, Xavier Bresson

Category: transformer-other
Keywords: transformer, graphs, neural networks, graph transformer, Laplacian eigenvectors
Year: 2021

We propose a generalization of transformer neural network architecture for
arbitrary graphs. The original transformer was designed for Natural Language
Processing (NLP), which operates on fully connected graphs representing all
connections between the words in a sequence. Such architecture does not leverage
the graph connectivity inductive bias, and can perform poorly when the graph
topology is important and has not been encoded into the node features. We
introduce a graph transformer with four new properties compared to the standard
model. First, the attention mechanism is a function of the neighborhood
connectivity for each node in the graph. Second, the positional encoding is
represented by the Laplacian eigenvectors, which naturally generalize the
sinusoidal positional encodings often used in NLP. Third, the layer
normalization is replaced by a batch normalization layer, which provides faster
training and better generalization performance. Finally, the architecture is
extended to edge feature representation, which can be critical to tasks such as
chemistry (bond type) or link prediction (entity relationship in knowledge
graphs). Numerical experiments on a graph benchmark demonstrate the performance
of the proposed graph transformer architecture. This work closes the gap between
the original transformer, which was designed for the limited case of line
graphs, and graph neural networks, that can work with arbitrary graphs. As our
architecture is simple and generic, we believe it can be used as a black box for
future applications that wish to consider transformer and graphs.

---

## Big Bird: Transformers for Longer Sequences

Manzil Zaheer, Guru Guruganesh, Avinava Dubey, Joshua Ainslie, Chris Alberti, Santiago Ontanon, Philip Pham, Anirudh Ravula, Qifan Wang, Li Yang, Amr Ahmed

Category: transformer-other
Keywords: Transformers, NLP, Sparse Attention, Sequence Functions, Genomics
Year: 2021

Transformers-based models, such as BERT, have been one of the most successful
deep learning models for NLP. Unfortunately, one of their core limitations is
the quadratic dependency (mainly in terms of memory) on the sequence length due
to their full attention mechanism. To remedy this, we propose, BIGBIRD, a sparse
attention mechanism that reduces this quadratic dependency to linear. We show
that BIGBIRD is a universal approximator of sequence functions and is Turing
complete, thereby preserving these properties of the quadratic, full attention
model. Along the way, our theoretical analysis reveals some of the benefits of
having O(1) global tokens (such as CLS), that attend to the entire sequence as
part of the sparse attention mechanism. The proposed sparse attention can handle
sequences of length up to 8x of what was previously possible using similar
hardware. As a consequence of the capability to handle longer context, BIGBIRD
drastically improves performance on various NLP tasks such as question answering
and summarization. We also propose novel applications to genomics data.

---

## Capturing Row and Column Semantics in Transformer Based Question Answering over Tables

Michael Glass, Mustafa Canim, Alfio Gliozzo, Saneem Chemmengath, Vishwajeet Kumar, Rishav Chakravarti, Avi Sil, Feifei Pan, Samarth Bharadwaj, Nicolas Rodolfo Fauceglia

Category: transformer-other
Keywords: transformer, question answering, tables, row-column intersection, RCI
Year: 2021

Transformer based architectures are recently used for the task of answering
questions over tables. In order to improve the accuracy on this task,
specialized pre-training techniques have been developed and applied on millions
of open-domain web tables. In this paper, we propose two novel approaches
demonstrating that one can achieve superior performance on table QA task without
even using any of these specialized pre-training techniques. The first model,
called RCI interaction, leverages a transformer based architecture that
independently classifies rows and columns to identify relevant cells. While this
model yields extremely high accuracy at finding cell values on recent
benchmarks, a second model we propose, called RCI representation, provides a
significant efficiency advantage for online QA systems over tables by
materializing embeddings for existing tables. Experiments on recent benchmarks
prove that the proposed methods can effectively locate cell values on tables (up
to ∼98% Hit@1 accuracy on WikiSQL lookup questions). Also, the interaction model
outperforms the state-of-the-art transformer based approaches, pre-trained on
very large table corpora (TAPAS and TABERT), achieving ∼3.4% and ∼18.86%
additional precision improvement on the standard WikiSQL benchmark.

---

## Big Bird: Transformers for Longer Sequences

Manzil Zaheer, Guru Guruganesh, Avinava Dubey, Joshua Ainslie, Chris Alberti, Santiago Ontanon, Philip Pham, Anirudh Ravula, Qifan Wang, Li Yang, Amr Ahmed

Category: transformer-other
Keywords: Big Bird, Transformers, Sparse Attention, Sequence Length, NLP
Year: 2021

Transformers-based models, such as BERT, have been one of the most successful
deep learning models for NLP. Unfortunately, one of their core limitations is
the quadratic dependency (mainly in terms of memory) on the sequence length due
to their full attention mechanism. To remedy this, we propose, BIGBIRD, a sparse
attention mechanism that reduces this quadratic dependency to linear. We show
that BIGBIRD is a universal approximator of sequence functions and is Turing
complete, thereby preserving these properties of the quadratic, full attention
model. Along the way, our theoretical analysis reveals some of the benefits of
having O(1) global tokens (such as CLS), that attend to the entire sequence as
part of the sparse attention mechanism. The proposed sparse attention can handle
sequences of length up to 8x of what was previously possible using similar
hardware. As a consequence of the capability to handle longer context, BIGBIRD
drastically improves performance on various NLP tasks such as question answering
and summarization. We also propose novel applications to genomics data.

---

## FNet: Mixing Tokens with Fourier Transforms

James Lee-Thorp, Joshua Ainslie, Ilya Eckstein, Santiago Ontañón

Category: transformer-other
Keywords: Transformers, Fourier Transform, FNet, NLP, text classification
Year: 2021

We show that Transformer encoder architectures can be massively sped up, with
limited accuracy costs, by replacing the self-attention sublayers with simple
linear transformations that 'mix' input tokens. These linear mixers, along with
standard nonlinearities in feed-forward layers, prove competent at modeling
semantic relationships in several text classification tasks. Most surprisingly,
we find that replacing the self-attention sublayer in a Transformer encoder with
a standard, unparameterized Fourier Transform achieves 92-97% of the accuracy of
BERT counterparts on the GLUE benchmark, but trains nearly seven times faster on
GPUs and twice as fast on TPUs. The resulting model, FNet, also scales very
efficiently to long inputs. Specifically, compared to the 'efficient'
Transformers on the Long Range Arena benchmark, FNet matches the accuracy of the
most accurate models, but is faster than the fastest models across all sequence
lengths on GPUs (and across relatively shorter lengths on TPUs). Finally, FNet
has a light memory footprint and is particularly efficient at smaller model
sizes: for a fixed speed and accuracy budget, small FNet models outperform
Transformer counterparts.

---

## Multimodal Few-Shot Learning with Frozen Language Models

Maria Tsimpoukelli, Jacob Menick, Serkan Cabi, S. M. Ali Eslami, Oriol Vinyals, Felix Hill

Category: transformer-other
Keywords: multimodal learning, few-shot learning, language models, vision and language, transformers
Year: 2021

When trained at sufficient scale, auto-regressive language models exhibit the
notable ability to learn a new language task after being prompted with just a
few examples. Here, we present a simple, yet effective, approach for
transferring this few-shot learning ability to a multimodal setting (vision and
language). Using aligned image and caption data, we train a vision encoder to
represent each image as a sequence of continuous embeddings, such that a pre-
trained, frozen language model prompted with this prefix generates the
appropriate caption. The resulting system is a multimodal few-shot learner, with
the surprising ability to learn a variety of new tasks when conditioned on
examples, represented as a sequence of multiple interleaved image and text
embeddings. We demonstrate that it can rapidly learn words for new objects and
novel visual categories, do visual question-answering with only a handful of
examples, and make use of outside knowledge, by measuring a single model on a
variety of established and new benchmarks.

---

## Big Bird: Transformers for Longer Sequences

Manzil Zaheer, Guru Guruganesh, Avinava Dubey, Joshua Ainslie, Chris Alberti, Santiago Ontanon, Philip Pham, Anirudh Ravula, Qifan Wang, Li Yang, Amr Ahmed

Category: transformer-other
Keywords: Transformers, Sparse Attention, NLP, Sequence Length, Big Bird
Year: 2021

Transformers-based models, such as BERT, have been one of the most successful
deep learning models for NLP. Unfortunately, one of their core limitations is
the quadratic dependency (mainly in terms of memory) on the sequence length due
to their full attention mechanism. To remedy this, we propose BIGBIRD, a sparse
attention mechanism that reduces this quadratic dependency to linear. We show
that BIGBIRD is a universal approximator of sequence functions and is Turing
complete, thereby preserving these properties of the quadratic, full attention
model. Along the way, our theoretical analysis reveals some of the benefits of
having O(1) global tokens (such as CLS), that attend to the entire sequence as
part of the sparse attention mechanism. The proposed sparse attention can handle
sequences of length up to 8x of what was previously possible using similar
hardware. As a consequence of the capability to handle longer context, BIGBIRD
drastically improves performance on various NLP tasks such as question answering
and summarization. We also propose novel applications to genomics data.

---

## A Generalization of Transformer Networks to Graphs

Vijay Prakash Dwivedi, Xavier Bresson

Category: transformer-other
Keywords: transformer networks, graph neural networks, attention mechanism, positional encoding, Laplacian eigenvectors
Year: 2021

We propose a generalization of the transformer neural network architecture for
arbitrary graphs. The original transformer was designed for Natural Language
Processing (NLP), which operates on fully connected graphs representing all
connections between the words in a sequence. Such architecture does not leverage
the graph connectivity inductive bias and can perform poorly when the graph
topology is important and has not been encoded into the node features. We
introduce a graph transformer with four new properties compared to the standard
model. First, the attention mechanism is a function of the neighborhood
connectivity for each node in the graph. Second, the positional encoding is
represented by the Laplacian eigenvectors, which naturally generalize the
sinusoidal positional encodings often used in NLP. Third, the layer
normalization is replaced by a batch normalization layer, which provides faster
training and better generalization performance. Finally, the architecture is
extended to edge feature representation, which can be critical to tasks such as
chemistry (bond type) or link prediction (entity relationship in knowledge
graphs). Numerical experiments on a graph benchmark demonstrate the performance
of the proposed graph transformer architecture. This work closes the gap between
the original transformer, designed for the limited case of line graphs, and
graph neural networks, which can work with arbitrary graphs. As our architecture
is simple and generic, we believe it can be used as a black box for future
applications that wish to consider transformer and graphs.

---

## Hopfield Networks is All You Need

Hubert Ramsauer, Bernhard Schäfl, Johannes Lehner, Philipp Seidl, Michael Widrich, Thomas Adler, Lukas Gruber, Markus Holzleitner, Milena Pavlovic, Geir Kjetil Sandve, Victor Greiff, David Kreil, Michael Kopp, Günter Klambauer, Johannes Brandstetter, Sepp Hochreiter

Category: transformer-other
Keywords: Hopfield networks, continuous states, deep learning, transformer models, attention mechanism
Year: 2021

We introduce a modern Hopfield network with continuous states and a
corresponding update rule. The new Hopfield network can store exponentially
(with the dimension of the associative space) many patterns, retrieves the
pattern with one update, and has exponentially small retrieval errors. It has
three types of energy minima (fixed points of the update): (1) global fixed
point averaging over all patterns, (2) metastable states averaging over a subset
of patterns, and (3) fixed points which store a single pattern. The new update
rule is equivalent to the attention mechanism used in transformers. This
equivalence enables a characterization of the heads of transformer models. These
heads perform in the first layers preferably global averaging and in higher
layers partial averaging via metastable states. The new modern Hopfield network
can be integrated into deep learning architectures as layers to allow the
storage of and access to raw input data, intermediate results, or learned
prototypes. These Hopfield layers enable new ways of deep learning, beyond
fully-connected, convolutional, or recurrent networks, and provide pooling,
memory, association, and attention mechanisms. We demonstrate the broad
applicability of the Hopfield layers across various domains. Hopfield layers
improved state-of-the-art on three out of four considered multiple instance
learning problems as well as on immune repertoire classification with several
hundreds of thousands of instances. On the UCI benchmark collections of small
classification tasks, where deep learning methods typically struggle, Hopfield
layers yielded a new state-of-the-art when compared to different machine
learning methods. Finally, Hopfield layers achieved state-of-the-art on two drug
design datasets. The implementation is available at: https://github.com/ml-
jku/hopfield-layers

---

## The Devil is in the Detail: Simple Tricks Improve Systematic Generalization of Transformers

Róbert Csordás, Kazuki Irie, Jürgen Schmidhuber

Category: transformer-other
Keywords: systematic generalization, transformers, neural networks, relative positional embedding, universal transformer
Year: 2021

Recently, many datasets have been proposed to test the systematic generalization
ability of neural networks. The companion baseline Transformers, typically
trained with default hyper-parameters from standard tasks, are shown to fail
dramatically. Here we demonstrate that by revisiting model configurations as
basic as scaling of embeddings, early stopping, relative positional embedding,
and Universal Transformer variants, we can drastically improve the performance
of Transformers on systematic generalization. We report improvements on five
popular datasets: SCAN, CFQ, PCFG, COGS, and Mathematics dataset. Our models
improve accuracy from 50% to 85% on the PCFG productivity split, and from 35% to
81% on COGS. On SCAN, relative positional embedding largely mitigates the EOS
decision problem, yielding 100% accuracy on the length split with a cutoff at
26. Importantly, performance differences between these models are typically
invisible on the IID data split. This calls for proper generalization validation
sets for developing neural networks that generalize systematically. We publicly
release the code to reproduce our results.

---

## Compressive Transformers for Long-Range Sequence Modelling

Jack W. Rae, Anna Potapenko, Siddhant M. Jayakumar, Chloe Hillier, Timothy P. Lillicrap

Category: transformer-other
Keywords: Compressive Transformer, sequence model, long-range sequence learning, language modelling, memory mechanism, reinforcement learning
Year: 2020

We present the Compressive Transformer, an attentive sequence model which
compresses past memories for long-range sequence learning. We find the
Compressive Transformer obtains state-of-the-art language modelling results in
the WikiText-103 and Enwik8 benchmarks, achieving 17.1 ppl and 0.97 bpc
respectively. We also find it can model high-frequency speech effectively and
can be used as a memory mechanism for RL, demonstrated on an object matching
task. To promote the domain of long-range sequence learning, we propose a new
open-vocabulary language modelling benchmark derived from books, PG-19.

---

## Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer

Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, Peter J. Liu

Category: transformer-other
Keywords: transfer learning, natural language processing, multi-task learning, attention-based models, deep learning
Year: 2020

Transfer learning, where a model is first pre-trained on a data-rich task before
being fine-tuned on a downstream task, has emerged as a powerful technique in
natural language processing (NLP). The effectiveness of transfer learning has
given rise to a diversity of approaches, methodology, and practice. In this
paper, we explore the landscape of transfer learning techniques for NLP by
introducing a unified framework that converts all text-based language problems
into a text-to-text format. Our systematic study compares pre-training
objectives, architectures, unlabeled data sets, transfer approaches, and other
factors on dozens of language understanding tasks. By combining the insights
from our exploration with scale and our new 'Colossal Clean Crawled Corpus', we
achieve state-of-the-art results on many benchmarks covering summarization,
question answering, text classification, and more. To facilitate future work on
transfer learning for NLP, we release our data set, pre-trained models, and
code.

---

## Efficient Transformers: A Survey

Yi Tay, Mostafa Dehghani, Dara Bahri, Donald Metzler

Category: transformer-other
Keywords: Transformers, Self-attention, Efficient Transformers, NLP, Complexity Reduction
Year: 2020

Transformers have become the defacto standard for many NLP tasks. However, the
quadratic complexity of self-attention in Transformers makes it challenging to
scale to long sequences. In this survey, we provide a comprehensive overview of
the recent advances in efficient Transformers that aim to reduce the
computational complexity. We categorize the approaches into four groups:
structured, low-rank, kernel-based, and sparsity-based methods. We also discuss
the trade-offs and future directions in the design of efficient Transformers.

---

## Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer

Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, Peter J. Liu

Category: transformer-other
Keywords: transfer learning, natural language processing, multi-task learning, attention-based models, deep learning
Year: 2020

Transfer learning, where a model is first pre-trained on a data-rich task before
being fine-tuned on a downstream task, has emerged as a powerful technique in
natural language processing (NLP). The effectiveness of transfer learning has
given rise to a diversity of approaches, methodology, and practice. In this
paper, we explore the landscape of transfer learning techniques for NLP by
introducing a unified framework that converts all text-based language problems
into a text-to-text format. Our systematic study compares pre-training
objectives, architectures, unlabeled data sets, transfer approaches, and other
factors on dozens of language understanding tasks. By combining the insights
from our exploration with scale and our new 'Colossal Clean Crawled Corpus', we
achieve state-of-the-art results on many benchmarks covering summarization,
question answering, text classification, and more. To facilitate future work on
transfer learning for NLP, we release our data set, pre-trained models, and
code.

---

## Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer

Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, Peter J. Liu

Category: transformer-other
Keywords: transfer learning, natural language processing, multi-task learning, attention-based models, deep learning
Year: 2020

Transfer learning, where a model is first pre-trained on a data-rich task before
being fine-tuned on a downstream task, has emerged as a powerful technique in
natural language processing (NLP). The effectiveness of transfer learning has
given rise to a diversity of approaches, methodology, and practice. In this
paper, we explore the landscape of transfer learning techniques for NLP by
introducing a unified framework that converts all text-based language problems
into a text-to-text format. Our systematic study compares pre-training
objectives, architectures, unlabeled data sets, transfer approaches, and other
factors on dozens of language understanding tasks. By combining the insights
from our exploration with scale and our new 'Colossal Clean Crawled Corpus', we
achieve state-of-the-art results on many benchmarks covering summarization,
question answering, text classification, and more. To facilitate future work on
transfer learning for NLP, we release our data set, pre-trained models, and
code.

---

## SPECTER: Document-level Representation Learning using Citation-informed Transformers

Arman Cohan, Sergey Feldman, Iz Beltagy, Doug Downey, Daniel S. Weld

Category: transformer-other
Keywords: representation learning, transformers, scientific documents, citation graph, document-level embedding
Year: 2020

Representation learning is a critical ingredient for natural language processing
systems. Recent Transformer language models like BERT learn powerful textual
representations, but these models are targeted towards token- and sentence-level
training objectives and do not leverage information on inter-document
relatedness, which limits their document-level representation power. For
applications on scientific documents, such as classification and recommendation,
the embeddings power strong performance on end tasks. We propose SPECTER, a new
method to generate document-level embedding of scientific documents based on
pretraining a Transformer language model on a powerful signal of document-level
relatedness: the citation graph. Unlike existing pretrained language models,
SPECTER can be easily applied to downstream applications without task-specific
fine-tuning. Additionally, to encourage further research on document-level
models, we introduce SCIDOCS, a new evaluation benchmark consisting of seven
document-level tasks ranging from citation prediction, to document
classification and recommendation. We show that SPECTER outperforms a variety of
competitive baselines on the benchmark.

---

## Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer

Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, Peter J. Liu

Category: transformer-other
Keywords: transfer learning, natural language processing, multi-task learning, attention-based models, deep learning
Year: 2020

Transfer learning, where a model is first pre-trained on a data-rich task before
being fine-tuned on a downstream task, has emerged as a powerful technique in
natural language processing (NLP). The effectiveness of transfer learning has
given rise to a diversity of approaches, methodology, and practice. In this
paper, we explore the landscape of transfer learning techniques for NLP by
introducing a unified framework that converts all text-based language problems
into a text-to-text format. Our systematic study compares pre-training
objectives, architectures, unlabeled data sets, transfer approaches, and other
factors on dozens of language understanding tasks. By combining the insights
from our exploration with scale and our new "Colossal Clean Crawled Corpus", we
achieve state-of-the-art results on many benchmarks covering summarization,
question answering, text classification, and more. To facilitate future work on
transfer learning for NLP, we release our data set, pre-trained models, and
code.

---

## Reformer: The Efficient Transformer

Nikita Kitaev, Łukasz Kaiser, Anselm Levskaya

Category: transformer-other
Keywords: Transformer, efficiency, locality-sensitive hashing, reversible residual layers, memory-efficiency
Year: 2020

Large Transformer models routinely achieve state-of-the-art results on a number
of tasks but training these models can be prohibitively costly, especially on
long sequences. We introduce two techniques to improve the efficiency of
Transformers. For one, we replace dot-product attention by one that uses
locality-sensitive hashing, changing its complexity from O(L^2) to O(L log L),
where L is the length of the sequence. Furthermore, we use reversible residual
layers instead of the standard residuals, which allows storing activations only
once in the training process instead of N times, where N is the number of
layers. The resulting model, the Reformer, performs on par with Transformer
models while being much more memory-efficient and much faster on long sequences.

---

## Graph Transformer Networks

Seongjun Yun, Minbyul Jeong, Raehyun Kim, Jaewoo Kang, Hyunwoo J. Kim

Category: transformer-other
Keywords: Graph Neural Networks, Graph Transformer Networks, Node Classification, Link Prediction, Meta-paths
Year: 2020

Graph neural networks (GNNs) have been widely used in representation learning on
graphs and achieved state-of-the-art performance in tasks such as node
classification and link prediction. However, most existing GNNs are designed to
learn node representations on the fixed and homogeneous graphs. The limitations
especially become problematic when learning representations on a misspecified
graph or a heterogeneous graph that consists of various types of nodes and
edges. In this paper, we propose Graph Transformer Networks (GTNs) that are
capable of generating new graph structures, which involve identifying useful
connections between unconnected nodes on the original graph, while learning
effective node representation on the new graphs in an end-to-end fashion. Graph
Transformer layer, a core layer of GTNs, learns a soft selection of edge types
and composite relations for generating useful multi-hop connections so-called
meta-paths. Our experiments show that GTNs learn new graph structures, based on
data and tasks without domain knowledge, and yield powerful node representation
via convolution on the new graphs. Without domain-specific graph preprocessing,
GTNs achieved the best performance in all three benchmark node classification
tasks against the state-of-the-art methods that require pre-defined meta-paths
from domain knowledge.

---

## Reformer: The Efficient Transformer

Nikita Kitaev, Łukasz Kaiser, Anselm Levskaya

Category: transformer-other
Keywords: Transformer, efficiency, locality-sensitive hashing, reversible residual layers, memory efficiency
Year: 2020

Large Transformer models routinely achieve state-of-the-art results on a number
of tasks but training these models can be prohibitively costly, especially on
long sequences. We introduce two techniques to improve the efficiency of
Transformers. For one, we replace dot-product attention by one that uses
locality-sensitive hashing, changing its complexity from O(L^2) to O(L log L),
where L is the length of the sequence. Furthermore, we use reversible residual
layers instead of the standard residuals, which allows storing activations only
once in the training process instead of N times, where N is the number of
layers. The resulting model, the Reformer, performs on par with Transformer
models while being much more memory-efficient and much faster on long sequences.

---

## Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer

Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, Peter J. Liu

Category: transformer-other
Keywords: transfer learning, natural language processing, multi-task learning, attention-based models, deep learning
Year: 2020

Transfer learning, where a model is first pre-trained on a data-rich task before
being fine-tuned on a downstream task, has emerged as a powerful technique in
natural language processing (NLP). The effectiveness of transfer learning has
given rise to a diversity of approaches, methodology, and practice. In this
paper, we explore the landscape of transfer learning techniques for NLP by
introducing a unified framework that converts all text-based language problems
into a text-to-text format. Our systematic study compares pre-training
objectives, architectures, unlabeled data sets, transfer approaches, and other
factors on dozens of language understanding tasks. By combining the insights
from our exploration with scale and our new “Colossal Clean Crawled Corpus”, we
achieve state-of-the-art results on many benchmarks covering summarization,
question answering, text classification, and more. To facilitate future work on
transfer learning for NLP, we release our data set, pre-trained models, and
code.

---

## Longformer: The Long-Document Transformer

Iz Beltagy, Matthew E. Peters, Arman Cohan

Category: transformer-other
Keywords: transformer, long sequences, self-attention, Longformer, language modeling, sequence-to-sequence
Year: 2020

Transformer-based models are unable to process long sequences due to their self-
attention operation, which scales quadratically with the sequence length. To
address this limitation, we introduce the Longformer with an attention mechanism
that scales linearly with sequence length, making it easy to process documents
of thousands of tokens or longer. Longformer’s attention mechanism is a drop-in
replacement for the standard self-attention and combines a local windowed
attention with a task motivated global attention. Following prior work on long-
sequence transformers, we evaluate Longformer on character-level language
modeling and achieve state-of-the-art results on text8 and enwik8. In contrast
to most prior work, we also pretrain Longformer and finetune it on a variety of
downstream tasks. Our pretrained Longformer consistently outperforms RoBERTa on
long document tasks and sets new state-of-the-art results on WikiHop and
TriviaQA. We finally introduce the Longformer-Encoder-Decoder (LED), a
Longformer variant for supporting long document generative sequence-to-sequence
tasks, and demonstrate its effectiveness on the arXiv summarization dataset.

---

## BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding

Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova

Category: transformer-other
Keywords: BERT, language representation, transformers, pre-training, fine-tuning, NLP
Year: 2019

We introduce a new language representation model called BERT, which stands for
Bidirectional Encoder Representations from Transformers. Unlike recent language
representation models, BERT is designed to pre-train deep bidirectional
representations from unlabeled text by jointly conditioning on both left and
right context in all layers. As a result, the pre-trained BERT model can be
fine-tuned with just one additional output layer to create state-of-the-art
models for a wide range of tasks, such as question answering and language
inference, without substantial task-specific architecture modifications. BERT is
conceptually simple and empirically powerful. It obtains new state-of-the-art
results on eleven natural language processing tasks, including pushing the GLUE
score to 80.5%, MultiNLI accuracy to 86.7%, SQuAD v1.1 question answering Test
F1 to 93.2, and SQuAD v2.0 Test F1 to 83.1.

---

## Fast Transformer Decoding: One Write-Head is All You Need

Noam Shazeer

Category: transformer-other
Keywords: Transformer, multi-head attention, multi-query attention, incremental inference, neural sequence model
Year: 2019

Multi-head attention layers, as used in the Transformer neural sequence model,
are a powerful alternative to RNNs for moving information across and between
sequences. While training these layers is generally fast and simple, due to
parallelizability across the length of the sequence, incremental inference
(where such parallelization is impossible) is often slow, due to the memory-
bandwidth cost of repeatedly loading the large "keys" and "values" tensors. We
propose a variant called multi-query attention, where the keys and values are
shared across all of the different attention "heads", greatly reducing the size
of these tensors and hence the memory bandwidth requirements of incremental
decoding. We verify experimentally that the resulting models can indeed be much
faster to decode, and incur only minor quality degradation from the baseline.

---

## Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context

Zihang Dai, Zhilin Yang, Yiming Yang, Jaime Carbonell, Quoc V. Le, Ruslan Salakhutdinov

Category: transformer-other
Keywords: Transformer-XL, language modeling, long-term dependency, positional encoding, recurrence mechanism
Year: 2019

Transformers have a potential of learning longer-term dependency, but are
limited by a fixed-length context in the setting of language modeling. We
propose a novel neural architecture Transformer-XL that enables learning
dependency beyond a fixed length without disrupting temporal coherence. It
consists of a segment-level recurrence mechanism and a novel positional encoding
scheme. Our method not only enables capturing longer-term dependency, but also
resolves the context fragmentation problem. As a result, Transformer-XL learns
dependency that is 80% longer than RNNs and 450% longer than vanilla
Transformers, achieves better performance on both short and long sequences, and
is up to 1,800+ times faster than vanilla Transformers during evaluation.
Notably, we improve the state-of-the-art results of bpc/perplexity to 0.99 on
enwiki8, 1.08 on text8, 18.3 on WikiText-103, 21.8 on One Billion Word, and 54.5
on Penn Treebank (without finetuning). When trained only on WikiText-103,
Transformer-XL manages to generate reasonably coherent, novel text articles with
thousands of tokens. Our code, pretrained models, and hyperparameters are
available in both Tensorflow and PyTorch.

---

## Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context

Zihang Dai, Zhilin Yang, Yiming Yang, Jaime Carbonell, Quoc V. Le, Ruslan Salakhutdinov

Category: transformer-other
Keywords: Transformer-XL, language modeling, long-term dependency, neural architecture, positional encoding
Year: 2019

Transformers have a potential of learning longer-term dependency, but are
limited by a fixed-length context in the setting of language modeling. We
propose a novel neural architecture Transformer-XL that enables learning
dependency beyond a fixed length without disrupting temporal coherence. It
consists of a segment-level recurrence mechanism and a novel positional encoding
scheme. Our method not only enables capturing longer-term dependency, but also
resolves the context fragmentation problem. As a result, Transformer-XL learns
dependency that is 80% longer than RNNs and 450% longer than vanilla
Transformers, achieves better performance on both short and long sequences, and
is up to 1,800+ times faster than vanilla Transformers during evaluation.
Notably, we improve the state-of-the-art results of bpc/perplexity to 0.99 on
enwiki8, 1.08 on text8, 18.3 on WikiText-103, 21.8 on One Billion Word, and 54.5
on Penn Treebank (without finetuning). When trained only on WikiText-103,
Transformer-XL manages to generate reasonably coherent, novel text articles with
thousands of tokens. Our code, pretrained models, and hyperparameters are
available in both Tensorflow and PyTorch.

---

## BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding

Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova

Category: transformer-other
Keywords: BERT, language representation, transformers, natural language processing, pre-training
Year: 2019

We introduce a new language representation model called BERT, which stands for
Bidirectional Encoder Representations from Transformers. Unlike recent language
representation models, BERT is designed to pre-train deep bidirectional
representations from unlabeled text by jointly conditioning on both left and
right context in all layers. As a result, the pre-trained BERT model can be
fine-tuned with just one additional output layer to create state-of-the-art
models for a wide range of tasks, such as question answering and language
inference, without substantial task-specific architecture modifications. BERT is
conceptually simple and empirically powerful. It obtains new state-of-the-art
results on eleven natural language processing tasks, including pushing the GLUE
score to 80.5% (7.7% point absolute improvement), MultiNLI accuracy to 86.7%
(4.6% absolute improvement), SQuAD v1.1 question answering Test F1 to 93.2 (1.5
point absolute improvement), and SQuAD v2.0 Test F1 to 83.1 (5.1 point absolute
improvement).

---

## Are Sixteen Heads Really Better than One?

Paul Michel, Omer Levy, Graham Neubig

Category: transformer-other
Keywords: attention, multi-headed attention, natural language processing, transformer, model pruning
Year: 2019

Attention is a powerful and ubiquitous mechanism for allowing neural models to
focus on particular salient pieces of information by taking their weighted
average when making predictions. In particular, multi-headed attention is a
driving force behind many recent state-of-the-art natural language processing
(NLP) models such as Transformer-based MT models and BERT. These models apply
multiple attention mechanisms in parallel, with each attention “head”
potentially focusing on different parts of the input, which makes it possible to
express sophisticated functions beyond the simple weighted average. In this
paper we make the surprising observation that even if models have been trained
using multiple heads, in practice, a large percentage of attention heads can be
removed at test time without significantly impacting performance. In fact, some
layers can even be reduced to a single head. We further examine greedy
algorithms for pruning down models, and the potential speed, memory efficiency,
and accuracy improvements obtainable therefrom. Finally, we analyze the results
with respect to which parts of the model are more reliant on having multiple
heads, and provide precursory evidence that training dynamics play a role in the
gains provided by multi-head attention.

---

## Are Sixteen Heads Really Better than One?

Paul Michel, Omer Levy, Graham Neubig

Category: transformer-other
Keywords: multi-headed attention, transformers, natural language processing, model pruning, attention mechanisms
Year: 2019

Attention is a powerful and ubiquitous mechanism for allowing neural models to
focus on particular salient pieces of information by taking their weighted
average when making predictions. In particular, multi-headed attention is a
driving force behind many recent state-of-the-art natural language processing
(NLP) models such as Transformer-based MT models and BERT. These models apply
multiple attention mechanisms in parallel, with each attention “head”
potentially focusing on different parts of the input, which makes it possible to
express sophisticated functions beyond the simple weighted average. In this
paper we make the surprising observation that even if models have been trained
using multiple heads, in practice, a large percentage of attention heads can be
removed at test time without significantly impacting performance. In fact, some
layers can even be reduced to a single head. We further examine greedy
algorithms for pruning down models, and the potential speed, memory efficiency,
and accuracy improvements obtainable therefrom. Finally, we analyze the results
with respect to which parts of the model are more reliant on having multiple
heads, and provide precursory evidence that training dynamics play a role in the
gains provided by multi-head attention.

---

## BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding

Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova

Category: transformer-other
Keywords: BERT, language representation model, bidirectional transformers, pre-training, natural language processing, fine-tuning
Year: 2019

We introduce a new language representation model called BERT, which stands for
Bidirectional Encoder Representations from Transformers. Unlike recent language
representation models, BERT is designed to pre-train deep bidirectional
representations from unlabeled text by jointly conditioning on both left and
right context in all layers. As a result, the pre-trained BERT model can be
fine-tuned with just one additional output layer to create state-of-the-art
models for a wide range of tasks, such as question answering and language
inference, without substantial task-specific architecture modifications. BERT is
conceptually simple and empirically powerful. It obtains new state-of-the-art
results on eleven natural language processing tasks, including pushing the GLUE
score to 80.5% (7.7% point absolute improvement), MultiNLI accuracy to 86.7%
(4.6% absolute improvement), SQuAD v1.1 question answering Test F1 to 93.2 (1.5
point absolute improvement) and SQuAD v2.0 Test F1 to 83.1 (5.1 point absolute
improvement).

---

## RoBERTa: A Robustly Optimized BERT Pretraining Approach

Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, Veselin Stoyanov

Category: transformer-other
Keywords: RoBERTa, BERT, pretraining, language models, hyperparameters, performance gains, state-of-the-art
Year: 2019

Language model pretraining has led to significant performance gains but careful
comparison between different approaches is challenging. Training is
computationally expensive, often done on private datasets of different sizes,
and, as we will show, hyperparameter choices have significant impact on the
final results. We present a replication study of BERT pretraining (Devlin et
al., 2019) that carefully measures the impact of many key hyperparameters and
training data size. We find that BERT was significantly undertrained, and can
match or exceed the performance of every model published after it. Our best
model achieves state-of-the-art results on GLUE, RACE and SQuAD. These results
highlight the importance of previously overlooked design choices, and raise
questions about the source of recently reported improvements. We release our
models and code.

---

## Are Sixteen Heads Really Better than One?

Paul Michel, Omer Levy, Graham Neubig

Category: transformer-other
Keywords: multi-head attention, transformers, NLP, model pruning, efficiency
Year: 2019

Attention is a powerful and ubiquitous mechanism for allowing neural models to
focus on particular salient pieces of information by taking their weighted
average when making predictions. In particular, multi-headed attention is a
driving force behind many recent state-of-the-art natural language processing
(NLP) models such as Transformer-based MT models and BERT. These models apply
multiple attention mechanisms in parallel, with each attention “head”
potentially focusing on different parts of the input, which makes it possible to
express sophisticated functions beyond the simple weighted average. In this
paper we make the surprising observation that even if models have been trained
using multiple heads, in practice, a large percentage of attention heads can be
removed at test time without significantly impacting performance. In fact, some
layers can even be reduced to a single head. We further examine greedy
algorithms for pruning down models, and the potential speed, memory efficiency,
and accuracy improvements obtainable therefrom. Finally, we analyze the results
with respect to which parts of the model are more reliant on having multiple
heads, and provide precursory evidence that training dynamics play a role in the
gains provided by multi-head attention.

---

## Compressive Transformers for Long-Range Sequence Modelling

Jack W. Rae, Anna Potapenko, Siddhant M. Jayakumar, Chloe Hillier, Timothy P. Lillicrap

Category: transformer-other
Keywords: Compressive Transformer, long-range sequence learning, language modelling, WikiText-103, Enwik8, PG-19
Year: 2019

We present the Compressive Transformer, an attentive sequence model which
compresses past memories for long-range sequence learning. We find the
Compressive Transformer obtains state-of-the-art language modelling results in
the WikiText-103 and Enwik8 benchmarks, achieving 17.1 ppl and 0.97 bpc
respectively. We also find it can model high-frequency speech effectively and
can be used as a memory mechanism for RL, demonstrated on an object matching
task. To promote the domain of long-range sequence learning, we propose a new
open-vocabulary language modelling benchmark derived from books, PG-19.

---

## Fast Transformer Decoding: One Write-Head is All You Need

Noam Shazeer

Category: transformer-other
Keywords: Transformer, multi-head attention, multi-query attention, incremental inference, neural sequence model
Year: 2019

Multi-head attention layers, as used in the Transformer neural sequence model,
are a powerful alternative to RNNs for moving information across and between
sequences. While training these layers is generally fast and simple, due to
parallelizability across the length of the sequence, incremental inference
(where such parallelization is impossible) is often slow, due to the memory-
bandwidth cost of repeatedly loading the large 'keys' and 'values' tensors. We
propose a variant called multi-query attention, where the keys and values are
shared across all of the different attention 'heads', greatly reducing the size
of these tensors and hence the memory bandwidth requirements of incremental
decoding. We verify experimentally that the resulting models can indeed be much
faster to decode, and incur only minor quality degradation from the baseline.

---

## Deep Equilibrium Models

Shaojie Bai, J. Zico Kolter, Vladlen Koltun

Category: transformer-other
Keywords: deep equilibrium models, sequential data modeling, root-finding, implicit differentiation, memory efficiency
Year: 2019

We present a new approach to modeling sequential data: the deep equilibrium
model (DEQ). Motivated by an observation that the hidden layers of many existing
deep sequence models converge towards some fixed point, we propose the DEQ
approach that directly finds these equilibrium points via root-finding. Such a
method is equivalent to running an infinite depth (weight-tied) feedforward
network, but has the notable advantage that we can analytically backpropagate
through the equilibrium point using implicit differentiation. Using this
approach, training and prediction in these networks require only constant
memory, regardless of the effective “depth” of the network. We demonstrate how
DEQs can be applied to two state-of-the-art deep sequence models: self-attention
transformers and trellis networks. On large-scale language modeling tasks, such
as the WikiText-103 benchmark, we show that DEQs 1) often improve performance
over these state-of-the-art models (for similar parameter counts); 2) have
similar computational requirements to existing models; and 3) vastly reduce
memory consumption (often the bottleneck for training large sequence models),
demonstrating an up-to 88% memory reduction in our experiments. The code is
available at https://github.com/locuslab/deq.

---

## Are Sixteen Heads Really Better than One?

Paul Michel, Omer Levy, Graham Neubig

Category: transformer-other
Keywords: multi-headed attention, transformers, natural language processing, model pruning, attention mechanism
Year: 2019

Attention is a powerful and ubiquitous mechanism for allowing neural models to
focus on particular salient pieces of information by taking their weighted
average when making predictions. In particular, multi-headed attention is a
driving force behind many recent state-of-the-art natural language processing
(NLP) models such as Transformer-based MT models and BERT. These models apply
multiple attention mechanisms in parallel, with each attention “head”
potentially focusing on different parts of the input, which makes it possible to
express sophisticated functions beyond the simple weighted average. In this
paper we make the surprising observation that even if models have been trained
using multiple heads, in practice, a large percentage of attention heads can be
removed at test time without significantly impacting performance. In fact, some
layers can even be reduced to a single head. We further examine greedy
algorithms for pruning down models, and the potential speed, memory efficiency,
and accuracy improvements obtainable therefrom. Finally, we analyze the results
with respect to which parts of the model are more reliant on having multiple
heads, and provide precursory evidence that training dynamics play a role in the
gains provided by multi-head attention.

---

## What Does BERT Look At? An Analysis of BERT’s Attention

Kevin Clark, Urvashi Khandelwal, Omer Levy, Christopher D. Manning

Category: transformer-other
Keywords: BERT, attention mechanisms, linguistic features, syntax, coreference, transformer models
Year: 2019

Large pre-trained neural networks such as BERT have had great recent success in
NLP, motivating a growing body of research investigating what aspects of
language they are able to learn from unlabeled data. Most recent analysis has
focused on model outputs (e.g., language model surprisal) or internal vector
representations (e.g., probing classifiers). Complementary to these works, we
propose methods for analyzing the attention mechanisms of pre-trained models and
apply them to BERT. BERT’s attention heads exhibit patterns such as attending to
delimiter tokens, specific positional offsets, or broadly attending over the
whole sentence, with heads in the same layer often exhibiting similar behaviors.
We further show that certain attention heads correspond well to linguistic
notions of syntax and coreference. For example, we find heads that attend to the
direct objects of verbs, determiners of nouns, objects of prepositions, and
coreferent mentions with remarkably high accuracy. Lastly, we propose an
attention-based probing classifier and use it to further demonstrate that
substantial syntactic information is captured in BERT’s attention.

---

## What Does BERT Look At? An Analysis of BERT’s Attention

Kevin Clark, Urvashi Khandelwal, Omer Levy, Christopher D. Manning

Category: transformer-other
Keywords: BERT, attention mechanisms, linguistic features, syntax, coreference
Year: 2019

Large pre-trained neural networks such as BERT have had great recent success in
NLP, motivating a growing body of research investigating what aspects of
language they are able to learn from unlabeled data. Most recent analysis has
focused on model outputs or internal vector representations. Complementary to
these works, we propose methods for analyzing the attention mechanisms of pre-
trained models and apply them to BERT. BERT’s attention heads exhibit patterns
such as attending to delimiter tokens, specific positional offsets, or broadly
attending over the whole sentence, with heads in the same layer often exhibiting
similar behaviors. We further show that certain attention heads correspond well
to linguistic notions of syntax and coreference. For example, we find heads that
attend to the direct objects of verbs, determiners of nouns, objects of
prepositions, and coreferent mentions with remarkably high accuracy. Lastly, we
propose an attention-based probing classifier and use it to further demonstrate
that substantial syntactic information is captured in BERT’s attention.

---

## BERT Rediscovers the Classical NLP Pipeline

Ian Tenney, Dipanjan Das, Ellie Pavlick

Category: transformer-other
Keywords: BERT, NLP, text encoders, linguistic information, interpretability
Year: 2019

Pre-trained text encoders have rapidly advanced the state of the art on many NLP
tasks. We focus on one such model, BERT, and aim to quantify where linguistic
information is captured within the network. We find that the model represents
the steps of the traditional NLP pipeline in an interpretable and localizable
way, and that the regions responsible for each step appear in the expected
sequence: POS tagging, parsing, NER, semantic roles, then coreference.
Qualitative analysis reveals that the model can and often does adjust this
pipeline dynamically, revising lower-level decisions on the basis of
disambiguating information from higher-level representations.

---

## BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding

Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova

Category: transformer-other
Keywords: BERT, language representation, transformers, natural language processing, pre-training
Year: 2019

We introduce a new language representation model called BERT, which stands for
Bidirectional Encoder Representations from Transformers. Unlike recent language
representation models (Peters et al., 2018a; Radford et al., 2018), BERT is
designed to pretrain deep bidirectional representations from unlabeled text by
jointly conditioning on both left and right context in all layers. As a result,
the pre-trained BERT model can be fine-tuned with just one additional output
layer to create state-of-the-art models for a wide range of tasks, such as
question answering and language inference, without substantial task-specific
architecture modifications. BERT is conceptually simple and empirically
powerful. It obtains new state-of-the-art results on eleven natural language
processing tasks, including pushing the GLUE score to 80.5% (7.7% point absolute
improvement), MultiNLI accuracy to 86.7% (4.6% absolute improvement), SQuAD v1.1
question answering Test F1 to 93.2 (1.5 point absolute improvement) and SQuAD
v2.0 Test F1 to 83.1 (5.1 point absolute improvement).

---

## Are Sixteen Heads Really Better than One?

Paul Michel, Omer Levy, Graham Neubig

Category: transformer-other
Keywords: multi-headed attention, Transformers, NLP, attention mechanism, model pruning
Year: 2019

Attention is a powerful and ubiquitous mechanism for allowing neural models to
focus on particular salient pieces of information by taking their weighted
average when making predictions. In particular, multi-headed attention is a
driving force behind many recent state-of-the-art natural language processing
(NLP) models such as Transformer-based MT models and BERT. These models apply
multiple attention mechanisms in parallel, with each attention “head”
potentially focusing on different parts of the input, which makes it possible to
express sophisticated functions beyond the simple weighted average. In this
paper we make the surprising observation that even if models have been trained
using multiple heads, in practice, a large percentage of attention heads can be
removed at test time without significantly impacting performance. In fact, some
layers can even be reduced to a single head. We further examine greedy
algorithms for pruning down models, and the potential speed, memory efficiency,
and accuracy improvements obtainable therefrom. Finally, we analyze the results
with respect to which parts of the model are more reliant on having multiple
heads, and provide precursory evidence that training dynamics play a role in the
gains provided by multi-head attention.

---

## Attention Is All You Need

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin

Category: transformer-other
Keywords: Transformer, attention mechanism, machine translation, sequence transduction, deep learning
Year: 2017

The dominant sequence transduction models are based on complex recurrent or
convolutional neural networks that include an encoder and a decoder. The best
performing models also connect the encoder and decoder through an attention
mechanism. We propose a new simple network architecture, the Transformer, based
solely on attention mechanisms, dispensing with recurrence and convolutions
entirely. Experiments on two machine translation tasks show these models to be
superior in quality while being more parallelizable and requiring significantly
less time to train. Our model achieves 28.4 BLEU on the WMT 2014 English-to-
German translation task, improving over the existing best results, including
ensembles, by over 2 BLEU. On the WMT 2014 English-to-French translation task,
our model establishes a new single-model state-of-the-art BLEU score of 41.8
after training for 3.5 days on eight GPUs, a small fraction of the training
costs of the best models from the literature. We show that the Transformer
generalizes well to other tasks by applying it successfully to English
constituency parsing both with large and limited training data.

---

## Attention Is All You Need

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin

Category: transformer-other
Keywords: Transformer, attention mechanism, sequence transduction, machine translation, model architecture
Year: 2017

The dominant sequence transduction models are based on complex recurrent or
convolutional neural networks that include an encoder and a decoder. The best
performing models also connect the encoder and decoder through an attention
mechanism. We propose a new simple network architecture, the Transformer, based
solely on attention mechanisms, dispensing with recurrence and convolutions
entirely. Experiments on two machine translation tasks show these models to be
superior in quality while being more parallelizable and requiring significantly
less time to train. Our model achieves 28.4 BLEU on the WMT 2014 English-to-
German translation task, improving over the existing best results, including
ensembles, by over 2 BLEU. On the WMT 2014 English-to-French translation task,
our model establishes a new single-model state-of-the-art BLEU score of 41.8
after training for 3.5 days on eight GPUs, a small fraction of the training
costs of the best models from the literature. We show that the Transformer
generalizes well to other tasks by applying it successfully to English
constituency parsing both with large and limited training data.