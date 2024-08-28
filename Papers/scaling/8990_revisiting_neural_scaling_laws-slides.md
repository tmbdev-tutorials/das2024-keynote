# Revisiting Neural Scaling Laws in Language and Vision

- Ibrahim Alabdulmohsin, Behnam Neyshabur, Xiaohua Zhai
- Google Research, Brain Team and Blueshift Team
- Zürich, Switzerland; Mountain View, United States
- NeurIPS 2022

# Introduction

- Scale drives progress in deep learning
- Empirical prediction of benefits from scaling
- Propose rigorous methodology using extrapolation loss
- Present a recipe for estimating scaling law parameters
- Benchmark dataset with 90 evaluation tasks

# Background/Related Work

- Scale improvements in vision and NLP domains
- Studies on scaling data size, model size, and training schedule
- Performance often follows a power law \( f(x) \sim \beta x^c \)
- Previous works report best-fitting parameters, leading to misleading results

# Contributions

- Rigorous validation of scaling law parameters using extrapolation
- Recipe for reliable estimation of scaling laws from learning curves
- Study impact of architecture type and size on scaling exponents
- Release a benchmark dataset with 90 tasks

# Objective

- Validate scaling law parameters based on extrapolation, not interpolation
- Propose a new estimator M4 for better extrapolation
- Verify estimator across multiple domains: image classification, NMT, language modeling

# Methodology Overview

- Use learning curves to estimate scaling law parameters
- Evaluate the performance using extrapolation error
- Experiment with multiple architecture families and domains

# Datasets

- Image classification using JFT-300M
- Neural machine translation datasets
- Language modeling with LaMDA architecture
- BIG-Bench evaluation benchmark with 90 tasks

# Model Details

## Model Architecture

- Big-transfer residual networks (BiT)
- Vision transformers (ViT)
- MLP mixers (MiX)

## Training Procedure

- Pretrain on JFT-300M
- Evaluate on few-shot accuracy downstream tasks
- Measure performance using RMSE for extrapolation

# Experiments

## Image Classification

- Pretrain on JFT-300M, evaluate on ImageNet, Birds 200, CIFAR100, Caltech101
- Use architectures: BiT, ViT, MiX
- Report 5/10/25-shot accuracy

## Neural Machine Translation (NMT)

- Encoder-decoder transformer models
- Measure log-perplexity on hold-out dataset
- Fit scaling law parameters and predict performance on larger datasets

## Language Modeling

- LaMDA architecture with various model sizes
- Rescale validation loss to unit interval
- Evaluate scaling law estimators using RMSE

# Results

## Performance Metrics

- M4 estimator outperforms previous methods in extrapolation
- Scaling exponents vary with architecture type and size
- Larger models show more favorable scaling behavior

## Performance Comparisons with Prior Work

- M4 provides better extrapolation than M1, M2, M3
- Consistent results across image classification, NMT, language modeling

## Ablation Studies

- Evaluate impact of different components in M4
- Show M4 works well even when data deviates from power law behavior

# Discussion

- Importance of accurate extrapolation for scaling laws
- Potential applications in sample size planning and neural architecture search
- Future work to apply the new recipe to other domains and applications

# Conclusion

- Rigorous validation of scaling law parameters using extrapolation
- New estimator M4 for better extrapolation accuracy
- Release of benchmark dataset to facilitate further research
- Future directions include applying the estimator to neural architecture search

# References

- Key citations include works by Bahri et al. (2021), Bansal et al. (2022), Kaplan et al. (2020), Hestness et al. (2017)

# Acknowledgements

- Thanks to Behrooz Ghorbani, Ambrose Slone, Lucas Beyer, Daniel Keysers, Olivier Bousquet for feedback and discussions

# Q&A

- Invitation for questions
