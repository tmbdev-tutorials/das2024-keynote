# CLIP

- Contrastive Language–Image Pretraining (2021; Radford et al. at OpenAI)
- based on transformer architecture; connects images and text by embedding them in a shared latent space
- performs zero-shot learning across various vision and language tasks
- image classification, object recognition, retrieval, captioning
- key advancement: aligned multimodal embeddings, allowing for open-vocabulary visual understanding

# ALIGN

- Large-scale Image-Language Pretraining (2021; Jia et al. at Google Research)
- transformer-based architecture; focuses on noisy web data for pretraining
- aligns visual and textual embeddings for downstream tasks like image-text retrieval
- image-text retrieval, zero-shot learning, cross-modal transfer
- novel approach: leveraging large-scale noisy datasets without the need for manual annotation

# ViLT

- Vision-and-Language Transformer (2021; Kim et al. at NAVER AI Lab)
- transformer model that handles both image and text data without using convolutional neural networks (CNNs)
- processes multimodal inputs end-to-end, integrating image patches directly into the transformer encoder
- visual question answering (VQA), image-text retrieval, visual reasoning
- contribution: efficient transformer architecture by eliminating the need for pre-trained CNN backbones

# VisualBERT

- Visual BERT (2019; Li et al. at Facebook AI)
- transformer model extending BERT for multimodal tasks by integrating visual embeddings
- fuses vision and language tasks by embedding visual features into BERT's architecture
- visual question answering (VQA), visual commonsense reasoning, visual entailment
- significant feature: pre-trained on both image-text pairs and language data, enabling robust cross-modal understanding

# UNITER

- UNiversal Image-TExt Representation (2019; Chen et al. at Microsoft Research)
- transformer-based model learning joint image-text representations using a pretraining scheme
- introduces four types of pretraining tasks: masked language modeling, masked region modeling, image-text matching, and word-region alignment
- visual question answering (VQA), image-text retrieval, visual reasoning
- innovation: effectively unifying visual and textual information via joint cross-modal embedding strategies

# LXMERT

- Learning Cross-Modality Encoder Representations from Transformers (2019; Tan & Bansal at UNC Chapel Hill)
- multimodal transformer architecture for vision and language tasks; processes images and text separately before merging
- leverages visual and textual encoders, followed by cross-attention layers for task integration
- visual question answering (VQA), image captioning, image retrieval
- key contribution: cross-modal attention layers for effective fusion of visual and textual representations

# VLMo

- Vision-Language MoE (2022; Wang et al. at Microsoft Research)
- transformer-based model using mixture-of-experts (MoE) for vision-language tasks
- experts are dynamically selected based on input modality, optimizing computational efficiency
- visual question answering (VQA), image-text retrieval, visual reasoning
- novelty: introduces a unified vision-language model with dynamic expert selection for diverse tasks

# FLAVA

- A Foundational Language and Vision Alignment Model (2022; Singh et al. at Meta AI)
- transformer architecture jointly pre-trained on images, text, and image-text pairs
- supports unimodal, cross-modal, and multimodal tasks, using a unified architecture
- image-text classification, image-text retrieval, visual question answering (VQA)
- distinctive feature: achieves competitive performance across both unimodal and multimodal benchmarks

# BLIP

- Bootstrapping Language-Image Pretraining (2022; Li et al. at Salesforce Research)
- multimodal model designed for image-language pretraining using vision-language bootstrapping
- supports various tasks by pretraining on noisy web data and fine-tuning on downstream tasks
- image-text retrieval, image captioning, visual question answering (VQA)
- innovation: combines bootstrapping techniques to improve performance on noisy datasets

# OF-ViT

- Omni-Format Vision Transformer (2022; Zhang et al. at Microsoft Research)
- transformer model designed to process visual data in multiple formats (e.g., images, videos)
- trained to handle various visual tasks by leveraging different input formats
- video recognition, image classification, visual question answering (VQA)
- contribution: versatile architecture that adapts to different input formats and tasks efficiently

# Kosmos-1

- Language-Centric Multimodal Foundation Model (2023; Huang et al. at Microsoft Research)
- transformer model that integrates language and multimodal inputs, focusing on tasks with strong language components
- performs well on text-to-image generation, visual reasoning, and other multimodal tasks
- visual question answering (VQA), multimodal reasoning, natural language generation
- key feature: emphasizes language-driven multimodal learning for a variety of tasks

# Kosmos-2

- Language-Centric Multimodal Foundation Model, Version 2 (2023; Huang et al. at Microsoft Research)
- improved version of Kosmos-1 with enhanced multimodal understanding and generative capabilities
- handles more complex tasks across modalities, with improved fine-tuning capabilities
- text-to-image generation, visual question answering (VQA), multimodal reasoning
- novelty: enhanced performance on complex multimodal reasoning tasks with better integration of vision and language

# Florence

- A Unified Image-Language Foundation Model (2022; Yuan et al. at Microsoft Research)
- large-scale pretrained transformer model for visual recognition and multimodal tasks
- pretrained on diverse datasets, supporting a wide range of vision and vision-language tasks
- image classification, image-text retrieval, visual reasoning, visual question answering (VQA)
- innovation: achieves state-of-the-art performance across a broad spectrum of visual and multimodal benchmarks

# Perceiver IO

- Generalized Perception with Iterative Attention (2021; Jaegle et al. at DeepMind)
- transformer-based architecture designed to handle a wide variety of input modalities, from images to point clouds
- processes any type of data input through a flexible and scalable architecture using iterative attention
- visual recognition, natural language processing, multimodal fusion tasks
- contribution: generalized transformer that scales efficiently across diverse data types without modality-specific adaptations

# CoCa

- Contrastive Captioners (2022; Yu et al. at Google Research)
- multimodal transformer model combining contrastive learning and captioning objectives
- jointly trains on image-text contrastive learning and autoregressive captioning for versatile performance
- image captioning, image-text retrieval, visual question answering (VQA)
- distinctive feature: unifies contrastive learning with generative tasks for enhanced multimodal understanding

# Unified-IO

- Unified Input-Output (2022; Lu et al. at Allen Institute for AI)
- transformer model capable of handling both vision and language tasks in a unified framework
- unifies the architecture to process different input-output formats without separate heads or task-specific modules
- visual recognition, language understanding, multimodal reasoning
- key contribution: single architecture for diverse input-output tasks, removing the need for task-specific design

# Segment Anything Model (SAM)

- A Foundation Model for Image Segmentation (2023; Kirillov et al. at Meta AI)
- transformer-based architecture designed for promptable segmentation tasks, enabling zero-shot generalization across different domains
- accepts various prompts such as points, boxes, or text to produce image masks
- image segmentation, object recognition, instance segmentation
- key innovation: enables flexible, task-agnostic segmentation with minimal user input across diverse image types

# Mask R-CNN

- Mask Region-Based Convolutional Neural Network (2017; He et al. at Facebook AI)
- extension of Faster R-CNN; adds a branch for predicting segmentation masks on top of object detection
- simultaneously detects objects and generates high-quality segmentation masks
- instance segmentation, object detection, keypoint detection
- contribution: first to achieve both object detection and instance segmentation in a unified framework

# DeepLab

- Deep Convolutional Networks for Semantic Image Segmentation (2017; Chen et al. at Google Research)
- convolutional neural network architecture using atrous convolutions and fully connected CRFs for semantic segmentation
- produces dense predictions for semantic segmentation tasks at multiple scales
- semantic segmentation, object recognition
- novelty: introduced atrous convolution and dense CRF for better segmentation in fine details and boundary refinement

# U-Net

- Convolutional Networks for Biomedical Image Segmentation (2015; Ronneberger et al. at University of Freiburg)
- U-shaped CNN architecture for precise segmentation of biomedical images, using skip connections for feature fusion
- designed to work with small datasets by augmenting with rotated and deformed images
- biomedical image segmentation, medical imaging
- key feature: symmetric encoder-decoder architecture with skip connections for high-resolution output segmentation

# SETR

- Segmentation Transformer (2021; Zheng et al. at Zhejiang University)
- transformer-based model applied to image segmentation, using transformer encoders instead of convolutional backbones
- processes image patches via transformers for pixel-level classification
- semantic segmentation, scene parsing
- innovation: leverages transformer encoders for pixel-wise segmentation, departing from traditional CNN-based approaches

# Detectron2

- Next-Generation Library for Object Detection and Segmentation (2019; Wu et al. at Facebook AI)
- modular framework built on PyTorch for object detection, instance segmentation, and keypoint detection
- incorporates state-of-the-art models like Mask R-CNN, RetinaNet, and Faster R-CNN
- object detection, instance segmentation, keypoint detection
- contribution: provides a high-performance, flexible, and scalable library for state-of-the-art visual recognition models

# YOLACT

- You Only Look at Coefficients (2019; Bolya et al. at Washington University in St. Louis)
- real-time instance segmentation model, balancing speed and accuracy by decoupling instance masks from object detection
- generates prototype masks and coefficients for each object, enabling real-time performance
- instance segmentation, object detection
- key innovation: achieves high-speed instance segmentation by separating mask generation from object detection

# Swin Transformer

- Hierarchical Vision Transformer using Shifted Windows (2021; Liu et al. at Microsoft Research)
- transformer-based model with a hierarchical architecture, enabling computation on non-overlapping windows shifted across layers
- performs effectively on dense prediction tasks such as object detection and segmentation
- object detection, semantic segmentation, image classification
- contribution: combines transformer efficiency with multi-scale feature extraction, excelling in both vision and dense prediction tasks

# Segmenter

- Transformer-Based Semantic Segmentation Model (2021; Strudel et al. at INRIA)
- end-to-end transformer model for semantic segmentation, using masked patch embeddings for prediction
- processes image patches via transformers to produce segmentation masks at a pixel level
- semantic segmentation, object recognition
- key contribution: leverages transformer architectures for dense prediction tasks, demonstrating superior performance without convolutions

# ViTGAN

- Vision Transformer Generative Adversarial Network (2021; Lee et al. at KIST AI)
- adapts Vision Transformers (ViT) for GAN architectures, generating high-quality images
- uses a transformer-based discriminator and generator for image synthesis
- image generation, style transfer
- novelty: integrates transformer architecture in GAN frameworks, achieving competitive generative performance without convolutions

# NVIDIA MaxViT

- Multi-Axis Vision Transformer (2022; Tu et al. at NVIDIA)
- hybrid model combining transformers and CNNs, with multi-axis attention for capturing both local and global dependencies
- excels in dense prediction tasks by balancing performance and efficiency
- image classification, object detection, semantic segmentation
- contribution: novel multi-axis attention mechanism, offering efficient processing and strong generalization on various vision tasks

# StyleGAN3

- Generative Adversarial Network for Image Synthesis (2021; Karras et al. at NVIDIA)
- focuses on alias-free image synthesis, improving temporal coherence and fine details
- employs continuous signals to generate high-quality, high-resolution images with spatial consistency
- image generation, style transfer, facial synthesis
- key feature: eliminates aliasing artifacts in GAN-generated images, enabling smoother transitions and more realistic outputs

# TransGAN

- Transformer-Based Generative Adversarial Network (2021; Jiang et al. at Huawei Noah’s Ark Lab)
- pure transformer model used for generative tasks, without relying on convolutional layers
- transforms image patches into high-quality outputs via transformer blocks in both generator and discriminator
- image generation, style transfer
- innovation: first to propose a fully transformer-based architecture for GANs, achieving competitive generative results

# SWIN-UNet

- Swin Transformer U-Net for Medical Image Segmentation (2021; Cao et al. at Tencent Youtu Lab)
- hybrid model combining Swin Transformer architecture with U-Net for high-precision medical image segmentation
- integrates hierarchical Swin Transformer blocks into U-Net’s encoder-decoder structure
- medical image segmentation, biomedical applications
- contribution: achieves state-of-the-art performance on medical image tasks by blending transformer efficiency with U-Net's resolution-preserving architecture

# TAdaConv

- Temporally Adaptive Convolutions (2022; Meng et al. at NVIDIA)
- dynamic convolutional layer that adapts over time for video understanding tasks, adjusting kernel weights temporally
- used for video classification and action recognition by capturing temporal dependencies efficiently
- video classification, action recognition
- key feature: introduces temporally adaptive convolutions, enhancing performance on dynamic visual tasks

# DINO

- Self-Supervised Vision Transformer (2021; Caron et al. at Facebook AI Research)
- self-supervised learning framework using Vision Transformers (ViT) for feature representation without labeled data
- leverages a student-teacher architecture to learn from unlabelled images, achieving robust visual representations
- image classification, object recognition, feature learning
- novelty: demonstrates the power of self-supervised learning with transformers, achieving state-of-the-art performance in unsupervised tasks


