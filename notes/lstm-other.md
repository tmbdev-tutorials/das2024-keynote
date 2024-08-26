
## On orthogonality and learning recurrent networks with long term dependencies

Eugene Vorontsov, Chiheb Trabelsi, Samuel Kadoury, Chris Pal

Category: lstm-other
Keywords: orthogonality, recurrent neural networks, vanishing gradient, exploding gradient, long term dependencies
Year: 2017

It is well known that it is challenging to train deep neural networks and
recurrent neural networks for tasks that exhibit long term dependencies. The
vanishing or exploding gradient problem is a well known issue associated with
these challenges. One approach to addressing vanishing and exploding gradients
is to use either soft or hard constraints on weight matrices so as to encourage
or enforce orthogonality. Orthogonal matrices preserve gradient norm during
backpropagation and may therefore be a desirable property. This paper explores
issues with optimization convergence, speed and gradient stability when
encouraging or enforcing orthogonality. To perform this analysis, we propose a
weight matrix factorization and parameterization strategy through which we can
bound matrix norms and therein control the degree of expansivity induced during
backpropagation. We find that hard constraints on orthogonality can negatively
affect the speed of convergence and model performance.
---

## Semantic Object Parsing with Local-Global Long Short-Term Memory

Xiaodan Liang, Xiaohui Shen, Donglai Xiang, Jiashi Feng, Liang Lin, Shuicheng Yan

Category: lstm-other
Keywords: semantic object parsing, local-global LSTM, computer vision, contextual information, feature learning
Year: 2016

Semantic object parsing is a fundamental task for understanding objects in
detail in the computer vision community, where incorporating multi-level
contextual information is critical for achieving such fine-grained pixel-level
recognition. Prior methods often leverage the contextual information through
post-processing predicted confidence maps. In this work, we propose a novel deep
Local-Global Long Short-Term Memory (LG-LSTM) architecture to seamlessly
incorporate short-distance and long-distance spatial dependencies into the
feature learning over all pixel positions. In each LG-LSTM layer, local guidance
from neighboring positions and global guidance from the whole image are imposed
on each position to better exploit complex local and global contextual
information. Individual LSTMs for distinct spatial dimensions are also utilized
to intrinsically capture various spatial layouts of semantic parts in the
images, yielding distinct hidden and memory cells of each position for each
dimension. In our parsing approach, several LG-LSTM layers are stacked and
appended to the intermediate convolutional layers to directly enhance visual
features, allowing network parameters to be learned in an end-to-end way. The
long chains of sequential computation by stacked LG-LSTM layers also enable each
pixel to sense a much larger region for inference benefiting from the
memorization of previous dependencies in all positions along all dimensions.
Comprehensive evaluations on three public datasets well demonstrate the
significant superiority of our LG-LSTM over other state-of-the-art methods.
---

## Unitary Evolution Recurrent Neural Networks

Martin Arjovsky, Amar Shah, Yoshua Bengio

Category: lstm-other
Keywords: Recurrent Neural Networks, Vanishing Gradients, Exploding Gradients, Unitary Matrices, Long-term Dependencies
Year: 2016

Recurrent neural networks (RNNs) are notoriously difficult to train. When the
eigenvalues of the hidden to hidden weight matrix deviate from absolute value 1,
optimization becomes difficult due to the well-studied issue of vanishing and
exploding gradients, especially when trying to learn long-term dependencies. To
circumvent this problem, we propose a new architecture that learns a unitary
weight matrix, with eigenvalues of absolute value exactly 1. The challenge we
address is that of parametrizing unitary matrices in a way that does not require
expensive computations (such as eigendecomposition) after each weight update. We
construct an expressive unitary weight matrix by composing several structured
matrices that act as building blocks with parameters to be learned. Optimization
with this parameterization becomes feasible only when considering hidden states
in the complex domain. We demonstrate the potential of this architecture by
achieving state-of-the-art results in several hard tasks involving very long-
term dependencies.
---

## Scene Labeling with LSTM Recurrent Neural Networks

Wonmin Byeon, Thomas M. Breuel, Federico Raue, Marcus Liwicki

Category: lstm-other
Keywords: scene labeling, LSTM, recurrent neural networks, pixel-level segmentation, natural scene images
Year: 2015

This paper addresses the problem of pixel-level segmentation and classification
of scene images with an entirely learning-based approach using Long Short Term
Memory (LSTM) recurrent neural networks, which are commonly used for sequence
classification. We investigate two-dimensional (2D) LSTM networks for natural
scene images taking into account the complex spatial dependencies of labels.
Prior methods generally have required separate classification and image
segmentation stages and/or pre- and post-processing. In our approach,
classification, segmentation, and context integration are all carried out by 2D
LSTM networks, allowing texture and spatial model parameters to be learned
within a single model. The networks efficiently capture local and global
contextual information over raw RGB values and adapt well for complex scene
images. Our approach, which has a much lower computational complexity than prior
methods, achieved state-of-the-art performance over the Stanford Background and
the SIFT Flow datasets. In fact, if no pre- or post-processing is applied, LSTM
networks outperform other state-of-the-art approaches. Hence, only with a
single-core Central Processing Unit (CPU), the running time of our approach is
equivalent or better than the compared state-of-the-art approaches which use a
Graphics Processing Unit (GPU). Finally, our networks' ability to visualize
feature maps from each layer supports the hypothesis that LSTM networks are
overall suited for image processing tasks.
---

## Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting

Xingjian Shi, Zhourong Chen, Hao Wang, Dit-Yan Yeung, Wai-kin Wong, Wang-chun Woo

Category: lstm-other
Keywords: precipitation nowcasting, spatiotemporal sequence forecasting, convolutional LSTM, weather forecasting, machine learning
Year: 2015

The goal of precipitation nowcasting is to predict the future rainfall intensity
in a local region over a relatively short period of time. Very few previous
studies have examined this crucial and challenging weather forecasting problem
from the machine learning perspective. In this paper, we formulate precipitation
nowcasting as a spatiotemporal sequence forecasting problem in which both the
input and the prediction target are spatiotemporal sequences. By extending the
fully connected LSTM (FC-LSTM) to have convolutional structures in both the
input-to-state and state-to-state transitions, we propose the convolutional LSTM
(ConvLSTM) and use it to build an end-to-end trainable model for the
precipitation nowcasting problem. Experiments show that our ConvLSTM network
captures spatiotemporal correlations better and consistently outperforms FC-LSTM
and the state-of-the-art operational ROVER algorithm for precipitation
nowcasting.