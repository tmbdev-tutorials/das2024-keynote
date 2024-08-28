# ONE SENTENCE SUMMARY:
This paper updates Chinchilla scaling laws to include inference costs, showing that smaller, longer-trained models are more cost-effective for high inference demands.

# MAIN POINTS:
1. Large language model (LLM) scaling laws predict model quality changes with increased parameters and training data.
2. Current scaling laws, including Chinchilla, ignore inference costs.
3. Modified scaling laws calculate optimal LLM size considering both training and inference costs.
4. Analysis based on compute budget and real-world costs shows smaller, longer-trained models are more efficient for high inference demands.
5. Training and inference costs are influenced by model size and user query volume.
6. Hoffmann et al. found that parameters and tokens should grow equally for optimal scaling.
7. Chinchilla models are not optimal when considering inference costs.
8. Inference costs are lower for smaller models, justifying the extra training compute.
9. The paper uses pre-training cross-entropy loss and floating-point operations (FLOPs) for analysis.
10. Real-world cost analysis includes hardware utilization differences and quantization.

# TAKEAWAYS:
1. Smaller, longer-trained models are more cost-effective for high inference demands.
2. Including inference costs shifts optimal model size and training data requirements.
3. Pre-training loss can be a proxy for model quality in cost calculations.
4. Real-world deployments benefit from considering both training and inference hardware utilization.
5. Modified scaling laws offer a more practical approach for LLM practitioners expecting significant inference demand.
