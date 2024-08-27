# ONE SENTENCE SUMMARY:
A semi-supervised learning method enhances OCR post-correction for endangered languages by combining self-training and lexically aware decoding, reducing error rates significantly.

# MAIN POINTS:
1. Vast textual data in endangered languages remain non-digitized.
2. OCR systems produce digitized text but often contain errors.
3. Neural post-correction models improve OCR outputs but need extensive curated data.
4. Semi-supervised learning leverages raw images for better performance.
5. Self-training iteratively enhances model accuracy using its outputs.
6. Lexically aware decoding ensures consistency by using a count-based language model.
7. Weighted finite-state automata (WFSA) facilitate efficient decoding.
8. Experiments on four languages showed 15%–29% error reduction.
9. The combined self-training and lexical decoding method was crucial for improvements.
10. The approach uses minimal manually transcribed data and larger unannotated datasets.

# TAKEAWAYS:
1. Combining self-training with lexically aware decoding significantly improves OCR post-correction.
2. The method reduces dependency on extensive manually curated data.
3. Lexically aware decoding uses a count-based language model for efficient predictions.
4. Self-training iteratively improves model performance with pseudo-training data.
5. The approach is effective across multiple endangered languages, reducing error rates up to 29%.
