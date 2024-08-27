## ONE SENTENCE SUMMARY:
This thesis explores the use of fine-tuned Large Language Models (LLMs) for improving the accuracy of Optical Character Recognition (OCR) outputs, showing significant character error rate reductions in modern documents.

## MAIN POINTS:
1. OCR post-correction improves text accuracy by fixing mistakes in OCR outputs from images/documents.
2. Pretrained LLMs like ByT5 can enhance OCR accuracy but require fine-tuning.
3. ByT5 models showed better performance than state-of-the-art methods for correcting OCR errors.
4. Preprocessing techniques like lowercasing and removing strange characters improve model effectiveness.
5. Optimal context length for ByT5 was found to be 50 characters.
6. Few-shot learning was ineffective for teaching LLMs OCR correction without fine-tuning.
7. The ByT5 model achieved up to 56% Character Error Rate (CER) reduction in modern documents.
8. LLMs struggled with historic documents due to language differences in the pretraining data.
9. The baseline method had higher precision but lower recall compared to ByT5 models.
10. Larger LLMs and domain-specific pretraining could further enhance OCR post-correction.

## TAKEAWAYS:
1. Fine-tuned ByT5 models outperform state-of-the-art methods for modern OCR post-correction.
2. Preprocessing techniques like lowercasing and strange character removal are crucial for improving LLM performance.
3. Context length significantly impacts the effectiveness of LLMs in OCR correction, with 50 characters being optimal.
4. Few-shot learning is insufficient for OCR correction tasks without model fine-tuning.
5. Larger and domain-specific pretrained LLMs hold potential for further improvements in OCR post-correction.
