# Summary: Accuracy, recall, precision, and related metrics

## Summary: Accuracy, Recall, Precision, and Related Metrics

**Main Topic:** This note defines and explains accuracy, recall (true positive rate), false positive rate, and precision, all common metrics used to evaluate classification models at a fixed threshold. It emphasizes their mathematical definitions, interpretations in the context of spam classification, and their relevance (or lack thereof) for imbalanced datasets. The note also highlights the trade-offs between precision and recall.

**Key Points:**

*   **Threshold Dependence:** All metrics discussed are calculated at a single, fixed threshold, and their values change when the threshold changes. Users often tune the threshold to optimize one of these metrics.
*   **Accuracy:**
    *   Definition: Proportion of all classifications (positive and negative) that are correct.
    *   Usefulness: Good coarse-grained measure for balanced datasets. Can be misleading for imbalanced datasets.
*   **Recall (True Positive Rate):**
    *   Definition: Proportion of actual positives correctly classified as positive. Also known as "probability of detection."
    *   Usefulness: More meaningful than accuracy for imbalanced datasets with few positive instances.
*   **False Positive Rate:**
    *   Definition: Proportion of actual negatives incorrectly classified as positive. Also known as "probability of false alarm."
    *   Usefulness: Less meaningful when the number of actual negatives is very low.
*   **Precision:**
    *   Definition: Proportion of positive classifications that are actually positive.
    *   Usefulness: Less meaningful when the number of actual positives is very low.
*   **Precision-Recall Tradeoff:** Improving precision often worsens recall, and vice versa, due to the effect of the classification threshold on false positives and false negatives.
*   **Choice of Metric:** The choice of metric to prioritize depends on the specific costs, benefits, and risks associated with the problem.

