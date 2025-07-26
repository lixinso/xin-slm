### A research project for Small Language Model (SLM) that train & run on local machine (Mac Mini M4)

## Evaluation Metrics

The following metrics are commonly used when training or evaluating small language models:

- **Cross-entropy loss** – the training objective that measures the negative log-likelihood of the correct token.
- **Perplexity** – the exponential of cross-entropy, providing an interpretable measure of model uncertainty.
- **Bits per character (BPC)** – an alternative to perplexity for character-level models, representing log-likelihood normalized by the number of characters.
- **Accuracy** – useful for discrete prediction tasks such as classification or restricted-vocabulary next-token prediction.
- **Task-specific metrics** – scores like BLEU, ROUGE, or F1 are used when the model is evaluated on particular tasks (e.g., translation or summarization).
