# Evaluation Metrics

The following metrics are commonly used when training or evaluating small language models:

- **Cross-entropy loss** – the training objective that measures the negative log-likelihood of the correct token.
- **Perplexity** – the exponential of cross-entropy, providing an interpretable measure of model uncertainty.
- **Bits per character (BPC)** – an alternative to perplexity for character-level models, representing log-likelihood normalized by the number of characters.
- **Accuracy** – useful for discrete prediction tasks such as classification or restricted-vocabulary next-token prediction.
- **Task-specific metrics** – scores like BLEU, ROUGE, or F1 are used when the model is evaluated on particular tasks (e.g., translation or summarization).

# Benchmark Tests

## Image Reasoning
- **MMMU** – Multimodal understanding benchmark
- **MathVista** – Mathematical visual reasoning

## Image Understanding
- **ChartQA** – Chart question answering
- **DocVQA** – Document visual question answering

## Coding
- **LiveCodeBench** – Real-world coding tasks evaluation

## Reasoning & Knowledge
- **MMLU Pro** – Massive multitask language understanding (professional level)
- **GPQA Diamond** – Graduate-level science questions
- **MATH-500** – Mathematical problem solving

## Multilingual
- **Multilingual MMLU** – Cross-lingual understanding evaluation

## Long Context
- **MTOB** – Machine translation of books (evaluates long context understanding)