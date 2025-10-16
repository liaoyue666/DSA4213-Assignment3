# Financial Sentiment Fine-Tuning (BERT-base & FinBERT)
This project fine-tunes BERT-base and FinBERT on Financial PhraseBank for 3-class sentiment (Bullish/Neutral/Bearish) using two strategies:
Full fine-tuning
LoRA (PEFT)
It also logs efficiency (trainable params, memory, throughput), exports confusion matrices, and writes error-case CSVs (slices, hard errors, near-misses) for analysis.
