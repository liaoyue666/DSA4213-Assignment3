# Financial Sentiment Fine-Tuning (BERT-base & FinBERT)
This project fine-tunes BERT-base and FinBERT on Financial PhraseBank for 3-class sentiment (Bullish/Neutral/Bearish) using two strategies:
Full fine-tuning
LoRA (PEFT)
It also logs efficiency (trainable params, memory, throughput), exports confusion matrices, and writes error-case CSVs (slices, hard errors, near-misses) for analysis.

## 1) Environment Setup
## Option A — Conda (recommended)
```bash
conda create -n finnlp python=3.10 -y
conda activate finnlp
pip install --upgrade pip
pip install "transformers==4.44.2" "datasets==2.19.1" "pyarrow==16.1.0" \
            "accelerate>=0.33.0" peft==0.11.1 scikit-learn==1.5.1 \
            matplotlib pandas numpy

## Option B - Virtualenv
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install "transformers==4.44.2" "datasets==2.19.1" "pyarrow==16.1.0" \
            "accelerate>=0.33.0" peft==0.11.1 scikit-learn==1.5.1 \
            matplotlib pandas numpy
## Notes
If you hit pyarrow / C-extension issues, ensure the pinned versions above and remove older builds:
pip uninstall -y pyarrow datasets fsspec && pip cache purge then reinstall.
GPU is optional; accelerate will pick CUDA if available.


