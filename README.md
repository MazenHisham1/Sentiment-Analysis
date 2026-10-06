# Sentiment140: TF-IDF + Logistic Regression vs DistilBERT

Binary sentiment classification on 1.6M tweets, comparing **model-specific text cleaning**: heavy normalisation for a classical baseline, minimal cleaning for a transformer.

[![Kaggle](https://img.shields.io/badge/Kaggle-Notebook-20BEFF?logo=kaggle&logoColor=white)](YOUR_KAGGLE_NOTEBOOK_LINK)
![Python](https://img.shields.io/badge/python-3.10+-blue)
![License](https://img.shields.io/badge/code%20license-MIT-green)

## Results

| Model | Cleaning | Train size | Test size | Accuracy | F1 |
|---|---|---|---|---|---|
| TF-IDF (1-2 grams) + Logistic Regression | Heavy | ~1.34M | 149,412 | 80.59% | 0.806 (macro) |
| DistilBERT (`distilbert-base-uncased`) | Minimal | 220,000 | 160,000 | 85.22% | 0.850 |

DistilBERT gains about **4.6 accuracy points** over the baseline while training on roughly 6x less data (2 epochs, about 10 minutes on a Colab GPU).

> Note: the two test sets differ slightly (the TF-IDF pipeline removes empty and duplicate cleaned tweets before splitting). See *Next steps* for the planned same-split comparison.

## Dataset

[Sentiment140](https://www.kaggle.com/datasets/kazanova/sentiment140): 1.6M tweets, labels 0 (negative) and 4 (positive), mapped to 0/1. The CSV is `latin-1` encoded with no header row.

- Labels were generated automatically from emoticons (distant supervision), then the emoticons were removed. Labels are **noisy**, so the realistic accuracy ceiling is below 90%.
- The dataset is **not included** in this repo, and it has no clearly stated license. Check the original authors' page and Twitter/X terms before reuse.

Citation: Go, A., Bhayani, R. and Huang, L. (2009). *Twitter sentiment classification using distant supervision.* CS224N Project Report, Stanford.

## Cleaning pipelines

| Step | TF-IDF + LR | DistilBERT |
|---|---|---|
| HTML unescape | Yes | Yes |
| Markdown links | Removed | - |
| Lowercase | Yes | Tokenizer (uncased) |
| Emojis | Converted to text | Left as is |
| URLs | Removed | Replaced with `http` |
| @mentions | Removed | Replaced with `@user` |
| Hashtags | Split into words (`wordninja`) | Left as is |
| Contractions | Expanded (keeps "not") | Left as is |
| Repeated letters | Capped at 2 | Capped at 3 |
| Punctuation / digits | Keep only letters, `!`, `?` | Kept |
| Stopwords | Removed, **negations kept** | Never removed |
| Dedupe / empty rows | Removed after cleaning | - |
| Tokeniser | TF-IDF, `min_df=5`, `max_df=0.9`, `sublinear_tf` | WordPiece, `max_length=64` |

Why: bag-of-words models treat every string variant as a separate feature, so cleaning merges variants and drops noise. Transformers were pre-trained on raw text and use context (negations, punctuation, stopwords), so heavy cleaning removes information they rely on.

## Training setup

- **TF-IDF + LR:** stratified 90/10 split, `ngram_range=(1,2)`, up to 500k features, `LogisticRegression(C=2.0, solver="saga")`, seed 42.
- **DistilBERT:** stratified 90/10 split, 220k training tweets, 2 epochs, `lr=2e-5`, batch size 64, weight decay 0.01, fp16, seed 42. Epoch 1: 84.75% accuracy; epoch 2: 85.22%.

## Repository

```
.
├── notebooks/
│   ├── TF_IDF___LR_Sentiment_Analysis.ipynb
│   └── DISTILBERT_Sentiment_Analysis.ipynb
├── docs/sentiment140_guide.html     # step-by-step reasoning for each cleaning choice
├── requirements.txt
└── README.md
```

## How to run

```bash
pip install pandas scikit-learn nltk emoji wordninja matplotlib joblib \
            transformers datasets torch kagglehub
python -c "import nltk; nltk.download('stopwords')"
```

Open either notebook in Kaggle or Colab. The dataset is downloaded with `kagglehub.dataset_download("kazanova/sentiment140")`. A GPU is needed for the DistilBERT notebook.

Quick inference with the saved baseline:

```python
import joblib
pipe = joblib.load("sentiment_model.joblib")
pipe.predict([clean_classical("I absolutely love this!!!")])   # [1]
```

## Limitations

- Labels come from emoticons, not human annotation, so accuracy partly measures how well a model recovers emoticon polarity.
- Data is from 2009 and English only; slang and topics have changed.
- No neutral class. Sarcasm and mixed sentiment are likely main error sources.

## Next steps

- [ ] Evaluate both models on the **same** test split (same row IDs)
- [ ] Separate validation set for model selection (currently the test split is also used for `load_best_model_at_end`)
- [ ] Cleaning ablation for TF-IDF + LR (raw, + each cleaning step, default vs negation-safe stopwords)
- [ ] Confusion matrix, ROC-AUC, and error analysis of 50-100 misclassified tweets
- [ ] Multiple seeds (mean ± std)
- [ ] Optional: BERT-base comparison
- [ ] Gradio / Streamlit demo on Hugging Face Spaces

## License

Code is released under the MIT License. This covers the code only, not the Sentiment140 dataset.
