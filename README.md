# Cross-Domain Review Sentiment Classification

**Course:** ARTI 404 — Natural Language Processing, Section 7MA2
**Group:** 4
**Instructor:** Dr. Mohammed Mosuily

**Team members:**

| Name | Student ID | Role |
|---|---|---|
| Ali Jafar Altaweel | 2240006456 | Team leader |
| Mohammed Moayed Alqurini | 2240007325 | Member |
| Danyal Salman Alhashem | 2240002713 | Member |

## 1. Task and intended user

This project implements Option 2 (Review Sentiment Analysis). It trains a binary sentiment classifier (Positive/Negative) on Amazon product reviews and IMDb movie reviews, then evaluates it on an in-domain held-out test and on a cross-domain test of Yelp restaurant reviews. The system takes a short English review as input and returns Positive or Negative.

The intended user is someone who needs a first-pass sort of customer feedback, such as a reviews analyst. It is an academic text-classification demonstration, not a production decision-making system.

**Approved research question:**

> Which method performs better on restaurant reviews when trained on product and movie reviews?

The main judgment is Macro F1 on the Yelp test, with Accuracy reported alongside; the in-domain test is a supporting comparison only. Settings for both methods are selected on Development data only — neither test set is used for tuning.

## 2. Dataset

**Source:** [UCI Sentiment Labelled Sentences](https://archive.ics.uci.edu/dataset/331/sentiment+labelled+sentences) (Amazon product reviews, IMDb movie reviews, Yelp restaurant reviews), licensed under **CC BY 4.0**.

| Source | Original count | Final count (after cleaning) |
|---|---:|---:|
| Amazon | 1000 | 987 |
| IMDb | 1000 | 996 |
| Yelp | 1000 | 996 |
| **Total** | **3000** | **2979** |

Cleaning removed 21 exact-duplicate sentences (18 within a source, 1 across sources) and 0 label conflicts; see `results/data_audit.csv` for the full per-source, per-label breakdown. Amazon and IMDb were split 70/15/15 into Training / Development / In-domain Test (stratified by source and label, `RANDOM_STATE = 42`); the remaining Yelp rows form a frozen Cross-domain Test that is never used for training or tuning.

## 3. Methods

**Method 1 — TF-IDF + Logistic Regression.** An sklearn `Pipeline` of `TfidfVectorizer` and `LogisticRegression`, fit on Training only. Grid search over `ngram_range ∈ {(1,1), (1,2)}` and `C ∈ {0.1, 1, 10}` (6 candidates), selected by highest Development Macro F1.
Selected on Development: `ngram_range = (1,1)`, `C = 10` — Development accuracy 0.822, Development Macro F1 0.821.

**Method 2 — Frozen MiniLM + Logistic Regression.** Sentence embeddings from `sentence-transformers/all-MiniLM-L6-v2` (384-dim, `normalize_embeddings=True`, CPU, used frozen — no fine-tuning), followed by a `LogisticRegression` trained on the Training embeddings. Grid search over `C ∈ {0.1, 1, 10}` (3 candidates), selected by highest Development Macro F1.
Selected on Development: `C = 0.1` — Development accuracy 0.916, Development Macro F1 0.916.

Both grids use a fixed tie-break policy (highest Development Macro F1, then higher Development accuracy, then smaller `C`, then unigrams for TF-IDF, then first in the declared grid order); full Development results are in `results/dev_results.csv`.

## 4. Results

Final test metrics (`results/test_metrics.csv`, rounded to 3 decimal places):

| Method | Test set | n | Accuracy | Macro F1 |
|---|---|---:|---:|---:|
| TF-IDF_LR | in_domain | 298 | 0.815 | 0.815 |
| MiniLM_LR | in_domain | 298 | 0.886 | 0.886 |
| TF-IDF_LR | yelp | 996 | 0.742 | 0.742 |
| MiniLM_LR | yelp | 996 | 0.891 | 0.890 |

**Answer to the research question:**

On the Yelp test, the main judgment for this question, MiniLM_LR outperforms TF-IDF_LR: macro F1 0.890 vs 0.742, and accuracy 0.891 vs 0.742. Accuracy and macro F1 agree on this ranking, so there is no disagreement to report between the two metrics.

The in-domain test is a supporting comparison. MiniLM_LR is essentially stable moving from in-domain to Yelp (macro F1 0.886 to 0.890, Δ = +0.5 pp), while TF-IDF_LR drops noticeably (macro F1 0.815 to 0.742, Δ = -7.4 pp). This is a descriptive pattern only: it shows TF-IDF_LR's score fell more than MiniLM_LR's when the text domain changed, not why it fell, and not that MiniLM's representations "understand" sentiment or language in general. Under this one transfer setup, with this training/development data and these settings, the frozen MiniLM sentence-embedding features produced classifier inputs that carried over to restaurant review text better than the TF-IDF vocabulary learned from Amazon and IMDb reviews did. It does not establish that this holds for other domain shifts, and the test set sizes here (298 in-domain, 996 Yelp) mean the exact percentage-point figures carry some sampling noise.

Confusion matrices for both methods on both test sets are saved as PNG files in `results/`. Error analysis on the Yelp test (selected examples, OOV word checks) is in `results/error_analysis.csv` and in the notebook's Results section.

## 5. How to run

Requirements: Python 3.12, VS Code with the Jupyter extension, and internet access for the first run.

```bash
# from the project root
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python -m ipykernel install --user --name nlp-sentiment --display-name "NLP Sentiment"
```

Then in VS Code:

1. Open `sentiment_project.ipynb`.
2. Select the `NLP Sentiment` (`.venv`) kernel in the top-right kernel picker.
3. Run **Restart** (restart the kernel) followed by **Run All**.

The first run downloads the UCI dataset archive and the MiniLM model weights, so it needs internet access; later runs reuse the files already saved under `data/raw/` and the Hugging Face cache. All paths in the notebook are relative to the project root (`Path.cwd()`), so the project can be run from any machine without editing paths.

The notebook can also be run end-to-end from a terminal:

```bash
.venv\Scripts\python.exe -m jupyter nbconvert --to notebook --execute --inplace sentiment_project.ipynb
```

## 6. Repository structure

```
README.md                          this file
requirements.txt                   pinned package versions
sentiment_project.ipynb            main notebook (Setup, Data, Method 1, Method 2, Results, Try it, Sources)
data/raw/                          downloaded UCI files + download_info.json
data/processed/split_manifest.csv  example_id, source, label, split, duplicate_key for every row
results/data_audit.csv             per-source/label counts before and after cleaning
results/dev_results.csv            all Development grid-search candidates for both methods
results/test_metrics.csv           final accuracy / macro F1 for both methods, both test sets
results/test_predictions.csv       per-example predictions used for evaluation and error analysis
results/error_analysis.csv         selected Yelp error examples with explanations
results/original_examples.csv      six original labelled examples, two per team member
results/confusion_matrix_*.png     confusion matrices for both methods, both test sets
models/tfidf_model.joblib          saved TF-IDF + Logistic Regression pipeline
models/minilm_classifier.joblib    saved MiniLM classifier (encoder is reloaded by model ID, not saved)
evidence/contribution_log.csv      dated log of each member's contributions
```

## 7. Using the Try it interface

The notebook's **Try it** section defines `predict_review(text)`, which runs a new piece of text through both trained models (no re-fitting) and prints each method's predicted label and probability. An empty or whitespace-only input prints a prompt to enter a short English review instead of predicting.

After running all cells, an interactive widget is also available at the end of the Try it section: type a review into the text box and click **Analyse** to see both methods' predictions side by side, or **Clear** to reset it. This lets the system be demoed live without editing any code. A plain code-cell fallback (`predict_review(my_review)`) is kept directly above the widget for environments where the widget does not render.

## 8. Limitations

- This is a single domain transfer (Amazon/IMDb → Yelp); it does not establish generalization to other domains or genuine language understanding.
- Yelp text was already new to the task classifiers at training time, but MiniLM's pretrained weights may have seen similar restaurant-review text during their own pretraining; this cannot be ruled out from our experiment.
- Test set sizes differ (298 in-domain vs. 996 Yelp), and the in-domain test is small, so small differences should be read cautiously.
- Domain and data source are confounded in this dataset (each domain comes from exactly one source), so the effect of domain and of source cannot be fully separated.
- Mixed, neutral, or sarcastic reviews may not be well represented by the binary Positive/Negative labels or by this dataset.
- The dataset is limited in size and somewhat dated; it may not represent current review-writing styles.

## 9. References

1. D. Kotzias, M. Denil, N. de Freitas, and P. Smyth (2015). *From Group to Individual Labels Using Deep Features.* ACM SIGKDD, pp. 597–606. DOI: [10.1145/2783258.2783380](https://doi.org/10.1145/2783258.2783380).
2. N. Reimers and I. Gurevych (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.* EMNLP-IJCNLP, pp. 3982–3992. [Paper](https://aclanthology.org/D19-1410/). DOI: 10.18653/v1/D19-1410.
3. [UCI Sentiment Labelled Sentences](https://archive.ics.uci.edu/dataset/331/sentiment+labelled+sentences) — source of the files, counts, labels, and licence (CC BY 4.0).
4. [all-MiniLM-L6-v2 model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) — source of the pretrained weights used frozen in Method 2.
5. [scikit-learn Pipelines](https://scikit-learn.org/stable/modules/compose.html).
6. [TfidfVectorizer documentation](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html).
7. [LogisticRegression documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html).
8. [f1_score documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html).
9. `NLP_Project_Student_Brief.pdf` — course assignment instructions.
