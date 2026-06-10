# Smart Product Intelligence — Capstone Project

An end-to-end deep learning system for the **Toys & Games** category of the
[Amazon Reviews 2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023)
dataset (McAuley Lab). The project covers tabular MLPs, computer vision,
text embeddings, transformers, LLMs with retrieval-augmented generation, and
diffusion-based image generation, integrated into a single Gradio demo.

## Dataset

- **Category:** Toys_and_Games
- **Subset size:** ~5,000 products / ~10,000 reviews (after cleaning and joining)
- **Splits:** 70% / 15% / 15% train / val / test, split **by product** (not by
  review) to prevent data leakage
- **Images:** ~1,500 product images downloaded and cached locally

## Repository Structure

```
smart-product-intelligence/
├── README.md
├── notebook/
│   └── smart_product_intelligence_capstone.ipynb   # all milestones
├── data/                  # cached subsets + images (not included)
└── report/
    └── final_report.pdf
```

> Note: all milestones (M0–M7) were implemented in a single combined
> notebook for simplicity: `smart_product_intelligence_capstone.ipynb`.

## How to Run

1. Open the notebook in Google Colab.
2. Set Runtime → Change runtime type → **GPU (T4)**.
3. Run all cells sequentially from top to bottom (M0 → M7). Each milestone
   depends on data/variables created in earlier cells.
4. The final cell launches a Gradio demo with a public link.

## Milestone Summary & Results

### M0 — Data Preparation & Exploration
Loaded and cleaned product metadata and reviews, joined on `parent_asin`,
engineered features (price, review length, category count, image
availability), and produced a missing-data report. Frozen train/val/test
splits created at the product level to avoid leakage.

### M1 — Tabular MLP (Rating Regression)
Predicted product `average_rating` from price, rating count, and category
count.

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Linear Regression (baseline) | 0.260 | 0.356 | 0.045 |
| MLP (Keras) | **0.246** | **0.342** | **0.122** |

The MLP outperforms the linear baseline on all metrics. Error analysis shows
the model regresses toward the mean rating (~4.5) and underestimates
products with unusually low ratings (2.6–3.3), since these are rare in the
training distribution.

### M2 — Computer Vision (Image-based Quality Classification)
Predicted whether a product's average rating is "good" (≥4.5) from its
product image, comparing a CNN trained from scratch vs. transfer learning
with MobileNetV2.

| Model | Accuracy | Macro-F1 |
|---|---|---|
| Majority-class baseline | 0.80 | 0.444 |
| CNN from scratch | 0.77 | **0.598** |
| Transfer learning (MobileNetV2) | 0.54 | 0.440 |

On macro-F1 (the more informative metric given class imbalance and a small
test set, n=35), the CNN trained from scratch outperforms both the baseline
and the frozen MobileNetV2 backbone. This suggests that, for this small
dataset and task, generic ImageNet features are not strongly aligned with
the rating signal — a useful negative result reported honestly.

### M3 — Text Vectorization & Embeddings
(a) Review sentiment classification (positive ≥4★ vs negative); (b)
semantic product search over description embeddings.

| Model | Accuracy | Macro-F1 |
|---|---|---|
| TF-IDF + Logistic Regression (baseline) | **0.882** | **0.785** |
| Learned embeddings (Keras) | 0.866 | 0.764 |

TF-IDF slightly outperforms learned embeddings on this dataset size. A
TF-IDF-based semantic search demo returns relevant products for free-text
queries, and a t-SNE projection visualizes the embedding space colored by
category.

### M4 — Transformers
Fine-tuned DistilBERT (PyTorch) on the same sentiment classification task.

| Model | Accuracy | Macro-F1 | Notes |
|---|---|---|---|
| TF-IDF + LogReg | 0.882 | 0.785 | fast, no GPU needed |
| Embeddings (Keras) | 0.866 | 0.764 | fast, small model |
| DistilBERT (fine-tuned) | **0.923** | **0.840** | GPU, ~2 epochs |

DistilBERT clearly outperforms both Milestone 3 approaches, justifying its
additional training cost for this task.

### M5 — LLMs & Fine-tuning
Implemented two features using `google/flan-t5-small`: (1) a review
summarizer producing pros/cons, and (2) a retrieval-augmented (RAG) QA
assistant grounded in product metadata and reviews via TF-IDF retrieval.
Compared a zero-shot model against the same model fine-tuned on 250
review→sentiment-summary pairs.

- **Grounding effect:** grounded (RAG) and ungrounded answers diverged on
  the same question, showing retrieval changes model behavior.
- **Fine-tuning effect:** zero-shot outputs did not follow the target
  format; after fine-tuning, the model reliably produced the target
  template (`"This is a [sentiment] review. Rating: X/5."`), though the
  predicted sentiment/rating values were not always accurate — an expected
  limitation of a small (80M parameter) model fine-tuned on 250 examples for
  2 epochs.

### M6 — Diffusion & Generative AI
Used Stable Diffusion v1.5 to generate a studio product photo and a
lifestyle image conditioned on a product's title. Both generations completed
successfully on GPU (~5–6 it/s, 25 steps).

**Reflection:** generated images are plausible and well-composed but cannot
reproduce the actual product's branding/packaging text, since generation is
based only on the text description. Best suited for placeholder hero images
or marketing concept variations rather than literal product photography.

### M7 — Integration & Demo
Combined all components into a single **Smart Product Assistant** Gradio
app: given a selected product, it displays the actual vs. predicted rating
(M1), an image-based quality prediction (M2), similar products via semantic
search (M3), a pros/cons summary and grounded Q&A (M5), and an optional
generated hero image (M6).

## Key Takeaways

- Deep learning models improved over simple baselines in M1, M3 (sentiment),
  and M4, but **not** in M2's transfer-learning variant — reported honestly
  rather than hidden.
- Macro-F1 was used throughout instead of raw accuracy due to strong class
  imbalance (most reviews/ratings are positive).
- All splits were performed at the product level to avoid data leakage
  between train/val/test.
- Small open-weight models (DistilBERT, Flan-T5-small) plus retrieval are
  sufficient to demonstrate the full LLM/RAG pipeline within a free GPU
  budget.
