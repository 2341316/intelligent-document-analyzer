# Error Analysis – Document Section Classification

## 1. Overview

This analysis evaluates the performance of both the classical machine learning model and the transformer-based model used for classifying sections of corporate annual reports.

The classical model uses TF-IDF features with Logistic Regression, while the transformer model uses DistilBERT fine-tuning to capture contextual relationships in text.

The dataset used for training and evaluation was generated from multiple company annual reports and processed through the full pipeline including:

* PDF ingestion
* text cleaning
* sentence-aware chunking
* automatic section labeling

Each document was divided into text chunks which were then classified into predefined document sections.

**Dataset statistics**

Total dataset size: 1418 chunks
Classical TF-IDF model accuracy: ~86%
Transformer (DistilBERT) model accuracy: ~84%

Although transformer models are generally more powerful, the results in this project highlight how dataset characteristics strongly influence model performance.



# 2. Dataset Characteristics

The dataset exhibits a significant class imbalance, which strongly impacts classification behavior.

| Section               | Count |
| --------------------- | ----- |
| Financial_Statements  | 964   |
| Sustainability        | 307   |
| Governance            | 92    |
| Management_Discussion | 29    |
| Risk_Management       | 20    |
| Corporate_Overview    | 6     |

Financial Statements account for approximately 68% of the dataset, making it the dominant class.

Some categories such as Corporate Overview contain extremely few examples, which makes it difficult for models to learn meaningful patterns for these sections.

This imbalance causes both classical and transformer models to favor predictions toward the dominant classes, particularly Financial Statements.



# 3. Chunk Length Observations

The statistics for chunk length are:
Mean length ≈ 392 words
Maximum length = 600 words

Chunks were generated using sentence-aware chunking with a target size between 400–600 words.

While this approach preserves semantic context, longer chunks may sometimes contain multiple topics.

For example, a single chunk may include both:
* financial performance information
* sustainability initiatives
* governance commentary

When multiple topics appear in the same chunk, classification becomes more difficult because the text does not clearly represent a single document section.



# 4. Confusion Matrix Analysis (TF-IDF Model)

The confusion matrix for the TF-IDF classifier reveals several patterns in model predictions.

### Strong performance on Financial Statements

The model correctly predicts most Financial Statement chunks.

Financial sections contain distinctive terminology such as:
* revenue
* profit
* assets
* liabilities
* cash flow

These keywords are highly informative in TF-IDF representations, allowing the model to identify financial sections with high confidence.



### Confusion between Sustainability and Financial sections

Some Sustainability sections are misclassified as Financial Statements.

This occurs because sustainability reports frequently include financial discussions when describing investments in environmental initiatives.

Examples include:
* investment in renewable energy
* carbon reduction funding
* sustainable infrastructure spending

Because TF-IDF relies primarily on word frequency, these overlapping terms can cause misclassification.



### Confusion between Governance and Financial sections

Governance sections sometimes include terms such as:
* audit committee
* financial oversight
* internal controls
* regulatory compliance

These words frequently appear in financial contexts as well, which leads the model to incorrectly classify some Governance text as Financial Statements.



### Poor performance on minority classes

Categories such as:
* Corporate_Overview
* Risk_Management
* Management_Discussion

contain very few training examples.

With such limited data, the model cannot learn reliable vocabulary patterns for these sections.



# 5. Limitations of TF-IDF Models

The TF-IDF representation treats documents as a bag of words, meaning it considers word frequency but ignores deeper semantic relationships.

For example, the sentence:

> The company invested $500 million in renewable energy projects.

contains words like:
* invested
* million
* projects

These words appear frequently in both financial and sustainability discussions.

Since TF-IDF does not capture contextual meaning, it struggles to distinguish between these different semantic uses.



# 6. Role of Transformer Models

Transformer models such as **DistilBERT** analyze text differently from classical models.

Instead of relying only on word frequency, transformers learn contextual relationships between words using attention mechanisms.

For example, a transformer can distinguish between:
* investment in renewable energy initiatives
* capital investment reported in financial statements

because it considers surrounding words and sentence structure.

This ability allows transformer models to better understand complex corporate documents.



# 7. Transformer Model Evaluation

A transformer-based classifier using DistilBERT was trained to compare performance with the classical TF-IDF model.

The model was fine-tuned using the HuggingFace Transformers library for 1 training epoch.

### Performance metrics

Accuracy ≈ 84%
Weighted F1 ≈ 0.80
Macro F1 ≈ 0.28

While the overall accuracy is comparable to the TF-IDF model, the macro F1 score reveals that the transformer struggles with minority classes.

### Observed prediction behavior

The confusion matrix shows that the transformer primarily predicts two dominant classes:

* Financial_Statements
* Sustainability

Minority categories such as:

* Governance
* Risk_Management
* Management_Discussion
* Corporate_Overview

are rarely predicted.

This indicates that the model effectively simplifies the problem into a two-class classification task dominated by the largest categories.

### Root causes

This behavior is primarily caused by:

1. Severe dataset imbalance
2. Extremely small minority classes
3. Limited training (only 1 epoch)

Transformer models generally require larger and more balanced datasets along with longer training to fully capture complex distinctions between document sections.



# 8. Key Insights

The main findings from this error analysis are:

1. The dataset contains severe class imbalance, with Financial Statements dominating the dataset.
2. Vocabulary overlap between financial, sustainability, and governance sections leads to classification ambiguity.
3. Long text chunks can contain multiple topics, reducing classification clarity.
4. Minority classes such as Corporate Overview and Risk Management have very limited training samples.
5. The transformer model largely predicts dominant categories due to the dataset distribution.
6. Model complexity alone cannot compensate for imbalanced or insufficient training data.



# 9. Potential Improvements

Several improvements could enhance model performance in future iterations:

* Increasing dataset size by incorporating additional company annual reports
* Balancing the dataset to provide sufficient samples for minority classes
* Reducing chunk size to minimize multi-topic segments
* Training transformer models for additional epochs
* Applying class weighting or sampling techniques to mitigate class imbalance
* Exploring hierarchical document classification strategies
* Incorporating section metadata to assist classification



# Conclusion

Error analysis highlights the importance of data quality, class balance, and contextual representation in document classification tasks.

While transformer models offer powerful contextual understanding, their performance remains heavily dependent on the structure and distribution of the training data.

This analysis provides valuable insights that can guide future improvements in both dataset preparation and model design.
