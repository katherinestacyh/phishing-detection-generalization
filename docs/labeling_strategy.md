
# Revised Labeling Strategy - Spam Detection

## Revised Research Question

How well do machine-learning spam detection models
generalize to emails from unseen data sources when
source-specific patterns and duplicate leakage are controlled?

## Reason for Changing the Research Focus

Our initial investigation found that most of the
original corpora were created for spam detection
rather than phishing-specific classification.

Only Nazario is clearly phishing-focused, while
Nigerian_Fraud contains fraud and scam messages.

Using these sources together for phishing detection
could cause the model to learn source-specific
patterns instead of phishing characteristics.

We are therefore revising our focus to spam detection
while keeping the original generalization methodology.

This is a proposed scope revision from our approved
phishing research question.

## Main Modeling Sources

We will use four corpora:

- CEAS_08: 39,154 emails
- Enron: 29,767 emails
- Ling: 2,859 emails
- SpamAssasin: 5,809 emails

Total: 77,589 emails before duplicate control.

## Binary Classification Labels

Original label 0 = Legitimate (ham)
Original label 1 = Spam

We will confirm the curated label mappings and
preserve the original labels in every record.

## Excluded Sources

Nazario:
Excluded from the main spam-versus-legitimate
experiment because it is phishing-focused
and contains only positive examples.

Nigerian_Fraud:
Excluded from the main experiment because it
contains fraud/scam emails and only positive examples.

Both sources will remain in our original dataset
documentation and exploratory analysis.

## Data Preparation

- Preserve source_id and original labels.
- Preserve subject, body, and available metadata.
- Document missing and unusual values.
- Identify and control exact and near-duplicates.
- Keep related emails in the same data split.
- Do not modify the original CSV files.

## Evaluation Strategy

1. Conventional random train/validation/test split.
2. Duplicate-controlled evaluation.
3. Source-disjoint evaluation where one entire
   corpus is held out for testing.

Models:
- TF-IDF + Logistic Regression
- TF-IDF + Linear SVM

Metrics:
- Precision
- Recall
- F1
- PR-AUC
- Confusion matrices
- False positives and false negatives
- Performance by source

## Remaining Risks

Switching to spam detection does not automatically
eliminate source bias.

We still need to test whether the models learn
generalizable spam patterns or rely on differences
between the original corpora.

We must also prevent overlapping and near-duplicate
emails from leaking across training and test sets.
