
# Dataset Source Documentation

## 1. CEAS_08

Original source: CEAS 2008 Spam Filter Challenge

Original purpose:
The CEAS 2008 challenge was designed to evaluate
email spam-filtering systems.

Original categories:
- Ham: legitimate email
- Spam: unsolicited or unwanted email

Important limitation:
The original challenge focuses on spam detection,
not specifically phishing detection. Therefore,
emails labeled as spam cannot automatically be
considered confirmed phishing emails.

Curated dataset:
- File: CEAS_08.csv
- Records: 39,154
- Original label 0: 17,312 records
- Original label 1: 21,842 records

Label interpretation:
The binary labels appear to follow the convention
0 = legitimate and 1 = spam, but the numeric mapping
still needs confirmation against the curated
dataset's documentation.

Original source:
https://ceas.cc/2008/challenge.html

Curated dataset:
https://zenodo.org/records/8339691

Status:
Source purpose verified.
Individual phishing status not verified.
