
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


## 2. Enron

Original source: Enron-Spam Dataset

Original purpose:
The Enron-Spam dataset was created to evaluate
email spam-filtering methods. It combines legitimate
emails from Enron employees with spam emails
collected from several other sources.

Original categories:
- Ham: legitimate email
- Spam: unwanted or unsolicited email

Important limitation:
The original dataset was designed for spam detection,
not specifically phishing detection. Therefore,
emails labeled as spam cannot automatically be
considered confirmed phishing emails.

Another concern is that some spam messages came
from collections such as SpamAssassin, which could
create overlapping emails across our datasets.
This will need to be checked during duplicate analysis.

Curated dataset:
- File: Enron.csv
- Records: 29,767
- Original label 0: 15,791 records
- Original label 1: 13,976 records
- Available columns: subject, body, label
- Date column: Not available

Collection period:
The Enron-Spam research was published in 2006.
The original paper describes email collections
from different periods, including spam collected
between 2001 and 2005.
The exact date range for our curated Enron.csv
has not yet been verified.

Label interpretation:
The source distinguishes legitimate email from
general spam. The numeric mapping in our curated
CSV still requires confirmation against its
documentation.

Original research:
https://www2.aueb.gr/users/ion/docs/ceas2006_paper.pdf

Curated dataset:
https://zenodo.org/records/8339691

Status:
Source purpose documented.
Original spam and legitimate categories identified.
Numeric label mapping pending confirmation.
Individual phishing status not verified.



## 3. Ling (Ling-Spam)

Original source: Ling-Spam Corpus

Original purpose:
The Ling-Spam corpus was developed to evaluate
email spam-filtering methods.

Original categories:
- Ham: legitimate messages from a linguistics mailing list
- Spam: unwanted emails collected separately

Important limitation:
Ling-Spam was created for general spam detection,
not specifically phishing detection. Therefore,
spam messages cannot automatically be considered
confirmed phishing emails.

Curated dataset:
- File: Ling.csv
- Records: 2,859
- Original label 0: 2,401 records
- Original label 1: 458 records
- Available columns: subject, body, label
- Date column: Not available

Difference from original corpus:
The original Ling-Spam corpus contains 2,893
messages: 2,412 legitimate and 481 spam.

Our curated CSV contains 34 fewer messages.
The reason for this difference has not yet
been verified.

Collection period:
Not yet verified for the curated CSV.
The original corpus was described in research
published around 2000-2001.

Label interpretation:
The original corpus distinguishes legitimate
email from general spam.

The curated numeric labels appear consistent
with 0 = legitimate and 1 = spam, but the exact
mapping requires confirmation.

Privacy and licensing:
The original messages were drawn from a public
mailing list and a separate spam collection.
Source-specific licensing terms still need review.

Original research:
https://aclanthology.org/W01-0506.pdf

Curated dataset:
https://zenodo.org/records/8339691

Status:
Source purpose documented.
Original ham and spam categories identified.
Curated dataset differences identified.
Numeric label mapping pending confirmation.
Individual phishing status not verified.



## 4. Nazario

Original source: Nazario Phishing Corpus

Original purpose:
The Nazario corpus is a collection of phishing
emails associated with researcher Jose Nazario.
It has been used to study phishing techniques
and how phishing messages attempt to deceive users.

Original categories:
- Phishing emails
- No legitimate email class in our curated file

Important limitation:
Unlike CEAS, Enron, and Ling, this source focuses
specifically on phishing rather than general spam.

However, the curated file contains only positive
examples. This creates a limitation for
source-disjoint evaluation because some metrics
cannot be calculated when the test set contains
only one class.

Curated dataset:
- File: Nazario.csv
- Records: 1,565
- Original label 1: 1,565 records
- Original label 0: 0 records
- Available columns: sender, receiver, date,
  subject, body, urls, label

Date analysis:
- Originally missing dates: 1
- Successfully parsed dates: 1,555
- Non-missing dates that failed parsing: 9
- Earliest parsed date: October 30, 2015
- Latest parsed date: December 27, 2022

Collection period:
The parsed email timestamps span 2015-2022.
This does not necessarily establish the original
collection period or confirm that all messages
were collected during those years.

Label interpretation:
The curated source identifies its messages as
phishing. All records have original label 1.
Individual email labels have not been independently
verified.

Privacy and licensing:
The collection contains email text and potentially
identifying sender or recipient information.
We will not upload raw email records to our public
GitHub repository. Original source-specific usage
terms still need verification.

Original corpus:
https://monkey.org/~jose/phishing/

Curated dataset:
https://zenodo.org/records/8339691

Status:
Phishing-focused source documented.
Original label distribution verified.
Date quality assessed.
Individual labels not independently verified.



## 5. Nigerian_Fraud

Original source: Nigerian Fraud Email Corpus

Original purpose:
This corpus contains fraudulent emails, including
messages associated with Nigerian advance-fee
fraud, also known as 419 scams.

These emails typically attempt to deceive recipients
into sending money or responding to fraudulent
financial opportunities.

Original categories:
- Fraud or scam emails
- No legitimate email class in our curated file

Important limitation:
The curated dataset identifies Nigerian Fraud as
a phishing-related source. However, financial fraud
and credential-stealing phishing are not always
the same type of attack.

Therefore, we cannot automatically assume that
every fraudulent email meets our project's
definition of phishing.

The source also contains only positive examples,
which limits the evaluation metrics available
when holding out this source for testing.

Curated dataset:
- File: Nigerian_Fraud.csv
- Records: 3,332
- Original label 1: 3,332 records
- Original label 0: 0 records
- Available columns: sender, receiver, date,
  subject, body, urls, label

Date analysis:
- Originally missing dates: 482
- Successfully parsed dates: 2,850
- Non-missing dates that failed parsing: 0
- Dates before 1995: 13
- Dates after 2026: 1
- Earliest date within screening range:
  January 1, 1998
- Latest date within screening range:
  December 1, 2015

Collection period:
The parsed email timestamps include dates
from 1998 through 2015 within our initial
screening range.

The original collection period for this exact
curated file has not yet been independently
verified. Some timestamps require further review.

Label interpretation:
All records have original label 1.
The source contains fraudulent emails, but
individual phishing labels have not been verified.

Privacy and licensing:
The records may contain sender and recipient
information. Raw emails will remain in our local
data folder and will not be uploaded to GitHub.

The curated collection is available under
CC BY 4.0 on Figshare. Original source-specific
usage terms still need verification.

Curated dataset:
https://zenodo.org/records/8339691

Curated collection documentation:
https://figshare.com/articles/dataset/Phishing_Email_11_Curated_Datasets/24952503

Status:
Fraud-focused source documented.
Original label distribution verified.
Date quality assessed.
Original collection provenance still under review.
Individual phishing status not verified.




## 6. SpamAssasin (SpamAssassin)

Original source: Apache SpamAssassin Public Mail Corpus

Original purpose:
The SpamAssassin Public Mail Corpus was created
to help developers test email spam-filtering
systems.

Original categories:
- Ham: legitimate or non-spam email
- Spam: unwanted email

The original corpus includes:
- Easy ham: legitimate messages that are easier
  to distinguish from spam
- Hard ham: legitimate messages that resemble spam
- Spam: unwanted email messages

Important limitation:
The original corpus was created for spam detection,
not specifically phishing detection.

Therefore, spam emails cannot automatically be
considered confirmed phishing emails.

Curated dataset:
- File: SpamAssasin.csv
- Records: 5,809
- Original label 0: 4,091 records
- Original label 1: 1,718 records
- Available columns: sender, receiver, date,
  subject, body, urls, label

Difference from original corpus:
The official SpamAssassin documentation describes
a corpus containing 6,047 messages.

Our curated CSV contains 5,809 messages,
which is 238 fewer.

The reason for this difference has not yet
been verified.

Date analysis:
- Originally missing dates: 0
- Successfully parsed dates: 5,517
- Non-missing dates that failed parsing: 292
- Dates before 1995: 63
- Dates after 2026: 1
- Earliest date within screening range:
  January 4, 1997
- Latest date within screening range:
  October 8, 2020

Collection period:
The original SpamAssassin corpus was assembled
and updated during 2002-2005.

Some timestamps in our curated file fall
outside that period, so we cannot assume
all email headers contain accurate dates.

Label interpretation:
The original corpus distinguishes legitimate
email from general spam.

The numeric mapping in our curated CSV appears
to follow 0 = legitimate and 1 = spam,
but this still requires confirmation.

Privacy and licensing:
The original documentation explains that
some email addresses and hostnames were
obfuscated, while other headers were preserved.

Copyright for message text remains with the
original senders.

We will keep the raw email data out of our
public GitHub repository.

Original corpus documentation:
https://spamassassin.apache.org/old/publiccorpus/readme.html

Curated dataset:
https://zenodo.org/records/8339691

Status:
Original corpus purpose documented.
Original ham and spam categories identified.
Differences in record counts identified.
Date quality assessed.
Exact numeric label mapping still requires
confirmation.
Individual phishing status not verified.
