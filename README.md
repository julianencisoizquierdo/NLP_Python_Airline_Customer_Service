# Tweet Analysis for the Airline Industry using NLP

## Content of the project
We are working for the customer service team of a company in the airline industry. Our overarching goal is to benchmark against the competition by analyzing their tweets. We are provided with the dataset (`airline.csv`) and are tasked to answer the following four questions:
1) What is the average length of a social customer service reply?
2) What type of links were referenced most often? URL links, phone numbers or direct messages?
3) How many people should be on a social media customer service team? 
4) How many social replies are reasonable for a customer service representative to handle?


## Instructions
The main file used for the analysis is `airline.csv`. It contains the text of the tweet, as well as information about the days in which those tweets were written.


## Rationale and Methodology


In this case, we are analyzing an “Amazon Customer Reviews” dataset, which includes a large and varied text content, making it a perfect fit for sentiment analysis. However, this requires converting unstructured text into a format for machine learning, known as vectorization, which can be very challenging. For this, we are using various vectorization methods like Bag of Words with TF-IDF, embeddings with FastText and topic modeling with TDA that we will then compare to choose the most appropriate and performant method for our analysis.

For some cases, we use text preprocessing, lemmatization in our case, that also affects data analysis, as it is helpful in reducing word form variability.

The aim of our analysis is to understand how different vectorization techniques and preprocessing methods impact sentiment classification models using the Amazon Customer Reviews dataset.

## Methodology 

## Data Overview 

The dataset is composed of different column, mainly: 

Review Text: This column contains the full text of the customers reviews. It is an all-inclusive field that includes the reviewer's experience, feedback on the product. It presents a rich resource for NLP applications, serving as our foundation for sentiment analysis.

Overall Rating: Accompanying each review is a star rating, ranging from 1 to 5, found in the "overall" column. This rating is a quantitative reflection of the customer's satisfaction with the product. We define reviews with star ratings of 4 or 5 as positive representation of the high satisfaction of the client with the product, and ratings of 1 or 2 as a negative representation showing the client’s dissatisfaction with the product. The plot shows the count of the reviews.



We convert the ratings to a binary representation, where “1” and ”2” ratings from the overall columns are stored as 0, to represent the negative sentiment of the client and “4” and ”5” are stored as 1, representing the positive sentiment of the client. This new variable is stored in the sentiment column, that is our target variable. The resulting class counts appear to be relatively balanced, with 150,000 counts for positive sentiment and 144,240 counts for negative sentiment. 

### Data splitting

Data splitting is performed before any preprocessing steps to avoid data leakage. Following the methodology outlined by Pinjosovsky (2023), all preprocessing steps performed on the training set are then replicated on the test set. 

Adhering to best practices in machine learning methodology, 80% of our data, totaling 235,392 observations, are allocated for training and the remaining 20% are used to test the performance of the chosen machine learning models. 

### Data Preparation and Preprocessing

The preprocessing is important to prepare the data set for further analysis and modeling. In this step, we perform preprocessing of non-text and text columns.

#### Pre-processing non-text columns

First, we augment the features with a length variable that stores each review’s character count, which can be a useful indicator of the customer’s sentiment.  

Moreover, we convert the ‘unixReviewTime’ column, that represents the time at which the review was left, from its original format to a much more interpretable datetime format. This allows us to extract important information such as the year, the month and the day the review was done by the customer. 

Next, we convert the ‘verified’ feature, that indicates if the purchase was verified or not, from its true/false original format to a binary format, where ‘True’ is represented by 1 and ‘False’ is represented by 0. This step is important to make the features set fitted for machine learning. 
Finally, we decide to remove certain columns from the dataset to reduce complexity and improve the efficiency of our analysis. Specifically, we eliminate the ‘reviewTime,’  'summary', 'ReviewerID', and 'asin' columns. The 'summary' and ‘reviewTime’ columns are redundant as they contain information duplicated in other variables. Considering our focus on sentiment classification, the ‘ReviewerID’ and ‘asin’ (amazon product identification key) are judged unlikely to significantly contribute to the predictive power of our models. On the other hand, retaining them would have led to high dimensionality after one-hot encoding due to their numerous unique values, which would have increased computational complexity. 

#### Pre-processing the ‘reviewText’ column

The first step is to remove punctuation marks from the ‘reviewText’ column, in order to reduce noise and standardize the text format. This step is important to delete all the unnecessary characters and information from the text.

We observe that some reviews can contain several repeated letters and characters which can affect our analysis (e.g. “waaaay too big”). To remedy that, we define a function that uses regular expressions to identify any 3 or more similar successive letters and replace them with a single character. 

We then create a pipeline to apply the steps above, as well as to transform the text to lowercase. The pipeline is applied to both the test set and the training set, to ensure consistency.

Finally, we define a set of stopwords to be used in the vectorization methods. We download a standard set of English stopwords from NLTK that we adjust by adding or removing specific words. 

### Choice of Vectorization techniques

Since our goal is to compare the effect of vectorization techniques on model performance, we select 3 distinct methods ranging from traditional bag-of-words approaches to more advanced topic modeling and word embedding methods. Each method offers a different approach to representing textual data, effectively capturing different aspects of the underlying semantics in the text. 

#### Word Embeddings with FastText

We initially set out to employ BERT embeddings, a contextual method which considers the information surrounding each word to generate embeddings. However, we encountered challenges related to the significant computational resources required. Compared to deep contextual embeddings, such as BERT, non-contextual embeddings (e.g. GloVe, FastText) require negligible time with minimal compromise in performance, particularly on tasks with large labeled data and simple language (Arora et al., 2020.) Considering this trade-off between performance and computational efficiency, we opt for the FastText algorithm.

We start by breaking down the text in the ‘reviewText’ column into tokens using the word_tokenize function from the Natural Language Toolkit library. These tokenized reviews are then used to train a FastText model, which transforms words into vectors, such that similar words are assigned closer values in the vector space. A vector size of 100 is set to allow for more nuanced representations while remaining computationally manageable. The minimum count parameter is set to 1, so that words with frequencies below this threshold are excluded from the vocabulary.

After training the FastText model, we obtain dense vector representations for each word in the vocabulary. We then define a function (get_review_vector) to aggregate these word vectors into a single vector representation for each review. This process involves averaging the word vectors for all tokens present in the review text. If a token is not found in the FastText vocabulary, it is assigned a zero vector.

Finally, we apply another aggregation function (aggregate_vectors) to transform each review vector into a single value corresponding to the mean of the word vectors across all tokens in the review. 

The final features in our training set are: verified, review_vectors, length, year, month, and day. 

#### Topic Modeling with LDA

We employ CountVectorizer to transform the textual data, removing stop words and excluding rare terms (min_df=0.001) and very common terms (max_df=0.95). The resulting count vectors are then used as input for Latent Dirichlet Allocation (LDA). Initially, the LDA model is configured with 10 components, a random state of 42 and it utilizes parallel processing for efficiency (n_jobs=-1).

An overview of the 10 generated groups reveals a mixed quality of topics, with some groups displaying coherent themes while others seem less informative. For instance, Topic 03 includes a cluster of terms associated with negative sentiments, such as "cheap," "money," “bad,” "waste," and “never,” suggesting customer dissatisfaction. On the other hand, Topic 09 seems to capture positive sentiments, with terms like "good," "works," “nice,” and "great.” However, not all topics are as discernible. Topic 08, for example, exhibits less coherence, combining terms such as "broke" "get," "excellent" and “months”, from which we can hardly derive meaningful interpretations. 

To refine our topic modeling, we experiment with increasing the number of topics to 25. This adjustment yields a broader range of topics, still with varying degrees of coherence. For instance, Topic 03 in the 25-component model includes terms such as "return," and "broken," which differs from the terms observed in the 10-component model, but still indicates negative sentiment. However, many of the topics generated seem to combine terms more arbitrarily, such as topic 20 with the terms  “plastic,” “make,” “enough,” “lot,” and “hold.” The interpretability of these topics remains subjective, with some displaying clearer themes while others appear more ambiguous.

Both the 10-components and the 25-components models are tested with the classification algorithms. The training set for the 10-components LDA contains a total of 15 features, 10 representing the LDA generated topics and 5 representing the non-text features (verified, day, month, year, length). Similarly, the training set for the 25-components LDA contains the 5 non-text features and the 25 LDA generated topics. 

#### Bag of Words with TF-IDF

For TF-IDF vectorization, we use the ’TfidfVectorizer’ from Scikit-learn to transform the textual data into numerical vectors. We specify the stop_words parameter to filter out common stopwords and set the min_df and max_df arguments to exclude terms appearing in less than 0.1% or in more than 95% of the corpus. 

We obtain a sparse matrix, where each row corresponds to a text review and each column corresponds to a term from the vocabulary. In total, our vocabulary includes 758 terms. To integrate this information with our non-text features, we do not transform the sparse matrix into a dataframe due to the high dimensionality. Instead, we horizontally stack the TF-IDF vectors with the other features from the training set, resulting in a combined sparse matrix of shape (235392, 763). 

Furthermore, we are interested in investigating whether lemmatization could help reduce the dimensionality and possibly improve the models’ performance. The same TF-IDF process is thus replicated on the review text data after lemmatization. The resulting vocabulary is 708.

### Choice of Models and Evaluation Metrics

For each vectorization technique, two models are trained: a Ridge Regression and a random forest classifier. Several factors are considered in choosing our models.

Firstly, computational efficiency plays a crucial role in our decision, given the high dimensionality of our dataset resulting from TF-IDF vectorization. Ridge regression and Random Forest are both able to handle sparse data sets and large feature spaces, with the former offering the additional advantage of tackling overfitting and multicollinearity issues (Gupta et al., 2017). 

In terms of explainability, ridge regression also offers the benefit of transparency with explicit coefficients for each feature allowing for a straightforward interpretation of its impact on the prediction. Random forest, on the other hand, is often considered a black-box due to its ensemble nature, but still offers some insights into feature importance through metrics such as Gini importance or mean decrease in impurity (Lee, 2017). 

For both models, we choose to adhere to the default hyperparameters provided within the Scikit-learn library. The default settings often provide acceptable results without the need for extensive tuning and are chosen to provide a reasonable balance between bias and variance. This reduces the risk of overfitting, which is particularly important in our scenario given the high dimensionality of the feature space. For the ridge regression, a threshold of 0.5 is selected to convert probabilities into classifications. 

Finally, to evaluate and compare the models, a function is created to compute a series of performance metrics: accuracy, precision, recall, F1-score and the confusion matrix.

Our findings suggest that the use of TF-IDF vectorization generally provides the best results. TF-IDF with lemmatization shows a marginal decrease in performance which can potentially be attributed to contextual nuances that were lost. On the other end of the spectrum, FastText vectorization leads to a relatively poorer model performance across all metrics and both models.

For both TF-IDF and FastText, the Ridge regression and the Random Forest perform rather similarly, with a slight advantage for Random Forest. On the other hand, results for LDA show more variability depending on the number of components and the model chosen. When using LDA with 10 topics, the Random Forest model significantly outperforms the Ridge regression, which suggests that Random Forest might be better at leveraging the thematic structures uncovered by LDA. Increasing the number of topics to 25 improves the Ridge model’s performance, but has the opposite effect of the random forest model. 

Overall, the analysis illustrates that vectorization significantly influences the performance of classification models. The choice between different vectorization techniques and preprocessing steps (e.g., lemmatization vs. non-lemmatization) can lead to varied results in sentiment analysis. 


## Usage

To run the analysis, open the `NLP_BI_Airline_Industry.ipynb` notebook and execute each cell sequentially. Ensure that the required datasets are in the correct file paths.


## Dependencies

The following libraries are used in different parts of the project. Proceed to their installation with the following code:

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import html
import math
import re
import glob
import os
import sys
import json
import random
import pprint as pp
import textwrap
import sqlite3
import logging
from fractions import Fraction

import spacy
import nltk

import seaborn as sns
sns.set_style("darkgrid")

from tqdm.auto import tqdm
tqdm.pandas()

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay, f1_score, cohen_kappa_score, roc_auc_score, average_precision_score, precision_score, recall_score
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
```

## Installation
Ensure that you have Python installed on your system. If not, you can download it from [python.org](https://www.python.org/downloads/).


## Usage
You are allowed to view and fork the repository for personal use. If you have any questions or would like to discuss potential collaborations, feel free to reach out.


## Contributing
Although this project is not open-source, I welcome feedback, bug reports, and suggestions. If you encounter any issues or have ideas for improvements, feel free to send me an email to julian.enciso.izquierdo@gmail.com.


## License
This project is not open-source, and it does not come with a specific open-source license. All rights are reserved, and usage, modification, or distribution of the code is not permitted without explicit permission.

If you are interested in using or collaborating on this project, please send me an email to julian.enciso.izquierdo@gmail.com.

## Acknowledgments
Special thanks to Abir and Aya for their collaboration.


