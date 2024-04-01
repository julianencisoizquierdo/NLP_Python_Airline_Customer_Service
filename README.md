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
For Case 2, we are expected to review the issue of an imbalanced dataset. We are provided with the eclipse_jdt.csv.

### Step 1: Recode the outcome into a binary variables

After importing the dataset, we start by pulling its first few rows and information, so as to get an idea of the data we are working with. The Duplicated_issue column has many missing values and will not be useful later on, so we drop this column.

We will use the priority level as the outcome variable, however it currently has multiple possible values. We change it into 2 categories: P3, or else. To do so, we use a lambda function on the Priority column which keeps “P3” as a value if it is already “P3”, and for any other values changes it to “else”. 

We check whether this worked properly by counting the values of each possible priority level, and we see then that the dataset is highly imbalanced with 88% of the observations having a P3 priority.

Next, we concatenate the Title and Description columns into a single Text column, and drop the Title and Description columns. This way, we will only have one column with all text to be analyzed. Then, we clean this new column, using the Clean function from the preparation module, and only keep clean text with more than 20 characters.

Finally, we split our dataset into test and training samples. We use 80% of the data as a training set and hold out the remaining 20% into a test set. We use the stratify argument to keep the original class distribution (88% of P3) in the training and test sets. We see that we have over 36k observations in the training set and over 9k observations in the test set.

### Step 2: Vectorization

We decide to use TF-IDF vectorization on the training sample, by using the TfidfVectorizer. This is because our dataset is composed of a Text for each observation, and we have thousands of observations. We thus want to consider not only how important a certain term is in the observed document (Text of that observation), but also we want to account for its frequency in the entire corpus (the set of observations).

We use the vectorizer and not the transformer, as it combines in a single operation both the text tokenization and the TF-IDF scores. This way, it streamlines the pipeline.
As settings for the TfidfVectorizer, we only keep tokens that have a frequency of more than 10, we include bigrams and we use the usual english stopwords.

### Step 3: Build reference results

As we will assess our models against the held out test set, we first also vectorize java_X_test.
We build a trivial classifier, which will always predict the majority class (P3), that we later use as a baseline to assess the other models. To do so, we use the DummyClassifier and apply the “most frequent” strategy. We then use this trivial classifier to predict the baseline Y, by applying our model to the vectorized test set.

We then choose five other machine learning classifiers to train and evaluate against the trivial classifier:
- K-Nearest Neighbor (KNN),
- Support Vector Machine (SVM),
- Logistic Regression,
- Random Forest,
- Gradient Boosting Classifier.

We select a KNN model, as it is a simple non-parametric classification technique (Putatunda, 2019), and it is regarded as among the best algorithms in data mining (Mucherino et al., 2009; Steinbach & Tan, 2009; Zhang, 2022). However, this algorithm does not explicitly learn classification, but rather finds the k most similar examples in the training set based on the defined measure (Pereira et al., 2009), in our case k = 5 and the metric is the Euclidean distance.

We choose SVM as it is a popular model in classification tasks for its generalization performance (Yue et al., 2003). It has been compared to other classification models and shows similar performance overall (Colas & Brazdil, 2006). It works by separating the data points in classes, through a decision boundary (Pereira et al., 2009).

Another linear classifier that we choose is the Logistic Regression, which is an algorithm that can be used in the case of binary classification and has shown better performance than other models in some datasets (Hassan et al., 2022).

Next, we run a Random Forest algorithm, as its accuracy on binary classification has been adduced by many (Buckinx & Van den Poel, 2005; Lunetta et al., 2004; Schwender et al., 2004). It is an ensemble method designed to remedy the issue of robustness in Decision Trees.

Finally, we execute a Gradient Boosting Classifier, an ensemble method that can be used with small amounts of data (Islam Khan et al., 2022). It has also been compared to the previously mentioned algorithm in certain studies and shown better results (Chakrabarty et al., 2019; Islam Khan et al., 2022).

For all these algorithms, we first instantiate the model, then fit them on our training set and before using them to do predictions on our test set.

Once we have the predictions from all the models, we run the required metrics to assess the models against each other. Please find a summary of the performance of the selected models per metric in the following table. In green is the best and in red the worst scorer per metric.

The Gradient Boosting Classifier shows the best performance for all the metrics, so is the best model in this case. The Trivial Classifier has the worst score on all metrics but the F1-score, where the K-Nearest Neighbor is 0.0001 below, we still conclude that all the models perform better than a dummy model.

### Step 4: Undersampling the majority class

As one solution to address the problem of the imbalanced classes, we undersample the majority class, so it has the same number of observations as the minority class.

To do so, we first create a majority and a minority dataframes, that hold only observations with “P3” and “else” priorities respectively. This allows us to treat the majority and minority classes differently later on. We then use the resample function from the Scikit-Learn package to perform sampling without replacement of the majority class. The objective is to create a sample of the majority class that is the same size as the minority class, randomly. Once we have this sample of the majority class called undersampled_majority, we concatenate it with the entire minority class observations, to obtain a new dataset, undersampled_java, which has 50% of “P3” observations and 50% of “else” observations.

Just like previously, we split this undersampled_java set into 80% training and 20% testing sets. We use the stratify argument on the “Priority” column to ensure that this equal distribution of classes is kept in both the train and test sets. We then vectorize these train and test sets, before fitting our Trivial Classifier and 5 other models again on the undersampled dataset. Finally, we predict the Y with all the models and assess them again on the chosen metrics.

While the Gradient Boosting Classifier was the best model in the original set, it does not perform as well as the Support Vector Machine when undersampling the majority class. Indeed the SVM is now the best performing model, having achieved the best score in all the chosen metrics. The Trivial Classifier still performs the worst on all metrics but the F1-Score, where the KNN and Random Forest have a lower score. Like in the previous step, we conclude that all models perform better than the dummy model.

### Step 5: Oversampling the minority class

Next, we explore another solution to the class imbalance problem, oversampling the minority class. In this step, we will duplicate the observations in the minority class to obtain the same size of both classes.

To determine how much we need to duplicate the minority class, we calculate the ratio of majority to minority observations using an integer division. We then create a duplicated_minority dataframe by concatenating minority to itself as many times as the ratio we previously calculated. We then concatenate this duplicated_minority dataframe to the majority dataframe to get our oversampled_java.

Afterwards, we split our oversampled_java into 80% train set and 20% test set, again stratifying to ensure the even class distribution is kept. These two sets are vectorized and we use our vectorized oversampled train set to fit our models. Finally, we use the fitted models to predict Y and compute the assessment metrics.

With oversampling the minority class, the best performing model, with the top score in all metrics, is the Random Forest, followed by the SVM. Like previously, the Trivial Classifier continues to perform the worst, indicating that all models perform better than a dummy model.


### Step 6: SMOTE

SMOTE refers to Synthetic Minority Oversampling Technique, and it is another technique to oversample the minority class. Instead of simply duplicating the minority data which can lead to overfitting, SMOTE creates new data points synthetically using a nearest neighbor approach. This way we can obtain a balanced dataset, without having repeated observations.

To perform SMOTE, we use the SMOTE from the IMB Learn package. For SMOTE, the outcome class must be numeric, hence we convert ‘P3’ instances to 1 and ‘else’ to 0, and ensure they are of type integer. We then define a X pandas series as the Text column only, a Y pandas series as the Priority column only, and vectorize X.

After these preprocessing steps, we instantiate SMOTE and apply its fit_resample method on the vectorized X and Y. This outcome is then split into our train and test datasets, still with a 20% held out test set and stratification to ensure the newly achieved class balance is maintained.

From there on, we train and assess our Trivial Classifier and each of our five models again.

However, in this step we encountered failures with the KNN model. We failed to train the KNN model and predict the test set. It took over 3 hours to execute, before failing each time. We attempted to fit the model with just 30% of the training data, but it also failed. We can thus conclude that the K-Nearest Neighbor is not adapted to this type of dataset, as it is too computationally intensive.

### Step 8: Conclusion

Conducting these different experiences highlight the ineffectiveness of using a dummy classifier, which always predicts the majority class, and indicates the need to employ more sophisticated prediction models. 

Before implementing any data adjustments to address class imbalance, the Gradient Boosting Classifier emerges as the top performer across all evaluated metrics. This highlights its inherent robustness and adaptability in handling diverse data distributions. However, when employing undersampling techniques, the Support Vector Machine (SVM) proves to have higher performance. This can be attributed to SVM's effectiveness to establish decision boundaries in reduced feature spaces, which suits the altered data distribution resulting from undersampling. 
The oversampling technique, along with SMOTE (a type of over sampling), both demonstrate Random Forest as the top performer.

The consistent performance of Random Forest as the top performer for both oversampling techniques, suggests that Random Forest is robust and adaptable to different data balancing strategies. This consistency may stem from Random Forest's inherent ability to handle complex datasets and mitigate overfitting, making it resilient to variations in the data distribution caused by oversampling. Additionally, Random Forest's ensemble nature, which combines multiple decision trees, allows it to capture diverse patterns and relationships within the data, enhancing its effectiveness in both oversampled scenarios.

In implementing the various techniques to handle the class imbalance, it is evident that overall, the oversampling technique results in much higher performance compared to other methods. SMOTE, a more advanced form of oversampling, closely follows Over Sampling in terms of performance, aligning with the findings of Juez-Gil et al. (2021), which emphasized the effectiveness of simpler methods, such as Random-Over-Sampling in providing better results than complex ones, such as SMOTE.

These results emphasize the importance of exploring different strategies for managing imbalanced data and emphasize the advantages of ensemble methods like Gradient Boosting and Random Forest, along with algorithm-level approaches such as adjusting misclassification costs and modifying decision thresholds, in addressing the challenges posed by imbalanced datasets.


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
Special thanks to Jade, Julie, Abir and Aya for their collaboration.


