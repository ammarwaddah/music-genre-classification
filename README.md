# Music Genre Classification
Using Data Science with Machine Learning to detect the genre of a music using significant features given by the most linked features that are taken into consideration when evaluating genre of music.

## Table of Contents
* [Introduction](#introduction)
* [Dataset General info](#dataset-general-info)
* [Evaluation](#evaluation)
* [Technologies](#technologies)
* [Setup](#setup)
* [Features](#features)
* [Run Example](#run-example)
* [Sources](#sources)

## Introduction
This project is a part of my training at SHAI FOR AI.\
During the stages of learning artificial intelligence, there is an important stage called classification, which indicates an important stage in distinguishing the output in a way that is close to human understanding and perception, especially when applied during practical stages that help people in their work. This is an introduction to the importance of the idea of multiple classification presented in the following project.\
Based on this introduction, I present to you my project in solving the problem of Music Genre Classification, and my suggestions for solving it with the best possible ways and the current capabilities using Machine Learning.\
Hoping to improve it gradually in the coming times.

## Dataset General info
**General info about the dataset:**

    artist: Name of the Artist.

    song: Name of the Track.

    popularity: The higher the value the more popular the song is.

    danceability: Danceability describes how suitable a track is for dancing based on a combination of musical elements including tempo, rhythm

    energy: Energy is a measure from 0.0 to 1.0 and represents a perceptual measure of intensity and activity.

    key: The key the track is in. Integers map to pitches using standard Pitch Class notation. E.g. 0 = C, 1 = C♯/D♭, 2 = D, and so on..

    loudness: The overall loudness of a track in decibels (dB). Loudness values are averaged across the entire track and are useful for comparing relative

    mode: Mode indicates the modality (major or minor) of a track, the type of scale from which its melodic content is derived. Major is represented by 1 and minor is 0.

    speechiness: Speechiness detects the presence of spoken words in a track. The more exclusively speech-like the recording (e.g. talk show, audio book, poetry), the closer to 1.0 the attribute value. Values above 0.66 describe tracks that are probably made entirely of spoken words. Values between 0.33 and 0.66 describe tracks that may contain both music and speech, either in sections or layered, including such cases as rap music. Values below 0.33 most likely represent music and other non-speech-like tracks.

    acousticness: A confidence measure from 0.0 to 1.0 of whether the track is acoustic. 1.0 represents high confidence the track is acoustic.

    instrumentalness: Predicts whether a track contains no vocals. "Ooh" and "aah" sounds are treated as instrumental in this context. Rap or spoken word tracks are clearly "vocal". The closer the instrumentalness value is to 1.0, the greater likelihood the track contains no vocal content. Values above 0.5 are intended to represent instrumental tracks, but confidence is higher as the value approaches 1.0.

    liveness: Detects the presence of an audience in the recording. Higher liveness values represent an increased probability that the track was performed live. A value above 0.8 provides strong likelihood that the track is live.

    valence: A measure from 0.0 to 1.0 describing the musical positiveness conveyed by a track. Tracks with high valence sound more positive (e.g. happy, cheerful, euphoric), while tracks with low valence sound more negative (e.g. sad, depressed, angry).

    tempo: The overall estimated tempo of a track in beats per minute (BPM). In musical terminology, tempo is the speed or pace of a given piece and derives directly from the average beat duration.

    duration in milliseconds:Time of the song

    time_signature: a notational convention used in Western musical notation to specify how many beats (pulses) are contained in each measure (bar), and which note value is equivalent to a beat.

    Class: Genre of the track.
    
## Evaluation
The F1 score can be interpreted as a harmonic mean of the precision and recall, where an F1 score reaches its best value at 1 and worst score at 0. The relative contribution of precision and recall to the F1 score are equal. The formula for the F1 score is:
                F1 = 2 * (precision * recall) / (precision + recall)
## Technologies
* Programming language: Python.
* Libraries: Numpy, Matplotlib, Pandas, Seaborn, tabulate, sklearn, plotly, scipy, imblearn, xgboost. 
* Application: Jupyter Notebook.

## Setup
To run this project setup the following libraries on your local machine using pip on the terminal after installing Python:\
'''\
pip install numpy\
pip install matplotlib\
pip install pandas\
pip install seaborn\
pip install tableau\
pip install scikit-learn\
pip install plotly\
pip install scipy\
pip install imblearn\
pip install xgboost\
'''\
To install these packages with conda run:\
'''\
conda install -c anaconda numpy\
conda install -c conda-forge matplotlib\
conda install -c anaconda pandas\
conda install -c anaconda seaborn\
conda install -c conda-forge tableauserverclient\
conda install -c anaconda scikit-learn\
conda install -c plotly plotly\
conda install -c anaconda scipy\
conda install -c conda-forge imbalanced-learn\
conda install -c anaconda py-xgboost\
'''

## Features
* I present to you my project solving the problem of Music Genre Classification using a lot of effective algorithm and techniques with a good analysis (EDA), and comparing between them using logical thinking, and put my suggestions for solving it in the best possible ways and the current capabilities using Machine Learning.

### To Do:
**Briefly about the process of the project work, here are (some) insights that I took care of it:**

* Dealing with missing data using regressor to estimate the best results, its estimation depend on the distributions, or remove it.
* Checking correlations between variables using corr() method, and Features Selection using Chi-2, f_classif, REF(recursive feature elimination), lasso regulation, logistic regression, and important features of Random Forest Classifier to select the best features with the target.
* Trying to Handle Imbalanced Data through up-sampling and downsampling techniques.
* Trying to improve the model accuracy by using PCA.
* Scaled features using min-max, standard-scaler, and Rubost depending on its distributions.
  
 *Used various classifying algorithms to aim best results:

* Decision Tree, Random Forest, KNN, SVC, Naive Bayes, XGBoost, SGD, Voting, Ada Boost, Gradient Boosting, Bagging, Logistic Regression, Stacking, Hist Gradient Boosting Classifiers.

 *Also making analysis using:
1. Confusion matrix analysis.
2. Features Selection.
3. Precision vs. Recall curves, and their best thresholds, also ROC plot for better understanding.

**Finally:**

* Using Halving Grid Search CV to tuning the best models parameters.

## Run Example
To run and show analysis, insights, correlation, and results between any set of features of the dataset, here is a simple example of it:

* Note: you have to use a jupyter notebook to open this code file.

1. Run the importing stage

2. Run the dataset

3. Select which cell you would like to run and show its output.

4. Run Selection/Line in Python Terminal command (Shift+Enter)

## Sources
This data was taken from SHAI competition (Music Genre Classification competition)
(https://www.kaggle.com/competitions/shai-training-level-1-b/)
