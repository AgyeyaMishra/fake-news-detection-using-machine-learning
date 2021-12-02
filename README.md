# Fake News Detection Using Machine Learning

This is a course project done for **Department of Computer Engineering, Delhi Technological University (formerly, Delhi College of Engineering)** for a course in **Machine Learning (Course Code - CO327)** for the **Fall Semester**.

---

# Introduction

- Fake news exist way before from social media but it was multifold when social media was introduced.
- Fake news is a piece of news designed to deliberately spread hoaxes, propaganda, and disinformation.
- Fake news stories usually spread through social media sites like Facebook, Twitter, etc.
- People are profiting by publishing fake news online.
- In general, the goal is profiting through clickbaits. 
- Such clickbaits lure users and entice curiosity with flashy headlines or designs to click links to increase advertisements revenues.

---

# Some Major Problems

Some of the major problems related to fake news that are generally faced by people are:

1. By clicking on a clickbait, users are led to a page that contains false information.
2. Fake news influences people’s perception.
3. The risk of fake news has become a global problem that even major tech companies like Facebook and google are struggling to solve. It can be difficult to determine whether a text is factual without additional context and human judgement.

---

# Purpose 

- This project aims to develop a method for detecting and classifying the news stories using Natural Language Processing.
- The main goal is to identify fake news, which is a classic text classification issue.
- We gathered our data, pre-processed the text, and translated our articles into supervised model features.
- Our goal is to develop a model that classifies a given news article as either fake or true.

---

# Delimitations

There are two delimitations associated with our project:
1. Our system does not guarantee 100% accuracy.
2. The system is unable to test data that is unrelated to the training database.

# Types of Fake News

There are basically two types of fake news:

1. Visual based type: Visual based are mainly photoshopped images, and videos which are posted on various social media platforms.
2. Linguistic based type: Linguistic based are mainly the manipulation of text and string content. These issues are with blogs, news, and emails.

---

# Natural Language Processing (NLP)

- There have been attempts made to use Artificial Intelligence techniques specifically machine learning/deep learning technique and natural language processing (NLP), to automate the process of detecting fake news and stopping it from spreading any further.
- It can be possible to teach a computer how to read and differentiate between real news and fake news using Natural Language Processing (NLP).
- The building blocks of our project are Dataset and Machine Learning.

# Abou the Dataset

1. It has the following attributes:
- id: unique id for a news article
- title: the title of a news article
- author: author of the news article
- text: the text of the article; could be incomplete
- label: a label that marks the article as potentially unreliable
2. The size of the dataset is 7796 x 4, which means it has 7796 rows and 4 columns.
3. The dataset is published on Kaggle for research and educational purposes.
4. We have 80% of data as training data and 20% data as testing data.
<p align = "center">
<img src="https://user-images.githubusercontent.com/53916781/144471315-ebaee0a2-a3b0-459a-a6a5-7d44893fdef7.png" />
</p>

---

# Workflow

The steps involved in the workflow of our project are:

1. First, we import the necessary libraries required for our project.
2. Then, we try to read the data into a DataFrame and get the shape of the data and the first 5 records.
3. After this, we get the labels from the DataFrame.
4. Now, we split our dataset into training and testing sets. The training set would contain the data which will be fed into the model. The testing set will contain the data on which we test the trained and validated model.
5. Now, we initialize a TfidfVectorizer with stop words from the English language and a maximum document frequency of 0.7 (terms with a higher document frequency will be discarded).
6. Next, we’ll initialize a PassiveAggressiveClassifier and try to calculate the accuracy.
7. Finally, we print out a confusion matrix to gain insight into the number of false and true negatives and positives.

---

# Architecture

<p align = "center"><img src = "https://user-images.githubusercontent.com/53916781/144471657-8324b082-2a03-4465-9fb9-9d8ef870aa67.png" /> </p>  
<p align = "center">Fig: ML Model Architecture</p>

# Techniques

- We used TF-IDF for feature extraction.
- We trained our data by using PA algorithm.
- We tested the efficiency of the classifier using accuracy.

<p align = "center"> <img src = "https://user-images.githubusercontent.com/53916781/144473394-ef8a1750-711c-438a-b92f-bba0d1ba36ca.png" /> </p>

---

# TF-IDF
