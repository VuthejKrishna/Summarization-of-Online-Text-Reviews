# Summarization of Online Text Reviews

2020 Version
Release Date: March 18, 2020

----------------------------------------------------------------------------------------------------------------------

# README CONTENTS

1.1 Introduction

1.2 Description

1.3 Code Explanation

1.4 Challenges Faced

1.5 Conclusion/Future Work

2.0 DSCI_521_Term_Project_Phase1.ipynb

2.1 DSCI_521_Term_Project_Phase2.ipynb

2.2 JSON Files

----------------------------------------------------------------------------------------------------------------------

# 1.1 Introduction

Online reviews are seemingly growing by the increased e-commerce users.
Many users rely on these online reviews in building up their opinion on the products that are being purchased.
More time needs to be invested by the user inorder to look through these giant set of reviews and to get a jist out of them.
Our attempt is to reduce this time of the user by providing an automated summary of all the reviews present on the product, 
thus making it easier for the user to arrive at a conclusion by just looking at one single review that our model generates.

Not just the users, even the companies that sell the products online can benifit from this application by saving huge amount of time.
They can understand the trend amongst the users and can exactly locate the pros & cons provided by the users.

----------------------------------------------------------------------------------------------------------------------

# 1.2 Description

We are using the data sets that we've build under DSCI-511 Term project by performing web-scrapping on www.91mobiles.com.
This website displays the user reviews from e-commerce giants Flipkart and Amazon on mobile phones and other electronic devices.
We have limited our scope to mobile phones and have built two JSON files one for each Flipkart and Amazon reviews respectively.

Building up on these data sets, in the current project under DSCI-521, we are trying to generate a summary of all the reviews without
missing out on the curucial information that the user provides. This will help saving the time from reading through all the reviews.

To achieve this, we are using the unsupervised LDA (Latent Dirichlet Allocation) model.
Below is the step by step process that we have followed.
	- Cleaning the data to remove special characters and emoticons
	- Feature Selection
	- Tokenization of the reviews
	- Removal of stop-words and contractions
	- Performing Stemming and Lemmatization
	- Applying bag of words model for vectorization
	- Applying TF-IDF for vectorization
	- Above generated vectorized data can then be fed to the LDA model to generate the specified number of topics
	- NLTK Sentiment Vader helps assessing the polarity of the reviews
	- Bi-grams and Tri-grams are then being applied to the output from LDA model
	- Thus generating a review with the most important topics on both the polarities (Positive and Negative)

Upon our initial EDA, we've understood that all the reviews needs to be drilled a step further inorder to understand which feature a user is specificall
talking about. This helped us in creating the summary more accurately by including all the postives and negatives that are frequently discussed in the reviews.
Final summary that has been generated contains both highly positive and highly negative contents that users have posted.

----------------------------------------------------------------------------------------------------------------------
# 1.3 Code Explanation

Section - 1
Code to import all the necessary modules and packages necessary.

Section - 2
Code to load '521_amazon.json' data which has amazon reviews for the selected 11 mobile phones.

Section - 3
Code to load the '521_amazon.json' data which has amazon reviews for the selected 11 mobile phones.

Section - 4
Code to make below modifications to the input data..
	'reviews' -- To convert all cases to lower and to strip spaces towards the ends.
	'useful' -- To convert string data to integer i.e. '7 users found this review helpful' to '7'
	'rating' -- To convert string data to integer i.e.  '1 stars' to '1'
	'date' -- To convert string data to datetime object i.e. '02 Jan 2019,' to datetime.date(2019, 1, 2)

Section - 5
Code to convert short notations in the English language to an elaborated one.

Section - 6
This code helps in identifying the stopwords which can be further used while converting word to vector.
However, there will be certain stopwords in the reviews which we need to retain in order to correctly assess the sentiment.
Hence we have created a list of words to pull out from the stopwords.

Section - 7
Function 'Lemmatize_stemming' helps in identifying the root form of all the words in the reviews.
Function 'preprocess' helps removing the identified stop words from the reviews. It also filter out the words with length less than 2.

Section - 8
This code utilizes the contractions list along with the above specified functions to lemmatizze and perform pre-processing.
In addition to pre-process function, we are filtering out below aspects..
    -> All the special characters
    -> All the single characters
    -> Replace multiple spaces with single space

Note : Though Sentiment Vader provides sentiment for emojis and slang words, we felt they might dilute the overall sentiment if in case
the reivew seems to be sarcastic. Based on this assumption, we have removed emojis but retained slang words.

Section - 9
This code creates a dataframe with modified values from ['review', 'useful', 'rating', 'date']. Additionnally adds a column for index.
'Processed_review' column further maps all the rootwords from the reviews.
Addition of the sentiment ('positive', 'negative', 'neutral' and 'compound') obtained from sentiment vader.

Section - 10
In order to have the plots correctly display the reviews in chronological order, we have sorted them based on the entry date of reviews.

Section - 11
Code to display the scatter plots for 'iPhone 11' and 'iPhone 11 Max' for date vs sentiment.

Section - 12
Code to display the variation in reviews received during the given duration for a sample phone. 'Apple iPhone 11'.

Section - 13
Code to display the variation in ratings received during the given duration for a sample phone. 'Apple iPhone 11'.

Section - 14
Code to display the count of people that actually found a review as useful during the given duration for a sample phone. 'Apple iPhone 11'.

Section - 15
Code to display the distribution of review ratings for a sample phone. 'Apple iPhone 11'

Section - 16
In order for us to better understand the count of reviews received per phone we have plotted below barplot.

Section - 17
Bar plot by applying logarithm to the number of reviews i.e. y-axis

Section - 18
From this section onwards, our main focus will be on the review text to generate the sentiment on Topics which will be created by LDA.

Section - 19
Code to save the cleaned reviews of each phone into a dictionary

Section - 20
We've identified common features and then divided the reviews based on features for every phone.

Section - 21
This function initially splits the words in reviews and compares itself with the words in paramaetes/features listed above.
Seggregation of reviews is then done based on the features.

In order to avoid the mixing of features which might be the case when we take an entire review, as a reviewer might discuss about 
various features in a single review. So by separating the review into sentences, we can capture insights for each feature more precisely.

Section - 22
To create individual dataframes for every listed feature in every phone.

Section - 23
For the vectorizer to function, we are providing the root words in verb form from 'preprocess' function which is defined at the beginning.

Section - 24
For efficiency purpose, rather than passing the entire data to build the corpus, we have individually built the dictionaries for
each feature in every phone.

Section - 25
Code to build corpus for each review based on dictionary that has been defined for that specific feature from above section.
Bag of Words Model is being used by doc2bow.

Section - 26
TF-IDF Model is being applied on the above created bow_corpus.

Section - 27
Applying Latent Dirichlet Allocation (LDA) model to the bag of words corpus (bow_corpus).
We have now divided entire reviews from each feature to a cluster of 3. 
For reviews falling in 'others' category, number of clusters will be 10.

Section - 28
Simple display of the topics that have been generated by LDA model as done above.

Section - 29
Applying Latent Dirichlet Allocation (LDA) model to the TF-IDF corpus (corpus_tfidf).
We have now divided entire reviews from each feature to a cluster of 3. 
For reviews falling in 'others' category, number of clusters will be 10.

Section - 30
Similarity Measurement for each sentence towards the 3 clusters from LDA Bag_of_words model.

Section - 31
Similarity Measurement for each sentence towards the 3 clusters from LDA TF-IDF model.

Section - 32
By using Sentiment Vader, we are generating polarities for each sentence.

Section - 33
Code to display the distribution of compound score for camera feature in sample phone (Google Pixel 3 XL) across 3 topics.

Section - 34
Assigning the aggregated polarities (positive, neutral, negative, negative_compound, positive_compound) for each feature in every phone to a dictionary.

Section - 35
Curating the input that needs to be provided to generate Bigrams and Trigrams.
This code produces one single string from the set of rootwords obtained in the begining for each statement in the review.

Section - 36
Generating Bigrams would help us in understanding the topics provided by LDA more clearly.

Section - 37
Generating Trigrams would help us in understanding the topics provided by LDA more clearly.

Section - 38
Giving LDA model score to each sentence

Section - 39
Finding an average score for a topic

Section - 40
Finally, generating a short summary

Section - 41
Finding short summaries for each topics.

Section - 42
Finding short summary for all the sentences with negative compound score.

Section - 43
Finding short summary for all the sentences with positive compound score.

Section - 44
Final Ouput function to display summarized topics, bigrams, trigrams and then the final statement.
This function also includes a grpah depicting the topics vs compound score for a specific feature from phone as input.

Section - 45
User Provided Input for Phone and Feature Selection.


Our model has been built in a way that it just requires 'Phone Selction' along with the 'Feature' that the user want to know about and
complete reviews will be processed to produce relavant output.

----------------------------------------------------------------------------------------------------------------------

# 1.4 Challenges Faced and Overcomed


LDA model provided the output in terms of topics with words which has certain score.
But these were just single words and doesn't necessarily congratulate or
negate a particular feature of the phone as it's just a single word.

To overcome this situation, we have come across the idea of utilizing the Bi-grams and Tri-grams
to produce two and three word pairs for positive, negative and neutral polarities.
These pairs helped us in better understanding the feature that are actually positive, neutral and negative.
It also provided a compound score.

There is a chance that our input data might have been diluted by the Opinion Spam and if these reviews go unidentified,
then the final sentiment about a particular feature might change in an incorrect direction.

It has been assumed that all the reviews would be entirely in English language but that wasn't the case.
We could also observe slang words and emoticons in the reviews.
These slang words and emoticons are difficult to understand mainly because they can also be treated as a sarcastic comment.
We have currently filtered out all the special characters thus eliminating the complexity of involving emoticons.

Slang words in the reviews could be better evaluated had there been a dedicated embedding model that supported them.

----------------------------------------------------------------------------------------------------------------------

# 1.5 Conclusion/Future Work

We've used the unsupervised LDA as this fits the data we have. 
Futher enhancement can be done with either Labeled-LDA or Multi-Grain LDA.
But the limitations with the data set that we have is blocking us from implementing them. 
Reason being that the Labeled-LDA and Multi-Grain LDA needs the labels from input data and current data sets we have lack this.

By improving the input data sets we have, i.e. by including the labels which Labeled-LDA and Multi-Grain LDA utilises,
enhancements can be done to the model.

We can further incorporate the emoticons and slang words while building the final summary.

We are able to build the final summary and this can be further smoothened to include punctuations to accurately frame the sentences.

----------------------------------------------------------------------------------------------------------------------

# 2.0 DSCI_521_Term_Project_Phase1.ipynb

This is the jupyter notebook with information relative to project scoping sumitted in phase-1.

----------------------------------------------------------------------------------------------------------------------

# 2.1 DSCI_521_Term_Project_Phase2.ipynb

This is the jupyter notebook where we have all our code along with the description of what the code in each cell does.

----------------------------------------------------------------------------------------------------------------------

# 2.2 JSON Files

'521_flipkart.json' and '521_amazon.json' files provide the dictionary structure of the data that is being used as input.
These files consists of multi-layered dictionaries.


<end of file>
