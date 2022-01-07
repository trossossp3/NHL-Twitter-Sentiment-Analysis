# utilities
import re
import string
import numpy as np
from numpy.lib.function_base import vectorize
import pandas as pd
# plotting
import seaborn as sns
from wordcloud import WordCloud
import matplotlib.pyplot as plt
# nltk
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
# sklearn
from sklearn.metrics import roc_curve, auc
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics import confusion_matrix, classification_report
import pickle

DATASET_COLUMNS = ['target', 'ids', 'date', 'flag', 'user', 'text']
DATASET_ENCODING = 'ISO-8859-1'

df = pd.read_csv('training.csv', encoding=DATASET_ENCODING, names=DATASET_COLUMNS)
# print(df.sample(5)) #displays 5 random entries from the training dataset
# df.head() #

data = df[['text', 'target']]
data['target'] = data['target'].replace(4,1) #change positive values (which were 4) to now be 1 so a negative sentiment has target 0 and positive sentiment has target 1

data_pos = data[data['target'] == 1]
data_neg = data[data['target'] == 0]

data_pos = data_pos.iloc[:int(20000)]
data_neg = data_neg.iloc[:int(20000)]

dataset = pd.concat([data_pos, data_neg])

dataset['text'] = dataset['text'].str.lower() #sets all text to lowecase

# print(dataset['text'].tail()) # prints last 5 entries


def cleaning_stopwords(text):
    return " ".join([word for word in str(text).split() if word not in ENGLISH_STOP_WORDS]) #removes the stop words
dataset['text'] = dataset['text'].apply(lambda text: cleaning_stopwords(text)) #.apply appliues a functiona long an axis

punctuations = string.punctuation

def cleaning_punctionations(text):
    translator = str.maketrans('','', punctuations) #makes translation table to get rid of the puctioations from text
    return text.translate(translator)
dataset['text'] = dataset['text'].apply(lambda word: cleaning_punctionations(word))


def cleaning_repeating_char(text):
    return re.sub(r'(.)1+', r'1', text)
dataset['text'] = dataset['text'].apply(lambda x: cleaning_repeating_char(x))
# print(dataset['text'].head())

tokenizer = RegexpTokenizer(r"[\w']+") #create a tokenizer to split at spaces essentialy
dataset['text'] = dataset['text'].apply(tokenizer.tokenize)
# print(dataset['text'].head())

st = nltk.PorterStemmer()
def stemming_on_text(data):
    text = [st.stem(word) for word in data]
    return text
dataset['text']= dataset['text'].apply(lambda x: stemming_on_text(x))




lm = nltk.WordNetLemmatizer()
def lemmatizer_on_text(data):
    text = [lm.lemmatize(word) for word in data]
    return text
dataset['text'] = dataset['text'].apply(lambda x: lemmatizer_on_text(x))

x = data.text
y = data.target


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.05, random_state =26105111) #splits into train and test sets



vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=500000)
vectorizer.fit(x_train)


x_train = vectorizer.transform(x_train)
x_test = vectorizer.transform(x_test)


def model_Evaluate(model):
    # Predict values for Test dataset
    y_pred = model.predict(x_test)
    # Print the evaluation metrics for the dataset.
    print(classification_report(y_test, y_pred))
    # Compute and plot the Confusion matrix
    cf_matrix = confusion_matrix(y_test, y_pred)
    categories = ['Negative','Positive']
    group_names = ['True Neg','False Pos', 'False Neg','True Pos']
    group_percentages = ['{0:.2%}'.format(value) for value in cf_matrix.flatten() / np.sum(cf_matrix)]
    labels = [f'{v1}n{v2}' for v1, v2 in zip(group_names,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    sns.heatmap(cf_matrix, annot = labels, cmap = 'Blues',fmt = '',
    xticklabels = categories, yticklabels = categories)
    plt.xlabel("Predicted values", fontdict = {'size':14}, labelpad = 10)
    plt.ylabel("Actual values" , fontdict = {'size':14}, labelpad = 10)
    plt.title ("Confusion Matrix", fontdict = {'size':18}, pad = 20)


LRmodel = LogisticRegression(C = 2, max_iter = 1000, n_jobs=-1)
LRmodel.fit(x_train, y_train)


model_Evaluate(LRmodel)
y_pred = LRmodel.predict(x_test)
# predTest = LRmodel.predict(testvVector)
# print(predTest)
out_path = "data/"
pickle.dump(LRmodel, open(out_path + "model.sav",'wb'))
pickle.dump(vectorizer, open(out_path+ "vectorizer.pickle",'wb'))
# fpr, tpr, thresholds = roc_curve(y_test, y_pred)
# roc_auc = auc(fpr, tpr)

# vec = pickle.load(open('vectorizer.pickle'))
# vec.transform("hello")