
# coding: utf-8

# # Sentiment Analysis Of Reviews

# In[4]:


# Loading the dataset
import pandas as pd


# In[5]:


df_yelp = pd.read_table("yelp.txt")
df_amazon = pd.read_table("amazon.txt")
df_imdb = pd.read_table("imdb.txt")


# In[6]:


frames = [df_yelp, df_amazon, df_imdb ]


# In[7]:


for col_name in frames:
    col_name.columns = ["Message", "Target"]


# In[8]:


for col_name in frames:
    print(col_name.columns)


# In[9]:


ind_key = ["YELP","AMAZON","IMDB"]


# In[10]:


df = pd.concat(frames, keys=ind_key)


# In[11]:


df.shape


# In[12]:


df.head()


# In[13]:


df.tail()


# In[14]:


df.isnull().sum()


# # Spacy Operations 

# In[15]:


import spacy
nlp = spacy.load('en')


# In[16]:


from spacy.lang.en import STOP_WORDS
stopwords = list(STOP_WORDS)


# In[17]:


import string
punctuations = string.punctuation


# In[18]:


from spacy.lang.en import English
parser = English()


# In[19]:


def spacy_tokenizer(sentence):
    my_token= parser(sentence)
    my_token = [word.lemma_.lower().strip() if word.lemma_ != "PRON" else word.lower_ for word in my_token ]
    my_token = [word for word in my_token if word not in stopwords and word not in punctuations ]
    return my_token


# #  Using ML

# In[20]:


# ML Packages
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.metrics import accuracy_score 
from sklearn.base import TransformerMixin 
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


# In[21]:


# Custom transformer using spacy
class predictors(TransformerMixin):
    def transform(self, X, **transform_params):
        return [clean_text(text) for text in X ]
    def fit(self, X, y=None , **fit_params):
        return self
    def get_params(self, deep= True):
        return{}
    
# Function for text cleaning: - 
def clean_text(text):
    return text.strip().lower()


# In[22]:


# Vectorization
vectorizer = CountVectorizer(tokenizer = spacy_tokenizer, ngram_range=(1,1))
classifier = LinearSVC()


# In[23]:


#Using Tfdif
tfvectorizer = TfidfVectorizer(tokenizer = spacy_tokenizer)


# In[24]:


from sklearn.model_selection import train_test_split


# In[25]:


X=df["Message"]
ylabels= df["Target"]


# In[26]:


X_train, X_test, y_train , y_test = train_test_split(X,ylabels, test_size=0.2, random_state = 42)


# In[27]:


#Create a pipeline to clean
pipe = Pipeline([("cleaner",predictors()),
                ("vectorizer", vectorizer),
                ("classifier", classifier)])


# In[28]:


pipe.fit(X_train, y_train)


# In[29]:


# Predicting with a dataset
sample_prediction = pipe.predict(X_test)


# In[30]:


# Prediction Results
# 1 = positive review
# 0 = negative review
for (sample,pred) in zip(X_test,sample_prediction):
    print(sample,"Prediction=>",pred)


# In[41]:


# Accuracy
print("Accuracy: ",pipe.score(X_test, y_test))


# # Fetch The Reviews

# In[32]:


import urllib.request
import urllib.parse
import urllib.error
from bs4 import BeautifulSoup


# In[33]:


#Creating an html parser
url=("https://www.amazon.com/All-New-Amazon-Echo-Dot-Add-Alexa-To-Any-Room/product-reviews/B01DFKC2SO/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews")
html = urllib.request.urlopen(url).read()
soup = BeautifulSoup(html, 'html.parser')
html = soup.prettify('utf-8')


# In[34]:


#fetching short reviews from amazon for echo-dot-2 speaker
product_review = []
for tags in soup.findAll('a',
                           attrs={'class': 'a-size-base a-link-normal review-title a-color-base a-text-bold'
                           }):
    short_review = tags.text.strip()
    product_review.append(short_review)


# In[35]:


product_review


# In[42]:


#ex =["This movie  could have been better"]


# In[43]:


#pipe.predict(ex)


# # Do The Analysis

# In[36]:


pipe.predict(product_review)


# In[37]:


#Storing predicted result in a dictionary
sentiment = {"Positive":0, "Negative":0}
def get_sentiment():
    for i in pipe.predict(product_review):
            if i == 1:
                sentiment["Positive"]= sentiment["Positive"]+1 
            else:
                sentiment["Negative"]= sentiment["Negative"]+1
    print(sentiment)


# In[38]:


get_sentiment()


# In[40]:


from matplotlib import pyplot as plt
slices = [sentiment["Positive"],sentiment["Negative"]]
activities = ['Positive','Negative']
cols = ['g','r']

plt.pie(slices,
        labels=activities,
        colors=cols,
        shadow= True,
        autopct='%1.1f%%')

plt.title('Sentiment Analysis of Reviews')
plt.legend()
plt.show()

