---
layout: post
title:  "Sentiment Analysis for Social Media, from Zero to BERT"
date:   2020-07-26 14:00:00 +0800
categories: nlp
---

When I first researched about sentiment analysis, it seemed that most of the resources/artices on the subject were about academic and clean datasets. For instance there are hundreds of tutorials out there about how to classify movie reviews from the super-popular IMDB dataset. Yet information about real world sentiment analysis, like how to deal with messy social media messages, is hard to find.

I assume this is because people tend to gravitate towards commonly-used datasets, but also because dealing with social media messages is a tough problem that has not been fully cracked yet. Sarcasm, humour, subjectivity, smileys... there are many challenges along the way.

The good news is that there has been major improvements of the standard techniques in the NLP toolbox during the past few years, notably with the introduction of large pre-trained models based on the Transformer architecture like BERT. These new models enable much richer and comprehensive representations of messages which can be used to classify sentiment more accurately.

I wrote this post to share some of what I learned in my journey building [BrandImage.io](https://brandimage.io), an online tool to better understand what people like/dislike about a given company and its products. I hope you'll find it useful as a starting point to learn how to build robust pipelines to deal with social media messages.

The goal here is not to beat the state-of-the-art accuracy but to offer a **practical guide** to the problem, so we will focus on straightforward and scalable solutions. Most attention will be given to different ways of pre-processing and representing the messages for classification, which in my experience is where 80% of the performance improvement happens! We will not build a large/deep model for the classification as this would make deployment harder. However we will look at BERT embeddings towards the end if you are interested in using more advanced representations.

You can also find this guide on [Github](https://github.com/jonathanbgn/social-media-sentiment-analysis/blob/master/Practical_Introduction_BERT.ipynb) or [Google Colab](https://colab.research.google.com/github/jonathanbgn/social-media-sentiment-analysis/blob/master/Practical_Introduction_BERT.ipynb) to experiment with the code yourself.

## Data Preprocessing

To illustrate the problem, we will use tweets from the **SemEval-2017** competition, where teams compete in various Twitter classification challenges. We will use the combined data of all the previous years for the Task 4-A, which you can [download here](http://alt.qcri.org/semeval2017/task4/index.php?id=data-and-tools). For more details about the task and winning teams, you can also have a look at the [official SemEval-2017 Task 4 paper](https://arxiv.org/pdf/1912.00741.pdf).

The objective will be to classify tweets in 3 buckets: negative, positive or neutral.


```python
import os, re, html, csv
import numpy as np
import pandas as pd

data_dir = 'Subtask_A/'
train_files = [
    'twitter-2013train-A.tsv',
    'twitter-2013dev-A.tsv',
    'twitter-2013test-A.tsv',
    'twitter-2014sarcasm-A.tsv',
    'twitter-2014test-A.tsv',
    'twitter-2015train-A.tsv',
    'twitter-2015test-A.tsv',
    'twitter-2016train-A.tsv',
    'twitter-2016dev-A.tsv',
    'twitter-2016devtest-A.tsv',
    'twitter-2016test-A.tsv',
]

def load_dataframe(file_path):
    return pd.read_csv(
        file_path,
        sep='\t',
        quoting=csv.QUOTE_NONE,
        usecols=[0,1,2],
        names=['id', 'label', 'message'],
        index_col=0,
        dtype={'label': 'category'})

train_dfs = []
for f in train_files:
    train_dfs.append(load_dataframe(os.path.join(data_dir, 'downloaded/', f)))
tweets_train = pd.concat(train_dfs)
tweets_train.drop_duplicates(inplace=True)
tweets_train = tweets_train.sample(frac=1.0, random_state=42)

# Clean and prepare messages:
def preprocess_messages(messages):
    messages = messages.str.decode('unicode_escape', errors='ignore')
    messages = messages.apply(html.unescape)
    messages = messages.str.strip('"')  # remove left-most and right-most "
    messages = messages.str.replace('""', '"', regex=False)
    return messages
tweets_train['message'] = preprocess_messages(tweets_train['message'])

tweets_train_y = tweets_train['label'].cat.codes
labels = tweets_train.label.cat.categories.tolist()
labels_codes = {}
for i, label in enumerate(labels):
    labels_codes[label] = i

print('Total number of examples for training: {}\nDistribution of classes:\n{}'.format(
    len(tweets_train),
    tweets_train['label'].value_counts() / len(tweets_train),
))

tweets_train.head()
```

    Total number of examples for training: 49675
    Distribution of classes:
    neutral     0.448032
    positive    0.395994
    negative    0.155974
    Name: label, dtype: float64





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>label</th>
      <th>message</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>640329403277438976</th>
      <td>neutral</td>
      <td>[ARIRANG] SIMPLY KPOP - Kim Hyung Jun - Cross ...</td>
    </tr>
    <tr>
      <th>640810454730833920</th>
      <td>neutral</td>
      <td>@TyTomlinson just read a politico article abou...</td>
    </tr>
    <tr>
      <th>111344128507392000</th>
      <td>neutral</td>
      <td>I just typed in "the Bazura Project" into goog...</td>
    </tr>
    <tr>
      <th>641414049083691009</th>
      <td>neutral</td>
      <td>Fast Lerner: Subpoenaed tech guy who worked on...</td>
    </tr>
    <tr>
      <th>637666734300905472</th>
      <td>negative</td>
      <td>Sony rewards app is like a lot of 19 y.o femal...</td>
    </tr>
  </tbody>
</table>
</div>



Above we do a bit of preprocessing on the messages (drop duplicates, trim spaces, replace unicode code points with the actual characters...). The data structure is pretty simple: the tweet and its class.

Note that the dataset is highly imbalanced. We have much more neutral messages than the rest, and the negative messages are less than half the positive ones! We should be very cautious about this. It is easy to get fooled thinking our model is performing quite well, when in reality it is strongly overfitting to the neutral/positive class and performs poorly on the negative one.

## Objective and Metrics

This is a standard text-classification problem with 3 classes, and we need to make sure that our model performs equally well on all these classes. This is why the SemEval-2017 competition used **Macro Recall** as its primary metric as it is more robust to class imbalance than accuracy. Macro Recall is the unweighted average of the recall on each class.

In the original competition, 38 teams participated and the two best teams achieved an identical **macro-average recall of 68.1%**. Both teams used deep learning: one used an ensemble of LSTMs and CNNs with multiple convolution operations, while the other used deep LSTM networks with an attention mechanism.

On top of **Macro Recall**, the organizers also used **Accuracy** and **Macro F1** (only over the positive and negative classes) as secondary metrics. We will define a convenient function to evaluate our model on these 3 metrics through 3-fold cross-validation. We will also measure the recall of the negative class since it is the minority one, and the one we should pay the most attention too.


```python
from sklearn import metrics
from sklearn.model_selection import cross_validate

f1_pos_neg = metrics.make_scorer(
    metrics.f1_score,
    average='macro',
    labels=[labels_codes['negative'], labels_codes['positive']])
recall_neg = metrics.make_scorer(
    metrics.recall_score,
    average='micro',
    labels=[labels_codes['negative']])

def evaluate_model(model, features, labels, cv=3, fit_params=None):
    scores = cross_validate(
        model,
        features,
        labels,
        cv=cv,
        fit_params=fit_params,
        scoring={
            'recall_macro': 'recall_macro',
            'f1_pos_neg': f1_pos_neg,
            'accuracy': 'accuracy',
            'recall_neg': recall_neg,
        },
        n_jobs=-1,
    )

    results = pd.DataFrame(scores).drop(['fit_time', 'score_time'], axis=1)
    results.columns = pd.MultiIndex.from_tuples([c.split('_', maxsplit=1) for c in results.columns])    
    summary = results.describe()
    results = results.append(summary.loc[['mean', 'std']])

    def custom_style(row):
        color = 'white'
        if row.name == 'mean':
            color = 'yellow'
        return ['background-color: %s' % color]*len(row.values)
    results = results[sorted(results.columns, key=lambda x: x[0], reverse=True)]
    results = results.style.apply(custom_style, axis=1)

    return results
```

## Baseline

Before starting to experiment, let's have an idea of what performance we could reach by using an off-the-shelf library to classify the sentiment of tweets. We will use TextBlob, a popular python library to analyze text, which implemented a basic algorithm for sentiment analysis based on a lexicon.


```python
from sklearn.base import BaseEstimator
from textblob import TextBlob

class TextBlobClassifier(BaseEstimator):
    def __init__(self, threshold=0.1):
        self.threshold = threshold

    def fit(self, X, y):
        return self  # nothing to do

    def predict(self, X):
        labels = []      
        for m in X:
            blob = TextBlob(m)
            polarity = blob.sentiment.polarity
            if polarity > self.threshold:
                labels.append(labels_codes['positive'])
            elif polarity < -self.threshold:
                labels.append(labels_codes['negative'])
            else:
                labels.append(labels_codes['neutral'])
        return labels

evaluate_model(TextBlobClassifier(), tweets_train['message'], tweets_train_y)
```




<style  type="text/css" >
    #T_9400664c_cf52_11ea_9194_0242ac1c0002row0_col0 {
            background-color:  white;
        }    #T_9400664c_cf52_11ea_9194_0242ac1c0002row0_col1 {
            background-color:  white;
        }    #T_9400664c_cf52_11ea_9194_0242ac1c0002row0_col2 {
            background-color:  white;
        }    #T_9400664c_cf52_11ea_9194_0242ac1c0002row0_col3 {
            background-color:  white;
        }    #T_9400664c_cf52_11ea_9194_0242ac1c0002row1_col0 {
            background-color:  white;
        }    #T_9400664c_cf52_11ea_9194_0242ac1c0002row1_col1 {
            background-color:  white;
        }    #T_9400664c_cf52_11ea_9194_0242ac1c0002row1_col2 {
            background-color:  white;
        }    #T_9400664c_cf52_11ea_9194_0242ac1c0002row1_col3 {
            background-color:  white;
        }    #T_9400664c_cf52_11ea_9194_0242ac1c0002row2_col0 {
            background-color:  white;
        }    #T_9400664c_cf52_11ea_9194_0242ac1c0002row2_col1 {
            background-color:  white;
        }    #T_9400664c_cf52_11ea_9194_0242ac1c0002row2_col2 {
            background-color:  white;
        }    #T_9400664c_cf52_11ea_9194_0242ac1c0002row2_col3 {
            background-color:  white;
        }    #T_9400664c_cf52_11ea_9194_0242ac1c0002row3_col0 {
            background-color:  yellow;
        }    #T_9400664c_cf52_11ea_9194_0242ac1c0002row3_col1 {
            background-color:  yellow;
        }    #T_9400664c_cf52_11ea_9194_0242ac1c0002row3_col2 {
            background-color:  yellow;
        }    #T_9400664c_cf52_11ea_9194_0242ac1c0002row3_col3 {
            background-color:  yellow;
        }    #T_9400664c_cf52_11ea_9194_0242ac1c0002row4_col0 {
            background-color:  white;
        }    #T_9400664c_cf52_11ea_9194_0242ac1c0002row4_col1 {
            background-color:  white;
        }    #T_9400664c_cf52_11ea_9194_0242ac1c0002row4_col2 {
            background-color:  white;
        }    #T_9400664c_cf52_11ea_9194_0242ac1c0002row4_col3 {
            background-color:  white;
        }</style><table id="T_9400664c_cf52_11ea_9194_0242ac1c0002" ><thead> <tr>        <th class="blank level1" ></th>        <th class="col_heading level1 col0" >recall_macro</th>        <th class="col_heading level1 col1" >f1_pos_neg</th>        <th class="col_heading level1 col2" >accuracy</th>        <th class="col_heading level1 col3" >recall_neg</th>    </tr></thead><tbody>
                <tr>
                        <th id="T_9400664c_cf52_11ea_9194_0242ac1c0002level0_row0" class="row_heading level0 row0" >0</th>
                        <td id="T_9400664c_cf52_11ea_9194_0242ac1c0002row0_col0" class="data row0 col0" >0.496095</td>
                        <td id="T_9400664c_cf52_11ea_9194_0242ac1c0002row0_col1" class="data row0 col1" >0.468637</td>
                        <td id="T_9400664c_cf52_11ea_9194_0242ac1c0002row0_col2" class="data row0 col2" >0.535117</td>
                        <td id="T_9400664c_cf52_11ea_9194_0242ac1c0002row0_col3" class="data row0 col3" >0.341670</td>
            </tr>
            <tr>
                        <th id="T_9400664c_cf52_11ea_9194_0242ac1c0002level0_row1" class="row_heading level0 row1" >1</th>
                        <td id="T_9400664c_cf52_11ea_9194_0242ac1c0002row1_col0" class="data row1 col0" >0.489455</td>
                        <td id="T_9400664c_cf52_11ea_9194_0242ac1c0002row1_col1" class="data row1 col1" >0.458633</td>
                        <td id="T_9400664c_cf52_11ea_9194_0242ac1c0002row1_col2" class="data row1 col2" >0.533156</td>
                        <td id="T_9400664c_cf52_11ea_9194_0242ac1c0002row1_col3" class="data row1 col3" >0.322080</td>
            </tr>
            <tr>
                        <th id="T_9400664c_cf52_11ea_9194_0242ac1c0002level0_row2" class="row_heading level0 row2" >2</th>
                        <td id="T_9400664c_cf52_11ea_9194_0242ac1c0002row2_col0" class="data row2 col0" >0.489174</td>
                        <td id="T_9400664c_cf52_11ea_9194_0242ac1c0002row2_col1" class="data row2 col1" >0.460622</td>
                        <td id="T_9400664c_cf52_11ea_9194_0242ac1c0002row2_col2" class="data row2 col2" >0.531646</td>
                        <td id="T_9400664c_cf52_11ea_9194_0242ac1c0002row2_col3" class="data row2 col3" >0.326983</td>
            </tr>
            <tr>
                        <th id="T_9400664c_cf52_11ea_9194_0242ac1c0002level0_row3" class="row_heading level0 row3" >mean</th>
                        <td id="T_9400664c_cf52_11ea_9194_0242ac1c0002row3_col0" class="data row3 col0" >0.491575</td>
                        <td id="T_9400664c_cf52_11ea_9194_0242ac1c0002row3_col1" class="data row3 col1" >0.462631</td>
                        <td id="T_9400664c_cf52_11ea_9194_0242ac1c0002row3_col2" class="data row3 col2" >0.533306</td>
                        <td id="T_9400664c_cf52_11ea_9194_0242ac1c0002row3_col3" class="data row3 col3" >0.330244</td>
            </tr>
            <tr>
                        <th id="T_9400664c_cf52_11ea_9194_0242ac1c0002level0_row4" class="row_heading level0 row4" >std</th>
                        <td id="T_9400664c_cf52_11ea_9194_0242ac1c0002row4_col0" class="data row4 col0" >0.003917</td>
                        <td id="T_9400664c_cf52_11ea_9194_0242ac1c0002row4_col1" class="data row4 col1" >0.005296</td>
                        <td id="T_9400664c_cf52_11ea_9194_0242ac1c0002row4_col2" class="data row4 col2" >0.001740</td>
                        <td id="T_9400664c_cf52_11ea_9194_0242ac1c0002row4_col3" class="data row4 col3" >0.010194</td>
            </tr>
    </tbody></table>



This doesn't look so great. Our recall macro average is below 50%. Sure this is higher than random, but the metrics for the negative class look pretty bad (only 33% of actual negative messages are recognized as such).

This illustrates the limitations of the lexicon approach for social media messages, surely we can do better than this!

## Choosing the Right Feature Representation

A lot of time in machine learning, choosing the best way to represent the data is even more important than what kind of classifier to use. To demonstrate this, I will go through different ways we could represent our social media messages. Later we will see how big the impact is on our model's classification performance.


### Bags of Words

The bag of words representation makes a bold simplifying assumption: that word orders do not matter. For a given message, its feature representation is just the count for each possible word in our vocabulary.

The first step to transform our tweets into word counts is to **tokenize each message into a list of words**. The choice of the tokenization method is important as it can be ambiguous where to split words. Here I use the Spacy tokenizer, which I customized a bit for Twitter messages by defining special tokens for smileys, user handles...


```python
import spacy
from sklearn.base import TransformerMixin

class TweetTokenizer(BaseEstimator, TransformerMixin):
    # I rewrote most preprocessing rules according to the original GloVe's tokenizer (in Ruby):
    # https://nlp.stanford.edu/projects/glove/preprocess-twitter.rb

    eyes_regex = r'[8:=;]'
    nose_regex = r"['`\-]?"

    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm', disable = ['ner', 'tagger', 'parser', 'textcat'])
        for token in ['<url>', '<user>', '<smile>', '<lolface>', '<sadface>', '<neutralface>', '<heart>', '<number>', '<repeat>', '<elong>', '<hashtag>']:
            self.nlp.tokenizer.add_special_case(token, [{spacy.symbols.ORTH: token}])

    def fit(self, X, y=None):
        return self

    def transform(self, messages):
        messages =  messages.str.replace(r'https?://\S+\b|www\.(\w+\.)+\S*', '<URL>')

        # Force splitting words appended with slashes/parenthesis/brackets (once we tokenized the URLs, of course)
        messages = messages.str.replace(r'([/()\[\]])',r' \1 ')

        messages =  messages.str.replace(r'@\w+', '<USER>') #@mention
        messages =  messages.str.replace(r'[-+]?[.\d]*[\d]+[:,.\d]*', ' <NUMBER> ')

        def replace_hash_text(match):
            hash_text = match.group(1)
            if hash_text.isupper():
                return '<HASHTAG> ' + hash_text
            else:
                return '<HASHTAG> ' + ' '.join(re.findall(r'([a-zA-Z0-9]+?)(?=\b|[A-Z0-9_])', hash_text))

        messages =  messages.str.replace(r'#(\S+)', replace_hash_text)
        messages =  messages.str.replace(self.eyes_regex + self.nose_regex + r'[)d]+|[)d]+' + self.nose_regex + self.eyes_regex, '<SMILE>', flags=re.IGNORECASE)
        messages =  messages.str.replace(self.eyes_regex + self.nose_regex + r'p+', '<LOLFACE>', flags=re.IGNORECASE)
        messages =  messages.str.replace(self.eyes_regex + self.nose_regex + r'\(+|\)+' + self.nose_regex + self.eyes_regex, '<SADFACE>')
        messages =  messages.str.replace(self.eyes_regex + self.nose_regex + r'[/|l*]', '<NEUTRALFACE>')
        messages =  messages.str.replace(r'<3', '<HEART>')

        # Mark punctuation repetitions (eg. "!!!" => "! <REPEAT>")
        messages =  messages.str.replace(r'([!?.]){2,}', r'\1 <REPEAT>')

        # Mark elongated words (eg. "wayyyy" => "way <ELONG>")
        messages =  messages.str.replace(r'\b(\S*?)(.)\2{2,}\b', r'\1\2 <ELONG>')    

        # Replace all whitespace characters by only one space
        messages =  messages.str.replace(r'\s+', ' ')
        messages = messages.str.strip()
        messages =  messages.str.lower()

        return messages.apply(lambda msg: [token.text for token in self.nlp(msg)])

# let's see some examples:
tweets_train_tokenized = TweetTokenizer().fit_transform(tweets_train['message'])
tweets_train_tokenized[:5]
```




    id
    640329403277438976    [[, arirang, ], simply, kpop, -, kim, hyung, j...
    640810454730833920    [<user>, just, read, a, politico, article, abo...
    111344128507392000    [i, just, typed, in, ", the, bazura, project, ...
    641414049083691009    [fast, lerner, :, subpoenaed, tech, guy, who, ...
    637666734300905472    [sony, rewards, app, is, like, a, lot, of, <nu...
    Name: message, dtype: object



You can see  that the messages are now properly broken down into words and special characters like `[`, `-`, `:`. We also replace words with special meaning like usernames or number with unique tokens `<user>`, `<number>` for better interpretation by our model.

Now we can transform these tokenized sentences into word counts:


```python
from collections import Counter

Counter(tweets_train_tokenized.iloc[0])
```




    Counter({'(': 1,
             ')': 1,
             '-': 2,
             '.': 1,
             '<url>': 1,
             '[': 1,
             ']': 1,
             'arirang': 1,
             'cross': 1,
             'feat': 1,
             'ha': 1,
             'hyung': 1,
             'jun': 1,
             'kim': 1,
             'kpop': 1,
             'line': 1,
             'of': 1,
             'playback': 1,
             'simply': 1,
             'the': 1,
             'yeong': 1})



Above is an example of a bag of words representation for one message. You may think that it would be hard for our model to guess the overall sentiment only based on this, but it turns out that this simple representation already contains quite a lot of relevant information! For example it will be easy for the model to recognize and learn about positive/negative words.

These word counts will then be vectorized into a giant 2D array where each row represents one message and each row contains 19,558 integers (the number of tokens in our vocabulary). Notice that the matrix will be very sparse as a given message only contains a few of those words.

### Normalizing Bags of Words with TF-IDF

We could even improve further these bags of words representations by applying a simple transformation to improve their information content.

**Term Frequency - Inverse Document Frequency** is a method to give more importance to rare or unusual words and less importance to common words such as "the", "a"...

The idea is simple, you just normalize (divide) each word count by the frequency of this word in all documents (hence the "*inverse document frequency*").


```python
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

class BagOfWords(BaseEstimator, TransformerMixin):
    def __init__(self, min_frequency=2, clip_counts=False, use_tfidf=False):
        self.min_frequency = min_frequency # to reduce our total vocabulary size, we only keep words that appear at least n times
        self.clip_counts = clip_counts # clip the counts to a maximum of 1 (is the word present or not)
        self.use_tfidf = use_tfidf

    def fit(self, X, y=None):
        self.keep_columns = None
        self.vectorizer = DictVectorizer(dtype=np.int)

        self.tfidf_transformer = None
        if self.use_tfidf:
            self.tfidf_transformer = TfidfTransformer()

        if self.clip_counts:
            bags_of_words = X.apply(lambda tokens: Counter(set(tokens)))
        else:
            bags_of_words = X.apply(lambda tokens: Counter(tokens))

        X_vectors = self.vectorizer.fit_transform(bags_of_words)
        self.keep_columns = np.array(X_vectors.sum(axis=0) >= self.min_frequency).squeeze()

        if self.use_tfidf:
            self.tfidf_transformer.fit(X_vectors[:, self.keep_columns])

        return self

    def transform(self, X):
        if self.clip_counts:
            bags_of_words = X.apply(lambda tokens: Counter(set(tokens)))
        else:
            bags_of_words = X.apply(lambda tokens: Counter(tokens))

        X_vectors = self.vectorizer.transform(bags_of_words)
        X_vectors = X_vectors[:, self.keep_columns]
        if self.use_tfidf:
            X_vectors = self.tfidf_transformer.transform(X_vectors)

        return X_vectors


print("Examples of bags of words without normalization (raw counts per document):")
print(BagOfWords(min_frequency=2, use_tfidf=False).fit_transform(tweets_train_tokenized[:5]).toarray(), end='\n\n\n')

print("The same examples as above but with counts clipped to 1:")
print(BagOfWords(min_frequency=2, clip_counts=True).fit_transform(tweets_train_tokenized[:5]).toarray(), end='\n\n\n')

print("The same examples as above but with TF-IDF normalization:")
print(np.around(BagOfWords(min_frequency=2, use_tfidf=True).fit_transform(tweets_train_tokenized[:5]).toarray(), decimals=1))
```

    Examples of bags of words without normalization (raw counts per document):
    [[0 0 1 1 2 1 0 1 0 0 0 0 0 1 0 0 1 0 0]
     [0 1 0 0 0 2 1 0 1 0 0 1 1 1 1 0 0 1 0]
     [2 0 0 0 0 2 1 0 1 1 2 1 0 1 0 1 2 0 1]
     [0 1 1 1 0 0 1 1 0 0 0 0 0 0 1 1 1 1 0]
     [0 0 0 0 0 1 2 0 2 1 1 0 1 1 0 0 0 0 1]]


    The same examples as above but with counts clipped to 1:
    [[0 1 1 1 0 1 0 0 0 0 0 1 0 0 1 0 0]
     [1 0 0 1 1 0 1 0 0 1 1 1 1 0 0 1 0]
     [0 0 0 1 1 0 1 1 1 1 0 1 0 1 1 0 1]
     [1 1 1 0 1 1 0 0 0 0 0 0 1 1 1 1 0]
     [0 0 0 1 1 0 1 1 1 0 1 1 0 0 0 0 1]]


    The same examples as above but with TF-IDF normalization:
    [[0.  0.  0.3 0.3 0.8 0.2 0.  0.3 0.  0.  0.  0.  0.  0.2 0.  0.  0.3 0.
      0. ]
     [0.  0.3 0.  0.  0.  0.5 0.2 0.  0.3 0.  0.  0.3 0.3 0.2 0.3 0.  0.  0.3
      0. ]
     [0.5 0.  0.  0.  0.  0.3 0.2 0.  0.2 0.2 0.4 0.2 0.  0.2 0.  0.2 0.4 0.
      0.2]
     [0.  0.3 0.3 0.3 0.  0.  0.2 0.3 0.  0.  0.  0.  0.  0.  0.3 0.3 0.3 0.3
      0. ]
     [0.  0.  0.  0.  0.  0.2 0.4 0.  0.5 0.3 0.3 0.  0.3 0.2 0.  0.  0.  0.
      0.3]]


As you can see, after the TF-IDF transformation, all 0s stay 0s but the positive count integers seen above are turned into floats after being divided by the frequency of each term across all documents.

Here all TF-IDF scores are between 0 and 1, but this is because the Scikit-Learn TF-IDF implementation also applies an L2 normalization step to each row (divide each row by its L2 norm). In other implementations these numbers could actually be greater than 1.


### Word Embeddings and BERT

Another powerful representation are **word embeddings**, where we map each individual word to a continuous vector in a high-dimensional space. The main intuition behind this technique is that similar words should be close together in the embedding space. For example the embeddings for `love` and `affection` are much closer than the ones for `horse` and `shoes`.

There exist many ways to construct embeddings for a given vocabulary. Some of the most popular methods include [word2vec](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf) or [GloVe](https://nlp.stanford.edu/projects/glove/). However in recent years people have started to use more and more **contextual word embeddings**, where embeddings are computed not only from the target word but also from the context it appears in. This allows for much richer representations and disambiguation of identical words in different settings. For example the embedding for `right` would be quite different in each of these 2 sentences:

* He was *right*.
* Turn *right* after the sign.

Being able to differentiate meaning between otherwise identical-looking words is important for sentiment analysis. [BERT](https://arxiv.org/abs/1810.04805) is one model which allow us to extract embeddings which take into account the context, so it will be a great representation for our social media messages.

Let's now encode all our tweets into BERT embeddings, we will use the convenient [Transformers library](https://github.com/huggingface/transformers) for this. BERT is quite a large model, and it can take some time to encode all the messages in the training set. I highly suggest using a GPU for that if you have one, or you can use [Google Colab](https://colab.research.google.com) which provides a free GPU for experimentation.


```python
import torch
import transformers
import tqdm

class BertEmbeddings(TransformerMixin):
    def __init__(self, max_sequence_length=50, batch_size=32, device='cpu'):
        self.max_sequence_length = max_sequence_length
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-cased')
        self.model = transformers.BertModel.from_pretrained('bert-base-cased', output_hidden_states=True)
        self.model.eval()
        self.model.to(self.device)

    def transform(self, messages):
        embeddings = np.zeros((len(messages),768), dtype=np.float32)
        with torch.no_grad():
            for i in tqdm.tqdm(range(0, len(messages), self.batch_size)):
                encoded = self.tokenizer.batch_encode_plus(
                    messages[i:i+self.batch_size],
                    max_length=self.max_sequence_length,
                    pad_to_max_length=True,
                    truncation=True)
                output = self.model(
                    input_ids=torch.tensor(encoded['input_ids']).to(self.device),
                    attention_mask=torch.tensor(encoded['attention_mask']).to(self.device),
                    token_type_ids=torch.tensor(encoded['token_type_ids']).to(self.device))

                # the indice `2` of the output contains a tuple of hidden states for all layers + final embeddings output    
                full_embeddings = output[2][-2].cpu().numpy()
                for j in range(len(encoded['input_ids'])):
                    sentence_length = len(np.nonzero(encoded['input_ids'][j])[0])
                    words_embeddings = full_embeddings[j,:sentence_length]
                    # We multiply the avg. embedding by the square root of the number of words
                    # to compensate for the fact that the mean of n vectors gets shorter as n grows
                    embeddings[i+j] = np.mean(words_embeddings, axis=0) * np.sqrt(len(words_embeddings))

        return embeddings

tweets_train_bert = BertEmbeddings(device='cuda').transform(tweets_train['message'].to_numpy())
print('Finished encoding the messages! The final shape of our array is {}'.format(tweets_train_bert.shape))
```

    100%|██████████| 1553/1553 [02:28<00:00, 10.46it/s]

    Finished encoding the messages! The final shape of our array is (49675, 768)





We now have a giant matrix where each message is encoded into a vector of length 768. Note that our final embeddings are actually message-level representations instead of word-level embeddings. We obtained these message-level vectors by averaging the individual word embeddings across each message. This is needed because we will later use a simple model that requires fixed-length features to classify the tweets.

It would be also possible to directly use separate word embeddings for classification, but we would need a more complex model that can handle variable-length messages (like a recurrent neural network for example). In order to keep things simple for this guide we will only use the message-level embeddings for now!

## Training the Model

Ok we now have a bunch of different representations for our tweets. Let's start building models to experiment with these.


### Naive Bayes

Naive Bayes is a popular classifier for text-related problems. As its name indicates, this classifier makes a naive assumption: the conditional independence assumption. Basically the model considers that the probability of each word presence in a message is completely independent from the presence of other words. It is easy to see how far a stretch this assumption is, as words are heavily correlated in a sentence (ex: if 'cake' is present then the probability of 'delicious' being also present is slightly higher).

Nevertheless the simplicity of this model doesn't impact too much its performance and Naive Bayes is usually one of the first thing people try when they have to deal with textual data.

This is what we will do too, and we train this model on the bags of words representations of our messages:


```python
from sklearn.naive_bayes import MultinomialNB

# we set fit_prior=False to use a uniform distribution as the prior (better for imbalanced classification)
evaluate_model(
    MultinomialNB(fit_prior=False),
    BagOfWords(min_frequency=2, clip_counts=True).fit_transform(tweets_train_tokenized),
    tweets_train_y)
```




<style  type="text/css" >
    #T_f8f824fe_cf52_11ea_9194_0242ac1c0002row0_col0 {
            background-color:  white;
        }    #T_f8f824fe_cf52_11ea_9194_0242ac1c0002row0_col1 {
            background-color:  white;
        }    #T_f8f824fe_cf52_11ea_9194_0242ac1c0002row0_col2 {
            background-color:  white;
        }    #T_f8f824fe_cf52_11ea_9194_0242ac1c0002row0_col3 {
            background-color:  white;
        }    #T_f8f824fe_cf52_11ea_9194_0242ac1c0002row1_col0 {
            background-color:  white;
        }    #T_f8f824fe_cf52_11ea_9194_0242ac1c0002row1_col1 {
            background-color:  white;
        }    #T_f8f824fe_cf52_11ea_9194_0242ac1c0002row1_col2 {
            background-color:  white;
        }    #T_f8f824fe_cf52_11ea_9194_0242ac1c0002row1_col3 {
            background-color:  white;
        }    #T_f8f824fe_cf52_11ea_9194_0242ac1c0002row2_col0 {
            background-color:  white;
        }    #T_f8f824fe_cf52_11ea_9194_0242ac1c0002row2_col1 {
            background-color:  white;
        }    #T_f8f824fe_cf52_11ea_9194_0242ac1c0002row2_col2 {
            background-color:  white;
        }    #T_f8f824fe_cf52_11ea_9194_0242ac1c0002row2_col3 {
            background-color:  white;
        }    #T_f8f824fe_cf52_11ea_9194_0242ac1c0002row3_col0 {
            background-color:  yellow;
        }    #T_f8f824fe_cf52_11ea_9194_0242ac1c0002row3_col1 {
            background-color:  yellow;
        }    #T_f8f824fe_cf52_11ea_9194_0242ac1c0002row3_col2 {
            background-color:  yellow;
        }    #T_f8f824fe_cf52_11ea_9194_0242ac1c0002row3_col3 {
            background-color:  yellow;
        }    #T_f8f824fe_cf52_11ea_9194_0242ac1c0002row4_col0 {
            background-color:  white;
        }    #T_f8f824fe_cf52_11ea_9194_0242ac1c0002row4_col1 {
            background-color:  white;
        }    #T_f8f824fe_cf52_11ea_9194_0242ac1c0002row4_col2 {
            background-color:  white;
        }    #T_f8f824fe_cf52_11ea_9194_0242ac1c0002row4_col3 {
            background-color:  white;
        }</style><table id="T_f8f824fe_cf52_11ea_9194_0242ac1c0002" ><thead>   <tr>        <th class="blank level1" ></th>        <th class="col_heading level1 col0" >recall_macro</th>        <th class="col_heading level1 col1" >f1_pos_neg</th>        <th class="col_heading level1 col2" >accuracy</th>        <th class="col_heading level1 col3" >recall_neg</th>    </tr></thead><tbody>
                <tr>
                        <th id="T_f8f824fe_cf52_11ea_9194_0242ac1c0002level0_row0" class="row_heading level0 row0" >0</th>
                        <td id="T_f8f824fe_cf52_11ea_9194_0242ac1c0002row0_col0" class="data row0 col0" >0.620937</td>
                        <td id="T_f8f824fe_cf52_11ea_9194_0242ac1c0002row0_col1" class="data row0 col1" >0.602270</td>
                        <td id="T_f8f824fe_cf52_11ea_9194_0242ac1c0002row0_col2" class="data row0 col2" >0.618093</td>
                        <td id="T_f8f824fe_cf52_11ea_9194_0242ac1c0002row0_col3" class="data row0 col3" >0.611305</td>
            </tr>
            <tr>
                        <th id="T_f8f824fe_cf52_11ea_9194_0242ac1c0002level0_row1" class="row_heading level0 row1" >1</th>
                        <td id="T_f8f824fe_cf52_11ea_9194_0242ac1c0002row1_col0" class="data row1 col0" >0.615800</td>
                        <td id="T_f8f824fe_cf52_11ea_9194_0242ac1c0002row1_col1" class="data row1 col1" >0.599981</td>
                        <td id="T_f8f824fe_cf52_11ea_9194_0242ac1c0002row1_col2" class="data row1 col2" >0.618009</td>
                        <td id="T_f8f824fe_cf52_11ea_9194_0242ac1c0002row1_col3" class="data row1 col3" >0.588691</td>
            </tr>
            <tr>
                        <th id="T_f8f824fe_cf52_11ea_9194_0242ac1c0002level0_row2" class="row_heading level0 row2" >2</th>
                        <td id="T_f8f824fe_cf52_11ea_9194_0242ac1c0002row2_col0" class="data row2 col0" >0.615914</td>
                        <td id="T_f8f824fe_cf52_11ea_9194_0242ac1c0002row2_col1" class="data row2 col1" >0.595471</td>
                        <td id="T_f8f824fe_cf52_11ea_9194_0242ac1c0002row2_col2" class="data row2 col2" >0.617768</td>
                        <td id="T_f8f824fe_cf52_11ea_9194_0242ac1c0002row2_col3" class="data row2 col3" >0.592334</td>
            </tr>
            <tr>
                        <th id="T_f8f824fe_cf52_11ea_9194_0242ac1c0002level0_row3" class="row_heading level0 row3" >mean</th>
                        <td id="T_f8f824fe_cf52_11ea_9194_0242ac1c0002row3_col0" class="data row3 col0" >0.617550</td>
                        <td id="T_f8f824fe_cf52_11ea_9194_0242ac1c0002row3_col1" class="data row3 col1" >0.599241</td>
                        <td id="T_f8f824fe_cf52_11ea_9194_0242ac1c0002row3_col2" class="data row3 col2" >0.617957</td>
                        <td id="T_f8f824fe_cf52_11ea_9194_0242ac1c0002row3_col3" class="data row3 col3" >0.597443</td>
            </tr>
            <tr>
                        <th id="T_f8f824fe_cf52_11ea_9194_0242ac1c0002level0_row4" class="row_heading level0 row4" >std</th>
                        <td id="T_f8f824fe_cf52_11ea_9194_0242ac1c0002row4_col0" class="data row4 col0" >0.002933</td>
                        <td id="T_f8f824fe_cf52_11ea_9194_0242ac1c0002row4_col1" class="data row4 col1" >0.003459</td>
                        <td id="T_f8f824fe_cf52_11ea_9194_0242ac1c0002row4_col2" class="data row4 col2" >0.000169</td>
                        <td id="T_f8f824fe_cf52_11ea_9194_0242ac1c0002row4_col3" class="data row4 col3" >0.012142</td>
            </tr>
    </tbody></table>



Not bad for a first try! Notice how our **avg. macro recall of ~62%** is much higher than our baseline (<50%). According to the SemEval-2017 paper this would actually put us among the top 20 of the official ranking, despite the simplicity of the model and features representation.

*Note: This is not exactly an apple-to-apple comparison since our results are obtained from cross-validation on the training data. The SemEval-2017 teams report their performance on the official test data. We will do so too at the end.*

Although Naive Bayes seems to perform great out-of-the-box, [it has been demonstrated](http://ai.stanford.edu/~ang/papers/nips01-discriminativegenerative.pdf) that Logistic regression, which takes longer to converge, can reach a lower error than Naive Bayes when properly fine-tuned. So let's try that!

### Logisitic Regression

This is usually a binary classifier but we can extend it to work for multiple classes by replacing the binary loss with the cross-entropy loss.

Let's first build a simple model with default settings:
* We choose `multi_class='multimonial'` to specify that we want to use Softmax/cross-entropy loss (and not 3 different binary  classifiers)
* We use `solver='lbfgs'` to pick a solver that supports Softmax
* We set `class_weight='balanced'` to give more importance to the less represented classes during training (i.e. negative messages)

We use bags of words to train the model here too.


```python
from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression(multi_class='multinomial', solver='lbfgs', class_weight='balanced')

evaluate_model(
    log_reg,
    BagOfWords().fit_transform(tweets_train_tokenized),
    tweets_train_y)
```




<style  type="text/css" >
    #T_fe0a497c_cf52_11ea_9194_0242ac1c0002row0_col0 {
            background-color:  white;
        }    #T_fe0a497c_cf52_11ea_9194_0242ac1c0002row0_col1 {
            background-color:  white;
        }    #T_fe0a497c_cf52_11ea_9194_0242ac1c0002row0_col2 {
            background-color:  white;
        }    #T_fe0a497c_cf52_11ea_9194_0242ac1c0002row0_col3 {
            background-color:  white;
        }    #T_fe0a497c_cf52_11ea_9194_0242ac1c0002row1_col0 {
            background-color:  white;
        }    #T_fe0a497c_cf52_11ea_9194_0242ac1c0002row1_col1 {
            background-color:  white;
        }    #T_fe0a497c_cf52_11ea_9194_0242ac1c0002row1_col2 {
            background-color:  white;
        }    #T_fe0a497c_cf52_11ea_9194_0242ac1c0002row1_col3 {
            background-color:  white;
        }    #T_fe0a497c_cf52_11ea_9194_0242ac1c0002row2_col0 {
            background-color:  white;
        }    #T_fe0a497c_cf52_11ea_9194_0242ac1c0002row2_col1 {
            background-color:  white;
        }    #T_fe0a497c_cf52_11ea_9194_0242ac1c0002row2_col2 {
            background-color:  white;
        }    #T_fe0a497c_cf52_11ea_9194_0242ac1c0002row2_col3 {
            background-color:  white;
        }    #T_fe0a497c_cf52_11ea_9194_0242ac1c0002row3_col0 {
            background-color:  yellow;
        }    #T_fe0a497c_cf52_11ea_9194_0242ac1c0002row3_col1 {
            background-color:  yellow;
        }    #T_fe0a497c_cf52_11ea_9194_0242ac1c0002row3_col2 {
            background-color:  yellow;
        }    #T_fe0a497c_cf52_11ea_9194_0242ac1c0002row3_col3 {
            background-color:  yellow;
        }    #T_fe0a497c_cf52_11ea_9194_0242ac1c0002row4_col0 {
            background-color:  white;
        }    #T_fe0a497c_cf52_11ea_9194_0242ac1c0002row4_col1 {
            background-color:  white;
        }    #T_fe0a497c_cf52_11ea_9194_0242ac1c0002row4_col2 {
            background-color:  white;
        }    #T_fe0a497c_cf52_11ea_9194_0242ac1c0002row4_col3 {
            background-color:  white;
        }</style><table id="T_fe0a497c_cf52_11ea_9194_0242ac1c0002" ><thead>    <tr>        <th class="blank level1" ></th>        <th class="col_heading level1 col0" >recall_macro</th>        <th class="col_heading level1 col1" >f1_pos_neg</th>        <th class="col_heading level1 col2" >accuracy</th>        <th class="col_heading level1 col3" >recall_neg</th>    </tr></thead><tbody>
                <tr>
                        <th id="T_fe0a497c_cf52_11ea_9194_0242ac1c0002level0_row0" class="row_heading level0 row0" >0</th>
                        <td id="T_fe0a497c_cf52_11ea_9194_0242ac1c0002row0_col0" class="data row0 col0" >0.641287</td>
                        <td id="T_fe0a497c_cf52_11ea_9194_0242ac1c0002row0_col1" class="data row0 col1" >0.619628</td>
                        <td id="T_fe0a497c_cf52_11ea_9194_0242ac1c0002row0_col2" class="data row0 col2" >0.647080</td>
                        <td id="T_fe0a497c_cf52_11ea_9194_0242ac1c0002row0_col3" class="data row0 col3" >0.613240</td>
            </tr>
            <tr>
                        <th id="T_fe0a497c_cf52_11ea_9194_0242ac1c0002level0_row1" class="row_heading level0 row1" >1</th>
                        <td id="T_fe0a497c_cf52_11ea_9194_0242ac1c0002row1_col0" class="data row1 col0" >0.631539</td>
                        <td id="T_fe0a497c_cf52_11ea_9194_0242ac1c0002row1_col1" class="data row1 col1" >0.610504</td>
                        <td id="T_fe0a497c_cf52_11ea_9194_0242ac1c0002row1_col2" class="data row1 col2" >0.645126</td>
                        <td id="T_fe0a497c_cf52_11ea_9194_0242ac1c0002row1_col3" class="data row1 col3" >0.575136</td>
            </tr>
            <tr>
                        <th id="T_fe0a497c_cf52_11ea_9194_0242ac1c0002level0_row2" class="row_heading level0 row2" >2</th>
                        <td id="T_fe0a497c_cf52_11ea_9194_0242ac1c0002row2_col0" class="data row2 col0" >0.637179</td>
                        <td id="T_fe0a497c_cf52_11ea_9194_0242ac1c0002row2_col1" class="data row2 col1" >0.611158</td>
                        <td id="T_fe0a497c_cf52_11ea_9194_0242ac1c0002row2_col2" class="data row2 col2" >0.646998</td>
                        <td id="T_fe0a497c_cf52_11ea_9194_0242ac1c0002row2_col3" class="data row2 col3" >0.596980</td>
            </tr>
            <tr>
                        <th id="T_fe0a497c_cf52_11ea_9194_0242ac1c0002level0_row3" class="row_heading level0 row3" >mean</th>
                        <td id="T_fe0a497c_cf52_11ea_9194_0242ac1c0002row3_col0" class="data row3 col0" >0.636669</td>
                        <td id="T_fe0a497c_cf52_11ea_9194_0242ac1c0002row3_col1" class="data row3 col1" >0.613763</td>
                        <td id="T_fe0a497c_cf52_11ea_9194_0242ac1c0002row3_col2" class="data row3 col2" >0.646402</td>
                        <td id="T_fe0a497c_cf52_11ea_9194_0242ac1c0002row3_col3" class="data row3 col3" >0.595119</td>
            </tr>
            <tr>
                        <th id="T_fe0a497c_cf52_11ea_9194_0242ac1c0002level0_row4" class="row_heading level0 row4" >std</th>
                        <td id="T_fe0a497c_cf52_11ea_9194_0242ac1c0002row4_col0" class="data row4 col0" >0.004894</td>
                        <td id="T_fe0a497c_cf52_11ea_9194_0242ac1c0002row4_col1" class="data row4 col1" >0.005089</td>
                        <td id="T_fe0a497c_cf52_11ea_9194_0242ac1c0002row4_col2" class="data row4 col2" >0.001105</td>
                        <td id="T_fe0a497c_cf52_11ea_9194_0242ac1c0002row4_col3" class="data row4 col3" >0.019121</td>
            </tr>
    </tbody></table>





As you can see we gain a few percentage points of macro recall compared with Naive Bayes. We can even go further by using normalized word counts with TF-IDF:


```python
evaluate_model(
    log_reg,
    BagOfWords(use_tfidf=True).fit_transform(tweets_train_tokenized),
    tweets_train_y)
```




<style  type="text/css" >
    #T_032d7dc0_cf53_11ea_9194_0242ac1c0002row0_col0 {
            background-color:  white;
        }    #T_032d7dc0_cf53_11ea_9194_0242ac1c0002row0_col1 {
            background-color:  white;
        }    #T_032d7dc0_cf53_11ea_9194_0242ac1c0002row0_col2 {
            background-color:  white;
        }    #T_032d7dc0_cf53_11ea_9194_0242ac1c0002row0_col3 {
            background-color:  white;
        }    #T_032d7dc0_cf53_11ea_9194_0242ac1c0002row1_col0 {
            background-color:  white;
        }    #T_032d7dc0_cf53_11ea_9194_0242ac1c0002row1_col1 {
            background-color:  white;
        }    #T_032d7dc0_cf53_11ea_9194_0242ac1c0002row1_col2 {
            background-color:  white;
        }    #T_032d7dc0_cf53_11ea_9194_0242ac1c0002row1_col3 {
            background-color:  white;
        }    #T_032d7dc0_cf53_11ea_9194_0242ac1c0002row2_col0 {
            background-color:  white;
        }    #T_032d7dc0_cf53_11ea_9194_0242ac1c0002row2_col1 {
            background-color:  white;
        }    #T_032d7dc0_cf53_11ea_9194_0242ac1c0002row2_col2 {
            background-color:  white;
        }    #T_032d7dc0_cf53_11ea_9194_0242ac1c0002row2_col3 {
            background-color:  white;
        }    #T_032d7dc0_cf53_11ea_9194_0242ac1c0002row3_col0 {
            background-color:  yellow;
        }    #T_032d7dc0_cf53_11ea_9194_0242ac1c0002row3_col1 {
            background-color:  yellow;
        }    #T_032d7dc0_cf53_11ea_9194_0242ac1c0002row3_col2 {
            background-color:  yellow;
        }    #T_032d7dc0_cf53_11ea_9194_0242ac1c0002row3_col3 {
            background-color:  yellow;
        }    #T_032d7dc0_cf53_11ea_9194_0242ac1c0002row4_col0 {
            background-color:  white;
        }    #T_032d7dc0_cf53_11ea_9194_0242ac1c0002row4_col1 {
            background-color:  white;
        }    #T_032d7dc0_cf53_11ea_9194_0242ac1c0002row4_col2 {
            background-color:  white;
        }    #T_032d7dc0_cf53_11ea_9194_0242ac1c0002row4_col3 {
            background-color:  white;
        }</style><table id="T_032d7dc0_cf53_11ea_9194_0242ac1c0002" ><thead>    <tr>        <th class="blank level1" ></th>        <th class="col_heading level1 col0" >recall_macro</th>        <th class="col_heading level1 col1" >f1_pos_neg</th>        <th class="col_heading level1 col2" >accuracy</th>        <th class="col_heading level1 col3" >recall_neg</th>    </tr></thead><tbody>
                <tr>
                        <th id="T_032d7dc0_cf53_11ea_9194_0242ac1c0002level0_row0" class="row_heading level0 row0" >0</th>
                        <td id="T_032d7dc0_cf53_11ea_9194_0242ac1c0002row0_col0" class="data row0 col0" >0.656339</td>
                        <td id="T_032d7dc0_cf53_11ea_9194_0242ac1c0002row0_col1" class="data row0 col1" >0.625988</td>
                        <td id="T_032d7dc0_cf53_11ea_9194_0242ac1c0002row0_col2" class="data row0 col2" >0.653119</td>
                        <td id="T_032d7dc0_cf53_11ea_9194_0242ac1c0002row0_col3" class="data row0 col3" >0.662408</td>
            </tr>
            <tr>
                        <th id="T_032d7dc0_cf53_11ea_9194_0242ac1c0002level0_row1" class="row_heading level0 row1" >1</th>
                        <td id="T_032d7dc0_cf53_11ea_9194_0242ac1c0002row1_col0" class="data row1 col0" >0.656458</td>
                        <td id="T_032d7dc0_cf53_11ea_9194_0242ac1c0002row1_col1" class="data row1 col1" >0.627115</td>
                        <td id="T_032d7dc0_cf53_11ea_9194_0242ac1c0002row1_col2" class="data row1 col2" >0.655574</td>
                        <td id="T_032d7dc0_cf53_11ea_9194_0242ac1c0002row1_col3" class="data row1 col3" >0.655306</td>
            </tr>
            <tr>
                        <th id="T_032d7dc0_cf53_11ea_9194_0242ac1c0002level0_row2" class="row_heading level0 row2" >2</th>
                        <td id="T_032d7dc0_cf53_11ea_9194_0242ac1c0002row2_col0" class="data row2 col0" >0.658039</td>
                        <td id="T_032d7dc0_cf53_11ea_9194_0242ac1c0002row2_col1" class="data row2 col1" >0.624048</td>
                        <td id="T_032d7dc0_cf53_11ea_9194_0242ac1c0002row2_col2" class="data row2 col2" >0.656480</td>
                        <td id="T_032d7dc0_cf53_11ea_9194_0242ac1c0002row2_col3" class="data row2 col3" >0.661634</td>
            </tr>
            <tr>
                        <th id="T_032d7dc0_cf53_11ea_9194_0242ac1c0002level0_row3" class="row_heading level0 row3" >mean</th>
                        <td id="T_032d7dc0_cf53_11ea_9194_0242ac1c0002row3_col0" class="data row3 col0" >0.656945</td>
                        <td id="T_032d7dc0_cf53_11ea_9194_0242ac1c0002row3_col1" class="data row3 col1" >0.625717</td>
                        <td id="T_032d7dc0_cf53_11ea_9194_0242ac1c0002row3_col2" class="data row3 col2" >0.655058</td>
                        <td id="T_032d7dc0_cf53_11ea_9194_0242ac1c0002row3_col3" class="data row3 col3" >0.659783</td>
            </tr>
            <tr>
                        <th id="T_032d7dc0_cf53_11ea_9194_0242ac1c0002level0_row4" class="row_heading level0 row4" >std</th>
                        <td id="T_032d7dc0_cf53_11ea_9194_0242ac1c0002row4_col0" class="data row4 col0" >0.000949</td>
                        <td id="T_032d7dc0_cf53_11ea_9194_0242ac1c0002row4_col1" class="data row4 col1" >0.001552</td>
                        <td id="T_032d7dc0_cf53_11ea_9194_0242ac1c0002row4_col2" class="data row4 col2" >0.001739</td>
                        <td id="T_032d7dc0_cf53_11ea_9194_0242ac1c0002row4_col3" class="data row4 col3" >0.003896</td>
            </tr>
    </tbody></table>



Wow the macro recall jumped by more than 2% just from this simple normalization procedure.

Finally let's experiment with BERT embeddings:


```python
evaluate_model(
    log_reg,
    tweets_train_bert,
    tweets_train_y)
```

    /usr/local/lib/python3.6/dist-packages/joblib/externals/loky/process_executor.py:691: UserWarning: A worker stopped while some jobs were given to the executor. This can be caused by a too short worker timeout or by a memory leak.
      "timeout or by a memory leak.", UserWarning





<style  type="text/css" >
    #T_17103198_cf53_11ea_9194_0242ac1c0002row0_col0 {
            background-color:  white;
        }    #T_17103198_cf53_11ea_9194_0242ac1c0002row0_col1 {
            background-color:  white;
        }    #T_17103198_cf53_11ea_9194_0242ac1c0002row0_col2 {
            background-color:  white;
        }    #T_17103198_cf53_11ea_9194_0242ac1c0002row0_col3 {
            background-color:  white;
        }    #T_17103198_cf53_11ea_9194_0242ac1c0002row1_col0 {
            background-color:  white;
        }    #T_17103198_cf53_11ea_9194_0242ac1c0002row1_col1 {
            background-color:  white;
        }    #T_17103198_cf53_11ea_9194_0242ac1c0002row1_col2 {
            background-color:  white;
        }    #T_17103198_cf53_11ea_9194_0242ac1c0002row1_col3 {
            background-color:  white;
        }    #T_17103198_cf53_11ea_9194_0242ac1c0002row2_col0 {
            background-color:  white;
        }    #T_17103198_cf53_11ea_9194_0242ac1c0002row2_col1 {
            background-color:  white;
        }    #T_17103198_cf53_11ea_9194_0242ac1c0002row2_col2 {
            background-color:  white;
        }    #T_17103198_cf53_11ea_9194_0242ac1c0002row2_col3 {
            background-color:  white;
        }    #T_17103198_cf53_11ea_9194_0242ac1c0002row3_col0 {
            background-color:  yellow;
        }    #T_17103198_cf53_11ea_9194_0242ac1c0002row3_col1 {
            background-color:  yellow;
        }    #T_17103198_cf53_11ea_9194_0242ac1c0002row3_col2 {
            background-color:  yellow;
        }    #T_17103198_cf53_11ea_9194_0242ac1c0002row3_col3 {
            background-color:  yellow;
        }    #T_17103198_cf53_11ea_9194_0242ac1c0002row4_col0 {
            background-color:  white;
        }    #T_17103198_cf53_11ea_9194_0242ac1c0002row4_col1 {
            background-color:  white;
        }    #T_17103198_cf53_11ea_9194_0242ac1c0002row4_col2 {
            background-color:  white;
        }    #T_17103198_cf53_11ea_9194_0242ac1c0002row4_col3 {
            background-color:  white;
        }</style><table id="T_17103198_cf53_11ea_9194_0242ac1c0002" ><thead>    <tr>        <th class="blank level1" ></th>        <th class="col_heading level1 col0" >recall_macro</th>        <th class="col_heading level1 col1" >f1_pos_neg</th>        <th class="col_heading level1 col2" >accuracy</th>        <th class="col_heading level1 col3" >recall_neg</th>    </tr></thead><tbody>
                <tr>
                        <th id="T_17103198_cf53_11ea_9194_0242ac1c0002level0_row0" class="row_heading level0 row0" >0</th>
                        <td id="T_17103198_cf53_11ea_9194_0242ac1c0002row0_col0" class="data row0 col0" >0.674462</td>
                        <td id="T_17103198_cf53_11ea_9194_0242ac1c0002row0_col1" class="data row0 col1" >0.645385</td>
                        <td id="T_17103198_cf53_11ea_9194_0242ac1c0002row0_col2" class="data row0 col2" >0.652455</td>
                        <td id="T_17103198_cf53_11ea_9194_0242ac1c0002row0_col3" class="data row0 col3" >0.745645</td>
            </tr>
            <tr>
                        <th id="T_17103198_cf53_11ea_9194_0242ac1c0002level0_row1" class="row_heading level0 row1" >1</th>
                        <td id="T_17103198_cf53_11ea_9194_0242ac1c0002row1_col0" class="data row1 col0" >0.671421</td>
                        <td id="T_17103198_cf53_11ea_9194_0242ac1c0002row1_col1" class="data row1 col1" >0.640748</td>
                        <td id="T_17103198_cf53_11ea_9194_0242ac1c0002row1_col2" class="data row1 col2" >0.650079</td>
                        <td id="T_17103198_cf53_11ea_9194_0242ac1c0002row1_col3" class="data row1 col3" >0.742835</td>
            </tr>
            <tr>
                        <th id="T_17103198_cf53_11ea_9194_0242ac1c0002level0_row2" class="row_heading level0 row2" >2</th>
                        <td id="T_17103198_cf53_11ea_9194_0242ac1c0002row2_col0" class="data row2 col0" >0.674438</td>
                        <td id="T_17103198_cf53_11ea_9194_0242ac1c0002row2_col1" class="data row2 col1" >0.644842</td>
                        <td id="T_17103198_cf53_11ea_9194_0242ac1c0002row2_col2" class="data row2 col2" >0.652857</td>
                        <td id="T_17103198_cf53_11ea_9194_0242ac1c0002row2_col3" class="data row2 col3" >0.745257</td>
            </tr>
            <tr>
                        <th id="T_17103198_cf53_11ea_9194_0242ac1c0002level0_row3" class="row_heading level0 row3" >mean</th>
                        <td id="T_17103198_cf53_11ea_9194_0242ac1c0002row3_col0" class="data row3 col0" >0.673440</td>
                        <td id="T_17103198_cf53_11ea_9194_0242ac1c0002row3_col1" class="data row3 col1" >0.643659</td>
                        <td id="T_17103198_cf53_11ea_9194_0242ac1c0002row3_col2" class="data row3 col2" >0.651797</td>
                        <td id="T_17103198_cf53_11ea_9194_0242ac1c0002row3_col3" class="data row3 col3" >0.744579</td>
            </tr>
            <tr>
                        <th id="T_17103198_cf53_11ea_9194_0242ac1c0002level0_row4" class="row_heading level0 row4" >std</th>
                        <td id="T_17103198_cf53_11ea_9194_0242ac1c0002row4_col0" class="data row4 col0" >0.001749</td>
                        <td id="T_17103198_cf53_11ea_9194_0242ac1c0002row4_col1" class="data row4 col1" >0.002535</td>
                        <td id="T_17103198_cf53_11ea_9194_0242ac1c0002row4_col2" class="data row4 col2" >0.001501</td>
                        <td id="T_17103198_cf53_11ea_9194_0242ac1c0002row4_col3" class="data row4 col3" >0.001523</td>
            </tr>
    </tbody></table>



This is our best performance so far! With BERT features we can reach **67% macro recall**.

Can we improve this model even further? Before concluding this notebook, let's fine-tune our model hyper-parameters to optimize to the maximum potential of this model.

### Fine-tuning the Model

Here are some possible optimizations we can try:

* **Standardization**: With simple linear classifier like Logistic Regression, it is always a good idea to normalize the features to have mean 0 and a standard deviation of 1.
* **Regularization Strength**: Regularizing our model could help us improve the model generalization and performance on unseen data.
* **Training Duration**: Training the model for more iterations can give it more time to converge. Previously we only trained with 100 iterations.




```python
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def grid_search_report(model, params, X, y, cv=3):

    grid_search = GridSearchCV(model, params, cv=cv, scoring='recall_macro', return_train_score=True, verbose=1, refit=False, n_jobs=-1)
    grid_search.fit(X, y)

    results = pd.DataFrame(grid_search.cv_results_)
    keep_cols = ['param_' + p for p in params[0]]
    keep_cols += ['mean_train_score', 'mean_test_score', 'std_test_score']
    results = pd.DataFrame(grid_search.cv_results_).sort_values(by='mean_test_score', ascending=False).reset_index()
    results = results[keep_cols]

    def custom_style(row):
        color = 'white'
        if row.name == 0:
            color = 'yellow'
        return ['background-color: %s' % color]*len(row.values)
    results = results.head(10).style.apply(custom_style, axis=1)

    return results

grid_search_report(
    model=Pipeline([
        ('standardization', StandardScaler()),
        ('cls', LogisticRegression(multi_class='multinomial', solver='lbfgs', class_weight='balanced'))
    ]),
    params=[{
        'cls__C': [None, 1.0, 0.1, 0.01, 0.001, 0.0001],
    }],
    X=tweets_train_bert,
    y=tweets_train['label'].cat.codes,
)
```

    Fitting 3 folds for each of 6 candidates, totalling 18 fits


    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 2 concurrent workers.
    [Parallel(n_jobs=-1)]: Done  18 out of  18 | elapsed:  2.3min finished





<style  type="text/css" >
    #T_68a23bc8_cf53_11ea_9194_0242ac1c0002row0_col0 {
            background-color:  yellow;
        }    #T_68a23bc8_cf53_11ea_9194_0242ac1c0002row0_col1 {
            background-color:  yellow;
        }    #T_68a23bc8_cf53_11ea_9194_0242ac1c0002row0_col2 {
            background-color:  yellow;
        }    #T_68a23bc8_cf53_11ea_9194_0242ac1c0002row0_col3 {
            background-color:  yellow;
        }    #T_68a23bc8_cf53_11ea_9194_0242ac1c0002row1_col0 {
            background-color:  white;
        }    #T_68a23bc8_cf53_11ea_9194_0242ac1c0002row1_col1 {
            background-color:  white;
        }    #T_68a23bc8_cf53_11ea_9194_0242ac1c0002row1_col2 {
            background-color:  white;
        }    #T_68a23bc8_cf53_11ea_9194_0242ac1c0002row1_col3 {
            background-color:  white;
        }    #T_68a23bc8_cf53_11ea_9194_0242ac1c0002row2_col0 {
            background-color:  white;
        }    #T_68a23bc8_cf53_11ea_9194_0242ac1c0002row2_col1 {
            background-color:  white;
        }    #T_68a23bc8_cf53_11ea_9194_0242ac1c0002row2_col2 {
            background-color:  white;
        }    #T_68a23bc8_cf53_11ea_9194_0242ac1c0002row2_col3 {
            background-color:  white;
        }    #T_68a23bc8_cf53_11ea_9194_0242ac1c0002row3_col0 {
            background-color:  white;
        }    #T_68a23bc8_cf53_11ea_9194_0242ac1c0002row3_col1 {
            background-color:  white;
        }    #T_68a23bc8_cf53_11ea_9194_0242ac1c0002row3_col2 {
            background-color:  white;
        }    #T_68a23bc8_cf53_11ea_9194_0242ac1c0002row3_col3 {
            background-color:  white;
        }    #T_68a23bc8_cf53_11ea_9194_0242ac1c0002row4_col0 {
            background-color:  white;
        }    #T_68a23bc8_cf53_11ea_9194_0242ac1c0002row4_col1 {
            background-color:  white;
        }    #T_68a23bc8_cf53_11ea_9194_0242ac1c0002row4_col2 {
            background-color:  white;
        }    #T_68a23bc8_cf53_11ea_9194_0242ac1c0002row4_col3 {
            background-color:  white;
        }    #T_68a23bc8_cf53_11ea_9194_0242ac1c0002row5_col0 {
            background-color:  white;
        }    #T_68a23bc8_cf53_11ea_9194_0242ac1c0002row5_col1 {
            background-color:  white;
        }    #T_68a23bc8_cf53_11ea_9194_0242ac1c0002row5_col2 {
            background-color:  white;
        }    #T_68a23bc8_cf53_11ea_9194_0242ac1c0002row5_col3 {
            background-color:  white;
        }</style><table id="T_68a23bc8_cf53_11ea_9194_0242ac1c0002" ><thead>    <tr>        <th class="blank level0" ></th>        <th class="col_heading level0 col0" >param_cls__C</th>        <th class="col_heading level0 col1" >mean_train_score</th>        <th class="col_heading level0 col2" >mean_test_score</th>        <th class="col_heading level0 col3" >std_test_score</th>    </tr></thead><tbody>
                <tr>
                        <th id="T_68a23bc8_cf53_11ea_9194_0242ac1c0002level0_row0" class="row_heading level0 row0" >0</th>
                        <td id="T_68a23bc8_cf53_11ea_9194_0242ac1c0002row0_col0" class="data row0 col0" >0.001000</td>
                        <td id="T_68a23bc8_cf53_11ea_9194_0242ac1c0002row0_col1" class="data row0 col1" >0.696250</td>
                        <td id="T_68a23bc8_cf53_11ea_9194_0242ac1c0002row0_col2" class="data row0 col2" >0.676795</td>
                        <td id="T_68a23bc8_cf53_11ea_9194_0242ac1c0002row0_col3" class="data row0 col3" >0.002522</td>
            </tr>
            <tr>
                        <th id="T_68a23bc8_cf53_11ea_9194_0242ac1c0002level0_row1" class="row_heading level0 row1" >1</th>
                        <td id="T_68a23bc8_cf53_11ea_9194_0242ac1c0002row1_col0" class="data row1 col0" >0.010000</td>
                        <td id="T_68a23bc8_cf53_11ea_9194_0242ac1c0002row1_col1" class="data row1 col1" >0.702407</td>
                        <td id="T_68a23bc8_cf53_11ea_9194_0242ac1c0002row1_col2" class="data row1 col2" >0.673495</td>
                        <td id="T_68a23bc8_cf53_11ea_9194_0242ac1c0002row1_col3" class="data row1 col3" >0.002558</td>
            </tr>
            <tr>
                        <th id="T_68a23bc8_cf53_11ea_9194_0242ac1c0002level0_row2" class="row_heading level0 row2" >2</th>
                        <td id="T_68a23bc8_cf53_11ea_9194_0242ac1c0002row2_col0" class="data row2 col0" >0.000100</td>
                        <td id="T_68a23bc8_cf53_11ea_9194_0242ac1c0002row2_col1" class="data row2 col1" >0.679654</td>
                        <td id="T_68a23bc8_cf53_11ea_9194_0242ac1c0002row2_col2" class="data row2 col2" >0.670958</td>
                        <td id="T_68a23bc8_cf53_11ea_9194_0242ac1c0002row2_col3" class="data row2 col3" >0.003752</td>
            </tr>
            <tr>
                        <th id="T_68a23bc8_cf53_11ea_9194_0242ac1c0002level0_row3" class="row_heading level0 row3" >3</th>
                        <td id="T_68a23bc8_cf53_11ea_9194_0242ac1c0002row3_col0" class="data row3 col0" >1.000000</td>
                        <td id="T_68a23bc8_cf53_11ea_9194_0242ac1c0002row3_col1" class="data row3 col1" >0.702548</td>
                        <td id="T_68a23bc8_cf53_11ea_9194_0242ac1c0002row3_col2" class="data row3 col2" >0.670369</td>
                        <td id="T_68a23bc8_cf53_11ea_9194_0242ac1c0002row3_col3" class="data row3 col3" >0.002391</td>
            </tr>
            <tr>
                        <th id="T_68a23bc8_cf53_11ea_9194_0242ac1c0002level0_row4" class="row_heading level0 row4" >4</th>
                        <td id="T_68a23bc8_cf53_11ea_9194_0242ac1c0002row4_col0" class="data row4 col0" >0.100000</td>
                        <td id="T_68a23bc8_cf53_11ea_9194_0242ac1c0002row4_col1" class="data row4 col1" >0.702850</td>
                        <td id="T_68a23bc8_cf53_11ea_9194_0242ac1c0002row4_col2" class="data row4 col2" >0.670230</td>
                        <td id="T_68a23bc8_cf53_11ea_9194_0242ac1c0002row4_col3" class="data row4 col3" >0.002478</td>
            </tr>

    </tbody></table>



We gain a bit of extra performance through this ultimate fine-tuning. We can now retrain our final model on the full dataset with the optimal hyper-parameters before evaluating on the test set.


```python
final_model = Pipeline([
    ('standardization', StandardScaler()),
    ('cls', LogisticRegression(multi_class='multinomial', solver='lbfgs', class_weight='balanced', C=0.001, max_iter=1000))
])

final_model.fit(tweets_train_bert, tweets_train_y)
```




    Pipeline(memory=None,
             steps=[('standardization',
                     StandardScaler(copy=True, with_mean=True, with_std=True)),
                    ('cls',
                     LogisticRegression(C=0.001, class_weight='balanced',
                                        dual=False, fit_intercept=True,
                                        intercept_scaling=1, l1_ratio=None,
                                        max_iter=1000, multi_class='multinomial',
                                        n_jobs=None, penalty='l2',
                                        random_state=None, solver='lbfgs',
                                        tol=0.0001, verbose=0, warm_start=False))],
             verbose=False)



# Estimating the Generalization Error

Now it's time to measure our final error on the official test set. We will need to convert the test messages in the same way as the training data, transforming words into sentence-level BERT embeddings:


```python
tweets_test = load_dataframe('Subtask_A/gold/SemEval2017-task4-test.subtask-A.english.txt')
tweets_test['message'] = preprocess_messages(tweets_test['message'])
tweets_test_bert = BertEmbeddings(device='cuda').transform(tweets_test['message'].to_numpy())
tweets_test_y = tweets_test['label'].map(lambda label: labels_codes[label])

print('\nTotal number of examples for testing: {}\nDistribution of classes:\n{}'.format(
    len(tweets_test),
    tweets_test['label'].value_counts() / len(tweets_test),
))
```

    100%|██████████| 384/384 [00:36<00:00, 10.61it/s]


    Total number of examples for testing: 12284
    Distribution of classes:
    neutral     0.483312
    negative    0.323347
    positive    0.193341
    Name: label, dtype: float64






```python
tweets_test_predictions = final_model.predict(tweets_test_bert)
print(metrics.classification_report(tweets_test_y, tweets_test_predictions, target_names=labels))
```

                  precision    recall  f1-score   support

        negative       0.58      0.81      0.68      3972
         neutral       0.70      0.57      0.63      5937
        positive       0.65      0.55      0.60      2375

        accuracy                           0.64     12284
       macro avg       0.65      0.64      0.63     12284
    weighted avg       0.66      0.64      0.64     12284



Our final score is **64% macro recall** and **65% accuracy**. Notice how much sensitive our model is towards the negative class, where it achieves a high recall of 81%. This is due to how we trained our logistic regression. By specifying `class_weight='balanced'`, we told Scikit-learn to pay much more attention to errors on the negative class. Because there were less negative  examples in the training data, the model had to compensate by adding bias for predicting this class more often.

You may notice that our final model's macro recall is not as high as the best score from the official SemEval competition (68.1%). This is because we only used a **simple logistic regression classifier** to keep things simple and practical for this introduction.

However there are many improvements possible over our current model such as:
* Adding depth to our classifier by turning our logistic regression model into a neural network with more than one layer.
* Using a recurrent neural network (eg. LSTM) to take advantage of individual word embeddings instead of averaging into a message-level representation.
* Fine-tuning the entire BERT model directly on our dataset instead of extracting embeddings to use in a separate logistic regression model.

Although we obtained the best performance using BERT, the bags of words classifier is probably the most practical option if you are looking to easily deploy an API to serve a web application, thanks to its lightness. If you have access to more resources you could also deploy BERT encodings for extra-performance. You might also want to look at smaller and faster versions of BERT like [DistilBERT](https://arxiv.org/abs/1910.01108) to speed things up.
