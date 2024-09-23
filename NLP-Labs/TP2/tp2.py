import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from joblib import dump, load
import pickle
from gensim.models import Word2Vec, FastText, KeyedVectors
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
nltk.download('punkt')


df = pd.read_csv('spooky_cleaned.csv')
df.head()

df.dropna(inplace=True)

# # B. Encoding of the Target Variable

#Encode the labels using an encoding technique.
le = LabelEncoder()
df['author'] = le.fit_transform(df['author'])
df.head()


# # C. Construction of Training and Testing Sets

X = df['stemmed_text']
y = df['author']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

# # D. Vectorization Methods
# 

# 1. Use the lexical frequency method and one-hot encoding to vectorize the training and testing datasets.
cv = CountVectorizer()
X_train_cv = cv.fit_transform(X_train)
X_test_cv = cv.transform(X_test)

# 2. Train a TF-IDF vectorization model on the training part and vectorize it.
tfidf = TfidfVectorizer()
X_train_tfidf = tfidf.fit_transform(X_train)
X_train_tfidf = tfidf.transform(X_train)
# 3. Using the same model, vectorize the testing part.
X_test_tfidf = tfidf.transform(X_test)

# # E. Training

model1 = MLPClassifier(hidden_layer_sizes=(32, 64), max_iter=1, activation='relu')
model2 = MLPClassifier(hidden_layer_sizes=(32, 64), max_iter=1, activation='logistic')
model3 = MLPClassifier(hidden_layer_sizes=(32, 64), max_iter=1, activation='tanh')

def train(vectorizer, model, epochs = 10):
    X = None
    y = None

    if vectorizer == 'cv':
        X = X_train_cv
        y = y_train
    elif vectorizer == 'tfidf':
        X = X_train_tfidf
        y = y_train
    else:
        raise ValueError("Invalid vectorizer")
    

    losses = []
    prev_accuracy = None 

    for epoch in range(epochs):
        model.partial_fit(X, y, classes=np.unique(y_train))

        y_train_pred = model.predict(X)

        accuracy = accuracy_score(y, y_train_pred)
        precision = precision_score(y, y_train_pred, average='weighted')
        recall = recall_score(y, y_train_pred, average='weighted')
        f1 = f1_score(y, y_train_pred, average='weighted')

        print(f"Epoch {epoch+1}/{epochs} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")
        
        if prev_accuracy is not None:
            loss = accuracy - prev_accuracy
            losses.append(loss)
        prev_accuracy = accuracy
    
    dump(model, 'models/'+f"{model.activation}_{vectorizer}.joblib")

    return losses

vectorizers = ['cv', 'tfidf']
models = [model1, model2, model3]
losses = {}

for model in models:
    for vectorizer in vectorizers:
        print(f"Training model with {model.activation} activation functions and {vectorizer} vectorizer")
        loss = train(vectorizer, model, epochs=50)
        losses[f"{model.activation}_{vectorizer}"] = loss
        

#plot the losses each in a different figure
for key, loss in losses.items():
    plt.figure()
    plt.plot(loss)
    plt.title(key)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()


# # F. Testing

#load the models from models folder and display the classification report for each model
X = None
#create a dataframe to store the accuracy, precision, recall, f1-score for each model
evaluation = pd.DataFrame(columns=['Model', 'Vectorizer', 'Accuracy', 'Precision', 'Recall', 'F1-score'])

for model in models:
    for vectorizer in vectorizers:
        print(f"Model with {model.activation} activation function and {vectorizer} vectorizer")
        path = 'models/'+f'{model.activation}_{vectorizer}.joblib'
        clf = load(path)
        print(clf)
        if vectorizer == 'cv':
            X = X_test_cv
        elif vectorizer == 'tfidf':
            X = X_test_tfidf

        y_pred = clf.predict(X)
        print(classification_report(y_test, y_pred, target_names=le.classes_))
        print('---------------------------------------------------')
        #save accuracy, precision, recall, f1-score in a dataframe
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        evaluation = evaluation.append({'Model': model.activation, 'Vectorizer': vectorizer, 'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1-score': f1}, ignore_index=True)

evaluation

# # G. Word Embedding

def vectorize(vectors, words):
    words = [word for word in words if word in vectors]
    if len(words) == 0:
        return np.zeros(50)
    return np.mean(vectors[words], axis=0)


# use Word2Vec to vectorize the training and testing datasets. (COW)
X_train_w2v_cbow = [word_tokenize(text) for text in X_train]
X_test_w2v_cbow = [word_tokenize(text) for text in X_test]

vectors = Word2Vec(X_train_w2v_cbow, vector_size=50, window=5, min_count=1, workers=4)

X_train_w2v_cbow = [vectorize(vectors.wv, sentence) for sentence in X_train_w2v_cbow]
X_test_w2v_cbow = [vectorize(vectors.wv, sentence) for sentence in X_test_w2v_cbow]

X_train_w2v_sg = [word_tokenize(text) for text in X_train]
X_test_w2v_sg = [word_tokenize(text) for text in X_test]

vectors = Word2Vec(X_train_w2v_sg, vector_size=50, window=5, min_count=1, workers=4, sg=1)

X_train_w2v_sg = [vectorize(vectors.wv, sentence) for sentence in X_train_w2v_sg]
X_test_w2v_sg = [vectorize(vectors.wv, sentence) for sentence in X_test_w2v_sg]

#use FastText
X_train_ft = [word_tokenize(text) for text in X_train]
X_test_ft = [word_tokenize(text) for text in X_test]

model = FastText(X_train_ft, vector_size=50, window=5, min_count=1, workers=4)

X_train_ft = [vectorize(model.wv, sentence) for sentence in X_train_ft]
X_test_ft = [vectorize(model.wv, sentence) for sentence in X_test_ft]

#use glove
X_train_gl = [word_tokenize(text) for text in X_train]
X_test_gl = [word_tokenize(text) for text in X_test]

glove = KeyedVectors.load_word2vec_format('glove.6B.50d.txt', binary=False, no_header=True)

X_train_glove = [vectorize(glove, sentence) for sentence in X_train_gl]
X_test_glove = [vectorize(glove, sentence) for sentence in X_test_gl]


def train(vectorizer, model, epochs = 10):
    X = None
    y = y_train

    if vectorizer == 'cv':
        X = X_train_cv
    elif vectorizer == 'tfidf':
        X = X_train_tfidf
    elif vectorizer == 'cbow':
        X = X_train_w2v_cbow
    elif vectorizer == 'sg':
        X = X_train_w2v_sg
    elif vectorizer == 'ft':
        X = X_train_ft
    elif vectorizer == 'gl':
        X = X_train_glove
    else:
        raise ValueError("Invalid vectorizer")
    

    losses = []
    prev_accuracy = None 

    for epoch in range(epochs):
        model.partial_fit(X, y, classes=np.unique(y_train))

        y_train_pred = model.predict(X)

        accuracy = accuracy_score(y, y_train_pred)
        precision = precision_score(y, y_train_pred, average='weighted')
        recall = recall_score(y, y_train_pred, average='weighted')
        f1 = f1_score(y, y_train_pred, average='weighted')

        print(f"Epoch {epoch+1}/{epochs} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")
        
        if prev_accuracy is not None:
            loss = accuracy - prev_accuracy
            losses.append(loss)
        prev_accuracy = accuracy
    
    dump(model, 'models/'+f"{model.activation}_{vectorizer}.joblib")

    return losses

model1 = MLPClassifier(hidden_layer_sizes=(32, 64), max_iter=1, activation='relu')
model2 = MLPClassifier(hidden_layer_sizes=(32, 64), max_iter=1, activation='logistic')
model3 = MLPClassifier(hidden_layer_sizes=(32, 64), max_iter=1, activation='tanh')

models = [model1, model2, model3]
vectorizers = ['cbow', 'sg', 'ft', 'gl']

for model in models:
    for vectorizer in vectorizers:
        print(f"Training model with {model.activation} hidden layers and {vectorizer} vectorizer")
        loss = train(vectorizer, model, 100)
        losses[f"{model.activation}_{vectorizer}"] = loss

#plot the losses each in a different figure
for key, loss in losses.items():
    plt.figure()
    plt.plot(loss)
    plt.title(key)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()

#load the models from models folder and display the classification report for each model
X = None
#create a dataframe to store the accuracy, precision, recall, f1-score for each model
evaluation = pd.DataFrame(columns=['Model', 'Vectorizer', 'Accuracy', 'Precision', 'Recall', 'F1-score'])

for model in models:
    for vectorizer in vectorizers:
        print(f"Model with {model.activation} hidden layers and {vectorizer} vectorizer")
        path = 'models/'+f'{model.activation}_{vectorizer}.joblib'
        clf = load(path)
        print(clf)
        if vectorizer == 'cbow':
            X = X_test_w2v_cbow
        elif vectorizer == 'sg':
            X = X_test_w2v_sg
        elif vectorizer == 'ft':
            X = X_test_ft
        elif vectorizer == 'gl':
            X = X_test_glove
        else:
            raise ValueError("Invalid vectorizer")

        y_pred = clf.predict(X)
        #save accuracy, precision, recall, f1-score in a dataframe
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        evaluation = evaluation.append({'Model': model.activation, 'Vectorizer': vectorizer, 'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1-score': f1}, ignore_index=True)

evaluation




