from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import warnings

warnings.filterwarnings('ignore')

news = pd.read_csv('./datasets/fake_or_real_news.csv')

x_train, x_test, y_train, y_test = train_test_split(news['text'], news['label'], test_size=0.2)

tfidf = TfidfVectorizer(stop_words='english', max_df=0.8)
x_train = tfidf.fit_transform(x_train)
x_test = tfidf.transform(x_test)

models = [PassiveAggressiveClassifier(max_iter=300), DecisionTreeClassifier(), RandomForestClassifier(),
          SVC(), LogisticRegression()]
labels = ['PassiveAggressiveClassifier', 'DecisionTreeClassifier', 'RandomForestClassifier',
          'SVC', 'LogisticRegression']
accuracies = []

for model in models:
    model.fit(x_train, y_train)
    prediction = model.predict(x_test)
    accuracies.append(accuracy_score(y_test, prediction))

for i in range(len(models)):
    print("Model: %s, accuracy: %f" % (labels[i], accuracies[i]))

test_data = pd.read_csv('./datasets/fake-news-test-set.csv')
test_accuracies = []
for model in models:
    prediction = model.predict(tfidf.transform(test_data['text']))
    prediction_ = [1 if el == "REAL" else 0 for el in prediction]
    test_accuracies.append(accuracy_score(test_data['true'], prediction_))

print("\n\n\n========\n")
for i in range(len(models)):
    print("Model: %s, accuracy: %f" % (labels[i], test_accuracies[i]))
