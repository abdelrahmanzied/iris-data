#Import
from sklearn.datasets import load_iris
from sklearn .model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns


#Data
X, y = load_iris(return_X_y=True)


#Splitting
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=.33, random_state=18)


#Model
clf = LogisticRegression(random_state=18, solver='saga', max_iter=2000)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
y_pred_prob = clf.predict_proba(X_test)

print('Train Score: ', clf.score(X_train, y_train))
print('Test Score: ', clf.score(X_test, y_test))
print('Number of Itterations: ', clf.n_iter_)
print('Classes: ', clf.classes_)


#Confusion Matrix
CM = confusion_matrix(y_test, y_pred)
print('Confusion Matrix: \n', CM)
sns.heatmap(CM, center=True)


#Accuracy Score
AccScore = accuracy_score(y_test, y_pred)
print('Accuracy Score: ', AccScore)