from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

classifier = RandomForestClassifier()
dataset_df=pd.read_pickle("dataset_img_hough5cross_fullsize.pkl").dropna(axis=1)
print(dataset_df)
X=dataset_df.drop(['target', 'filename'], axis=1)
y=dataset_df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=11, stratify=y)

# Aprenda a clasificar el conjunto de entrenamiento
classifier.fit(X_train, y_train)
y_predicted = classifier.predict(X_test)
conf_mat=metrics.confusion_matrix(y_test, y_predicted)
print(conf_mat)
print("Classification report for classifier {}\n{}\n".format(classifier, metrics.classification_report(y_test, y_predicted)))
