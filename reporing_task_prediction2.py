from sklearn import tree
import pandas as pd
import numpy as np
import pickle

df = pd.read_csv("data/reporting_task_prediction.csv")
df.replace(to_replace='yes', value=1, inplace=True)
df.replace(to_replace='no', value=0, inplace=True)
df.replace(to_replace='good', value=1, inplace=True)
df.replace(to_replace='Not good', value=0, inplace=True)

X = np.array(df.drop(['WillMissReportDueDate'], axis=1))
y = np.array(df['WillMissReportDueDate'])

clf = tree.DecisionTreeClassifier()

clf.fit(X,y)

#storing the classifier
with open('model.pickle', 'wb') as f:
	pickle.dump(clf, f)

pickle_in = open('model.pickle', 'rb')
clf = pickle.load(pickle_in)

prediction = clf.predict([[1, 11, 130, 1, 0]])

print (prediction)