from flask import Flask
from flask import jsonify
from flask import request
from sklearn import tree
import pandas as pd
import numpy as np
import pickle
import pydotplus 

app = Flask(__name__)

df = pd.read_csv("data/reporting_task_prediction.csv")
df.replace(to_replace='yes', value=1, inplace=True)
df.replace(to_replace='no', value=0, inplace=True)
df.replace(to_replace='good', value=1, inplace=True)
df.replace(to_replace='Not good', value=0, inplace=True)

X = np.array(df.drop(['WillMissReportDueDate'], axis=1))
y = np.array(df['WillMissReportDueDate'])

print(df)

@app.route('/train',methods=['GET'])
def startTraining():
	clf = tree.DecisionTreeClassifier()

	clf.fit(X,y)

	#storing the classifier
	with open('model.pickle', 'wb') as f:
		pickle.dump(clf, f)
	return jsonify({'response':'success'})

@app.route('/test',methods=['POST'])
def startTest():
	print(request.json['IsMissedLastYear'])
	print(request.json['LeavesInMonth'])
	print(request.json['BusyHoursInMonth'])
	print(request.json['PreviousMonth'])
	print(request.json['SystemInfo'])
	y_test = np.array([request.json['IsMissedLastYear'], request.json['LeavesInMonth'], 
		request.json['BusyHoursInMonth'], request.json['PreviousMonth'], request.json['SystemInfo']])

	print(y_test)

	pickle_in = open('model.pickle', 'rb')
	clf = pickle.load(pickle_in)

	prediction = clf.predict([y_test])
	print(prediction)
	return jsonify({'response':'success', 'result':str(prediction[0])})

@app.route('/pdf',methods=['GET'])
def generatePdf():	
	pickle_in = open('model.pickle', 'rb')
	clf = pickle.load(pickle_in)

	dot_data = tree.export_graphviz(clf, out_file=None)  
	graph = pydotplus.graph_from_dot_data(dot_data) 
	
	return graph.write_pdf("model.pdf") 

if __name__ == '__main__':
	app.run()