from flask import Flask
from flask import jsonify
from flask import request
from sklearn import tree
import pickle
import pydotplus 

app = Flask(__name__)

X = [	
		[1, 10, 120],
		[1, 12, 150],
		[0, 3, 100],
		[0, 15, 150],
		[1, 4, 90],
		[1, 11, 130],
		[1, 12, 160],
		[0, 4, 100],
		[0, 14, 140], 
		[1, 5, 80]
	]

Y = ['yes', 'yes', 'no', 'yes', 'no', 'yes', 'yes', 'no', 
	'yes', 'no']

@app.route('/train',methods=['GET'])
def startTraining():
	clf = tree.DecisionTreeClassifier()

	clf.fit(X,Y)

	#storing the classifier
	with open('model.pickle', 'wb') as f:
		pickle.dump(clf, f)
	return jsonify({'response':'success'})

@app.route('/test',methods=['GET'])
def startTest():
	pickle_in = open('model.pickle', 'rb')
	clf = pickle.load(pickle_in)

	prediction = clf.predict([[1,6,170]])
	print(prediction)
	return jsonify({'response':'success', 'result':prediction[0]})

@app.route('/pdf',methods=['GET'])
def generatePdf():	
	pickle_in = open('model.pickle', 'rb')
	clf = pickle.load(pickle_in)

	dot_data = tree.export_graphviz(clf, out_file=None)  
	graph = pydotplus.graph_from_dot_data(dot_data) 
	
	return graph.write_pdf("model.pdf") 

if __name__ == '__main__':
	app.run()