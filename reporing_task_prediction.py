from sklearn import tree
import pickle
# 1 -> Yes  
# 0 -> No
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

clf = tree.DecisionTreeClassifier()

clf.fit(X,Y)

#storing the classifier
with open('model.pickle', 'wb') as f:
	pickle.dump(clf, f)

pickle_in = open('model.pickle', 'rb')
clf = pickle.load(pickle_in)

prediction = clf.predict([[1,6,170], [1,2,150]])

print (prediction)

#viz code
import pydotplus 
dot_data = tree.export_graphviz(clf, out_file=None)  
graph = pydotplus.graph_from_dot_data(dot_data) 
graph.write_pdf("model.pdf") 
