from sklearn import tree
from sklearn.svm import SVC
from sklearn.linear_model import Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np

# data and labels
# [height, weight, shoe_size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']

#classifiers 
clf_tree = tree.DecisionTreeClassifier()
clf_svm=SVC()
clf_per=Perceptron()
clf_KNN=KNeighborsClassifier()


#training the models
clf_tree.fit(X, Y)
clf_svm.fit(X, Y)
clf_per.fit(X, Y)
clf_KNN.fit(X, Y)

#testing the models using same data

predict_tree = clf_tree.predict(X)
accuracy_tree=accuracy_score(Y,predict_tree)*100
print("Accuracy for dicision tree: {}".format(accuracy_tree))

predict_KNN=clf_KNN.predict(X)
accuracy_KNN=accuracy_score(Y,predict_KNN)*100
print("Accuracy for KNN: {}".format(accuracy_KNN))


predict_svm=clf_svm.predict(X)
accuracy_svm=accuracy_score(Y,predict_svm)*100
print("Accuracy for SVM: {}".format(accuracy_svm))

predict_per=clf_per.predict(X)
accuracy_per=accuracy_score(Y,predict_per)*100
print("Accuracy for Perceptron: {}".format(accuracy_per))


#best classifier
idx=np.argmax([accuracy_svm,accuracy_KNN,accuracy_per])
c={0:'SVM',1:'KNN',2:'Perceptron'}
print("Best Gender Classifier is {}".format(c[idx]))

