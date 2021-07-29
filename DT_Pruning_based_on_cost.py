import sklearn
sklearn.__version__


%matplotlib inline

print(__doc__)
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier

X, y = load_breast_cancer(return_X_y=True)


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

X_train.shape, X_test.shape, y_train.shape, y_test.shape

clf = DecisionTreeClassifier(random_state=0)
clf.fit(X_train,y_train)

tpred=clf.predict(X_train)
from sklearn.metrics import accuracy_score
print("Training set accuracy:",accuracy_score(y_train, tpred))

pred=clf.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy_score(y_test, pred)

from sklearn import tree
plt.figure(figsize=(15,10))
tree.plot_tree(clf,filled=True)


path = clf.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities

clfs = []
for ccp_alpha in ccp_alphas:
    clf = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
    clf.fit(X_train, y_train)
    print("Alpha",ccp_alpha)
    print("Node count",clf.tree_.node_count,"Leaf Count",clf.get_n_leaves())
    clfs.append(clf)
print("Number of nodes in the last tree is: {} with ccp_alpha: {}".format(
      clfs[-1].tree_.node_count, ccp_alphas[-1]))
      
train_scores = []
for clf in clfs:
  train_scores.append(clf.score(X_train, y_train))
  
train_scores = [clf.score(X_train, y_train) for clf in clfs]
test_scores = [clf.score(X_test, y_test) for clf in clfs]

fig, ax = plt.subplots()
ax.set_xlabel("alpha")
ax.set_ylabel("accuracy")
ax.set_title("Accuracy vs alpha for training and testing sets")
ax.plot(ccp_alphas, train_scores, marker='o', label="train",
        drawstyle="steps-post")
ax.plot(ccp_alphas, test_scores, marker='o', label="test",
        drawstyle="steps-post")
ax.legend()
plt.show()

clf = DecisionTreeClassifier(random_state=0, ccp_alpha=0.012)
clf.fit(X_train,y_train)


pred=clf.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy_score(y_test, pred)

from sklearn import tree
plt.figure(figsize=(15,10))
tree.plot_tree(clf,filled=True)

