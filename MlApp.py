import streamlit as st
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

st.title("IRIS FLOWER PREDICTION")
dataset_names = st.sidebar.selectbox("Select dataset", ("Iris","Breast Cancer","Wine Dataset"))
st.write(dataset_names)
classifier_names = st.sidebar.selectbox("Select dataset", ("KNN","SVM","Random Forest"))

def get_dataset(dataset_names):
    if dataset_names == "Iris":
        data = datasets.load_iris()
    elif dataset_names == "Breast Cancer":
        data = datasets.load_breast_cancer()
    else:
        data = datasets.load_wine()
    x = data.data
    y = data.target
    return x,y

x,y = get_dataset(dataset_names)
st.write("shape of dataset", x.shape)
st.write("No of classes", len(np.unique(y)))

def add_parameter_ui(clf_name):
    params = dict()
    if clf_name == "KNN":
        k = st.sidebar.slider("K", 1, 15)
        params["K"] = k
    elif clf_name == "SVM":
        c = st.sidebar.slider("C", 0.01, 10.0)
        params["C"] = c
    else:
        max_depth = st.sidebar.slider("max_depth", 2, 15)
        n_estimators = st.sidebar.slider("n_estimators", 1, 100)
        params["max_depth"] = max_depth
        params["n_estimators"] = n_estimators
    return params

params = add_parameter_ui(classifier_names)

def get_classifier(clf_name, params):
    if clf_name == "KNN":
        clf = KNeighborsClassifier(n_neighbors=params["K"])
    elif clf_name == "SVM":
        clf = SVC(C=params["C"])
    else:
        clf = RandomForestClassifier(n_estimators=params["n_estimators"]
                                     ,max_depth=params["max_depth"], random_state=1234)

    return clf

clf = get_classifier(classifier_names, params)

#Classification
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state= 1234)
clf.fit(x_train,y_train)
y_predict = clf.predict(x_test)

acc = accuracy_score(y_test, y_predict)
st.write(f"classifier = {classifier_names}")
st.write(f"accuracy = {acc}")

#Plotting
pca = PCA(2)
x_projected = pca.fit_transform(x)

x1 = x_projected[:, 0]
x2 = x_projected[:, 1]

fig = plt.figure()
plt.scatter(x1, x2, c=y, alpha = 0.8, cmap = "viridis")
plt.xlabel("Principle Component 1")
plt.ylabel("Principle Component 2")
plt.colorbar()

st.pyplot()








