import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble._forest import RandomForestClassifier
from sklearn.metrics import plot_confusion_matrix, plot_precision_recall_curve, plot_roc_curve 

#to load the mushroom dataset
@st.cache(persist=True)
def load_mushroom_data():
    df = pd.read_csv("datasets/mushroom dataset binary classification.csv")
    labelencoder = LabelEncoder()
    for col in df.columns:
        df[col] = labelencoder.fit_transform(df[col])
    return df

# train test split
@st.cache(persist=True)
def split(df):
    y = df["type"]
    X = df.drop(columns = ["type"])
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2,random_state = 0)
    return X_train, X_test, y_train, y_test

def plot_metrics(metrics_list,model,X_test,y_test,labels):
    if "Confusion matrix" in metrics_list:
        st.subheader("Confusion Matrix")
        plot_confusion_matrix(model,X_test,y_test,display_labels=labels)
        st.pyplot()
    if "Precision Recall Curve" in metrics_list:
        st.subheader("Precision Recall Curve")
        plot_precision_recall_curve(model,X_test,y_test)
        st.pyplot()
    if "ROC Curve" in metrics_list:
        st.subheader("ROC Curve")
        plot_roc_curve(model,X_test,y_test)
        st.pyplot()

if __name__ == "__main__":
    st.title("Binary Classification Web App")
    st.write("Are our mushrooms edible or poisonous? 🍄")
    st.sidebar.title("Binary Classification")
    st.sidebar.write("Are our mushrooms edible or poisonous? 🍄")
    df = load_mushroom_data()
    classes = ["edible","poisonous"]
    X_train, X_test, y_train, y_test = split(df)
    st.sidebar.subheader("Choose your classifier")
    classifier = st.sidebar.selectbox("Which ML classifier do u want the dataset to be trained on?",("Logistic Regression","Support Vector Classifier(SVC)","Decision Tree Classifier","Random Forest Classifier"))

    if classifier == "Logistic Regression":
        st.sidebar.write("Choose Hyperparameters for your model")
        C = st.sidebar.number_input("Inverse of Regularization Strength C",0.0,10.0,1.0,0.1)
        max_iter = st.sidebar.number_input("Maximum number of iterations to converge max_iter",1,500,100,10)
        metrics = st.sidebar.multiselect("What all metrics to plot?",
                            ("Confusion matrix","Precision Recall Curve","ROC Curve"))
        if st.sidebar.button("Classify"):
            st.subheader("Logistic Regression Results")
            model = LogisticRegression(C=C,max_iter=max_iter)
            model.fit(X_train,y_train)
            accuracy = model.score(X_test,y_test)
            y_predicted = model.predict(X_test)
            st.write("Accuracy: ", accuracy.round(2))
            st.write("Precision: ", precision_score(y_test,y_predicted,classes))
            st.write("Recall: ", recall_score(y_test,y_predicted,classes))
            plot_metrics(metrics,model,X_test,y_test,classes)
            #st.write(classification_report(y_test,y_predicted,labels = classes))
    
    if classifier == "Support Vector Classifier(SVC)":
        st.sidebar.write("Choose Hyperparameters for your model")
        C = st.sidebar.number_input("Inverse of Regularization Strength C",0.0,10.0,1.0,0.1)
        kernel = st.sidebar.selectbox("Kernel Type",("rbf","linear","poly","sigmoid"))
        gamma = st.sidebar.selectbox("Kernal coefficient gamma",("scale","auto"))
        metrics = st.sidebar.multiselect("What all metrics to plot?",
                            ("Confusion matrix","Precision Recall Curve","ROC Curve"))
        if st.sidebar.button("Classify"):
            st.subheader("Support Vector Classifier Results")
            model = SVC(C=C,kernel=kernel,gamma=gamma)
            model.fit(X_train,y_train)
            accuracy = model.score(X_test,y_test)
            y_predicted = model.predict(X_test)
            st.write("Accuracy: ", accuracy.round(2))
            st.write("Precision: ", precision_score(y_test,y_predicted,classes))
            st.write("Recall: ", recall_score(y_test,y_predicted,classes))
            plot_metrics(metrics,model,X_test,y_test,classes)

    if classifier == "Decision Tree Classifier":
        st.sidebar.write("Choose Hyperparameters for your model")
        criterion = st.sidebar.selectbox("criterion",("gini","entropy"))
        max_depth = st.sidebar.number_input("max_depth of the tree",1,500,10,10)
        metrics = st.sidebar.multiselect("What all metrics to plot?",
                            ("Confusion matrix","Precision Recall Curve","ROC Curve"))
        if st.sidebar.button("Classify"):
            st.subheader("Decision Tree Classifier Results")
            model = DecisionTreeClassifier(criterion=criterion,max_depth=max_depth)
            model.fit(X_train,y_train)
            accuracy = model.score(X_test,y_test)
            y_predicted = model.predict(X_test)
            st.write("Accuracy: ", accuracy.round(2))
            st.write("Precision: ", precision_score(y_test,y_predicted,classes))
            st.write("Recall: ", recall_score(y_test,y_predicted,classes))
            plot_metrics(metrics,model,X_test,y_test,classes)

    if classifier == "Random Forest Classifier":
        st.sidebar.write("Choose Hyperparameters for your model")
        n_estimators = st.sidebar.number_input("number of trees in the forest",1,500,10,10)
        criterion = st.sidebar.selectbox("criterion",("gini","entropy"))
        max_depth = st.sidebar.number_input("max_depth of the trees",1,500,10,10)
        bootstrap = st.sidebar.selectbox("bootstrap",(True,False))
        metrics = st.sidebar.multiselect("What all metrics to plot?",
                            ("Confusion matrix","Precision Recall Curve","ROC Curve"))
        if st.sidebar.button("Classify"):
            st.subheader("Random Forest Classifier Results")
            model = RandomForestClassifier(n_estimators=n_estimators,criterion=criterion,max_depth=max_depth,bootstrap=bootstrap)
            model.fit(X_train,y_train)
            accuracy = model.score(X_test,y_test)
            y_predicted = model.predict(X_test)
            st.write("Accuracy: ", accuracy.round(2))
            st.write("Precision: ", precision_score(y_test,y_predicted,classes))
            st.write("Recall: ", recall_score(y_test,y_predicted,classes))
            plot_metrics(metrics,model,X_test,y_test,classes)

    if st.checkbox("Show dataset", False):
        st.subheader("Mushroom Dataset")
        st.dataframe(df)
        st.markdown("This [data set](https://archive.ics.uci.edu/ml/datasets/Mushroom) includes "  "descriptions of hypothetical samples corresponding to 23 species of gilled mushrooms "
        "in the Agaricus and Lepiota Family (pp. 500-525). Each species is identified as definitely edible, definitely poisonous, "
        "or of unknown edibility and not recommended. This latter class was combined with the poisonous one.")


