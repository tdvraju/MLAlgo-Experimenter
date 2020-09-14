import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score

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

def main():
    st.title("Binary Classification Web App")
    st.write("Are our mushrooms edible or poisonous? 🍄")
    st.sidebar.title("Binary Classification")
    st.sidebar.write("Are our mushrooms edible or poisonous? 🍄")
    df = load_mushroom_data()
    classes = ["edible","poisonous"]
    X_train, X_test, y_train, y_test = split(df)
    st.sidebar.subheader("Choose your classifier")
    classifier = st.sidebar.selectbox("Which ML classifier do u want the dataset to be trained on?",("Logistic Regression","Support Vector Classifier(SVC)"))

    if classifier == "Logistic Regression":
        st.sidebar.write("Choose Hyperparameters for your model")
        C = st.sidebar.number_input("Inverse of Regularization Strength C",0.0,10.0,1.0,0.1)
        max_iter = st.sidebar.number_input("Maximum number of iterations to converge max_iter",1,500,100,10)
        if st.sidebar.button("Classify"):
            st.subheader("Logistic Regression Results")
            model = LogisticRegression(C=C,max_iter=max_iter)
            model.fit(X_train,y_train)
            accuracy = model.score(X_test,y_test)
            y_predicted = model.predict(X_test)
            st.write("Accuracy: ", accuracy.round(2))
            st.write("Precision: ", precision_score(y_test,y_predicted,classes))
            st.write("Recall: ", recall_score(y_test,y_predicted,classes))


if __name__ == "__main__":
    main()
