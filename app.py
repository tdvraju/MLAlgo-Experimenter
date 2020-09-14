import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

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
    

if __name__ == "__main__":
    main()
