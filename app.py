import streamlit as st
import numpy as np
import pandas as pd

def main():
    st.title("Binary Classification Web App")
    st.write("Are our mushrooms edible or poisonous? 🍄")
    st.sidebar.title("Binary Classification")
    st.sidebar.write("Are our mushrooms edible or poisonous? 🍄")

if __name__ == "__main__":
    main()
