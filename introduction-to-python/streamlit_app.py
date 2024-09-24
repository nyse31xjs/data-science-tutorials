import streamlit as st
import pandas as pd
import time

# Title of the app
st.title("Gaussian Vectors")

st.latex(r"f(\mathbf{X}) = \frac{1}{(2\pi)^{n/2} |\boldsymbol{\Sigma}|^{1/2}} \exp\left( -\frac{1}{2} (\mathbf{X} - \boldsymbol{\mu})^\top \boldsymbol{\Sigma}^{-1} (\mathbf{X} - \boldsymbol{\mu}) \right)")

#manage cache (see other function  like st.cache_resource)
@st.cache_data
def load_data():
    return pd.read_csv('/Users/hugorameil/Documents/GitHub/data-science-tutorials/introduction-to-python/train.csv')

data = load_data()


# Getting user input
column = st.text_input("Column name:")

# Displaying the inputs back to the user
if st.button("Show column"):
    st.write(f"latex expression:")
    st.latex(r"e^{i\pi} + 1 = 0")
    st.latex(r"f(\mathbf{X}) = \frac{1}{(2\pi)^{n/2} |\boldsymbol{\Sigma}|^{1/2}} \exp\left( -\frac{1}{2} (\mathbf{X} - \boldsymbol{\mu})^\top \boldsymbol{\Sigma}^{-1} (\mathbf{X} - \boldsymbol{\mu}) \right)")
    st.write(f"Column:")
    st.dataframe(data.filter([column])) 
    st.write(data.filter([column]).describe())

if st.button("Show progress bar"):    
    progress = st.progress(0)
    for i in range(100):
        time.sleep(0.1)
        progress.progress(i+1)

if st.button("Show spinner"):   
    with st.spinner("Loading..."):
        time.sleep(5)
if st.button("Show success message"):
    st.success("Success!")


# Setting the width of the text input
st.text_input("Enter your name:", max_chars=30, key="text", placeholder="Type here")

# Setting the width of a button (via CSS)
st.markdown("""
    <style>
    .stButton button {
        width: 50%;
    }
    </style>
    """, unsafe_allow_html=True)

st.button("Click me")

with st.container():
    st.header("This is a container")
    st.write("Widgets in this container will be grouped together")
    st.button("Container Button")

# Widgets outside the container
st.write("This is outside the container")

with st.expander("Click to expand"):
    st.write("This content is hidden until the user clicks the expander.")
    st.checkbox("Checkbox inside the expander")
    
st.sidebar.header("Sidebar")
st.sidebar.slider("Sidebar Slider", 0, 100, 25)
st.sidebar.button("Sidebar Button")
st.sidebar.button("ML Button")

# Main content
st.title("Main Content Area")
st.write("Widgets and plots in the main area")

st.markdown(
    """
    <style>
    div.stButton > button:first-child {
        background-color: grey;
        color:blue;
        padding: 10px 20px;
        border-radius: 30px;
        height: 2em;
        width: 40%;
    }
    </style>
    """, unsafe_allow_html=True)

st.button("Styled Button")

#objective : create a gaussian vector application 
#1. page : univariate case

#2. page : multivariate case

#1. create a function that generates a gaussian vector
def gaussian_vector(n, mu, sigma):
    return np.random.multivariate_normal(mu, sigma, n)