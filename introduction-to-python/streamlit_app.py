import streamlit as st
import pandas as pd
import time
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm

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



####################################################

# Streamlit Inputs for Univariate Gaussian parameters
st.title('Univariate Gaussian Sample Generator with Empirical and Theoretical Density')

# Input number of samples (n)
n_samples = st.number_input('Number of samples (n)', value=100)

# Input mean (mu) and variance (sigma^2)
mean = st.number_input('Mean (μ)', value=0.0)
variance = st.number_input('Variance (σ²)', value=1.0)
std_dev = np.sqrt(variance)  # Standard deviation

# Step 2: Generate the Univariate Gaussian samples
samples = np.random.normal(loc=mean, scale=std_dev, size=n_samples)

# Step 3: Create bins for histogram and calculate the theoretical density curve
x_vals = np.linspace(np.min(samples) - 1, np.max(samples) + 1, 1000)
theoretical_density = norm.pdf(x_vals, mean, std_dev)

# Step 4: Plot the histogram (empirical density) and overlay the theoretical density curve
fig = go.Figure()

# Histogram of the samples (empirical density)
fig.add_trace(go.Histogram(x=samples, nbinsx=50, histnorm='probability density', 
                           name='Empirical Density', marker_color='blue', opacity=0.6))

# Theoretical density curve
fig.add_trace(go.Scatter(x=x_vals, y=theoretical_density, mode='lines', 
                         name='Theoretical Density', line=dict(color='red', width=2)))

# Update layout for better visibility
fig.update_layout(
    title="Univariate Gaussian Distribution: Empirical vs Theoretical Density",
    xaxis_title="X",
    yaxis_title="Density",
    showlegend=True,
    autosize=False,
    width=800, height=500
)

# Step 5: Display the plot in Streamlit
st.plotly_chart(fig)

# Display additional information
st.write(f"Generated {n_samples} samples from a Univariate Gaussian distribution with mean {mean} and variance {variance} (σ = {std_dev:.2f}).")
