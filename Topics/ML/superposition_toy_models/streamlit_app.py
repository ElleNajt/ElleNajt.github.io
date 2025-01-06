#!/usr/bin/env python3


import streamlit as st
import torch
import numpy as np
import pandas as pd
from helper_functions import run_experiment, LinearAutoEncoder, ReLUAutoEncoder, SAEAutoEncoder, seed_everything

@st.cache_data
def cached_experiment(model_class, n_dim, h_dim, n_samples, sparsity, importance_base, epochs, learning_rate=0.01):
    fig, df, criterion, X_samples, losses, initial_W = run_experiment(
        model_class,
        n_dim=n_dim,
        h_dim=h_dim,
        n_samples=n_samples,
        sparsity=sparsity,
        importance_base=importance_base,
        epochs=epochs,
        learning_rate=learning_rate
    )
    return fig, df, criterion, X_samples, losses, initial_W

def main():
    st.title("Toy models of superposition visualization")

    # Sidebar parameters
    st.sidebar.header("Parameters")
    n_dim = st.sidebar.slider("Input Dimension", 2, 20, 5)
    h_dim = st.sidebar.slider("Hidden Dimension", 1, 10, 2)
    sparsity_exp = st.sidebar.slider("Sparsity exponent (10^-x)", 0.00, 5.00, 1.00)
    sparsity = 1 - 10**(-sparsity_exp)
    st.sidebar.write(f"Actual sparsity: {sparsity:.6f}")
    importance_base = st.sidebar.slider("Importance Base", 0.00, 1.0, 0.8)
    n_samples = st.sidebar.number_input("Number of Samples", 100, 10000, 5000)
    epochs = st.sidebar.number_input("Epochs", 100, 20000, 2000)
    seed = st.sidebar.number_input("Seed", 1, 100, 42)


    model_type = st.sidebar.selectbox(
        "Model Type",
        ["ReLU", "Linear", "SAE"]
    )

    # Map selection to model class
    model_map = {
        "Linear": LinearAutoEncoder,
        "ReLU": ReLUAutoEncoder,
        "SAE": SAEAutoEncoder
    }

    if st.button("Run Experiment"):
        # Run experiment and display results
        model_class = model_map[model_type]
        seed_everything(seed)
        fig, df, criterion, X_samples, losses, initial_W = cached_experiment(
        model_class,
        n_dim=n_dim,
        h_dim=h_dim,
        n_samples=n_samples,
        sparsity=sparsity,
        importance_base=importance_base,
        epochs=epochs)

        st.pyplot(fig)


        st.write("W matrix (n_dim × h_dim):")
        st.dataframe(df)

        st.write("Loss curve:")
        st.line_chart(losses)

        st.write("Sample data points:")
        st.dataframe(pd.DataFrame(X_samples.numpy()))

        st.write("Initial weights:")
        st.dataframe(pd.DataFrame(initial_W.detach().numpy()))

if __name__ == "__main__":
    main()
