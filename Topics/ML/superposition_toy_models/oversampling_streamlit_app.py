#!/usr/bin/env python3


import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
from oversampled_point_sae_functions import run_experiment

def main():
    st.title("Toy example of sparse auto encoder")
    st.markdown(""" Given a dataset of random points with one repeatedly sample point, I wanted to see if a sparse auto encoder would develop a feature for the
            oversampled vector in the hidden layer.""")

    # params_space, cached_results = precompute_experiments()

    with st.expander("ℹ️ Mathematical Details", expanded=False):

        col1, col2 = st.columns(2)

        with col1:

            st.markdown("### Data generating process ### \n We sample n points from a gaussian, and repeat the last one k times. The repeated point is the oversampled point.")
            st.latex(r"X' = \text{ sample } n \text{ points from } \mathcal{N}(0, I_d)")
            st.latex(r"X = X \cup X[-1] * k")


        with col2:
            st.markdown("""Metrics""")

            st.markdown("""
                ### Hidden Layer Neuron Activations ###
                - Blue bars show average activation for normal points
                - Orange bars show activation for the oversampled point
                - Higher orange bars indicate neurons specialized for the oversampled point
            """)

            st.markdown("""
                ### Selectivity Distribution ###
                - Shows selectivity scores for all points
                - Red bar highlights the oversampled point
                - Lower selectivity indicates stronger specialization
            """)

            st.markdown("""
                ### Neuron Selectivity Analysis ###
                - Shows top 3 most active neurons for the oversampled point
                - X-axis: correlation with oversampled point
                - Y-axis: neuron activation strength
                - Red dot: oversampled point's activation
                - Blue dots: normal points' activations
                - Red dot high and uncorrelated blue dots low indicates that the feature has learned about the oversampled point
            """)

            st.latex(r"\text{Selectivity} = \sum_i a_i(1 - |\text{corr}_i|)")


    # Initialize session state
    if 'history' not in st.session_state:
        st.session_state.history = []
    if 'current_index' not in st.session_state:
        st.session_state.current_index = -1

    # Create sliders in the sidebar
    st.sidebar.header("Parameters")
    dim_orig = st.sidebar.slider("Original Dimension", 5, 50, 20)
    dim_hidden = st.sidebar.slider("Hidden Dimension", 5, 50, 30)
    lambda_l1 = st.sidebar.slider("L1 Lambda", 0.005, 0.3, 0.01, format="%.3f")
    n_points = st.sidebar.slider("Number of Points", 50, 500, 100)
    n_oversamples = st.sidebar.slider("Number of Oversamples", 50, 1000, 100)
    seed = st.sidebar.number_input("Random Seed", value=42)

    # Navigation buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Previous", disabled=st.session_state.current_index <= 0):
            st.session_state.current_index -= 1
    with col2:
        if st.button("Next →", disabled=st.session_state.current_index >= len(st.session_state.history) - 1):
            st.session_state.current_index += 1

    if st.sidebar.button("Run Experiment"):
        params = {
            'dim_orig': dim_orig,
            'dim_hidden': dim_hidden,
            'lambda_l1': lambda_l1,
            'n_points': n_points,
            'n_oversamples': n_oversamples,
            'seed': seed
        }
        figures = run_experiment(**params)

        # Add to history
        st.session_state.history.append({'params': params, 'figures': figures})
        st.session_state.current_index = len(st.session_state.history) - 1
        st.rerun()  # Force streamlit to rerun with new state

    # Display current experiment if exists
    if st.session_state.current_index >= 0:
        current = st.session_state.history[st.session_state.current_index]

        # Create 2 columns for top row
        col1, col2 = st.columns(2)

        with col1:
            st.pyplot(current['figures']['activations'])

        with col2:
            st.pyplot(current['figures']['distribution'])

        # Bottom row
        st.pyplot(current['figures']['selectivity'])


        st.write(f"Experiment {st.session_state.current_index + 1} of {len(st.session_state.history)}")

        st.write("Parameters:", current['params'])

        st.write(" ### DIAGNOSTICS ###")
        st.pyplot(current['figures']['loss'])


if __name__ == "__main__":
    main()
