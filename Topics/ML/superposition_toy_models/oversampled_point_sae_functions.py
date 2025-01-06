#!/usr/bin/env python3

import torch

import streamlit as st
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

class AutoEncoder(nn.Module):
    def __init__(self, dim_orig=20, dim_hidden=10):
        super().__init__()
        A = torch.randn(dim_hidden, dim_orig)
        A = F.normalize(A, p=2, dim=1)  # Normalize rows
        self.A = nn.Parameter(A)
        self.b = nn.Parameter(torch.randn(dim_hidden))
        self.relu = nn.ReLU()

    def forward(self, x):
        A_normalized = F.normalize(self.A, p=2, dim=1)  # Keep rows normalized during forward pass
        c = self.relu(torch.matmul(x, A_normalized.t()) + self.b)
        return c


def train_SAE(dim_orig, dim_hidden, X, lambda_l1, num_epochs = 1000):
    # Modified training loop with weighted L1 loss
    model = AutoEncoder(dim_orig = dim_orig, dim_hidden =  dim_hidden)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    losses = []
    sparsity = []
    relative_recon_losses = []

    X_norm = torch.norm(X)

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        c = model(X)
        reconstruction = torch.matmul(c, model.A)
        recon_loss = torch.norm(reconstruction - X)
        relative_recon_loss = recon_loss / X_norm
        l1_loss = lambda_l1 * torch.norm(c, p=1)  # Weighted L1 loss
        loss = recon_loss + l1_loss
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        relative_recon_losses.append(relative_recon_loss.item())
        zeros = (c.abs() < 1e-3).float().mean()
        sparsity.append(zeros.item())

    return model, optimizer, losses


def generate_gaussian_points_with_oversampling(n_points=100, n_dim=2, n_oversamples=100, mean=0, std=1, seed=None):
    """
    Generate random Gaussian points with one point heavily oversampled.

    Args:
        n_points: Base number of points
        n_dim: Dimension of each point
        n_oversamples: Number of times to repeat one selected point
        mean: Mean of Gaussian
        std: Standard deviation of Gaussian
        seed: Random seed
    """
    if seed is not None:
        torch.manual_seed(seed)

    # Generate base points
    X_base = torch.randn(n_points, n_dim) * std + mean

    point_to_oversample = X_base[-1:]  # Keep as 2D tensor

    # Create oversampled points
    X_oversampled = point_to_oversample.repeat(n_oversamples, 1)

    # Concatenate base and oversampled points
    X = torch.cat([X_base, X_oversampled], dim=0)

    return X, point_to_oversample


@st.cache_data
def run_experiment(
dim_orig = 20,
dim_hidden = 30,
lambda_l1 = 0.01,
n_points = 100,
n_oversamples = 100,
seed = 42
):
    figures = {}

    X, oversampled_point = generate_gaussian_points_with_oversampling(
        n_points=n_points,
        n_dim=dim_orig,
        n_oversamples=n_oversamples,
        seed=seed
    )

    model, optimizer, losses = train_SAE(dim_orig, dim_hidden, X, lambda_l1, num_epochs = 2000)

    with torch.no_grad():
        c = model(X)
        c_oversampled = model(oversampled_point)

        # For each neuron in hidden layer
        for i in range(dim_hidden):
            over_act = c_oversampled[0,i].abs().mean()
            normal_act = c[:n_points - 1,i].abs().mean()
            ratio = over_act/normal_act if normal_act != 0 else 0

            # if i < 20:
            #     print(f"Neuron {i}: oversampled={over_act:.4f}, normal={normal_act:.4f}, ratio={ratio:.4f}")

    # Plotting
    fig_loss = plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    figures['loss'] = fig_loss

    fig_activations = plt.figure(figsize=(10, 5))
    bar_width = 0.35
    index = np.arange(dim_hidden)

    normal_acts = np.array([c[:n_points - 1,i].abs().mean() for i in range(dim_hidden)])
    over_acts = np.array([c_oversampled[0,i].abs().mean() for i in range(dim_hidden)])

    plt.bar(index, normal_acts, bar_width, label='Normal points', alpha=0.8)
    plt.bar(index + bar_width, over_acts, bar_width, label='Oversampled point', alpha=0.8)

    plt.xlabel('Neuron')
    plt.ylabel('Mean Activation')
    plt.title('Neuron Activations')
    plt.xticks(index + bar_width/2, [f'N{i}' for i in range(dim_hidden)])
    plt.legend()
    figures['activations'] = fig_activations

    print("Selectivity for top activated hidden neurons for the oversampled point.")
    top_neurons = torch.argsort(c_oversampled[0].abs(), descending=True)[:3]

    correlations = torch.nn.functional.cosine_similarity(
        X[:n_points], oversampled_point.repeat(n_points, 1))

    fig_selectivity = plt.figure(figsize=(15, 4))
    for idx, i in enumerate(top_neurons):
        plt.subplot(1, 3, idx+1)

        # Get activations for this neuron
        activations = c[:n_points-1, i].abs()
        correlations_normal = correlations[:n_points-1]  # exclude oversampled point

        # Scatter plot for normal points
        plt.scatter(correlations_normal, activations, alpha=0.6, label='Normal points')

        # Add point for oversampled point
        plt.scatter(correlations[-1], c_oversampled[0,i].abs(), color='red', s=100, label='Oversampled')

        plt.xlabel('Correlation with oversampled point')
        plt.ylabel(f'Neuron {i} activation')
        plt.title(f'Neuron {i} Selectivity')
        plt.legend()

    plt.tight_layout()
    figures['selectivity'] = fig_selectivity

    # # Calculate for each point including oversampled
    results = []
    for i in range(n_points-1):
        selectivity, neuron = calculate_selectivity(X, c, i, n_points)
        results.append({
            'point': i,
            'selectivity': selectivity,
            'neuron': neuron
        })

    # Calculate for oversampled point
    selectivity, neuron = calculate_selectivity(X, c, n_points-1, n_points)
    results.append({
        'point': 'oversampled',
        'selectivity': selectivity,
        'neuron': neuron
    })

    # Print results sorted by selectivity
    # for r in sorted(results, key=lambda x: x['selectivity']):
    #     print(f"Point {r['point']}: Neuron {r['neuron']}, Selectivity = {r['selectivity']:.4f}")
    fig_distribution = plt.figure(figsize=(12, 6))

    # Sort results by selectivity
    sorted_results = sorted(results, key=lambda x: x['selectivity'])
    selectivities = [r['selectivity'] for r in sorted_results]
    points = [r['point'] for r in sorted_results]

    # Create bar plot
    plt.bar(range(len(selectivities)), selectivities, alpha=0.6)

    # Highlight oversampled point
    oversampled_idx = points.index('oversampled')
    if selectivities[oversampled_idx] == 0:
        plt.scatter(oversampled_idx, selectivities[oversampled_idx],
                color='red', s=50, zorder=3, label='Oversampled point')
    else:
        plt.bar(oversampled_idx, selectivities[oversampled_idx],
                color='red', label='Oversampled point')

    plt.xlabel('Point index (sorted by selectivity)')
    plt.ylabel('Selectivity Score')
    plt.title('Distribution of Neuron Selectivity Scores')
    plt.legend()

    # Add grid for readability
    plt.grid(True, alpha=0.3)
    figures['distribution'] = fig_distribution
    return figures

def calculate_selectivity(X, c, point_idx, n_points):
    # Get correlations with target point
    correlations = torch.nn.functional.cosine_similarity(
        X[:n_points-1], X[point_idx].unsqueeze(0).repeat(n_points-1, 1))

    # Get most active neuron for this point
    point_activations = c[point_idx].abs()
    top_neuron = torch.argmax(point_activations)

    # Get activations for this neuron
    neuron_activations = c[:n_points-1, top_neuron].abs()

    # Calculate selectivity metric
    selectivity = torch.sum(neuron_activations * (1 - correlations.abs()))

    return selectivity.item(), top_neuron.item()
