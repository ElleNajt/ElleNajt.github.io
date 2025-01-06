#!/usr/bin/env python3


import random
import pandas as pd
from sklearn.decomposition import PCA
from matplotlib.colors import LinearSegmentedColormap
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

class BaseAutoEncoder(nn.Module):
    def __init__(self, n_dim, h_dim):
        super().__init__()
        self.encoder = nn.Linear(n_dim, h_dim)
        self.bias = nn.Parameter(torch.zeros(n_dim))


        # prefer this over a random initialization
        # because it doesn't add a random project, which is confusing for small importance base
        # and then adding randomness to first row so it has a gradient
        nn.init.normal_(self.encoder.weight, mean=0.0, std=0.001)  # very small std


    def get_importance_matrix(self):
        W = self.encoder.weight
        WtW = torch.mm(W.t(), W)
        return WtW

    def get_feature_importance(self):
        WtW = self.get_importance_matrix()
        importance = torch.diag(WtW)
        return importance

class LinearAutoEncoder(BaseAutoEncoder):
    def forward(self, x):
        h = self.encoder(x)
        x_recon = torch.nn.functional.linear(h,
                                           self.encoder.weight.t()) + self.bias
        return x_recon

class ReLUAutoEncoder(BaseAutoEncoder):
    def __init__(self, n_dim, h_dim):
        super().__init__(n_dim, h_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        h = self.encoder(x)  # Wx
        x_recon = torch.nn.functional.linear(h, self.encoder.weight.t()) + self.bias  # W^T(Wx) + b
        x_recon = self.relu(x_recon)  # ReLU(W^T(Wx) + b)
        return x_recon

class SAEAutoEncoder(BaseAutoEncoder):
    def __init__(self, n_dim, h_dim):
        super().__init__(n_dim, h_dim)
        self.relu = nn.ReLU()
        self.hidden_bias = nn.Parameter(torch.zeros(h_dim))

    def forward(self, x):
        h = self.encoder(x) + self.hidden_bias  # Wx + b
        c = self.relu(h) # Relu(h)
        x_recon = torch.nn.functional.linear(c, self.encoder.weight.t()) + self.bias  # W^T(Relu(Wx + b)) + b
        return x_recon

class WeightedMSELoss(nn.Module):
    def __init__(self, importance_base):
        super().__init__()
        self.importance_base = importance_base

    def forward(self, output, target):

        if self.importance_base == 0:
            self.importance_weights = torch.zeros(output.shape[1], dtype=output.dtype)
            self.importance_weights[0] = 1.0

        else:
            self.importance_weights = self.importance_base ** torch.arange(output.shape[1], dtype=output.dtype)
        # print("Importance weights:", self.importance_weights)  # Debug print

        squared_errors = (output - target) ** 2

        # print("Squared errors:", squared_errors.mean(dim=0))
        weighted_errors = self.importance_weights.unsqueeze(0) * squared_errors
        # print("Weighted errors sum per dimension:", weighted_errors.sum(dim=0))  # Debug print
        return weighted_errors.sum()

def generate_sparse_data(n_samples, n_dim, sparsity):
    """Generate sparse data where each coordinate is:
    - 0 with probability sparsity
    - Uniform[0,1] with probability (1-sparsity)
    """
    # Generate random mask: 1 with probability (1-sparsity), 0 with probability sparsity
    mask = torch.bernoulli(torch.ones(n_samples, n_dim) * (1 - sparsity))
    # Generate uniform values
    values = torch.rand(n_samples, n_dim)
    # Apply mask to get final data
    return mask * values

def analyze_model(model):
    W = model.encoder.weight.detach()
    norms = torch.norm(W, dim=0)
    W_normalized = W / (norms[None,:] + 1e-8)
    interference = torch.sum((W_normalized.T @ W_normalized), dim=1) - 1

    # Find antipodal pairs
    cosine_sim = W_normalized.T @ W_normalized

    # Find pairs with similarity close to -1 (allowing some tolerance)
    antipodal_pairs = []
    for i in range(len(cosine_sim)):
        for j in range(i+1, len(cosine_sim)):
            if cosine_sim[i,j] < -0.9:  # threshold for considering antipodal
                antipodal_pairs.append((i, j, cosine_sim[i,j].item()))

    # Sort by norm
    sorted_idx = torch.argsort(norms, descending=True)

    print("\nModel Analysis:")
    print("-" * 40)
    print(f"Top 5 feature norms: {norms[sorted_idx][:5].tolist()}")
    print(f"Top 5 interference values: {interference[sorted_idx][:5].tolist()}")
    print(f"Number of features with norm > 0.1: {(norms > 0.1).sum().item()}")
    print(f"Number of antipodal pairs: {len(antipodal_pairs)}")
    if antipodal_pairs:
        print("Antipodal pairs (feature_i, feature_j, cosine_similarity):")
        for i, j, sim in antipodal_pairs:
            print(f"  {i:2d}, {j:2d}: {sim:.3f}")
    print("-" * 40)


def plot_model_analysis(model, criterion, h_dim):
    plt.close('all')
    fig = plt.figure(figsize=(20, 5))

    W = model.encoder.weight.detach()
    norms = torch.norm(W, dim=0)

    # 1. Original W^T W matrix
    plt.subplot(131)
    WtW = model.get_importance_matrix().detach()
    # print(f"WtW min: {WtW.min().item():.3f}, max: {WtW.max().item():.3f}")
    cmap = LinearSegmentedColormap.from_list('custom', ['blue', 'white', 'red'])
    vmax = max(abs(WtW.min().item()), abs(WtW.max().item()))
    plt.imshow(WtW, cmap=cmap, vmin=-vmax, vmax=vmax)
    plt.colorbar()
    plt.title('WᵀW Matrix')


    # # 2. Normalized W^T W matrix
    # plt.subplot(132)
    W_normalized = W / (norms[None,:] + 1e-8)
    WtW_normalized = W_normalized.T @ W_normalized


    # # Mask out diagonal and zero-norm features
    # mask = torch.ones_like(WtW_normalized)
    # mask[torch.eye(len(mask), dtype=bool)] = 0  # mask diagonal
    # mask[norms < 1e-4, :] = 0  # mask zero-norm rows
    # mask[:, norms < 1e-4] = 0  # mask zero-norm columns
    # WtW_normalized = WtW_normalized * mask

    # print(f"Normalized WtW min: {WtW_normalized.min().item():.3f}, max: {WtW_normalized.max().item():.3f}")
    # plt.imshow(WtW_normalized, cmap=cmap, vmin=-1, vmax=1)
    # plt.colorbar()
    # plt.title('Normalized WᵀW (Cosine Similarities)')

    ax = plt.subplot(132)
    norms = torch.norm(W, dim=0)
    mask = norms > 1e-8
    W_normalized = torch.zeros_like(W)
    W_normalized[:, mask] = W[:, mask] / norms[mask][None, :]
    dots = (W_normalized.T @ W)
    squared_dots = dots * dots
    interference = squared_dots.sum(dim=1) - squared_dots.diagonal()
    interference = interference * mask
    # Sort by importance weights instead of norms
    importance_weights = criterion.importance_weights
    sorted_idx = torch.argsort(importance_weights, descending=True)
    interference_cmap = LinearSegmentedColormap.from_list('custom', ['black', 'yellow'])
    bars = ax.bar(range(len(norms)),
                norms[sorted_idx],
                color=interference_cmap(interference[sorted_idx]))
    plt.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0, 1),
                                    cmap=interference_cmap),
                ax=ax)
    plt.title('Feature Norms (height) & Interference (color)')
    plt.xlabel('Feature Index (sorted by importance)')
    plt.ylabel('Feature Norm ||W_i||')



    # Add PCA visualization
    W = model.encoder.weight.detach()  # Shape: [n_dim, h_dim]
    W_numpy = W.t().numpy()  # Transpose and convert to numpy

    pca = PCA(n_components=min(3, h_dim))
    W_pca = pca.fit_transform(W_numpy)


    importance_cmap = LinearSegmentedColormap.from_list('custom', ['#2E8B57', '#90EE90', '#F4E87C'])  # Sea green -> light green -> soft yellow
    ax = plt.subplot(133, projection='3d' if h_dim > 2 else None)
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.1)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.1)

    # Set bounds based on maximum vector length
    max_norm = np.max(np.linalg.norm(W_numpy, axis=0))
    print(max_norm)

    plt.xlim(-max_norm, max_norm)
    plt.ylim(-max_norm, max_norm)

    if h_dim > 2:
        plt.zlim(-max_norm, max_norm)

    if h_dim == 1:
        # 1D visualization
        for i in range(W_numpy.shape[0]):
            plt.quiver(0, 0, W_numpy[i,0], 0,
                    color=importance_cmap(importance_weights[i].item()),
                    angles='xy', scale_units='xy', scale=1, width=.005)
            plt.annotate(f'W_{i}', (W_numpy[i,0], 0))
    elif h_dim == 2:
        # 2D visualization (current implementation)
        for i in range(W_numpy.shape[0]):
            plt.quiver(0, 0, W_numpy[i,0], W_numpy[i,1],
                    color=importance_cmap(importance_weights[i].item()),
                    angles='xy', scale_units='xy', scale=1, width=.005)
            plt.annotate(f'W_{i}', (W_numpy[i,0], W_numpy[i,1]))
    else:
        # 3D visualization using first 3 components
        for i in range(W_numpy.shape[0]):
            ax.quiver(0, 0, 0,
                    W_pca[i,0], W_pca[i,1], W_pca[i,2],
                    color=importance_cmap(importance_weights[i].item()))
            ax.text(W_pca[i,0], W_pca[i,1], W_pca[i,2], f'W_{i}')

    # Add colorbar with importance label
    plt.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0, 1),
                                    cmap=importance_cmap),
                ax=ax,
                label='Importance')


    plt.draw()

    plt.title('Weight Vectors')
    plt.axis('equal')  # This ensures circle appears circular

    plt.xlim(-max_norm, max_norm)
    plt.ylim(-max_norm, max_norm)
    plt.tight_layout()
    return fig

def run_experiment(model_version,
n_dim = 20,
h_dim = 5,
n_samples = 1000,  # total number of data points
sparsity = 0.7,
importance_base = .7,
epochs = 1000,
learning_rate = 0.01,):
    # Initialize model and optimizer
    model = model_version(n_dim, h_dim)

    initial_W = model.encoder.weight.clone()  # Store initial weights
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = WeightedMSELoss(importance_base)


    # Training loop
    losses = []
    for _ in range(epochs):

        X = generate_sparse_data(n_samples, n_dim, sparsity)
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, X)

        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    # Visualization
    fig = plot_model_analysis(model, criterion, h_dim)
    W = model.encoder.weight.detach()
    df = pd.DataFrame(W)
    print(W)

    sample_indices = torch.randint(0, n_samples, (5,))  # Get 5 random samples
    X_samples = X[sample_indices]

    return fig, df, criterion, X_samples, losses, initial_W

    # W_numpy = W.t().numpy()  # Transpose and convert to numpy
    # pca = PCA(n_components=2)

    # W_pca = pca.fit_transform(W_numpy)
    # print(pd.DataFrame(W_pca))
    # analyze_model(model)

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
