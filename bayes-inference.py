import pandas as pd
import numpy as np
import pymc3 as pm
import arviz as az
import torch
import matplotlib.pyplot as plt

TEST_TRAIN_SPLIT = .8
MODEL_ITERATION = 10000
POSTERIOR_SAMPLES = 500
DRAWS = 1000
PERCENT_OF_MIXED_LABELS = 0.3

# Load the mushroom dataset
data = pd.read_csv("../data/mushrooms.csv")

# Set column names
col_names = ['class', 'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor', 'gill-attachment',
             'gill-spacing', 'gill-size', 'gill-color', 'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',
             'stalk-surface-below-ring', 'stalk-color-above-ring', 'stalk-color-below-ring', 'veil-type',
             'veil-color', 'ring-number', 'ring-type', 'spore-print-color', 'population', 'habitat']
data.columns = col_names

# Convert categorical features to numeric
for col in data.columns:
    data[col] = pd.Categorical(data[col]).codes

X = data.drop('class', axis=1)
X = np.array(X)
y = data['class']

train_size = int(len(data) * TEST_TRAIN_SPLIT)
train_data = data.iloc[:train_size, :]
test_data = data.iloc[train_size:, :]

# mix class label randomly
num_train = len(train_data)
num_to_mix = int(num_train*PERCENT_OF_MIXED_LABELS)
mix_indices = np.random.choice(num_train, num_to_mix, replace=False)
for idx in mix_indices:
    train_data.at[idx, 'class'] = 1 - train_data.at[idx, 'class']

# Define the model
with pm.Model() as model:

    X = pm.Data('X', train_data.iloc[:, 1:].T)
    # Priors
    alpha = pm.Normal('alpha', mu=0, sd=10)
    beta = pm.Normal('beta', mu=0, sd=10, shape=(22,))
    # Linear combination of features
    mu = alpha + pm.math.dot(beta, X)
    # Likelihood function
    y_obs = pm.Bernoulli('y_obs', logit_p=mu, observed=train_data['class'])
    # Run variational inference
    approx = pm.fit(method='advi', n=MODEL_ITERATION)

# Get the posterior distribution
trace = approx.sample(draws=DRAWS)


# Get posterior predictive distribution for test data
with model:
    pm.set_data({'X': test_data.iloc[:, 1:].T})

    ppc = pm.sample_posterior_predictive(trace, model=model, samples=POSTERIOR_SAMPLES)
    y_pred = ppc['y_obs'].mean(axis=0)


# Compute accuracy on test data
accuracy = np.mean((y_pred > 0.5) == test_data['class'])
print(f'Test accuracy: {accuracy}')

# Step 3: Calculate the prediction interval for each input
max_probs, _ = torch.max(torch.tensor(y_pred), dim=1)  # shape: (batch_size,)
pred_intervals = torch.stack([0.5 * torch.ones_like(max_probs), max_probs], dim=1)  # shape: (batch_size, 2)

# Step 4: Determine whether the actual target value falls within the prediction interval
correct_preds = torch.logical_and(test_data['class'] == 1, torch.logical_and(y_pred[:,1] >= pred_intervals[:,0], y_pred[:,1] <= pred_intervals[:,1]))

# Step 5: Calculate the PICP
picp = torch.mean(correct_preds.float()) * 100.0
print(f"Prediction interval coverage probability: {picp:.2f}%")
