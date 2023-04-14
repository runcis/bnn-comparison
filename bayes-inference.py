import pandas as pd
import numpy as np
import pymc3 as pm
import arviz as az
import torch
import matplotlib.pyplot as plt
import time

TRAIN_TEST_SPLIT = .8
MODEL_ITERATION = 35000
POSTERIOR_SAMPLES = 100
DRAWS = 100
PERCENT_OF_MIXED_LABELS = 0.4

# Load the mushroom dataset
data = pd.read_csv("./data/mushrooms.csv")

class_labels = pd.read_csv("./data/mushrooms.csv", header=None).drop([0])[0]

# initialize arrays for the train and test indices
train_indices = np.array([], dtype=int)
test_indices = np.array([], dtype=int)

num_test = int(round(len(class_labels) * (1 - TRAIN_TEST_SPLIT) * 0.5))
# loop over the unique class labels
for label in class_labels.unique():
    # get the indices for the current class
    class_indices = np.where(class_labels == label)[0]

    # perform stratified sampling on the current class
    np.random.seed(43)
    test_indices = np.concatenate((test_indices, np.random.choice(class_indices, size=num_test, replace=False)))
    train_indices = np.concatenate((train_indices, np.setdiff1d(class_indices, test_indices)))

np.random.shuffle(test_indices)
np.random.shuffle(train_indices)

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

train_data = data.iloc[train_indices, :]
test_data = data.iloc[test_indices, :]

# mix class label randomly
num_rows_to_switch = int(len(train_data) * PERCENT_OF_MIXED_LABELS)
rows_to_switch = np.random.choice(train_data.index, size=num_rows_to_switch, replace=False)
train_data.loc[rows_to_switch, 'class'] = 1 - train_data.loc[rows_to_switch, 'class']

start_time = time.time()
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
    # Plot the KL divergence over iterations
    hist = approx.hist
    plt.plot(hist)
    plt.xlabel('Iterācijas')
    plt.ylabel('KL diverģence')
    plt.title('KL diverģence izmantojot VI')
    plt.show()
# Get the posterior distribution
trace = approx.sample(draws=DRAWS)

# Output obtained posterior mean values
posterior_mean = approx.mean.eval()
print('Posterior Mean:')
print('alpha:', posterior_mean[0])
print('beta:', posterior_mean[1])
print('sigma:', posterior_mean[2])


# Get posterior predictive distribution for test data
with model:
    pm.set_data({'X': test_data.iloc[:, 1:].T})

    ppc = pm.sample_posterior_predictive(trace, model=model, samples=POSTERIOR_SAMPLES)
    y_pred = ppc['y_obs'].mean(axis=0)
    y_pred_vars = ppc['y_obs'].var(axis=0)


# Compute accuracy on test data
result = [1 if x else 0 for x in (y_pred > 0.5)]

accuracy = np.mean(result == test_data['class'])
print(f'Test accuracy: {accuracy}', ' variance: ', y_pred_vars.mean())

def createConfusionMatrix(actual_max, result):
    confusion_matrix = np.zeros((2, 2))

    # Populate the matrix with actual vs. predicted values
    for a, p in zip(actual_max, result):
        confusion_matrix[a][p] += 1
    return confusion_matrix

print('confusion matrx : ', createConfusionMatrix(test_data['class'], result))
print("--- %s seconds ---" % (time.time() - start_time))
print('VI: mixed labels: ',PERCENT_OF_MIXED_LABELS)

# # Step 3: Calculate the prediction interval for each input
# max_probs, _ = torch.max(torch.tensor(y_pred), dim=1)  # shape: (batch_size,)
# pred_intervals = torch.stack([0.5 * torch.ones_like(max_probs), max_probs], dim=1)  # shape: (batch_size, 2)
#
# # Step 4: Determine whether the actual target value falls within the prediction interval
# correct_preds = torch.logical_and(test_data['class'] == 1, torch.logical_and(y_pred[:,1] >= pred_intervals[:,0], y_pred[:,1] <= pred_intervals[:,1]))
