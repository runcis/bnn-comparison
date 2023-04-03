import torchbnn as bnn
from torch.utils.data import DataLoader, Dataset
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt


PERCENT_OF_MIXED_LABELS = 0
BATCH_SIZE = 64
TRAIN_TEST_SPLIT = 0.7
LEARNING_RATE = 0.01
NUMBER_OF_EPOCHS = 300
HIDDEN_LAYER_NODES_1 = 50
HIDDEN_LAYER_NODES_2 = 20
HIDDEN_LAYER_NODES_3 = 10

WEIGHT_OF_CROSS_ENTROPY_LOSS = 0.9
WEIGHT_OF_KL_DIVERGENCE_LOSS = 0.1
PRIOR_MU_FIRST_LAYER = 0
PRIOR_MU_SECOND_LAYER = 0
PRIOR_MU_THIRD_LAYER = 0
PRIOR_MU_FORTH_LAYER = 0
PRIOR_SIGMA_FIRST_LAYER = 0.01
PRIOR_SIGMA_SECOND_LAYER = 0.01
PRIOR_SIGMA_THIRD_LAYER = 0.01
PRIOR_SIGMA_FOURTH_LAYER = 0.01

OUTPUT_DIMENSION = 2
# Load the mushroom dataset
data_path = "../data/mushrooms.csv"

class MushroomDataset(torch.utils.data.Dataset):
    def __init__(self, train=False):

        data = pd.read_csv(data_path, header=None)

        # Encode y values
        labelValues = pd.Categorical(data[0][1:]).codes
        self.y = torch.tensor(pd.get_dummies(labelValues).values, dtype=torch.float32)

        # Encode x values
        data = data.drop(columns=[0])
        self.X = []
        self.inputSize = 0
        for column in data:
            values = pd.Categorical(data[column]).codes
            tensor = torch.tensor(pd.get_dummies(values).values, dtype=torch.float32)
            self.X.append(tensor)
            self.inputSize += len(data[column].unique())

        if train:
            self.X = [x[:round(len(x) * TRAIN_TEST_SPLIT)] for x in self.X]
            self.y = self.y[:round(len(self.y) * TRAIN_TEST_SPLIT)]

            # create a mask of indices to flip
            if (PERCENT_OF_MIXED_LABELS):
                mask = np.random.choice(len(self.y), int(len(self.y) * PERCENT_OF_MIXED_LABELS), replace=False)
                # flip the values at the selected indices
                self.y[mask] = 1 - self.y[mask]
            self.num_samples = len(self.y)

        else:
            self.X = [x[round(len(x) * TRAIN_TEST_SPLIT):] for x in self.X]
            self.y = self.y[round(len(self.y) * TRAIN_TEST_SPLIT):]
            self.num_samples = len(self.y)

    def __getitem__(self, index):
        # return the feature and label at the given index
        return [x[index] for x in self.X], self.y[index]

    def __len__(self):
        # return the total number of samples in the dataset
        return self.num_samples


mushroom_data = pd.read_csv(data_path)

train_dataset = MushroomDataset(train=True)
test_dataset = MushroomDataset(train=False)

dataloader_train = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

dataloader_test = torch.utils.data.DataLoader(
    dataset=test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)


class BNN(nn.Module):
    def __init__(self, input_dim):
        super(BNN, self).__init__()
        self.fc1 = bnn.BayesLinear(prior_mu=PRIOR_MU_FIRST_LAYER, prior_sigma=PRIOR_SIGMA_FIRST_LAYER, in_features=input_dim, out_features=HIDDEN_LAYER_NODES_1)
        self.fc2 = bnn.BayesLinear(prior_mu=PRIOR_MU_SECOND_LAYER, prior_sigma=PRIOR_SIGMA_SECOND_LAYER, in_features=HIDDEN_LAYER_NODES_1, out_features=HIDDEN_LAYER_NODES_2)
        self.fc3 = bnn.BayesLinear(prior_mu=PRIOR_MU_THIRD_LAYER, prior_sigma=PRIOR_SIGMA_THIRD_LAYER, in_features=HIDDEN_LAYER_NODES_2, out_features=HIDDEN_LAYER_NODES_3)
        self.fc4 = bnn.BayesLinear(prior_mu=PRIOR_MU_FOURTH_LAYER, prior_sigma=PRIOR_SIGMA_FOURTH_LAYER, in_features=HIDDEN_LAYER_NODES_3, out_features=OUTPUT_DIMENSION)

    def forward(self, x):
        x = torch.cat(x, dim=1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        return x


kl_loss = bnn.BKLLoss(reduction='mean', last_layer_only=False)
def train(model, optimizer, criterion, X_train, y_train):
    optimizer.zero_grad()
    output = model(X_train)
    ce = criterion(output, y_train) * WEIGHT_OF_CROSS_ENTROPY_LOSS
    kl = kl_loss(model) * WEIGHT_OF_KL_DIVERGENCE_LOSS
    loss = ce + kl

    optimizer.step()
    loss.backward()
    optimizer.step()
    return loss.item(), ce.item(), kl.item()


def evaluate(model, criterion, X_test, y_test):
    output = model(X_test)
    loss = criterion(output, y_test)
    pred_max = torch.argmax(output, dim=1)
    actual_max = torch.argmax(y_test, dim=1)
    correct_count = (pred_max == actual_max).sum().item()
    acc = correct_count / y_test.shape[0]
    return loss.item(), acc


# Set the model hyperparameters
model = BNN(train_dataset.inputSize)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

# Train the model with BBB

loss_plot_train = []
loss_plot_test = []
acc_plot_train = []
acc_plot_test = []
for step in range(NUMBER_OF_EPOCHS):
    losses = []
    accs = []
    ce_loss = []
    kll = []
    for dataloader in [dataloader_train, dataloader_test]:
        losses = []
        accs = []
        for x, y in dataloader:

            if dataloader == dataloader_train:
                loss = train(model, optimizer, criterion, x, y)
            else:
                acc, loss = evaluate(model, criterion, x, y)
                accs.append(acc)

            losses.append(loss)

        if dataloader == dataloader_train:
            loss_plot_train.append(np.mean(losses))
        else:
            loss_plot_test.append(np.mean(losses))
            acc_plot_test.append(np.mean(accs))

    if step % 10 == 0:
        _, axes = plt.subplots(nrows=3, ncols=1, figsize=(10,10))
        ax1 = axes[0]
        ax1.set_title("Training loss")
        ax1.plot(loss_plot_train, 'r-', label='train')
        ax1.legend()
        ax1.set_ylabel("loss")

        ax1 = axes[1]
        ax1.set_title("Testing Loss")
        ax1.plot(loss_plot_test, 'r-', label='test')
        ax1.legend()
        ax1.set_ylabel("loss.")

        ax1 = axes[2]
        ax1.set_title("Accuracy")
        ax1.plot(acc_plot_test, 'g-', label='test')
        ax1.legend()
        ax1.set_ylabel("Accuracy")

        ax1.set_xlabel("Epoch")
        plt.subplots_adjust(hspace=0.5)
        plt.show()
        print('Step: ', step, 'got accuracy: ', acc_plot_test[-1])



print('BBB: mixed labels: ',PERCENT_OF_MIXED_LABELS, 'got accuracy: ', acc_plot_test[-1])