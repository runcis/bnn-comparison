import torchbnn as bnn
from torch.utils.data import DataLoader, Dataset
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
import time

PERCENT_OF_MIXED_LABELS = 0
PERCENT_OF_MIXED_PARAMETERS = 0
BATCH_SIZE = 64
TRAIN_TEST_SPLIT = .8
LEARNING_RATE = 0.001
NUMBER_OF_EPOCHS = 101
HIDDEN_LAYER_NODES_1 = 100
HIDDEN_LAYER_NODES_2 = 20
HIDDEN_LAYER_NODES_3 = 20

WEIGHT_OF_CROSS_ENTROPY_LOSS = 0.9
WEIGHT_OF_KL_DIVERGENCE_LOSS = 0.1
PRIOR_MU_FIRST_LAYER = 0
PRIOR_MU_SECOND_LAYER = 0
PRIOR_SIGMA_FIRST_LAYER = 0.01
PRIOR_SIGMA_SECOND_LAYER = 0.01
SAMPLES = 100

OUTPUT_DIMENSION = 2
# Load the mushroom dataset
data_path = "./data/mushrooms.csv"

# initialize arrays for the train and test indices
train_indices = np.array([], dtype=int)
test_indices = np.array([], dtype=int)

class_labels = pd.read_csv(data_path, header=None).drop([0])[0]
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

class MushroomDataset(torch.utils.data.Dataset):
    def __init__(self, train=False):

        data = pd.read_csv(data_path, header=None)
        # Encode y values
        labelValues = pd.Categorical(data[0][1:]).codes
        self.y = torch.tensor(pd.get_dummies(labelValues).values, dtype=torch.float32)

        # Drop 1 column and 1 row (output class and labels)
        data = data.drop(columns=[0])
        data = data.drop([0])
        # Encode X values
        self.X = []
        self.inputSize = 0
        for column in data:
            values = pd.Categorical(data[column]).codes
            tensor = torch.tensor(pd.get_dummies(values).values, dtype=torch.float32)
            self.X.append(tensor)
            self.inputSize += len(data[column].unique())

        if train:
            self.X = [tensor[train_indices] for tensor in self.X]
            self.y = [self.y[i] for i in train_indices]
            self.num_samples = len(self.y)

            # create a mask of indices to flip
            if (PERCENT_OF_MIXED_LABELS):
                mask = np.random.choice(self.num_samples, int(self.num_samples * PERCENT_OF_MIXED_LABELS), replace=False)
                # flip the values at the selected indices
                temp = np.array(self.y)
                temp[mask] = 1 - temp[mask]
                self.y = [tensor for tensor in temp]

            if (PERCENT_OF_MIXED_PARAMETERS):
                mask = np.random.choice(self.num_samples, int(self.num_samples * PERCENT_OF_MIXED_PARAMETERS), replace=False)

                for id in mask:
                    featureId = np.random.randint(0, 21)
                    tensorToChange = self.X[featureId][id]
                    self.X[featureId][id] = torch.roll(tensorToChange, shifts=torch.randint(0, len(tensorToChange), (1,)).item())

        else:
            self.X = [tensor[test_indices] for tensor in self.X]
            self.y = [self.y[i] for i in test_indices]
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


start_time = time.time()
class BNN(nn.Module):
    def __init__(self, input_dim):
        super(BNN, self).__init__()
        self.fc1 = bnn.BayesLinear(prior_mu=PRIOR_MU_FIRST_LAYER, prior_sigma=PRIOR_SIGMA_FIRST_LAYER, in_features=input_dim, out_features=HIDDEN_LAYER_NODES_1)

        self.fc2 = bnn.BayesLinear(prior_mu=PRIOR_MU_SECOND_LAYER, prior_sigma=PRIOR_SIGMA_SECOND_LAYER,
                               in_features=HIDDEN_LAYER_NODES_1, out_features=OUTPUT_DIMENSION)

    def forward(self, x):
        x = torch.cat(x, dim=1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.softmax(x, dim=-1)
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


def calculateAccuracy(max_indices, y_test):
    # Accuracy calculation
    actual_max = torch.argmax(y_test, dim=1)
    correct_count = (max_indices == actual_max).sum().item()
    accuracy = correct_count / y_test.shape[0]
    return actual_max, accuracy


def createConfusionMatrix(actual_max, max_indices):
    confusion_matrix = np.zeros((2, 2))

    # Populate the matrix with actual vs. predicted values
    for a, p in zip(actual_max, max_indices):
        confusion_matrix[a][p] += 1
    return confusion_matrix


def evaluate(model, X_test):
    outputs = []
    for i in range(SAMPLES):
        output = model(X_test)
        outputs.append(output)
    return torch.stack(outputs)


# Set the model hyperparameters
model = BNN(train_dataset.inputSize)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.BCELoss()

# Train the model with BBB

loss_plot_train = []
loss_plot_test = []
acc_plot_train = []
acc_plot_test = []
vars_plot_test = []
for step in range(NUMBER_OF_EPOCHS):
    losses = []
    accs = []
    ce_loss = []
    kll = []
    confusion_matrix = np.zeros((2, 2))
    videjaVariance = 0
    test_predictions = torch.tensor([])
    test_true_values = torch.tensor([])

    for dataloader in [dataloader_train, dataloader_test]:
        losses = []
        accs = []
        vars = []
        for x, y in dataloader:

            if dataloader == dataloader_train:
                loss = train(model, optimizer, criterion, x, y)
            else:
                outputs = evaluate(model, x)

                mean = outputs.mean(dim=0)
                predictions, max_indices = torch.max(mean, dim=1)

                variances = outputs.var(dim=0)
                prediction_variances = torch.gather(variances, dim=1, index=max_indices.unsqueeze(1))

                # Loss calculation
                loss = criterion(mean, y).item()

                # Accuracy calculation
                actual_max, accuracy = calculateAccuracy(max_indices, y)

                test_predictions = torch.cat((test_predictions, mean.reshape(-1)))
                test_true_values = torch.cat((test_true_values, y.reshape(-1)))

                confusion_matrix = confusion_matrix + createConfusionMatrix(actual_max, max_indices)

                accs.append(accuracy)
                vars.append(torch.mean(prediction_variances.float()).item())

            losses.append(loss)

        if dataloader == dataloader_train:
            loss_plot_train.append(np.mean(losses))
        else:
            loss_plot_test.append(np.mean(losses))
            acc_plot_test.append(np.mean(accs))
            vars_plot_test.append(np.mean(vars))

    if step % 10 == 0:
        print('confusion matrix : ', confusion_matrix, )
        _, axes = plt.subplots(nrows=4, ncols=1, figsize=(10,14))
        ax1 = axes[0]
        ax1.set_title("Apmācības kļūda")
        ax1.plot(loss_plot_train, 'r-')
        ax1.legend()
        ax1.set_ylabel("kļūda")

        ax1 = axes[1]
        ax1.set_title("Testēšanas kļūda")
        ax1.plot(loss_plot_test, 'r-')
        ax1.legend()
        ax1.set_ylabel("kļūda")

        ax1 = axes[2]
        ax1.set_title("Testēšanas rezultātu vidējā dispersija")
        ax1.plot(vars_plot_test, 'b-')
        ax1.legend()
        ax1.set_ylabel("dispersija")

        ax1 = axes[3]
        ax1.set_title("Testēšanas pareizi kategorizētie ieraksti")

        ax1.plot(acc_plot_test, 'g-')
        ax1.legend()
        ax1.set_ylabel("Precizitāte")

        ax1.set_xlabel("Cikls")
        plt.subplots_adjust(hspace=0.5)
        plt.show()
        print('Step: ', step, 'got accuracy: ', acc_plot_test[-1], 'got variance: ', vars_plot_test[-1])
        print("--- %s seconds ---" % (time.time() - start_time))


print('BBB: mixed labels: ',PERCENT_OF_MIXED_LABELS,'BBB: mixed parameters: ',PERCENT_OF_MIXED_PARAMETERS, 'got accuracy: ', acc_plot_test[-1],
      acc_plot_test[-1], 'got variance: ', vars_plot_test[-1])