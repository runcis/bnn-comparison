import time

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt

PERCENT_OF_MIXED_LABELS = 0
PERCENT_OF_MIXED_PARAMETERS = 0
BATCH_SIZE = 64
TRAIN_TEST_SPLIT = 0.8
LEARNING_RATE = 0.01
NUMBER_OF_EPOCHS = 101
HIDDEN_LAYER_NODES_1 = 50
HIDDEN_LAYER_NODES_2 = 20
HIDDEN_LAYER_NODES_3 = 20

SAMPLES = 100
DROPOUT_RATE = 0.3

OUTPUT_DIMENSION = 2 #don't change:

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

            if PERCENT_OF_MIXED_PARAMETERS:
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
    shuffle=False
)


start_time = time.time()
class BNN(nn.Module):
    def __init__(self, input_dim):
        super(BNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, HIDDEN_LAYER_NODES_1)
        self.fc2 = nn.Linear(HIDDEN_LAYER_NODES_1, HIDDEN_LAYER_NODES_2)
        self.fc3 = nn.Linear(HIDDEN_LAYER_NODES_2, OUTPUT_DIMENSION)
        self.dropout = nn.Dropout(p=DROPOUT_RATE)

    def forward(self, x):
        x = torch.cat(x, dim=1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        x = F.softmax(x, dim=-1)
        return x


def train(model, optimizer, criterion, X_train, y_train):
    model.train()
    optimizer.zero_grad()
    output = model(X_train)

    loss = criterion(output, y_train)
    loss.backward()
    optimizer.step()
    return loss.item()

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


def reliability_curve(y_true, y_prob, n_bins):
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_prob, bin_edges) - 1
    bin_sums = np.bincount(bin_indices, minlength=n_bins, weights=y_prob)
    bin_true = np.bincount(bin_indices, minlength=n_bins, weights=y_true)
    bin_total = np.bincount(bin_indices, minlength=n_bins)

    nonzero = bin_total != 0
    observed_frequency = np.divide(bin_true[nonzero], bin_total[nonzero], out=np.zeros_like(bin_true[nonzero]),
                                   where=bin_total[nonzero] != 0)
    mean_predicted_probability = np.divide(bin_sums[nonzero], bin_total[nonzero], out=np.zeros_like(bin_sums[nonzero]),
                                           where=bin_total[nonzero] != 0)

    return mean_predicted_probability, observed_frequency

def evaluate(model, X_test):
    model.eval()

    # Run model on a batch of inputs, NUMBER_OF_SAMPLE times
    outputs = []
    for i in range(SAMPLES):
        output = model(X_test)
        outputs.append(output.detach().numpy())
    return np.stack(outputs)

# Initialize the model and optimizer
model = BNN(train_dataset.inputSize)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.BCELoss()

# Train the model with Monte Carlo Dropout

loss_plot_train = []
loss_plot_test = []
acc_plot_train = []
acc_plot_test = []
vars_plot_test = []
for step in range(NUMBER_OF_EPOCHS):
    print(step)

    for dataloader in [dataloader_train, dataloader_test]:
        losses = []
        accs = []
        vars = []
        confusion_matrix = np.zeros((2, 2))
        test_predictions = torch.tensor([])
        test_true_values = torch.tensor([])

        for x , y in dataloader:

            if dataloader == dataloader_train:
                loss = train(model, optimizer, criterion, x, y)
            else:
                outputs = evaluate(model, x)

                # Create tensor from mean of samples
                mean_output = torch.tensor(np.mean(outputs, axis=0), dtype=torch.float32)

                # Calculate variance from samples
                var_output = np.var(outputs, axis=0)
                max_indices = torch.argmax(mean_output, dim=1)
                var_output = var_output[torch.arange(var_output.shape[0]), max_indices]

                test_predictions = torch.cat((test_predictions, mean_output.reshape(-1)))
                test_true_values = torch.cat((test_true_values, y.reshape(-1)))

                # Calculate loss
                loss = criterion(mean_output, y)

                # Accuracy calculation
                actual_max, accuracy = calculateAccuracy(max_indices, y)


                confusion_matrix = confusion_matrix + createConfusionMatrix(actual_max, max_indices)
                accs.append(accuracy)
                vars.append(np.mean(var_output))

            losses.append(loss)

        if dataloader == dataloader_train:
            loss_plot_train.append(np.mean(losses))
        else:
            loss_plot_test.append(np.mean(losses))
            acc_plot_test.append(np.mean(accs))
            vars_plot_test.append(np.mean(vars))


    if step % 10 == 0:
        print(step, ' : ', confusion_matrix)
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

        ax1.set_xlabel("Epoha")
        plt.subplots_adjust(hspace=0.5)
        plt.show()
        print('Step: ', step, 'got accuracy: ', acc_plot_test[-1], 'got variance: ', vars_plot_test[-1])


        # mean_predicted_prob, observed_freq = reliability_curve(test_true_values.tolist(), test_predictions.tolist(), n_bins=20)
        #
        # plt.figure(figsize=(8, 8))
        # plt.plot(mean_predicted_prob, observed_freq, 'o-', label='Modelis')
        # plt.plot([0, 1], [0, 1], '--', label='Perfekti kalibrēts', color='gray')
        # plt.xlabel('Vidējā sagaidāmā varbūtība')
        # plt.ylabel('Novērotā frekvence')
        # plt.legend(loc='best')
        # plt.title('Uzticamības diagramma')
        # plt.show()

        print("--- %s seconds ---" % (time.time() - start_time))

print('DROPOUT: mixed labels: ',PERCENT_OF_MIXED_LABELS, 'got accuracy: ',
      acc_plot_test[-1], 'got variance: ', vars_plot_test[-1])