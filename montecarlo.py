import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt

PERCENT_OF_MIXED_LABELS = 0
BATCH_SIZE = 128
TRAIN_TEST_SPLIT = 0.7
LEARNING_RATE = 0.01
NUMBER_OF_EPOCHS = 300
HIDDEN_LAYER_NODES_1 = 50
HIDDEN_LAYER_NODES_2 = 50
HIDDEN_LAYER_NODES_3 = 50

NUMBER_OF_SAMPLE = 500
DROPOUT_RATE = 0.3

OUTPUT_DIMENSION = 2 #don't change:
DEVICE = 'cpu'
if torch.cuda.is_available():
    print('cuda')
    DEVICE = 'cuda'
else:
    print('cpu')

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
    shuffle=False
)


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
        return x


def train(model, optimizer, criterion, X_train, y_train):
    model.train()
    optimizer.zero_grad()
    output = model(X_train)

    for k in range(len(output)):
        output[k] = output[k].to(DEVICE)
    loss = criterion(output, y_train)
    loss.backward()
    optimizer.step()
    return loss.item()

def evaluate(model, criterion, X_test, y_test):
    model.eval()

    # Run model on a batch of inputs, NUMBER_OF_SAMPLE times
    outputs = []
    for i in range(NUMBER_OF_SAMPLE):
        output = model(X_test)
        outputs.append(output.cpu().detach().numpy())
    outputs = np.stack(outputs)

    # Create tensor from mean of samples
    mean_output = torch.tensor(np.mean(outputs, axis=0), dtype=torch.float32).to(DEVICE)

    # Calculate variance from samples
    #var_output = np.var(outputs, axis=0)
    max_indices = torch.argmax(mean_output, dim=1)
    #var_output = var_output[torch.arange(var_output.shape[0]), max_indices]

    # Calculate loss
    loss = criterion(mean_output, y_test).cpu()

    # Calculate accurately predicted labels
    pred_max = torch.argmax(mean_output, dim=1)
    actual_max = torch.argmax(y_test, dim=1)
    correct_count = (pred_max == actual_max).sum().item()
    acc = correct_count / y_test.shape[0]

    return acc, loss


# Initialize the model and optimizer
model = BNN(train_dataset.inputSize)
model.to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

# Train the model with Monte Carlo Dropout

loss_plot_train = []
loss_plot_test = []
acc_plot_train = []
acc_plot_test = []
for step in range(NUMBER_OF_EPOCHS):
    print(step)

    for dataloader in [dataloader_train, dataloader_test]:
        losses = []
        accs = []
        for x , y in dataloader:

            for i in range(len(x)):
                x[i] = x[i].to(DEVICE)
            y = y.to(DEVICE)

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



print('DROPOUT: mixed labels: ',PERCENT_OF_MIXED_LABELS, 'got accuracy: ', acc_plot_test[-1])