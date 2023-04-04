Repository contains the implementation of three BNN algorithms for the mushroom dataset - Bayes by backprop (BBB), Monte Carlo Dropout (MC-D), Variation Inference (VI).

# Things to check:
1. Kā embedding ir implementēts algoritmos:

Dataset objektā katrai kategorijai tiek izveidots tensors.
~~~
self.X = []
self.inputSize = 0
for column in data:
    values = pd.Categorical(data[column]).codes
    tensor = torch.tensor(pd.get_dummies(values).values, dtype=torch.float32)
    self.X.append(tensor)
    self.inputSize += len(data[column].unique())
~~~

Forward cikla sakumā tensori tiek apvienoti vienā garā vektorā: ar shape Bx117(katra unikālā vērtība katrai kategorijai, piem [0.,0.,0.,1.,0.,...])
~~~
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
~~~

2. Pašlaik BBB un MC-D implementācijas sasniedz ~70-90% accuracy un neuzlabojas to algoritmam ejot uz priekšu

Novērotie rezultāti līdz šim:

### BBB:
![bbb](media/bbb_result.PNG)


### MC-D:
![mcd](media/mcd_result.PNG)

3. Kā implementēts label mixup:

* tāpēc ka dataloader subsets nevar izmainīt, tas tiek darīts dataset klasē:
~~~ 
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
  ~~~

4. Vai bayes inference implementēts pareizi?