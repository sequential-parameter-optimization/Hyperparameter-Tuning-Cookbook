---
execute:
  cache: false
  eval: true
  echo: true
  warning: false
title: Hyperparameter Tuning with `spotpython` and `PyTorch` Lightning Using a MultiNet Model
jupyter: python3
---

```{python}
#| label: 606_user-user-imports
#| echo: false
import os
from math import inf
import warnings
warnings.filterwarnings("ignore")
```

* First, we generate an 80-dim dataframe with 10000 samples, where the first two columns are random integers and the rest are random floats.
* Then, we generate a target variable as the sum of the squared values.
* The dataframe is converted to a tensor and split into a training, validation, and testing set. The corresponding data loaders are created.

```{python}
import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
np.random.seed(42)
n_samples = 10_000
n_int = 2
n_float = 76
input_dim = n_int + n_float
output_dim = 1
data = np.random.rand(n_samples, n_float)
data = np.hstack((np.random.randint(0, 10, (n_samples, n_int)), data))
df = pd.DataFrame(data)
df['y'] = np.sum(df.iloc[:, 2:]**2, axis=1)
df.head()
X = torch.tensor(df.iloc[:, :-1].values, dtype=torch.float32)
y = torch.tensor(df.iloc[:, -1].values, dtype=torch.float32)
dataset = TensorDataset(X, y)
print(f"Dataset with input tensor shape: {dataset.tensors[0].shape}")
print(f"Dataset with target tensor shape: {dataset.tensors[1].shape}")
# print(dataset[0][0])
# print(dataset[0][1])
train_size_0 = int(0.8 * len(dataset))
train_size = int(0.8 * train_size_0)
val_size = train_size_0 - train_size
test_size = len(dataset) - train_size_0
train_dataset_0, test_dataset = random_split(dataset, [train_size_0, test_size])
train_dataset, val_dataset = random_split(train_dataset_0, [train_size, val_size])
BATCH_SIZE = 128
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
```

* We define a standard neural network which will be used as the base model for the MultiNet model.

```{python}
from spotpython.light.regression import NNLinearRegressor
from torch import nn
import lightning as L
batch_x, batch_y = next(iter(train_loader))
print(batch_x.shape)
print(batch_y.shape)
net_light_base = NNLinearRegressor(l1=128,
                                    batch_norm=True,
                                    epochs=10,
                                    batch_size=BATCH_SIZE,
                                    initialization='xavier',
                                    act_fn=nn.ReLU(),
                                    optimizer='Adam',
                                    dropout_prob=0.1,
                                    lr_mult=0.1,
                                    patience=5,
                                    _L_in=input_dim,
                                    _L_out=output_dim,
                                    _torchmetric="mean_squared_error",)
trainer = L.Trainer(max_epochs=2,  enable_progress_bar=True)
trainer.fit(net_light_base, train_loader)
trainer.validate(net_light_base, val_loader)
trainer.test(net_light_base, test_loader)
```

