---
execute:
  cache: false
  eval: true
  echo: true
  warning: false
jupyter: python3
---

# Physics Informed Neural Networks {#sec-pinn-601}

## PINNs
```{python}
#| label: import_601_pinns
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as thdat
import functools
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()
torch.manual_seed(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# boundaries for the frequency range
a = 0
b = 500
```

## Generation and Visualization of the Training Data and the Ground Truth (Function)

* Definition of the (unknown) differential equation:

```{python}
#| label: ode_601_pinns
def ode(frequency, loc, sigma, R):
    """Computes the amplitude. Defining equation, used
    to generate data and train models.
    The equation itself is not known to the model.

    Args:
        frequency: (N,) array-like
        loc: float
        sigma: float
        R: float
    
    Returns:
        (N,) array-like
    
    Examples:
        >>> ode(0, 25, 100, 0.005)
        100.0
    """
    A = np.exp(-R * (frequency - loc)**2/sigma**2)
    return A
```

* Setting the parameters for the ode

```{python}
#| label: parameters_601_pinns
np.random.seed(10)
loc = 250
sigma = 100
R = 0.5
```

* Generating the data

```{python}
#| label: amp_data_601_pinns
frequencies = np.linspace(a, b, 1000)
eq = functools.partial(ode, loc=loc, sigma=sigma, R=R)
amplitudes = eq(frequencies)
```

* Now we have the ground truth for the full frequency range and can take a look at the first 10 values:

```{python}
#| label: first_10_601_pinns
import pandas as pd
df = pd.DataFrame({'Frequency': frequencies[:10], 'Amplitude': amplitudes[:10]})
print(df)
```

* We generate the training data as a subset of the full frequency range and add some noise:

```{python}
#| label: training_data_601_pinns
t = np.linspace(a, 2*b/3, 10)
A = eq(t) +  0.2 * np.random.randn(10)
```

* Plot of the training data and the ground truth:

```{python}
#| label: fig-plot_data_601_pinns
plt.plot(frequencies, amplitudes)
plt.plot(t, A, 'o')
plt.legend(['Equation (ground truth)', 'Training data'])
plt.ylabel('Amplitude')
plt.xlabel('Frequency')
```

## Gradient With Autograd

```{python}
#| label: autograd_601_pinns
def grad(outputs, inputs):
    """Computes the partial derivative of 
    an output with respect to an input.

    Args:
        outputs: (N, 1) tensor
        inputs: (N, D) tensor

    Returns:
        (N, D) tensor
    
    Examples:
        >>> x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        >>> y = x**2
        >>> grad(y, x)
        tensor([2., 4., 6.])
    """
    return torch.autograd.grad(
        outputs, inputs, grad_outputs=torch.ones_like(outputs), create_graph=True
    )
```

* Autograd example:

```{python}
#| label: autograd_example_601_pinns
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x**2
grad(y, x)
```

## Network

```{python}
#| label: numpy2torch_601_pinns
def numpy2torch(x):
    """Converts a numpy array to a pytorch tensor.

    Args:
        x: (N, D) array-like

    Returns:
        (N, D) tensor

    Examples:
        >>> numpy2torch(np.array([1,2,3]))
        tensor([1., 2., 3.])
    """
    n_samples = len(x)
    return torch.from_numpy(x).to(torch.float).to(DEVICE).reshape(n_samples, -1)
```

```{python}
#| label: Net_601_pinns
class Net(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        n_units=100,
        epochs=1000,
        loss=nn.MSELoss(),
        lr=1e-3,
        loss2=None,
        loss2_weight=0.1,
    ) -> None:
        super().__init__()

        self.epochs = epochs
        self.loss = loss
        self.loss2 = loss2
        self.loss2_weight = loss2_weight
        self.lr = lr
        self.n_units = n_units

        self.layers = nn.Sequential(
            nn.Linear(input_dim, self.n_units),
            nn.ReLU(),
            nn.Linear(self.n_units, self.n_units),
            nn.ReLU(),
            nn.Linear(self.n_units, self.n_units),
            nn.ReLU(),
            nn.Linear(self.n_units, self.n_units),
            nn.ReLU(),
        )
        self.out = nn.Linear(self.n_units, output_dim)

    def forward(self, x):
        h = self.layers(x)
        out = self.out(h)
        return out

    def fit(self, X, y):
        Xt = numpy2torch(X)
        yt = numpy2torch(y)

        optimiser = optim.Adam(self.parameters(), lr=self.lr)
        self.train()
        losses = []
        for ep in range(self.epochs):
            optimiser.zero_grad()
            outputs = self.forward(Xt)
            loss = self.loss(yt, outputs)
            if self.loss2:
                loss += self.loss2_weight + self.loss2_weight * self.loss2(self)
            loss.backward()
            optimiser.step()
            losses.append(loss.item())
            if ep % int(self.epochs / 10) == 0:
                print(f"Epoch {ep}/{self.epochs}, loss: {losses[-1]:.2f}")
        return losses

    def predict(self, X):
        self.eval()
        out = self.forward(numpy2torch(X))
        return out.detach().cpu().numpy()
```

* Extended network for parameter estimation of parameter `r`:

```{python}
#| label: PINNParam_601_pinns_ext
class PINNParam(Net):
    def __init__(
        self,
        input_dim,
        output_dim,
        n_units=100,
        epochs=1000,
        loss=nn.MSELoss(),
        lr=0.001,
        loss2=None,
        loss2_weight=0.1,
    ) -> None:
        super().__init__(
            input_dim, output_dim, n_units, epochs, loss, lr, loss2, loss2_weight
        )

        self.r = nn.Parameter(data=torch.tensor([1.]))
        self.sigma = nn.Parameter(data=torch.tensor([100.]))
        self.loc = nn.Parameter(data=torch.tensor([100.]))
```

## Basic Neutral Network

* Network without regularization:

```{python}
#| label: fig-net_601_pinns_wo_reg
net = Net(1,1, loss2=None, epochs=2000, lr=1e-5).to(DEVICE)

losses = net.fit(t, A)

plt.plot(losses)
plt.yscale('log')
```

* Adding L2 regularization:

```{python}
#| label: l2_reg_601_pinns_reg
def l2_reg(model: torch.nn.Module):
    """L2 regularization for the model parameters.

    Args:
        model: torch.nn.Module

    Returns:
        torch.Tensor

    Examples:
        >>> l2_reg(Net(1,1))
        tensor(0.0001, grad_fn=<SumBackward0>)
    """
    return torch.sum(sum([p.pow(2.) for p in model.parameters()]))
```

```{python}
#| label: fig-net_601_pinns_reg
netreg = Net(1,1, loss2=l2_reg, epochs=20000, lr=1e-5, loss2_weight=.1).to(DEVICE)
losses = netreg.fit(t, A)
plt.plot(losses)
plt.yscale('log')
```

```{python}
#| label: fig-plot_results_601_pinns
predsreg = netreg.predict(frequencies)
preds = net.predict(frequencies)
plt.plot(frequencies, amplitudes, alpha=0.8)
plt.plot(t, A, 'o')
plt.plot(frequencies, preds, alpha=0.8)
plt.plot(frequencies, predsreg, alpha=0.8)

plt.legend(labels=['Equation','Training data', 'Network', 'L2 Regularization Network'])
plt.ylabel('Amplitude')
plt.xlabel('Frequency')
```

## PINNs

* Calculate the physics-informed loss (similar to the L2 regularization):

```{python}
#| label: physics_loss_601_pinns
def physics_loss(model: torch.nn.Module):
    """Computes the physics-informed loss for the model.

    Args:
        model: torch.nn.Module

    Returns:
        torch.Tensor

    Examples:
        >>> physics_loss(Net(1,1))
        tensor(0.0001, grad_fn=<MeanBackward0>)
    """
    ts = torch.linspace(a, b, steps=1000).view(-1,1).requires_grad_(True).to(DEVICE)
    amplitudes = model(ts)
    dT = grad(amplitudes, ts)[0]
    ode = -2*R*(ts-loc)/ sigma**2 * amplitudes - dT
    return torch.mean(ode**2)
```

* Train the network with the physics-informed loss and plot the training error:

```{python}
#| label: fig-net_601_pinns_pinn_plot_loss
net_pinn = Net(1,1, loss2=physics_loss, epochs=2000, loss2_weight=1, lr=1e-5).to(DEVICE)
losses = net_pinn.fit(t, A)
plt.plot(losses)
plt.yscale('log')
```

* Predict the amplitude and plot the results:

```{python}
#| label: fig-net_601_pinns_pinn_plot_results
preds_pinn = net_pinn.predict(frequencies)
plt.plot(frequencies, amplitudes, alpha=0.8)
plt.plot(t, A, 'o')
plt.plot(frequencies, preds, alpha=0.8)
plt.plot(frequencies, predsreg, alpha=0.8)
plt.plot(frequencies, preds_pinn, alpha=0.8)
plt.legend(labels=['Equation','Training data', 'NN', "R2", 'PINN'])
plt.ylabel('Amplitude')
plt.xlabel('Frequency')
```

### PINNs: Parameter Estimation

```{python}
#| label: pinn_param_601_pinns_param_est
def physics_loss_estimation(model: torch.nn.Module):
    ts = torch.linspace(a, b, steps=1000,).view(-1,1).requires_grad_(True).to(DEVICE)
    amplitudes = model(ts)
    dT = grad(amplitudes, ts)[0]
    ode = -2*model.r*(ts-model.loc)/ (model.sigma)**2 * amplitudes - dT
    return torch.mean(ode**2)
```

```{python}
#| label: fig-pinn_param_601_pinns_param_est
pinn_param = PINNParam(1, 1, loss2=physics_loss_estimation, loss2_weight=1, epochs=4000, lr= 5e-6).to(DEVICE)
losses = pinn_param.fit(t, A)
plt.plot(losses)
plt.yscale('log')
```

```{python}
#| label: print-pinn_param_601_pinns_param_est_plot_results
preds_disc = pinn_param.predict(frequencies)
print(f"Estimated r: {pinn_param.r}")
print(f"Estimated sigma: {pinn_param.sigma}")
print(f"Estimated loc: {pinn_param.loc}")
```

```{python}
#| label: fig-pinn_param_601_pinns_param_est_plot_results
plt.plot(frequencies, amplitudes, alpha=0.8)
plt.plot(t, A, 'o')
plt.plot(frequencies, preds_disc, alpha=0.8)
plt.legend(labels=['Equation','Training data', 'estimation PINN'])
plt.ylabel('Amplitude')
plt.xlabel('Frequency')
```

```{python}
#| label: fig-pinn_param_601_pinns_param_est_amplitude
plt.plot(frequencies, amplitudes, alpha=0.8)
plt.plot(t, A, 'o')
plt.plot(frequencies, preds, alpha=0.8)
plt.plot(frequencies, predsreg, alpha=0.8)
plt.plot(frequencies, preds_pinn, alpha=0.8)
plt.plot(frequencies, preds_disc, alpha=0.8)
plt.legend(labels=['Equation','Training data', 'NN', "R2", 'PINN', 'paramPINN'])
plt.ylabel('Amplitude')
plt.xlabel('Frequency')
```

```{python}
#| label: fig-pinn_param_601_pinns_param_est_plot_results_all
plt.plot(frequencies, amplitudes, alpha=0.8)
plt.plot(t, A, 'o')
plt.plot(frequencies, preds, alpha=0.8)
plt.plot(frequencies, predsreg, alpha=0.8)
plt.plot(frequencies, preds_disc, alpha=0.8)
plt.legend(labels=['Equation','Training data', 'NN', "R2", 'paramPINN'])
plt.ylabel('Amplitude')
plt.xlabel('Frequency')
```

```{python}
#| label: fig-pinn_param_601_pinns_param_ground
plt.plot(frequencies, amplitudes, alpha=0.8)
plt.plot(t, A, 'o')
plt.plot(frequencies, preds, alpha=0.8)
plt.plot(frequencies, predsreg, alpha=0.8)
plt.plot(frequencies, preds_disc, alpha=0.8)
plt.legend(labels=['Grundwahrheit','Trainingsdaten', 'NN', "NN+R2", 'PINN'])
plt.ylabel('Amplitude')
plt.xlabel('Frequenz')
# save the plot as a pdf
plt.savefig('pinns.pdf')
plt.savefig('pinns.png')
```

## Summary

* Results strongly depend on the parametrization(s)
* PINN parameter estimation not robust
* Hyperparameter tuning is crucial
* Use SPOT before further analysis is done

