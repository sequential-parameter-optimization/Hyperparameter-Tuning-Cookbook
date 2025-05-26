---
execute:
  cache: false
  eval: true
  echo: true
  warning: false
---

# Matrices

## Derivatives of Quadratic Forms {#sec-derivative-quadratic-form}

We present a step-by-step derivation of the general formula 
$$
\frac{\partial}{\partial \mathbf{v}} (\mathbf{v}^T \mathbf{A} \mathbf{v}) = \mathbf{A} \mathbf{v} + \mathbf{A}^T \mathbf{v}.
$$ {#eq-derivative-quadratic-form}

1. Define the components. Let $\mathbf{v}$ be a vector of size $n \times 1$, and let $\mathbf{A}$ be a matrix of size $n \times n$.
2. Write out the quadratic form in summation notation. The product $\mathbf{v}^T \mathbf{A} \mathbf{v}$ is a scalar. It can be expanded and be rewritten as a double summation: 
$$
\mathbf{v}^T \mathbf{A} \mathbf{v} = \sum_{i=1}^n \sum_{j=1}^n v_i a_{ij} v_j.
$$

3. Calculate the partial derivative with respect to a component $v_k$: The derivative of the scalar $\mathbf{v}^T \mathbf{A} \mathbf{v}$ with respect to the vector $\mathbf{v}$ is the gradient vector, whose $k$-th component is $\frac{\partial}{\partial v_k} (\mathbf{v}^T \mathbf{A} \mathbf{v})$. We need to find $\frac{\partial}{\partial v_k} \left( \sum_{i=1}^n \sum_{j=1}^n v_i a_{ij} v_j \right)$. Consider the terms in the summation that involve $v_k$. A term $v_i a_{ij} v_j$ involves $v_k$ if $i=k$ or $j=k$ (or both).
    * Terms where $i=k$: $v_k a_{kj} v_j$. The derivative with respect to $v_k$ is $a_{kj} v_j$.
    * Terms where $j=k$: $v_i a_{ik} v_k$. The derivative with respect to $v_k$ is $v_i a_{ik}$.
    * The term where $i=k$ and $j=k$: $v_k a_{kk} v_k = a_{kk} v_k^2$. Its derivative with respect to $v_k$ is $2 a_{kk} v_k$. Notice this term is included in both cases above when $i=k$ and $j=k$. When $i=k$, the term is $v_k a_{kk} v_k$, derivative is $a_{kk} v_k$. When $j=k$, the term is $v_k a_{kk} v_k$, derivative is $v_k a_{kk}$. Summing these two gives $2 a_{kk} v_k$.

4. Let's differentiate the sum $\sum_{i=1}^n \sum_{j=1}^n v_i a_{ij} v_j$ with respect to $v_k$: 
$$
\frac{\partial}{\partial v_k} \left( \sum_{i=1}^n \sum_{j=1}^n v_i a_{ij} v_j \right) = \sum_{i=1}^n \sum_{j=1}^n \frac{\partial}{\partial v_k} (v_i a_{ij} v_j).
$$

5. The partial derivative $\frac{\partial}{\partial v_k} (v_i a_{ij} v_j)$ is non-zero only if $i=k$ or $j=k$.
    * If $i=k$ and $j \ne k$: $\frac{\partial}{\partial v_k} (v_k a_{kj} v_j) = a_{kj} v_j$.
    * If $i \ne k$ and $j = k$: $\frac{\partial}{\partial v_k} (v_i a_{ik} v_k) = v_i a_{ik}$.
    * If $i=k$ and $j=k$: $\frac{\partial}{\partial v_k} (v_k a_{kk} v_k) = \frac{\partial}{\partial v_k} (a_{kk} v_k^2) = 2 a_{kk} v_k$.

6. So, the partial derivative is the sum of derivatives of all terms involving $v_k$: $\frac{\partial}{\partial v_k} (\mathbf{v}^T \mathbf{A} \mathbf{v}) = \sum_{j \ne k} (a_{kj} v_j) + \sum_{i \ne k} (v_i a_{ik}) + (2 a_{kk} v_k)$.

7. We can rewrite this by including the $i=k, j=k$ term back into the summations: $\sum_{j \ne k} (a_{kj} v_j) + a_{kk} v_k + \sum_{i \ne k} (v_i a_{ik}) + v_k a_{kk}$ (since $v_k a_{kk} = a_{kk} v_k$) $= \sum_{j=1}^n a_{kj} v_j + \sum_{i=1}^n v_i a_{ik}$.

8. Convert back to matrix/vector notation: The first summation $\sum_{j=1}^n a_{kj} v_j$ is the $k$-th component of the matrix-vector product $\mathbf{A} \mathbf{v}$.The second summation $\sum_{i=1}^n v_i a_{ik}$ can be written as $\sum_{i=1}^n a_{ik} v_i$. Recall that the element in the $k$-th row and $i$-th column of the transpose matrix $\mathbf{A}^T$ is $(A^T)_{ki} = a_{ik}$. So, $\sum_{i=1}^n a_{ik} v_i = \sum_{i=1}^n (A^T)_{ki} v_i$, which is the $k$-th component of the matrix-vector product $\mathbf{A}^T \mathbf{v}$.

9. Assemble the gradient vector: The $k$-th component of the gradient $\frac{\partial}{\partial \mathbf{v}} (\mathbf{v}^T \mathbf{A} \mathbf{v})$ is $(\mathbf{A} \mathbf{v})_k + (\mathbf{A}^T \mathbf{v})_k$. Since this holds for all $k = 1, \dots, n$, the gradient vector is the sum of the two vectors $\mathbf{A} \mathbf{v}$ and $\mathbf{A}^T \mathbf{v}$.
Therefore, the general formula for the derivative is $\frac{\partial}{\partial \mathbf{v}} (\mathbf{v}^T \mathbf{A} \mathbf{v}) = \mathbf{A} \mathbf{v} + \mathbf{A}^T \mathbf{v}$.



## The Condition Number {#sec-conditon-number}

A small value, `eps`, can be passed to the function `build_Psi` to improve the condition number. For example, `eps=sqrt(spacing(1))` can be used. The numpy function `spacing()` returns the distance between a number and its nearest adjacent number.

The condition number of a matrix is a measure of its sensitivity to small changes in its elements. It is used to estimate how much the output of a function will change if the input is slightly altered.

A matrix with a low condition number is well-conditioned, which means its behavior is relatively stable, while a matrix with a high condition number is ill-conditioned, meaning its behavior is unstable with respect to numerical precision.

```{python}
import numpy as np

# Define a well-conditioned matrix (low condition number)
A = np.array([[1, 0.1], [0.1, 1]])
print("Condition number of A: ", np.linalg.cond(A))

# Define an ill-conditioned matrix (high condition number)
B = np.array([[1, 0.99999999], [0.99999999, 1]])
print("Condition number of B: ", np.linalg.cond(B))
```


## The Moore-Penrose Pseudoinverse {#sec-matrix-pseudoinverse}

### Definitions

The Moore-Penrose pseudoinverse is a generalization of the inverse matrix for non-square or singular matrices.
It is computed as

$$
A^+ = (A^* A)^{-1} A^*,
$$
where $A^*$ is the conjugate transpose of $A$.

It satisfies the following properties:

1. $AA^+A = A$
2. $A^+AA^+ = A^+$
3. $(AA^+)^* = AA^+$.
4. $(A^+A)^* = A^+A$
5. $A^+ = (A^*)^+$
6. $A^+ = A^T$ if $A$ is a square matrix and $A$ is invertible.

The pseudoinverse can be computed using Singular Value Decomposition (SVD).

### Implementation in Python
```{python}
import numpy as np
from numpy.linalg import pinv
A = np.array([[1, 2], [3, 4], [5, 6]])
print(f"Matrix A:\n {A}")
A_pseudo_inv = pinv(A)
print(f"Moore-Penrose Pseudoinverse:\n {A_pseudo_inv}")
```


## Strictly Positive Definite Kernels {#sec-strictly-positive-definite}

### Definition

::: {#def-strictly-positive-definite}
### Strictly Positive Definite Kernel

A kernel function $k(x,y)$ is called strictly positive definite if for any finite collection of distinct points ${x_1, x_2, \ldots, x_n}$ in the input space and any non-zero vector of coefficients $\alpha = (\alpha_1, \alpha_2, \ldots, \alpha_n)$, the following inequality holds:

$$
\sum_{i=1}^{n} \sum_{j=1}^{n} \alpha_i \alpha_j k(x_i, x_j) > 0.
$$ {#eq-strictly-positive-definite}
:::

In contrast, a kernel function $k(x,y)$ is called positive definite (but not strictly) if the "$>$" sign is replaced by "$\geq$" in the above inequality.


### Connection to Positive Definite Matrices

The connection between strictly positive definite kernels and positive definite matrices lies in the Gram matrix construction:

* When we evaluate a kernel function $k(x,y)$ at all pairs of data points in our sample, we construct the Gram matrix $K$ where $K_{ij} = k(x_i, x_j)$.
* If the kernel function $k$ is strictly positive definite, then for any set of distinct points, the resulting Gram matrix will be symmetric positive definite.

A symmetric matrix is positive definite if and only if for any non-zero vector $\alpha$, the quadratic form $\alpha^T K \alpha > 0$, which directly corresponds to the kernel definition above.

### Connection to RBF Models

For RBF models, the kernel function is the radial basis function itself:
$$
k(x,y) = \psi(||x-y||).
$$

The Gaussian RBF kernel $\psi(r) = e^{-r^2/(2\sigma^2)}$ is strictly positive definite in $\mathbb{R}^n$ for any dimension $n$.
The inverse multiquadric kernel $\psi(r) = (r^2 + \sigma^2)^{-1/2}$ is also strictly positive definite in any dimension.

This mathematical property guarantees that the interpolation problem has a unique solution (the weight vector $\vec{w}$ is uniquely determined).
The linear system $\Psi \vec{w} = \vec{y}$ can be solved reliably using Cholesky decomposition.
The RBF interpolant exists and is unique for any distinct set of centers.



## Cholesky Decomposition and Positive Definite Matrices

We consider the definiteness of a matrix, before discussing the Cholesky decomposition.

::: {#def-positive-definite}
### Positive Definite Matrix

A symmetric matrix $A$ is positive definite if all its eigenvalues are positive.

:::

::: {#exm-positive-definite}
### Positive Definite Matrix

Given a symmetric matrix $A = \begin{pmatrix} 9 & 4 \\ 4 & 9 \end{pmatrix}$,
the eigenvalues of $A$ are $\lambda_1 = 13$ and $\lambda_2 = 5$.
Since both eigenvalues are positive, the matrix $A$ is positive definite.

:::

::: {#def-negative-definite}
### Negative Definite, Positive Semidefinite, and Negative Semidefinite Matrices

Similarily, a symmetric matrix $A$ is negative definite if all its eigenvalues are negative.
It is positive semidefinite if all its eigenvalues are non-negative, and negative semidefinite if all its eigenvalues are non-positive.

:::


The covariance matrix must be positive definite for a multivariate normal distribution for a couple of reasons:

* Semidefinite vs Definite: A covariance matrix is always symmetric and positive semidefinite. However, for a multivariate normal distribution, it must be positive definite, not just semidefinite. This is because a positive semidefinite matrix can have zero eigenvalues, which would imply that some dimensions in the distribution have zero variance, collapsing the distribution in those dimensions. A positive definite matrix has all positive eigenvalues, ensuring that the distribution has positive variance in all dimensions.
* Invertibility: The multivariate normal distribution's probability density function involves the inverse of the covariance matrix. If the covariance matrix is not positive definite, it may not be invertible, and the density function would be undefined.

In summary, the covariance matrix being positive definite ensures that the multivariate normal distribution is well-defined and has positive variance in all dimensions.


The definiteness of a matrix can be checked by examining the eigenvalues of the matrix. If all eigenvalues are positive, the matrix is positive definite.

```{python}
import numpy as np

def is_positive_definite(matrix):
    return np.all(np.linalg.eigvals(matrix) > 0)

matrix = np.array([[9, 4], [4, 9]])
print(is_positive_definite(matrix))  # Outputs: True
```

 However, a more efficient way to check the definiteness of a matrix is through the Cholesky decomposition.

::: {#def-cholesky-decomposition}
### Cholesky Decomposition
For a given symmetric positive-definite matrix $A \in \mathbb{R}^{n \times n}$,
there exists a unique lower triangular matrix $L \in \mathbb{R}^{n \times n}$ with positive diagonal elements such that:

$$
A = L L^T.
$$

Here, $L^T$ denotes the transpose of $L$.
:::



::: {#exm-cholesky-decomposition}
### Cholesky decomposition using `numpy`

`linalg.cholesky` computes the Cholesky decomposition of a matrix, i.e., it computes a lower triangular matrix $L$ such that $LL^T = A$. If the matrix is not positive definite, an error (`LinAlgError`) is raised.


```{python}
import numpy as np

# Define a Hermitian, positive-definite matrix
A = np.array([[9, 4], [4, 9]]) 

# Compute the Cholesky decomposition
L = np.linalg.cholesky(A)

print("L = \n", L)
print("L*LT = \n", np.dot(L, L.T))

```

::: 


::: {#exm-cholesky-decomposition}
### Cholesky Decomposition

Given a symmetric positive-definite matrix $A = \begin{pmatrix} 9 & 4 \\ 4 & 9 \end{pmatrix}$,
the Cholesky decomposition computes the lower triangular matrix $L$ such that $A = L L^T$.
The matrix $L$ is computed as:
$$
L = \begin{pmatrix} 3 & 0 \\ 4/3 & 2 \end{pmatrix},
$$
so that
$$
L L^T = \begin{pmatrix} 3 & 0 \\ 4/3 & \sqrt{65}/3 \end{pmatrix} \begin{pmatrix} 3 & 4/3 \\ 0 & \sqrt{65}/3 \end{pmatrix} = \begin{pmatrix} 9 & 4 \\ 4 & 9 \end{pmatrix} = A.
$$

:::

An efficient implementation of the definiteness-check based on Cholesky is already available in the `numpy` library.
It provides the `np.linalg.cholesky` function to compute the Cholesky decomposition of a matrix.
This more efficient `numpy`-approach can be used as follows:

```{python}
import numpy as np

def is_pd(K):
    try:
        np.linalg.cholesky(K)
        return True
    except np.linalg.linalg.LinAlgError as err:
        if 'Matrix is not positive definite' in err.message:
            return False
        else:
            raise
matrix = np.array([[9, 4], [4, 9]])
print(is_pd(matrix))  # Outputs: True
```






### Example of Cholesky Decomposition

We consider dimension $k=1$ and $n=2$ sample points.
The sample points are located at $x_1=1$ and $x_2=5$. 
The response values are $y_1=2$ and $y_2=10$.
The correlation parameter is $\theta=1$ and $p$ is set to $1$.
Using @eq-krigingbase, we can compute the correlation matrix $\Psi$:

$$
\Psi = \begin{pmatrix}
1 & e^{-1}\\
e^{-1} & 1
\end{pmatrix}.
$$

To determine MLE as in @eq-mle-yhat, we need to compute $\Psi^{-1}$:

$$
\Psi^{-1} = \frac{e}{e^2 -1} \begin{pmatrix}
e & -1\\
-1 & e
\end{pmatrix}.
$$

Cholesky-decomposition of $\Psi$ is recommended to compute $\Psi^{-1}$. Cholesky decomposition is a decomposition of a positive definite symmetric matrix into the product of a lower triangular matrix $L$, a diagonal matrix $D$ and the transpose of $L$, which is denoted as $L^T$.
Consider the following example:

$$
LDL^T=
\begin{pmatrix}
1 & 0 \\
l_{21} & 1
\end{pmatrix}
\begin{pmatrix}
d_{11} & 0 \\
0 & d_{22}
\end{pmatrix}
\begin{pmatrix}
1 & l_{21} \\
0 & 1
\end{pmatrix}=
$$

$$
\begin{pmatrix}
d_{11} & 0 \\
d_{11} l_{21} & d_{22}
\end{pmatrix}
\begin{pmatrix}
1 & l_{21} \\
0 & 1
\end{pmatrix}
=
\begin{pmatrix}
d_{11} & d_{11} l_{21} \\
d_{11} l_{21} & d_{11} l_{21}^2 + d_{22}
\end{pmatrix}.
$$ {#eq-cholex}



Using @eq-cholex, we can compute the Cholesky decomposition of $\Psi$:

1. $d_{11} = 1$,
2. $l_{21}d_{11} = e^{-1} \Rightarrow l_{21} = e^{-1}$, and
3. $d_{11} l_{21}^2 + d_{22} = 1 \Rightarrow d_{22} = 1 - e^{-2}$.

The Cholesky decomposition of $\Psi$ is
$$
\Psi = \begin{pmatrix}
1 & 0\\
e^{-1} & 1\\
\end{pmatrix}
\begin{pmatrix}
1 & 0\\
0 & 1 - e^{-2}\\
\end{pmatrix}
\begin{pmatrix}
1 & e^{-1}\\
0 & 1\\
\end{pmatrix}
= LDL^T$$

Some programs use $U$ instead of $L$. The Cholesky decomposition of $\Psi$ is
$$
\Psi = LDL^T = U^TDU.
$$

Using 
$$
\sqrt{D} =\begin{pmatrix}
1 & 0\\
0 & \sqrt{1 - e^{-2}}\\
\end{pmatrix},
$$
we can write the Cholesky decomposition of $\Psi$ without a diagonal matrix $D$ as
$$
\Psi = \begin{pmatrix}
1 & 0\\
e^{-1} & \sqrt{1 - e^{-2}}\\
\end{pmatrix}
\begin{pmatrix}
1 & e^{-1}\\
0 & \sqrt{1 - e^{-2}}\\
\end{pmatrix}
= U^TU.
$$


### Inverse Matrix Using Cholesky Decomposition

To compute the inverse of a matrix using the Cholesky decomposition, you can follow these steps:

1. Decompose the matrix $A$ into $L$ and $L^T$, where $L$ is a lower triangular matrix and $L^T$ is the transpose of $L$.
2. Compute $L^{-1}$, the inverse of $L$.
3. The inverse of $A$ is then $(L^{-1})^T  L^-1$.

Please note that this method only applies to symmetric, positive-definite matrices.

The inverse of the matrix $\Psi$ from above is:

$$
\Psi^{-1} = \frac{e}{e^2 -1} \begin{pmatrix}
e & -1\\
-1 & e
\end{pmatrix}.
$$


Here’s an example of how to compute the inverse of a matrix using Cholesky decomposition in Python:

```{python}
import numpy as np
from scipy.linalg import cholesky, inv
E = np.exp(1)

# Psi is a symmetric, positive-definite matrix 
Psi = np.array([[1, 1/E], [1/E, 1]])
L = cholesky(Psi, lower=True)
L_inv = inv(L)
# The inverse of A is (L^-1)^T * L^-1
Psi_inv = np.dot(L_inv.T, L_inv)

print("Psi:\n", Psi)
print("Psi Inverse:\n", Psi_inv)
```



## Jupyter Notebook

:::{.callout-note}

* The Jupyter-Notebook of this lecture is available on GitHub in the [Hyperparameter-Tuning-Cookbook Repository](https://github.com/sequential-parameter-optimization/Hyperparameter-Tuning-Cookbook/blob/main/006_matrices.ipynb)

:::


