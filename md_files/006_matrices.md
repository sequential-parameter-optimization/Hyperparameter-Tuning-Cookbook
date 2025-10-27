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

## Nyström Approximation

### What's the Big Idea?

Imagine you have a huge, detailed map of a country. Working with the full, high-resolution map is slow and takes up a lot of computer memory. The Nyström method is like creating a smaller-scale summary map by only looking at a few key, representative locations.

In machine learning, we often work with a **kernel matrix** (or Gram matrix), which tells us how similar every pair of data points is to each other. For very large datasets, this matrix can become massive, making it computationally expensive to store and process.

The Nyström method provides an efficient way to create a **low-rank approximation** of this large kernel matrix. In simple terms, it finds a "simpler" version of the matrix that captures its most important properties without needing to compute or store the whole thing.

---

### How Does It Work?

The core idea is to select a small, random subset of the columns of the full kernel matrix and use them to reconstruct the entire matrix. Let's say our full kernel matrix is $K$.

1. **Sample:** Randomly select $l$ columns from the $n$ total columns of $K$. Let $C$ be the $n \times l$ matrix of these sampled columns.
2. **Intersect:** Take the rows of $C$ corresponding to the sampled column indices to form the $l \times l$ matrix $W$.
3. **Approximate:** Using $C$ and $W$, calculate the Nyström approximation $\tilde{K}$ of $K$:
    $$
    \tilde{K} \approx C W^{+} C^T
    $$
    where $W^{+}$ is the pseudoinverse of $W$.

---

### Example

Suppose we have 4 data points and the full kernel matrix $K$ is:
$$
K = \begin{pmatrix}
9 & 6 & 3 & 1 \\
6 & 4 & 2 & 0.5 \\
3 & 2 & 1 & 0.25 \\
1 & 0.5 & 0.25 & 0.1
\end{pmatrix}
$$

Let's approximate it by sampling 2 columns ($l=2$):

1. **Sample:** Pick the 1st and 3rd columns:
    $$
    C = \begin{pmatrix}
    9 & 3 \\
    6 & 2 \\
    3 & 1 \\
    1 & 0.25
    \end{pmatrix}
    $$
2. **Intersect:** Take the 1st and 3rd rows from $C$ to form $W$:
    $$
    W = \begin{pmatrix}
    9 & 3 \\
    3 & 1
    \end{pmatrix}
    $$
3. **Approximate:** Suppose the pseudoinverse of $W$ is:
    $$
    W^{+} = \begin{pmatrix}
    0.09 & -0.27 \\
    -0.27 & 0.81
    \end{pmatrix}
    $$
    Then,
    $$
    \tilde{K} = C W^{+} C^T = \begin{pmatrix}
    9 & 6 & 3 & 0.675 \\
    6 & 4 & 2 & 0.45 \\
    3 & 2 & 1 & 0.225 \\
    0.675 & 0.45 & 0.225 & 0.05
    \end{pmatrix}
    $$

$\tilde{K}$ is a good approximation of the original $K$, especially in the top-left portion.

---

### Why Is This Useful?

- **Speed:** The Nyström method is much faster than computing the full kernel matrix. The complexity is roughly $O(l^2 n)$ instead of $O(n^2 d)$ (where $d$ is the number of features).
- **Scalability:** It allows kernel methods (like SVM or Kernel PCA) to be used on much larger datasets.
- **Feature Mapping:** The method can be used to project new data points into the same feature space for prediction tasks.

The quality of the approximation depends on the columns you sample. Uniform random sampling is common and often effective, but more advanced techniques exist to select more informative columns.

### Applying the Nyström Approximation: How Nyström Approximation Helps Kriging

Kriging can significantly benefit from the Nyström approximation, especially when dealing with large datasets.
Kriging is a spatial interpolation method used to estimate values at unmeasured locations based on observed points. It relies on a **covariance matrix** (often denoted as **K**) that describes the spatial correlation between all observed data points.

**The Problem with Standard Kriging:**

The main computational challenge in Kriging is solving for the weights needed for prediction, which requires **inverting the covariance matrix K**. For `n` data points, **K** is an `n x n` matrix, and inverting it has computational complexity $O(n^3)$. This becomes impractical for large datasets.

**The Nyström Solution:**

Since the covariance matrix in Kriging is a type of kernel matrix, we can use the Nyström method to create a low-rank approximation, $\tilde{K}$. Instead of inverting the full matrix, we use the **Woodbury matrix identity** on the Nyström approximation, allowing us to efficiently compute $\tilde{K}^{-1}$ without forming the full matrix. This reduces computational complexity to roughly $O(l^2 n)$, where `l` is the number of sampled columns.

In summary, Nyström makes Kriging feasible for large-scale problems by replacing expensive matrix inversion with a faster, memory-efficient approximation.

---

### Example: Predicting Temperature with Nyström-Kriging

Suppose we have temperature readings from 100 weather stations (`n=100`) and want to predict the temperature at a new location.

**Data:**

* Observed Locations (X): 100 coordinate pairs
* Observed Temperatures (y): 100 values
* Prediction Location (x*): Coordinates of the new location

#### Step 1: Nyström Approximation of the Covariance Matrix

1. **Sample Representative Points:** Randomly select `l=10` stations as landmarks.
2. **Compute C and W:**
    - **C:** Covariance between all 100 stations and the 10 landmarks (`100x10` matrix)
    - **W:** Covariance among the 10 landmarks (`10x10` matrix)

Nyström approximation: $\tilde{K} = C W^{+} C^T$

#### Step 2: Modeling and Prediction

Standard Kriging prediction:
$$
y(x^*) = \mathbf{k}^{*T} \mathbf{K}^{-1} \mathbf{y}
$$
where $\mathbf{k}^{*T}$ is the covariance vector between the prediction location and all observed locations.

Nyström-Kriging prediction:
$$
y(x^*) \approx \mathbf{k}^{*T} (\text{fast\_approx\_inverse}(\mathbf{C}, \mathbf{W})) \mathbf{y}
$$

**Prediction Steps:**

1. Calculate $\mathbf{k}^{*T}$: Covariance between new location and all stations.
2. Approximate the inverse term using the Woodbury identity with **C** and **W**.
3. Make the prediction: Take the dot product of $\mathbf{k}^{*T}$ and the weights vector.

This yields an accurate prediction efficiently, enabling rapid mapping for large regions.

### Details: Woodbury Matrix Identity for Avoiding the Big Inversion

First, what is the **Woodbury matrix identity**? It's a mathematical rule that tells you how to find the inverse of a matrix that's been modified slightly. Its most useful form is for a "low-rank update":

$$
(A + UCV)^{-1} = A^{-1} - A^{-1}U(C^{-1} + VA^{-1}U)^{-1}VA^{-1}
$$

This looks complicated, but the core idea is simple:

- If you have a matrix $A$ that is **easy to invert** (like a diagonal matrix).
- And you add a low-rank matrix to it (the $UCV$ part, where $C$ is small).
- You can find the new inverse without directly inverting the big $(A + UCV)$ matrix. Instead, you only need to invert the much smaller matrix in the middle of the formula: $(C^{-1} + VA^{-1}U)$.

**How does this apply to the Nyström approximation?**

In many machine learning and Kriging applications, we don't just need the kernel matrix $\tilde{K}$, but a "regularized" version, $(\lambda I + \tilde{K})$, where $\lambda I$ is a diagonal matrix that helps prevent overfitting. We need to find the inverse of this:

$$
(\lambda I + \tilde{K})^{-1}
$$

Substituting the Nyström formula $\tilde{K} = C W^{+} C^T$, we get:

$$
(\lambda I + C W^{+} C^T)^{-1}
$$

This expression fits the Woodbury identity perfectly!

- $A = \lambda I$ (very easy to invert: $A^{-1} = \frac{1}{\lambda}I$)
- $U = C$ (our $n \times l$ matrix)
- $C$ (middle matrix) $= W^{+}$ (our small $l \times l$ matrix)
- $V = C^T$ (our $l \times n$ matrix)

By plugging these into the Woodbury formula, we get an expression for the inverse that only requires inverting a small $l \times l$ matrix. This means we never have to build the full $n \times n$ matrix $\tilde{K}$ or invert it directly. This is the source of the massive speed-up.

---

### The Example: Step-by-Step

Let's reuse our 4-point example and show both the slow way and the fast Woodbury way.

**Recall our matrices:**

- $C = \begin{pmatrix} 9 & 3 \\ 6 & 2 \\ 3 & 1 \\ 1 & 0.25 \end{pmatrix}$
- $W^{+} = \begin{pmatrix} 0.09 & -0.27 \\ -0.27 & 0.81 \end{pmatrix}$
- Let's use a regularization value $\lambda = 0.1$.

#### Method 1: The Slow Way (Forming the full matrix)

1. **Construct $\tilde{K}$:** First, we explicitly calculate the full $4 \times 4$ Nyström approximation $\tilde{K} = C W^{+} C^T$.

    $$
    \tilde{K} = \begin{pmatrix}
    9 & 6 & 3 & 0.675 \\
    6 & 4 & 2 & 0.45 \\
    3 & 2 & 1 & 0.225 \\
    0.675 & 0.45 & 0.225 & 0.05
    \end{pmatrix}
    $$

2. **Add the regularization:** Now we compute $(\lambda I + \tilde{K})$.

    $$
    (\lambda I + \tilde{K}) = \begin{pmatrix}
    9.1 & 6 & 3 & 0.675 \\
    6 & 4.1 & 2 & 0.45 \\
    3 & 2 & 1.1 & 0.225 \\
    0.675 & 0.45 & 0.225 & 0.15
    \end{pmatrix}
    $$

3. **Invert the $4 \times 4$ matrix:** This is the expensive step. The result is:

    $$
    (\lambda I + \tilde{K})^{-1} \approx
    \begin{pmatrix}
    9.85 & -14.78 & -0.07 & 0.27 \\
    -14.78 & 22.22 & 0.09 & -0.41 \\
    -0.07 & 0.09 & 0.91 & -0.03 \\
    0.27 & -0.41 & -0.03 & 6.67
    \end{pmatrix}
    $$

This works for our tiny $4 \times 4$ example, but it would be computationally infeasible if $n$ was 10,000.

#### Method 2: The Fast Way (Using Woodbury Identity)

We use the Woodbury formula to get the same result without ever creating a $4 \times 4$ matrix. The formula simplifies to:

$$
(\lambda I + \tilde{K})^{-1} = \frac{1}{\lambda}I - \frac{1}{\lambda^2} C \left(W + \frac{1}{\lambda}C^T C\right)^{-1} C^T
$$

1. **Compute the small $2 \times 2$ pieces:**
    - $C^T C = \begin{pmatrix} 127 & 42.25 \\ 42.25 & 14.0625 \end{pmatrix}$
    - $W = \begin{pmatrix} 9 & 3 \\ 3 & 1 \end{pmatrix}$
    - The matrix to invert is $W + \frac{1}{0.1}C^T C = W + 10 \cdot (C^T C)$, which is:
      $$
      \begin{pmatrix} 9 & 3 \\ 3 & 1 \end{pmatrix} +
      \begin{pmatrix} 1270 & 422.5 \\ 422.5 & 140.625 \end{pmatrix} =
      \begin{pmatrix} 1279 & 425.5 \\ 425.5 & 141.625 \end{pmatrix}
      $$

2. **Invert the small $2 \times 2$ matrix:** This is the only inversion we need, and it's extremely fast.

    $$
    (W + \frac{1}{\lambda}C^T C)^{-1} \approx
    \begin{pmatrix}
    0.22 & -0.66 \\
    -0.66 & 1.99
    \end{pmatrix}
    $$

3. **Combine the results:** Now we plug this small inverse back into the full formula. The rest is just matrix multiplication, no more inversions.

    - First, calculate the middle term: $M = \frac{1}{\lambda^2} C (\dots)^{-1} C^T$. This will result in a $4 \times 4$ matrix.
    - Then, calculate the final result: $\frac{1}{\lambda}I - M$.

After performing these multiplications, you will get the **exact same $4 \times 4$ inverse matrix** as in the slow method.

The crucial difference is that the most expensive operation—the matrix inversion—was performed on a tiny $2 \times 2$ matrix instead of a $4 \times 4$ one. For a large-scale problem, this is the difference between a calculation that takes seconds and one that could take hours or even be impossible.


## Extending spotpython’s Kriging Surrogate with Nyström Approximation for Enhanced Scalability

### Introduction: Overcoming the Scalability Challenge in Kriging for Sequential Optimization

The Sequential Parameter Optimization Toolbox (spotpython) is a framework for hyperparameter tuning and black-box optimization based on Sequential Model-Based Optimization (SMBO). At the core of SMBO lies a surrogate model that approximates the true, expensive objective. Kriging (Gaussian Process regression) is a premier choice because it provides both predictions and a principled measure of uncertainty. This uncertainty enables a balance between exploration and exploitation. In each SMBO iteration, the Kriging model is updated with new evaluations, refining its approximation and proposing the next points.

Standard Kriging requires constructing and inverting an $n \times n$ covariance matrix, where $n$ is the number of data points. Matrix inversion scales as $O(n^3)$. During SMBO, $n$ can reach hundreds or thousands; refitting the surrogate each iteration becomes prohibitively expensive. This cubic scaling is the key obstacle to applying Kriging at larger scales.

We integrate the Nyström method into the spotpython Kriging class. The Nyström method yields a low-rank approximation of a symmetric positive semidefinite (SPSD) kernel matrix by selecting $l \ll n$ “landmark” points. It approximates the full $n \times n$ covariance while requiring inversion of only an $l \times l$ matrix, reducing fitting cost from $O(n^3)$ to $O(n\,l^2)$. This makes Kriging viable even when the number of function evaluations is large.

### Report Objectives and Structure
- Review theoretical foundations of Kriging and Nyström approximation
- Present documented Python code updates for Kriging (as in kriging.py)
- Explain changes to `__init__`, `fit`, and `predict`
- Show how mixed variable types are preserved via `build_Psi` and `build_psi_vec`
- Provide practical usage guidance and a formal complexity analysis

---

## Theoretical Foundations: The Nyström–Kriging Framework

### A Primer on Kriging (Gaussian Process Regression)
Kriging models $f(x)$ as a Gaussian Process with mean function $m(\cdot)$ and covariance (kernel) $k(\cdot,\cdot)$. For training inputs $X = \{x_1,\dots,x_n\}$ and observations $y = \{y_1,\dots,y_n\}$:
$$
y \sim \mathcal{N}\!\big(m(X),\, K(X,X) + \sigma_n^2 I\big)
$$
For a new point $x_\ast$:
$$
\mu(x_\ast) = k(x_\ast, X)\,[K(X,X) + \sigma_n^2 I]^{-1} y
$$
$$
\sigma^2(x_\ast) = k(x_\ast, x_\ast) - k(x_\ast, X)\,[K(X,X) + \sigma_n^2 I]^{-1} k(X, x_\ast)
$$
The challenge is inverting the $n \times n$ matrix $K(X,X) + \sigma_n^2 I$.

### The Nyström Method for Low-Rank Kernel Approximation
Select $l$ landmark points $X_m \subset X$. Let:
- $C = K_{nm} = K(X, X_m) \in \mathbb{R}^{n \times l}$
- $W = K_{mm} = K(X_m, X_m) \in \mathbb{R}^{l \times l}$
Then the Nyström approximation is:
$$
\tilde{K}_{nn} = C\,W^{+}\,C^\top = K_{nm}\,K_{mm}^{+}\,K_{mn}
$$
where $W^{+}$ is the pseudoinverse of $W$. The approximation has rank $\le l$.

### Justification for Landmark Selection
Uniform sampling without replacement is an effective and inexpensive strategy for selecting landmarks across varied datasets and kernels.

---

## Implementation: A Scalable Kriging Class for spotpython

### Updated kriging.py with Nyström Approximation (excerpt)

````python
"""
Kriging surrogate with optional Nyström approximation.
"""
import numpy as np
from scipy.spatial.distance import cdist
from scipy.linalg import cholesky, cho_solve, solve_triangular

class Kriging:
    def __init__(self, fun_control, n_theta=None, theta=None, p=2.0,
                 corr="squared_exponential", isotropic=False,
                 approximation="None", n_landmarks=100):
        self.fun_control = fun_control
        self.dim = self.fun_control["lower"].shape
        self.p = p
        self.corr = corr
        self.isotropic = isotropic
        self.approximation = approximation
        self.n_landmarks = n_landmarks
        self.factor_mask = self.fun_control["var_type"] == "factor"
        self.ordered_mask = ~self.factor_mask
        self.n_theta = 1 if isotropic else (n_theta or self.dim)
        self.theta = np.full(self.n_theta, 0.1) if theta is None else theta
        self.X_, self.y_, self.L_, self.alpha_ = None, None, None, None
        self.landmarks_, self.W_cho_, self.nystrom_alpha_ = None, None, None

    def fit(self, X, y):
        self.X_, self.y_ = X, y
        n_samples = X.shape[0]
        if self.approximation.lower() == "nystroem" and n_samples > self.n_landmarks:
            return self._fit_nystrom(X, y)
        return self._fit_standard(X, y)

    def _fit_standard(self, X, y):
        Psi = self.build_Psi(X, X)
        Psi[np.diag_indices_from(Psi)] += 1e-8
        try:
            self.L_ = cholesky(Psi, lower=True)
            self.alpha_ = cho_solve((self.L_, True), y)
        except np.linalg.LinAlgError:
            self.L_ = None
            self.alpha_ = np.linalg.pinv(Psi) @ y

    def _fit_nystrom(self, X, y):
        n_samples = X.shape[0]
        idx = np.random.choice(n_samples, self.n_landmarks, replace=False)
        self.landmarks_ = X[idx, :]
        W = self.build_Psi(self.landmarks_, self.landmarks_) + 1e-8 * np.eye(self.n_landmarks)
        C = self.build_Psi(X, self.landmarks_)
        try:
            self.W_cho_ = cholesky(W, lower=True)
            self.nystrom_alpha_ = cho_solve((self.W_cho_, True), C.T @ y)
        except np.linalg.LinAlgError:
            self.W_cho_ = None
            self._fit_standard(X, y)

    def predict(self, X_star):
        if self.approximation.lower() == "nystroem" and self.landmarks_ is not None:
            return self._predict_nystrom(X_star)
        return self._predict_standard(X_star)

    def _predict_standard(self, X_star):
        psi = self.build_Psi(X_star, self.X_)
        y_pred = psi @ self.alpha_
        if self.L_ is not None:
            v = solve_triangular(self.L_, psi.T, lower=True)
            y_mse = 1.0 - np.sum(v**2, axis=0)
        else:
            Psi = self.build_Psi(self.X_, self.X_) + 1e-8 * np.eye(self.X_.shape[0])
            pi_Psi = np.linalg.pinv(Psi)
            y_mse = 1.0 - np.sum((psi @ pi_Psi) * psi, axis=1)
        y_mse[y_mse < 0] = 0
        return y_pred, y_mse.reshape(-1, 1)

    def _predict_nystrom(self, X_star):
        psi_star_m = self.build_Psi(X_star, self.landmarks_)
        y_pred = psi_star_m @ self.nystrom_alpha_
        if self.W_cho_ is not None:
            v = cho_solve((self.W_cho_, True), psi_star_m.T)
            quad = np.sum(psi_star_m * v.T, axis=1)
            y_mse = 1.0 - quad
        else:
            y_mse = np.ones(X_star.shape[0])
        y_mse[y_mse < 0] = 0
        return y_pred, y_mse.reshape(-1, 1)

    def build_Psi(self, X1, X2):
        n1 = X1.shape[0]
        Psi = np.zeros((n1, X2.shape[0]))
        for i in range(n1):
            Psi[i, :] = self.build_psi_vec(X1[i, :], X2)
        return Psi

    def build_psi_vec(self, x, X_):
        theta10 = np.full(self.dim, 10**self.theta) if self.isotropic else 10**self.theta
        D = np.zeros(X_.shape[0])
        if self.ordered_mask.any():
            Xo = X_[:, self.ordered_mask]
            xo = x[self.ordered_mask]
            D += cdist(xo.reshape(1, -1), Xo, metric="sqeuclidean",
                       w=theta10[self.ordered_mask]).ravel()
        if self.factor_mask.any():
            Xf = X_[:, self.factor_mask]
            xf = x[self.factor_mask]
            D += cdist(xf.reshape(1, -1), Xf, metric="hamming",
                       w=theta10[self.factor_mask]).ravel() * self.factor_mask.sum()
        return np.exp(-D) if self.corr == "squared_exponential" else np.exp(-(D**self.p))
````
## Implementation Details

### Architectural Enhancements to `init`

* New argument `approximation="None"` for backward-compatible selection between exact Kriging and Nyström
* New argument `n_landmarks` (default 100) controls the number of inducing points when using Nyström
* State attributes for both exact and Nyström paths are maintained separately

### The `fit()` Method: A Dual-Pathway Approach

* Dispatcher selecting exact or Nyström pathway
* The Nyström fit Pathway (`_fit_nystrom`):
    * Landmark selection via uniform sampling without replacement
    * Core matrices:
        * $W = K_{mm}$ (landmark-landmark)
        * $C = K_{nm}$ (data-landmark)
    * Cholesky factorization of $W$ (with jitter) for stability
    * Pre-computation: $\alpha_{nys} = W^{-1} C^T y$ via `cho_solve`
* The Standard fit Pathway (`_fit_standard`):
    * Full $\Psi$ construction, Cholesky decomposition, and solve for $\alpha$
    * Fallback to pseudoinverse if Cholesky fails

### The `predict()` Method: Conditional Prediction Logic

* Routes to Nyström or standard prediction path based on fitted model state
* The Nyström predict Pathway (`_predict_nystrom`):
    * Cross-covariance $\psi$ between test points and landmarks
    * Mean: $\psi \cdot \alpha_{nys}$
    * Variance: uses `cho_solve` with $W$ Cholesky; non-negative clipping
* The Standard predict Pathway (`_predict_standard`):
    * Cross-covariance with all training points
    * Mean from $\alpha$; variance via triangular solves or pseudoinverse fallback

### Critical Detail: Preserving Mixed Variable Type Functionality

The Significance of `build_psi_vec`:

* Mixed spaces: continuous (ordered) and categorical (factor) variables
* Distances:
    * Weighted squared Euclidean for ordered variables
    * Weighted Hamming for factors
* Anisotropic kernel via per-dimension length-scales $\theta$
* Nyström path reuses `build_Psi` → `build_psi_vec`, preserving mixed-type handling

### Seamless Integration into the Nyström Workflow

All covariance computations ($W$, $C$, predictive cross-covariance) use `build_Psi`, ensuring identical handling for mixed variable types in both standard and Nyström modes.

## Jupyter Notebook

:::{.callout-note}

* The Jupyter-Notebook of this lecture is available on GitHub in the [Hyperparameter-Tuning-Cookbook Repository](https://github.com/sequential-parameter-optimization/Hyperparameter-Tuning-Cookbook/blob/main/006_matrices.ipynb)

:::


