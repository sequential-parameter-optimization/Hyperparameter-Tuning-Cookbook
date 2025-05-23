---
execute:
  cache: false
  eval: true
  echo: true
  warning: false
jupyter: python3
---

# Classification


In classification we have a qualitative response variable.

![Classification. Taken from @Jame14a ](./figures_static/0218a.png){width=100% #fig-0218a}

Here the response variable $Y$ is qualitative, e.g., email is one of $\cal{C} = (spam, ham)$, where ham is good email,
digit class is one of $\cal{C} = \{ 0, 1, \ldots, 9 \}$.
Our goals are to:

* Build a classifier $C(X)$ that assigns a class label from $\cal{C}$ to a future unlabeled observation $X$.
* Assess the uncertainty in each classification
* Understand the roles of the different predictors among $X = (X_1,X_2, \ldots, X_p)$.

Simulation example depicted in@fig-0218a.
$Y$ takes two values, zero and one, and $X$ has only one value.
Big sample: each single vertical bar indicates an occurrance of a zero (orange) or one (blue) as a function of the $X$s.
Black curve generated the data: it is the probability of generating a one. For high values of $X$, the probability of ones is increasing.
What is an ideal classifier $C(X)$?

Suppose the $K$ elements in $\cal{C}$ are numbered $1,2,\ldots, K$. Let
$$
p_k(x) = Pr(Y = k|X = x), k = 1,2,\ldots,K.
$$

These are the **conditional class probabilities** at $x$; e.g. see little barplot at $x = 5$.
Then the  **Bayes optimal classifier** at $x$ is 
$$
C(x) = j \qquad \text{ if }  p_j(x) = \max \{p_1(x),p_2(x),\ldots, p_K(x)\}.
$$
At $x=5$ there is an 80\% probability of one, and an 20\% probability of a zero.
So, we classify this point to the class with the highest probability, the majority
class.

Nearest-neighbor averaging can be used as before. This is illustrated in Fig.~\ref{fig:0219a}.
Here, we consider 100 points only.
Nearest-neighbor averaging also breaks down as dimension grows. 
However, the impact on $\hat{C}(x)$ is less than on $\hat{p}_k (x)$, 
$k = 1, \ldots, K$.

![Classification. Taken from @Jame14a ](./figures_static/0219a.png){width=100% #fig-0219a}


## Classification: Some Details

Average number of errors made to measure the performance. Typically we measure the performance of $\hat{C}(x)$ using the **misclassification error rate**:
$$
Err_{Te} = Ave_{i\in Te} I[y_i \neq \hat{C} (x_i) ].
$$
The Bayes classifier (using the true $p_k(x)$) has smallest error (in the population).


## k-Nearest Neighbor Classification

Consider k-nearest neighbors in two dimensions. Orange and blue dots label the true class memberships of the underlying  points in the 2-dim plane.
Dotted line is the decision boundary, that is the contour with equal probability for both classes.

Nearest-neighbor averaging in 2-dim. At any given point we want to classify, we spread out a little neighborhood,
say $K=10$ points from the neighborhood and calulated the percentage of blue and orange. We assign the color with the highest probability to this point.
If this is done for every point in the plane, we obtain the solid black curve as the esitmated decsion boundary.

We can use $K=1$. This is the **nearest-neighbor classifier**.
The decision boundary is piecewise linear. Islands occur. Approximation is rather noisy.

$K=100$ leads to a smooth decision boundary. But gets uninteresting.

![K-nearest neighbors in two dimensions. Taken from @Jame14a ](./figures_static/0213.png){width=70% #fig-0213}

![K-nearest neighbors in two dimensions. Taken from @Jame14a ](./figures_static/0215.png){width=70% #fig-0215}

![K-nearest neighbors in two dimensions. Taken from @Jame14a ](./figures_static/0216.png){width=70% #fig-0216}

$K$ large means higher bias, so $1/K$ is chosen, because we go from low to high complexity on the $x$-error, see @fig-0217.
Horizontal dotted line is the base error.

![K-nearest neighbors classification error. Taken from @Jame14a ](./figures_static/0217.png){width=70% #fig-0217}


::: {def-minkowski-distance}

### Minkowski Distance

The Minkowski distance of order $p$ (where $p$ is an integer) between two points
$X=(x_1,x_2,\ldots,x_n)\text{ and }Y=(y_1,y_2,\ldots,y_n) \in \mathbb{R}^n$
is defined as:
$$
D \left( X,Y \right) = \left( \sum_{i=1}^n |x_i-y_i|^p \right)^\frac{1}{p}.
$$
:::


* Video: [StatQuest: K-nearest neighbors, Clearly Explained](https://youtu.be/HVXime0nQeI?si=wTGGkn_6vIshrTk0)

## Decision and Classification Trees

### Decision Trees

#### [Decision and Classification Trees, Clearly Explained](https://youtu.be/_L39rN6gz7Y?si=KtY-CsLGeAbIJN-f)

#### [StatQuest: Decision Trees, Part 2 - Feature Selection and Missing Data](https://youtu.be/wpNl-JwwplA?si=R7qiQ4rVzsrW1GAI)


### Regression Trees

#### [Regression Trees, Clearly Explained!!!](https://youtu.be/g9c66TUylZ4?si=aXOCqkDl-fGAFRx2)

#### [How to Prune Regression Trees, Clearly Explained!!!](https://youtu.be/D0efHEJsfHo?si=OKizIPtcrWDOSCRF)



## The Confusion Matrix

* Video: [Machine Learning Fundamentals: The Confusion Matrix](https://youtu.be/Kdsp6soqA7o?si=pOEUeyk1Crt9heg1)



### Sensitivity and Specificity

* Video: [Machine Learning Fundamentals: Sensitivity and Specificity](https://youtu.be/vP06aMoz4v8?si=9O6FfcKtOWSdx84t)



## Naive Bayes

* Video: [Naive Bayes, Clearly Explained!!!](https://youtu.be/O2L2Uv9pdDA?si=CTRhu0XXwTZuxxwS)



## Gaussian Naive Bayes

* Video: [Gaussian Naive Bayes, Clearly Explained!!!](https://youtu.be/H3EjCKtlVog?si=cXWTWaQ1cw5wbFXr)



