---
execute:
  cache: false
  eval: true
  echo: true
  warning: false
jupyter: python3
---

# Addressing Multicollinearity: Principle Component Analysis (PCA) and Factor Analysis (FA)

## Introduction

The concepts of Principal Component Analysis (PCA) and Factor Analysis (FA) are both  dimensionality reduction techniques. They operate on different assumptions and serve distinct purposes.
PCA aims to transform correlated variables into a smaller set of uncorrelated principal components that capture maximum variance, whereas Factor Analysis seeks to explain the correlations between observed variables in terms of a smaller number of unobserved, underlying factors. 

After loading and preprocessing the data in @sec-data-preprocessing, we will explore these methods to reduce dimensions and address multicollinearity. In @sec-fit-ols we will conduct linear regression on the extracted components or factors.
@sec-collinearity-diagnostics provides diagnostics for multicollinearity, including the coefficient table, eigenvalues, condition indices, and the KMO measure. 
@sec-pca explains how PCA is applied to the data, while @sec-fa discusses Factor Analysis. Both methods are used to mitigate multicollinearity issues in regression models. @sec-other-models shows how the reduced dimensions can be used in other machine learning models, such as Random Forests.

The following packages are used in this chapter:
```{python}
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_kmo
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import copy
```

## The Car-Sales Data Set {#sec-data-preprocessing}

First, the data is preprocessed to ensure that it does not contain any NaN or infinite values.
We load the data set, which contains information about car sales, including various features such as price, engine size, horsepower, and more. The initial shape of the DataFrame is `(157, 27)`.

```{python}
#| label: load_car_sales_data
df = pd.read_csv("data/car_sales.csv", encoding="utf-8")
print(df.shape)
df.head()
```

The first column is removed as it's an index or non-informative column.
```{python}
#| label: drop_first_column
df = df.drop(df.columns[0], axis=1)
print(df.columns)
```
### The Target Variable

The `sales` variable, which is our target, is transformed to a log scale. Missing or zero values are handled by replacing them with the median.

```{python}
#| label: transform_sales_variable
df['ln_sales'] = np.log(df['sales'].replace(0, np.nan))
if df['ln_sales'].isnull().any() or np.isinf(df['ln_sales']).any():
    df['ln_sales'] = df['ln_sales'].fillna(df['ln_sales'].median()) # Or any other strategy
y = df['ln_sales']
```

### The Features
#### Numerical Features

The following steps are performed during data preprocessing for numerical features:

1.  Check for NaN or infinite values in X.
2.  Replace NaN and infinite values with the median of the respective column.
3.  Remove constant or nearly constant columns (not explicitly shown in code but stated in preprocessing steps).
4.  Standardize the numerical predictors in X using StandardScaler.
5.  Verify that X_scaled does not contain any NaN or infinite values.

```{python}
# Use columns from 'price' to 'mpg' as predictors
independent_var_columns = ['price', 'engine_s', 'horsepow', 'wheelbas',
                           'width', 'length', 'curb_wgt', 'fuel_cap', 'mpg']
# Select those columns, ensuring they are numeric
X = df[independent_var_columns].apply(pd.to_numeric, errors='coerce')
# Handle missing/nans in features by using an appropriate imputation strategy
X = X.fillna(X.median()) # Impute with median or any other appropriate strategy
# Display the first few rows of the features
X.head()
```

```{python}
if X.isnull().any().any():
    print("NaNs detected in X. Filling with column medians.")
    X = X.fillna(X.median())

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
if np.isnan(X_scaled).any() or np.isinf(X_scaled).any():
    raise ValueError("X_scaled contains NaN or infinite values after preprocessing.")
# Convert the scaled data back to a DataFrame
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
# Display the first few rows of the scaled features
X_scaled.head()
```

#### Categorical Features

Categorical features (like 'type') are one-hot encoded and then combined with the scaled numerical features.

```{python}
categorical_cols = ['type'] # Replace if more categorical variables exist
encoder = OneHotEncoder(drop='first', sparse_output=False)
X_categorical_encoded = encoder.fit_transform(df[categorical_cols])
# Convert encoded data into a DataFrame
X_categorical_encoded_df = pd.DataFrame(X_categorical_encoded,
                                        columns=encoder.get_feature_names_out(categorical_cols))
X_categorical_encoded_df.describe(include='all')
```

### Combine non-categorical and categorical (encoded) data
```{python}
X_encoded = pd.concat([X_scaled, X_categorical_encoded_df], axis=1)
X_encoded.describe(include='all')
```

## Fit the Linear Regression Model using statsmodels {#sec-fit-ols}

An Ordinary Least Squares (OLS) regression model is fitted using the preprocessed and combined features (`X_encoded`).

```{python}
X_encoded_with_const = sm.add_constant(X_encoded) # Adds a constant term (intercept) to the model
model = sm.OLS(df['ln_sales'], X_encoded_with_const).fit()
```

### Model Summary and Interpretation

#### Model Summary (ANOVA Table)
The ANOVA table shows a significant F-value (Prob (F-statistic) close to zero), indicating that the model is statistically significant and better than simply estimating the mean. The Adj. R-squared value, close to 0.40, suggests that nearly 40% of the variation in `ln_sales` is explained by the model.

```{python}
print(model.summary())
```

Despite the positive model fit, many predictors show non-significant coefficients (P>|t| much larger than 0.05), suggesting they contribute little to the model.

## Collinearity Diagnostics {#sec-collinearity-diagnostics}

### The Coefficient Table
The coefficient table provides further evidence of multicollinearity. The function `compute_coefficients_table()`  from the `spotpython` package is used here for comprehensive diagnostics.

```{python}
from spotpython.utils.stats import compute_coefficients_table
coeffs_table = compute_coefficients_table(
    model=model, X_encoded=X_encoded_with_const, y=y, vif_table=None
)
print("\nCoefficients Table:")
print(coeffs_table)
```

For most predictors, the partial correlations (Partial r) decrease significantly compared to the zero-order correlations (Zero-Order r), which suggests multicollinearity. Tolerance values (1 minus the proportion of variance explained by other predictors) are low, indicating that approximately 70%-90% of a given predictor's variance can be explained by other predictors. Tolerances close to 0 signify high multicollinearity. A Variance Inflation Factor (VIF) greater than 2 is typically considered problematic, and in this table, the smallest VIF is already greater than 2, confirming serious multicollinearity.

### Eigenvalues and Condition Indices

Eigenvalues indicate how many factors or components can be meaningfully extracted. An eigenvalue greater than 1 suggests that the factor/component explains more variance than a single variable.

#### Eigenvalues

We use the `FactorAnalyzer` function from the `factor_analyzer` package to compute eigenvalues.

```{python}
fa_temp = FactorAnalyzer(n_factors=X_encoded.shape[1], method="principal", rotation=None)
try:
    fa_temp.fit(X_encoded)
    ev, _ = fa_temp.get_eigenvalues()
    ev = np.sort(ev) # The source prints in ascending order
    print("Eigenvalues for each component:\n", ev)
except Exception as e:
    print(f"Error during factor analysis fitting: {e}")
    print("Consider reducing multicollinearity or removing problematic features.")
```

The eigenvalue-based diagnostics confirm severe multicollinearity.
Several eigenvalues are close to 0, indicating strong correlations among predictors.

#### Condition Indices

From `spotpython.utils.stats`, we can compute the condition index, which is a measure of multicollinearity. A condition index greater than 15 suggests potential multicollinearity issues, and values above 30 indicate severe problems.

Condition indices, calculated as the square roots of the ratios of the largest eigenvalue to each subsequent eigenvalue, also highlight the issue.

::: {#def-condition_index}
###### Condition Index

The Condition Index ($CI_i$) for the $i$-th eigenvalue is defined as:
$$
\text{CI}_i = \sqrt{\frac{\lambda{\max}}{\lambda_i}},
$$
where $\lambda_{\max}$ is the largest eigenvalue of the scaled predictor correlation matrix, and $\lambda_i$ is the $i$-th eigenvalue of the same matrix.
::: 

$CI_i$-values greater than 15 suggest a potential problem, and values over 30 indicate a severe problem.

```{python}
from spotpython.utils.stats import condition_index
X_cond = copy.deepcopy(X_encoded)
condition_index_df = condition_index(X_cond)
print("\nCondition Index:")
print(condition_index_df)
```

###  Kayser-Meyer-Olkin (KMO) Measure

The KMO (Kaiser-Meyer-Olkin) measure is a metric for assessing the suitability of data for Factor Analysis. A KMO value of 0.6 or higher is generally considered acceptable, while a value below 0.5 indicates that the data is not suitable for Factor Analysis.

The KMO measure is based on the correlation and partial correlation between variables. It is calculated as the ratio of the squared sums of correlations to the squared sums of correlations plus the squared sums of partial correlations. KMO values range between 0 and 1, where values close to 1 suggest strong correlations and suitability for Factor Analysis, and values close to 0 indicate weak correlations and unsuitability.

```{python}
kmo_all, kmo_model = calculate_kmo(X_encoded)
print(f"\nKMO measure: {kmo_model:.3f} (0.6+ is often considered acceptable)")
```

A KMO measure of 0.835 indicates that the data is well-suited for Factor Analysis.

## Addressing Multicollinearity with Principal Component Analysis (PCA) {#sec-pca}

::: {#def-multicoll}
###### Multicollinearity and Multicorrelation

Multicorrelation is a general term that describes correlation between multiple variables. Multicollinearity is a specific problem in regression models caused by strong correlations between independent variables, making model interpretation difficult.
:::


### Introduction to PCA

Principal Component Analysis (PCA) is a popular unsupervised dimensionality reduction technique. It transforms a set of possibly correlated variables into a set of linearly uncorrelated variables called principal components. The first principal component accounts for as much of the variability in the data as possible, and each succeeding component accounts for as much of the remaining variability as possible. PCA is primarily used for data compression and simplifying complex datasets.

### Application of PCA in Regression Problems:

*   **Dimensionality Reduction:** PCA reduces the number of explanatory variables by transforming original variables into a smaller set of uncorrelated principal components, making regression algorithms less prone to overfitting, especially with many features.
*   **Reducing Multicollinearity:** PCA effectively eliminates multicollinearity in linear regression models because the resulting principal components are orthogonal (uncorrelated) to each other, leading to more stable coefficient estimates.
*   **Handling High-Dimensional Data:** It can reduce the dimensions of datasets with many variables to a manageable level before regression.
*   **Reduced Overfitting Tendencies:** By removing redundant and highly correlated variables, PCA helps reduce the risk of overfitting by focusing the model on the most influential features.
*   **Improved Model Performance:** Performing regression on the most important principal components often leads to better generalization and improved model performance on new data.
*   **Interpretation of Feature Importance:** PCA provides insights into the importance of original features through the variance explained by each principal component, which can identify combinations of variables best representing the data.

### Scree Plot

::: {#def-scree-plot}
#### Scree Plot
A scree plot is a graphical representation of the eigenvalues of a covariance or correlation matrix in descending order. It is used to determine the number of significant components or factors in dimensionality reduction techniques.

Mathematically, the eigenvalues $\lambda_1, \lambda_2, \dots, \lambda_p$ are plotted against their corresponding component or factor indices $i = 1, 2, \dots, p$, where $p$ is the total number of components or factors.

The eigenvalues are defined as:

$$
\lambda_i = \text{Var}(\mathbf{z}_i),
$$

where $\mathbf{z}_i$ is the $i$-th principal component or factor, and $\text{Var}(\mathbf{z}_i)$ is its variance.

The scree plot is constructed by plotting the points $(i, \lambda_i)$ for $i = 1, 2, \dots, p$. The "elbow" in the plot, where the eigenvalues start to level off, indicates the optimal number of components or factors to retain.
::: 



### Loading Scores (for PCA)

Loading scores in the context of Principal Component Analysis (PCA) represent the correlation or relationship between the original variables and the principal components. 

::: {#def-loading_scores}
#### Loading Scores
The loading score for the $j$-th variable on the $i$-th principal component is defined as:

$$
L_{ij} = \mathbf{a}_i^\top \mathbf{x}_j,
$$

where:

$\mathbf{a}_i$ is the eigenvector corresponding to the $i$-th principal component,
$\mathbf{x}_j$ is the standardized value of the $j$-th variable.
:::

In PCA, the loading scores indicate how much each original variable contributes to a given principal component. High absolute values of $L_{ij}$ suggest that the $j$-th variable strongly influences the $i$-th principal component.
In PCA, loading scores can be viewed as directional vectors in the feature space. The magnitude of the score indicates how dominant the variable is in a component, while the sign represents the direction of the relationship. A high positive loading means a positive influence and correlation with the component, and a high negative loading indicates a negative correlation. Loading score values also show how much each original variable contributes to the explained variance in its respective principal component.

::: {.callout-note}
### Summary of Loading Scores
Loading scores are used in Principal Component Analysis (PCA).

* Definition: Loading scores represent the correlation or relationship between the original variables and the principal components.
* Purpose: They indicate how much each original variable contributes to a given principal component.
* Mathematical Representation: In PCA, the loading scores are the elements of the eigenvectors of the covariance (or correlation) matrix, scaled by the square root of the corresponding eigenvalues.
* Interpretation: High absolute values of loading scores suggest that the variable strongly influences the corresponding principal component.
::: 

@sec-loading-scores-vs-factor-loadings explains the difference between loading scores in PCA and factor loadings in FA. 



### PCA for Car Sales Example

#### Computing the Principal Components
The Principal Component Analysis (PCA) is applied only to the features (`X_encoded`), not to the target variable. We will use `sklearn.decomposition.PCA` to perform PCA.

```{python}
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def get_pca(df):
    """
    Scale the numeric data and perform PCA.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        tuple: A tuple containing the PCA object, scaled data, feature names, sample names, and PCA-transformed data.
    """
    # Select only numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    
    # Scale the numeric data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numeric_df)
    feature_names = numeric_df.columns
    sample_names = df.index

    # Perform PCA
    pca = PCA()
    pca.fit(scaled_data)
    pca_data = pca.transform(scaled_data)

    return pca, scaled_data, feature_names, sample_names, pca_data
```

```{python}
def plot_pca_scree(pca, df_name="", max_scree=None, figsize=(12, 6)):
    """
    Plot the scree plot for PCA.

    Args:
        pca (PCA): Fitted PCA object.
        df_name (str): Name of the dataset.
        max_scree (int): Maximum number of principal components to plot.
        figsize (tuple): Size of the figure.
    """
    per_var = np.round(pca.explained_variance_ratio_ * 100, decimals=1)
    full_labels = ["PC" + str(x) for x in range(1, len(per_var) + 1)]

    # Limit the number of PCs in the scree plot
    if max_scree is not None:
        per_var = per_var[:max_scree]
        scree_labels = full_labels[:max_scree]
    else:
        scree_labels = full_labels

    plt.figure(figsize=figsize)
    plt.plot(range(1, len(per_var) + 1), per_var, marker='o', linestyle='--')
    plt.xticks(range(1, len(per_var) + 1), scree_labels)
    plt.ylabel("Percentage of Explained Variance")
    plt.xlabel("Principal Component")
    plt.title(f"Scree Plot. {df_name}")
    plt.grid(True)
    plt.show()
```

```{python}
def plot_pca1vs2(pca, pca_data, df_name="", figsize=(12, 6)):
    """
    Plot the first two principal components.

    Args:
        pca (PCA): Fitted PCA object.
        pca_data (array-like): PCA-transformed data.
        df_name (str): Name of the dataset.
        figsize (tuple): Size of the figure.
    """
    pca_df = pd.DataFrame(pca_data, columns=["PC" + str(i + 1) for i in range(pca_data.shape[1])])

    plt.figure(figsize=figsize)
    plt.scatter(pca_df["PC1"], pca_df["PC2"])
    for sample in pca_df.index:
        plt.annotate(sample, (pca_df.PC1.loc[sample], pca_df.PC2.loc[sample]))
    plt.title(f"PCA Graph. {df_name}")
    plt.xlabel(f"PC1 - {np.round(pca.explained_variance_ratio_[0] * 100, 1)}%")
    plt.ylabel(f"PC2 - {np.round(pca.explained_variance_ratio_[1] * 100, 1)}%")
    # add grid
    plt.grid(True)
    plt.show()

def get_pca_topk(pca, feature_names, k=10):
    """
    Get the top k features influencing PC1 and PC2.

    Args:
        pca (PCA): Fitted PCA object.
        feature_names (list): List of feature names.
        k (int): Number of top features to select.

    Returns:
        tuple: Two lists containing the names of the top k features most influential on PC1 and PC2.
    """
    loading_scores_pc1 = pd.Series(pca.components_[0], index=feature_names)
    loading_scores_pc2 = pd.Series(pca.components_[1], index=feature_names)

    sorted_loading_scores_pc1 = loading_scores_pc1.abs().sort_values(ascending=False)
    sorted_loading_scores_pc2 = loading_scores_pc2.abs().sort_values(ascending=False)

    top_k_features_pc1 = sorted_loading_scores_pc1.head(k).index.tolist()
    top_k_features_pc2 = sorted_loading_scores_pc2.head(k).index.tolist()

    return top_k_features_pc1, top_k_features_pc2
```

```{python}
# Step 1: Perform PCA and scale the data
pca, scaled_data, feature_names, sample_names, pca_data = get_pca(df=X_encoded)

# Step 2: Plot the scree plot
plot_pca_scree(pca, df_name="Car Sales Data", max_scree=10)

# Step 3: Plot the first two principal components
plot_pca1vs2(pca, pca_data, df_name="Car Sales Data")

# Step 4: Get the top k features influencing PC1 and PC2
top_k_features_pc1, top_k_features_pc2 = get_pca_topk(pca, feature_names, k=10)
print("Top 10 features influencing PC1:", top_k_features_pc1)
print("Top 10 features influencing PC2:", top_k_features_pc2)
```

#### Loading Scores for PCA (10 Components)

```{python}
#| label: fig-pca_loading_scores-10
#| fig.cap: "PCA Loading Scores Heatmap showing the influence of original features on the principal components."
#| fig.width: 10
#| fig.height: 8
#| fig.align: center
#| fig.show: true
loading_scores = pd.DataFrame(pca.components_.T, columns=[f"PC{i+1}" for i in range(pca.n_components_)], index=X_encoded.columns)
print("Loading Scores for PCA:\n", loading_scores)
# Plot the loading scores heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(
    loading_scores, annot=True, fmt=".2f", cmap="coolwarm",
    cbar_kws={"label": "Loading Score"}, linewidths=0.5
)
plt.title("PCA Loading Scores Heatmap")
plt.xlabel("Principal Components")
plt.ylabel("Original Features")
plt.tight_layout()
plt.show()
```



#### Using Three Principal Components only

Next, we apply PCA to the encoded features. 
For this example, we select three principal components, similar to the Factor Analysis example below.

```{python}
n_components_pca = 3
pca = PCA(n_components=n_components_pca)
X_pca_scores = pca.fit_transform(X_encoded)

# Convert principal components to a DataFrame
pca_columns = [f"PC{i+1}" for i in range(X_pca_scores.shape[1])]
df_pca_components = pd.DataFrame(data=X_pca_scores, columns=pca_columns)

print(f"Shape of PCA components DataFrame: {df_pca_components.shape}")
print("First 5 rows of PCA components:\n", df_pca_components.head())

# Loading scores for interpretation
pca_loading_scores = pd.DataFrame(pca.components_.T, columns=pca_columns, index=X_encoded.columns)
print("\nPCA Loading Scores:\n", pca_loading_scores)

# Explained variance
explained_variance_ratio = pca.explained_variance_ratio_
print("\nExplained variance ratio per principal component:\n", explained_variance_ratio)
print("Cumulative explained variance:", np.cumsum(explained_variance_ratio))
```

#### Plotting the Scree Plot

```{python}
#| label: fig-scree_plot_pca
#| fig.cap: "Scree plot for PCA showing the explained variance ratio for each principal component."
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, marker='o', linestyle='--')
plt.title('Scree Plot for PCA')
plt.xlabel('Number of Principal Components')
plt.ylabel('Explained Variance Ratio')
plt.grid(True)
plt.show()
```

The scree plot and cumulative explained variance help in determining the optimal number of components. Here, 10 components explain 100% of the variance, as we've chosen to extract all possible components.

#### Plotting the Loading Scores (3 Components)

```{python}
#| label: fig-pca_loading_scores-3
#| fig.cap: "PCA Loading Scores Heatmap showing the influence of original features on the principal components."
#| fig.width: 10
#| fig.height: 8
#| fig.align: center
#| fig.show: true
plt.figure(figsize=(10, 8))
sns.heatmap(
    pca_loading_scores, annot=True, fmt=".2f", cmap="coolwarm",
    cbar_kws={"label": "Loading Score"}, linewidths=0.5
)
plt.title("PCA Loading Scores Heatmap")
plt.xlabel("Principal Components")
plt.ylabel("Original Features")
plt.tight_layout()
plt.show()
```


### Creating the Regression Model with Principal Components
Now, a linear regression model is fitted using the principal components derived from PCA. These components are uncorrelated, which should eliminate multicollinearity issues.

```{python}
X_pca_model_with_const = sm.add_constant(df_pca_components)
model_pca = sm.OLS(y, X_pca_model_with_const).fit()
print("\nRegression on PCA Components:")
print(model_pca.summary())

# Verify collinearity statistics for PCA components (VIF and Tolerance)
# A custom function would be needed here for VIF of PC components,
# but since PCs are orthogonal, VIFs will be 1.0.
# The condition number from the OLS summary (Cond. No.) for model_pca
# should also be close to 1, indicating no multicollinearity.
```


As expected, the condition number for `model_pca` is 1.00, indicating no multicollinearity among the principal components. This confirms that PCA successfully addresses the multicollinearity problem. The R-squared and Adjusted R-squared values remain the same as the original OLS model since PCA preserves the total variance when all components are retained.

### Comparison of Model Performance (PCA vs. OLS)
```{python}
# Predictions from the Linear Regression Model (model)
predictions_linear = model.predict(X_encoded_with_const)
# Predictions from the PCA Regression Model (model_pca)
predictions_pca = model_pca.predict(X_pca_model_with_const)

# Calculate R-squared and Adjusted R-squared for both models
r2_linear = model.rsquared
adj_r2_linear = model.rsquared_adj

r2_pca = model_pca.rsquared
adj_r2_pca = model_pca.rsquared_adj

# Calculate Mean Squared Error (MSE) and Root Mean Squared Error (RMSE) for both models
mse_linear = mean_squared_error(y, predictions_linear)
rmse_linear = np.sqrt(mse_linear)

mse_pca = mean_squared_error(y, predictions_pca)
rmse_pca = np.sqrt(mse_pca)

# Print the comparison
print("Comparison of OLS and PCA Regression Models")
print("\nLinear Regression Model (Original Features):")
print(f"R-squared: {r2_linear:.4f}")
print(f"Adjusted R-squared: {adj_r2_linear:.4f}")
print(f"MSE: {mse_linear:.4f}")
print(f"RMSE: {rmse_linear:.4f}")

print("\nPCA Regression Model (Principal Components):")
print(f"R-squared: {r2_pca:.4f}")
print(f"Adjusted R-squared: {adj_r2_pca:.4f}")
print(f"MSE: {mse_pca:.4f}")
print(f"RMSE: {rmse_pca:.4f}")
```

When all principal components are retained, the PCA regression model performs identically to the original OLS model in terms of R-squared, Adjusted R-squared, MSE, and RMSE. This is because PCA merely rotates the data, preserving all variance if all components are used. Its benefit lies in handling multicollinearity and enabling dimensionality reduction if fewer components are chosen without significant loss of information.

## Addressing Multicollinearity and Latent Structure with Factor Analysis (FA) {#sec-fa}

### Introduction to Factor Analysis
Factor Analysis (FA) is a statistical method used to describe variability among observed, correlated variables in terms of a potentially lower number of unobserved variables called factors or latent variables. Unlike PCA, which is primarily a data reduction technique focused on maximizing variance explained, FA assumes that the observed variables are linear combinations of these underlying factors plus an error term. FA's main goal is to uncover the underlying structure that explains the correlations among observed variables.

### Determining the Number of Factors for Factor Analysis
For Factor Analysis, the number of factors to extract is a crucial decision. A common approach, consistent with the KMO measure, is to consider factors with eigenvalues greater than 1 (Kaiser's criterion). Factor analysis is then performed, often with a rotation method like Varimax to improve factor interpretability.

```{python}
anz_fak = 10 # Number of factors to extract, similar to the components in PCA
n_factors = min(anz_fak, X_encoded.shape[1])
fa = FactorAnalyzer(n_factors=n_factors, method="principal", rotation="varimax")
fa.fit(X_encoded) # Fit the Factor Analyzer
actual_factors = fa.loadings_.shape[1] # Number of factors actually extracted
print(f"actual_factors: {actual_factors}")
if actual_factors < n_factors:
    print(
        f"\nWarning: Only {actual_factors} factors could be extracted "
        f"(requested {n_factors})."
    )
factor_columns = [f"Factor{i+1}" for i in range(actual_factors)]
```


### Scree Plot for Factor Analysis


@fig-scree_plot_fa shows the eigenvalues for each factor extracted from Factor Analysis. The scree plot helps in determining the number of factors to retain by identifying the "elbow" point where the eigenvalues start to level off, indicating diminishing returns in explained variance.

```{python}
#| label: fig-scree_plot_fa
#| fig-cap: "Scree plot for Factor Analysis showing the eigenvalues for each factor."
plt.figure(figsize=(10, 6))
ev_fa, _ = fa.get_eigenvalues()
plt.plot(range(1, len(ev_fa) + 1), ev_fa, marker='o', linestyle='--')
plt.title('Scree Plot for Factor Analysis')
plt.xlabel('Number of Factors')
plt.ylabel('Eigenvalue')
plt.grid(True)
plt.xticks(range(1, len(ev_fa) + 1))
plt.show()
```



### Factor Loadings

Factor Loadings indicate how strongly each original variable is correlated with the extracted factors. High absolute values suggest that the variable has a significant influence on, or is strongly associated with, that factor. Loadings help in interpreting the meaning of each underlying factor.

::: {.callout-note}
#### Summary of Factor Loadings
Factor loadings are used in Factor Analysis (FA).
* Definition: Factor loadings represent the correlation or relationship between the observed variables and the latent factors.
* Purpose: They indicate how much each observed variable is explained by a given factor.
* Mathematical Representation: In FA, factor loadings are derived from the factor model, where observed variables are expressed as linear combinations of latent factors plus error terms.
* Interpretation: High absolute values of factor loadings suggest that the variable is strongly associated with the corresponding factor.
:::

@sec-loading-scores-vs-factor-loadings explains the difference between loading scores in PCA and factor loadings in FA. 

```{python}
# Print factor loadings with 2 decimals
factor_loadings = fa.loadings_
print("Factor Loadings (rounded to 2 decimals):\n", np.round(factor_loadings, 2))

# Create a DataFrame for the factor loadings for better visualization
factor_loadings_df = pd.DataFrame(
    factor_loadings, index=X_encoded.columns, # Original feature names
    columns=factor_columns # Factor names
)

# Plot the heatmap for factor loadings
plt.figure(figsize=(10, 8))
sns.heatmap(
    factor_loadings_df, annot=True, # Annotate with values
    fmt=".2f", # Format values to 2 decimals
    cmap="coolwarm", # Color map
    cbar=True # Show color bar
)
plt.title("Factor Loadings Heatmap")
plt.xlabel("Factors")
plt.ylabel("Features")
plt.tight_layout()
plt.show()
```

### Factor Scores
The factor scores are the transformed values of the original variables based on the extracted factors. These scores represent the values of the latent factors for each observation and can be used as new features in regression models, similar to principal components in PCA.

::: {#def-factor_scores}
#### Factor Scores
A **factor score** represents the value of a latent factor for a given observation, calculated as a linear combination of the observed variables weighted by the factor score coefficients.

Mathematically, the factor score for the $i$-th factor and the $j$-th observation is defined as:

$$
F_{ji} = w_{i1} x_{j1} + w_{i2} x_{j2} + \cdots + w_{ip} x_{jp} = \sum_{k=1}^p w_{ik} x_{jk},
$$

where

* $F_{ji}$ is the factor score for factor $i$ and observation $j$,  
* $w_{ik}$ is the factor score coefficient for variable $k$ on factor $i$,  
* $x_{jk}$ is the standardized value of variable $k$ for observation $j$, and 
* $p$ is the number of observed variables.
::: 


```{python}
# Factor scores for each row (shape: [n_samples, actual_factors])
X_factor_scores = fa.transform(X_encoded)
print(f"X_factor_scores shape: {X_factor_scores.shape}")

# Adapt the factor column names to the actual factor count
df_factors = pd.DataFrame(X_factor_scores, columns=factor_columns)
print(f"df_factors shape: {df_factors.shape}")
print(f"df_factors head:\n{df_factors.head()}")
```


### Creating the Regression Model with Extracted Factors (from FA)
A linear regression model is built using all ten extracted factors from Factor Analysis. The expectation is that these factors are uncorrelated, addressing multicollinearity.

```{python}
X_model_fa = sm.add_constant(df_factors)
model_factors = sm.OLS(y, X_model_fa).fit()
print("\nRegression on Factor Scores (all 10 factors):")
print(model_factors.summary())

# Verify collinearity statistics for Factor Analysis scores (VIF and Tolerance)
coeffs_table_fa = compute_coefficients_table(
    model=model_factors, X_encoded=X_model_fa, y=y, vif_table=None
)
print("\nCoefficients Table (Factor Analysis Model):")
print(coeffs_table_fa)

# Verify condition indices
X_cond_fa = copy.deepcopy(df_factors)
condition_index_df_fa = condition_index(X_cond_fa)
print("\nCondition Index (Factor Analysis Model):")
print(condition_index_df_fa)
```

As expected, the collinearity statistics (VIF and Tolerance) for the factor values show that they are uncorrelated (VIF=1, Tolerance=1). The condition indices are also all close to 1, confirming that Factor Analysis successfully mitigates multicollinearity. The coefficient estimates are larger relative to their standard errors compared to the original model, which can lead to more factors being identified as statistically significant.

### Comparison of Model Performance (FA vs. OLS)
```{python}
# Predictions from the Linear Regression Model (model)
predictions_linear = model.predict(X_encoded_with_const)
# Predictions from the Factor Analysis Regression Model (model_factors)
predictions_factors = model_factors.predict(X_model_fa)

# Calculate R-squared and Adjusted R-squared for both models
r2_linear = model.rsquared
adj_r2_linear = model.rsquared_adj

r2_factors = model_factors.rsquared
adj_r2_factors = model_factors.rsquared_adj

# Calculate Mean Squared Error (MSE) and Root Mean Squared Error (RMSE) for both models
mse_linear = mean_squared_error(y, predictions_linear)
rmse_linear = np.sqrt(mse_linear)

mse_factors = mean_squared_error(y, predictions_factors)
rmse_factors = np.sqrt(mse_factors)

# Print the comparison
print("Comparison of OLS and Factor Analysis Regression Models (all 10 factors)")
print("\nLinear Regression Model (Original Features):")
print(f"R-squared: {r2_linear:.4f}")
print(f"Adjusted R-squared: {adj_r2_linear:.4f}")
print(f"MSE: {mse_linear:.4f}")
print(f"RMSE: {rmse_linear:.4f}")

print("\nFactor Analysis Regression Model (Factor Scores):")
print(f"R-squared: {r2_factors:.4f}")
print(f"Adjusted R-squared: {adj_r2_factors:.4f}")
print(f"MSE: {mse_factors:.4f}")
print(f"RMSE: {rmse_factors:.4f}")
```


If the R-squared and Adjusted R-squared values for `model_factors` are close to those of the original `model`, it indicates that the regression model based on Factor Analysis performs similarly well, while successfully reducing multicollinearity. When all factors are used, the predictive performance metrics are identical to the original OLS model.

### Factor Analysis: Creating the Regression Model with three Extracted Factors only

To demonstrate the effect of dimensionality reduction, a regression model is created using only the first three extracted factors from Factor Analysis.

```{python}
# Create a regression model using only the first three factors
df_factors_reduced = df_factors.iloc[:, :3] # select the first three factors
X_model_fa_reduced = sm.add_constant(df_factors_reduced)
model_factors_reduced = sm.OLS(y, X_model_fa_reduced).fit()
print("\nRegression on Factor Scores (three factors only):")
print(model_factors_reduced.summary())

# Verify collinearity statistics for reduced FA scores
coeffs_table_fa_reduced = compute_coefficients_table(
    model=model_factors_reduced, X_encoded=X_model_fa_reduced, y=y, vif_table=None
)
print("\nCoefficients Table (Reduced Factor Analysis Model):")
print(coeffs_table_fa_reduced)

# Verify condition indices for reduced FA scores
X_cond_fa_reduced = copy.deepcopy(df_factors_reduced)
condition_index_df_fa_reduced = condition_index(X_cond_fa_reduced)
print("\nCondition Index (Reduced Factor Analysis Model):")
print(condition_index_df_fa_reduced)
```

The collinearity statistics for the reduced factor set continue to show that they are uncorrelated, with VIFs of 1.0 and condition indices close to 1.

### Comparison of Model Performance of the Reduced FA Model and the Full OLS Model
```{python}
# Predictions from the Linear Regression Model (model)
predictions_linear = model.predict(X_encoded_with_const)
# Predictions from the Reduced Factor Analysis Regression Model (model_factors_reduced)
predictions_factors_reduced = model_factors_reduced.predict(X_model_fa_reduced)

# Calculate R-squared and Adjusted R-squared for both models
r2_linear = model.rsquared
adj_r2_linear = model.rsquared_adj

r2_factors_reduced = model_factors_reduced.rsquared
adj_r2_factors_reduced = model_factors_reduced.rsquared_adj

# Calculate Mean Squared Error (MSE) and Root Mean Squared Error (RMSE) for both models
mse_linear = mean_squared_error(y, predictions_linear)
rmse_linear = np.sqrt(mse_linear)

mse_factors_reduced = mean_squared_error(y, predictions_factors_reduced)
rmse_factors_reduced = np.sqrt(mse_factors_reduced)

# Print the comparison
print("Comparison of OLS and Reduced Factor Analysis Regression Models")
print("\nLinear Regression Model (Original Features):")
print(f"R-squared: {r2_linear:.4f}")
print(f"Adjusted R-squared: {adj_r2_linear:.4f}")
print(f"MSE: {mse_linear:.4f}")
print(f"RMSE: {rmse_linear:.4f}")

print("\nReduced Factor Analysis Regression Model (3 Factor Scores):")
print(f"R-squared: {r2_factors_reduced:.4f}")
print(f"Adjusted R-squared: {adj_r2_factors_reduced:.4f}")
print(f"MSE: {mse_factors_reduced:.4f}")
print(f"RMSE: {rmse_factors_reduced:.4f}")
```


When reducing the number of factors from 10 to 3, the R-squared and Adjusted R-squared values for the Factor Analysis model decrease significantly (from ~0.48 to ~0.35). This indicates a trade-off: while reducing dimensionality successfully addresses multicollinearity, retaining too few factors can lead to information loss and reduced predictive accuracy. Lower MSE and RMSE values still suggest better predictive performance for the full OLS model in this specific comparison, as it retains more information.

## Summary: Comparing OLS, PCA, and Factor Analysis Models {#sec-summary-comparison}

### Interpretation of the Regression Models

*   **OLS Model (`model`):** This model uses the original variables directly. Coefficients indicate the direct relationship between each original variable and the target variable.
*   **PCA Regression Model (`model_pca`):** This model uses principal components, which are linear combinations of the original variables, as predictors. The coefficients show the relationship between the target variable and these abstract components.
*   **Factor Analysis Model (`model_factors`):** This model uses extracted factors, which are also linear combinations of original variables, designed to capture underlying latent structures. Coefficients indicate the relationship between the target variable and these latent factors.

### Differences Compared to the Standard OLS Model

| Feature           | OLS Model (Standard)                                        | PCA Regression Model                                                      | Factor Analysis Model                                                      |
| :---------------- | :---------------------------------------------------------- | :------------------------------------------------------------------------ | :------------------------------------------------------------------------- |
| **Input Variables** | Uses original variables (e.g., `X_encoded`) as predictors. | Uses principal components (e.g., `df_pca_components`) as predictors.       | Uses extracted factors (e.g., `df_factors`) as predictors.        |
| **Multicollinearity** | Can suffer from multicollinearity if predictors are highly correlated, leading to unstable coefficients and inflated standard errors. | Reduces multicollinearity because principal components are orthogonal (uncorrelated). | Reduces multicollinearity by using uncorrelated factors as predictors. |
| **Interpretability** | Coefficients correspond directly to original variables, making interpretation straightforward. | Coefficients relate to abstract principal components, making direct interpretation of original variable influence more challenging. | Coefficients relate to abstract factors, making interpretation more challenging. Factor loadings must be analyzed for meaning. |
| **Dimensionality** | Uses all original variables, potentially including redundant or irrelevant features. | Reduces the number of predictors by combining original variables into fewer principal components. | Reduces the number of predictors by combining original variables into fewer factors. |
| **Purpose**       | Direct relationship modeling, inference.                    | Dimensionality reduction, variance maximization, multicollinearity mitigation. | Discovering latent structures, explaining correlations.                     |
| **Assumptions**   | None on underlying structure beyond linear relationship.    | Does not assume an underlying causal model.                               | Assumes observed variables are caused by underlying factors.               |
| **Error Variance** | Does not explicitly separate unique variance.               | Does not separate unique variance from common variance.                   | Explicitly models unique variance for each variable.                       |

### Key Differences Between Loading Scores (PCA) and Factor Loadings (FA) {#sec-loading-scores-vs-factor-loadings}

| Aspect                | Loading Scores (PCA)                                   | Factor Loadings (FA)                                   |
|-----------------------|--------------------------------------------------------|-------------------------------------------------------|
| **Context**           | Principal Component Analysis (PCA)                    | Factor Analysis (FA)                                  |
| **Purpose**           | Describe the contribution of variables to principal components. | Describe the relationship between variables and latent factors. |
| **Underlying Model**  | No assumption of latent structure; purely variance-based. | Assumes a latent structure explaining observed variables. |
| **Error Term**        | PCA does not explicitly model error variance.          | FA explicitly models unique (error) variance for each variable. |
| **Interpretability**  | Components are orthogonal (uncorrelated).              | Factors may not be orthogonal, depending on rotation. |

While both loading scores and factor loadings describe relationships between variables and derived components or factors, **loading scores** are specific to PCA and focus on maximizing variance, while **factor loadings** are specific to FA and aim to uncover latent structures.




### Advantages of Using PCA and FA

**Principal Component Analysis (PCA):**

*   **Reduced Multicollinearity:** By using uncorrelated principal components, the model avoids instability caused by multicollinearity.
*   **Dimensionality Reduction:** The model uses fewer predictors if desired, improving computational efficiency and potentially generalization by removing noise.
*   **Variance Maximization:** Components are constructed to capture the maximum possible variance from the original data.

**Factor Analysis (FA):**

*   **Reduced Multicollinearity:** Similar to PCA, using uncorrelated factors prevents instability from multicollinearity.
*   **Dimensionality Reduction:** Reduces the number of predictors, improving computational efficiency and generalization.
*   **Focus on Underlying Structure:** Factor analysis aims to capture the latent structure of the data, potentially providing better insights into the fundamental relationships between variables.

### Disadvantages of Using PCA and FA

**Principal Component Analysis (PCA):**

*   **Loss of Interpretability:** Principal components are abstract combinations of the original variables, making it harder to directly interpret the coefficients. Understanding individual variable influence requires examining loading scores.
*   **Potential Information Loss:** If too few components are retained, information from the original variables may be lost, potentially reducing predictive accuracy.

**Factor Analysis (FA):**

*   **Loss of Interpretability:** Factors are abstract combinations of the original variables, making it harder to directly interpret the coefficients. Factor loadings must be analyzed to understand the influence of individual variables.
*   **Potential Information Loss:** If too few factors are retained, information from the original variables may be lost, reducing predictive accuracy.
*   **Complexity:** The process of extracting factors and interpreting their meaning adds complexity to the modeling process.
*   **Dependence on Factor Selection:** The number of factors to retain is subjective and can affect model performance. Too few factors may oversimplify, while too many may reintroduce multicollinearity.
*   **Assumption of Latent Structure:** Relies on the assumption that an underlying latent structure exists, which may not always be true for all datasets.

### When to Use Which Method
*   Use PCA when the primary goal is **dimensionality reduction**, **data compression**, and **multicollinearity resolution**, especially when interpretability of the new components is secondary to predictive performance.
*   Use Factor Analysis when the goal is to **uncover underlying latent constructs or factors** that explain the relationships among variables, and when you seek to understand the conceptual meaning of these latent variables, even if it adds complexity.
*   The original OLS model is preferable when **interpretability of original variables is crucial** and multicollinearity is not a significant issue.

## Using Principal Components / Factors in Other Models {#sec-other-models}

The principal components from PCA or factors from Factor Analysis can also be effectively used as predictors in other machine learning models, not just linear regression.

### Random Forest Regressor with the Full Dataset
First, a Random Forest Regressor is trained using the original, full dataset (`X_encoded`).

```{python}
# 1. Prepare Data # 
# Use the original input features (X_encoded) as predictors
X_original = X_encoded
# Split the data into training and testing sets
X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(X_original, y, test_size=0.2, random_state=42)

# 2. Fit Random Forest Model 
from sklearn.ensemble import RandomForestRegressor
rf_model_orig = RandomForestRegressor(n_estimators=100, random_state=42)
# Train the model on the training data
rf_model_orig.fit(X_train_orig, y_train_orig)

# 3. Evaluate the Model
# Make predictions on the test set
y_pred_orig = rf_model_orig.predict(X_test_orig)
# Calculate evaluation metrics
r2_rf_orig = r2_score(y_test_orig, y_pred_orig)
mse_rf_orig = mean_squared_error(y_test_orig, y_pred_orig)
rmse_rf_orig = np.sqrt(mse_rf_orig)

# Print the results
print("\nRandom Forest Model (using original data):")
print(f"R-squared: {r2_rf_orig:.4f}")
print(f"MSE: {mse_rf_orig:.4f}")
print(f"RMSE: {rmse_rf_orig:.4f}")
```


### Random Forest Regressor with PCA Components
Next, a Random Forest Regressor is trained using the principal components derived from PCA. This tests if the dimensionality reduction and multicollinearity resolution of PCA benefit non-linear models.

```{python}
# 1. Prepare Data 
# Use the extracted PCA components as predictors (using the 10 components)
X_pca_rf = df_pca_components

# Split the data into training and testing sets
X_train_pca_rf, X_test_pca_rf, y_train_pca_rf, y_test_pca_rf = train_test_split(X_pca_rf, y, test_size=0.2, random_state=42)

# 2. Fit Random Forest Model
# Initialize the Random Forest Regressor
rf_model_pca = RandomForestRegressor(n_estimators=100, random_state=42)
# Train the model on the training data
rf_model_pca.fit(X_train_pca_rf, y_train_pca_rf)

#  3. Evaluate the Model 
# Make predictions on the test set
y_pred_pca_rf = rf_model_pca.predict(X_test_pca_rf)
# Calculate evaluation metrics
r2_rf_pca = r2_score(y_test_pca_rf, y_pred_pca_rf)
mse_rf_pca = mean_squared_error(y_test_pca_rf, y_pred_pca_rf)
rmse_rf_pca = np.sqrt(mse_rf_pca)

# Print the results
print("\nRandom Forest Model (using PCA components):")
print(f"R-squared: {r2_rf_pca:.4f}")
print(f"MSE: {mse_rf_pca:.4f}")
print(f"RMSE: {rmse_rf_pca:.4f}")
```


##### Random Forest Regressor with Extracted Factors (from FA)
Finally, a Random Forest Regressor is trained using the extracted factors from Factor Analysis (using the 3 factors from the reduced model for this example to illustrate potential impact of reduction).

```{python}
# 1. Prepare Data 
# Use the extracted factors as predictors (using the 3 factors from the reduced FA model)
X_factors_rf = df_factors_reduced

# Split the data into training and testing sets
X_train_fa_rf, X_test_fa_rf, y_train_fa_rf, y_test_fa_rf = train_test_split(X_factors_rf, y, test_size=0.2, random_state=42)

# 2. Fit Random Forest Model 
# Initialize the Random Forest Regressor
rf_model_fa = RandomForestRegressor(n_estimators=100, random_state=42)
# Train the model on the training data
rf_model_fa.fit(X_train_fa_rf, y_train_fa_rf)

# 3. Evaluate the Model 
# Make predictions on the test set
y_pred_fa_rf = rf_model_fa.predict(X_test_fa_rf)
# Calculate evaluation metrics
r2_rf_fa = r2_score(y_test_fa_rf, y_pred_fa_rf)
mse_rf_fa = mean_squared_error(y_test_fa_rf, y_pred_fa_rf)
rmse_rf_fa = np.sqrt(mse_rf_fa)

# Print the results
print("\nRandom Forest Model (using extracted factors):")
print(f"R-squared: {r2_rf_fa:.4f}")
print(f"MSE: {mse_rf_fa:.4f}")
print(f"RMSE: {rmse_rf_fa:.4f}")
```

##### Comparison of the Random Forest Models {#sec-rf-comparison}
```{python}
# Print comparison of Random Forest models
print("\nComparison of Random Forest Models:")
print("\nUsing Original Data:")
print(f"R-squared: {r2_rf_orig:.4f}")
print(f"MSE: {mse_rf_orig:.4f}")
print(f"RMSE: {rmse_rf_orig:.4f}")

print("\nUsing PCA Components:")
print(f"R-squared: {r2_rf_pca:.4f}")
print(f"MSE: {mse_rf_pca:.4f}")
print(f"RMSE: {rmse_rf_pca:.4f}")

print("\nUsing Extracted Factors (from FA):")
print(f"R-squared: {r2_rf_fa:.4f}")
print(f"MSE: {mse_rf_fa:.4f}")
print(f"RMSE: {rmse_rf_fa:.4f}")
```


In this example, for Random Forest, using all PCA components results in identical performance to the original data, suggesting PCA successfully preserved information for this non-linear model. However, using the reduced set of 3 factors from Factor Analysis led to a decrease in R-squared and an increase in MSE/RMSE compared to using the original variables. This highlights that while dimensionality reduction can be beneficial, choosing too few components or factors can lead to information loss, negatively impacting predictive performance.

#### Multicollinearity: Conclusion and Recommendation
Multicollinearity is a common issue in regression models that can lead to unstable and difficult-to-interpret coefficients. Both Principal Component Analysis (PCA) and Factor Analysis (FA) are powerful techniques for addressing multicollinearity and reducing dimensionality.

*   **PCA** is a standard method for addressing multicollinearity by transforming correlated variables into uncorrelated principal components. These components can be effectively used in linear regression and other models like Random Forest. While PCA components are not always easy to interpret directly in terms of original variables, they excel at data compression and reducing model complexity.
*   **Factor Analysis** provides a way to simplify data by identifying underlying latent structures (factors) that explain correlations among variables. It also results in uncorrelated factors, making it suitable for regression problems affected by multicollinearity. Interpretation of factors relies on factor loadings.

The choice between PCA and Factor Analysis depends on the specific goals: PCA for dimensionality reduction and variance explanation, FA for discovering latent constructs. Both are valuable tools in the data scientist's toolkit for handling complex, highly correlated datasets.

## Videos: Principal Component Analysis (PCA)

* Video: [Principal Component Analysis (PCA), Step-by-Step](https://youtu.be/FgakZw6K1QQ?si=lmXhc-bpOqb7RmDP)
* Video: [PCA - Practical Tips](https://youtu.be/oRvgq966yZg?si=TIUsxNItfyYOjTLt)
* Video: [PCA in Python](https://youtu.be/Lsue2gEM9D0?si=_fV_RzK8j1jwcb-e)


## Jupyter Notebook

:::{.callout-note}

* The Jupyter-Notebook of this lecture is available on GitHub in the [Hyperparameter-Tuning-Cookbook Repository](https://github.com/sequential-parameter-optimization/Hyperparameter-Tuning-Cookbook/blob/main/100_ddmo_pca.ipynb)

:::

