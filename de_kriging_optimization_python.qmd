---
lang: de
eval: true
---

# Lernmodul: Erweiterung des Kriging-Modells: Numerische Optimierung der Hyperparameter (Python Code)


```{python}
import numpy as np
import matplotlib.pyplot as plt
from numpy import (exp, multiply, eye, linspace, spacing, sqrt)
from numpy.linalg import cholesky, solve
from scipy.spatial.distance import squareform, pdist, cdist
from scipy.optimize import minimize # Für die Optimierung

def build_Psi(X, theta, eps=sqrt(spacing(1))):
    """
    Berechnet die Korrelationsmatrix Psi basierend auf paarweisen
    quadratischen euklidischen Distanzen zwischen Eingabelokationen,
    skaliert mit theta.
    Fügt ein kleines Epsilon zur Diagonalen für numerische Stabilität
    hinzu (Nugget-Effekt).
    Hinweis: p_j ist implizit 2 aufgrund der 'sqeuclidean'-Metrik.
    """
    # Sicherstellen, dass theta ein 1D-Array für das 'w'-Argument
    # von cdist/pdist ist
    if not isinstance(theta, np.ndarray) or theta.ndim == 0:
        theta = np.array([theta])

    D = squareform(pdist(X, metric='sqeuclidean', w=theta))
    Psi = exp(-D)
    # Ein kleiner Wert wird zur Diagonalen hinzugefügt für
    # numerische Stabilität (Nugget)
    # Korrektur: X.shape für die Anzahl der Zeilen der
    # Identitätsmatrix
    Psi += multiply(eye(X.shape[0]), eps)
    return Psi

def build_psi(X_train, x_predict, theta):
    """
    Berechnet den Korrelationsvektor (oder Matrix) psi zwischen
    neuen Vorhersageorten und Trainingsdatenlokationen.
    """
    # Sicherstellen, dass theta ein 1D-Array für das 'w'-Argument
    # von cdist/pdist ist
    if not isinstance(theta, np.ndarray) or theta.ndim == 0:
        theta = np.array([theta])

    D = cdist(x_predict, X_train, metric='sqeuclidean', w=theta)
    psi = exp(-D)
    return psi.T 
    # Transponieren, um konsistent mit der Literatur zu sein
    # (n x m oder n x 1)

def neg_log_likelihood(params, X_train, y_train):
    """
    Berechnet die negative konzentrierte Log-Likelihood für das Kriging-Modell.
    params: ein 1D-Numpy-Array, wobei params theta ist.
            (Falls auch p optimiert würde, wäre es params usw.)
    X_train: (n, k)-Matrix der Trainings-Eingabelokationen
    y_train: (n, 1)-Vektor der Trainings-Ausgabewerte
    """
    theta = params
    # Für dieses Beispiel ist p implizit auf 2 festgelegt
    # (durch 'sqeuclidean' in build_Psi).
    # Falls p optimiert würde, müsste es hier aus 'params' extrahiert
    # und an build_Psi übergeben werden
    n = X_train.shape[0]

    # 1. Korrelationsmatrix Psi aufbauen
    Psi = build_Psi(X_train, theta)

    # 2. mu_hat berechnen (MLE des Mittelwerts)
    # Verwendung der Cholesky-Zerlegung für stabile Inversion
    try:
        # numpy.cholesky gibt L (untere Dreiecksmatrix) zurück,
        # daher transponieren für U (obere)
        U = cholesky(Psi).T
    except np.linalg.LinAlgError:
        # Bei Fehlern (z.B. wenn Psi nicht positiv definit ist,
        # durch schlechte theta-Werte)
        # einen sehr großen Wert zurückgeben, um diese Parameter zu bestrafen
        return 1e15

    one = np.ones(n).reshape(-1, 1)
    # Stabile Berechnung von Psi_inv @ y und Psi_inv @ one
    Psi_inv_y = solve(U, solve(U.T, y_train))
    Psi_inv_one = solve(U, solve(U.T, one))

    # Berechnung von mu_hat
    mu_hat = (one.T @ Psi_inv_y) / (one.T @ Psi_inv_one)
    mu_hat = mu_hat.item() # Skalaren Wert extrahieren

    # 3. sigma_hat_sq berechnen (MLE der Prozessvarianz)
    y_minus_mu_one = y_train - one * mu_hat
    # Korrekte Berechnung: (y-1*mu_hat).T @ Psi_inv @ (y-1*mu_hat) / n
    sigma_hat_sq = (y_minus_mu_one.T @ \
                    solve(U, solve(U.T, y_minus_mu_one))) / n
    sigma_hat_sq = sigma_hat_sq.item()

    if sigma_hat_sq < 1e-10: # Sicherstellen, dass sigma_hat_sq
        # nicht-negativ und nicht zu klein ist
        return 1e15 # Sehr großen Wert zurückgeben zur Bestrafung

    # 4. Log-Determinante von Psi mittels Cholesky-Zerlegung für
    # Stabilität berechnen.
    # ln(|Psi|) = 2 * Summe(ln(L_ii)) wobei L die untere
    # Dreiecksmatrix der Cholesky-Zerlegung ist
    log_det_Psi = 2 * np.sum(np.log(np.diag(U.T))) # U.T ist L

    # 5. Negative konzentrierte Log-Likelihood berechnen
    # ln(L) = - (n/2) * ln(sigma_hat_sq) - (1/2) * ln(|Psi|)
    # Zu minimieren ist -ln(L)
    nll = 0.5 * n * np.log(sigma_hat_sq) + 0.5 * log_det_Psi
    return nll

n_train = 4  # Anzahl der Stichprobenlokationen
X_train = np.linspace(0, 2 * np.pi, n_train, endpoint=False).reshape(-1, 1)
y_train = np.sin(X_train)  # Zugehörige y-Werte (Sinus von x)

initial_theta_guess = np.array([1.0])  # Startwert für Theta
bounds = [(0.001, 100.0)]  # Suchbereich für Theta

print("\n--- Starte Hyperparameter-Optimierung für Theta ---")
result = minimize(neg_log_likelihood, initial_theta_guess,
                 args=(X_train, y_train),
                 method='L-BFGS-B', bounds=bounds)

opt_theta = result.x
opt_nll = result.fun

print(f"Optimierung erfolgreich: {result.success}")
print(f"Optimales Theta: {opt_theta[0]:.4f}")
print(f"Minimaler Negativer Log-Likelihood: {opt_nll:.4f}")

# Berechne Vorhersage mit optimiertem Theta
Psi_opt = build_Psi(X_train, opt_theta)
U_opt = cholesky(Psi_opt).T
one_opt = np.ones(n_train).reshape(-1, 1)
mu_hat_opt = (one_opt.T @ solve(U_opt, solve(U_opt.T, y_train))) / \
             (one_opt.T @ solve(U_opt, solve(U_opt.T, one_opt)))
mu_hat_opt = mu_hat_opt.item()

m_predict = 100  # Anzahl der neuen Lokationen für die Vorhersage
x_predict = np.linspace(0, 2 * np.pi, m_predict, endpoint=True).reshape(-1, 1)
psi_opt = build_psi(X_train, x_predict, opt_theta)
f_predict_opt = mu_hat_opt * np.ones(m_predict).reshape(-1, 1) + \
                psi_opt.T @ solve(U_opt, solve(U_opt.T, y_train - one_opt * mu_hat_opt))

# Visualisierung
plt.figure(figsize=(10, 6))
plt.plot(x_predict, np.sin(x_predict), color="grey", linestyle='--',
         label="Wahre Sinusfunktion")
plt.plot(X_train, y_train, "bo", markersize=8,
         label=f"Messpunkte ({n_train} Punkte)")
plt.plot(x_predict, f_predict_opt, color="orange",
         label=f"Kriging-Vorhersage (Theta={opt_theta[0]:.2f})")
plt.title(f"Kriging-Vorhersage der Sinusfunktion mit {n_train} Punkten\n" + 
          f"Optimierter Aktivitätsparameter Theta={opt_theta[0]:.2f}")
plt.xlabel("x")
plt.ylabel("y")
plt.legend(loc='upper right')
plt.grid(True)
plt.show()


