---
lang: de
eval: true
---

# Lernmodul: Die Cholesky-Zerlegung

## Einführung

Die Cholesky-Zerlegung ist ein grundlegendes Werkzeug in der numerischen linearen Algebra, das speziell für symmetrische, positiv definite Matrizen entwickelt wurde. Sie zerlegt eine solche Matrix in das Produkt einer unteren Dreiecksmatrix und ihrer Transponierten. Diese Zerlegung ist nicht nur rechnerisch effizient, sondern auch numerisch stabil, was sie zu einer bevorzugten Methode in vielen angewandten Bereichen macht, insbesondere im wissenschaftlichen Rechnen und bei der Modellierung von Systemen, die durch teure Computersimulationen beschrieben werden.

Im Kontext von Ersatzmodellen (*surrogate models*) und Gauß-Prozessen (Kriging) spielt die Cholesky-Zerlegung eine zentrale Rolle bei der effizienten Lösung linearer Gleichungssysteme und der Berechnung von Determinanten, die für die Modellanpassung und Vorhersage erforderlich sind.

## Definition und Eigenschaften

### Symmetrische, positiv definite Matrizen

Die Cholesky-Zerlegung ist ausschließlich für **symmetrische, positiv definite Matrizen** definiert.

*   Eine Matrix $A$ ist **symmetrisch**, wenn sie gleich ihrer Transponierten ist, d.h., $A = A^T$.
*   Eine symmetrische Matrix $A$ ist **positiv definit**, wenn alle ihre Eigenwerte positiv sind. Eine äquivalente Definition besagt, dass für jeden von Null verschiedenen Vektor $\vec{x}$ gilt: $\vec{x}^T A \vec{x} > 0$. Diese Eigenschaft ist entscheidend, da sie die Eindeutigkeit einer Lösung garantiert und numerische Stabilität gewährleistet. Wenn eine Matrix nicht positiv definit ist, kann die Cholesky-Zerlegung fehlschlagen und einen Fehler auslösen.

### Die Zerlegung

Für eine symmetrische, positiv definite Matrix $A$ findet die Cholesky-Zerlegung eine untere Dreiecksmatrix $L$ (oder eine obere Dreiecksmatrix $U$) derart, dass:

$$
A = L L^T
$$

oder

$$
A = U^T U.
$$

Hierbei ist $L^T$ die Transponierte von $L$. Wenn NumPy's `cholesky`-Funktion verwendet wird, liefert sie standardmäßig den unteren Dreiecksfaktor $L$; um den oberen Dreiecksfaktor $U$ zu erhalten, muss $L$ transponiert werden (`U = cholesky(Psi).T`).

###  Vorteile der Cholesky-Zerlegung

Die Cholesky-Zerlegung bietet erhebliche Vorteile gegenüber der direkten Matrixinversion:

*   **Recheneffizienz**: Sie reduziert die rechnerische Komplexität von $O(n^3)$ für die direkte Inversion auf etwa $O(n^3/3)$.
*   **Numerische Stabilität**: Die Methode ist numerisch äußerst stabil und robust gegenüber Rundungsfehlern bei Gleitkomma-Berechnungen. Dies ist besonders wichtig bei schlecht konditionierten Matrizen, wo die Determinante nahe Null liegt, was zu Instabilität führen kann.

## Berechnung der Cholesky-Zerlegung

Die Cholesky-Zerlegung kann algorithmisch durchgeführt werden. Für eine Matrix 
$A = \begin{pmatrix} a_{11} & a_{12} \\ a_{21} & a_{22} \end{pmatrix}$ und $L = \begin{pmatrix} l_{11} & 0 \\ l_{21} & l_{22} \end{pmatrix}$ gilt $A = LL^T$.

Dies führt zu den Gleichungen:

\begin{align}
a_{11} &= l_{11}^2 \\
a_{21} &= l_{21} l_{11} \\
a_{22} &= l_{21}^2 + l_{22}^2
\end{align}
Die Elemente von $L$ können dann wie folgt bestimmt werden:
\begin{align}
l_{11} &= \sqrt{a_{11}} \\
l_{21} &= \frac{a_{21}}{l_{11}} \\
l_{22} &= \sqrt{a_{22} - l_{21}^2}
\end{align}

Dies ist der Kernalgorithmus der Cholesky-Zerlegung, der sich auf größere Matrizen erweitern lässt.


### Beispiel mit $\Psi$

Betrachten wir die Korrelationsmatrix $\Psi$ aus den Quellen:
$$
\Psi = \begin{pmatrix} 1 & e^{-1}\ e^{-1} & 1 \end{pmatrix}
$$
Um die Cholesky-Zerlegung $\Psi = LDL^T$ (oder $U^TDU$) zu berechnen, setzen wir:
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
\end{pmatrix}
$$
Multipliziert man dies aus, erhält man:
$$
\begin{pmatrix}
d_{11} & d_{11} l_{21} \\
d_{11} l_{21} & d_{11} l_{21}^2 + d_{22}
\end{pmatrix}
$$
Durch Koeffizientenvergleich mit $\Psi$:

1.  $d_{11} = 1$
2.  $l_{21}d_{11} = e^{-1} \Rightarrow l_{21} = e^{-1}$
3.  $d_{11} l_{21}^2 + d_{22} = 1 \Rightarrow d_{22} = 1 - e^{-2}$

Die Cholesky-Zerlegung von $\Psi$ ist somit:
$$
\Psi = 
\begin{pmatrix}
1 & 0 \\
e^{-1} & 1 
\end{pmatrix}
\begin{pmatrix}
1 & 0 \\
0 & 1 - e^{-2}
\end{pmatrix}
\begin{pmatrix}
1 & e^{-1} \\
0 & 1
\end{pmatrix} = LDL^T
$$
Alternativ, ohne die explizite Diagonalmatrix $D$:
$$
\Psi =
\begin{pmatrix}
1 & 0 \\
e^{-1} & \sqrt{1 - e^{-2}}
\end{pmatrix}
\begin{pmatrix}
1 & e^{-1}\\
0 & \sqrt{1 - e^{-2}}
\end{pmatrix} = U^TU.
$$


## Anwendungen der Cholesky-Zerlegung

Die Cholesky-Zerlegung ist ein vielseitiges Werkzeug in der numerischen linearen Algebra:

### Lösung linearer Gleichungssysteme

Anstatt die Inverse $A^{-1}$ explizit zu berechnen (was numerisch instabil und teuer ist),
kann ein lineares System 
$$
A\vec{x} = \vec{b}
$$
mittels Cholesky-Zerlegung in zwei einfachere Dreieckssysteme zerlegt werden.

1.  **Vorwärtssubstitution**: Löse $L\vec{y} = \vec{b}$ nach $\vec{y}$. Da $L$ eine untere Dreiecksmatrix ist, lässt sich dies leicht rekursiv lösen.
2.  **Rückwärtssubstitution**: Löse $L^T\vec{x} = \vec{y}$ nach $\vec{x}$. Da $L^T$ (oder $U$) eine obere Dreiecksmatrix ist, lässt sich dies ebenfalls leicht rekursiv lösen.

Dieser zweistufige Prozess ist viel schneller und numerisch stabiler als die direkte Inversion.
Im Python-Code wird dies oft durch Funktionen wie `scipy.linalg.cho_solve` oder `numpy.linalg.solve` nach der Cholesky-Zerlegung (`L = cholesky(Psi, lower=True)`) erledigt.

### Berechnung von Determinanten

Für die Berechnung des Logarithmus des Absolutwerts der Determinante einer symmetrischen, positiv definiten Matrix $\Psi$, was in der Maximum-Likelihood-Schätzung oft vorkommt, ist die Cholesky-Zerlegung besonders nützlich. Es gilt:
$$
\ln(|\Psi|) = 2\sum_{i=1}^{n} \ln(L_{ii})
$$
wobei $L_{ii}$ die Diagonalelemente der Cholesky-Faktorisierung $L$ sind.
Dieser Ansatz vermeidet die direkte Berechnung der Determinante, die bei schlecht konditionierten Matrizen sehr kleine Werte annehmen und zu numerischer Instabilität führen kann.

### Kriging und Gauß-Prozess-Regression (GPR)

Im Bereich der Gauß-Prozess-Modellierung, auch bekannt als Kriging, ist die Cholesky-Zerlegung ein Kernbestandteil.

*   **Modellanpassung (MLE)**: Bei der Schätzung der Modellparameter über die Maximum-Likelihood-Methode erfordert die Berechnung der Likelihood-Funktion (oder der konzentrierten Log-Likelihood) mehrere Matrixinversionen. Die Cholesky-Zerlegung, gefolgt von Vorwärts- und Rückwärtssubstitution, ist der schnellste und stabilste Weg, dies zu tun.
*   **Vorhersage (BLUP)**: Die Vorhersage neuer Werte im Kriging-Modell, bekannt als Best Linear Unbiased Predictor (BLUP), beinhaltet ebenfalls die Lösung linearer Systeme mit der Korrelationsmatrix. Auch hier kommt die Cholesky-Zerlegung zum Einsatz, um die Vorhersage effizient und stabil zu berechnen.
*   **Numerische Stabilität und Nugget-Effekt**: Wenn Trainingspunkte sehr nahe beieinander liegen, kann die Korrelationsmatrix schlecht konditioniert oder nahezu singulär werden, was die Cholesky-Zerlegung zum Scheitern bringen kann. Um dies zu verhindern, wird ein kleiner positiver Wert, der so genannte **Nugget-Effekt** ($\lambda$ oder `eps`), zur Diagonale der Korrelationsmatrix addiert:
$$
\Psi_{new} = \Psi + \lambda I.
$$
Dieser Nugget stellt sicher, dass die Matrix streng positiv definit und somit die Cholesky-Zerlegung erfolgreich ist. Der Nugget kann auch als statistischer Parameter interpretiert werden, der Rauschen im Modell berücksichtigt und das Modell von einem exakten Interpolator zu einem rauschfilternden Regressor ändert.

### Generierung von Stichproben aus multivariaten Normalverteilungen

Die Cholesky-Zerlegung kann auch verwendet werden, um Zufallsstichproben aus einer multivariaten Normalverteilung zu generieren.
Wenn $\vec{u}$ ein Vektor von unabhängigen standardnormalverteilten Zufallsvariablen ist und $K = LL^T$ die Cholesky-Zerlegung der Kovarianzmatrix $K$ ist, dann hat der Vektor $\vec{x} = \vec{\mu} + L\vec{u}$ die gewünschte multivariate Normalverteilung mit Mittelwert $\vec{\mu}$ und Kovarianzmatrix $K$. Auch hier kann ein kleiner "Nugget"-Term zur Kovarianzmatrix hinzugefügt werden, um numerische Stabilität zu gewährleisten, da die Eigenwerte von Kovarianzmatrizen schnell abfallen können.

### Anwendungen in Optimierungsalgorithmen

Obwohl die Cholesky-Zerlegung selbst kein Optimierungsalgorithmus ist (im Gegensatz zu Gradientenabstieg, Newton-Verfahren oder BFGS), ist sie ein entscheidendes Werkzeug zur Lösung der linearen Systeme, die in vielen fortgeschrittenen Optimierungsverfahren auftreten. Beispielsweise erfordert das Newton-Verfahren zur Minimierung einer Funktion die Lösung eines linearen Systems mit der Hesse-Matrix. Wenn diese Hesse-Matrix symmetrisch und positiv definit ist, kann die Cholesky-Zerlegung für die effiziente und stabile Lösung dieses Systems genutzt werden.


## Implementierung in Python

Python-Bibliotheken wie `numpy` bieten effiziente Funktionen zur Cholesky-Zerlegung, z.B. `np.linalg.cholesky(A)`. Wenn die Matrix nicht positiv definit ist, wird ein `LinAlgError` ausgelöst.


:::{#exm-py-cholesky-de}

#### Cholesky-Zerlegung in Python

```{python}
import numpy as np

# Definieren der symmetrischen, positiv-definiten Matrix A 
A = np.array([[25., 15., -5.],
              [15., 18.,  0.],
              [-5.,  0., 11.]])

# Versuch, die Cholesky-Zerlegung durchzuführen
try:
    # Berechne die untere Dreiecksmatrix L
    L = np.linalg.cholesky(A)
    
    print("Matrix A:\n", A)
    print("\nUntere Dreiecksmatrix L:\n", L)
    
    # Überprüfung: L * L^T sollte wieder A ergeben
    # np.dot für Matrixmultiplikation, L.T für Transponierung
    A_reconstructed = np.dot(L, L.T)
    print("\nRekonstruierte Matrix L * L.T:\n", A_reconstructed)
    
    # Überprüfen, ob die Rekonstruktion erfolgreich war
    print("\nIst die Rekonstruktion nahe an der Originalmatrix?:", 
          np.allclose(A, A_reconstructed))

except np.linalg.LinAlgError:
    print("\nFehler: Die Matrix ist nicht positiv-definit.")
```
:::


:::{#exm-py-cholesky-lgs-de}
#### Lösung von Ax=b
Die primäre Anwendung der Cholesky-Zerlegung ist die effiziente Lösung von linearen Gleichungssystemen 
$$
A\vec{x} = \vec{b},
$$ bei denen $A$ eine symmetrische, positiv definite Matrix ist. Anstatt die Matrix $A$ direkt zu invertieren – ein rechenintensiver und numerisch oft instabiler Prozess – zerlegt man das Problem in zwei einfachere Schritte:

1. **Zerlegung:** Zuerst wird $A$ in $LL^T$ zerlegt. Das System wird zu $LL^T\vec{x} = \vec{b}$.
2. **Substitution:** Das System wird in zwei Schritten gelöst, indem ein Hilfsvektor $\vec{y}$ eingeführt wird:
    (i) **Vorwärtssubstitution:** Löse $L\vec{y} = \vec{b}$ nach $\vec{y}$. Da $L$ eine untere Dreiecksmatrix ist, kann dies sehr effizient geschehen.
    (ii) **Rückwärtssubstitution:** Löse $L^T\vec{x} = \vec{y}$ nach $\vec{x}$. Da $L^T$ eine obere Dreiecksmatrix ist, ist auch dieser Schritt sehr effizient.

Die Bibliothek `SciPy` bietet optimierte Funktionen für diesen Prozess.

```{python}
from scipy.linalg import solve_triangular, cho_solve
import numpy as np

# Verwenden der gleichen Matrix A und der berechneten Matrix L
A = np.array([[25., 15., -5.],
              [15., 18.,  0.],
              [-5.,  0., 11.]])
L = np.linalg.cholesky(A)

# Definieren des Vektors b
b = np.array([10., 5., 8.])

# --- Methode 1: Manuelle Vorwärts- und Rückwärtssubstitution ---
# Schritt 2a: Löse Ly = b für y (Vorwärtssubstitution)
y = solve_triangular(L, b, lower=True)
print("Zwischenvektor y:\n", y)

# Schritt 2b: Löse L^T x = y für x (Rückwärtssubstitution)
x1 = solve_triangular(L.T, y, lower=False)
print("\nLösung x (manuelle Substitution):\n", x1)

# --- Methode 2: Direkte Lösung mit cho_solve ---
# Diese Funktion kombiniert die Schritte für maximale Effizienz
# Sie benötigt die Cholesky-Faktorisierung (L) und b
x2 = cho_solve((L, True), b) # (L, True) bedeutet, dass L eine untere Dreiecksmatrix ist
print("\nLösung x (mit cho_solve):\n", x2)

# Überprüfung der Lösung: A * x sollte wieder b ergeben
print("\nÜberprüfung A * x:\n", np.dot(A, x2))
```
:::


## Fazit

Die Cholesky-Zerlegung ist ein unverzichtbares Werkzeug in der numerischen linearen Algebra für symmetrische, positiv definite Matrizen. Ihre Effizienz und Robustheit machen sie zur bevorzugten Methode für Aufgaben wie die Lösung linearer Gleichungssysteme, die Berechnung von Determinanten und insbesondere in den rechenintensiven Prozessen von Gauß-Prozessen und Ersatzmodellen. Das Verständnis ihrer Funktionsweise und ihrer Anwendungsbereiche ist für jeden Ingenieur und Naturwissenschaftler, der mit komplexen Berechnungen arbeitet, von grundlegender Bedeutung.

## Zusatzmaterialien

:::{.callout-note}
#### Interaktive Webseite

* Eine interaktive Webseite zum Thema **Cholesky-Zerlegung** ist hier zu finden: [Cholesky Interaktiv](https://advm1.gm.fh-koeln.de/~bartz/bart21i/de_cholesky_interactive.html).

:::

:::{.callout-note}
#### Audiomaterial

* Eine Audio zum Thema **Cholesky-Zerlegung** ist hier zu finden: [Cholesky Audio](https://advm1.gm.fh-koeln.de/~bartz/bart21i/audio/cholesky.m4a).


:::

:::{.callout-note}
#### Jupyter-Notebook

* Das Jupyter-Notebook für dieses Lernmodul ist auf GitHub im [Hyperparameter-Tuning-Cookbook Repository](https://github.com/sequential-parameter-optimization/Hyperparameter-Tuning-Cookbook/blob/main/de_cholesky.ipynb) verfügbar.

:::
