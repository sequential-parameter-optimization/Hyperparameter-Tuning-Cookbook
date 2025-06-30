---
lang: de
eval: false
---

# Lernmodul: Eine Einführung in Kriging

## Konzeptionelle Grundlagen des Kriging

### Von einfachen Modellen zur intelligenten Interpolation

In der modernen ingenieur- und naturwissenschaftlichen Forschung werden Praktiker häufig mit „Black-Box“-Funktionen konfrontiert. Dies sind Systeme oder Simulationen, deren interne Funktionsweise entweder unbekannt oder so komplex ist, dass sie praktisch undurchschaubar ist. Ein gängiges Beispiel ist eine hochpräzise Simulation der numerischen Strömungsmechanik (CFD), bei der Eingaben wie die Flügelgeometrie oder die Strömungsgeschwindigkeit Ausgaben wie Auftrieb und Luftwiderstand erzeugen. Jede Auswertung dieser Black Box kann außerordentlich teuer sein und Stunden oder sogar Tage an Supercomputerzeit in Anspruch nehmen. Wenn das Ziel darin besteht, den Designraum zu erkunden oder ein optimales Design zu finden, ist die Durchführung von Tausenden dieser Auswertungen oft nicht durchführbar.

Diese Herausforderung führt zur Notwendigkeit von **Surrogatmodellen**, auch bekannt als Metamodelle oder Antwortflächenmodelle "Response-Surface". Ein Surrogatmodell ist eine rechengünstige Annäherung an eine teure Black-Box-Funktion. Es wird konstruiert, indem eine kleine Anzahl sorgfältig ausgewählter Auswertungen der wahren Funktion durchgeführt und dann ein mathematisches Modell an diese beobachteten Datenpunkte angepasst wird. Dieses „Modell eines Modells“ kann dann Tausende Male zu vernachlässigbaren Kosten ausgewertet werden, was eine effiziente Optimierung, Sensitivitätsanalyse und Erkundung des Designraums ermöglicht.

#### Eine Brücke zum Kriging: Verständnis von Radialen Basisfunktionen (RBFs)

Eine leistungsstarke und intuitive Klasse von Surrogatmodellen ist das **Modell der Radialen Basisfunktionen (RBF)**. Die grundlegende Idee hinter einem RBF ist es, eine komplexe Funktion als gewichtete Summe einfacherer, gut verstandener Basisfunktionen darzustellen. Jede Basisfunktion ist an einem der bekannten Datenpunkte zentriert, und ihr Wert hängt nur vom Abstand zu diesem Zentrum ab.

Mathematisch hat ein RBF-Modell die Form:
$$
\hat{f}(\vec{x}) = \sum_{i=1}^{n} w_i \psi(||\vec{x} - \vec{c}^{(i)}||)
$$
wobei $\hat{f}(\vec{x})$ der vorhergesagte Wert an einem neuen Punkt $\vec{x}$ ist, $w_i$ die Gewichte sind, $\vec{c}^{(i)}$ die Zentren der Basisfunktionen (typischerweise die Standorte der bekannten Datenpunkte, $\vec{x}^{(i)}$) und $\psi$ die radiale Basisfunktion selbst ist, die auf dem euklidischen Abstand $||\vec{x} - \vec{c}^{(i)}||$ operiert.

Gängige Wahlen für $\psi$ umfassen die linearen, kubischen, Gauß'schen oder multiquadratischen Funktionen [@Forr08a]. Indem wir fordern, dass das Modell exakt durch alle bekannten Datenpunkte verläuft (ein Prozess, der als Interpolation bezeichnet wird), können wir ein System linearer Gleichungen aufstellen, um die unbekannten Gewichte $w_i$ zu lösen. Dies wird typischerweise in Matrixform geschrieben als:
$$
\Psi \vec{w} = \vec{y}
$$
wobei $\Psi$ eine Matrix der Auswertungen der Basisfunktionen ist, $\vec{w}$ der Vektor der Gewichte und $\vec{y}$ der Vektor der beobachteten Antworten ist. Das Lösen nach den Gewichten ist dann eine Frage der Matrixinversion: $\vec{w} = \Psi^{-1} \vec{y}$. Die Schönheit dieses Ansatzes liegt darin, dass er ein potenziell hochgradig nichtlineares Modellierungsproblem in ein unkompliziertes lineares Algebraproblem umwandelt [@Forr08a].

Diese Struktur weist eine bemerkenswerte Ähnlichkeit mit anderen Modellierungsparadigmen auf. Die RBF-Formulierung ist funktional identisch mit einem einschichtigen künstlichen neuronalen Netz, bei dem die Neuronen eine radiale Aktivierungsfunktion verwenden. In dieser Analogie ist die Eingabe für jedes Neuron der Abstand von einem Zentrum, die Aktivierungsfunktion des Neurons ist die Basisfunktion $\psi$, und die Ausgabe des Netzwerks ist die gewichtete Summe dieser Aktivierungen. Diese Verbindung bietet ein nützliches mentales Modell für diejenigen, die mit maschinellem Lernen vertraut sind, und rahmt RBFs nicht als esoterische statistische Technik, sondern als nahen Verwandten von neuronalen Netzen ein, die beide leistungsstarke universelle Funktionsapproximatoren sind.

#### Einordnung des Kriging

In dieser Landschaft tritt das **Kriging** als eine besonders anspruchsvolle und flexible Art eines RBF-Modells hervor. Ursprünglich aus dem Bereich der Geostatistik durch die Arbeit von Danie G. Krige und Georges Matheron stammend, wurde es entwickelt, um Erzkonzentrationen im Bergbau vorherzusagen [@Forr08a]. Seine Anwendung auf deterministische Computerexperimente wurde von @Sack89a vorangetrieben und ist seitdem zu einem Eckpfeiler des Ingenieurdesigns und der Optimierung geworden.

Im Bereich des maschinellen Lernens ist Kriging besser bekannt als **Gauß-Prozess-Regression (GPR)**. Obwohl sich die Terminologie unterscheidet, ist das zugrunde liegende mathematische Gerüst dasselbe. Kriging unterscheidet sich von einfacheren RBF-Modellen durch seine einzigartige Basisfunktion und seine statistische Grundlage, die nicht nur eine Vorhersage, sondern auch ein Maß für die Unsicherheit dieser Vorhersage liefert.

### Die Kernphilosophie des Kriging: Eine stochastische Prozessperspektive

Um das Kriging wirklich zu verstehen, muss man einen konzeptionellen Sprung wagen, der zunächst kontraintuitiv sein kann. Selbst bei der Modellierung eines perfekt deterministischen Computercodes – bei dem dieselbe Eingabe immer genau dieselbe Ausgabe erzeugt – behandelt das Kriging die Ausgabe der Funktion als eine einzelne Realisierung eines **stochastischen (oder zufälligen) Prozesses**.

Das bedeutet nicht, dass wir annehmen, die Funktion sei zufällig. Stattdessen drücken wir unsere Unsicherheit über den Wert der Funktion an nicht beobachteten Stellen aus. Bevor wir Daten haben, könnte der Wert der Funktion an jedem Punkt alles sein. Nachdem wir einige Punkte beobachtet haben, ist unsere Unsicherheit reduziert, aber sie existiert immer noch überall sonst. Das stochastische Prozessgerüst bietet eine formale mathematische Sprache, um diese Unsicherheit zu beschreiben.

#### Das Prinzip der Lokalität und Korrelation

Dieser angenommene stochastische Prozess ist nicht völlig unstrukturiert. Er wird von einer **Korrelationsstruktur** bestimmt, die eine grundlegende Annahme über die Welt verkörpert: das Prinzip der Lokalität. Dieses Prinzip besagt, dass Punkte, die im Eingaberaum nahe beieinander liegen, erwartungsgemäß ähnliche Ausgabewerte haben (d. h. sie sind hoch korreliert), während Punkte, die weit voneinander entfernt sind, erwartungsgemäß unähnliche oder unzusammenhängende Ausgabewerte haben (d. h. sie sind unkorreliert). Diese Annahme gilt für die große Mehrheit der physikalischen Phänomene und glatten mathematischen Funktionen, die keine chaotischen, diskontinuierlichen Sprünge aufweisen. Die Korrelation zwischen zwei beliebigen Punkten wird durch eine **Kovarianzfunktion** oder einen **Kernel** quantifiziert, der das Herzstück des Kriging-Modells ist.

#### Gauß-Prozess-Prior

Speziell nimmt das Kriging an, dass dieser stochastische Prozess ein **Gauß-Prozess** ist. Ein Gauß-Prozess ist eine Sammlung von Zufallsvariablen, von denen jede endliche Anzahl eine gemeinsame multivariate Normalverteilung (MVN) hat. Dies ist eine starke Annahme, da eine multivariate Normalverteilung vollständig durch nur zwei Komponenten definiert ist: einen **Mittelwertvektor ($\vec{\mu}$)** und eine **Kovarianzmatrix ($\Sigma$)** [@Forr08a].

Dies ist als der **Gauß-Prozess-Prior** bekannt. Es ist unsere „vorherige Überzeugung“ über die Natur der Funktion, bevor wir Daten gesehen haben. Wir glauben, dass die Funktionswerte an jedem Satz von Punkten gemeinsam gaußverteilt sein werden, zentriert um einen gewissen Mittelwert, mit einer Kovarianzstruktur, die durch den Abstand zwischen den Punkten diktiert wird. Wenn wir Daten beobachten, verwenden wir Bayes'sche Inferenz, um diese vorherige Überzeugung zu aktualisieren, was zu einem **Gauß-Prozess-Posterior** führt. Dieser Posterior ist ebenfalls ein Gauß-Prozess, aber sein Mittelwert und seine Kovarianz wurden aktualisiert, um mit den beobachteten Daten konsistent zu sein. Der Mittelwert dieses posterioren Prozesses gibt uns die Kriging-Vorhersage, und seine Varianz gibt uns ein Maß für die Unsicherheit über diese Vorhersage. Diese statistische Grundlage ist es, die das Kriging auszeichnet und es ihm ermöglicht, nicht nur zu interpolieren, sondern auch seine eigene Zuverlässigkeit zu quantifizieren.

## Die mathematische Architektur eines Kriging-Modells

Um vom konzeptionellen Ansatz des Kriging zur praktischen Umsetzung zu gelangen, ist es unerlässlich, seine mathematischen Komponenten zu verstehen. Dieser Abschnitt analysiert die Architektur des Modells und verbindet konsequent die abstrakte mathematische Notation aus Referenztexten mit den konkreten Variablen, die im bereitgestellten Python-Code verwendet werden.

### Glossar der Kriging-Notation

 @tbl-glossar-de dient als Glossar, um die Notation aus [@Forr08a], dem Kochbuch [@bart23iArXiv] und dem in diesem Dokument in @sec-example-de bereitgestellten Python-Code abzugleichen.

| Mathematisches Symbol | Konzeptionelle Bedeutung | Python-Variable |
| :--- | :--- | :--- |
| $n$ | Anzahl der Trainings-/Stichprobenpunkte | `n` oder `X_train.shape` |
| $k$ | Anzahl der Eingabedimensionen/Variablen | `X_train.shape` |
| $m$ | Anzahl der Punkte für die Vorhersage | `m` oder `x_predict.shape` |
| $X$ | $n \times k$ Matrix der Trainingspunkt-Standorte | `X_train` |
| $y$ | $n \times 1$ Vektor der beobachteten Antworten | `y_train` |
| $\vec{x}$ | Ein neuer Standort für die Vorhersage | Eine Zeile in `x_predict` |
| $\Psi$ (Psi) | $n \times n$ Korrelationsmatrix der Trainingsdaten | `Psi` |
| $\vec{\psi}$ (psi) | $n \times m$ Vorhersage-Trainings-Korrelationsmatrix | `psi` |
| $\vec{\theta}$ (theta) | $k \times 1$ Vektor der Aktivitäts-/Breiten-Hyperparameter | `theta` |
| $\vec{p}$ | $k \times 1$ Vektor der Glattheits-Hyperparameter | Implizit $p=2$ im Code |
| $\mu$ (mu) | Der globale Mittelwert des stochastischen Prozesses | `mu_hat` |
| $\sigma^2$ (sigma-quadrat) | Die Varianz des stochastischen Prozesses | Im Code nicht explizit berechnet |
| $\lambda$ (lambda) | Der Regressions-/Nugget-Parameter | `eps` |
| $\hat{y}(\vec{x})$ | Die Kriging-Vorhersage am Punkt $\vec{x}$ | `f_predict` |

: Glossar {#tbl-glossar-de}

### Der Korrelationskernel: Quantifizierung von Beziehungen

Der Kern des Kriging-Modells ist seine spezialisierte Basisfunktion, auch als Kernel oder Kovarianzfunktion bekannt. Diese Funktion definiert die Korrelation zwischen zwei beliebigen Punkten im Designraum. Die gebräuchlichste Form, und die in unseren Referenztexten verwendete, ist der Gauß'sche Kernel.

Die Kriging-Basisfunktion ist definiert als:
$$\psi(\vec{x}^{(i)}, \vec{x}) = \exp\left(-\sum_{j=1}^{k} \theta_j |x_j^{(i)} - x_j|^{p_j}\right)$$
Diese Gleichung berechnet die Korrelation zwischen einem bekannten Punkt $\vec{x}^{(i)}$ und jedem anderen Punkt $\vec{x}$. Sie wird von zwei Schlüsselsätzen von Hyperparametern gesteuert: $\vec{\theta}$ und $\vec{p}$.

#### Hyperparameter $\vec{\theta}$ (Theta): Der Aktivitätsparameter

Der Parametervektor $\vec{\theta} = \{\theta_1, \theta_2,..., \theta_k\}^T$ ist wohl der wichtigste Hyperparameter im Kriging-Modell. Jede Komponente $\theta_j$ steuert, wie schnell die Korrelation mit dem Abstand entlang der $j$-ten Dimension abfällt. Er wird oft als „Aktivitäts“- oder „Breiten“-Parameter bezeichnet.

*   Ein **großes $\theta_j$** zeigt an, dass die Funktion sehr empfindlich auf Änderungen in der $j$-ten Variablen reagiert. Die Korrelation wird sehr schnell abfallen, wenn sich die Punkte in dieser Dimension voneinander entfernen, was zu einer „schmalen“ Basisfunktion führt. Dies impliziert, dass die zugrunde liegende Funktion entlang dieser Achse sehr „aktiv“ ist oder sich schnell ändert.
*   Ein **kleines $\theta_j$** zeigt an, dass die Funktion relativ unempfindlich auf Änderungen in der $j$-ten Variablen reagiert. Die Korrelation wird langsam abfallen, was zu einer „breiten“ Basisfunktion führt, die ihren Einfluss über einen größeren Bereich ausdehnt.
Da die Korrelation auch über größere Distanzen hinweg hoch bleibt, bedeutet dies, dass das Modell davon ausgeht, dass Punkte, die in der $j$-ten Dimension weiter voneinander entfernt sind, immer noch stark korreliert sind. und dass der Einfluss eines Datenpunktes sich über einen großen Bereich im Eingaberaum erstreckt, bevor die Korrelation signifikant abnimmt. Das Modell geht also davon aus, dass sich die zugrunde liegende Funktion entlang dieser Achse sehr "langsam" ändert oder "inaktiv" ist.

Die Tatsache, dass $\vec{\theta}$ ein Vektor ist – mit einem separaten Wert für jede Eingabedimension – ist ein entscheidendes Merkmal, das dem Kriging immense Leistungsfähigkeit verleiht, insbesondere bei mehrdimensionalen Problemen. Dies ist als **anisotrope Modellierung** bekannt. Indem die Korrelationslänge für jede Variable unterschiedlich sein kann, kann sich das Modell an Funktionen anpassen, die sich entlang verschiedener Achsen unterschiedlich verhalten. Zum Beispiel könnte eine Funktion sehr schnell auf Temperaturänderungen, aber sehr langsam auf Druckänderungen reagieren. Ein anisotropes Kriging-Modell kann dieses Verhalten erfassen, indem es ein großes $\theta$ für die Temperatur und ein kleines $\theta$ für den Druck lernt.

Diese Fähigkeit hat eine tiefgreifende Konsequenz: **automatische Relevanzbestimmung**. Während des Modellanpassungsprozesses (den wir in @sec-mle diskutieren werden) findet der Optimierungsalgorithmus die $\vec{\theta}$-Werte, die die Daten am besten erklären. Wenn eine bestimmte Eingangsvariable $x_j$ wenig oder keinen Einfluss auf die Ausgabe $y$ hat, wird das Modell einen sehr kleinen Wert für $\theta_j$ lernen. Ein kleines $\theta_j$ macht den Term $\theta_j|x_j^{(i)} - x_j|^{p_j}$ nahe null, was die Korrelation effektiv unempfindlich gegenüber Änderungen in dieser Dimension macht. Daher kann ein Ingenieur nach der Anpassung des Modells den optimierten $\vec{\theta}$-Vektor inspizieren, um eine Sensitivitätsanalyse durchzuführen. Die Dimensionen mit den größten $\theta_j$-Werten sind die einflussreichsten Treiber der Systemantwort. Dies verwandelt das Surrogatmodell von einem einfachen Black-Box-Approximator in ein Werkzeug zur Generierung wissenschaftlicher und technischer Erkenntnisse. Der im [Hyperparameter Tuning Cookbook](https://sequential-parameter-optimization.github.io/Hyperparameter-Tuning-Cookbook/006_num_gp.html#sec-kriging-example-006) bereitgestellte und in @sec-example-de besprochene Python-Code, als eindimensionales Beispiel, vereinfacht dies durch die Verwendung eines einzelnen skalaren `theta`, aber das Verständnis seiner Rolle als Vektor ist entscheidend, um den Nutzen des Kriging in realen Anwendungen zu schätzen [@bart23icode].

#### Hyperparameter $\vec{p}$: Der Glattheitsparameter

Der Parametervektor $\vec{p} = \{p_1, p_2,..., p_k\}^T$ steuert die Glattheit der Funktion an den Datenpunkten. Sein Wert ist typischerweise auf das Intervall $[1,2]$  beschränkt [@Forr08a]. Die Wahl von $p_j$ hat tiefgreifende Auswirkungen auf die Form der resultierenden Basisfunktion:

*   Wenn **$p_j = 2$**, ist die resultierende Funktion unendlich differenzierbar, was bedeutet, dass sie sehr glatt ist. Dies ist im bereitgestellten Python-Code der Fall, was durch die Verwendung der `sqeuclidean`-Distanzmetrik (quadrierter Abstand entspricht $p=2$) implizit ist. Die `sqeuclidean`-Metrik ist auf [https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.sqeuclidean.html](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.sqeuclidean.html) beschrieben.
*   Wenn **$p_j = 1$**, ist die resultierende Funktion stetig, aber an den Datenpunkten nicht differenzierbar, was ihr ein „spitzeres“ oder „stacheligeres“ Aussehen verleiht.

Die Wahl von $p$ spiegelt eine Annahme über die Natur der zugrunde liegenden Funktion wider. Die meisten physikalischen Prozesse sind glatt, was $p=2$ zu einer gängigen und robusten Wahl macht. Für Funktionen mit bekannten scharfen Merkmalen kann jedoch die Optimierung von $p$ in Richtung 1 eine bessere Anpassung ermöglichen.

### Aufbau des Systems: Die Korrelationsmatrizen $\Psi$ und $\vec{\psi}$

Mit dem definierten Korrelationskernel können wir nun die Matrizen konstruieren, die den Kern des Kriging-Systems bilden. Diese Matrizen quantifizieren die Beziehungen zwischen allen Punkten in unserem Problem: den bekannten Trainingspunkten und den neuen Vorhersagepunkten.

#### Die `Psi`-Matrix ($\Psi$): Korrelation der Trainingsdaten

Die Matrix $\Psi$ ist die $n \times n$ Korrelationsmatrix der $n$ Trainingsdaten mit sich selbst. Jedes Element $\Psi_{ij}$ ist die Korrelation zwischen dem Trainingspunkt $\vec{x}^{(i)}$ und dem Trainingspunkt $\vec{x}^{(j)}$, berechnet mit der Basisfunktion.
$$
\Psi_{ij} = \text{corr}(\vec{x}^{(i)}, \vec{x}^{(j)}) = \exp\left(-\sum_{l=1}^{k} \theta_l |x_l^{(i)} - x_l^{(j)}|^{p_l}\right).
$$
Da die Korrelation eines Punktes mit sich selbst perfekt ist, sind die diagonalen Elemente $\Psi_{ii}$ immer gleich 1. Die Matrix ist auch symmetrisch, da der Abstand von Punkt $A$ zu $B$ derselbe ist wie von $B$ zu $A$.

**Code-Analyse (`build_Psi`):** Die bereitgestellte Python-Funktion `build_Psi` implementiert diese Berechnung effizient.

```python
from scipy.spatial.distance import pdist, squareform, cdist
from numpy.linalg import cholesky
import numpy as np
from numpy import sqrt, spacing, exp, multiply, eye
from numpy.linalg import solve
from scipy.spatial.distance import pdist, squareform, cdist
def build_Psi(X, theta, eps=sqrt(spacing(1))):
    D = squareform(pdist(X, metric='sqeuclidean', out=None, w=theta))
    Psi = exp(-D)
    Psi += multiply(eye(X.shape[0]), eps)
    return Psi
```

1.  `D = squareform(pdist(X, metric='sqeuclidean', out=None, w=theta))`: Diese Zeile ist der rechnerische Kern. Die Funktion `scipy.spatial.distance.pdist` berechnet die paarweisen Abstände zwischen allen Zeilen in der Eingabematrix `X_train`.
    *   `metric='sqeuclidean'` gibt an, dass der quadrierte euklidische Abstand, $(x_i - x_j)^2$, verwendet werden soll. Dies setzt implizit den Hyperparameter $p=2$.
    *   `w=theta` wendet den Aktivitätsparameter als Gewicht auf den quadrierten Abstand jeder Dimension an und berechnet $\theta_j(x_{ij} - x_{kj})^2$ für jedes Paar von Punkten $i, k$ und jede Dimension $j$. Für den 1D-Fall im Code ist dies einfach `theta * (x_i - x_j)^2`.
    *   `squareform` wandelt dann den von `pdist` zurückgegebenen komprimierten Abstandsvektor in die vollständige, symmetrische $n \times n$ Abstandsmatrix um, die wir $D$ nennen können.
2.  `Psi = exp(-D)`: Dies führt eine elementweise Potenzierung des Negativen der Abstandsmatrix durch und vervollständigt die Berechnung des Gauß'schen Kernels.
3.  `Psi += multiply(eye(X.shape), eps)`: Diese Zeile addiert eine kleine Konstante `eps` zur Diagonale der $\Psi$-Matrix. Dies ist der „Nugget“-Term, eine entscheidende Komponente sowohl für die numerische Stabilität als auch für die Rauschmodellierung, die in @sec-numerical-best-practices behandelt wird.

#### Der `psi`-Vektor/Matrix ($\vec{\psi}$): Vorhersage-Trainings-Korrelation

Die Matrix $\vec{\psi}$ ist die $n \times m$ Matrix der Korrelationen zwischen den $n$ bekannten Trainingspunkten und den $m$ neuen Punkten, an denen wir eine Vorhersage machen möchten. Jedes Element $\psi_{ij}$ ist die Korrelation zwischen dem $i$-ten Trainingspunkt $\vec{x}^{(i)}$ und dem $j$-ten Vorhersagepunkt $\vec{x}_{pred}^{(j)}$.
$$\psi_{ij} = \text{corr}(\vec{x}^{(i)}, \vec{x}_{pred}^{(j)})$$

**Code-Analyse (`build_psi`):** Die Funktion `build_psi` berechnet diese Matrix.

```python
def build_psi(X_train, x_predict, theta):
    D = cdist(x_predict, X_train, metric='sqeuclidean', out=None, w=theta)
    psi = exp(-D)
    return psi.T
```

1.  `D = cdist(x_predict, X_train, metric='sqeuclidean', out=None, w=theta)`: Hier wird `scipy.spatial.distance.cdist` verwendet, siehe [https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html), da wir Abstände zwischen Punkten aus zwei *verschiedenen* Mengen berechnen: den $m$ Vorhersagepunkten in `x_predict` und den $n$ Trainingspunkten in `X_train`. Dies führt zu einer $m \times n$ Matrix gewichteter quadrierter Abstände.
2.  `psi = exp(-D)`: Wie zuvor vervollständigt dies die Kernel-Berechnung.
3.  `return psi.T`: Die resultierende Matrix wird transponiert, um die Größe $n \times m$ zu haben. Dies ist eine Konvention, um mit der in den Vorhersageformeln in Referenztexten wie @Forr08a dargestellten Matrixalgebra übereinzustimmen.

### Modellkalibrierung durch Maximum-Likelihood-Schätzung (MLE) {#sec-mle}

Sobald die Struktur des Modells durch den Kernel definiert ist, müssen wir die optimalen Werte für seine Hyperparameter, nämlich $\vec{\theta}$ und $\vec{p}$, bestimmen. Der Code in @sec-example-de umgeht diesen Schritt, indem er `theta = 1.0` fest codiert, aber in jeder praktischen Anwendung müssen diese Parameter aus den Daten gelernt werden. Eine gebräuchliche Methode hierfür ist die **Maximum-Likelihood-Schätzung (MLE)**.

Die Kernidee der MLE besteht darin, die Frage zu beantworten: „Welche Werte der Hyperparameter machen bei unseren beobachteten Daten diese Daten am wahrscheinlichsten?“ Wir finden die Parameter, die die Wahrscheinlichkeit maximieren, die Daten beobachtet zu haben, die wir tatsächlich gesammelt haben.

#### Die Likelihood-Funktion

Unter der Annahme des Gauß-Prozesses wird die gemeinsame Wahrscheinlichkeit, den Antwortvektor $\vec{y}$ bei gegebenen Parametern zu beobachten, durch die multivariate normale Wahrscheinlichkeitsdichtefunktion beschrieben. Diese Funktion ist unsere Likelihood, $L$:
$$
L(\mu, \sigma^2, \vec{\theta}, \vec{p} | \vec{y}) = \frac{1}{(2\pi\sigma^2)^{n/2}|\Psi|^{1/2}} \exp\left[ - \frac{(\vec{y} - \vec{1}\mu)^T \vec{\Psi}^{-1}(\vec{y} - \vec{1}\mu) }{2 \sigma^2}\right].
$$ {#eq-likelihood-de}
Hier sind $\mu$ und $\sigma^2$ der globale Mittelwert und die Varianz des Prozesses, und $\Psi$ ist die Korrelationsmatrix, die implizit von $\vec{\theta}$ und $\vec{p}$ abhängt. Beachten Sie, dass $|\Psi|$ die Determinante (also ein reellwertiger Skalar) der Korrelationsmatrix ist.

#### Die konzentrierte Log-Likelihood

Die direkte Maximierung der Likelihood-Funktion (@eq-likelihood-de) ist schwierig. Aus Gründen der rechnerischen Stabilität und mathematischen Bequemlichkeit arbeiten wir stattdessen mit ihrem natürlichen Logarithmus, der Log-Likelihood (siehe Gleichung (2.29) in @Forr08a):
$$
\ln(L) = -\frac{n}{2}\ln(2\pi) - \frac{n}{2}\ln(\sigma^2) - \frac{1}{2}\ln|\Psi| - \frac{(\vec{y} - \vec{1}\mu)^T \Psi^{-1}(\vec{y} - \vec{1}\mu)}{2\sigma^2}. 
$$ {#eq-log-likelihood-de}


Eine wesentliche Vereinfachung ergibt sich, da wir für jedes gegebene $\vec{\theta}$ und $\vec{p}$ (und damit ein festes $\Psi$) die optimalen Werte für $\mu$ und $\sigma^2$ analytisch finden können, indem wir Ableitungen bilden und sie auf null setzen. Die MLE für den Mittelwert ist (siehe Gleichung (2.30) in @Forr08a):
$$
\hat{\mu} = \frac{\mathbf{1}^T \Psi^{-1} \vec{y}}{\mathbf{1}^T \Psi^{-1} \mathbf{1}}.
$$ {#eq-mu-hat-de}
Die Berechnung in @eq-mu-hat-de kann als Berechung eines verallgemeinerten gewichteten Durchschnitts der beobachteten Antworten interpretiert werden, wobei die Gewichtung die Korrelationsstruktur berücksichtigt.

Ein ähnlicher Ausdruck existiert für die optimale Varianz, $\hat{\sigma}^2$ (siehe Gleichung (2.31) in @Forr08a):
$$
\hat{\sigma}^2 = \frac{(\vec{y} - \mathbf{1}\hat{\mu})^T \Psi^{-1} (\vec{y} - \mathbf{1}\hat{\mu})}{n}.
$$ 

Indem wir diese analytischen Ausdrücke für $\hat{\mu}$ und $\hat{\sigma}^2$ wieder in die Log-Likelihood-Funktion (@eq-log-likelihood-de) einsetzen, erhalten wir die **konzentrierte Log-Likelihood-Funktion** (siehe Gleichung (2.32) in @Forr08a):
$$
\ln(L) \approx -\frac{n}{2}\ln(\hat{\sigma}^2) - \frac{1}{2}\ln|\Psi|.
$$
Diese vereinfachte Funktion hängt nun nur noch von den Hyperparametern $\vec{\theta}$ und $\vec{p}$ ab, die in $\Psi$ und $\hat{\sigma}^2$ eingebettet sind.

#### Numerische Optimierung

Wir können die optimalen $\vec{\theta}$ und $\vec{p}$ nicht analytisch bestimmen.
Die konzentrierte Log-Likelihood ist jedoch eine skalare Funktion, die relativ schnell zu berechnen ist.
Wir können daher einen numerischen Optimierungsalgorithmus – wie einen genetischen Algorithmus, Nelder-Mead oder Differential Evolution – verwenden, um den Hyperparameterraum zu durchsuchen und die Werte von $\vec{\theta}$ und $\vec{p}$ zu finden, die diese Funktion maximieren.
Dieser Suchprozess ist die „Trainings“- oder „Anpassungs“-Phase beim Aufbau eines Kriging-Modells.
Sobald die optimalen Hyperparameter gefunden sind, ist das Modell vollständig kalibriert und bereit, Vorhersagen zu treffen.

## Implementierung und Vorhersage

Nachdem der theoretische und mathematische Rahmen geschaffen wurde, konzentriert sich dieser Teil auf die praktische Anwendung des Kriging-Modells: wie man es zur Erstellung von Vorhersagen verwendet und welche wesentlichen numerischen Techniken sicherstellen, dass der Prozess sowohl effizient als auch robust ist.

### Der Kriging-Prädiktor: Generierung neuer Werte

Das ultimative Ziel beim Aufbau eines Kriging-Modells ist die Vorhersage des Funktionswerts an neuen, unbeobachteten Punkten. Die hierfür verwendete Formel ist als **Bester Linearer Unverzerrter Prädiktor (BLUP)** bekannt. Sie liefert den Mittelwert des posterioren Gauß-Prozesses am Vorhersageort, was unsere beste Schätzung des Funktionswerts ist.

Die Vorhersageformel lautet (siehe Gleichung (2.40) in @Forr08a):
$$
\hat{y}(\vec{x}) = \hat{\mu} + \vec{\psi}^T \Psi^{-1} (\vec{y} - \mathbf{1}\hat{\mu}),
$$ {#eq-blup-de}
wobei $\hat{y}(\vec{x})$ die Vorhersage an einem neuen Punkt $\vec{x}$ ist und alle anderen Terme wie im Glossar definiert sind.Im Folgenden beschreiben wir die einzelnen Komponenten dieser Gleichung und ihre Bedeutung:

1.  **$\hat{\mu}$**: Die Vorhersage beginnt mit dem geschätzten globalen Mittelwert des Prozesses. Dies ist unsere Basisvermutung, bevor wir den Einfluss der lokalen Datenpunkte berücksichtigen.
2.  **$(\vec{y} - \mathbf{1}\hat{\mu})$**: Dies ist der Vektor der Residuen. Er stellt die Differenz zwischen jedem beobachteten Datenpunkt und dem globalen Mittelwert dar. Dieser Vektor erfasst die spezifischen, lokalen Informationen, die unsere Trainingsdaten liefern.
3.  **$\Psi^{-1} (\vec{y} - \mathbf{1}\hat{\mu})$**: Dieser Term kann als ein Vektor von Gewichten betrachtet werden, nennen wir ihn $\vec{w}$. Indem wir die Residuen mit der Inversen der Korrelationsmatrix multiplizieren, berechnen wir einen Satz von Gewichten, der die Interkorrelationen zwischen den Trainingspunkten berücksichtigt. Dieser Schritt sagt im Wesentlichen: „Wie viel sollte jedes Residuum beitragen, wenn man bedenkt, dass die Datenpunkte selbst nicht unabhängig sind?“
4.  **$\vec{\psi}^T \vec{w}$**: Die endgültige Vorhersage wird durch eine gewichtete Summe dieser berechneten Gewichte angepasst. Die Gewichte für diese Summe sind die Korrelationen ($\vec{\psi}$) zwischen dem neuen Vorhersagepunkt $\vec{x}$ und jedem der Trainingspunkte. Im Wesentlichen besagt die Formel: „Beginne mit dem globalen Mittelwert und füge dann eine Korrektur basierend auf den beobachteten Residuen hinzu. Der Einfluss jedes Residuums wird durch die Korrelation des neuen Punktes mit dem entsprechenden Trainingspunkt bestimmt.“

**Code-Analyse (`mu_hat` und `f_predict`):** Der bereitgestellte Python-Code implementiert diesen Vorhersageprozess direkt.

```python
Psi = build_Psi(X, theta)
U = cholesky(Psi).T
one = np.ones(n).reshape(-1, 1)
mu_hat = (one.T @ solve(U, solve(U.T, y_train))) / \
         (one.T @ solve(U, solve(U.T, one)))
f_predict = mu_hat * np.ones(m).reshape(-1, 1) \
            + psi.T @ solve(U, solve(U.T, y_train - one * mu_hat))
```

*   `mu_hat = (one.T @ solve(U, solve(U.T, y_train))) / (one.T @ solve(U, solve(U.T, one)))`: Dies ist eine direkte und numerisch stabile Implementierung der Formel für $\hat{\mu}$, siehe @eq-mu-hat-de. Anstatt `Psi_inv` explizit zu berechnen, wird der auf Cholesky basierende Löser verwendet, der im nächsten Abschnitt detailliert wird. Der Ausdruck `solve(U, solve(U.T, y_train))` ist äquivalent zu `Psi_inv @ y_train`.
*   `f_predict = mu_hat *... + psi.T @ solve(U, solve(U.T, y_train - one * mu_hat))`: Diese Zeile ist eine direkte Übersetzung der BLUP-Formel aus @eq-blup-de. Sie berechnet die Residuen `y_train - one * mu_hat`, multipliziert sie mit `Psi_inv` unter Verwendung des Cholesky-Lösers und berechnet dann das Skalarprodukt mit `psi.T`, bevor das Ergebnis zum Basiswert `mu_hat` addiert wird.

### Numerische Best Practices und der Nugget-Effekt {#sec-numerical-best-practices}

Eine direkte Implementierung der Kriging-Gleichungen unter Verwendung der Standard-Matrixinversion kann sowohl langsam als auch numerisch anfällig sein. Professionelle Implementierungen stützen sich auf spezifische numerische Techniken, um Effizienz und Robustheit zu gewährleisten.

#### Effiziente Inversion mit Cholesky-Zerlegung

Die Berechnung der Matrixinversen $\Psi^{-1}$ ist der rechenintensivste Schritt sowohl im MLE-Prozess als auch bei der endgültigen Vorhersage. Eine direkte Inversion hat eine Rechenkomplexität von etwa $O(n^3)$. Wenn die Matrix schlecht konditioniert ist (fast singulär), kann die direkte Inversion außerdem zu großen numerischen Fehlern führen.

Ein überlegener Ansatz ist die Verwendung der **Cholesky-Zerlegung**. Diese Methode gilt für symmetrische, positiv definite Matrizen wie $\Psi$. Sie zerlegt $\Psi$ in das Produkt einer unteren Dreiecksmatrix $L$ und ihrer Transponierten $L^T$ (oder einer oberen Dreiecksmatrix $U$ und ihrer Transponierten $U^T$), sodass $\Psi = LL^T$. Diese Zerlegung ist schneller als die Inversion, mit einer Komplexität von ungefähr $O(n^3/3)$ [@Forr08a].

Sobald $\Psi$ zerlegt ist, wird das Lösen eines linearen Systems wie $\Psi \vec{w} = \vec{b}$ zu einem zweistufigen Prozess der Lösung zweier viel einfacherer Dreieckssysteme, ein Verfahren, das als Vorwärts- und Rückwärtssubstitution bekannt ist:

1.  Löse $L\vec{v} = \vec{b}$ nach $\vec{v}$.
2.  Löse $L^T\vec{w} = \vec{v}$ nach $\vec{w}$.

Genau das macht der Python-Code. Die Zeile `U = cholesky(Psi).T` führt die Zerlegung durch (NumPys `cholesky` gibt den unteren Dreiecksfaktor $L$ zurück, also wird er transponiert, um den oberen Dreiecksfaktor $U$ zu erhalten). Anschließend implementieren Ausdrücke wie `solve(U, solve(U.T,...))` die effiziente zweistufige Lösung, ohne jemals die vollständige Inverse von $\Psi$ zu bilden.

#### Der Nugget: Von numerischer Stabilität zur Rauschmodellierung

Die Cholesky-Zerlegung funktioniert nur, wenn die Matrix $\Psi$ streng positiv definit ist. Ein Problem tritt auf, wenn zwei Trainingspunkte $\vec{x}^{(i)}$ und $\vec{x}^{(j)}$ sehr nahe beieinander liegen. In diesem Fall wird ihre Korrelation nahe 1 sein, was die entsprechenden Zeilen und Spalten in $\Psi$ nahezu identisch macht. Dies führt dazu, dass die Matrix schlecht konditioniert oder fast singulär wird, was zum Scheitern der Cholesky-Zerlegung führen kann.

::: {.callout-note}
#### Naheliegende Punkte führen zu numerischen Problemen

Die Aussage, dass ein Problem auftritt, wenn zwei Trainingspunkte $\vec{x}^{(i)}$ und $\vec{x}^{(j)}$ sehr nahe beieinander liegen, und deren Korrelation dann nahe 1 ist, was die entsprechenden Zeilen und Spalten in der $\Psi$-Matrix nahezu identisch macht, ist ein wichtiger Aspekt der numerischen Stabilität und Modellierung beim Kriging.

Die $\Psi$-Matrix (sprich: Psi) ist die Korrelationsmatrix der Trainingsdaten mit sich selbst. Jedes Element $\Psi_{ij}$ quantifiziert die Korrelation zwischen zwei bekannten Trainingspunkten $\vec{x}^{(i)}$ und $\vec{x}^{(j)}$. Diese Korrelation wird durch die Kriging-Basisfunktion (oder den Gauß'schen Kernel) definiert:

$$
\psi(\vec{x}^{(i)}, \vec{x}^{(j)}) = \exp\left(-\sum_{l=1}^{k} \theta_l |x_l^{(i)} - x_l^{(j)}|^{p_l}\right)
$$

Hierbei ist $k$ die Anzahl der Eingabedimensionen, $\theta_l$ der Aktivitätshyperparameter für die $l$-te Dimension und $p_l$ der Glattheitsparameter. Die diagonalen Elemente $\Psi_{ii}$ sind immer 1, da die Korrelation eines Punktes mit sich selbst perfekt ist.

Das Kriging-Modell basiert auf dem Prinzip der Lokalität. Dieses besagt, dass Punkte, die im Eingaberaum nahe beieinander liegen, erwartungsgemäß ähnliche Ausgabewerte haben und somit hoch korreliert sind.

Wenn nun zwei Trainingspunkte $\vec{x}^{(i)}$ und $\vec{x}^{(j)}$ im Eingaberaum sehr nahe beieinander liegen, bedeutet dies, dass die Abstände $|x_l^{(i)} - x_l^{(j)}|$ für alle Dimensionen $l$ sehr klein sind. Infolgedessen wird der Term in der Summe $-\sum_{l=1}^{k} \theta_l |x_l^{(i)} - x_l^{(j)}|^{p_l}$ ebenfalls sehr klein (nahe Null) sein. Da $\exp(x)$ für $x \approx 0$ gegen 1 geht, wird die Korrelation $\psi(\vec{x}^{(i)}, \vec{x}^{(j)})$ einen Wert nahe 1 annehmen.

Lassen Sie uns diesen Sachverhalt an einem hypothetischen numerischen Beispiel in einem 3-dimensionalen Raum ($k=3$) verdeutlichen. Nehmen wir an, wir verwenden den Gauß'schen Kernel mit $p_l = 2$ für alle Dimensionen und die Aktivitätshyperparameter $\vec{\theta} = [1.0, 1.0, 1.0]$.

Betrachten wir drei Trainingspunkte:

*   $\vec{x}^{(1)} = [1.0, 2.0, 3.0]$
*   $\vec{x}^{(2)} = [1.0001, 2.0002, 3.0003]$ (Dieser Punkt ist extrem nahe an $\vec{x}^{(1)}$)
*   $\vec{x}^{(3)} = [5.0, 6.0, 7.0]$ (Dieser Punkt ist weit entfernt von $\vec{x}^{(1)}$ und $\vec{x}^{(2)}$)

Wir berechnen die Korrelationen:

1.  **Korrelation zwischen $\vec{x}^{(1)}$ und $\vec{x}^{(2)}$:**
    Die quadrierten Abstände sind:
    $$
    |x_1^{(1)} - x_1^{(2)}|^2 = |1.0 - 1.0001|^2 = (-0.0001)^2 = 0.00000001
    $$
    $$
    |x_2^{(1)} - x_2^{(2)}|^2 = |2.0 - 2.0002|^2 = (-0.0002)^2 = 0.00000004
    $$
    $$
    |x_3^{(1)} - x_3^{(2)}|^2 = |3.0 - 3.0003|^2 = (-0.0003)^2 = 0.00000009
    $$

    Die Summe der gewichteten quadrierten Abstände (da $\theta_l=1.0$):
    $1.0 \cdot 0.00000001 + 1.0 \cdot 0.00000004 + 1.0 \cdot 0.00000009 = 0.00000014$

    Die Korrelation ist:
    $\psi(\vec{x}^{(1)}, \vec{x}^{(2)}) = \exp(-0.00000014) \approx \mathbf{0.99999986}$

    **Wie man sieht, ist die Korrelation extrem nahe 1.**

2.  **Korrelation zwischen $\vec{x}^{(1)}$ und $\vec{x}^{(3)}$:**
    Die quadrierten Abstände sind:
    $$
    |x_1^{(1)} - x_1^{(3)}|^2 = |1.0 - 5.0|^2 = (-4.0)^2 = 16.0
    $$
    $$|x_2^{(1)} - x_2^{(3)}|^2 = |2.0 - 6.0|^2 = (-4.0)^2 = 16.0
    $$
    $$
    |x_3^{(1)} - x_3^{(3)}|^2 = |3.0 - 7.0|^2 = (-4.0)^2 = 16.0
    $$

    Die Summe der gewichteten quadrierten Abstände:
    $1.0 \cdot 16.0 + 1.0 \cdot 16.0 + 1.0 \cdot 16.0 = 48.0$

    Die Korrelation ist:
    $\psi(\vec{x}^{(1)}, \vec{x}^{(3)}) = \exp(-48.0) \approx \mathbf{1.39 \times 10^{-21}}$

    **Diese Korrelation ist praktisch Null, was zeigt, dass weit entfernte Punkte unkorreliert sind.**

Wenn wir eine $\Psi$-Matrix für diese drei Punkte aufstellen (und die Diagonalelemente $\Psi_{ii}=1$ sind, plus ein winziges `eps` für numerische Stabilität):

$$
\Psi =
\begin{pmatrix}
  \Psi_{11} & \Psi_{12} & \Psi_{13} \\
  \Psi_{21} & \Psi_{22} & \Psi_{23} \\
  \Psi_{31} & \Psi_{32} & \Psi_{33}
\end{pmatrix}
$$

Setzen wir die berechneten Werte ein (unter Vernachlässigung des `eps`-Terms für Klarheit in diesem Schritt):

$$
\Psi \approx
\begin{pmatrix}
  1.0 & 0.99999986 & 1.39 \times 10^{-21} \\
  0.99999986 & 1.0 & 1.39 \times 10^{-21} \\
  1.39 \times 10^{-21} & 1.39 \times 10^{-21} & 1.0
\end{pmatrix}
$$

Man sieht deutlich, dass die **erste und zweite Zeile (und Spalte)** der Matrix **nahezu identisch** sind, da $\Psi_{11} \approx \Psi_{21}$ und $\Psi_{12} \approx \Psi_{22}$ und $\Psi_{13} \approx \Psi_{23}$ (wobei letztere beide nahezu Null sind).

Wenn Zeilen oder Spalten einer Matrix nahezu identisch sind, bedeutet dies, dass die Matrix **schlecht konditioniert** oder **fast singulär** ist. Eine schlecht konditionierte Matrix ist für numerische Operationen, insbesondere für die Inversion ($\Psi^{-1}$), problematisch. Die Inversion einer solchen Matrix ist ein rechenintensiver und numerisch anfälliger Schritt sowohl im Maximum-Likelihood-Schätzprozess zur Modellkalibrierung als auch bei der finalen Vorhersage. Dies kann zum **Scheitern der Cholesky-Zerlegung** führen, einer häufig verwendeten Methode zur effizienten und stabilen Matrixinversion im Kriging.

:::


Dieses Problem wird gelöst, indem ein kleiner positiver Wert zur Diagonale der Korrelationsmatrix addiert wird: $\Psi_{new} = \Psi + \lambda I$, wobei $I$ die Identitätsmatrix ist. Diese kleine Addition, oft als **Nugget** bezeichnet, stellt sicher, dass die Matrix gut konditioniert und invertierbar bleibt. Im bereitgestellten Code dient die Variable `eps` diesem Zweck.

Obwohl es als numerischer „Hack“ beginnen mag, hat dieser Nugget-Term eine tiefgreifende und starke statistische Interpretation: Er modelliert Rauschen in den Daten. Dies führt zu zwei unterschiedlichen Arten von Kriging-Modellen:

1.  **Interpolierendes Kriging ($\lambda \approx 0$):** Wenn der Nugget null oder sehr klein ist (wie `eps` im Code), wird das Modell gezwungen, exakt durch jeden Trainingsdatenpunkt zu verlaufen. Dies ist für deterministische Computerexperimente geeignet, bei denen die Ausgabe rauschfrei ist.
2.  **Regressives Kriging ($\lambda > 0$):** Wenn bekannt ist, dass die Daten verrauscht sind (z. B. aus physikalischen Experimenten oder stochastischen Simulationen), würde das Erzwingen der Interpolation jedes Punktes dazu führen, dass das Modell das Rauschen anpasst, was zu einer übermäßig komplexen und „zappeligen“ Oberfläche führt, die schlecht generalisiert. Durch Hinzufügen eines größeren Nugget-Terms $\lambda$ zur Diagonale teilen wir dem Modell explizit mit, dass es Varianz (Rauschen) in den Beobachtungen gibt. Das Modell ist nicht mehr verpflichtet, exakt durch die Datenpunkte zu verlaufen. Stattdessen wird es eine glattere Regressionskurve erstellen, die den zugrunde liegenden Trend erfasst und gleichzeitig das Rauschen herausfiltert. Die Größe von $\lambda$ kann als weiterer zu optimierender Hyperparameter behandelt werden, der die Varianz des Rauschens darstellt.

Dieselbe mathematische Operation – das Hinzufügen eines Wertes zur Diagonale von $\Psi$ – dient somit einem doppelten Zweck. Ein winziges, festes `eps` ist eine pragmatische Lösung für die numerische Stabilität. Ein größeres, potenziell optimiertes $\lambda$ ist ein formaler statistischer Parameter, der das Verhalten des Modells grundlegend von einem exakten Interpolator zu einem rauschfilternden Regressor ändert.

## Eine vollständige exemplarische Vorgehensweise: Kriging der Sinusfunktion {#sec-example-de}

Dieser letzte Teil fasst die gesamte vorangegangene Theorie zusammen, indem er sie auf das bereitgestellte Python-Codebeispiel aus dem [Hyperparameter Tuning Cookbook](https://sequential-parameter-optimization.github.io/Hyperparameter-Tuning-Cookbook/006_num_gp.html#sec-kriging-example-006) anwendet. Wir werden das Skript Schritt für Schritt durchgehen und die Eingaben, Prozesse und Ausgaben in jeder Phase interpretieren, um eine konkrete Veranschaulichung des Kriging in Aktion zu geben.

Zunächst zeigen wir den gesamten Code, gefolgt von einer detaillierten Erklärung jedes Schrittes.



```{python}
#| eval: true
import numpy as np
import matplotlib.pyplot as plt
from numpy import (array, zeros, power, ones, exp, multiply, eye,
                  linspace, spacing, sqrt, arange, append, ravel)
from numpy.linalg import cholesky, solve
from scipy.spatial.distance import squareform, pdist, cdist

# --- 1. Kriging Basis Functions (Defining the Correlation) ---
# The core of Kriging uses a specialized basis function for correlation:
# psi(x^(i), x) = exp(- sum_{j=1}^k theta_j |x_j^(i) - x_j|^p_j)
# For this 1D example (k=1), and with p_j=2
# (squared Euclidean distance implicit from pdist usage)
# and theta_j = theta (a single value), it simplifies.

def build_Psi(X, theta, eps=sqrt(spacing(1))):
    """
    Computes the correlation matrix Psi based on pairwise squared Euclidean distances
    between input locations, scaled by theta.
    Adds a small epsilon to the diagonal for numerical stability (nugget effect).
    """
    # Calculate pairwise squared Euclidean distances (D) between points in X
    D = squareform(pdist(X, metric='sqeuclidean', out=None, w=theta))
    # Compute Psi = exp(-D)
    Psi = exp(-D)
    # Add a small value to the diagonal for numerical stability (nugget)
    # This is often done in Kriging implementations, though a regression method
    # with a 'nugget' parameter (Lambda) is explicitly mentioned for noisy data later.
    # The source code snippet for build_Psi explicitly includes
    # `multiply(eye(X.shape), eps)`.
    # FIX: Use X.shape to get the number of rows for the identity matrix
    Psi += multiply(eye(X.shape[0]), eps) # Corrected line
    return Psi

def build_psi(X_train, x_predict, theta):
    """
    Computes the correlation vector (or matrix) psi between new prediction locations
    and training data locations.
    """
    # Calculate pairwise squared Euclidean distances (D) between prediction points
    # (x_predict)
    # and training points (X_train).
    # `cdist` computes distances between each pair of the two collections of inputs.
    D = cdist(x_predict, X_train, metric='sqeuclidean', out=None, w=theta)
    # Compute psi = exp(-D)
    psi = exp(-D)
    return psi.T
    # Return transpose to be consistent with literature (n x m or n x 1)
```

```{python}
#| eval: true
# --- 2. Data Points for the Sinusoid Function Example ---
# The example uses a 1D sinusoid measured at eight equally spaced x-locations.
n = 8 # Number of sample locations
X_train = np.linspace(0, 2 * np.pi, n, endpoint=False).reshape(-1, 1)
y_train = np.sin(X_train) # Corresponding y-values (sine of x)
print("--- Training Data (X_train, y_train) ---")
print("x values:\n", np.round(X_train, 2))
print("y values:\n", np.round(y_train, 2))
print("-" * 40)
```

```{python}
#| eval: true
# Visualize the data points
plt.figure(figsize=(8, 5))
plt.plot(X_train, y_train, "bo", label=f"Measurements ({n} points)")
plt.title(f"Sin(x) evaluated at {n} points")
plt.xlabel("x")
plt.ylabel("sin(x)")
plt.grid(True)
plt.legend()
plt.show()
```

```{python}
#| eval: true
# --- 3. Calculating the Correlation Matrix (Psi) ---
# Psi is based on pairwise squared distances between input locations.
# theta is set to 1.0 for this 1D example.
theta = np.array([1.0])
Psi = build_Psi(X_train, theta)
print("\n--- Computed Correlation Matrix (Psi) ---")
print("Dimensions of Psi:", Psi.shape) # Should be (8, 8)
print("First 5x5 block of Psi:\n", np.round(Psi[:5,:5], 2))
print("-" * 40)
```

```{python}
#| eval: true
# --- 4. Selecting New Locations (for Prediction) ---
# We want to predict at m = 100 new locations in the interval [0, 2*pi].
m = 100 # Number of new locations
x_predict = np.linspace(0, 2 * np.pi, m, endpoint=True).reshape(-1, 1)
print("\n--- New Locations for Prediction (x_predict) ---")
print(f"Number of prediction points: {m}")
print("First 5 prediction points:\n", np.round(x_predict[:5], 2).flatten())
print("-" * 40)
```

```{python}
#| eval: true
# --- 5. Computing the psi Vector ---
# This vector contains correlations between each of the n observed data points
# and each of the m new prediction locations.
psi = build_psi(X_train, x_predict, theta)
print("\n--- Computed Prediction Correlation Matrix (psi) ---")
print("Dimensions of psi:", psi.shape) # Should be (8, 100)
print("First 5x5 block of psi:\n", np.round(psi[:5,:5], 2))
print("-" * 40)
```

```{python}
#| eval: true
# --- 6. Predicting at New Locations (Kriging Prediction) ---
# The Maximum Likelihood Estimate (MLE) for y_hat is calculated using the formula:
# y_hat(x) = mu_hat + psi.T @ Psi_inv @ (y - 1 * mu_hat)
# Matrix inversion is efficiently performed using Cholesky factorization.
# Step 6a: Cholesky decomposition of Psi
U = cholesky(Psi).T
# Note: `cholesky` in numpy returns lower triangular L,
# we need U (upper) so transpose L.
# Step 6b: Calculate mu_hat (estimated mean)
one = np.ones(n).reshape(-1, 1) # Vector of ones
mu_hat = (one.T @ solve(U, solve(U.T, y_train))) \
         / (one.T @ solve(U, solve(U.T, one)))
mu_hat = mu_hat.item() # Extract scalar value
print("\n--- Kriging Prediction Calculation ---")
print(f"Estimated mean (mu_hat): {np.round(mu_hat, 4)}")
```

```{python}
#| eval: true
# Step 6c: Calculate predictions f (y_hat) at new locations
# f = mu_hat * ones(m) + psi.T @ Psi_inv @ (y - one * mu_hat)
f_predict = mu_hat * np.ones(m).reshape(-1, 1) \
            + psi.T @ solve(U, solve(U.T, y_train - one * mu_hat))
print(f"Dimensions of predicted values (f_predict): {f_predict.shape}")
# Should be (100, 1)
print("First 5 predicted f values:\n", np.round(f_predict[:5], 2).flatten())
print("-" * 40)
```

```{python}
#| eval: true
# --- 7. Visualization ---
# Plot the original sinusoid function, the measured points, and the Kriging predictions.
plt.figure(figsize=(10, 6))
plt.plot(x_predict, f_predict, color="orange", label="Kriging Prediction")
plt.plot(x_predict, np.sin(x_predict), color="grey", linestyle='--', \
        label="True Sinusoid Function")
plt.plot(X_train, y_train, "bo", markersize=8, label="Measurements")
plt.title(f"Kriging prediction of sin(x) with {n} points. (theta: {theta})")
plt.xlabel("x")
plt.ylabel("y")
plt.legend(loc='upper right')
plt.grid(True)
plt.show()
```



### Schritt-für-Schritt-Codeausführung und Interpretation

Das Beispiel zielt darauf ab, die Funktion $y = \sin(x)$ unter Verwendung einer kleinen Anzahl von Stichprobenpunkten zu modellieren. Dies ist ein klassisches „Spielzeugproblem“, das nützlich ist, um zu visualisieren, wie sich das Modell verhält.


#### Schritt 1: Datengenerierung

```python
n = 8
X_train = np.linspace(0, 2 * np.pi, n, endpoint=False).reshape(-1, 1)
y_train = np.sin(X_train)
```

Hier generiert das Skript die Trainingsdaten.

*   `n = 8`: Wir entscheiden uns, die Funktion an acht verschiedenen Stellen abzutasten.
*   `X_train`: Dies erstellt ein Array von acht  gleichmäßig verteilten Punkten im Intervall [0, $2\pi$], die als Trainingspunkte dienen.
*   `y_train = np.sin(X_train)`: Dies berechnet die Sinuswerte an den Trainingspunkten. Das Ergebnis ist ein $8 \times 1$ Vektor, der die beobachteten Antworten darstellt.

#### Schritt 2: Definition der Korrelationsmatrix ($\Psi$)
```python
theta = np.array([1.0])
Psi = build_Psi(X_train, theta)
```

Dieser Schritt berechnet die $8 \times 8$ Korrelationsmatrix $\Psi$ für die Trainingsdaten.

*   `theta = np.array([1.0])`: Der Aktivitätshyperparameter $\theta$ wird auf 1.0 gesetzt. In einer realen Anwendung würde dieser Wert durch MLE gefunden, aber hier wird er zur Vereinfachung festgesetzt.
*   `Psi = build_Psi(X_train, theta)`: Die Funktion `build_Psi` wird aufgerufen. Sie berechnet den gewichteten quadrierten Abstand zwischen jedem Paar von Punkten in `X_train` und wendet dann den exponentiellen Kernel an. Die resultierende `Psi`-Matrix quantifiziert die angenommene Korrelation zwischen all unseren bekannten Datenpunkten. Die diagonalen Elemente sind 1 (plus ein winziges `eps`), und die außerdiagonalen Werte nehmen ab, wenn der Abstand zwischen den entsprechenden Punkten zunimmt.

#### Schritt 3: Definition der Vorhersagepunkte

```python
m = 100
x_predict = np.linspace(0, 2 * np.pi, m, endpoint=True).reshape(-1, 1)
```

Wir definieren nun die Orte, an denen wir neue Vorhersagen machen wollen. Wir erstellen ein dichtes Gitter von `m = 100` Punkten, das das gesamte Intervall [0, $2\pi$] abdeckt. Dies sind die Punkte, an denen wir unser Surrogatmodell auswerten werden, um eine glatte Kurve zu erzeugen.

#### Schritt 4: Berechnung der Vorhersagekorrelation ($\vec{\psi}$)

```python
psi = build_psi(X_train, x_predict, theta)
```

Dieser Schritt berechnet die $8 \times 100$ Korrelationsmatrix $\vec{\psi}$. Die Funktion `build_psi` berechnet die Korrelation zwischen jedem der acht Trainingspunkte und jedem der 100 Vorhersagepunkte. Jede Spalte der resultierenden `psi`-Matrix entspricht einem Vorhersagepunkt und enthält seine acht Korrelationswerte mit dem Trainingssatz. Diese Matrix verbindet die neuen, unbekannten Orte mit unserer bestehenden Wissensbasis.

#### Schritt 5: Berechnung der Vorhersage

```python
U = cholesky(Psi).T
one = np.ones(n).reshape(-1, 1)
mu_hat = (one.T @ solve(U, solve(U.T, y_train))) /
         (one.T @ solve(U, solve(U.T, one)))
f_predict = mu_hat * np.ones(m).reshape(-1, 1) +
            psi.T @ solve(U, solve(U.T, y_train - one * mu_hat))
```

Dies ist der entscheidende Schritt, in dem die tatsächlichen Vorhersagen gemacht werden.

1.  `U = cholesky(Psi).T`: Die numerisch entscheidende Cholesky-Zerlegung von $\Psi$ wird durchgeführt.
2.  `mu_hat =...`: Die Maximum-Likelihood-Schätzung für den globalen Mittelwert $\mu$ wird unter Verwendung des numerisch stabilen Cholesky-Lösers berechnet. Für diese spezifischen symmetrischen Daten wird `mu_hat` nahe null sein.
3.  `f_predict =...`: Die BLUP-Formel wird implementiert. Sie berechnet die Residuen `(y_train - one * mu_hat)`, findet die gewichteten Residuen unter Verwendung des Cholesky-Lösers (`solve(U, solve(U.T,...))`) und berechnet dann die endgültige Vorhersage durch eine gewichtete Summe basierend auf den Vorhersagekorrelationen `psi.T`. Das Ergebnis, `f_predict`, ist ein $100 \times 1$ Vektor, der die vorhergesagten Sinuswerte an jedem der `x_predict`-Orte enthält.

#### Schritt 6: Visualisierung

Der letzte Codeblock stellt die Ergebnisse grafisch dar.

*   **Messungen (Blaue Punkte):** Die ursprünglichen 8 Datenpunkte werden dargestellt.
*   **Wahre Sinusfunktion (Graue gestrichelte Linie):** Die tatsächliche $\sin(x)$-Funktion wird als Referenz dargestellt.
*   **Kriging-Vorhersage (Orange Linie):** Die vorhergesagten Werte `f_predict` werden gegen `x_predict` aufgetragen.

Die Grafik zeigt die Schlüsseleigenschaften des Kriging-Modells. Die orangefarbene Linie verläuft *exakt* durch jeden der blauen Punkte und demonstriert damit ihre **interpolierende** Natur (da `eps` sehr klein war). Wichtiger noch, zwischen den Stichprobenpunkten liefert die Vorhersage eine glatte und bemerkenswert genaue Annäherung an die wahre zugrunde liegende Sinuskurve, obwohl das Modell in diesen Bereichen keine anderen Informationen über die Funktion hatte als die acht Stichprobenpunkte und die angenommene Korrelationsstruktur.

### Fazit und Ausblick

Wir begannen damit, das Kriging als eine anspruchsvolle Form der Modellierung mit radialen Basisfunktionen einzuordnen und seine Kernphilosophie zu übernehmen, deterministische Funktionen als Realisierungen eines stochastischen Prozesses zu behandeln. Anschließend haben wir seine Architektur zerlegt: den leistungsstarken Korrelationskernel mit seinen Aktivitäts- ($\theta$) und Glattheits- ($p$) Hyperparametern, die Konstruktion der Korrelationsmatrizen ($\Psi$ und $\vec{\psi}$) und den Ansatz der Maximum-Likelihood-Schätzung zur Modellkalibrierung. Schließlich haben wir die numerischen Best Practices wie die Cholesky-Zerlegung untersucht und die doppelte Rolle des Nugget-Terms für die numerische Stabilität und die statistische Rauschmodellierung besprochen. Die schrittweise exemplarische Vorgehensweise des Sinusbeispiels lieferte eine konkrete Demonstration dieser Konzepte und zeigte, wie eine kleine Menge von Datenpunkten verwendet werden kann, um ein genaues, interpolierendes Modell einer unbekannten Funktion zu erzeugen.

Für den angehenden Praktiker ist dies nur der Anfang. Die Welt des Kriging und der Gauß-Prozess-Regression ist reich an fortgeschrittenen Techniken, die auf diesen Grundlagen aufbauen.

*   **Fehlerschätzungen und sequentielles Design:** Ein wesentliches Merkmal, das im Beispielcode nicht untersucht wurde, ist, dass das Kriging nicht nur eine mittlere Vorhersage, sondern auch eine **Varianz** an jedem Punkt liefert, die die Unsicherheit des Modells quantifiziert. Diese Varianz ist in Regionen weit entfernt von Datenpunkten hoch und in deren Nähe niedrig. Diese Fehlerschätzung ist die Grundlage für **aktives Lernen** oder **sequentielles Design**, bei dem Infill-Kriterien wie die **erwartete Verbesserung (EI)** verwendet werden, um den nächsten zu beprobenden Punkt intelligent auszuwählen, wobei ein Gleichgewicht zwischen der Notwendigkeit, vielversprechende Regionen auszunutzen (niedriger vorhergesagter Wert) und der Notwendigkeit, unsichere Regionen zu erkunden (hohe Varianz), hergestellt wird.

*   **Gradienten-erweitertes Kriging:** In vielen modernen Simulationsumgebungen ist es möglich, nicht nur den Funktionswert, sondern auch seine Gradienten (Ableitungen) zu geringen zusätzlichen Kosten zu erhalten. Das **Gradienten-erweiterte Kriging** integriert diese Gradienteninformationen in das Modell, was seine Genauigkeit drastisch verbessert und den Aufbau hochpräziser Modelle mit sehr wenigen Stichprobenpunkten ermöglicht.

*   **Multi-Fidelity-Modellierung (Co-Kriging):** Oft haben Ingenieure Zugang zu mehreren Informationsquellen mit unterschiedlicher Genauigkeit und Kosten – zum Beispiel ein schnelles, aber ungenaues analytisches Modell und eine langsame, aber hochpräzise CFD-Simulation. **Co-Kriging** ist ein Rahmenwerk, das diese unterschiedlichen Datenquellen verschmilzt, indem es die reichlich vorhandenen billigen Daten verwendet, um einen Basistrend zu etablieren, und die spärlichen teuren Daten, um ihn zu korrigieren, was zu einem Modell führt, das genauer ist als eines, das nur aus einer der Datenquellen allein erstellt wurde.


## Zusatzmaterialien

:::{.callout-note}
#### Interaktive Webseite

* Eine interaktive Webseite zum Thema **Kriging** ist hier zu finden: [Kriging Interaktiv](https://advm1.gm.fh-koeln.de/~bartz/bart21i/de_kriging_interactive.html).

* Eine interaktive Webseite zum Thema **MLE** ist hier zu finden: [MLE Interaktiv](https://advm1.gm.fh-koeln.de/~bartz/bart21i/de_mle_interactive.html).
:::

:::{.callout-note}
#### Audiomaterial

* Ein Audio zum Thema *Kriging** ist hier zu finden: [Cholesky Audio](https://advm1.gm.fh-koeln.de/~bartz/bart21i/audio/numerischeMatheKriging.m4a).
* Ein Audio zum Thema *Stochastische Prozesses** ist hier zu finden: [Cholesky Audio](https://advm1.gm.fh-koeln.de/~bartz/bart21i/audio/stochastischeProzesse.m4a).


:::

:::{.callout-note}
#### Jupyter-Notebook

* Das Jupyter-Notebook für dieses Lernmodul ist auf GitHub im [Hyperparameter-Tuning-Cookbook Repository](https://github.com/sequential-parameter-optimization/Hyperparameter-Tuning-Cookbook/blob/main/de_kriging.ipynb) verfügbar.

:::

