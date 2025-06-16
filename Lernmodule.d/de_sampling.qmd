---
lang: de
eval: true
---

# Lernmodul: Versuchspläne (Sampling-Pläne) für Computerexperimente

Dieses Dokument beschreibt die grundlegenden Ideen von Versuchsplänen, die in der Konzeption und Analyse von Computerexperimenten verwendet werden. Es stützt sich auf Kapitel 1 aus Forresters Buch "Engineering Design via Surrogate Modelling" [@Forr08a] und Kapitel 4 des "Hyperparameter Tuning Cookbook" [@bart23iArXiv].

## Einführung

:::{#def-sampling-plan}
#### Sampling-Plan
m Kontext von Computerexperimenten bezieht sich der Begriff **Sampling-Plan** auf die Menge der Eingabewerte, beispielsweise $X$, an denen der Computercode evaluiert wird.
:::

Das übergeordnete Ziel eines Sampling-Plans ist es, den Eingaberaum effizient zu erkunden, um das Verhalten eines Computercodes zu verstehen und ein Surrogatmodell zu erstellen, das das Verhalten des Codes genau abbildet. Traditionell wurde die Response Surface Methodology (RSM) zur Gestaltung von Sampling-Plänen für Computerexperimente verwendet, wobei Punkte mittels eines rechteckigen Rasters oder eines faktoriellen Designs generiert wurden.

In jüngerer Zeit hat sich jedoch Design and Analysis of Computer Experiments (DACE) als flexiblerer und leistungsfähigerer Ansatz für die Gestaltung von Sampling-Plänen etabliert. Der Prozess umfasst:

*   Die Abtastung diskreter Beobachtungen.
*   Die Verwendung dieser Abtastungen zur Konstruktion einer Approximation $\hat{f}$.
*   Die Sicherstellung, dass das Surrogatmodell wohlformuliert ist, d.h., es ist mathematisch gültig und kann Vorhersagen effektiv verallgemeinern.

Ein Sampling-Plan
$$
X = \left\{ x^{(i)} \in D | i = 1, \ldots, n \right\}
$$
bestimmt die räumliche Anordnung der Beobachtungen.

## Der "Fluch der Dimensionen"

### Das Volumen in hochdimensionalen Räume

Ein wesentliches Problem bei der Gestaltung von Sampling-Plänen, insbesondere in hochdimensionalen Räumen, ist der sogenannte **"Fluch der Dimensionen"**. Dieses Phänomen beschreibt, wie das Volumen des Eingaberaums exponentiell mit der Anzahl der Dimensionen zunimmt, was es schwierig macht, den Raum ausreichend abzutasten. Wenn eine bestimmte Vorhersagegenauigkeit durch die Abtastung eines eindimensionalen Raums an $n$ Stellen erreicht wird, sind $n^k$ Beobachtungen erforderlich, um die gleiche Abtastdichte in einem $k$-dimensionalen Raum zu erzielen.

Um diesem Problem zu begegnen, kann man entweder die Bereiche der Variablen begrenzen, sodass die zu modellierende Form ausreichend einfach ist, um aus sehr spärlichen Daten approximiert zu werden, oder viele Designwerte auf sinnvollen Werten festlegen und jeweils nur mit wenigen Variablen arbeiten.

### Volumen einer Kugel in hohen Dimensionen

::: {#rem-curse-of-dimensionality}
#### Volumen einer Kugel in hohen Dimensionen
Unsere Intuition legt nahe, dass das Volumen einer Kugel mit steigender Dimension immer weiter zunimmt. Das Gegenteil ist der Fall: Das Volumen einer Einheitskugel (Radius = 1) wächst nur bis zur fünften Dimension und beginnt dann stetig und unaufhaltsam gegen null zu schrumpfen. Der Grund: In hohen Dimensionen befindet sich der Großteil des Volumens eines umschließenden Hyperwürfels in dessen „Ecken“, sodass für die eingeschriebene Kugel kaum noch Raum bleibt.
:::

@rem-curse-of-dimensionality beschreibt eines der bekanntesten Paradoxa der hochdimensionalen Geometrie. Es widerspricht unserer in drei Dimensionen geschulten Intuition, lässt sich aber sowohl mathematisch als auch geometrisch gut erklären.

@rem-curse-of-dimensionality lässt sich auf zwei Arten veranschaulichen: den mathematischen Grund, der sich aus der Volumenformel ergibt, und den geometrischen Grund, der das Verhältnis zwischen einer Hypersphäre und einem sie umschließenden Hyperwürfel betrifft.

### Der mathematische Grund: Ein Wettlauf zweier Funktionen

Unsere Intuition, dass das Volumen mit der Dimension wachsen sollte, ist anfangs korrekt. Das Volumen einer Einheitskugel (Radius R=1) nimmt von Dimension 1 (eine Strecke der Länge 2) über Dimension 2 (ein Kreis mit Fläche π) bis Dimension 3 (eine Kugel mit Volumen  $4\pi/3$ tatsächlich zu. Dieser Trend setzt sich bis zur fünften Dimension fort.   

Der Grund für den anschließenden Rückgang liegt in der Formel für das Volumen einer n-dimensionalen Kugel:
$$
V_(R) = \frac{\pi^{n/2} R^n}{\Gamma(\frac{n}{2} + 1)},
$$

Hier kämpfen zwei Terme gegeneinander: 

* Der Zähler ($\pi^{n/2} R^n$): Dieser Term wächst mit zunehmender Dimension $n$ exponentiell an. Er ist verantwortlich für unsere Intuition und das anfängliche Wachstum des Volumens.
* Der Nenner ($\Gamma(\frac{n}{2} + 1)$): DieserTermenth enthält die Gammafunktion($\Gamma$), eine Verallgemeinerung der Fakultätsfunktion (z. B. $\Gamma(x+1) = x \cdot \Gamma(x)$). Die Gammafunktion wächst noch schneller als der Zähler. Bei niedrigen Dimensionen dominiert das Wachstum des Zählers, und das Volumen der Hypersphäre nimmt zu. Ab Dimension 5 überholt jedoch das extrem schnelle Wachstum der Gammafunktion im Nenner das Wachstum des Zählers, was dazu führt, dass das Gesamtvolumen wieder abnimmt und für $n \to \infty$ gegen null strebt. Das Maximum des Volumens einer Einheitskugel liegt bei etwa 5,26 Dimensionen.   

### Der geometrische Grund: Die Dominanz der Ecken 

Eine anschaulichere Erklärung liefert die Betrachtung eines Hyperwürfels, der die Hypersphäre umschließt. Stellen Sie sich eine Einheitskugel (Radius 1) vor, die perfekt in einen Hyperwürfel mit der Kantenlänge 2 passt.

* In 2 Dimensionen (Kreis im Quadrat): Der Kreis füllt einen erheblichen Teil der Fläche des Quadrats aus (ca. 78,5 %). Die Ecken des Quadrats machen nur einen kleinen Teil der Gesamtfläche aus.   
* In 3 Dimensionen (Kugel im Würfel): Die Kugel füllt bereits einen kleineren Anteil des Würfelvolumens aus (ca. 52,3 %). Die Ecken des Würfels beanspruchen relativ mehr Raum.   

Mit steigender Dimension $n$ geschehen zwei Dinge, die diesen Effekt dramatisch verstärken:

* Die Anzahl der Ecken explodiert: Ein n-dimensionaler Würfel hat $2^n$ Ecken. Ein 10-dimensionaler Würfel hat bereits 1024 Ecken, ein 20-dimensionaler über eine Million. Die Ecken entfernen sich vom Zentrum: Der Abstand vom Zentrum eines Würfels zu den Mittelpunkten seiner Seitenflächen (wo die eingeschriebene Kugel ihn berührt) bleibt konstant (hier: 1). Der Abstand zu den Ecken wächst jedoch mit der Dimension (er beträgt $\sqrt{n}$ für einen n-dimensionalen Würfel, z.B. $\sqrt{10} \approx 3.16$ für einen 10-dimensionalen Würfel und $\sqrt{20} \approx 4.47$ für einen 20-dimensionalen Würfel).
Das bedeutet, dass in hohen Dimensionen fast das gesamte Volumen des Hyperwürfels in seine zahlreichen, weit vom Zentrum entfernten Ecken gedrängt wird. Die eingeschriebene Hypersphäre, die in der Mitte "gefangen" ist, kann diese Ecken nicht erreichen. Ihr Volumen wird im Vergleich zum Volumen des Hyperwürfels, der "nur noch aus Ecken besteht", verschwindend gering.

Zusammenfassend lässt sich @rem-curse-of-dimensionality wie folgt erklärn:
Einerseits gilt mathematisch, dass die Gammafunktion im Nenner der Volumenformel schneller wächst als der Potenzterm im Zähler. Andererseits lässt sich geometrisch zeigen, dass in hohen Dimensionen das Volumen eines Würfels fast vollständig in seinen Ecken konzentriert ist, wodurch für die eingeschriebene Kugel im Zentrum kaum noch Volumen übrig bleibt.


## Entwurf von Vorab-Experimenten (Screening)

Bevor die Zielfunktion $f$ modelliert wird, ist es entscheidend, die Anzahl der Designvariablen ($x_1, x_2, \dots, x_k$) zu minimieren. Dieser Prozess wird als **Screening** bezeichnet und zielt darauf ab, die Dimensionalität zu reduzieren, ohne die Analyse zu beeinträchtigen. Der Morris-Algorithmus ist hierfür eine beliebte Methode, da er lediglich annimmt, dass die Zielfunktion deterministisch ist.

Morris' Methode zielt darauf ab, die Parameter der Verteilung von Elementareffekten zu schätzen, die mit jeder Variablen verbunden sind. Ein großes Maß an zentraler Tendenz (Mittelwert) deutet auf eine Variable mit wichtigem Einfluss auf die Zielfunktion hin, während ein großes Maß an Streuung auf Wechselwirkungen und/oder Nichtlinearität der Funktion in Bezug auf die Variable hinweist.


::: {#def-elementareffekt}
#### Elementareffekt

Für einen gegebenen Basiswert $x \in D$ bezeichne $d_i(x)$ den Elementareffekt von $x_i$, wobei:
$$
d_i(x) = \frac{f(x_1, \dots, x_i + \Delta, \dots, x_k) - f(x_1, \dots, x_i - \Delta, \dots, x_k)}{2\Delta}, \quad i = 1, \dots, k,
$$
wobei $\Delta$ die Schrittweite ist, definiert als der Abstand zwischen zwei benachbarten Levels im Raster.
:::

Um die Effizienz zu gewährleisten, sollte der vorläufige Sampling-Plan $X$ so gestaltet sein, dass jede Evaluierung der Zielfunktion $f$ zur Berechnung von zwei Elementareffekten beiträgt. Zusätzlich sollte der Sampling-Plan eine bestimmte Anzahl (z.B. $r$) von Elementareffekten für jede Variable liefern.

Die `spotpython`-Bibliothek bietet eine Python-Implementierung zur Berechnung der Morris-Screening-Pläne. Die Funktion `screeningplan()` generiert einen Screening-Plan, indem sie die Funktion `randorient()` $r$-mal aufruft, um $r$ zufällige Orientierungen zu erstellen.

Ein Beispiel aus dem `Hyperparameter-Tuning-Cookbook` demonstriert die Analyse der Variablenwichtigkeit für das `Aircraft Wing Weight Example`.

```{python}
import numpy as np
from spotpython.utils.effects import screeningplan

# Beispielparameter
k = 3  # Anzahl der Designvariablen (Dimensionen)
p = 3  # Anzahl der Levels im Raster für jede Variable
xi = 1 # Ein Parameter zur Berechnung der Schrittweite Delta
r = 25 # Anzahl der Elementareffekte pro Variable

# Generieren des Screening-Plans
X = screeningplan(k=k, p=p, xi=xi, r=r)
print(f"Form des generierten Screening-Plans: {X.shape}")
```

::: {.callout-note}
#### Hinweis

* Der Code generiert jedes Mal einen leicht unterschiedlichen Screening-Plan, da er zufällige Orientierungen der Abtastmatrix verwendet*.
:::

In der Praxis können durch Screening gewonnene Läufe für den eigentlichen Modellierungsschritt wiederverwendet werden, insbesondere wenn die Zielfunktion sehr teuer zu evaluieren ist. Dies ist am effektivsten, wenn sich Variablen als völlig inaktiv erweisen, doch da dies selten der Fall ist, muss ein Gleichgewicht zwischen der Wiederverwendung teurer Simulationsläufe und der Einführung potenziellen Rauschens in das Modell gefunden werden.

## Entwurf eines umfassenden Sampling-Plans

Ziel ist es, einen Sampling-Plan zu entwerfen, der eine gleichmäßige Verteilung der Punkte gewährleistet, um eine einheitliche Modellgenauigkeit im gesamten Designraum zu erreichen. Ein solcher Plan wird als **raumfüllend** bezeichnet.

### Stratifikation

Der einfachste Weg, einen Designraum gleichmäßig abzutasten, ist ein rechteckiges Gitter von Punkten, die sogenannte **vollfaktorielle Abtasttechnik**.

```{python}
import numpy as np
from spotpython.utils.sampling import fullfactorial
q = [3, 2]
edges = 1 # Punkte sind gleichmäßig von Rand zu Rand verteilt
X_full_factorial = fullfactorial(q, edges)
print(f"Vollfaktorieller Plan (q={q}, edges={edges}):\n{X_full_factorial}")
print(f"Form des vollfaktoriellen Plans: {X_full_factorial.shape}")
```

Allerdings hat dieser Ansatz zwei wesentliche Einschränkungen:

*   **Beschränkte Designgrößen:** Die Methode funktioniert nur für Designs, bei denen die Gesamtpunktzahl $n$ als Produkt der Anzahl der Levels in jeder Dimension ausgedrückt werden kann ($n = q_1 \times q_2 \times \cdots \times q_k$).
*   **Überlappende Projektionen:** Wenn die Abtastpunkte auf einzelne Achsen projiziert werden, können sich Punktsätze überlappen, was die Effektivität des Sampling-Plans reduziert und zu einer ungleichmäßigen Abdeckung führen kann.

Um die Gleichmäßigkeit der Projektionen für einzelne Variablen zu verbessern, kann der Bereich dieser Variablen in gleich große "Bins" unterteilt werden, und innerhalb dieser Bins können gleich große zufällige Teilstichproben generiert werden. Dies wird als **geschichtete Zufallsstichprobe** bezeichnet. Eine Erweiterung dieser Idee auf alle Dimensionen führt zu einem geschichteten Sampling-Plan, der üblicherweise mittels **Lateinischer Hyperwürfel-Abtastung (Latin Hypercube Sampling, LHS)** implementiert wird.

::: {#def-latin-hypercube}
#### Lateinische Quadrate und Hyperwürfel

Im Kontext der statistischen Stichproben ist ein quadratisches Gitter, das Abtastpositionen enthält, ein Lateinisches Quadrat, wenn (und nur wenn) in jeder Zeile und jeder Spalte nur eine Abtastung vorhanden ist. Ein Lateinischer Hyperwürfel ist die Verallgemeinerung dieses Konzepts auf eine beliebige Anzahl von Dimensionen, wobei jede Abtastung die einzige in jeder achsenparallelen Hyperebene ist, die sie enthält.
:::

Das Generieren eines Lateinischen Hyperwürfels führt zu einem randomisierten Sampling-Plan, dessen Projektionen auf die Achsen gleichmäßig verteilt sind (multidimensionale Stratifikation). Dies garantiert jedoch nicht, dass der Plan raumfüllend ist.

### Maximin-Pläne

Eine weit verbreitete Metrik zur Beurteilung der Gleichmäßigkeit oder "Raumfüllung" eines Sampling-Plans ist die **Maximin-Metrik**.

::: {#def-maximin2-de}
#### Maximin-Plan

Ein Sampling-Plan $X$ wird als Maximin-Plan betrachtet, wenn er unter allen Kandidatenplänen den kleinsten Zwischenpunktabstand $d_1$ maximiert. Unter den Plänen, die diese Bedingung erfüllen, minimiert er ferner $J_1$, die Anzahl der Paare, die durch diesen minimalen Abstand getrennt sind.
:::

Um die Stratifikationseigenschaften von Lateinischen Hyperwürfeln zu bewahren, konzentriert sich die Anwendung dieser Definition auf diese Klasse von Designs. Um das Problem potenziell mehrerer äquivalenter Maximin-Designs zu lösen, wird eine umfassendere "Tie-Breaker"-Definition nach Morris und Mitchell vorgeschlagen:

::: {#def-maximin-tie-breaker}
#### Maximin-Plan mit Tie-Breaker
Ein Sampling-Plan $X$ wird als Maximin-Plan bezeichnet, wenn er sequenziell die folgenden Bedingungen optimiert: Er maximiert $d_1$; unter diesen minimiert er $J_1$; unter diesen maximiert er $d_2$; unter diesen minimiert er $J_2$; und so weiter, bis er $J_m$ minimiert.
:::

Für die Berechnung von Distanzen in diesen Kontexten wird die **p-Norm** am häufigsten verwendet:

::: {#def-p-norm}
#### p-Norm

Die p-Norm eines Vektors $\vec{x} = (x_1, x_2, \ldots, x_k)$ ist definiert als:
$$
d_p(\vec{x}^{(i_1)}, \vec{x}^{(i_2)}) = \left( \sum_{j=1}^k |x_j^{(i_1)} - x_j^{(i_2)}|^p \right)^{1/p}.
$$
:::

Wenn $p=1$, definiert dies die **Rechteckdistanz** (oder Manhattan-Norm), und wenn $p=2$, die **Euklidische Norm**. Die Rechteckdistanz ist rechnerisch erheblich weniger aufwendig, was besonders bei großen Sampling-Plänen von Vorteil sein kann.

Die `spotpython` Bibliothek bietet Funktionen zur Implementierung dieser Kriterien, wie `mm()` für paarweise Vergleiche von Sampling-Plänen.

### Das Morris-Mitchell-Kriterium (Phi_q)

Um konkurrierende Sampling-Pläne in einer kompakten Form zu bewerten, definierten Morris und Mitchell (1995) die folgende skalarwertige Kriteriumsfunktion, die **Morris-Mitchell-Kriterium** genannt wird:


::: {#def-morris-mitchell}
#### Morris-Mitchell-Kriterium

Das Morris-Mitchell-Kriterium ist definiert als:
$$
\Phi_q (X) = \left(\sum_{j=1}^m J_j d_j^{-q}\right)^{1/q},
$$
wobei $X$ der Sampling-Plan ist, $d_j$ der Abstand zwischen den Punkten, $J_j$ die Vielfachheit dieses Abstands und $q$ ein benutzerdefinierter Exponent ist. Der Parameter $q$ kann angepasst werden, um den Einfluss kleinerer Abstände auf die Gesamtmetrik zu steuern. Ein kleinerer Wert von $\Phi_q$ deutet auf bessere raumfüllende Eigenschaften des Sampling-Plans hin.
:::

Größere Werte von $q$ stellen sicher, dass Terme in der Summe, die kleineren Zwischenpunktabständen entsprechen, einen dominanten Einfluss haben, was dazu führt, dass $\Phi_q$ die Sampling-Pläne in einer Weise ordnet, die der ursprünglichen Maximin-Definition (@def-maximin2) sehr genau entspricht. Kleinere $q$-Werte hingegen erzeugen eine $\Phi_q$-Landschaft, die der ursprünglichen Definition zwar nicht perfekt entspricht, aber im Allgemeinen optimierungsfreundlicher ist. Es wird empfohlen, $\Phi_q$ für eine Reihe von $q$-Werten (z.B. 1, 2, 5, 10, 20, 50 und 100) zu minimieren und dann den besten Plan aus diesen Ergebnissen anhand der tatsächlichen Maximin-Definition auszuwählen.

Die Funktionen `mmphi()` und `mmphi_intensive()` in `spotpython` berechnen das Morris-Mitchell-Kriterium, unterscheiden sich jedoch in ihrer Normalisierung. Die `mmphi_intensive()`-Funktion ist invariant gegenüber der Abtastgröße, da sie die Summe $\sum J_l d_l^{-q}$ durch $M$ (die Gesamtzahl der Paare, $M = N(N-1)/2$) teilt, was einen durchschnittlichen Beitrag pro Paar zur $-q$-ten Potenz des Abstands vor dem Ziehen der $q$-ten Wurzel berechnet. Dies ermöglicht aussagekräftigere Vergleiche der Raumfüllung zwischen Designs unterschiedlicher Größe. Ein kleinerer Wert zeigt bei beiden Kriterien ein besseres (raumfüllenderes) Design an.

```{python}
import numpy as np
from spotpython.utils.sampling import mmphi, mmphi_intensive, rlh

# Beispiel: Erstellen von zwei Latin Hypercube Designs
np.random.seed(42)
X1 = rlh(n=10, k=2) # 10 Punkte in 2 Dimensionen
X2 = rlh(n=20, k=2) # 20 Punkte in 2 Dimensionen

q_val = 2.0 # Exponent q
p_val = 2.0 # p-Norm (Euclidean distance)

# Berechne Phi_q für X1 und X2
phi_q_X1 = mmphi(X1, q_val, p_val)
phi_q_X2 = mmphi(X2, q_val, p_val)

# Berechne Phi_q_intensive für X1 und X2
phi_q_intensive_X1, _, _ = mmphi_intensive(X1, q_val, p_val)
phi_q_intensive_X2, _, _ = mmphi_intensive(X2, q_val, p_val)

print(f"Morris-Mitchell Criterion (Phi_q) für X1 (10 Pts): {phi_q_X1:.3f}")
print(f"Morris-Mitchell Criterion (Phi_q) für X2 (20 Pts): {phi_q_X2:.3f}")
print("-" * 30)
print(f"Morris-Mitchell Criterion (Phi_q_intensive) für X1 (10 Pts): {phi_q_intensive_X1:.3f}")
print(f"Morris-Mitchell Criterion (Phi_q_intensive) für X2 (20 Pts): {phi_q_intensive_X2:.3f}")
```

:::{#rem-morris-mitchell-de}
#### Morris-Mitchell-Kriterium
$\Phi_q$ steigt tendenziell mit der Anzahl der Punkte ($N$), während $\Phi_q^I$ (`mmphi_intensive`) abtastgrößeninvariant ist und daher besser für den Vergleich von Designs unterschiedlicher Größe geeignet ist.
:::

## Alternative Sampling-Pläne: Sobol-Sequenzen

Wenn die gesamte Rechenzeit budgetiert ist, aber nicht klar ist, wie viele Kandidatendesigns in dieser Zeit evaluiert werden können, bieten sich **Sobol-Sequenzen** als Alternative an. Diese Sampling-Pläne weisen gute raumfüllende Eigenschaften auf (zumindest für große $n$) und besitzen die Eigenschaft, dass für jedes $n$ und $k > 1$ die Sequenz für $n-1$ und $k$ eine Untermenge der Sequenz für $n$ und $k$ ist.

## Zusammenfassung

Die Auswahl des richtigen Sampling-Plans ist ein grundlegender Schritt in Computerexperimenten und der Surrogatmodellierung. Von traditionellen RSM-Ansätzen bis hin zu modernen DACE-Methoden, die den "Fluch der Dimensionen" mildern und effiziente Screenings ermöglichen, entwickeln sich die Techniken ständig weiter. Maximin-Pläne und das Morris-Mitchell-Kriterium bieten robuste Methoden zur Quantifizierung der Raumfüllung, wobei abtastgrößeninvariante Kriterien wie `mmphi_intensive()` Vergleiche über verschiedene Designgrößen hinweg ermöglichen.


## Zusatzmaterialien


Viele der in diesem Dokument beschriebenen Konzepte sind in den [Jupyter Notebooks](https://github.com/sequential-parameter-optimization/Hyperparameter-Tuning-Cookbook/tree/main/notebooks) zum "Hyperparameter Tuning Cookbook" verfügbar und können dort interaktiv erkundet werden. Die Python-Bibliothek `spotpython` ([GitHub Repository](https://github.com/sequential-parameter-optimization/spotPython)) bietet Implementierungen für viele dieser Probenahme- und Optimierungsstrategien.

:::{.callout-note}
#### Interaktive Webseite

* Eine interaktive Webseite zum Thema **Curse of Dimensionality** ist hier zu finden: [Curse of Dimensionality interaktiv](https://advm1.gm.fh-koeln.de/~bartz/bart21i/en_curse_interactive.html).

:::


:::{.callout-note}
#### Jupyter-Notebook

* Das Jupyter-Notebook für dieses Lernmodul ist auf GitHub im [Hyperparameter-Tuning-Cookbook Repository](https://github.com/sequential-parameter-optimization/Hyperparameter-Tuning-Cookbook/blob/main/de_kriging.ipynb) verfügbar.

:::

