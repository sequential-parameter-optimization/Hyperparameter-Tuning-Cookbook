---
lang: de
---

# Lernmodul: Aircraft Wing Weight Example (AWWE)

## Einleitung

Die Aircraft Wing Weight Example (AWWE) Funktion dient dazu, das Gewicht eines unlackierten Leichtflugzeugflügels als Funktion verschiedener Design- und Betriebsparameter zu verstehen. Diese Sektion basiert auf Kapitel 1.3 "A ten-variable weight function" in "Engineering Design via Surrogate Modelling: A Practical Guide" von Forrester, Sóbester und Keane (2008) [@Forr08a].

Die in diesem Kontext verwendete Gleichung ist keine reale Computersimulation, wird aber zu Illustrationszwecken als solche behandelt. Sie stellt eine mathematische Gleichung dar, deren funktionale Form durch die Kalibrierung bekannter physikalischer Beziehungen an Kurven aus bestehenden Flugzeugdaten abgeleitet wurde, wie in Raymer (2006) referenziert [@raym06a]. Im Wesentlichen fungiert sie als Ersatzmodell (*surrogate*) für tatsächliche Messungen des Flugzeuggewichts. Die Studie nimmt ein Cessna C172 Skyhawk Flugzeug als Referenzpunkt.

## Die AWWE-Gleichung

Die ursprüngliche AWWE-Gleichung beschreibt das Gewicht ($W$) eines unlackierten Leichtflugzeugflügels basierend auf neun Design- und Betriebsparametern:

$$
W = 0.036 S_W^{0.758} \times W_{fw}^{0.0035} \left( \frac{A}{\cos^2 \Lambda} \right)^{0.6} \times q^{0.006} \times \lambda^{0.04} \times \left( \frac{100 R_{tc}}{\cos \Lambda} \right)^{-0.3} \times (N_z W_{dg})^{0.49}$$

Im Rahmen einer Übung wird die Formel um das Lackgewicht ($W_p$) erweitert. Die aktualisierte Gleichung lautet dann:

$$
W = 0.036S_W^{0.758} \times W_{fw}^{0.0035} \times \left( \frac{A}{\cos^2 \Lambda} \right)^{0.6} \times q^{0.006} \times \lambda^{0.04} \times \left( \frac{100 R_{tc}}{\cos \Lambda} \right)^{-0.3} \times (N_z W_{dg})^{0.49} + S_w W_p
$$

## Eingabegrößen (Parameter)

Die AWWE-Funktion verwendet zehn Design- und Betriebsparameter. 

**Parameterbeschreibungen:**

- **$S_W$ (Flügelfläche):** Die projizierte Fläche des Flügels. Sie ist ein Haupttreiber für den Auftrieb, aber auch für das Gewicht und den Luftwiderstand.
- **$W_{fw}$ (Treibstoffgewicht im Flügel):** Das Gewicht des im Flügel untergebrachten Treibstoffs.
- **$A$ (Aspektverhältnis):** Das Verhältnis von Spannweite zum Quadrat zur Flügelfläche. Eine hohe Streckung (lange, schmale Flügel) ist aerodynamisch effizient, führt aber zu höheren Biegebelastungen und damit potenziell zu mehr Gewicht.
- **$\Lambda$ (Viertel-Chord-Pfeilung):** Der Winkel, um den der Flügel nach hinten geneigt ist. Die Pfeilung ist entscheidend für das Verhalten bei hohen Geschwindigkeiten, beeinflusst aber auch die Strukturlasten.
- **$q$ (Dynamischer Druck im Reiseflug):** Ein Maß für die aerodynamische Belastung, abhängig von Luftdichte und Fluggeschwindigkeit.
- **$\lambda$ (Konizität):** Das Verhältnis der Flügeltiefe an der Spitze zur Flügeltiefe an der Wurzel. Es beeinflusst die Auftriebsverteilung und die Struktureffizienz.
- **$R_{tc}$ (Relative Profildicke):** Das Verhältnis der maximalen Dicke des Flügelprofils zu seiner Tiefe (Profilsehne). Dickere Profile bieten mehr Platz für Struktur und Treibstoff, erzeugen aber mehr Luftwiderstand.
- **$N_z$ (Ultimativer Lastfaktor):** Der maximale Lastvielfache (g-Kraft), dem die Flügelstruktur standhalten muss, ohne zu versagen. Ein entscheidender Parameter für die strukturelle Auslegung.
- **$W_{dg}$ (Flugdesign-Bruttogewicht):** Das maximale Gewicht, für das das Flugzeug ausgelegt ist.
- **$W_p$ (Lackgewicht):** Das Gewicht der Lackierung pro Flächeneinheit.

Die Parameter sind in der folgenden Tabelle mit ihren Symbolen, Beschreibungen sowie Basis-, Minimal- und Maximalwerten aufgeführt:

| Symbol | Parameter | Basiswert | Minimum | Maximum |
| :----- | :---------------------------------- | :-------- | :------ | :------ |
| $S_W$ | Flügelfläche ($ft^2$) | 174 | 150 | 200 |
| $W_{fw}$ | Gewicht des Treibstoffs im Flügel (lb) | 252 | 220 | 300 |
| $A$ | Aspektverhältnis | 7.52 | 6 | 10 |
| $\Lambda$ | Viertel-Chord-Pfeilung (deg) | 0 | -10 | 10 |
| $q$ | Dynamischer Druck bei Reisegeschwindigkeit ($lb/ft^2$) | 34 | 16 | 45 |
| $\lambda$ | Konizität | 0.672 | 0.5 | 1 |
| $R_{tc}$ | Profil-Dicken-Sehnen-Verhältnis | 0.12 | 0.08 | 0.18 |
| $N_z$ | Ultimativer Lastfaktor | 3.8 | 2.5 | 6 |
| $W_{dg}$ | Flugdesign-Bruttogewicht (lb) | 2000 | 1700 | 2500 |
| $W_p$ | Lackgewicht ($lb/ft^2$) | 0.064 | 0.025 | 0.08 |


## Ausgabegröße

Die Ausgabegröße der AWWE-Funktion ist das berechnete Gewicht des Flugzeugflügels in Pfund (lb), das Flügelgewicht ($W$).

## Analyse von Effekten, Interaktionen und Wichtigkeit

###  Mathematische Eigenschaften und Ziele

Die mathematischen Eigenschaften der AWWE-Gleichung zeigen, dass die Reaktion des Modells in Bezug auf ihre Eingaben hochgradig nichtlinear ist. Obwohl die Anwendung des Logarithmus zur Vereinfachung von Gleichungen mit komplexen Exponenten üblich ist, bleibt die Reaktion selbst bei Modellierung des Logarithmus aufgrund des Vorhandenseins trigonometrischer Terme nichtlinear. Angesichts der Kombination aus Nichtlinearität und hoher Eingabedimension sind einfache lineare und quadratische Response-Surface-Approximationen für diese Analyse wahrscheinlich unzureichend.

Die primären Ziele der Studie umfassen das tiefgreifende Verständnis der Input-Output-Beziehungen und die Optimierung. Insbesondere besteht ein Interesse daran, das Gewicht des Flugzeugs zu minimieren, wobei bestimmte Einschränkungen, wie eine notwendige Nicht-Null-Flügelfläche für die Flugfähigkeit, berücksichtigt werden müssen. Eine globale Perspektive und der Einsatz flexibler Modellierung sind für solche (eingeschränkten) Optimierungsszenarien unerlässlich.

### Wichtigkeit der Variablen und deren Effekte

Die Analyse des "AWWE Landscape" und spezifischer Plot-Interpretationen liefert wichtige Einblicke in die Auswirkungen und Interaktionen der einzelnen Designparameter auf das Flügelgewicht:

####  Dominante Variablen (wichtigste Einflussfaktoren)

Die wichtigsten Variablen, die einen signifikanten Einfluss auf das Flügelgewicht haben, sind:

*   **Ultimativer Lastfaktor ($N_z$):** Dieser Faktor bestimmt die Größe der maximalen aerodynamischen Last auf den Flügel und ist sehr aktiv, da er stark mit anderen Variablen interagiert.
*   **Flügelfläche ($S_W$):** Ein entscheidender Parameter für das Gesamtgewicht.
*   **Flugdesign-Bruttogewicht ($W_{dg}$):** Hat ebenfalls einen sehr starken Einfluss auf das Flügelgewicht.

Diese Variablen zeigen in Screening-Studien sowohl große zentrale Tendenzwerte als auch hohe Standardabweichungen, was auf starke direkte Effekte und komplexe Interaktionen hindeutet.

#### Interaktionen zwischen Variablen

*   **Interaktion von $N_z$ (Ultimativer Lastfaktor) und $A$ (Aspektverhältnis):** Dies ist ein klassisches Beispiel für eine Interaktion, die zu einem schweren Flügel führt, insbesondere bei hohen Aspektverhältnissen und großen G-Kräften. Diese Beobachtung steht im Einklang mit der Designphilosophie von Kampfflugzeugen, die aufgrund der Notwendigkeit, hohen G-Kräften standzuhalten, keine effizienten, segelflugzeugähnlichen Flügel aufweisen können. Die leichte Krümmung der Konturlinien im $N_z$ vs. $A$-Plot deutet auf diese Interaktion hin. Der darstellte Ausgabebereich in diesem Plot (von ca. 160 bis 320) deckt fast den gesamten Bereich der Ausgaben ab, die aus verschiedenen Eingabeeinstellungen im vollen 9-dimensionalen Eingaberaum beobachtet wurden.
*   **Aspektverhältnis ($A$) und Profil-Dicken-Sehnen-Verhältnis ($R_{tc}$):** Diese beiden Variablen zeigen nichtlineare Interaktionen. Ihre hohen Standardabweichungen in Screening-Studien weisen auf signifikantes nichtlineares Verhalten und Interaktionen mit anderen Variablen hin.

#### Variablen mit geringem Einfluss

Einige Variablen haben, insbesondere innerhalb der untersuchten Bereiche, nur einen geringen Einfluss auf das Flügelgewicht:

*   **Dynamischer Druck ($q$):** Hat innerhalb des gewählten Bereichs nur begrenzte Auswirkungen auf das Flügelgewicht. Dies kann im Wesentlichen als Darstellung unterschiedlicher Reiseflughöhen bei gleicher Geschwindigkeit interpretiert werden.
*   **Konizität ($\lambda$):** Zeigt geringen Einfluss.
*   **Viertel-Chord-Pfeilung ($\Lambda$):** Hat ebenfalls geringen Einfluss, insbesondere innerhalb des engen Bereichs von -10° bis +10°, der typisch für leichte Flugzeuge ist.
*   **Lackgewicht ($W_p$):** Wie zu erwarten, trägt das Lackgewicht nur wenig zum Gesamtgewicht des Flügels bei.

Variablen mit minimalem Einfluss gruppieren sich in Screening-Plots nahe dem Ursprung, was auf ihren geringen Einfluss auf die Zielfunktion hinweist.

#### Variablen mit moderatem, linearem Einfluss

*   **Flügelkraftstoffgewicht ($W_{fw}$):** Während $W_{fw}$ immer noch nahe dem Nullpunkt liegt, zeigt es in Screening-Studien eine leicht größere zentrale Tendenz bei sehr geringer Standardabweichung. Dies deutet auf eine moderate Bedeutung hin, aber auf minimale Beteiligung an Interaktionen mit anderen Variablen. Der Vergleich der Konizität ($\lambda$) und des Kraftstoffgewichts ($W_{fw}$) zeigt, dass keiner der beiden Inputs das Flügelgewicht stark beeinflusst. $\lambda$ hat dabei einen geringfügig größeren Effekt, der weniger als 4 Prozent der Gewichtsspanne ausmacht, die im $A \times N_z$-Diagramm beobachtet wurde. Eine Interaktion ist zwischen $\lambda$ und $W_{fw}$ nicht erkennbar.

### Expertenwissen und Screening-Studien

Flugzeugkonstrukteure wissen, dass das Gesamtgewicht des Flugzeugs und die Flügelfläche auf ein Minimum reduziert werden müssen. Letztere wird oft durch Beschränkungen wie die erforderliche Strömungsabrissgeschwindigkeit, die Landestrecke oder die Wendegeschwindigkeit vorgegeben. Die Anforderung an einen hohen ultimativen Lastfaktor ($N_z$) führt unweigerlich zu der Notwendigkeit robuster, schwerer Flügel.

Die Durchführung von Screening-Studien, oft mittels der Morris-Methode, ist von entscheidender Bedeutung, um die Anzahl der Designvariablen zu minimieren, bevor die Zielfunktion modelliert wird. Diese Methode hilft, die Dimensionalität zu reduzieren, ohne die Analyse zu beeinträchtigen. Ein großer Wert des zentralen Trends in den Ergebnissen der Morris-Methode deutet darauf hin, dass eine Variable einen signifikanten Einfluss auf die Zielfunktion über den Designbereich hinweg hat. Eine große Streuung hingegen legt nahe, dass die Variable in Interaktionen involviert ist oder zur Nichtlinearität der Funktion beiträgt. Die Visualisierung dieser Ergebnisse (Stichprobenmittelwerte und Standardabweichungen) hilft dabei, die wichtigsten Variablen zu identifizieren und zu erkennen, ob ihre Effekte linear sind oder Interaktionen beinhalten.


## Zusatzmaterialien

:::{.callout-note}
#### Interaktive Webseite

* Eine interaktive Webseite zum Thema **Aircraft Wing Weight Example** ist hier zu finden: [Aircraft Wing Weight Interaktiv](https://advm1.gm.fh-koeln.de/~bartz/bart21i/de_awwe_interactive.html).


:::

:::{.callout-note}
#### Audiomaterial

* Eine Audio zum Thema **Aircraft Wing Weight Example** ist hier zu finden: [Aircraft Wing Weight Audio](https://advm1.gm.fh-koeln.de/~bartz/bart21i/audio/DecodingComplexity.mp3).


:::

:::{.callout-note}
#### Jupyter-Notebook

* Das Jupyter-Notebook für dieses Lernmodul ist auf GitHub im [Hyperparameter-Tuning-Cookbook Repository](https://github.com/sequential-parameter-optimization/Hyperparameter-Tuning-Cookbook/blob/main/de_kriging.ipynb) verfügbar.

:::

