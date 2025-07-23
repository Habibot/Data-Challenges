# 🪙 Prägefolgen antiker Münzen mit GNNs vorhersagen

Dieses Projekt nutzt ein Graph Neural Network (GNN) mit Bild-Embeddings, um **die Prägereihenfolge antiker Münzen** vorherzusagen.

![Visualisierung](img.png)

---

## 🔧 Vorbereitung

1. **Excel-Datei hinzufügen**
   `Stempelliste_bueschel_Neuses_einfach.xlsx` ins Projektverzeichnis legen.

2. **Bilder einfügen**
   Ordner `images/` erstellen und alle Münzbilder (eine Seite pro Münze, z. B. `123_a.jpg`) hineinlegen.

3. **Script ausführen**

   ```bash
   python script.py
   ```

---

## 🧠 Modell

* Verwendet **CLIP** für Bild-Embeddings.
* Lernt mit einem GAT-Modell Münzfolgen aus Ground-Truth-Ketten.
* Macht schrittweise Vorhersagen über mögliche Prägeabfolgen.

---

## 📊 Ausgabe

* Visualisierung des Graphs mit Bildern
* HTML-Dateien mit Vorhersagepfaden
* Schrittgenaue Accuracy-Auswertung
