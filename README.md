# Ingegneria-della-conoscenza
# Ingegneria della Conoscenza - Classificazione intelligente dei funghi

## Struttura della Repository
La repository è organizzata in diverse cartelle per una gestione più ordinata dei file:

Cartella code: contiene tutto il codice relativo al progetto

Cartella data: contiene file utili per il progetto

---

## Requisiti
Prima di eseguire il codice, assicurati di avere **Python 3.8+** installato sul tuo sistema.

### Installazione delle librerie richieste
Puoi installare tutte le librerie necessarie eseguendo i seguenti comandi:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn imbalanced-learn
```

Per usare **Prolog**, devi installare SWI-Prolog per poter eseguire la Knowledge Base.

---

## Esecuzione del Progetto
Per eseguire l'intero workflow del progetto

### 1️ Preprocessing dei Dati
```bash
python scripts/preprocessing.py
```

### 2️ Selezione delle Feature
```bash
python scripts/select_feature.py
```

### 3️ Generazione della Knowledge Base
```bash
python scripts/generate_kb.py
```

### 4️ Ottimizzazione degli Iperparametri
```bash
python scripts/hyperparameter_tuning.py
```

### 5️ Addestramento e Confronto dei Modelli
```bash
python scripts/learning.py
```
