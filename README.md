# Colon Cancer Precision Oncology

Projet Data Science + DevOps simple pour classifier l'etat d'un patient a partir de donnees d'expression genetique du cancer du colon.

Le modele utilise une Logistic Regression avec uniquement 6 genes selectionnes :

- M63391
- T62947
- D14812
- T51250
- H66976
- X55362

> Important : le dataset fourni est synthetique et sert a rendre le projet directement executable pour une demonstration. Il ne doit pas etre utilise comme outil medical.

## Architecture

```text
colon_cancer_precision_oncology/
|
├── data/
│   └── colon_cancer.csv
├── notebooks/
│   └── logistic_regression_analysis.ipynb
├── train/
│   ├── train.py
│   ├── requirements.txt
│   └── Dockerfile
├── deploy/
│   ├── app.py
│   ├── requirements.txt
│   ├── Dockerfile
│   ├── templates/
│   │   └── index.html
│   └── static/
│       └── style.css
├── models/
├── docker-compose.yml
└── README.md
```

## Pipeline

```text
Dataset -> Pretraitement -> StandardScaler -> LogisticRegression -> Sauvegarde modele -> Flask prediction
```

1. Le fichier `data/colon_cancer.csv` contient les 6 genes et le label `normal` ou `cancer`.
2. Le script d'entrainement encode les labels avec `LabelEncoder`.
3. Les valeurs genetiques sont standardisees avec `StandardScaler`.
4. Le modele `LogisticRegression` est entraine et evalue.
5. Les fichiers `model.pkl`, `scaler.pkl` et `label_encoder.pkl` sont sauvegardes dans `models/`.
6. L'application Flask charge ces fichiers et expose un formulaire web de prediction.

## Lancer Jupyter

Depuis le dossier `colon_cancer_precision_oncology/` :

```bash
docker-compose up jupyter
```

Puis ouvrir :

```text
http://localhost:8888
```

Le notebook se trouve dans :

```text
notebooks/logistic_regression_analysis.ipynb
```

## Entrainer le modele

```bash
docker-compose up --build train
```

Cette commande cree les fichiers suivants :

```text
models/model.pkl
models/scaler.pkl
models/label_encoder.pkl
```

## Lancer l'application Flask

```bash
docker-compose up --build deploy
```

Puis ouvrir :

```text
http://localhost:5000
```

Si le modele n'existe pas encore, Flask affiche :

```text
Model not found. Please run the training service first.
```

## Tout lancer

```bash
docker-compose up --build
```

Le service `deploy` depend du service `train`, afin que l'entrainement soit lance avant l'application web.

## Exemple de valeurs a tester

Prediction attendue proche de `normal` :

```text
M63391=5.10
T62947=6.24
D14812=4.02
T51250=7.12
H66976=2.80
X55362=4.60
```

Prediction attendue proche de `cancer` :

```text
M63391=8.14
T62947=3.12
D14812=7.38
T51250=4.16
H66976=7.52
X55362=8.63
```
