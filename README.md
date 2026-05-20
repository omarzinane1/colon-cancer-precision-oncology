# Colon Cancer Precision Oncology

Projet Data Science + DevOps simple pour classifier l'etat d'un patient a partir de donnees d'expression genetique du cancer du colon.

Le modele utilise une Logistic Regression avec uniquement 6 genes d'expression genetique :

- M63391
- T62947
- D14812
- T51250
- H66976
- X55362

> Important : ce projet est une demonstration pedagogique. Il ne doit pas etre utilise comme outil de diagnostic medical final.

## Dataset

Le fichier utilise est :

```text
data/colon_cancer.csv
```

Le CSV contient de nombreuses colonnes de genes. La colonne cible du dataset est :

```text
Class
```

Les classes predites sont :

- Normal
- Abnormal

Le pipeline ne garde que les 6 genes selectionnes afin de construire un modele simple, interpretable et facile a presenter.

## Selection des genes

Le dataset contient beaucoup plus de genes que d'echantillons. Une selection de variables est donc utile pour reduire la dimension, limiter la complexite du modele et garder une application Flask simple avec seulement 6 valeurs a saisir.

Dans la version initiale du projet, les 6 genes etaient definis manuellement dans le notebook, `train/train.py` et `deploy/app.py` :

- M63391
- T62947
- D14812
- T51250
- H66976
- X55362

Une verification a ete ajoutee dans le notebook avec Sequential Forward Selection (SFS). SFS commence sans variable, ajoute a chaque etape le gene qui ameliore le plus l'accuracy en validation croisee, et s'arrete ici apres avoir selectionne exactement 6 genes.

SFS a selectionne :

- R02593
- T55131
- H20709
- T63508
- M63391
- D00749

Comparaison sur le meme split train/test stratifie :

- Accuracy avec les 6 genes manuels : `0.9231`
- Accuracy avec les 6 genes SFS : `0.7692`

Comme SFS a choisi des genes differents mais n'a pas ameliore l'accuracy, le projet final conserve les 6 genes manuellement predefinis. Le notebook documente cette comparaison et montre le code SFS complet.

Ce projet utilise une selection de variables, pas PCA. SFS garde des genes originaux du dataset, ce qui rend le modele plus interpretable. PCA creerait de nouvelles composantes transformees, moins directes a expliquer dans ce contexte medical pedagogique.

Aucune affirmation biologique ou clinique n'est faite sur ces genes.

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
Dataset -> Selection/conservation des 6 genes -> Class -> StandardScaler -> LogisticRegression -> Sauvegarde modele -> Flask prediction
```

1. Le fichier `data/colon_cancer.csv` contient plusieurs colonnes de genes et la colonne cible `Class`.
2. Le script d'entrainement conserve les 6 genes finaux utilises comme features.
3. La colonne `Class` contient les classes `Normal` et `Abnormal`.
4. Les classes sont encodees avec `LabelEncoder`.
5. Les valeurs genetiques sont standardisees avec `StandardScaler`.
6. Le modele `LogisticRegression` est entraine et evalue.
7. Les fichiers `model.pkl`, `scaler.pkl` et `label_encoder.pkl` sont sauvegardes dans `models/`.
8. L'application Flask charge ces fichiers et expose un formulaire web de prediction.

## Lancer Jupyter

Depuis le dossier `colon_cancer_precision_oncology/` :

```bash
docker compose up jupyter
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
docker compose up --build train
```

Cette commande cree les fichiers suivants :

```text
models/model.pkl
models/scaler.pkl
models/label_encoder.pkl
```

## Lancer l'application Flask

```bash
docker compose up --build deploy
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
docker compose up --build
```

Le service `deploy` depend du service `train`, afin que l'entrainement soit lance avant l'application web.

## Exemple de valeurs a tester

Exemple issu du dataset, prediction attendue proche de `Abnormal` :

```text
M63391=552.65875
T62947=74.5325
D14812=438.24625
T51250=512.1238
H66976=115.395
X55362=315.52374
```

Exemple issu du dataset, prediction attendue proche de `Normal` :

```text
M63391=2314.9487
T62947=66.07
D14812=589.72375
T51250=946.5588
H66976=283.4325
X55362=379.9875
```
