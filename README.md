# QRT Data Challenge 2024 — Leukemia Risk Prediction

<p align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/4/4d/Logo-gustave-roussy.jpg/1200px-Logo-gustave-roussy.jpg" alt="Gustave Roussy" width="200"/>
  <img src="https://upload.wikimedia.org/wikipedia/en/thumb/3/3f/Qube_Research_%26_Technologies_Logo.svg/1200px-Qube_Research_%26_Technologies_Logo.svg.png" alt="QRT" width="150" style="margin-left: 20px;"/>
</p>

> **Objectif**: Prédire le risque de décès pour des patients atteints de leucémie myéloïde en utilisant des données cliniques et moléculaires.

## 📋 Table des Matières

- [Installation Rapide](#-installation-rapide)
- [Utilisation](#-utilisation)
- [Structure du Projet](#-structure-du-projet)
- [Méthodologie](#-méthodologie)
- [Résultats](#-résultats)
- [Historique des Modifications](#-historique-des-modifications)

---

## 🚀 Installation Rapide

### Prérequis

- Python 3.9+
- pip ou conda

### Installation

```bash
# 1. Cloner le repository
git clone https://github.com/chenenen13/QRT-Leukemia-DataChallenge.git
cd QRT-Leukemia-DataChallenge

# 2. Créer un environnement virtuel (recommandé)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou: venv\Scripts\activate  # Windows

# 3. Installer les dépendances
pip install -r requirements.txt
```

### Dépendances principales

| Package | Version | Description |
|---------|---------|-------------|
| numpy | ≥1.24.0 | Calculs numériques |
| pandas | ≥2.0.0 | Manipulation de données |
| scikit-learn | ≥1.3.0 | Machine learning |
| scikit-survival | ≥0.22.0 | Modèles de survie |
| lightgbm | ≥4.0.0 | Gradient boosting |
| numba | ≥0.58.0 | Optimisation JIT |

---

## 💻 Utilisation

### Option 1: Exécuter le rapport complet

```bash
# Ouvrir le notebook principal (rapport)
jupyter notebook main.ipynb
```

Ce notebook contient:
- Analyse exploratoire complète
- Tous les modèles (baseline, clustering, RSF)
- Visualisations et interprétations
- Génération du fichier de soumission

### Option 2: Utiliser les modules Python

```python
from src.data_loader import load_all_data
from src.features import build_molecular_features
from src.models import create_rsf_model
from src.evaluation import ipcw_cindex

# Charger les données
clinical_train, clinical_test, molecular_train, molecular_test, y_train = load_all_data()

# Feature engineering
mol_features = build_molecular_features(molecular_train)

# Créer et entraîner un modèle
model = create_rsf_model({"n_estimators": 400})
# ...
```

### Option 3: Notebook de développement

```bash
# Pour le notebook de développement détaillé
jupyter notebook DataChallenge_ML.ipynb
```

### Générer une soumission

Après exécution de `main.ipynb` ou `DataChallenge_ML.ipynb`:

```bash
# Le fichier submission.csv est créé à la racine
cat submission.csv | head
```

Format attendu:
```csv
ID,risk_score
P123456,2.345
P123457,1.234
...
```

---

## 📁 Structure du Projet

```
QRT-Leukemia-DataChallenge/
│
├── 📊 data/                        # Données brutes
│   ├── clinical_train.csv          # Données cliniques (train)
│   ├── clinical_test.csv           # Données cliniques (test)
│   ├── molecular_train.csv         # Mutations génétiques (train)
│   ├── molecular_test.csv          # Mutations génétiques (test)
│   └── target_train.csv            # Labels (OS_YEARS, OS_STATUS)
│
├── 📦 src/                         # Modules Python
│   ├── __init__.py                 # Package init
│   ├── config.py                   # Configuration et constantes
│   ├── data_loader.py              # Chargement et validation des données
│   ├── features.py                 # Feature engineering
│   ├── preprocessing.py            # Pipelines sklearn
│   ├── models.py                   # Définitions des modèles
│   ├── evaluation.py               # Métriques et cross-validation
│   ├── optimization.py             # Fonctions Numba optimisées
│   └── visualization.py            # Graphiques et visualisations
│
├── 📓 main.ipynb                   # Rapport principal (à soumettre)
├── 📓 DataChallenge_ML.ipynb       # Notebook de développement
├── 📓 Benchmark_nqBJ7fO.ipynb      # Benchmark fourni par QRT
│
├── 📄 requirements.txt             # Dépendances Python
├── 📄 submission.csv               # Fichier de soumission
└── 📄 README.md                    # Ce fichier
```

### Description des modules `src/`

| Module | Description |
|--------|-------------|
| `config.py` | Constantes, chemins, hyperparamètres par défaut |
| `data_loader.py` | Fonctions de chargement CSV, validation, fusion des datasets |
| `features.py` | Agrégation des mutations au niveau patient, extraction de features |
| `preprocessing.py` | Pipelines sklearn (imputation, scaling, TF-IDF, SVD) |
| `models.py` | Classes de modèles (Baseline, Clustering, RSF) |
| `evaluation.py` | IPCW C-index, cross-validation, grid search |
| `optimization.py` | Fonctions Numba pour calculs intensifs |
| `visualization.py` | Graphiques matplotlib pour le rapport |

---

## 🔬 Méthodologie

### Données

- **Train**: 3,323 patients avec labels (OS_YEARS, OS_STATUS)
- **Test**: 1,193 patients à prédire
- **24 centres cliniques**

### Pipeline

```
┌──────────────┐     ┌────────────────┐     ┌──────────────┐
│   Données    │ ──▶ │   Feature      │ ──▶ │  Preprocess  │
│   Brutes     │     │   Engineering  │     │  Pipeline    │
└──────────────┘     └────────────────┘     └──────────────┘
                                                   │
                                                   ▼
┌──────────────┐     ┌────────────────┐     ┌──────────────┐
│  Submission  │ ◀── │   Évaluation   │ ◀── │   Modèles    │
│  risk_score  │     │   IPCW C-index │     │   Survie     │
└──────────────┘     └────────────────┘     └──────────────┘
```

### Modèles implémentés

1. **Baseline (Ridge Regression)**: Régression sur OS_YEARS, ignore la censure
2. **KMeans Clustering**: Non-supervisé, risque par médiane de cluster
3. **Random Survival Forest**: Gère la censure, hyperparamètres optimisés

### Métrique

**IPCW C-index** (τ = 7 ans): Mesure la capacité à ordonner correctement les paires de patients selon leur survie.

---

## 📈 Résultats

| Modèle | IPCW C-index (validation) |
|--------|---------------------------|
| Baseline (Ridge) | ~0.64 |
| KMeans Clustering | ~0.62 |
| **Random Survival Forest** | **~0.70** |

### Features les plus importantes

1. `BM_BLAST` (blastes moelle osseuse)
2. `PLT` (plaquettes)
3. `HB` (hémoglobine)
4. `n_mut` (nombre de mutations)
5. `vaf_mean` (VAF moyen)

---

## 📝 Historique des Modifications

### Version 2.0 (Actuelle) — Restructuration complète

#### Changements majeurs

| Avant (`DataChallenge_ML.ipynb` v1) | Après (v2) |
|-------------------------------------|------------|
| Chargement via URL GitHub | Chargement local (`data/`) |
| Fonctions définies dans le notebook | Modules Python dans `src/` |
| Pas d'optimisation | Fonctions Numba JIT |
| Code monolithique | Architecture modulaire |
| Pas de requirements.txt | requirements.txt complet |

#### Détail des modifications

##### 1. **Imports des données** (Cellule 5 → `src/data_loader.py`)

**Avant:**
```python
BASE_URL = "https://raw.githubusercontent.com/.../main/data"
clinical_train = pd.read_csv(f"{BASE_URL}/clinical_train.csv")
```

**Après:**
```python
from src.data_loader import load_all_data
clinical_train, clinical_test, molecular_train, molecular_test, y_train = load_all_data()
```

##### 2. **Feature Engineering** (Cellules 10-11 → `src/features.py`)

**Avant:** Fonction `build_molecular_features()` définie dans le notebook (50+ lignes)

**Après:** Module dédié avec fonctions réutilisables
```python
from src.features import build_molecular_features, get_feature_columns
```

##### 3. **Preprocessing** (Cellule 18 → `src/preprocessing.py`)

**Avant:** ColumnTransformer défini inline avec 30+ lignes

**Après:**
```python
from src.preprocessing import get_default_preprocessor
preprocess = get_default_preprocessor(feature_cols)
```

##### 4. **Évaluation** (Cellules 27-28 → `src/evaluation.py`)

**Avant:** Fonctions `to_sksurv_y()`, `ipcw_cindex()` dans le notebook

**Après:** Module avec grid search et permutation importance
```python
from src.evaluation import ipcw_cindex, grid_search_survival, permutation_importance_survival
```

##### 5. **Modèles** (Cellules 30-37 → `src/models.py`)

**Avant:** Code inline pour chaque modèle

**Après:** Classes et factories
```python
from src.models import BaselineRiskModel, ClusteringRiskModel, create_rsf_model
```

##### 6. **Optimisation Numba** (Nouveau → `src/optimization.py`)

Fonctions JIT-compilées pour les calculs lourds:
- `fast_cindex()` — C-index parallélisé
- `fast_vaf_stats()` — Statistiques VAF par patient
- `fast_pairwise_euclidean()` — Distances euclidiennes
- `fast_aggregate_by_id()` — Agrégation parallèle

##### 7. **Visualisation** (Nouveau → `src/visualization.py`)

Fonctions de plotting standardisées:
- `plot_survival_distribution()`
- `plot_feature_importance()`
- `plot_model_comparison()`
- `plot_cluster_survival()`

##### 8. **Cellules supprimées**

Les cellules de debugging suivantes ont été retirées:
- Vérification de types intermédiaires
- Tests de preprocessing
- Cellules vides

#### Avantages de la nouvelle architecture

| Aspect | Amélioration |
|--------|--------------|
| **Maintenabilité** | Code modulaire, facile à modifier |
| **Réutilisabilité** | Modules importables dans d'autres projets |
| **Performance** | Optimisations Numba pour calculs intensifs |
| **Testabilité** | Fonctions isolées, faciles à tester |
| **Reproductibilité** | Configuration centralisée dans `config.py` |

---

## 📚 Références

- [scikit-survival Documentation](https://scikit-survival.readthedocs.io/)
- [IPCW C-index](https://scikit-survival.readthedocs.io/en/stable/api/generated/sksurv.metrics.concordance_index_ipcw.html)
- [Random Survival Forests](https://arxiv.org/abs/0811.1645)
- [Cox Proportional Hazards](https://en.wikipedia.org/wiki/Proportional_hazards_model)

---

## 📄 Licence

Ce projet est développé dans le cadre du QRT Data Challenge 2024 en partenariat avec l'Institut Gustave Roussy.

---

<p align="center">
  <b>QRT Data Challenge 2024</b><br>
  En partenariat avec l'Institut Gustave Roussy
</p>