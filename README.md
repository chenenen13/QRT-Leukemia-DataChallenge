# QRT Data Challenge 2024 — Leukemia Risk Prediction

<p align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/4/4d/Logo-gustave-roussy.jpg/1200px-Logo-gustave-roussy.jpg" alt="Gustave Roussy" width="200"/>
  <img src="https://upload.wikimedia.org/wikipedia/en/thumb/3/3f/Qube_Research_%26_Technologies_Logo.svg/1200px-Qube_Research_%26_Technologies_Logo.svg.png" alt="QRT" width="150" style="margin-left: 20px;"/>
</p>

> **Objectif**: Prédire le risque de décès pour des patients atteints de leucémie myéloïde en utilisant des données cliniques et moléculaires.

## 🏆 Résultats Actuels

| Modèle | IPCW C-index | Status |
|--------|--------------|--------|
| **Gradient Boosting Survival** | **0.7111** | ✅ Meilleur modèle |
| Random Survival Forest | 0.7040 | ✅ Testé |
| Baseline (Ridge) | 0.6537 | ✅ Référence |
| KMeans Clustering | 0.6182 | ✅ Testé |
| Challenge Winner | 0.7744 | 🎯 Objectif |

> **Gap à combler**: -0.063 (~6%) pour atteindre le score du winner

## 📋 Table des Matières

- [Résultats Actuels](#-résultats-actuels)
- [Installation Rapide](#-installation-rapide)
- [Utilisation](#-utilisation)
- [Structure du Projet](#-structure-du-projet)
- [Méthodologie](#-méthodologie)
- [Modèles Implémentés](#-modèles-implémentés)
- [Expériences en Cours](#-expériences-en-cours)
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

### Option 1: Voir le rapport final

```bash
# Ouvrir le notebook principal (résultats documentés)
jupyter notebook main.ipynb
```

Ce notebook contient:
- ✅ Analyse exploratoire complète
- ✅ Tous les modèles testés (Baseline → GBSA)
- ✅ Visualisations et interprétations
- ✅ Génération du fichier de soumission

### Option 2: Tester des améliorations

```bash
# Ouvrir le notebook d'expérimentation
jupyter notebook experiments.ipynb
```

Ce notebook contient:
- 🧪 Tests d'ensemble RSF + GBSA
- 🧪 Tuning GBSA avec grid search étendu
- 🧪 Tests avec plus de features génétiques
- 🧪 Features de co-mutations

### Option 3: Utiliser les modules Python

```python
from src.data_loader import load_all_data
from src.features import build_molecular_features
from src.models import create_rsf_model
from src.evaluation import ipcw_cindex
from sksurv.ensemble import GradientBoostingSurvivalAnalysis

# Charger les données
clinical_train, clinical_test, molecular_train, molecular_test, y_train = load_all_data()

# Feature engineering
mol_features = build_molecular_features(molecular_train)

# Meilleur modèle : Gradient Boosting Survival
gbsa = GradientBoostingSurvivalAnalysis(n_estimators=200, learning_rate=0.1, max_depth=3)
```

### Option 4: Notebook de développement (legacy)

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
├── 📓 main.ipynb                   # Rapport principal (résultats finaux)
├── 📓 experiments.ipynb            # Notebook d'expérimentation (tests avancés)
├── 📓 DataChallenge_ML.ipynb       # Notebook de développement (legacy)
├── 📓 Benchmark_nqBJ7fO.ipynb      # Benchmark fourni par QRT
│
├── 📄 requirements.txt             # Dépendances Python
├── 📄 submission.csv               # Fichier de soumission
└── 📄 README.md                    # Ce fichier
```

### Notebooks

| Notebook | Description | Quand l'utiliser |
|----------|-------------|------------------|
| **main.ipynb** | Rapport final documenté avec tous les résultats | Voir les résultats finaux, générer la soumission |
| **experiments.ipynb** | Tests d'amélioration (ensemble, tuning, etc.) | Tester de nouvelles idées |
| DataChallenge_ML.ipynb | Notebook original de développement | Référence historique |

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

1. **Baseline (Ridge Regression)**: Régression sur OS_YEARS, ignore la censure → **0.6537**
2. **KMeans Clustering**: Non-supervisé, risque par médiane de cluster → **0.6182**
3. **Random Survival Forest**: Gère la censure, hyperparamètres optimisés → **0.7040**
4. **Gradient Boosting Survival**: Meilleur modèle actuel → **0.7111** ✅

### Métrique

**IPCW C-index** (τ = 7 ans): Mesure la capacité à ordonner correctement les paires de patients selon leur survie, en tenant compte de la censure à droite.

$$C = \frac{\text{Paires Concordantes}}{\text{Paires Comparables}}$$

- **C = 1**: Classement parfait
- **C = 0.5**: Modèle aléatoire

---

## 🧪 Modèles Implémentés

### 1. Baseline Ridge Regression (0.6537)

Régression linéaire régularisée sur `OS_YEARS`. Simple mais **ignore la censure** (patients encore en vie traités comme décédés).

```python
from src.models import BaselineRiskModel
baseline = BaselineRiskModel(preprocessor=preprocess, alpha=1.0)
```

### 2. KMeans Clustering (0.6182)

Approche non-supervisée : cluster les patients, puis assigne un risque basé sur la survie médiane de chaque cluster.

```python
from src.models import ClusteringRiskModel
cluster_model = ClusteringRiskModel(preprocessor=preprocess, n_clusters=5)
```

### 3. Random Survival Forest (0.7040)

Forêt aléatoire adaptée à l'analyse de survie. Gère correctement la **censure à droite**.

**Hyperparamètres optimisés :**
- `n_estimators`: 200-300
- `min_samples_leaf`: 10-20
- `max_features`: 0.5

```python
from sksurv.ensemble import RandomSurvivalForest
rsf = RandomSurvivalForest(n_estimators=200, min_samples_leaf=20, random_state=42)
```

### 4. Gradient Boosting Survival ⭐ (0.7111)

**Meilleur modèle actuel.** Gradient boosting adapté à la survie, souvent meilleur que RSF.

**Hyperparamètres :**
- `n_estimators`: 200
- `learning_rate`: 0.1
- `max_depth`: 3

```python
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
gbsa = GradientBoostingSurvivalAnalysis(n_estimators=200, learning_rate=0.1, max_depth=3)
```

---

## 🔬 Expériences en Cours

Voir **experiments.ipynb** pour les tests d'amélioration.

### Expériences déjà implémentées

| # | Expérience | Description | Status |
|---|------------|-------------|--------|
| 1 | Ensemble RSF + GBSA | Moyenne pondérée des deux modèles | 🧪 À tester |
| 2 | GBSA Tuning | Grid search plus large (n_estimators, learning_rate, max_depth) | 🧪 À tester |
| 3 | Plus de gènes | Augmenter TOP_GENES de 30 à 50 | 🧪 À tester |
| 4 | Co-mutations | Features d'interaction gène-gène (ex: TP53 + RUNX1) | 🧪 À tester |

### Idées à explorer

- [ ] **CoxPH avec ElasticNet** — Modèle de Cox régularisé
- [ ] **Parser CYTOGENETICS** — Extraire del(5q), -7, complex karyotype
- [ ] **Stacking** — Meta-learner sur les prédictions des modèles
- [ ] **XGBoost AFT** — Accelerated Failure Time avec XGBoost
- [ ] **DeepSurv** — Réseau de neurones pour la survie

---

## 📈 Résultats Détaillés

| Modèle | IPCW C-index | Gap vs Winner | Commentaire |
|--------|--------------|---------------|-------------|
| **Gradient Boosting Surv** | **0.7111** | -0.063 | ✅ Meilleur modèle |
| Random Survival Forest | 0.7040 | -0.070 | Bon modèle de survie |
| Baseline (Ridge) | 0.6537 | -0.121 | Ignore la censure |
| KMeans Clustering | 0.6182 | -0.156 | Non-supervisé |
| Challenge Winner | 0.7744 | — | 🎯 Objectif |

### Progression des scores

```
Baseline      ████████████████████████████░░░░░░░░░░  0.6537
KMeans        ████████████████████████░░░░░░░░░░░░░░  0.6182
RSF           ████████████████████████████████░░░░░░  0.7040
GBSA          █████████████████████████████████░░░░░  0.7111 ← Actuel
Winner        ████████████████████████████████████░░  0.7744 ← Objectif
```

### Features les plus importantes

| Rang | Feature | Description | Impact |
|------|---------|-------------|--------|
| 1 | `BM_BLAST` | Blastes moelle osseuse (%) | Très élevé |
| 2 | `PLT` | Plaquettes (Giga/L) | Élevé |
| 3 | `HB` | Hémoglobine (g/dL) | Élevé |
| 4 | `n_mut` | Nombre total de mutations | Modéré |
| 5 | `vaf_mean` | VAF moyen des mutations | Modéré |
| 6 | `WBC` | Globules blancs (Giga/L) | Modéré |
| 7 | `GENE__TP53` | Mutation TP53 (binaire) | Cliniquement important |

---

## 📝 Historique des Modifications

### Version 2.1 (Janvier 2026) — Gradient Boosting + Expériences

#### Nouveautés

- ✅ **Gradient Boosting Survival** — Nouveau meilleur modèle (0.7111 vs 0.7040 RSF)
- ✅ **experiments.ipynb** — Notebook dédié aux expériences
- ✅ **Optimisation Grid Search** — Réduction de 72 à 16 fits, mode fast
- ✅ **Fix alignement colonnes** — Correction du bug KeyError sur test set

#### Scores atteints

| Étape | Score | Amélioration |
|-------|-------|--------------|
| v1 Baseline | 0.6537 | — |
| v2 RSF | 0.7040 | +0.050 |
| v2.1 GBSA | **0.7111** | +0.007 |

### Version 2.0 — Restructuration complète

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
- [Gradient Boosting Survival](https://scikit-survival.readthedocs.io/en/stable/api/generated/sksurv.ensemble.GradientBoostingSurvivalAnalysis.html)
- [Cox Proportional Hazards](https://en.wikipedia.org/wiki/Proportional_hazards_model)

---

## 🤝 Contribution

Pour contribuer au projet :

1. Ouvrir `experiments.ipynb` et tester une nouvelle idée
2. Si le score s'améliore, reporter les résultats dans `main.ipynb`
3. Mettre à jour ce README avec les nouveaux scores

---

## 📄 Licence

Ce projet est développé dans le cadre du QRT Data Challenge 2024 en partenariat avec l'Institut Gustave Roussy.

---

<p align="center">
  <b>QRT Data Challenge 2024</b><br>
  En partenariat avec l'Institut Gustave Roussy<br><br>
  <i>Score actuel : 0.7111 | Objectif : 0.7744</i>
</p>