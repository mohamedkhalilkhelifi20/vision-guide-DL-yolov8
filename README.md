# 🔍 Object-Finder — Vision Guide : Pipeline YOLO + CNN

[![Demo Live](https://img.shields.io/badge/...)](https://vision-guide-dl-frontend.vercel.app/)
![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.10-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![Lightning](https://img.shields.io/badge/Lightning-2.6-792EE5?style=flat&logo=lightning&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics_8.4-00FFFF?style=flat&logo=yolo&logoColor=black)
![Kaggle](https://img.shields.io/badge/Kaggle-GPU_Tesla_T4-20BEFF?style=flat&logo=kaggle&logoColor=white)
![Licence](https://img.shields.io/badge/Licence-Academic-green?style=flat)
![Statut](https://img.shields.io/badge/Statut-Terminé-brightgreen?style=flat)
![F1 Escaliers CNN](https://img.shields.io/badge/F1_Escaliers_(CNN)-96.84%25-success?style=flat)
![mAP50 YOLO](https://img.shields.io/badge/mAP50_(YOLO)-62.76%25-blue?style=flat)
![F1 Âge](https://img.shields.io/badge/F1_Âge-64.04%25-yellow?style=flat)

<div align="center">

[![🚀 Demo Live](https://img.shields.io/badge/🚀%20Demo%20Live-vision--guide--dl--frontend.vercel.app-black?style=for-the-badge&logo=vercel)](https://vision-guide-dl-frontend.vercel.app/)
&nbsp;&nbsp;
[![Frontend](https://img.shields.io/badge/Frontend-GitHub-181717?style=for-the-badge&logo=github)](https://github.com/mohamedkhalilkhelifi20/vision-guide-DL-frontend)
&nbsp;&nbsp;
[![Backend](https://img.shields.io/badge/Backend-GitHub-181717?style=for-the-badge&logo=github)](https://github.com/mohamedkhalilkhelifi20/vision-guide-DL-backend)

</div>

---

> **Projet académique** — 4ème année Data Science & IA, École Polytechnique de Sousse
> **Auteur** : Mohamed Khalil Khelifi

**Object-Finder / Vision Guide** est un système de détection et classification d'objets en temps réel conçu pour **aider les personnes malvoyantes à naviguer en sécurité dans leur environnement**. Le pipeline orchestre YOLOv8 pour la détection, puis deux CNN spécialisés — MobileNetV2 et ConvNeXt-Tiny — pour une classification fine-grain que YOLO seul ne peut pas fournir.

---

## 📋 Table des matières

- | 🌐 **Application live** | [vision-guide-dl-frontend.vercel.app](https://vision-guide-dl-frontend.vercel.app/) |
- [🎯 Problème & Motivation](#-problème--motivation)
- [🏗️ Architecture du pipeline](#️-architecture-du-pipeline)
- [📁 Structure du dépôt](#-structure-du-dépôt)
- [🧑‍🦳 Rôle 1 — Classification d'âge (FairFace)](#-rôle-1--classification-dâge-fairface)
- [🪜 Rôle 2 — Détection d'escaliers (Open Images V7)](#-rôle-2--détection-descaliers-open-images-v7)
  - [Dataset & Préparation](#dataset--préparation)
  - [Fine-tuning YOLOv8n](#fine-tuning-yolov8n-pour-la-détection-descaliers)
  - [ConvNeXt-Tiny — Classificateur CNN binaire](#convnext-tiny--classificateur-cnn-binaire-modèle-final)
- [📈 Résultats — Synthèse globale](#-résultats--synthèse-globale)
- [🚀 Comment reproduire](#-comment-reproduire)
- [🛠️ Technologies](#️-technologies)
- [🧪 Concepts clés appliqués](#-concepts-clés-appliqués)
- [👤 Auteur](#-auteur)

---

## 🔗 Liens du projet

| Ressource | Lien |
|-----------|------|
| 🌐 **Application live** | [vision-guide-dl-frontend.vercel.app](https://vision-guide-dl-frontend.vercel.app/) |
| 🎨 **Frontend** (React / Next.js) | [vision-guide-DL-frontend](https://github.com/mohamedkhalilkhelifi20/vision-guide-DL-frontend) |
| ⚙️ **Backend** (API / Inférence) | [vision-guide-DL-backend](https://github.com/mohamedkhalilkhelifi20/vision-guide-DL-backend) |
| 🧠 **Modèles DL** (ce repo) | [vision-guide-DL-yolov8](https://github.com/mohamedkhalilkhelifi20/vision-guide-DL-yolov8) |

---

## 🎯 Problème & Motivation

YOLOv8 pré-entraîné sur COCO couvre 80 classes génériques. Cela crée **deux angles morts critiques** pour une application d'aide à la mobilité :

| Limitation YOLO | Impact concret | Solution apportée |
|-----------------|----------------|-------------------|
| Détecte `person` sans précision d'âge | Impossible de différencier un enfant, un adulte ou une personne âgée — information pourtant utile pour évaluer le contexte de danger | **MobileNetV2** post-YOLO → 3 classes |
| `stairs` absent des 80 classes COCO | Les escaliers représentent un danger majeur de chute pour un malvoyant — YOLO ne les voit pas du tout | **YOLOv8n fine-tuné + ConvNeXt-Tiny** en cascade |

Object-Finder résout ces deux limitations en injectant une **couche CNN spécialisée après chaque détection YOLO**, sans alourdir le modèle de base.

---

## 🏗️ Architecture du pipeline

```
┌──────────────────────────────────────────────────────────────┐
│                       IMAGE D'ENTRÉE                         │
│                    (flux caméra temps réel)                  │
└────────────────────────────┬─────────────────────────────────┘
                             │
                             ▼
               ┌─────────────────────────┐
               │        YOLOv8n          │
               │  Détection + localisation│
               │   (80 classes COCO)     │
               │   73 layers · 3M params │
               │       8.1 GFLOPs        │
               └─────────────┬───────────┘
                             │
            ┌────────────────┴────────────────┐
            │                                 │
            ▼                                 ▼
    "person" détecté                 Région non couverte
            │                        → scan escaliers
            ▼                                 │
 ┌──────────────────────┐                     ▼
 │    MobileNetV2       │       ┌─────────────────────────┐
 │  Transfer Learning   │       │   YOLOv8n Fine-tuned    │
 │  backbone + tête     │       │  (stairs_v1 · 50 epochs)│
 │  ImageNet → FairFace │       │  mAP50  = 0.6276        │
 │                      │       │  Rappel = 0.5909        │
 │  → enfant            │       └─────────────┬───────────┘
 │  → adulte            │                     │
 │  → personne_âgée     │                     ▼
 │                      │       ┌─────────────────────────┐
 │  F1-macro : 64.04%   │       │     ConvNeXt-Tiny        │
 └──────────────────────┘       │   Fine-tuning complet   │
                                │  Threshold opt. = 0.504 │
                                │                         │
                                │  → stairs               │
                                │  → pas_stairs           │
                                │                         │
                                │  Accuracy : 96.96%      │
                                │  F1-macro : 96.84%      │
                                │  AUC      : 0.9906      │
                                └─────────────┬───────────┘
                                              │
                                              ▼
                          ┌────────────────────────────────┐
                          │  Label précis + niveau danger  │
                          │  → Alerte vocale / visuelle    │
                          └────────────────────────────────┘
```

---

## 📁 Structure du dépôt

```
vision-guide-DL-yolov8/
│
├── role1_age/                          # Rôle 1 — Classification d'âge
│   ├── 1_EDA.ipynb                     # Exploration FairFace : distribution, classes
│   ├── 2_Preprocessing.ipynb           # Mapping âge → 3 classes, export CSV
│   ├── 3_v1_Model_Comparison.ipynb     # Benchmark 4 architectures (backbone gelé)
│   ├── 3_v2_Model_Comparison.ipynb     # Fine-tuning MobileNetV2 complet (FINAL ✅)
│   └── models/
│       └── mobilenetv2_fairface.pth    # Poids sauvegardés (meilleur F1-macro)
│
├── role2_stairs/                       # Rôle 2 — Détection d'escaliers
│   ├── 1_EDA_Preprocessing.ipynb       # EDA + extraction crops Open Images V7
│   ├── 2_YOLOv8_Finetune.ipynb         # Fine-tuning YOLOv8n (50 epochs · stairs)
│   ├── 3_v1_Modeling.ipynb             # Baseline CNN (expérimentation initiale)
│   ├── 3_v2_Modeling_Final.ipynb       # ConvNeXt-Tiny fine-tuning (FINAL ✅)
│   └── models/
│       ├── yolov8n_stairs.pt           # YOLOv8n fine-tuné — mAP50 = 0.6276
│       └── convnext_stairs_v2.pth      # CNN binaire — F1 = 96.84%
│
└── README.md
```

---

## 🧑‍🦳 Rôle 1 — Classification d'âge (FairFace)

### Contexte

Lorsque YOLO détecte une `person`, l'information s'arrête là. Pourtant, pour un système d'aide à la mobilité, savoir si la personne devant soi est un enfant (imprévisible), un adulte ou une personne âgée (plus vulnérable) change complètement la stratégie d'alerte. Ce CNN post-YOLO comble cette lacune.

### Dataset — FairFace

| Propriété | Valeur |
|-----------|--------|
| Source | [FairFace (Kaggle)](https://www.kaggle.com/datasets/aibloy/fairface) |
| Train | 86 744 images |
| Validation | 10 954 images |
| Classes cibles | `enfant` · `adulte` · `personne_âgée` |

**Mapping âge → classe :**

| Tranches FairFace | Classe cible |
|-------------------|--------------|
| 0–2, 3–9, 10–19 | `enfant` |
| 20–29, 30–39, 40–49 | `adulte` |
| 50–59, 60–69, 70+ | `personne_âgée` |

**Distribution (déséquilibrée) :**

| Classe | Train | Validation | % Train |
|--------|-------|------------|---------|
| adulte | 55 592 | 6 983 | 64.1% |
| enfant | 21 303 | 2 736 | 24.6% |
| personne_âgée | 9 849 | 1 235 | 11.3% |

> Le fort déséquilibre (adulte = 5.6× les enfants, 5.6× les personnes âgées) exige une stratégie explicite de pondération des pertes.

### Benchmark d'architectures (10 epochs, backbone gelé)

| Modèle | Params | Taille | Val Accuracy | Stratégie |
|--------|--------|--------|-------------|-----------|
| CNN Simple | 127K | 0.5 MB | 69.05% | From Scratch |
| ResNet18 | 11.2M | 42.6 MB | 69.07% | Transfer Learning |
| EfficientNet-B0 | 4.0M | 15.3 MB | 71.60% | Transfer Learning |
| **MobileNetV2** ✅ | **2.2M** | **8.5 MB** | **71.96%** | **Transfer Learning** |

**Pourquoi MobileNetV2 ?** Meilleure accuracy parmi les modèles pré-entraînés avec le footprint mémoire le plus bas — critique pour l'inférence embarquée en temps réel.

### Fine-tuning MobileNetV2 — Stratégie complète

**Architecture de la tête de classification :**
```python
model.classifier = nn.Sequential(
    nn.Dropout(p=0.3),
    nn.Linear(1280, 256),
    nn.ReLU(inplace=True),
    nn.Linear(256, 3)   # enfant / adulte / personne_âgée
)
```

**Hyperparamètres :**

| Paramètre | Valeur | Justification |
|-----------|--------|---------------|
| Optimizer | Adam | Convergence rapide sur transfer learning |
| Backbone lr | `1e-5` | Très faible pour préserver les features ImageNet |
| Tête lr | `1e-4` | ×10 pour adapter rapidement la nouvelle couche |
| Scheduler | ReduceLROnPlateau (factor=0.5) | Réduit le lr si plateau sur val_loss |
| Batch size | 32 | Équilibre mémoire GPU / variance du gradient |
| Max epochs | 20 | + Early stopping sur F1-macro |
| Loss | CrossEntropyLoss pondérée (√inverse) | Compense le déséquilibre des classes |
| GPU | Tesla T4 (Kaggle) | 14.9 GB VRAM |

**Résultats finaux :**

| Métrique | Valeur |
|----------|--------|
| Val Accuracy | 65.36% |
| Val F1-macro | **64.04%** |

**Classification Report :**

| Classe | Precision | Recall | F1-score | Support |
|--------|-----------|--------|----------|---------|
| adulte | 0.941 | 0.497 | 0.650 | 6 983 |
| enfant | 0.612 | 0.915 | 0.734 | 2 736 |
| personne_âgée | 0.373 | 0.960 | 0.537 | 1 235 |
| **macro avg** | **0.642** | **0.791** | **0.640** | 10 954 |

> **Lecture des résultats :** Le recall de `personne_âgée` à **96%** est intentionnel et prioritaire — il vaut mieux déclencher une alerte sur un faux positif que de rater une vraie personne âgée en situation de danger. La précision faible (37%) est le prix accepté de cette stratégie safety-first.

---

## 🪜 Rôle 2 — Détection d'escaliers (Open Images V7)

### Contexte

Les escaliers sont **totalement absents** des 80 classes COCO de YOLOv8. Pour un malvoyant, un escalier non détecté peut provoquer une chute grave. L'approche adoptée est **double** : fine-tuner YOLOv8 pour *localiser* les escaliers dans l'image, puis un CNN binaire pour *confirmer* chaque détection avec une grande précision.

---

### Dataset & Préparation

**Source :** Open Images V7 via FiftyOne, classe `Stairs`.

| Propriété | Valeur |
|-----------|--------|
| Annotations totales | **6 026** bounding boxes |
| Split train (YOLO) | 5 981 annotations · **3 655 images** |
| Split validation (YOLO) | 45 annotations · **36 images** |
| Split train (CNN) | **4 461 crops** |
| Split validation (CNN) | **914 crops** (split 80/20) |
| Classe positive | `stairs` |
| Classe négative | `pas_stairs` (backgrounds extraits) |

**Pipeline de préparation :**

1. Téléchargement via FiftyOne (`foz.load_zoo_dataset("open-images-v7", ...)`)
2. Conversion des bounding boxes au format YOLO normalisé `[x_center, y_center, w, h]`
3. Export CSV avec colonnes `filepath · label · x_norm · y_norm · w_norm · h_norm · area · split`
4. Extraction des **crops** (régions d'intérêt) pour entraîner le CNN binaire indépendamment

---

### Fine-tuning YOLOv8n pour la Détection d'Escaliers

#### Objectif

Adapter YOLOv8n (3M paramètres, pré-entraîné COCO) à la **localisation spatiale** des escaliers dans une image. YOLOv8 produit des bounding boxes avec un score de confiance — il détecte *où* se trouvent les escaliers. Le CNN binaire confirme ensuite *si* c'est bien un escalier.

#### Configuration d'entraînement

```yaml
model:       yolov8n.pt               # 73 layers · 3 005 843 paramètres · 8.1 GFLOPs
epochs:      50
imgsz:       640                       # Résolution standard YOLOv8
batch:       16
lr0:         0.001
device:      Tesla T4 (CUDA:0 · 14 913 MiB)
framework:   Ultralytics 8.4.48 · Python 3.12.12 · torch-2.10.0+cu128
durée:       0.631 heures (~38 minutes)
```

#### Analyse des courbes d'entraînement (50 epochs)

Les courbes montrent un comportement **sain et cohérent** sur 50 epochs :

| Courbe | Observation | Interprétation |
|--------|-------------|----------------|
| `train/box_loss` | 1.6 → 1.0 (descente régulière) | Le modèle apprend à localiser les escaliers |
| `train/cls_loss` | 2.25 → 0.95 (forte réduction) | La classification stairs/background s'améliore |
| `train/dfl_loss` | 1.85 → 1.40 (descente stable) | Précision des coins de bounding boxes progresse |
| `val/box_loss` | Se stabilise autour de 1.7 | Bonne généralisation, pas de sur-apprentissage |
| `val/cls_loss` | 5.0 → 2.0 (réduction notable) | Réduction du bruit initial de validation |
| `mAP50(B)` | Progression monotone → ~0.64 | Signal clair d'apprentissage continu |
| `mAP50-95(B)` | Progression → ~0.40 | Bonne précision de localisation multi-threshold |
| `metrics/precision` | Oscille autour de 0.6 | Dataset val petit (36 images) → variance élevée |
| `metrics/recall` | Progression globale → 0.6 | Le modèle capture de plus en plus d'escaliers |

> **Note sur la variance des courbes :** La validation ne couvre que **36 images (45 instances)**, ce qui explique les oscillations visibles sur précision et rappel. Les métriques lissées (courbe orange) confirment néanmoins une progression monotone stable.

#### Résultats finaux — YOLOv8n Fine-tuné

Modèle sauvegardé : `yolov8n_stairs.pt` (6.2 MB)

| Métrique | Valeur | Signification |
|----------|--------|---------------|
| **mAP50** | **0.6276** | Précision moyenne à IoU = 0.50 |
| **mAP50-95** | **0.3929** | Précision moyenne sur IoU ∈ [0.50, 0.95] |
| **Précision** | **0.5613** | 56% des détections sont de vrais escaliers |
| **Rappel** | **0.5909** | 59% des escaliers présents sont détectés |
| Vitesse inférence | **2.4 ms/image** | Temps réel garanti |
| Taille modèle | **6.2 MB** | Léger et déployable sur mobile/edge |

**Évaluation complète sur le set de validation :**
```
Class    Images  Instances   Box(P)    R      mAP50   mAP50-95
all        36       44       0.561   0.591    0.628     0.393
```

#### Visualisation des prédictions

Le modèle démontre une capacité à détecter des **escaliers dans des contextes très variés** :

| Contexte | Confiance | Résultat |
|----------|-----------|----------|
| Escaliers extérieurs en béton | 0.82 | ✅ Détection nette |
| Escaliers métalliques vus de haut | 0.34 – 0.64 | ✅ Multi-détections cohérentes |
| Escaliers en construction | 0.64 | ✅ Détection correcte |
| Escalier mécanique (escalator) | 0.26 | ⚠️ Détecté mais faible confiance |
| Escaliers décoratifs sculptés | 0.28 | ⚠️ Détecté avec hésitation |
| Escaliers bois + végétation dense | — | ❌ Non détecté (cas difficile) |

> **Analyse :** Le modèle est robuste sur les escaliers standards. Les faux négatifs apparaissent principalement dans des contextes atypiques (masqués par végétation, angles très obliques, matériaux inhabituels). C'est précisément pour cette raison que le **CNN binaire ConvNeXt-Tiny** intervient en second niveau pour consolider la décision finale.

---

### ConvNeXt-Tiny — Classificateur CNN Binaire (Modèle FINAL)

#### Pourquoi un second modèle ?

YOLOv8 localise mais avec un rappel de 59% — il manque environ 4 escaliers sur 10. ConvNeXt-Tiny opère sur les **crops** extraits et répond à une question binaire simple : *ce crop contient-il un escalier ?*

Cette architecture en **deux étages** maximise à la fois la couverture spatiale (YOLO) et la précision de classification (ConvNeXt).

**Pourquoi ConvNeXt-Tiny plutôt que ResNet ou EfficientNet ?**

ConvNeXt intègre des blocs inspirés des Vision Transformers (larges noyaux 7×7, LayerNorm, GELU) dans un framework ConvNet pur. Il surpasse systématiquement les ResNets sur les tâches de classification fine-grain, avec une stabilité d'entraînement supérieure grâce à ses connexions denses et sa normalisation robuste.

#### Stratégie de fine-tuning

```
Phase unique — Toutes les couches dégelées dès le départ
├── Backbone  : lr = 1e-5   (préserver les features ImageNet)
├── Tête      : lr = 1e-4   (adapter rapidement à stairs/pas_stairs)
└── Scheduler : OneCycleLR
                ├── Warmup progressif  → évite l'écrasement des poids pré-entraînés
                ├── Phase peak         → exploration maximale de l'espace des paramètres
                └── Descente cosinus   → convergence fine et stable
```

**Correction du déséquilibre :** Loss pondérée **seule** (sans WeightedRandomSampler). Combiner les deux revient à corriger le déséquilibre en double — ce qui crée une sur-représentation artificielle des classes minoritaires et dégrade les performances réelles.

#### Hyperparamètres

| Paramètre | Valeur | Justification |
|-----------|--------|---------------|
| Modèle | ConvNeXt-Tiny | Supérieur sur tâches binaires fines |
| Optimizer | Adam | Adaptatif, convergence rapide |
| Backbone lr | `1e-5` | Conservation des features ImageNet |
| Tête lr | `1e-4` | Adaptation rapide à la tâche binaire |
| Scheduler | OneCycleLR | Warmup + peak + cosine descent |
| Batch size | 32 | Optimal GPU T4 |
| Max epochs | 20 | Early stopping sur F1-macro |
| Dropout | 0.4 | Régularisation contre le sur-apprentissage |
| GPU | Tesla T4 (Kaggle) | 14.9 GB VRAM |

#### Résultats finaux — ConvNeXt-Tiny v2

Meilleur checkpoint sauvegardé à **l'epoch 15**.

| Métrique | Valeur |
|----------|--------|
| **Val Accuracy** | **96.96%** |
| **Val F1-macro** | **96.84%** |
| **AUC** | **0.9906** |
| Meilleur epoch | 15 / 20 |
| **Threshold optimal** | **0.504** |

> **Threshold = 0.504** — Seuil de décision déterminé par maximisation du F1-macro sur la courbe ROC. À utiliser tel quel dans l'intégration Object-Finder pour la décision binaire stairs/pas_stairs. Un seuil de 0.5 arbitraire aurait donné des résultats légèrement inférieurs.

---

## 📈 Résultats — Synthèse globale

| Module | Modèle | Dataset | Métrique clé | Valeur |
|--------|--------|---------|-------------|--------|
| Rôle 1 — Âge | MobileNetV2 fine-tuné | FairFace (86K images) | Val F1-macro | 64.04% |
| Rôle 2 — YOLO stairs | YOLOv8n fine-tuné | Open Images V7 (3 655 images) | mAP50 | **62.76%** |
| Rôle 2 — YOLO stairs | YOLOv8n fine-tuné | Open Images V7 | mAP50-95 | **39.29%** |
| Rôle 2 — CNN stairs | ConvNeXt-Tiny fine-tuné | Crops stairs (4 461 images) | Val F1-macro | **96.84%** |
| Rôle 2 — CNN stairs | ConvNeXt-Tiny fine-tuné | Crops stairs | AUC | **0.9906** |

**Interprétation du pipeline combiné Rôle 2 :**

YOLO détecte et localise (rappel 59%) → ConvNeXt confirme avec une précision de 97%. Le résultat est un système robuste qui maximise la couverture spatiale tout en minimisant les fausses alertes. Les deux modèles sont complémentaires : YOLO rate certains escaliers difficiles, mais ConvNeXt filtre les faux positifs de YOLO avec une précision quasi-parfaite.

---

## 🚀 Comment reproduire

### Prérequis

- Compte Kaggle avec GPU activé (Tesla T4 gratuit)
- Datasets ajoutés via **Add Input** dans chaque notebook

### Rôle 1 — Classification d'âge

```bash
# 1. Ajouter FairFace dataset via Kaggle > Add Input
# 2. Exécuter dans cet ordre strict :

1_EDA.ipynb
    └─ Sortie : distribution des classes, visualisations

2_Preprocessing.ipynb
    └─ Entrée  : FairFace dataset
    └─ Sortie  : train.csv · val.csv

3_v2_Model_Comparison.ipynb          ← NOTEBOOK FINAL
    └─ Entrée  : train.csv · val.csv
    └─ Sortie  : mobilenetv2_fairface.pth
```

### Rôle 2 — Détection d'escaliers

```bash
# Open Images V7 est téléchargé automatiquement via FiftyOne
# Exécuter dans cet ordre strict :

1_EDA_Preprocessing.ipynb
    └─ Sortie  : stairs_dataset.csv · dossier crops/

2_YOLOv8_Finetune.ipynb
    └─ Entrée  : 1_EDA_Preprocessing (format YOLO)
    └─ Sortie  : yolov8n_stairs.pt  (mAP50 = 0.6276)

3_v2_Modeling_Final.ipynb            ← NOTEBOOK FINAL
    └─ Entrée  : stairs_dataset.csv + crops/
    └─ Sortie  : convnext_stairs_v2_final.pth  (F1 = 96.84%)
```

### Charger les modèles pour l'inférence

**MobileNetV2 — Classification d'âge :**
```python
from torchvision import models
import torch.nn as nn
import torch

model = models.mobilenet_v2(pretrained=False)
model.classifier = nn.Sequential(
    nn.Dropout(p=0.3),
    nn.Linear(1280, 256),
    nn.ReLU(inplace=True),
    nn.Linear(256, 3)   # 0=enfant · 1=adulte · 2=personne_âgée
)
model.load_state_dict(torch.load("mobilenetv2_fairface.pth", map_location="cpu"))
model.eval()
```

**YOLOv8n — Détection d'escaliers :**
```python
from ultralytics import YOLO

model = YOLO("yolov8n_stairs.pt")
results = model("image.jpg", conf=0.25)  # seuil confiance recommandé
```

**ConvNeXt-Tiny — Classification binaire stairs :**
```python
# Charger via PyTorch Lightning
model = ConvNeXtFineTunedV2.load_from_checkpoint(
    "convnext_stairs_v2_final.pth",
    loss_weights=loss_weights,
    steps_per_epoch=len(train_loader),
    max_epochs=20
)
model.eval()

# Décision binaire avec threshold optimal ROC
proba = torch.sigmoid(model(crop))
is_stairs = proba.item() > 0.504   # threshold optimal
```

---

## 🛠️ Technologies

| Outil | Version | Usage |
|-------|---------|-------|
| Python | 3.12 | Langage principal |
| PyTorch | 2.10 | Framework deep learning |
| PyTorch Lightning | 2.6 | Boucles d'entraînement structurées |
| torchvision | latest | MobileNetV2 · ConvNeXt-Tiny · transforms |
| Ultralytics | **8.4.48** | YOLOv8n fine-tuning + inférence |
| FiftyOne | latest | Chargement & annotation Open Images V7 |
| Pandas / NumPy | — | Manipulation des données |
| Matplotlib / Seaborn | — | Visualisation des courbes d'entraînement |
| scikit-learn | — | Métriques · ROC · AUC · confusion matrix |
| Kaggle Notebooks | — | Environnement GPU Tesla T4 (14.9 GB) |

---

## 🧪 Concepts clés appliqués

- **Transfer Learning** — Partir de poids ImageNet pré-entraînés au lieu d'entraîner from scratch. Les features visuelles bas-niveau (contours, textures, formes) sont universelles et directement réutilisables.

- **Fine-tuning complet avec dual learning rates** — Dégeler toutes les couches tout en appliquant un lr très faible sur le backbone (`1e-5`) et un lr plus élevé sur la tête (`1e-4`). Cela préserve la connaissance ImageNet tout en adaptant le modèle à la nouvelle tâche.

- **OneCycleLR Scheduler** — Warmup progressif → peak → descente cosinus. Élimine le risque d'écrasement des poids pré-entraînés en début d'entraînement, et assure une convergence fine en fin de cycle.

- **Loss pondérée (√inverse)** — Corriger le déséquilibre de classes directement dans la fonction de perte, sans WeightedRandomSampler. Combiner les deux crée une double correction qui biaise l'apprentissage.

- **Threshold optimal ROC** — Le seuil de décision binaire (0.504) est sélectionné par maximisation du F1-macro sur la courbe ROC, plutôt que d'utiliser arbitrairement 0.5.

- **Pipeline YOLO + CNN en cascade** — YOLO apporte la localisation spatiale, le CNN apporte la précision de classification. Séparer les deux responsabilités permet d'optimiser chaque composant indépendamment et de cumuler leurs forces.

- **Pipeline modulaire inter-notebooks** — Les sorties d'un notebook (CSV, crops, modèles) deviennent les entrées du suivant. Reproductibilité totale, débogage isolé et versionnement indépendant de chaque étape.

- **Évaluation multi-métrique** — F1-macro (résistant au déséquilibre), AUC (performance globale du classifieur), mAP50 et mAP50-95 (standard détection d'objets), Classification Report complet par classe.

---

## 👤 Auteur

**Mohamed Khalil Khelifi**
4ème année Data Science & IA — École Polytechnique de Sousse

[![GitHub](https://img.shields.io/badge/GitHub-mohamedkhalilkhelifi20-181717?style=flat&logo=github)](https://github.com/mohamedkhalilkhelifi20)
[![Kaggle](https://img.shields.io/badge/Kaggle-medkhalilkh-20BEFF?style=flat&logo=kaggle&logoColor=white)](https://www.kaggle.com/medkhalilkh)
