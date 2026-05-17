# 🔍 Object-Finder — Pipeline YOLO + CNN

![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.10-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF?style=flat&logo=yolo&logoColor=black)
![Kaggle](https://img.shields.io/badge/Kaggle-GPU%20T4-20BEFF?style=flat&logo=kaggle&logoColor=white)
![License](https://img.shields.io/badge/Licence-Academic-green?style=flat)
![Status](https://img.shields.io/badge/Statut-Terminé-brightgreen?style=flat)
![F1 Stairs](https://img.shields.io/badge/F1%20Escaliers-96.84%25-success?style=flat)
![F1 Age](https://img.shields.io/badge/F1%20Âge-64.04%25-yellow?style=flat)

<div align="center">

[![ Demo Live](https://img.shields.io/badge/🚀%20Demo%20Live-vision--guide--dl--frontend.vercel.app-black?style=for-the-badge&logo=vercel)](https://vision-guide-dl-frontend.vercel.app/)
&nbsp;&nbsp;
[![Frontend](https://img.shields.io/badge/Frontend-GitHub-181717?style=for-the-badge&logo=github)](https://github.com/mohamedkhalilkhelifi20/vision-guide-DL-frontend)
&nbsp;&nbsp;
[![Backend](https://img.shields.io/badge/Backend-GitHub-181717?style=for-the-badge&logo=github)](https://github.com/mohamedkhalilkhelifi20/vision-guide-DL-backend)

</div>

> **Projet académique** — 4ème année Data Science & IA, École Polytechnique de Sousse  
> **Auteur** : Mohamed Khalil Khelifi

Système de détection et classification d'objets en temps réel conçu pour **aider les personnes malvoyantes à naviguer dans leur environnement**. Le pipeline combine YOLOv8 pour la détection et deux CNN spécialisés pour une classification fine-grain.

---

## 📋 Table des matières

- [Liens du projet](#-liens-du-projet)
- [Problème](#-problème)
- [Architecture du pipeline](#️-architecture-du-pipeline)
- [Structure du dépôt](#-structure-du-dépôt)
- [Rôle 1 — Classification d'âge (FairFace)](#-rôle-1--classification-dâge-fairface)
- [Rôle 2 — Détection d'escaliers (Open Images V7)](#-rôle-2--détection-descaliers-open-images-v7)
- [Résultats](#-résultats)
- [Comment reproduire](#-comment-reproduire)
- [Technologies](#️-technologies)
- [Concepts clés appliqués](#-concepts-clés-appliqués)

---

## 🔗 Liens du projet

| Ressource | Lien |
|-----------|------|
| 🌐 **Application live** | [vision-guide-dl-frontend.vercel.app](https://vision-guide-dl-frontend.vercel.app/) |
| 🎨 **Frontend** (React / Next.js) | [vision-guide-DL-frontend](https://github.com/mohamedkhalilkhelifi20/vision-guide-DL-frontend) |
| ⚙️ **Backend** (API / Inférence) | [vision-guide-DL-backend](https://github.com/mohamedkhalilkhelifi20/vision-guide-DL-backend) |
| 🧠 **Modèles DL** (ce repo) | [vision-guide-DL-yolov8](https://github.com/mohamedkhalilkhelifi20/vision-guide-DL-yolov8) |

---

## 🎯 Problème

YOLOv8 pré-entraîné sur COCO couvre 80 classes génériques — il détecte une `personne`, mais ne dit pas si c'est un enfant, un adulte ou une personne âgée. Il ne détecte pas du tout les **escaliers**, danger majeur pour un utilisateur malvoyant.

Object-Finder résout ces deux limitations en ajoutant une couche CNN spécialisée **après** chaque détection YOLO.

---

## 🏗️ Architecture du pipeline

```
Image caméra
     │
     ▼
┌─────────────┐
│   YOLOv8n   │  ← Détection + localisation (80 classes COCO)
└─────────────┘
     │
     ├──── "person" détecté ─────────────────────────────────┐
     │                                                        ▼
     │                                          ┌─────────────────────────┐
     │                                          │     MobileNetV2         │
     │                                          │   Transfer Learning     │
     │                                          │  "person" → enfant /    │
     │                                          │   adulte / personne_âgée│
     │                                          └─────────────────────────┘
     │
     └──── "stairs" NON détecté ────────────────────────────┐
                                                             ▼
                                              ┌─────────────────────────┐
                                              │     ConvNeXt-Tiny       │
                                              │    Fine-tuning complet  │
                                              │   Détection binaire     │
                                              │  stairs / pas_stairs    │
                                              └─────────────────────────┘
                                                             │
                                                             ▼
                                              Résultat : label précis + niveau de danger
```

---

## 📁 Structure du dépôt

```
object-finder/
│
├── role1_age/                         # Rôle 1 — Classification d'âge
│   ├── 1_EDA.ipynb                    # Exploration du dataset FairFace
│   ├── 2_Preprocessing.ipynb          # Mapping âge → classe, sauvegarde CSV
│   ├── 3_v1_Model_Comparison.ipynb    # Comparaison 4 architectures CNN
│   ├── 3_v2_Model_Comparison.ipynb    # Fine-tuning MobileNetV2 sur FairFace (FINAL)
│   └── models/
│       └── mobilenetv2_fairface.pth   # Modèle final sauvegardé
│
├── role2_stairs/                      # Rôle 2 — Détection d'escaliers
│   ├── 1_EDA_Preprocessing.ipynb      # EDA + préparation dataset Open Images V7
│   ├── 2_YOLOv8_Finetune.ipynb        # Fine-tuning YOLOv8n pour "stairs"
│   ├── 3_v1_Modeling.ipynb            # Expérimentation initiale CNN (baseline)
│   ├── 3_v2_Modeling_Final.ipynb      # ConvNeXt-Tiny fine-tuning (FINAL)
│   └── models/
│       ├── yolov8n_stairs.pt          # YOLOv8n fine-tuné pour escaliers
│       └── convnext_stairs_v2.pth     # Modèle CNN final
│
└── README.md
```

---

## 🧑‍🦳 Rôle 1 — Classification d'âge (FairFace)

### Problème
YOLO détecte `person` — impossible de distinguer un enfant d'un adulte ou d'une personne âgée, information pourtant utile pour contextualiser un danger.

### Dataset — FairFace

| Propriété       | Valeur                        |
|-----------------|-------------------------------|
| Source          | [FairFace (Kaggle)](https://www.kaggle.com/datasets/aibloy/fairface) |
| Train           | 86 744 images                 |
| Validation      | 10 954 images                 |
| Classes cibles  | `enfant` / `adulte` / `personne_agée` |

**Mapping âge → classe :**

| Ages FairFace       | Classe cible    |
|---------------------|-----------------|
| 0-2, 3-9, 10-19     | `enfant`        |
| 20-29, 30-39, 40-49 | `adulte`        |
| 50-59, 60-69, 70+   | `personne_agée` |

**Distribution (déséquilibrée) :**

| Classe         | Train  | Val   |
|----------------|--------|-------|
| adulte         | 55 592 | 6 983 |
| enfant         | 21 303 | 2 736 |
| personne_agée  |  9 849 | 1 235 |

### Comparaison d'architectures (10 epochs, backbone gelé)

| Modèle          | Params  | Taille  | Val Acc | Stratégie        |
|-----------------|---------|---------|---------|------------------|
| CNN Simple      | 127K    | 0.5 MB  | 69.05%  | From Scratch     |
| ResNet18        | 11.2M   | 42.6 MB | 69.07%  | Transfer Learning|
| EfficientNet-B0 | 4.0M    | 15.3 MB | 71.60%  | Transfer Learning|
| **MobileNetV2** | **2.2M**| **8.5 MB** | **71.96%** | **Transfer Learning ✅** |

MobileNetV2 est sélectionné : meilleure accuracy avec le moins de paramètres parmi les modèles pré-entraînés — idéal pour l'inférence en temps réel.

### Fine-tuning MobileNetV2 sur FairFace

**Stratégie :**
- Toutes les couches dégelées (fine-tuning complet)
- Dual learning rates : backbone `lr=1e-5` / tête `lr=1e-4`
- Loss pondérée (racine carrée inversée) pour gérer le déséquilibre

**Hyperparamètres :**

| Paramètre      | Valeur                          |
|----------------|---------------------------------|
| Optimizer      | Adam                            |
| Scheduler      | ReduceLROnPlateau (factor=0.5)  |
| Batch size     | 32                              |
| Max epochs     | 20                              |
| Loss           | CrossEntropyLoss pondérée       |
| GPU            | Tesla T4 (Kaggle)               |

**Résultats finaux :**

| Métrique       | Valeur  |
|----------------|---------|
| Val Accuracy   | 65.36%  |
| Val F1-macro   | 64.04%  |

**Classification Report :**

| Classe        | Precision | Recall | F1-score |
|---------------|-----------|--------|----------|
| adulte        | 0.941     | 0.497  | 0.650    |
| enfant        | 0.612     | 0.915  | 0.734    |
| personne_agée | 0.373     | 0.960  | 0.537    |

> **Note** : Le recall élevé sur `personne_agée` (96%) est prioritaire — il est préférable de signaler un faux positif que de rater une vraie personne âgée en danger.

---

## 🪜 Rôle 2 — Détection d'escaliers (Open Images V7)

### Problème
Les escaliers sont absents des 80 classes COCO de YOLOv8. L'approche est double : fine-tuner YOLOv8 pour détecter les escaliers, **et** entraîner un CNN binaire pour raffiner la décision.

### Dataset — Open Images V7

| Propriété      | Valeur                             |
|----------------|------------------------------------|
| Source         | Open Images V7 (via FiftyOne)      |
| Annotations    | 6 026 bounding boxes               |
| Images train   | 4 461                              |
| Images val     | 914 (split 80/20 aléatoire)        |
| Classes        | `stairs` / `pas_stairs` (binaire)  |

### YOLOv8n Fine-tuning (Notebook 2)

**Objectif** : adapter YOLOv8n (pré-entraîné COCO) pour détecter la classe `stairs`.

| Paramètre      | Valeur  |
|----------------|---------|
| Modèle base    | YOLOv8n |
| Epochs         | 50      |
| Image size     | 640×640 |
| Batch size     | 16      |
| lr0            | 0.001   |

**Résultats YOLOv8n fine-tuné :**

| Métrique   | Valeur |
|------------|--------|
| mAP50      | —      |
| mAP50-95   | —      |
| Précision  | —      |
| Rappel     | —      |

> Les métriques détaillées sont disponibles dans `2_YOLOv8_Finetune.ipynb`.

### ConvNeXt-Tiny Fine-tuning — Modèle FINAL (Notebook 3 v2)

ConvNeXt-Tiny est sélectionné pour sa capacité supérieure sur les tâches binaires et sa robustesse grâce aux connexions denses.

**Stratégie fine-tuning :**
- Phase unique : toutes les couches dégelées
- Dual learning rates : backbone `lr=1e-5` / tête `lr=1e-4`
- Scheduler : **OneCycleLR** (warmup + peak + descente cosinus)
- Correction du déséquilibre : **Loss pondérée seule** (sans WeightedRandomSampler — évite la double correction)

**Hyperparamètres :**

| Paramètre      | Valeur                |
|----------------|-----------------------|
| Modèle         | ConvNeXt-Tiny         |
| Optimizer      | Adam                  |
| Scheduler      | OneCycleLR            |
| Batch size     | 32                    |
| Max epochs     | 20                    |
| Dropout        | 0.4                   |
| GPU            | Tesla T4 (Kaggle)     |

**Résultats finaux — ConvNeXt-Tiny v2 :**

| Métrique        | Valeur     |
|-----------------|------------|
| Val Accuracy    | **96.96%** |
| Val F1-macro    | **96.84%** |
| AUC             | **0.9906** |
| Meilleur epoch  | 15         |
| Threshold opt.  | 0.504      |

> **Threshold optimal = 0.504** — à utiliser dans l'intégration Object-Finder pour maximiser le F1-macro.

---

## 📈 Résultats — Synthèse

| Module               | Modèle          | Métrique principale | Valeur     |
|----------------------|-----------------|---------------------|------------|
| Rôle 1 (âge)         | MobileNetV2     | Val F1-macro        | 64.04%     |
| Rôle 2 (escaliers)   | ConvNeXt-Tiny   | Val F1-macro        | **96.84%** |
| Détection YOLO       | YOLOv8n fine-tuné | mAP50             | voir notebook |

---

## 🚀 Comment reproduire

### Prérequis
- Compte Kaggle (GPU gratuit Tesla T4)
- Datasets ajoutés via **Add Input** dans chaque notebook

### Ordre d'exécution

#### Rôle 1 — Classification d'âge

```bash
# 1. Ajouter le dataset FairFace via Kaggle > Add Input
# 2. Exécuter dans l'ordre :

1_EDA.ipynb
    └─ Output : analyse visuelle, distribution des classes

2_Preprocessing.ipynb
    └─ Input  : FairFace dataset
    └─ Output : train.csv, val.csv

3_v2_Model_Comparison.ipynb
    └─ Input  : 2_Preprocessing (train.csv / val.csv)
    └─ Output : mobilenetv2_fairface.pth
```

#### Rôle 2 — Détection escaliers

```bash
# 1. Ajouter Open Images V7 (via FiftyOne dans le notebook)
# 2. Exécuter dans l'ordre :

1_EDA_Preprocessing.ipynb
    └─ Output : stairs_dataset.csv, dossier crops/

2_YOLOv8_Finetune.ipynb
    └─ Input  : 1_EDA_Preprocessing
    └─ Output : yolov8n_stairs.pt

3_v2_Modeling_Final.ipynb
    └─ Input  : 1_EDA_Preprocessing (stairs_dataset.csv + crops/)
    └─ Output : convnext_stairs_v2_final.pth
```

### Charger les modèles sauvegardés

**MobileNetV2 (Rôle 1) :**
```python
from torchvision import models
import torch.nn as nn
import torch

model = models.mobilenet_v2()
model.classifier = nn.Sequential(
    nn.Dropout(p=0.3),
    nn.Linear(1280, 256),
    nn.ReLU(inplace=True),
    nn.Linear(256, 3)   # 3 classes : enfant / adulte / personne_agée
)
model.load_state_dict(torch.load("mobilenetv2_fairface.pth"))
model.eval()
```

**ConvNeXt-Tiny (Rôle 2) :**
```python
# Charger via PyTorch Lightning checkpoint
model = ConvNeXtFineTunedV2.load_from_checkpoint(
    "convnext_stairs_v2_final.pth",
    loss_weights=loss_weights,
    steps_per_epoch=len(train_loader),
    max_epochs=20
)
model.eval()
```

---

## 🛠️ Technologies

| Outil                   | Usage                              |
|-------------------------|------------------------------------|
| Python 3.12             | Langage principal                  |
| PyTorch 2.10            | Framework deep learning            |
| PyTorch Lightning 2.6   | Boucles d'entraînement structurées |
| torchvision             | MobileNetV2, ConvNeXt-Tiny         |
| Ultralytics             | YOLOv8n fine-tuning                |
| FiftyOne                | Chargement Open Images V7          |
| Pandas / NumPy          | Manipulation des données           |
| Matplotlib / Seaborn    | Visualisation                      |
| scikit-learn            | Métriques, train/val split         |
| Kaggle Notebooks        | Environnement GPU gratuit (T4)     |

---

## 🧪 Concepts clés appliqués

- **Transfer Learning** — partir de poids ImageNet pré-entraînés au lieu d'entraîner from scratch
- **Fine-tuning complet** — dégeler toutes les couches avec dual learning rates (backbone lent, tête rapide)
- **Data Augmentation** — flip horizontal, rotation ±15°, ColorJitter, RandomErasing pour robustesse
- **Loss pondérée** — gérer le déséquilibre de classes sans WeightedRandomSampler (évite double correction)
- **OneCycleLR Scheduler** — warmup progressif + peak + descente cosinus pour convergence stable
- **Early Stopping** — sauvegarde automatique du meilleur checkpoint (F1-macro)
- **Confusion Matrix + Classification Report** — évaluation multi-classe complète
- **Threshold optimal ROC** — maximiser le F1-macro en ajustant le seuil de décision (0.504)
- **Pipeline modulaire** — inter-notebook data flow via CSV (EDA → Preprocessing → Modeling)

---

## 👤 Auteur

**Mohamed Khalil Khelifi**  
4ème année Data Science & IA  
École Polytechnique de Sousse
