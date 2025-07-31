# LLM-VLM Planner - README

Ce dossier contient un pipeline complet pour la planification de tâches robotiques à partir d'une image, en utilisant un modèle de langage (LLM) et un modèle de vision (VLM), avec intégration de la base de données vectorielle ChromaDB pour le RAG (Retrieval Augmented Generation).

## Contenu du dossier

- `test_retour.py` : Script principal orchestrant l'interaction entre l'utilisateur, le LLM, le VLM et la base de documents. Permet de générer un plan d'action pour chaque objet détecté sur une image.
- `Images/` : Dossier contenant les images utilisées pour l'analyse visuelle (ex : `device_live.png`, `lego_live.png`).
- (Dépendances) :
  - `llm_vlm_planner/` : Module contenant les utilitaires, le planificateur de tâches (`TaskPlanner`), et les fonctions de génération de prompts et de récupération de documents.
  - `chromadb` : Base de données vectorielle pour l'indexation et la recherche de documents.
  - `ollama` : API pour interagir avec les modèles LLM et VLM (ex : `qwen3:4b`, `qwen2.5vl`).

## Fonctionnement général

1. **Description d'image** :
   - Le VLM analyse une image et retourne la liste des objets détectés sous forme de liste Python.
2. **Planification par objet** :
   - Pour chaque objet détecté, le LLM génère un plan d'action.
   - L'utilisateur peut affiner le plan en ajoutant des informations, qui sont indexées dans ChromaDB.
   - Le LLM peut demander des précisions au VLM si besoin ("ask vlm").
3. **RAG (Retrieval Augmented Generation)** :
   - Les documents ajoutés par l'utilisateur sont vectorisés et stockés dans ChromaDB.
   - Le LLM utilise les documents les plus pertinents pour améliorer la planification.
4. **Génération du plan final** :
   - Le planificateur (`TaskPlanner`) synthétise la réponse finale à partir du dernier échange LLM et des documents utiles.

## Utilisation

### Prérequis
- Python 3.10+
- Les dépendances suivantes installées :
  - `chromadb`
  - `ollama`
  - Les modules du dossier `llm_vlm_planner`
- Les modèles Ollama nécessaires doivent être installés localement (`qwen3:4b`, `qwen2.5vl`, etc.)

### Lancer le script principal

```bash
python test_retour.py
```

### Déroulement interactif
- Le script affiche la description de l'image et la liste des objets détectés.
- Pour chaque objet, il propose un plan d'action généré par le LLM.
- L'utilisateur peut ajouter des informations (taper du texte, puis "no" pour valider le plan).
- Le LLM peut demander des précisions au VLM automatiquement.
- À la fin, le planificateur génère et affiche le plan final pour chaque objet.

### Personnalisation
- Pour changer l'image analysée, modifier la variable `image_path` dans `main()`.
- Pour ajuster les prompts ou la logique, éditer les fonctions dans `test_retour.py` ou les utilitaires du module `llm_vlm_planner`.

## Structure recommandée

```
llm_vlm_planner/tests/
│   test_retour.py
│
Images/
│   device_live.png
│   lego_live.png
│   ...
llm_vlm_planner/
    task_planner.py
    utils/
        other.py
        ...
```

## Remarques
- Le script est interactif et nécessite une intervention utilisateur pour affiner les plans.
- Les documents ajoutés sont indexés de façon incrémentale pour éviter les doublons.
- Le code est modulaire et facilement extensible pour d'autres cas d'usage.

---

Pour toute question ou amélioration, modifier ou ouvrir une issue dans le dépôt principal.
