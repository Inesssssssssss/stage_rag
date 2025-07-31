# LLM-VLM Planner - README

Ce dossier propose un pipeline complet pour la planification de tâches robotiques à partir d'une image, en utilisant un modèle de langage (LLM) et un modèle de vision (VLM), avec intégration d'une base de données vectorielle ChromaDB pour le RAG (Retrieval Augmented Generation).

## Structure du dossier

- `src/llm_vlm_planner/` : Code source principal (planificateur, utilitaires, configuration).
    - `utils/` : Fonctions utilitaires (traitement, embeddings, etc).
    - `config/` : Prompts et fichiers de configuration.
    - `planners/` : (optionnel) Implémentations de planificateurs.
- `main/` : Scripts principaux de lancement.
    - `planner_vlm_llm.py` : Pipeline principal avec interaction utilisateur (l'utilisateur peut corriger ou enrichir le plan, mais il n'y a pas d'échange direct entre le LLM et le VLM pendant la planification).
    - `interactive_llm_vlm.py` : Script interactif avancé avec double interaction : l'utilisateur peut corriger/enrichir le plan ET le LLM peut interroger dynamiquement le VLM pour obtenir des précisions sur l'image (LLM <-> VLM <-> User).
- `experiment/` : Scripts d'expérimentation (batch, ablation, etc).
- `Images/` : Images d'entrée pour l'analyse visuelle (ex : `device_live.png`).
- `audio/` : Fichiers audio pour tests ou retours utilisateur.
- `results/` : Résultats générés (plots, logs, etc).
    - `plot_*.png` : Graphiques générés automatiquement.
    - `experiment_results.txt`, `results.txt` : Logs et résultats d'expérience.

## Fonctionnement général

1. **Description d'image** :
   - Le VLM analyse une image et retourne la liste des objets détectés.
2. **Planification par objet** :
   - Pour chaque objet détecté, le LLM génère un plan d'action.
   - L'utilisateur peut affiner le plan en ajoutant des informations, qui sont indexées dans ChromaDB.
   - Le LLM peut demander des précisions au VLM si besoin ("ask vlm").
3. **RAG (Retrieval Augmented Generation)** :
   - Les documents ajoutés sont vectorisés et stockés dans ChromaDB.
   - Le LLM utilise les documents les plus pertinents pour améliorer la planification.
4. **Génération du plan final** :
   - Le planificateur (`TaskPlanner`) synthétise la réponse finale à partir du dernier échange LLM et des documents utiles.

## Utilisation

### Prérequis
- Python 3.10+
- Dépendances :
  - `chromadb`
  - `ollama`
  - Modules du dossier `llm_vlm_planner`
- Modèles Ollama nécessaires installés localement (`qwen3:4b`, `qwen2.5vl`, etc.)



### Lancer le script interactif avancé (LLM <-> VLM <-> User)

```bash
python main/interactive_llm_vlm.py
```

- Le script affiche la description de l'image et la liste des objets détectés.
- Pour chaque objet, il propose un plan d'action généré par le LLM.
- L'utilisateur peut ajouter des informations (taper du texte, puis "no" pour valider le plan).
- Le LLM peut demander dynamiquement des précisions au VLM (ex : "ask vlm: ...").
- À la fin, le planificateur génère et affiche le plan final pour chaque objet.

### Lancer le pipeline interactif simple (User <-> LLM)

```bash
python main/planner_vlm_llm.py
```

- L'utilisateur peut corriger ou enrichir le plan proposé par le LLM pour chaque objet.
- Il n'y a pas d'interaction automatique entre le LLM et le VLM pendant la planification (le VLM n'est utilisé que pour la description initiale de l'image).

### Lancer une expérience batch

```bash
python experiment/experiment_RAG.py
```
(ou un autre script du dossier `experiment/`)

### Analyser les résultats

```bash
python results/analyze_log.py
```
Les graphiques seront enregistrés dans `results/`.

### Personnalisation
- Pour changer l'image analysée, modifier la variable `image_path` dans les scripts.
- Pour ajuster les prompts ou la logique, éditer les fonctions dans `src/llm_vlm_planner/utils/` ou les fichiers de configuration dans `config/`.

## Bonnes pratiques
- Ajouter un fichier `__init__.py` dans chaque dossier Python.
- Regrouper les résultats générés automatiquement dans `results/plots/` et `results/logs/`.
- Documenter les nouveaux scripts ou modules dans `docs/` si besoin.

---

Pour toute question ou amélioration, modifier ou ouvrir une issue dans le dépôt principal.
