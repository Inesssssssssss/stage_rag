# test_retour.py

Ce script permet d'orchestrer une boucle de planification interactive entre un LLM (modèle de langage) et un VLM (modèle de vision), en intégrant des retours utilisateur et une base de connaissances vectorielle (ChromaDB).

## Fonctionnalités principales
- **Description d'image** : Utilise un VLM pour extraire la liste des objets présents sur une image.
- **Planification LLM** : Génère un plan d'action pour chaque objet détecté via un LLM, avec possibilité de demander des précisions au VLM.
- **Boucle de feedback utilisateur** : Permet à l'utilisateur d'ajouter des informations ou corrections, qui sont indexées dans une base vectorielle pour améliorer la planification.
- **Stockage vectoriel** : Chaque retour utilisateur est encodé et stocké dans ChromaDB avec un identifiant unique.
- **Planification finale** : Génère un plan final pour chaque objet en tenant compte de tous les retours et documents utiles.

## Dépendances
- Python 3.8+
- [ollama](https://ollama.com/) (API Python)
- [chromadb](https://www.trychroma.com/)
- Un modèle LLM compatible (ex: `qwen3:4b`, `llama3.1:8b`)
- Un modèle VLM compatible (ex: `qwen2.5vl`)
- Le module `llm_vlm_planner` (avec `task_planner` et `utils`)

## Utilisation
1. Placez une image à analyser dans le dossier `Images/` (par défaut `Images/device_live.png`).
2. Lancez le script :

```bash
python test_retour.py
```

3. Suivez les instructions dans le terminal pour interagir avec le planificateur et ajouter des retours si besoin.

## Structure du code
- `describe_image` : Décrit l'image et extrait la liste des objets.
- `embed_and_store_document` : Ajoute un document utilisateur dans la base vectorielle avec un id unique.
- `llm_vlm_loop` : Boucle de dialogue entre LLM et VLM pour générer un plan initial.
- `user_feedback_loop` : Permet d'ajouter des retours utilisateur et d'améliorer le plan.
- `main` : Orchestration générale du workflow.

## Exemple de workflow
1. Le VLM décrit l'image et liste les objets.
2. Pour chaque objet, le LLM propose un plan.
3. L'utilisateur peut affiner le plan en ajoutant des informations.
4. Les nouveaux documents sont indexés et utilisés pour améliorer la planification.
5. Un plan final est généré et affiché.

## Remarques
- Le script suppose que les modèles Ollama sont accessibles localement.
- Le stockage vectoriel est réinitialisé à chaque exécution.
- Le script est interactif et nécessite des entrées utilisateur pour affiner les plans.
