# Prompts de l'assistant — guide d'édition

Ce dossier contient **les instructions données au LLM** pour chaque fonction de
l'assistant. Ce sont de simples fichiers texte : **les modifier ne demande
aucune compétence technique**.

## Quel fichier pour quel onglet ?

| Fichier | Onglet / fonction | Rôle |
|---|---|---|
| `question_libre.parseur.txt` | 🧭 Question libre | comprendre la question (filtres, comptage…) |
| `question_libre.reponse_liste.txt` | 🧭 Question libre | rédiger la réponse pour une liste |
| `question_libre.reponse_repartition.txt` | 🧭 Question libre | rédiger la réponse pour une répartition |
| `recherche_semantique.reponse.txt` | 🔍 Recherche sémantique | rédiger la réponse à partir des fiches trouvées |
| `agregation.parseur.txt` | 📊 Agrégation | comprendre la question de comptage |
| `actions.parseur.txt` | 🛠️ Actions | comprendre la question sur les actions |
| `actions.reponse_liste.txt` | 🛠️ Actions | rédiger la liste d'actions |
| `actions.reponse_comptage.txt` | 🛠️ Actions | rédiger un comptage d'actions |
| `recommandation.reponse.txt` | 💡 Recommandation | rédiger les recommandations d'actions |

## Règles d'édition (importantes)

1. **Prise d'effet immédiate** : enregistrez le fichier, posez une question
   dans l'onglet concerné — la modification est déjà active. Aucun
   redémarrage nécessaire.
2. **Ne touchez jamais aux mots entre accolades** comme `{question}`,
   `{contexte}`, `{total}` : ce sont des emplacements remplis automatiquement
   par le système. Les déplacer est permis, les renommer ou les supprimer casse
   la fonction.
3. **Formulez toujours en positif** (« fais X »), jamais en négatif de rappel
   (« surtout ne fais pas Y » avec Y détaillé) : mentionner un comportement
   pousse le modèle à le reproduire — c'est vérifié.
4. **Testez après chaque modification** avec 2-3 questions dans l'onglet
   concerné (idéalement celles des rapports `tests/rapport_*.json`).
5. **La connaissance métier ne va pas ici** : les exemples de questions, les
   synonymes, le sens des valeurs vivent dans le schéma YAML du module
   (`retex-ingestion/config/schemas/…` + publication). Ici, on ne règle que la
   FORME du raisonnement et des réponses.
6. Ces fichiers sont versionnés avec le code : toute modification durable doit
   être commitée (historique + retour arrière possible).
