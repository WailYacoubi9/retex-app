# Assistant RETEX — mode d'emploi du pilote

**Quoi :** un assistant qui répond en français aux questions sur les 9 191
incidents de sécurité (18 ans de REX). Un seul champ de saisie (onglet
**🎯 Automatique**) : il choisit lui-même la bonne façon de répondre.

**Accès :** [URL Streamlit interne — réseau ADL uniquement]

## Ce que vous pouvez lui demander

| Capacité | Exemple |
|---|---|
| **Compter** (chiffres exacts, vérifiés) | « Combien d'incidents de nuit en 2024 ? » |
| **Analyser** (proportions, tendances, croisements) | « Les incidents de nuit sont-ils proportionnellement plus graves ? » · « Parmi les incidents au roulage, quel facteur domine ? » |
| **Raconter** (recherche dans les fiches) | « Que s'est-il passé lors des incidents de dégivrage ? » |
| **Retrouver une fiche** | « Montre-moi la fiche FNE/AA/NNNN » |
| **Recommander** (mémoire des cas similaires) | Décrivez un événement : « Un camion a refusé la priorité à un avion au repoussage » → incidents similaires + actions qui avaient été prises |
| **Décoder un sigle** | « Que signifie SSLIA ? » |

## Ce qu'il ne fait pas (assumé)

- Les données absentes des fiches (coûts, météo, mouvements d'avions) → il refuse.
- **Toute question sur une personne** (« qui a fait… », « les actions de X ») →
  refus systématique : le REX analyse le système, jamais les individus.
- Les analyses très avancées (comparaisons complexes, durées) — en construction.
- Les chiffres des voies « compter/analyser » sont **calculés et exacts** ;
  les réponses « raconter/recommander » sont rédigées par IA à partir des
  fiches : gardez votre regard critique et signalez toute bizarrerie.

## Ce qu'on attend de vous (2 semaines)

1. **Posez de vraies questions**, celles de votre quotidien SMS — même si vous
   pensez qu'il va échouer : ses échecs nous servent autant que ses réussites.
2. **Votez 👍/👎** sous chaque réponse (2 secondes, c'est notre boussole).
3. Signalez toute réponse fausse ou étrange (commentaire du 👎 ou directement).

## Transparence

Vos questions et les réponses sont **enregistrées pour évaluation** — elles
sont **anonymisées à l'écriture** (aucun nom, aucun identifiant : masquage
automatique) et servent uniquement à améliorer l'assistant. Conformément à la
culture juste, aucun usage individuel n'est possible ni recherché.

## Critères de succès du pilote

≥ 50 vraies questions posées · % de 👍 mesuré par capacité · chaque échec
transformé en cas de test. Merci !
