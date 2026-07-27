#!/usr/bin/env python3
"""
Pour chaque sigle du glossaire, trouve TOUS les FNEs qui le mentionnent
dans neo4j (detail, action_corrective, analyse_chaud, titre, detail_verification).
Met à jour le champ 'references' avec la liste complète.
"""
import json, os, sys, re
from neo4j import GraphDatabase

# Identifiants via l'environnement (jamais en dur — cf. .env du serveur).
NEO4J_URI  = os.environ.get("NEO4J_URI", "bolt://ia-neo4j:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PWD  = os.environ.get("NEO4J_PASSWORD")
if not NEO4J_PWD:
    sys.exit("NEO4J_PASSWORD absent de l'environnement — refuse de tourner sans identifiant sécurisé.")

GLOSSAIRE_PATH = os.environ.get("GLOSSAIRE_PATH", "/app/glossaire_sigles.json")

# Champs texte à fouiller
TEXT_FIELDS = ["titre", "detail", "action_corrective", "analyse_chaud", "detail_verification"]

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PWD))

def find_fnes(sigle: str) -> list[str]:
    """Retourne tous les numero_fe qui contiennent le sigle dans n'importe quel champ texte."""
    # Pour les sigles courts ou ambigus (NP, iso), on cherche avec word boundary simulé
    # Pour les sigles longs, CONTAINS suffit
    conditions = " OR ".join(
        f"(i.{f} IS NOT NULL AND i.{f} CONTAINS $s)"
        for f in TEXT_FIELDS
    )
    cypher = (
        f"MATCH (i:IncidentSecu) "
        f"WHERE coalesce(i.is_test_data, false) = false AND ({conditions}) "
        f"RETURN i.numero_fe AS fe ORDER BY i.numero_fe"
    )
    with driver.session() as sess:
        result = sess.run(cypher, s=sigle)
        return [r["fe"] for r in result if r["fe"]]

def main():
    with open(GLOSSAIRE_PATH) as f:
        data = json.load(f)

    all_sections = [
        ("sigles_a_definir",            data["sigles_a_definir"]),
        ("sigles_deja_definis_a_verifier", data["sigles_deja_definis_a_verifier"]),
    ]

    for section_name, entries in all_sections:
        for entry in entries:
            sigle = entry["sigle"]
            fnes = find_fnes(sigle)
            entry["references_toutes"] = fnes
            print(f"  {sigle:8s} : {len(fnes)} FNEs trouvés")

    with open(GLOSSAIRE_PATH, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"\nFichier mis à jour : {GLOSSAIRE_PATH}")

if __name__ == "__main__":
    main()
