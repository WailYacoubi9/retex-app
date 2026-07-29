"""Retrieval hybride : index lexical BM25 en mémoire + fusion RRF avec le dense.

Rôle : ajouter un signal LEXICAL (BM25 Okapi maison, aucune dépendance) au retrieval
dense (bge-m3/Qdrant) pour remonter les cas que le dense rate à l'échelle du mot
(ex. « fuite de carburant » → les fiches contenant kérosène/avitaillement). Pur
INCLUSION (rappel) : le reranker cross-encoder en aval reste le juge de précision.

Conception (cf. plan) :
- Index construit UNE FOIS au démarrage en scrollant le texte déjà stocké dans le
  payload Qdrant (`texte`) — zéro ré-indexation, zéro changement d'ingestion.
- Réversible : `HYBRID_RETRIEVAL=0` ⇒ aucun index, aucun scroll, chemin dense intact.
- Mono-worker (uvicorn sans --workers) ⇒ un seul index partagé (singleton de module).
"""
from __future__ import annotations

import logging
import math
import os

from domain_synonyms import canonical_of, tokenize

logger = logging.getLogger(__name__)

# Doit rester aligné sur retrieval_incident_v2.SOURCE_MODULE (dupliqué pour éviter
# un import circulaire retrieval <-> hybrid).
SOURCE_MODULE = "incident_securite_v2"

# --- Configuration (lue au démarrage) ----------------------------------------
HYBRID_RETRIEVAL = os.environ.get("HYBRID_RETRIEVAL", "0") == "1"
RRF_K = int(os.environ.get("HYBRID_RRF_K", "60"))
TOP_K_DENSE = int(os.environ.get("HYBRID_TOP_K_DENSE", "50"))
TOP_K_BM25 = int(os.environ.get("HYBRID_TOP_K_BM25", "50"))
TOP_K_FUSED = int(os.environ.get("HYBRID_TOP_K_FUSED", "30"))
MIN_LEX_TERMS = int(os.environ.get("HYBRID_MIN_LEX_TERMS", "1"))
BM25_SCORE_FLOOR = float(os.environ.get("HYBRID_BM25_SCORE_FLOOR", "0.0"))

# Paramètres BM25 Okapi standards.
_K1 = 1.5
_B = 0.75

# Champs « solution » exclus du MATCHING de la reco/synthèse : on veut apparier
# problème↔problème (la SITUATION), pas la description sur l'action/vérification
# (= la solution qu'on cherche). N'affecte QUE le matching : les actions sont
# toujours EXTRAITES des fiches retrouvées (A_POUR_ACTION + champ action_corrective).
EXCLUDED_MATCH_FIELDS = {"action_corrective", "detail_verification"}


class Bm25Index:
    """Index BM25 en mémoire sur le texte des chunks (payload `texte`)."""

    def __init__(self) -> None:
        self.postings: dict[str, list[tuple[int, int]]] = {}  # terme -> [(doc_idx, tf)]
        self.doc_len: list[int] = []
        self.meta: list[dict] = []   # doc_idx -> {chunk_id, incident_id, numero_fe, field_canonical}
        self.idf: dict[str, float] = {}
        self.avgdl: float = 0.0
        self.N: int = 0

    def build_from_scroll(self, qdrant) -> "Bm25Index":
        rows = qdrant.scroll_all(source_module=SOURCE_MODULE, exclude_test_data=True)
        df: dict[str, int] = {}
        for r in rows:
            payload = r.get("payload") or {}
            # on n'indexe pas les champs « solution » (matching problème↔problème)
            if payload.get("field_canonical") in EXCLUDED_MATCH_FIELDS:
                continue
            toks = tokenize(payload.get("texte") or "")
            if not toks:
                continue
            tf: dict[str, int] = {}
            for t in toks:
                tf[t] = tf.get(t, 0) + 1
            doc_idx = len(self.meta)
            self.meta.append({
                "chunk_id": r.get("chunk_id"),
                "incident_id": payload.get("incident_id"),
                "numero_fe": payload.get("numero_fe"),
                "field_canonical": payload.get("field_canonical", "unknown"),
            })
            self.doc_len.append(len(toks))
            for t, f in tf.items():
                self.postings.setdefault(t, []).append((doc_idx, f))
                df[t] = df.get(t, 0) + 1
        self.N = len(self.meta)
        self.avgdl = (sum(self.doc_len) / self.N) if self.N else 0.0
        for t, d in df.items():
            self.idf[t] = math.log(1 + (self.N - d + 0.5) / (d + 0.5))
        logger.info("Index BM25 construit : %d chunks, %d termes, avgdl=%.1f",
                    self.N, len(self.postings), self.avgdl)
        return self

    def search(self, query_terms, top_k: int) -> list[tuple]:
        """Retourne [(chunk_id, score_bm25, n_overlap, meta)] triés par score.
        `n_overlap` = nb de groupes lexicaux canoniques distincts partagés
        (synonymes comptés une fois) → sert au plancher d'abstention."""
        if not self.N:
            return []
        scores: dict[int, float] = {}
        canon: dict[int, set[str]] = {}
        for term in query_terms:
            post = self.postings.get(term)
            if not post:
                continue
            idf_t = self.idf.get(term, 0.0)
            c = canonical_of(term)
            for doc_idx, tf in post:
                dl = self.doc_len[doc_idx]
                denom = tf + _K1 * (1 - _B + _B * dl / self.avgdl) if self.avgdl else (tf + _K1)
                scores[doc_idx] = scores.get(doc_idx, 0.0) + idf_t * tf * (_K1 + 1) / denom
                canon.setdefault(doc_idx, set()).add(c)
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])[:top_k]
        return [(self.meta[di]["chunk_id"], sc, len(canon[di]), self.meta[di])
                for di, sc in ranked]


# --- Singleton de module -----------------------------------------------------
_INDEX: Bm25Index | None = None


def build_index(qdrant) -> Bm25Index:
    """Construit (ou reconstruit) l'index global. Appelé au démarrage si activé."""
    global _INDEX
    _INDEX = Bm25Index().build_from_scroll(qdrant)
    return _INDEX


def get_index() -> Bm25Index | None:
    return _INDEX


def rrf_fuse(dense_ids: list, bm25_ids: list, k: int = RRF_K) -> dict:
    """Reciprocal Rank Fusion sur des listes de chunk_id (rangs à partir de 1)."""
    fused: dict = {}
    for rank, cid in enumerate(dense_ids, start=1):
        fused[cid] = fused.get(cid, 0.0) + 1.0 / (k + rank)
    for rank, cid in enumerate(bm25_ids, start=1):
        fused[cid] = fused.get(cid, 0.0) + 1.0 / (k + rank)
    return fused
