"""Recomendadores: popularidade (baseline / cold-start), aleatório e filtragem colaborativa."""
import numpy as np

from .data import Dataset


class BaseRecommender:
    """Interface comum. Subclasses implementam `_fit` e `scores`."""

    name = "base"

    def fit(self, ds: Dataset):
        self.ds = ds
        self._fit(ds)
        return self

    def _fit(self, ds: Dataset):
        pass

    def scores(self, u_idx: int) -> np.ndarray:
        """Pontuação de todos os itens para o usuário (maior = melhor)."""
        raise NotImplementedError

    def has_history(self, user_id) -> bool:
        u = self.ds.user_index(user_id)
        return u is not None and self.ds.matrix.indptr[u + 1] > self.ds.matrix.indptr[u]

    def recommend(self, user_id, n: int = 10, exclude_seen: bool = True) -> list[tuple]:
        """Top-N (item_id, score). Itens já conhecidos pelo usuário são excluídos.

        Usuário sem histórico (cold-start) recebe os itens mais populares.
        """
        if not self.has_history(user_id):
            return self._cold_start(n)
        u = self.ds.user_index(user_id)
        s = self.scores(u).astype(float).copy()
        if exclude_seen:
            seen = self.ds.matrix[u].indices
            s[seen] = -np.inf
        return self._top(s, n)

    def _cold_start(self, n: int) -> list[tuple]:
        pop = self if isinstance(self, PopularityRecommender) else PopularityRecommender().fit(self.ds)
        return pop._top(pop.item_scores.copy(), n)

    def _top(self, s: np.ndarray, n: int) -> list[tuple]:
        n = min(n, int(np.isfinite(s).sum()))
        idx = np.argpartition(-s, n - 1)[:n] if n > 0 else np.array([], dtype=int)
        idx = idx[np.argsort(-s[idx], kind="stable")]
        return [(self.ds.item_ids[i], float(s[i])) for i in idx]


class PopularityRecommender(BaseRecommender):
    """Média bayesiana das notas: (v*R + m*C) / (v + m).

    v = nº de avaliações do item, R = nota média do item, C = média global,
    m = mínimo de votos. Evita que itens com poucos votos dominem o ranking.
    """

    name = "Popularidade"

    def __init__(self, min_votes: int = 20):
        self.min_votes = min_votes

    def _fit(self, ds: Dataset):
        m = ds.matrix
        votes = np.diff(m.tocsc().indptr).astype(float)
        sums = np.asarray(m.sum(axis=0)).ravel()
        mean_item = np.divide(sums, votes, out=np.zeros_like(sums), where=votes > 0)
        c = m.data.mean()
        self.item_scores = (votes * mean_item + self.min_votes * c) / (votes + self.min_votes)

    def scores(self, u_idx: int) -> np.ndarray:
        return self.item_scores


class RandomRecommender(BaseRecommender):
    """Baseline mínimo de comparação (sorteia itens)."""

    name = "Aleatório"

    def __init__(self, seed: int = 42):
        self.seed = seed

    def scores(self, u_idx: int) -> np.ndarray:
        return np.random.default_rng(self.seed + u_idx).random(self.ds.n_items)
