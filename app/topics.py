from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional
from collections import Counter

import json
import math
import re
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from tenacity import retry, stop_after_attempt, wait_exponential

from .config import cfg
from .storage import save_parquet, load_parquet, save_json, load_json
from .yandex_gpt import get_yandex_gpt_client, has_yandex_gpt_credentials


logger = logging.getLogger(__name__)


RE_TICKER = re.compile(r"\b[A-Z]{2,6}\b")
RE_PERCENT_OR_NUMBER = re.compile(r"(\d+[.,]?\d*\s*%|\b\d+[.,]?\d*\b)")
RE_WORD_TOKEN = re.compile(r"[\dA-Za-zА-Яа-яёЁ%$€₽]+")
POSITIVE_TERMS = [
    "ipo", "листинг", "buyback", "дивиден", "сделк", "поглощ", "merge", "m&a",
    "одобр", "отчит", "финрезульт", "результат", "guidance", "ставк", "фрс",
    "цб", "санкц", "кредит", "размещ", "облигац", "рост", "снижен", "повыш",
    "buy", "sell", "upgrade", "downgrade", "прогноз", "пересмотр", "прибыль",
    "выручк", "маржин", "инвест", "финанс"
]
NEGATIVE_FORMAT_TERMS = [
    "розыгрыш", "итоги розыгрыша", "победител", "конкурс", "giveaway", "скидк",
    "промокод", "реклам", "спонсор", "итоги недели", "дайджест", "подпис", "репост",
    "акция дня", "лотерея", "гамификац"
]


def _chunked(seq: list[int], size: int) -> list[list[int]]:
    if size <= 0:
        size = 1
    return [seq[i:i + size] for i in range(0, len(seq), size)]


def _parse_json_from_text(text: str) -> list[dict[str, Any]]:
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass
    start = text.find('[')
    end = text.rfind(']')
    if start != -1 and end != -1 and end > start:
        snippet = text[start:end + 1]
        try:
            data = json.loads(snippet)
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            return []
    return []


def _auto_dbscan_params(X: np.ndarray, metric: str = 'cosine') -> tuple[float, int]:
    n_samples = len(X)
    min_samples = max(3, min(10, int(np.log(n_samples))))
    k = min_samples
    try:
        nbrs = NearestNeighbors(n_neighbors=k, metric=metric).fit(X)
        distances, _ = nbrs.kneighbors(X)
        k_distances = distances[:, k - 1]
        k_distances = np.sort(k_distances)
        eps = np.percentile(k_distances, 85)
        if metric == 'cosine':
            if eps < 0.2:
                eps = 0.3
            elif eps > 0.8:
                eps = 0.6
        else:
            if eps < 1.0:
                eps = 1.5
            elif eps > 10.0:
                eps = 5.0
        print(f"Auto DBSCAN params: eps={eps:.3f}, min_samples={min_samples}, data_points={n_samples}")
    except Exception as e:
        print(f"Ошибка в _auto_dbscan_params: {e}")
        eps, min_samples = 0.3, 5  # Fallback
    return eps, min_samples



@dataclass
class _DbscanAttempt:
    eps: float
    min_samples: int
    labels: np.ndarray
    cluster_count: int
    noise_ratio: float

@dataclass
class WeakClusterSummary:
    cluster_id: int
    size: int
    members: list[str]
    mean_similarity: float
    snn_threshold: int


def _summarize_labels(labels: np.ndarray) -> tuple[int, float]:
    if len(labels) == 0:
        return 0, 0.0
    noise_mask = labels == -1
    cluster_count = int(len(set(labels)) - (1 if -1 in labels else 0))
    noise_ratio = float(noise_mask.sum() / len(labels))
    return cluster_count, noise_ratio


def _fit_dbscan(X: np.ndarray, eps: float, min_samples: int, metric: str) -> _DbscanAttempt:
    model = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
    labels = model.fit_predict(X)
    cluster_count, noise_ratio = _summarize_labels(labels)
    return _DbscanAttempt(
        eps=float(eps),
        min_samples=int(min_samples),
        labels=labels,
        cluster_count=cluster_count,
        noise_ratio=noise_ratio,
    )


def _adaptive_dbscan_search(
    X: np.ndarray,
    metric: str,
    base_eps: float,
    base_min_samples: int,
    *,
    allow_eps_variation: bool,
    allow_min_samples_variation: bool,
) -> _DbscanAttempt:
    eps_candidates: list[float] = [max(0.05, float(base_eps))]
    if allow_eps_variation:
        for scale in (0.9, 0.8, 0.7, 0.6, 0.5, 0.42, 0.36, 0.3, 0.25, 0.22, 0.18, 0.15, 0.12, 0.1):
            candidate = max(0.05, float(base_eps) * scale)
            if all(abs(candidate - existing) > 1e-6 for existing in eps_candidates):
                eps_candidates.append(candidate)
    eps_candidates = sorted(eps_candidates, reverse=True)

    min_samples_candidates: set[int] = {max(3, int(round(base_min_samples)))}
    if allow_min_samples_variation:
        min_samples_candidates.update(
            {
                max(3, int(round(base_min_samples)) - 1),
                max(3, int(round(base_min_samples)) - 2),
                max(3, int(np.floor(base_min_samples * 0.75))),
            }
        )
    min_samples_list = sorted(min_samples_candidates)

    attempts: list[_DbscanAttempt] = []
    seen: set[tuple[int, int]] = set()
    for min_samples_value in min_samples_list:
        for eps_value in eps_candidates:
            key = (int(round(eps_value * 1000)), min_samples_value)
            if key in seen:
                continue
            attempt = _fit_dbscan(X, eps_value, min_samples_value, metric)
            attempts.append(attempt)
            seen.add(key)

    if not attempts:
        raise RuntimeError('DBSCAN auto-tuning produced no attempts')

    acceptable = [
        attempt
        for attempt in attempts
        if attempt.cluster_count >= 2 and attempt.noise_ratio <= 0.95
    ]

    def _attempt_score(attempt: _DbscanAttempt) -> tuple[int, float, float]:
        target_noise = 0.4
        return (
            attempt.cluster_count,
            -abs(attempt.noise_ratio - target_noise),
            -attempt.noise_ratio,
        )

    ranking_pool = acceptable if acceptable else attempts
    ranking_pool.sort(key=_attempt_score, reverse=True)
    return ranking_pool[0]


def project_embeddings(emb_df: pd.DataFrame, n_components: int = 5) -> pd.DataFrame:
    feat_cols = [c for c in emb_df.columns if c.startswith("e")]
    X = emb_df[feat_cols].values
    n_components = min(n_components, X.shape[1] - 1) if X.shape[1] > 1 else 1
    if n_components < 1:
        n_components = 1
    pca = PCA(n_components=n_components, random_state=42)
    Z = pca.fit_transform(X)
    proj = pd.DataFrame(Z, columns=[f"p{i:03d}" for i in range(Z.shape[1])])
    proj["message_id"] = emb_df["message_id"].values
    return proj


def cluster_embeddings_dbscan(
        proj_df: pd.DataFrame,
        eps: Optional[float] = None,
        min_samples: Optional[int] = None,
        metric: str = 'cosine',
        normalize: bool = False
) -> tuple[pd.DataFrame, dict[str, float | int | bool]]:
    feat_cols = [c for c in proj_df.columns if c.startswith("p")]
    if not feat_cols:
        raise ValueError("Projection DataFrame must contain columns starting with 'p'")
    X = proj_df[feat_cols].values
    labels = np.full(len(proj_df), -1, dtype=int)
    info: dict[str, float | int | bool] = {
        "eps": float(eps) if eps is not None else 0.0,
        "min_samples": int(min_samples) if min_samples is not None else 0,
        "cluster_count": 0,
        "noise_ratio": 0.0,
        "auto_tuned": False,
    }
    attempt: Optional[_DbscanAttempt] = None
    if len(proj_df) < 3:
        print("Too few points for DBSCAN; marking everything as noise.")
    else:
        X_proc = X
        if normalize and metric != 'cosine':
            scaler = StandardScaler()
            X_proc = scaler.fit_transform(X)
        if eps is not None and min_samples is not None:
            attempt = _fit_dbscan(X_proc, eps, min_samples, metric)
        else:
            auto_eps, auto_min_samples = _auto_dbscan_params(X_proc, metric=metric)
            base_eps = eps if eps is not None else auto_eps
            base_min_samples = min_samples if min_samples is not None else auto_min_samples
            print(
                f"DBSCAN auto base params: eps={auto_eps:.3f}, min_samples={auto_min_samples}, "
                f"data_points={len(X_proc)}"
            )
            attempt = _adaptive_dbscan_search(
                X_proc,
                metric,
                base_eps,
                base_min_samples,
                allow_eps_variation=eps is None,
                allow_min_samples_variation=min_samples is None,
            )
            info["auto_tuned"] = True
        if attempt is not None:
            labels = attempt.labels.copy()
            unique_labels = np.unique(labels[labels != -1])
            if len(unique_labels) > 0:
                label_map = {old: new for new, old in enumerate(sorted(unique_labels), start=1)}
                for idx in range(len(labels)):
                    if labels[idx] != -1:
                        labels[idx] = label_map[labels[idx]]
    cluster_count, noise_ratio = _summarize_labels(labels)
    info["cluster_count"] = cluster_count
    info["noise_ratio"] = noise_ratio
    if attempt is not None:
        info["eps"] = float(attempt.eps)
        info["min_samples"] = int(attempt.min_samples)
        print(
            f"DBSCAN using eps={attempt.eps:.3f}, min_samples={attempt.min_samples}, "
            f"clusters={cluster_count}, noise={noise_ratio:.1%}"
        )
    clusters = pd.DataFrame({
        "message_id": proj_df["message_id"],
        "cluster": labels,
        "is_noise": labels == -1
    })
    return clusters, info



def cluster_noise_with_snn(
        proj_df: pd.DataFrame,
        clusters_df: pd.DataFrame,
        k: int = 8,
        snn_threshold: int = 4,
        min_cluster_size: int = 2,
        max_cluster_size: int = 5,
        min_cosine: float = 0.82,
        split_large: bool = True,
        id_offset: int = 0,
        keep_strong_immutable: bool = True
) -> tuple[pd.DataFrame, list[WeakClusterSummary]]:
    """Поиск "слабых" кластеров в шуме DBSCAN через SNN-граф."""

    feat_cols = [c for c in proj_df.columns if c.startswith("p")]
    if not feat_cols:
        return clusters_df, []

    clusters_df = clusters_df.copy()
    if "cluster_strength" not in clusters_df.columns:
        clusters_df["cluster_strength"] = np.where(clusters_df["cluster"] == -1, "noise", "strong")
    else:
        clusters_df.loc[clusters_df["cluster"] != -1, "cluster_strength"] = "strong"

    strong_before = clusters_df.loc[clusters_df["cluster"] != -1, ["message_id", "cluster"]].copy()
    strong_before.set_index("message_id", inplace=True)

    noise_mask = clusters_df["cluster"] == -1
    noise_ids = clusters_df.loc[noise_mask, "message_id"].tolist()
    if len(noise_ids) < max(2, min_cluster_size):
        clusters_df.loc[noise_mask, "cluster_strength"] = "singleton"
        return clusters_df, []

    proj_indexed = proj_df.set_index("message_id")
    available_ids = [mid for mid in noise_ids if mid in proj_indexed.index]
    if len(available_ids) < max(2, min_cluster_size):
        clusters_df.loc[noise_mask, "cluster_strength"] = "singleton"
        return clusters_df, []

    X_noise = proj_indexed.loc[available_ids, feat_cols].to_numpy()
    n_points = X_noise.shape[0]
    if n_points < 2:
        clusters_df.loc[noise_mask, "cluster_strength"] = "singleton"
        return clusters_df, []

    # Подготовка kNN-графа по косинусу
    neigh_count = min(k + 1, n_points)
    knn = NearestNeighbors(
        n_neighbors=neigh_count,
        metric="cosine",
        algorithm="brute"
    )
    knn.fit(X_noise)
    distances, indices = knn.kneighbors(X_noise)

    neighbor_sets: list[set[int]] = []
    for row_idx in range(n_points):
        neigh_set: set[int] = set()
        for neigh_idx, dist in zip(indices[row_idx], distances[row_idx]):
            if neigh_idx == row_idx:
                continue
            cosine_sim = 1.0 - float(dist)
            if cosine_sim < min_cosine:
                continue
            neigh_set.add(int(neigh_idx))
        neighbor_sets.append(neigh_set)

    adjacency: list[set[int]] = [set() for _ in range(n_points)]
    for i in range(n_points):
        for j in neighbor_sets[i]:
            if j <= i:
                continue
            shared = len(neighbor_sets[i].intersection(neighbor_sets[j]))
            if shared >= snn_threshold:
                adjacency[i].add(j)
                adjacency[j].add(i)

    # Поиск компонент связности
    visited: set[int] = set()
    components: list[list[int]] = []
    for i in range(n_points):
        if i in visited:
            continue
        stack = [i]
        component: set[int] = set()
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            component.add(node)
            stack.extend(adjacency[node] - visited)
        components.append(sorted(component))

    existing_clusters = clusters_df.loc[clusters_df["cluster"] != -1, "cluster"]
    base_cluster_id = int(existing_clusters.max()) if not existing_clusters.empty else 0
    # if id_offset <= 0:
    #     id_offset = 100_000
    next_cluster_id = base_cluster_id + id_offset

    index_to_id = {idx: mid for idx, mid in enumerate(available_ids)}
    weak_summaries: list[WeakClusterSummary] = []
    assigned_indices: set[int] = set()

    def _create_summary(member_indices: list[int], new_cluster: int) -> WeakClusterSummary:
        vectors = X_noise[member_indices]
        if len(member_indices) <= 1:
            mean_sim = 1.0
        else:
            sims = cosine_similarity(vectors)
            tri_upper = sims[np.triu_indices(len(member_indices), k=1)]
            mean_sim = float(tri_upper.mean()) if tri_upper.size else 1.0
        members = [index_to_id[idx] for idx in member_indices]
        return WeakClusterSummary(
            cluster_id=new_cluster,
            size=len(member_indices),
            members=members,
            mean_similarity=mean_sim,
            snn_threshold=snn_threshold,
        )

    for component in components:
        if len(component) < min_cluster_size:
            continue

        member_indices_lists: list[list[int]]
        if len(component) > max_cluster_size and split_large:
            n_target = max(1, math.ceil(len(component) / max_cluster_size))
            sub_vectors = X_noise[component]
            try:
                aggl = AgglomerativeClustering(
                    n_clusters=n_target,
                    metric="cosine",
                    linkage="average"
                )
                sub_labels = aggl.fit_predict(sub_vectors)
                member_indices_lists = [
                    [component[idx] for idx, lbl in enumerate(sub_labels) if lbl == lbl_value]
                    for lbl_value in sorted(set(sub_labels))
                ]
            except Exception:
                member_indices_lists = [component]
        else:
            member_indices_lists = [component]

        for members in member_indices_lists:
            if len(members) < min_cluster_size:
                continue
            if len(members) > max_cluster_size:
                chunks = [members[i:i + max_cluster_size] for i in range(0, len(members), max_cluster_size)]
            else:
                chunks = [members]
            for chunk in chunks:
                if len(chunk) < min_cluster_size or len(chunk) > max_cluster_size:
                    continue
                next_cluster_id += 1
                weak_summary = _create_summary(chunk, next_cluster_id)
                weak_summaries.append(weak_summary)
                assigned_indices.update(chunk)
                for idx in chunk:
                    mid = index_to_id[idx]
                    mask = clusters_df["message_id"] == mid
                    clusters_df.loc[mask, "cluster"] = next_cluster_id
                    clusters_df.loc[mask, "cluster_strength"] = "weak"

    # Обновляем статусы шума и одиночек
    if keep_strong_immutable and not strong_before.empty:
        restore_map = strong_before["cluster"].astype(int).to_dict()
        strong_mask = clusters_df["message_id"].isin(restore_map.keys())
        if strong_mask.any():
            clusters_df.loc[strong_mask, "cluster"] = (
                clusters_df.loc[strong_mask, "message_id"].map(restore_map).values
            )
            clusters_df.loc[strong_mask, "cluster_strength"] = "strong"

    clusters_df["cluster"] = clusters_df["cluster"].astype(int)
    clusters_df["is_noise"] = clusters_df["cluster"] == -1
    clusters_df.loc[clusters_df["is_noise"], "cluster_strength"] = "singleton"

    return clusters_df, weak_summaries


def analyze_cluster_quality(clusters_df: pd.DataFrame) -> dict:
    """Анализ качества кластеризации для трендов"""
    total_messages = len(clusters_df)
    noise_messages = (clusters_df["cluster"] == -1).sum()
    unique_clusters = clusters_df[clusters_df["cluster"] != -1]["cluster"].nunique()

    cluster_sizes = clusters_df[clusters_df["cluster"] != -1]["cluster"].value_counts()
    avg_cluster_size = cluster_sizes.mean() if len(cluster_sizes) > 0 else 0

    # Конвертируем numpy.int64 в int для JSON
    cluster_size_distribution = {int(k): int(v) for k, v in cluster_sizes.to_dict().items()}

    quality_metrics = {
        "total_messages": int(total_messages),
        "noise_messages": int(noise_messages),
        "noise_ratio": float(noise_messages / total_messages if total_messages > 0 else 0),
        "unique_clusters": int(unique_clusters),
        "avg_cluster_size": float(avg_cluster_size),
        "largest_cluster_size": int(cluster_sizes.max() if len(cluster_sizes) > 0 else 0),
        "cluster_size_distribution": cluster_size_distribution
    }

    print(f"Качество кластеризации: {unique_clusters} кластеров, "
          f"шум: {noise_messages}/{total_messages} ({quality_metrics['noise_ratio']:.1%})")

    return quality_metrics


def _zscore_series(series: pd.Series) -> pd.Series:
    if series.empty:
        return pd.Series(np.zeros(len(series)), index=series.index)
    std = series.std(ddof=0)
    if std == 0 or np.isnan(std):
        return pd.Series(np.zeros(len(series)), index=series.index)
    mean = series.mean()
    return (series - mean) / (std + 1e-9)


def _contains_any(text: str, vocab: list[str]) -> int:
    if not text:
        return 0
    low = text.lower()
    return int(any(token in low for token in vocab))


def triage_singletons(
        messages_df: pd.DataFrame,
        proj_df: pd.DataFrame,
        clusters_df: pd.DataFrame,
        top_m: int = 10,
        weight_engagement: float = 0.5,
        weight_rarity: float = 0.3,
        weight_outlier: float = 0.2,
        freshness_half_life_days: float = 14.0
) -> pd.DataFrame:
    """Ранжирование оставшихся одиночек по интегральному скору важности."""

    if "cluster_strength" in clusters_df.columns:
        singleton_ids = clusters_df.loc[clusters_df["cluster_strength"] == "singleton", "message_id"].tolist()
    else:
        singleton_ids = clusters_df.loc[clusters_df["cluster"] == -1, "message_id"].tolist()

    if not singleton_ids:
        return pd.DataFrame(columns=[
            "message_id",
            "insight_score",
            "engagement_z",
            "rarity_z",
            "outlier_z",
            "freshness",
            "triage_bucket"
        ])

    messages_df = messages_df.copy()
    messages_df["message_id"] = messages_df["message_id"].astype(str)
    singleton_df = messages_df[messages_df["message_id"].isin(singleton_ids)].copy()
    if singleton_df.empty:
        return pd.DataFrame(columns=[
            "message_id",
            "insight_score",
            "engagement_z",
            "rarity_z",
            "outlier_z",
            "freshness",
            "triage_bucket"
        ])

    # --- Engagement ---
    engagement_cols: list[str] = []
    numeric_cols = singleton_df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        low = col.lower()
        if any(token in low for token in ["engagement", "like", "comment", "reply", "share", "view", "reaction", "repost", "forward"]):
            engagement_cols.append(col)
    if "engagement" in singleton_df.columns:
        engagement_cols = ["engagement"]
    if engagement_cols:
        singleton_df["engagement_raw"] = singleton_df[engagement_cols].sum(axis=1)
    else:
        singleton_df["engagement_raw"] = 0.0

    if "sender" in singleton_df.columns:
        singleton_df["engagement_z"] = singleton_df.groupby("sender")["engagement_raw"].transform(_zscore_series)
    else:
        singleton_df["engagement_z"] = _zscore_series(singleton_df["engagement_raw"])

    # --- Rarity ---
    try:
        vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), min_df=1)
        vectorizer.fit(messages_df["text"].astype(str).values)
        singleton_tfidf = vectorizer.transform(singleton_df["text"].astype(str).values)
        rarity_raw = []
        for row in singleton_tfidf:
            rarity_raw.append(float(row.data.mean()) if row.nnz else 0.0)
        singleton_df["rarity_raw"] = rarity_raw
    except ValueError:
        singleton_df["rarity_raw"] = 0.0
    singleton_df["rarity_z"] = _zscore_series(singleton_df["rarity_raw"])

    # --- Outlierness ---
    feat_cols = [c for c in proj_df.columns if c.startswith("p")]
    proj_indexed = proj_df.set_index("message_id")
    available_singletons = [mid for mid in singleton_ids if mid in proj_indexed.index]
    singleton_vectors = proj_indexed.loc[available_singletons, feat_cols].to_numpy() if available_singletons else np.empty((0, len(feat_cols)))

    strong_mask = clusters_df.get("cluster_strength") == "strong" if "cluster_strength" in clusters_df.columns else clusters_df["cluster"] != -1
    strong_centroids: list[np.ndarray] = []
    if strong_mask.any():
        strong_clusters = clusters_df.loc[strong_mask, "cluster"].unique()
        for cid in strong_clusters:
            member_ids = clusters_df.loc[clusters_df["cluster"] == cid, "message_id"].tolist()
            member_ids = [mid for mid in member_ids if mid in proj_indexed.index]
            if not member_ids:
                continue
            centroid = proj_indexed.loc[member_ids, feat_cols].mean(axis=0).to_numpy()
            strong_centroids.append(centroid)

    if strong_centroids and singleton_vectors.size:
        centroid_matrix = np.vstack(strong_centroids)
        sims = cosine_similarity(singleton_vectors, centroid_matrix)
        nearest_sim = sims.max(axis=1)
        outlier_raw = 1.0 - nearest_sim
        # align lengths if some singleton_ids missing in proj
        outlier_series = pd.Series(0.0, index=singleton_df.index)
        for mid, score in zip(available_singletons, outlier_raw):
            outlier_series.loc[singleton_df[singleton_df["message_id"] == mid].index] = score
        singleton_df["outlier_raw"] = outlier_series
    else:
        singleton_df["outlier_raw"] = 0.0
    singleton_df["outlier_z"] = _zscore_series(singleton_df["outlier_raw"])

    # --- Freshness ---
    if "date" in singleton_df.columns:
        singleton_df["date_dt"] = pd.to_datetime(singleton_df["date"], errors="coerce")
    else:
        singleton_df["date_dt"] = pd.NaT
    now = singleton_df["date_dt"].max()
    if pd.isna(now):
        now = pd.Timestamp.utcnow()
    delta_days = (now - singleton_df["date_dt"]).dt.total_seconds() / 86400.0
    delta_days = delta_days.fillna(delta_days.max() if not delta_days.dropna().empty else 0.0)
    decay = np.power(0.5, delta_days / max(freshness_half_life_days, 1e-6))
    singleton_df["freshness"] = decay.clip(lower=0.0, upper=1.0)

    # --- Итоговый скор ---
    singleton_df[["engagement_z", "rarity_z", "outlier_z"]] = singleton_df[["engagement_z", "rarity_z", "outlier_z"]].fillna(0.0)
    score = (
        weight_engagement * singleton_df["engagement_z"]
        + weight_rarity * singleton_df["rarity_z"]
        + weight_outlier * singleton_df["outlier_z"]
    )
    singleton_df["insight_score"] = score * singleton_df["freshness"]

    singleton_df.sort_values("insight_score", ascending=False, inplace=True)
    singleton_df.reset_index(drop=True, inplace=True)
    if top_m and top_m > 0:
        singleton_df["triage_bucket"] = np.where(singleton_df.index < top_m, "priority", "watchlist")
    else:
        singleton_df["triage_bucket"] = "watchlist"

    return singleton_df[[
        "message_id",
        "insight_score",
        "engagement_z",
        "rarity_z",
        "outlier_z",
        "freshness",
        "triage_bucket"
    ]]


def compute_message_usefulness(
        messages_df: pd.DataFrame,
        clusters_df: pd.DataFrame,
        half_life_days: float = 10.0
) -> pd.DataFrame:
    df = messages_df.copy()
    if df.empty:
        return pd.DataFrame(columns=[
            "message_id", "cluster", "usefulness_msg", "fresh", "eng_z", "is_format"
        ])

    df["message_id"] = df["message_id"].astype(str)
    clusters_view = clusters_df[["message_id", "cluster"]].copy()
    clusters_view["message_id"] = clusters_view["message_id"].astype(str)
    df = df.merge(clusters_view, on="message_id", how="left")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    engagement_cols = [
        col for col in numeric_cols
        if any(token in col.lower() for token in [
            "engagement", "like", "comment", "reply", "share", "repost", "view", "reaction", "forward"
        ])
    ]
    if engagement_cols:
        df["eng_raw"] = df[engagement_cols].sum(axis=1)
    else:
        df["eng_raw"] = 0.0
    if "sender" in df.columns:
        df["eng_z"] = df.groupby("sender")["eng_raw"].transform(_zscore_series)
    else:
        df["eng_z"] = _zscore_series(df["eng_raw"])
    df["eng_z"].fillna(0.0, inplace=True)

    texts = df["text"].fillna("").astype(str)
    df["has_ticker"] = texts.apply(lambda s: int(bool(RE_TICKER.search(s))))
    df["has_number"] = texts.apply(lambda s: int(bool(RE_PERCENT_OR_NUMBER.search(s))))
    df["pos_words"] = texts.apply(lambda s: _contains_any(s, POSITIVE_TERMS))
    df["neg_format"] = texts.apply(lambda s: _contains_any(s, NEGATIVE_FORMAT_TERMS))
    df["len_tokens"] = texts.apply(lambda s: len(s.split()))
    df["link_count"] = texts.str.count(r"http[s]?://")
    df["link_ratio"] = (df["link_count"] / df["len_tokens"].clip(lower=1)).fillna(0.0)

    if "date" in df.columns:
        df["date_dt"] = pd.to_datetime(df["date"], errors="coerce")
    else:
        df["date_dt"] = pd.NaT
    now = df["date_dt"].max()
    if pd.isna(now):
        now = pd.Timestamp.utcnow()
    delta_days = (now - df["date_dt"]).dt.total_seconds() / 86400.0
    if delta_days.dropna().empty:
        delta_days = pd.Series(np.zeros(len(df)), index=df.index)
    delta_days = delta_days.fillna(delta_days.max() if not delta_days.dropna().empty else 0.0)
    df["fresh"] = np.power(0.5, delta_days / max(half_life_days, 1e-6)).clip(0.0, 1.0)

    df["len_z"] = _zscore_series(df["len_tokens"].astype(float))
    df["link_z"] = _zscore_series(df["link_ratio"].astype(float))

    msg_score = (
        0.35 * df["has_ticker"]
        + 0.20 * df["has_number"]
        + 0.25 * df["pos_words"]
        + 0.15 * df["eng_z"].clip(lower=0.0)
        + 0.10 * df["len_z"].clip(lower=0.0)
        - 0.60 * df["neg_format"]
        - 0.15 * df["link_z"].clip(lower=0.0)
    )
    msg_score = msg_score.clip(lower=-2.0, upper=3.0)
    df["usefulness_msg"] = msg_score * df["fresh"]
    df["is_format"] = (df["neg_format"] == 1).astype(int)

    return df[[
        "message_id", "cluster", "usefulness_msg", "fresh", "eng_z", "is_format"
    ]]


def score_clusters(
        messages_usefulness: pd.DataFrame,
        clusters_df: pd.DataFrame,
        cluster_metadata: Optional[dict[int, dict[str, Any]]] = None
) -> pd.DataFrame:
    if messages_usefulness.empty:
        return pd.DataFrame(columns=[
            "cluster", "size", "usefulness_cluster", "hot_score", "format_ratio", "cluster_strength"
        ])

    df = messages_usefulness.copy()
    cluster_meta = clusters_df[["message_id", "cluster", "cluster_strength"]].copy()
    cluster_meta["message_id"] = cluster_meta["message_id"].astype(str)
    df = df.merge(cluster_meta, on=["message_id", "cluster"], how="left")
    df["cluster_strength"].fillna("unknown", inplace=True)

    df = df[df["cluster"] != -1]
    if df.empty:
        return pd.DataFrame(columns=[
            "cluster", "size", "usefulness_cluster", "hot_score", "format_ratio", "cluster_strength"
        ])

    df["w"] = df["fresh"] * (1.0 + 0.30 * df["eng_z"].clip(lower=0.0))
    df["w"].replace([np.inf, -np.inf], 0.0, inplace=True)
    df["w"].fillna(0.0, inplace=True)

    grouped = df.groupby("cluster", as_index=False).agg(
        size=("message_id", "count"),
        useful_sum=("usefulness_msg", lambda s: float(np.nansum(s))),
        weight_sum=("w", lambda s: float(np.nansum(s))),
        format_ratio=("is_format", "mean"),
        fresh_avg=("fresh", "mean"),
        eng_pos_ratio=("eng_z", lambda s: float((s > 0).mean()))
    )

    grouped["usefulness_cluster"] = np.where(
        grouped["weight_sum"] > 1e-6,
        grouped["useful_sum"] / grouped["weight_sum"].clip(lower=1e-6),
        0.0
    )
    grouped["usefulness_cluster"] = grouped["usefulness_cluster"].clip(-2.0, 3.0)
    grouped["hot_score"] = (
        0.5 * grouped["fresh_avg"] + 0.4 * grouped["eng_pos_ratio"] - 0.3 * grouped["format_ratio"]
    ).clip(-1.0, 1.0)
    grouped["format_ratio"] = grouped["format_ratio"].fillna(0.0)
    grouped["format_label"] = np.where(
        grouped["format_ratio"] >= 0.6,
        "format/giveaway",
        "content"
    )

    strength_map = clusters_df.groupby("cluster")["cluster_strength"].first().to_dict()
    grouped["cluster_strength"] = grouped["cluster"].map(strength_map).fillna("unknown")

    metadata_map: dict[int, dict[str, Any]] = {}
    if cluster_metadata:
        metadata_map = {int(k): v for k, v in cluster_metadata.items()}

    if metadata_map:
        grouped["llm_content_type"] = grouped["cluster"].map(
            lambda cid: metadata_map.get(int(cid), {}).get("content_type")
        )
        grouped["llm_is_promo"] = grouped["cluster"].map(
            lambda cid: bool(metadata_map.get(int(cid), {}).get("is_promo"))
        )
        grouped["llm_has_numbers"] = grouped["cluster"].map(
            lambda cid: bool(metadata_map.get(int(cid), {}).get("has_numbers"))
        )
        grouped["llm_has_entities"] = grouped["cluster"].map(
            lambda cid: bool(metadata_map.get(int(cid), {}).get("has_entities"))
        )
        grouped["llm_key_entities"] = grouped["cluster"].map(
            lambda cid: metadata_map.get(int(cid), {}).get("key_entities") or []
        )
        grouped["llm_sentiment"] = grouped["cluster"].map(
            lambda cid: metadata_map.get(int(cid), {}).get("sentiment")
        )
        grouped["llm_confidence"] = grouped["cluster"].map(
            lambda cid: metadata_map.get(int(cid), {}).get("confidence")
        )

        grouped["llm_key_entities"] = grouped["llm_key_entities"].apply(
            lambda v: v if isinstance(v, list) else ([v] if v not in (None, "", float('nan')) else [])
        )

        content_type_series = grouped["llm_content_type"].fillna("").str.lower()
        promo_mask = grouped["llm_is_promo"] | content_type_series.isin({
            "promo", "contest", "giveaway", "marketing", "advertisement", "ads"
        })
        grouped.loc[promo_mask, "format_label"] = "format/giveaway"
        grouped.loc[promo_mask, "usefulness_cluster"] -= 0.4

        positive_mask = content_type_series.isin({"news", "analysis", "insight"})
        grouped.loc[positive_mask & ~promo_mask, "usefulness_cluster"] += 0.2

        negative_mask = content_type_series.isin({"spam", "promo"})
        grouped.loc[negative_mask & ~promo_mask, "usefulness_cluster"] -= 0.25

        digest_mask = content_type_series.isin({"digest", "summary", "qa"})
        grouped.loc[digest_mask & ~promo_mask, "usefulness_cluster"] -= 0.1

        grouped.loc[grouped["llm_has_numbers"], "usefulness_cluster"] += 0.05
        grouped.loc[grouped["llm_has_entities"], "usefulness_cluster"] += 0.05

    grouped["usefulness_cluster"] = grouped["usefulness_cluster"].clip(-2.5, 3.5)

    size_vals = grouped["size"].astype(float)
    size_max = float(size_vals.max()) if not size_vals.empty else 0.0
    if size_max > 0:
        size_norm = np.log1p(size_vals) / np.log1p(size_max)
    else:
        size_norm = np.zeros(len(size_vals), dtype=float)

    usefulness_norm = np.clip((grouped["usefulness_cluster"] + 2.5) / 6.0, 0.0, 1.0)
    hot_norm = np.clip((grouped["hot_score"] + 1.0) / 2.0, 0.0, 1.0)

    grouped["priority_score"] = (
        0.5 * usefulness_norm
        + 0.4 * hot_norm
        + 0.1 * size_norm
    )
    grouped["priority_score"] = grouped["priority_score"].clip(0.0, 1.0)

    return grouped.sort_values(
        by=["priority_score", "usefulness_cluster", "hot_score", "size"],
        ascending=[False, False, False, False]
    )

def _fallback_terms_from_texts(texts: list[str], top_k: int) -> list[str]:
    tokens: list[str] = []
    for text in texts:
        if not text:
            continue
        words = RE_WORD_TOKEN.findall(text.lower())
        tokens.extend([w for w in words if len(w) > 2])
    counter = Counter(tokens)
    return [term for term, _ in counter.most_common(top_k)]


def _normalize_cluster_metadata(meta: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    content_type = meta.get("content_type")
    if isinstance(content_type, str):
        result["content_type"] = content_type.strip()
    elif content_type is not None:
        result["content_type"] = str(content_type)
    is_promo = meta.get("is_promo")
    if isinstance(is_promo, str):
        result["is_promo"] = is_promo.lower() in {"true", "1", "yes"}
    else:
        result["is_promo"] = bool(is_promo)
    has_numbers = meta.get("has_numbers")
    if isinstance(has_numbers, str):
        result["has_numbers"] = has_numbers.lower() in {"true", "1", "yes"}
    else:
        result["has_numbers"] = bool(has_numbers)
    has_entities = meta.get("has_entities")
    if isinstance(has_entities, str):
        result["has_entities"] = has_entities.lower() in {"true", "1", "yes"}
    else:
        result["has_entities"] = bool(has_entities)
    sentiment = meta.get("sentiment")
    if isinstance(sentiment, str):
        result["sentiment"] = sentiment.strip()
    elif sentiment is not None:
        result["sentiment"] = str(sentiment)
    key_entities = meta.get("key_entities")
    if isinstance(key_entities, list):
        result["key_entities"] = [str(x).strip() for x in key_entities if str(x).strip()][:6]
    elif isinstance(key_entities, str):
        result["key_entities"] = [x.strip() for x in key_entities.split(",") if x.strip()][:6]
    else:
        result["key_entities"] = []
    confidence = meta.get("confidence")
    if confidence is None:
        result["confidence"] = None
    else:
        try:
            result["confidence"] = float(confidence)
        except (TypeError, ValueError):
            result["confidence"] = None
    return result


def collect_cluster_samples(
        messages_df: pd.DataFrame,
        clusters_df: pd.DataFrame,
        max_samples: int = 3,
        max_chars: int = 320
) -> dict[int, list[str]]:
    merged = messages_df.merge(
        clusters_df[["message_id", "cluster"]],
        on="message_id",
        how="inner"
    )
    merged = merged[merged["cluster"] != -1]
    if merged.empty or "text" not in merged.columns:
        return {}
    if "date" in merged.columns:
        merged["date_dt"] = pd.to_datetime(merged["date"], errors="coerce")
    else:
        merged["date_dt"] = pd.NaT

    samples: dict[int, list[str]] = {}
    for cid, group in merged.groupby("cluster"):
        group = group.copy()
        if "date_dt" in group:
            group.sort_values(["date_dt"], ascending=[False], inplace=True)
        lengths = group["text"].astype(str).str.len()
        group = group.assign(_len=lengths.fillna(0))
        group.sort_values(["date_dt", "_len"], ascending=[False, False], inplace=True)
        texts: list[str] = []
        for text in group["text"].astype(str):
            snippet = re.sub(r"\s+", " ", text).strip()
            if not snippet:
                continue
            if len(snippet) > max_chars:
                snippet = snippet[:max_chars].rstrip() + "..."
            texts.append(snippet)
            if len(texts) >= max_samples:
                break
        if texts:
            samples[int(cid)] = texts
    return samples


def extract_cluster_metadata_with_llm(
        cluster_terms: dict[int, list[str]],
        cluster_samples: dict[int, list[str]],
        batch_size: int = 6
) -> dict[int, dict[str, Any]]:
    if not has_yandex_gpt_credentials():
        return {}
    candidate_ids = sorted({cid for cid in cluster_terms if cid != -1} & set(cluster_samples.keys()))
    if not candidate_ids:
        return {}
    client = get_yandex_gpt_client()
    metadata: dict[int, dict[str, Any]] = {}
    system_prompt = (
        "Ты аналитик по финтех-трендам. Для каждого кластера сформируй структурные признаки. "
        "Верни JSON-массив, где каждый объект содержит поля: "
        "cluster_id (int), content_type (news|analysis|promo|contest|digest|qa|spam|other), "
        "is_promo (bool), has_numbers (bool), has_entities (bool), sentiment (positive|neutral|negative), "
        "key_entities (list[str], до 4 элементов), confidence (float от 0 до 1)."
    )
    for chunk_ids in _chunked(candidate_ids, batch_size):
        items = []
        for cid in chunk_ids:
            terms = ", ".join(cluster_terms.get(cid, [])[:8])
            samples = cluster_samples.get(cid, [])[:3]
            items.append({
                "cluster_id": cid,
                "key_terms": terms,
                "sample_messages": samples,
            })
        user_payload = json.dumps(items, ensure_ascii=False, indent=2)
        messages = [
            {"role": "system", "content": system_prompt + " Отвечай строго JSON без пояснений."},
            {
                "role": "user",
                "content": (
                    "Данные кластеров:\n" + user_payload +
                    "\nОтветь JSON-массивом с указанными полями."
                ),
            },
        ]
        try:
            logger.info(
                "LLM extractor запрос: %d кластеров (ids=%s)",
                len(chunk_ids),
                ",".join(str(cid) for cid in chunk_ids),
            )
            resp = client.chat.completions.create(
                model=cfg.chat_model,
                messages=messages,
                temperature=0.1,
                max_tokens=600
            )
            text = resp.choices[0].message.content.strip()
            parsed = _parse_json_from_text(text)
            if not parsed:
                logger.warning("LLM extractor: не удалось распарсить ответ для chunk=%s", chunk_ids)
                continue
            for item in parsed:
                if not isinstance(item, dict):
                    continue
                try:
                    cid = int(item.get("cluster_id"))
                except (TypeError, ValueError):
                    continue
                metadata[cid] = _normalize_cluster_metadata(item)
            logger.info("LLM extractor: получили метаданные для %d кластеров", len(parsed))
        except Exception as exc:
            logger.exception("Ошибка LLM extractor (chunk=%s): %s", chunk_ids, exc)
    if metadata:
        logger.info("LLM extractor: всего получено метаданных для %d кластеров", len(metadata))
    else:
        logger.warning("LLM extractor: не удалось получить метаданные для кластеров")
    return metadata
def extract_cluster_terms(
        messages_df: pd.DataFrame,
        clusters_df: pd.DataFrame,
        top_k: int = 12,
        include_noise: bool = False
) -> dict[int, list[str]]:
    df = messages_df.merge(clusters_df, on="message_id", how="inner")
    if not include_noise:
        df = df[df["cluster"] != -1]
    if len(df) == 0:
        print("Нет данных после merge или все кластеры — шум")
        return {}
    df = df.reset_index(drop=True)
    message_id_to_idx = {mid: idx for idx, mid in enumerate(df["message_id"])}
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        stop_words=None,
        min_df=1,
        max_df=0.95
    )
    try:
        tfidf = vectorizer.fit_transform(df["text"].astype(str).values)
        vocab = np.array(vectorizer.get_feature_names_out())
    except ValueError as e:
        print(f"Ошибка TF-IDF: {e}. Используем fallback по словам.")
        tfidf = None
        vocab = np.array([])
    terms: dict[int, list[str]] = {}
    for cid, sub in df.groupby("cluster"):
        if len(sub) == 0:
            continue
        texts = sub["text"].astype(str).tolist()
        fallback_terms = _fallback_terms_from_texts(texts, top_k)
        cluster_terms: list[str]
        if tfidf is not None and vocab.size > 0:
            idx = [message_id_to_idx[mid] for mid in sub["message_id"]]
            cluster_tfidf = tfidf[idx]
            if cluster_tfidf.nnz > 0:
                scores = cluster_tfidf.mean(axis=0).A1
                top_idx = np.argsort(scores)[-top_k:][::-1]
                cluster_terms = vocab[top_idx].tolist()
                cluster_terms = [term for term in cluster_terms if len(term) > 2]
                if not cluster_terms:
                    cluster_terms = fallback_terms
            else:
                cluster_terms = fallback_terms
        else:
            cluster_terms = fallback_terms
        if not cluster_terms:
            cluster_terms = fallback_terms
        terms[int(cid)] = cluster_terms[:top_k]
    return terms


@retry(wait=wait_exponential(min=1, max=50), stop=stop_after_attempt(6))
def label_clusters_with_llm(cluster_terms: dict[int, list[str]]) -> dict[int, str]:
    if not has_yandex_gpt_credentials():
        return {cid: ", ".join(terms[:3]) for cid, terms in cluster_terms.items()}
    client = get_yandex_gpt_client()
    items = []
    for cid, terms in sorted(cluster_terms.items()):
        if cid == -1:
            continue
        items.append({
            "cluster_id": cid,
            "key_terms": ", ".join(terms[:8])
        })
    if not items:
        return {}
    logger.info("LLM разметка кластеров: %d элементов", len(items))
    prompt = (
        "Ты эксперт по анализу трендов ИИ и Финансов тем. "
        "Для каждого кластера ключевых терминов создай краткое, точное название тренда (3-6 слов). "
        "Фокусируйся на инвестиционной и финтех тематике. "
        "Примеры хороших названий: 'Рост криптовалют', 'ESG инвестиции', 'Цифровые CBDC', 'Ипотечные ставки'. "
        "В обязательном порядке разметь каждый кластер и определи тему. "
        "НЕ ПИШИ НИКАКОГО ВСТУПЛЕНИЯ! СРАЗУ НАЧИНАЙ С НАЗВАНИЙ ТРЕНДОВ! Если в кластере нет тренда, то называй его \"Нет релевантного тренда\""
    )
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": f"Кластеры терминов для анализа:\n{items}"},
    ]
    try:
        resp = client.chat.completions.create(
            model=cfg.chat_model,
            messages=messages,
            temperature=0.2,
            max_tokens=500
        )
        text = resp.choices[0].message.content.strip()
        lines = [l.strip(" -:\t") for l in text.splitlines() if l.strip()]
        labels: dict[int, str] = {}
        non_noise_terms = [(cid, t) for cid, t in sorted(cluster_terms.items()) if cid != -1]
        for (cid, _), line in zip(non_noise_terms, lines):
            clean_line = line.split(":")[-1].strip(" -\t\"'")
            labels[int(cid)] = clean_line
        logger.info("LLM разметка: успешно получили %d строк", len(lines))
    except Exception as e:
        logger.exception("Ошибка LLM разметки: %s", e)
        labels = {}
    for cid, terms in cluster_terms.items():
        if cid != -1 and int(cid) not in labels:
            labels[int(cid)] = ", ".join(terms[:3])
    return labels


def run_topic_modeling(
        messages_df: pd.DataFrame,
        emb_df: pd.DataFrame,
        n_components: int = 50,
        eps: Optional[float] = None,
        min_samples: Optional[int] = None,
        metric: str = 'cosine',
        normalize: bool = False,
        enable_snn: bool = True,
        snn_k: int = 8,
        snn_threshold: int = 3,
        snn_min_cluster_size: int = 2,
        snn_max_cluster_size: int = 5,
        snn_min_cosine: float = 0.7,
        snn_id_offset: int = 0,
        triage_top_m: int = 10,
        triage_half_life_days: float = 14.0
) -> tuple[pd.DataFrame, pd.DataFrame, dict[int, str], dict, dict[int, list[str]]]:
    """
    Полный пайплайн тематического моделирования с DBSCAN
    """
    messages_df["message_id"] = messages_df["message_id"].astype(str)
    emb_df["message_id"] = emb_df["message_id"].astype(str)

    print("Проекция эмбеддингов...")
    proj_df = project_embeddings(emb_df, n_components=n_components)
    save_parquet(proj_df, cfg.projections_parquet)

    print("DBSCAN кластеризация...")
    clusters_df, clustering_info = cluster_embeddings_dbscan(
        proj_df, eps=eps, min_samples=min_samples, metric=metric, normalize=normalize
    )
    save_parquet(clusters_df, cfg.clusters_parquet)

    clusters_df["cluster_strength"] = np.where(clusters_df["cluster"] == -1, "singleton", "strong")


    print("Анализ качества кластеризации...")
    quality_metrics = analyze_cluster_quality(clusters_df)
    quality_metrics.update({
        "dbscan_eps": float(clustering_info.get("eps", 0.0)),
        "dbscan_min_samples": int(clustering_info.get("min_samples", 0)),
        "dbscan_cluster_count": int(clustering_info.get("cluster_count", 0)),
        "dbscan_noise_ratio": float(clustering_info.get("noise_ratio", 0.0)),
        "dbscan_auto_tuned": bool(clustering_info.get("auto_tuned", False)),
    })
#    save_json(quality_metrics, cfg.artifacts_dir / "quality_metrics.json")

    weak_clusters: list[WeakClusterSummary] = []
    top_clusters_preview: list[dict[str, Any]] = []
    cluster_scores_summary: dict[str, Any] = {}
    if enable_snn and (clusters_df["cluster"] == -1).any():
        print("SNN дообработка шума...")
        # Сохраняем исходные сильные кластеры (ID неизменяемы)
        orig_clusters = clusters_df[["message_id", "cluster"]].copy()
        orig_clusters["message_id"] = orig_clusters["message_id"].astype(str)
        strong_mask_orig = orig_clusters["cluster"] != -1
        orig_restore_map = dict(zip(orig_clusters.loc[strong_mask_orig, "message_id"], orig_clusters.loc[strong_mask_orig, "cluster"].astype(int)))
        clusters_df, weak_clusters = cluster_noise_with_snn(
            proj_df,
            clusters_df,
            k=snn_k,
            snn_threshold=snn_threshold,
            min_cluster_size=snn_min_cluster_size,
            max_cluster_size=snn_max_cluster_size,
            min_cosine=snn_min_cosine,
            split_large=True,
            id_offset=snn_id_offset,
            keep_strong_immutable=True,
        )

        clusters_df["message_id"] = clusters_df["message_id"].astype(str)
        need_restore_mask = clusters_df["message_id"].isin(orig_restore_map.keys())
        before_unique = int(clusters_df.loc[need_restore_mask, "cluster"].nunique())
        clusters_df.loc[need_restore_mask, "cluster"] = (
            clusters_df.loc[need_restore_mask, "message_id"].map(orig_restore_map).astype(int).values
        )
        clusters_df.loc[need_restore_mask, "cluster_strength"] = "strong"
        after_unique = int(clusters_df.loc[need_restore_mask, "cluster"].nunique())
        if after_unique != before_unique:
            print(f"SNN guard: restored strong clusters (unique {before_unique} -> {after_unique})")
    else:
        clusters_df["is_noise"] = clusters_df["cluster"] == -1
        clusters_df.loc[clusters_df["cluster"] == -1, "cluster_strength"] = "singleton"

    save_parquet(clusters_df, cfg.clusters_parquet)

    print("Анализ качества после SNN...")
    quality_metrics = analyze_cluster_quality(clusters_df)
    quality_metrics.update({
        "dbscan_eps": float(clustering_info.get("eps", 0.0)),
        "dbscan_min_samples": int(clustering_info.get("min_samples", 0)),
        "dbscan_cluster_count": int(clustering_info.get("cluster_count", 0)),
        "dbscan_noise_ratio": float(clustering_info.get("noise_ratio", 0.0)),
        "dbscan_auto_tuned": bool(clustering_info.get("auto_tuned", False)),
    })
    quality_metrics["weak_clusters_count"] = len(weak_clusters)
    quality_metrics["weak_clusters"] = [
        {
            "cluster_id": summary.cluster_id,
            "size": summary.size,
            "members": summary.members,
            "mean_similarity": float(summary.mean_similarity),
            "snn_threshold": summary.snn_threshold,
        }
        for summary in weak_clusters
    ]

    if quality_metrics["weak_clusters"]:
        try:
            save_json({"clusters": quality_metrics["weak_clusters"]}, cfg.weak_clusters_json)
            quality_metrics["weak_clusters_path"] = str(cfg.weak_clusters_json)
        except Exception as exc:
            print(f"Не удалось сохранить weak_clusters.json: {exc}")

    print(f"Найдено {quality_metrics['unique_clusters']} кластеров, "
          f"шум: {quality_metrics['noise_ratio']:.1%}")

    print("Триаж одиночек...")
    singleton_triage = triage_singletons(
        messages_df,
        proj_df,
        clusters_df,
        top_m=triage_top_m,
        freshness_half_life_days=triage_half_life_days
    )
    quality_metrics["singleton_triage_total"] = int(len(singleton_triage))
    quality_metrics["singleton_triage_priority"] = int((singleton_triage.get("triage_bucket") == "priority").sum())
 #   if not singleton_triage.empty:
        # save_parquet(singleton_triage, cfg.singleton_triage_parquet)
        # quality_metrics["singleton_triage_path"] = str(cfg.singleton_triage_parquet)

    print("Оценка полезности сообщений...")
    message_usefulness = compute_message_usefulness(messages_df, clusters_df)
    if not message_usefulness.empty:
        save_parquet(message_usefulness, cfg.message_usefulness_parquet)
        quality_metrics["message_usefulness_path"] = str(cfg.message_usefulness_parquet)
        try:
            print(f"Message usefulness: {len(message_usefulness)} записей → {cfg.message_usefulness_parquet}")
        except Exception:
            pass

    print("Извлечение терминов для кластеров...")
    cluster_terms = extract_cluster_terms(messages_df, clusters_df)

    cluster_samples = collect_cluster_samples(messages_df, clusters_df)
    cluster_metadata = extract_cluster_metadata_with_llm(cluster_terms, cluster_samples)
    if cluster_metadata:
        try:
            save_json({str(k): v for k, v in cluster_metadata.items()}, cfg.cluster_metadata_json)
            quality_metrics["cluster_llm_metadata_path"] = str(cfg.cluster_metadata_json)
            preview_meta = {}
            for cid, meta in list(cluster_metadata.items())[:15]:
                preview_meta[str(cid)] = {
                    "content_type": meta.get("content_type"),
                    "is_promo": bool(meta.get("is_promo")),
                    "has_numbers": bool(meta.get("has_numbers")),
                    "has_entities": bool(meta.get("has_entities")),
                    "sentiment": meta.get("sentiment"),
                    "key_entities": meta.get("key_entities", [])[:4],
                }
            quality_metrics["cluster_llm_metadata_head"] = preview_meta
        except Exception as exc:
            print(f"Не удалось сохранить cluster_metadata.json: {exc}")

    print("Скоринг кластеров (usefulness / hot score)...")
    cluster_scores = score_clusters(message_usefulness, clusters_df, cluster_metadata=cluster_metadata)
    def _records_for_json(df: pd.DataFrame, limit: int | None = None) -> list[dict[str, Any]]:
        subset = df.head(limit).copy() if limit else df.copy()
        if "llm_key_entities" in subset.columns:
            subset["llm_key_entities"] = subset["llm_key_entities"].apply(
                lambda v: v if isinstance(v, list) else []
            )
        subset = subset.replace({np.nan: None})
        return subset.to_dict(orient="records")

    if not cluster_scores.empty:
        cluster_scores_to_save = cluster_scores.copy()
        if "llm_key_entities" in cluster_scores_to_save.columns:
            cluster_scores_to_save["llm_key_entities"] = cluster_scores_to_save["llm_key_entities"].apply(
                lambda v: json.dumps(v, ensure_ascii=False) if isinstance(v, list) else json.dumps([])
            )
        cluster_scores_to_save = cluster_scores_to_save.replace({np.nan: None})
        save_parquet(cluster_scores_to_save, cfg.cluster_usefulness_parquet)
        quality_metrics["cluster_usefulness_path"] = str(cfg.cluster_usefulness_parquet)
        top_clusters_preview = _records_for_json(cluster_scores, 10)
        quality_metrics["cluster_scores_top"] = _records_for_json(cluster_scores, 20)
        cluster_scores_summary = {
            "content": int((cluster_scores["format_label"] == "content").sum()),
            "format": int((cluster_scores["format_label"] == "format/giveaway").sum()),
            "path": str(cfg.cluster_usefulness_parquet),
        }
        format_clusters = cluster_scores.loc[
            cluster_scores["format_label"] == "format/giveaway", "cluster"
        ].tolist()
        if format_clusters:
            clusters_df.loc[clusters_df["cluster"].isin(format_clusters), "cluster_strength"] = "format"

    save_parquet(clusters_df, cfg.clusters_parquet)
    save_json(quality_metrics, cfg.artifacts_dir / "quality_metrics.json")

    print(
        "Оценка одиночек завершена: "
        f"{quality_metrics['singleton_triage_total']} сообщений, "
        f"{quality_metrics['singleton_triage_priority']} в приоритете"
    )


    save_json({str(k): v for k, v in cluster_terms.items()}, cfg.artifacts_dir / "cluster_terms.json")

    print("LLM разметка кластеров...")
    labels = label_clusters_with_llm(cluster_terms)

    labels_with_noise = labels.copy()
    if -1 in cluster_terms:
        labels_with_noise[-1] = "Нерелевантные сообщения"

    if top_clusters_preview:
        print("Top clusters by usefulness/hotness:")
        for idx, row in enumerate(top_clusters_preview, start=1):
            cid_raw = row.get("cluster")
            cid = int(cid_raw) if cid_raw is not None and not pd.isna(cid_raw) else -1
            label_text = labels_with_noise.get(cid) or labels.get(cid) or f"Cluster {cid}"
            use_val = row.get("usefulness_cluster")
            hot_val = row.get("hot_score")
            size_val = row.get("size", 0)
            format_label = row.get("format_label", "")
            strength = row.get("cluster_strength", "")
            content_type = row.get("llm_content_type", "")
            sentiment = row.get("llm_sentiment", "")
            is_promo_llm = bool(row.get("llm_is_promo"))
            use_str = f"{float(use_val):.2f}" if use_val is not None and not pd.isna(use_val) else "na"
            hot_str = f"{float(hot_val):.2f}" if hot_val is not None and not pd.isna(hot_val) else "na"
            try:
                size_int = int(size_val)
            except (TypeError, ValueError):
                size_int = 0
            tag_parts = [part for part in [strength, format_label] if part]
            if content_type:
                tag_parts.append(f"type:{content_type}")
            if sentiment:
                tag_parts.append(f"sent:{sentiment}")
            if is_promo_llm and "promo" not in " ".join(tag_parts):
                tag_parts.append("promo-llm")
            tag_str = f" ({', '.join(tag_parts)})" if tag_parts else ""
            key_entities = row.get("llm_key_entities") or []
            if isinstance(key_entities, str):
                key_entities = [key_entities]
            ent_str = ""
            if key_entities:
                ent_str = " | entities=" + ", ".join(str(e) for e in key_entities[:3])
            print(
                f"  {idx:>2}. [{cid}] {label_text} — use={use_str}, hot={hot_str}, size={size_int}{tag_str}{ent_str}"
            )

    if cluster_scores_summary:
        content_cnt = cluster_scores_summary.get("content")
        format_cnt = cluster_scores_summary.get("format")
        path_val = cluster_scores_summary.get("path")
        print(
            "Cluster scores saved → "
            f"{path_val} | content={content_cnt}, format/promos={format_cnt}"
        )

    save_json({str(k): v for k, v in labels_with_noise.items()}, cfg.cluster_labels_json)

    return proj_df, clusters_df, labels, quality_metrics, cluster_terms


def load_topic_artifacts() -> tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[dict[int, str]], Optional[dict], Optional[dict[int, list[str]]]]:
    """Загрузка сохраненных артефактов тематического моделирования"""
    proj = load_parquet(cfg.projections_parquet)
    clus = load_parquet(cfg.clusters_parquet)
    labels_raw = load_json(cfg.cluster_labels_json)
    labels = {int(k): v for k, v in labels_raw.items()} if labels_raw else None
    quality_metrics = load_json(cfg.artifacts_dir / "quality_metrics.json") or {}
    cluster_terms_raw = load_json(cfg.artifacts_dir / "cluster_terms.json")
    cluster_terms = {int(k): v for k, v in cluster_terms_raw.items()} if cluster_terms_raw else None
    return proj, clus, labels, quality_metrics, cluster_terms


def load_singleton_triage() -> Optional[pd.DataFrame]:
    """Получить результаты триажа одиночек, если файл уже сохранён."""
    return load_parquet(cfg.singleton_triage_parquet)


def load_cluster_scores() -> Optional[pd.DataFrame]:
    """Загрузить агрегированные скоринги кластеров."""
    return load_parquet(cfg.cluster_usefulness_parquet)


def load_message_usefulness() -> Optional[pd.DataFrame]:
    """Загрузить полезность сообщений, если она была рассчитана."""
    return load_parquet(cfg.message_usefulness_parquet)


def load_weak_clusters() -> list[dict]:
    """Загрузить описание слабых кластеров (SNN) из артефактов."""
    raw = load_json(cfg.weak_clusters_json)
    if not raw:
        return []
    clusters = raw.get("clusters", [])
    return clusters if isinstance(clusters, list) else []


def suggest_dbscan_params(emb_df: pd.DataFrame, n_components: int = 50, metric: str = 'cosine') -> dict:
    proj_df = project_embeddings(emb_df, n_components=n_components)
    feat_cols = [c for c in proj_df.columns if c.startswith("p")]
    if not feat_cols:
        raise ValueError("Projection DataFrame must contain columns starting with 'p'")
    X = proj_df[feat_cols].values
    if len(proj_df) < 3:
        dims = int(X.shape[1]) if X.ndim == 2 else 0
        return {
            "suggested_eps": 0.3,
            "suggested_min_samples": 3,
            "data_points": int(len(X)),
            "dimensions": dims,
            "auto_tuned": False,
        }
    auto_eps, auto_min_samples = _auto_dbscan_params(X, metric=metric)
    attempt = _adaptive_dbscan_search(
        X,
        metric,
        auto_eps,
        auto_min_samples,
        allow_eps_variation=True,
        allow_min_samples_variation=True,
    )
    return {
        "suggested_eps": float(attempt.eps),
        "suggested_min_samples": int(attempt.min_samples),
        "base_eps": float(auto_eps),
        "base_min_samples": int(auto_min_samples),
        "data_points": int(len(X)),
        "dimensions": int(X.shape[1]),
        "estimated_clusters": int(attempt.cluster_count),
        "estimated_noise_ratio": float(attempt.noise_ratio),
        "auto_tuned": True,
    }
