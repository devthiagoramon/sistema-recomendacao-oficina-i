"""Interface Streamlit do sistema de recomendação. Execute: streamlit run app.py"""
import sys
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from recsys.data import (append_rating, dataset_summary, load_config,  # noqa: E402
                         load_dataset, next_user_id)
from recsys.models import ItemKNN, PopularityRecommender, UserKNN  # noqa: E402

st.set_page_config(page_title="Recomendador", page_icon="🎬", layout="wide")

NEW_USER = "Novo usuário (sem histórico)"

st.markdown("""
<style>
.block-container {padding-top: 2rem; max-width: 1150px;}
.card {background:#1A1D26; border:1px solid #2A2E3B; border-radius:12px;
       padding:14px 18px; margin-bottom:10px;}
.card .title {font-size:1.05rem; font-weight:600;}
.card .rank {color:#E50914; font-weight:700; margin-right:8px;}
.card .why {color:#9AA0AE; font-size:.85rem; margin-top:4px;}
.chip {display:inline-block; background:#2A2E3B; color:#C9CEDA; border-radius:999px;
       padding:1px 10px; font-size:.75rem; margin:6px 6px 0 0;}
.score {float:right; color:#9AA0AE; font-size:.85rem;}
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner="Carregando e tratando os dados...")
def get_data(version: float):
    cfg = load_config()
    ds, report = load_dataset(cfg)
    return cfg, ds, report


@st.cache_resource(show_spinner="Treinando o modelo...")
def get_model(kind: str, version: float):
    cfg, ds, _ = get_data(version)
    m = cfg["model"]
    model = {
        "Item-based CF": lambda: ItemKNN(m["neighbors"], m["shrinkage"]),
        "User-based CF": lambda: UserKNN(m["neighbors"], m["shrinkage"]),
        "Popularidade": lambda: PopularityRecommender(m["popularity_min_votes"]),
    }[kind]()
    return model.fit(ds)


def data_version() -> float:
    """Muda quando novas avaliações são gravadas -> invalida os caches."""
    cfg = load_config()
    p = ROOT / cfg["storage"]["new_ratings_path"]
    return p.stat().st_mtime if p.exists() else 0.0


def chips(extra: str) -> str:
    return "".join(f'<span class="chip">{escape(g)}</span>'
                   for g in extra.split("|") if g and g != "nan")


def card(title: str, extra: str = "", rank: int | None = None,
         right: str = "", why: str = "") -> str:
    r = f'<span class="rank">{rank}</span>' if rank else ""
    w = f'<div class="why">{escape(why)}</div>' if why else ""
    return (f'<div class="card"><span class="score">{escape(right)}</span>'
            f'<div class="title">{r}{escape(title)}</div>{chips(extra)}{w}</div>')


version = data_version()
cfg, ds, report = get_data(version)

# ---------- Barra lateral ----------
st.sidebar.title("🎬 Recomendador")
st.sidebar.caption(cfg["dataset"]["name"])
st.session_state.setdefault("user_select", ds.user_ids.tolist()[0])
if st.session_state["user_select"] not in [NEW_USER, *ds.user_ids.tolist()]:
    st.session_state["user_select"] = NEW_USER
user_choice = st.sidebar.selectbox("Usuário", [NEW_USER, *ds.user_ids.tolist()], key="user_select")
model_kind = st.sidebar.selectbox("Modelo", ["Item-based CF", "User-based CF", "Popularidade"])
n_recs = st.sidebar.slider("Quantidade de recomendações", 5, 30, 10)
with st.sidebar.expander("Sobre a base"):
    st.json(dataset_summary(ds))

model = get_model(model_kind, version)
user_id = None if user_choice == NEW_USER else user_choice
user_hist = (ds.df[ds.df["user"] == user_id].sort_values(["rating", "timestamp"], ascending=False)
             if user_id is not None else ds.df.iloc[0:0])

st.title("Sistema de Recomendação")
st.caption("Filtragem colaborativa · recomendações personalizadas sem repetir itens já conhecidos")

tab_hist, tab_recs, tab_rate, tab_res = st.tabs(
    ["📚 Histórico", "✨ Recomendações", "⭐ Avaliar itens", "📊 Resultados"])

# ---------- Histórico ----------
with tab_hist:
    if user_hist.empty:
        st.info("Este usuário ainda não tem histórico. As recomendações usarão os itens mais "
                "populares (cold-start).")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Itens avaliados", len(user_hist))
        c2.metric("Nota média", f"{user_hist['rating'].mean():.2f}")
        c3.metric("Itens que gostou (≥ %g)" % cfg["dataset"]["relevance_threshold"],
                  int((user_hist["rating"] >= cfg["dataset"]["relevance_threshold"]).sum()))
        left, right = st.columns([3, 2])
        with left:
            st.subheader("Itens avaliados")
            for _, r in user_hist.head(30).iterrows():
                st.markdown(card(ds.title(r["item"]), ds.extra(r["item"]),
                                 right=f"★ {r['rating']:g}"), unsafe_allow_html=True)
            if len(user_hist) > 30:
                st.caption(f"Mostrando 30 de {len(user_hist)} itens (os mais bem avaliados).")
        with right:
            st.subheader("Distribuição das notas")
            st.bar_chart(user_hist["rating"].value_counts().sort_index(), color="#E50914")

# ---------- Recomendações ----------
with tab_recs:
    recs = model.recommend(user_id, n_recs)
    if user_hist.empty:
        st.warning("Usuário sem histórico: exibindo os itens mais populares "
                   "(média bayesiana das notas).")
    st.subheader(f"Top {len(recs)} para {user_choice if user_id is not None else 'novo usuário'}")
    for pos, (item, score) in enumerate(recs, 1):
        why = ""
        if isinstance(model, ItemKNN) and not user_hist.empty:
            ex = model.explain(user_id, item, 2)
            if ex:
                why = "Porque você avaliou: " + "; ".join(ds.title(j) for j, _ in ex)
        st.markdown(card(ds.title(item), ds.extra(item), rank=pos,
                         right=f"score {score:.2f}", why=why), unsafe_allow_html=True)

# ---------- Avaliar itens ----------
def save_rating(item_id, rating):
    """Callback do botão: grava a nota; usuário novo ganha um id e passa a ser o selecionado."""
    uid = user_id
    if uid is None:
        uid = next_user_id(ds)
        st.session_state["user_select"] = uid
    append_rating(cfg, uid, item_id, rating)
    st.session_state["saved_msg"] = f"Avaliação de «{ds.title(item_id)}» salva para o usuário {uid}."


with tab_rate:
    st.subheader("Avaliar um item")
    if user_id is None:
        st.info("Você está como **novo usuário**. Ao salvar a primeira avaliação, um usuário é "
                "criado e as recomendações passam a ser personalizadas.")
    if msg := st.session_state.pop("saved_msg", None):
        st.success(msg + " Veja as novas recomendações na aba ✨.")
    query = st.text_input("Buscar item pelo título", placeholder="ex.: matrix, toy story...")
    if query:
        found = ds.items[ds.items["title"].str.contains(query, case=False, regex=False)]
        found = found[found.index.isin(ds.item_ids)].head(20)
    else:
        found = ds.items.iloc[0:0]
    if query and found.empty:
        st.warning("Nenhum item encontrado.")
    if not found.empty:
        item_id = st.selectbox("Item", found.index.tolist(), format_func=ds.title)
        lo, hi = cfg["dataset"]["rating_scale"]
        step = 0.5 if lo % 1 else 1.0
        already = user_hist[user_hist["item"] == item_id]["rating"]
        default = float(already.iloc[0]) if len(already) else round((lo + hi) / 2 / step) * step
        rating = st.slider("Sua nota", float(lo), float(hi), default, step)
        if len(already):
            st.caption(f"Você já avaliou este item com {already.iloc[0]:g}; salvar substitui a nota.")
        st.button("Salvar avaliação", type="primary", on_click=save_rating, args=(item_id, rating))

# ---------- Resultados ----------
with tab_res:
    st.subheader("Avaliação offline dos modelos")
    metrics_path = ROOT / "results" / "metrics.csv"
    if metrics_path.exists():
        res = pd.read_csv(metrics_path, index_col="Modelo")
        st.dataframe(res.style.format("{:.4f}").highlight_max(axis=0, color="#4a1418"),
                     width="stretch")
        rank_cols = [c for c in res.columns if c.split("@")[0] in ("Precision", "Recall", "NDCG")]
        st.bar_chart(res[rank_cols])
        st.caption("Hold-out temporal: os 20% de avaliações mais recentes de cada usuário formam "
                   "o teste. Itens com nota ≥ limiar contam como relevantes. RMSE/MAE: quanto "
                   "menor, melhor; as demais métricas: quanto maior, melhor.")
    else:
        st.info("Rode `python scripts/evaluate.py` para gerar as métricas.")
    with st.expander("Tratamento dos dados (linhas restantes após cada etapa)"):
        st.json(report)
