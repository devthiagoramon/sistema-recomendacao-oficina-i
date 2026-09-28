"""Interface Streamlit do sistema de recomendação. Execute: streamlit run app.py"""
import sys
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from recsys.data import dataset_summary, load_config, load_dataset  # noqa: E402
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
def get_data(_version: float):
    cfg = load_config()
    ds, report = load_dataset(cfg)
    return cfg, ds, report


@st.cache_resource(show_spinner="Treinando o modelo...")
def get_model(kind: str, _version: float):
    cfg, ds, _ = get_data(_version)
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
user_choice = st.sidebar.selectbox("Usuário", [NEW_USER, *ds.user_ids.tolist()], index=1)
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

tab_hist, tab_recs = st.tabs(["📚 Histórico", "✨ Recomendações"])

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
