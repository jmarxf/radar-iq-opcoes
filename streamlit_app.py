
from __future__ import annotations

import io
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

APP_VERSION = "0.1.1-online"
PULSEFLAT_URL = (
    "https://raw.githubusercontent.com/PulseDataLabs/PulseFlat/"
    "main/data/b3_opcoes_posicoes_aberto.csv.gz"
)
HISTORY_PATH = Path("data/historico_iq.csv")
HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)

CALL_MONTH_CODES = set("ABCDEFGHIJKL")
PUT_MONTH_CODES = set("MNOPQRSTUVWX")


def norm(txt: str) -> str:
    txt = unicodedata.normalize("NFKD", str(txt))
    txt = "".join(c for c in txt if not unicodedata.combining(c))
    txt = txt.lower().strip()
    return re.sub(r"[^a-z0-9]+", "_", txt).strip("_")


ALIASES = {
    "ticker": [
        "opcao", "ticker", "codigo_opcao", "codigo_da_opcao",
        "codigo_negociacao", "codigo_de_negociacao", "serie", "codigo_serie",
        "cod_serie", "symbol", "option_symbol",
    ],
    "strike": [
        "strike", "preco_exercicio", "preco_de_exercicio",
        "valor_exercicio", "exercise_price",
    ],
    "close": [
        "fechamento", "preco_fechamento", "cotacao_fechamento",
        "ultimo", "ultimo_preco", "close", "closing_price",
    ],
    "covered": [
        "coberto", "coberta", "posicao_coberta", "posicoes_cobertas",
        "qtd_coberta", "quantidade_coberta", "covered",
    ],
    "locked": [
        "trava", "travado", "travada", "posicao_travada",
        "posicoes_travadas", "qtd_travada", "quantidade_travada", "locked",
    ],
    "uncovered": [
        "descoberto", "descoberta", "posicao_descoberta",
        "posicoes_descobertas", "qtd_descoberta",
        "quantidade_descoberta", "uncovered", "naked",
    ],
    "total": [
        "total", "posicao_total", "posicoes_abertas", "posicao_aberta",
        "open_interest", "oi", "total_posicoes",
    ],
    "holders": [
        "titular", "titulares", "numero_titulares", "n_titulares",
        "qtd_titulares", "quantidade_titulares", "clientes_titulares",
        "holders", "holder_count",
    ],
    "writers": [
        "lancador", "lancadores", "numero_lancadores", "n_lancadores",
        "qtd_lancadores", "quantidade_lancadores", "clientes_lancadores",
        "writers", "writer_count",
    ],
    "reference_date": [
        "data_referencia", "data_de_referencia", "data", "reference_date",
        "data_pregao", "pregao",
    ],
    "underlying": [
        "ativo_objeto", "ativo_subjacente", "ativo", "underlying",
        "codigo_ativo", "ticker_ativo",
    ],
    "option_type": [
        "tipo_opcao", "tipo_de_opcao", "tipo", "call_put",
        "option_type", "natureza",
    ],
    "expiration": [
        "vencimento", "data_vencimento", "data_de_vencimento",
        "expiration", "expiry",
    ],
}


def find_column(df: pd.DataFrame, key: str) -> str | None:
    normalized = {norm(c): c for c in df.columns}
    for alias in ALIASES[key]:
        if alias in normalized:
            return normalized[alias]

    # Busca parcial conservadora.
    for nc, original in normalized.items():
        for alias in ALIASES[key]:
            if len(alias) >= 6 and (alias in nc or nc in alias):
                return original
    return None


def detect_schema(df: pd.DataFrame) -> dict[str, str | None]:
    return {k: find_column(df, k) for k in ALIASES}


def to_number(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")

    x = s.astype(str).str.strip()

    # Formato brasileiro: 1.234.567,89
    has_comma = x.str.contains(",", regex=False).mean() > 0.10
    if has_comma:
        x = x.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)

    x = x.str.replace(r"[^\d\.\-]", "", regex=True)
    return pd.to_numeric(x, errors="coerce")


def derive_root(ticker: str) -> str:
    ticker = str(ticker).upper().strip()
    m = re.match(r"^([A-Z]{4})", ticker)
    return m.group(1) if m else ""


def derive_type(ticker: str) -> str:
    ticker = str(ticker).upper().strip()
    if len(ticker) < 5:
        return "?"
    c = ticker[4]
    if c in CALL_MONTH_CODES:
        return "CALL"
    if c in PUT_MONTH_CODES:
        return "PUT"
    return "?"


def normalize_type(v) -> str:
    s = norm(v)
    if s in {"call", "c", "compra", "opcao_de_compra"}:
        return "CALL"
    if s in {"put", "p", "venda", "opcao_de_venda"}:
        return "PUT"
    if "call" in s or "compra" in s:
        return "CALL"
    if "put" in s or "venda" in s:
        return "PUT"
    return "?"


@st.cache_data(ttl=900, show_spinner=False)
def load_pulseflat() -> pd.DataFrame:
    # pandas reconhece gzip pela extensão.
    return pd.read_csv(PULSEFLAT_URL, compression="gzip", low_memory=False)


def prepare(df_raw: pd.DataFrame):
    schema = detect_schema(df_raw)

    required = ["ticker", "holders", "writers"]
    missing = [k for k in required if not schema.get(k)]
    if missing:
        return None, schema, missing

    out = pd.DataFrame()
    out["Opção"] = df_raw[schema["ticker"]].astype(str).str.upper().str.strip()

    for key, title in [
        ("strike", "Strike"),
        ("close", "Fechamento"),
        ("covered", "Coberto"),
        ("locked", "Trava"),
        ("uncovered", "Descoberto"),
        ("total", "Total"),
        ("holders", "Titular"),
        ("writers", "Lançador"),
    ]:
        col = schema.get(key)
        out[title] = to_number(df_raw[col]) if col else np.nan

    # Se Total não vier da fonte, soma posições quando possível.
    if out["Total"].isna().all():
        pieces = out[["Coberto", "Trava", "Descoberto"]].fillna(0)
        out["Total"] = pieces.sum(axis=1)

    # IQ = titulares / lançadores.
    out["IQ"] = np.where(
        out["Lançador"] > 0,
        out["Titular"] / out["Lançador"],
        np.nan,
    )

    underlying_col = schema.get("underlying")
    if underlying_col:
        raw_under = df_raw[underlying_col].astype(str).str.upper().str.strip()
        out["Ativo"] = raw_under.str.extract(r"([A-Z]{4}\d{1,2}|[A-Z]{4})", expand=False)
    else:
        out["Ativo"] = out["Opção"].map(derive_root)

    type_col = schema.get("option_type")
    if type_col:
        out["Tipo"] = df_raw[type_col].map(normalize_type)
        bad = out["Tipo"].eq("?")
        out.loc[bad, "Tipo"] = out.loc[bad, "Opção"].map(derive_type)
    else:
        out["Tipo"] = out["Opção"].map(derive_type)

    ref_col = schema.get("reference_date")
    if ref_col:
        out["Data"] = pd.to_datetime(df_raw[ref_col], errors="coerce").dt.date
    else:
        out["Data"] = pd.NaT

    exp_col = schema.get("expiration")
    if exp_col:
        out["Vencimento"] = pd.to_datetime(df_raw[exp_col], errors="coerce").dt.date
    else:
        out["Vencimento"] = pd.NaT

    # Percentual descoberto no OI.
    out["% Descoberto"] = np.where(
        out["Total"] > 0,
        100 * out["Descoberto"] / out["Total"],
        np.nan,
    )

    # Percentil de posição descoberta por ativo/tipo.
    out["Perc. Descoberto"] = (
        out.groupby(["Ativo", "Tipo"])["Descoberto"]
        .rank(pct=True, method="average")
        .mul(100)
    )

    out = out.replace([np.inf, -np.inf], np.nan)
    return out, schema, []


def classify_barrier(row, iq_attention: float, iq_strong: float, pctl_attention: float):
    iq = row.get("IQ")
    pctl = row.get("Perc. Descoberto")

    if pd.isna(iq):
        return "SEM IQ"

    # Não é probabilidade; é uma classificação operacional do método.
    if iq >= iq_strong and (pd.isna(pctl) or pctl >= pctl_attention):
        return "BARREIRA FORTE"
    if iq >= iq_attention:
        return "ATENÇÃO"
    if iq < 0.5:
        return "IQ MUITO BAIXO"
    return "NORMAL"


def latest_reference_date(df: pd.DataFrame):
    if "Data" not in df or df["Data"].isna().all():
        return None
    return max(d for d in df["Data"].dropna())


def load_history() -> pd.DataFrame:
    if not HISTORY_PATH.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(HISTORY_PATH, low_memory=False)
    except Exception:
        return pd.DataFrame()


def append_history(df_filtered: pd.DataFrame, reference_date):
    HISTORY_PATH.parent.mkdir(exist_ok=True)
    save = df_filtered[[
        "Opção", "Ativo", "Tipo", "Strike", "Titular", "Lançador",
        "IQ", "Coberto", "Trava", "Descoberto", "Total", "% Descoberto"
    ]].copy()

    save["data_snapshot"] = (
        str(reference_date) if reference_date else datetime.now().date().isoformat()
    )
    save["capturado_em_utc"] = datetime.now(timezone.utc).isoformat()

    old = load_history()
    combined = pd.concat([old, save], ignore_index=True)

    # Um registro por série/data.
    if {"Opção", "data_snapshot"}.issubset(combined.columns):
        combined = combined.drop_duplicates(
            subset=["Opção", "data_snapshot"], keep="last"
        )

    combined.to_csv(HISTORY_PATH, index=False)


def add_previous_iq(df: pd.DataFrame) -> pd.DataFrame:
    hist = load_history()
    if hist.empty or "IQ" not in hist.columns:
        df["IQ anterior"] = np.nan
        df["Δ IQ"] = np.nan
        return df

    hist["IQ"] = pd.to_numeric(hist["IQ"], errors="coerce")
    hist["data_snapshot"] = pd.to_datetime(hist["data_snapshot"], errors="coerce")
    hist = hist.dropna(subset=["Opção", "data_snapshot"]).sort_values("data_snapshot")

    # Último IQ salvo de cada opção.
    prev = hist.groupby("Opção", as_index=False).tail(1)[["Opção", "IQ"]]
    prev = prev.rename(columns={"IQ": "IQ anterior"})

    out = df.merge(prev, on="Opção", how="left")
    out["Δ IQ"] = out["IQ"] - out["IQ anterior"]
    return out


st.set_page_config(
    page_title="Radar IQ — Opções B3",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Radar IQ — Opções B3")
st.caption(
    f"MVP {APP_VERSION} • IQ = Titulares ÷ Lançadores • "
    "Radar de posições e possíveis barreiras; não é probabilidade de exercício."
)

st.info(
    "Versão online: os dados atuais são carregados diretamente da fonte gratuita. "
    "O histórico salvo no servidor é temporário e pode ser apagado quando a hospedagem reiniciar."
)

with st.sidebar:
    st.header("Fonte de dados")
    source = st.radio(
        "Escolha a fonte",
        ["PulseFlat (gratuita)", "Carregar CSV/CSV.GZ"],
        index=0,
    )

    upload = None
    if source == "Carregar CSV/CSV.GZ":
        upload = st.file_uploader(
            "Arquivo",
            type=["csv", "gz"],
            help="Pode usar um arquivo exportado de outra fonte para validar a tabela.",
        )

    st.divider()
    st.header("Parâmetros do radar")
    iq_attention = st.number_input(
        "IQ de atenção",
        min_value=0.10, max_value=20.0, value=1.00, step=0.10,
    )
    iq_strong = st.number_input(
        "IQ de barreira forte",
        min_value=0.10, max_value=20.0, value=1.80, step=0.10,
        help="Valor empírico configurável; não representa probabilidade.",
    )
    pctl_attention = st.slider(
        "Percentil mínimo de Descoberto",
        0, 100, 75, 5,
        help="Compara a posição descoberta com outras séries do mesmo ativo/tipo.",
    )

    st.divider()
    st.caption(
        "A leitura de 'barreira' é uma heurística operacional inspirada "
        "no método do vídeo; valide junto com gráfico, vencimento, liquidez e risco."
    )

try:
    if source == "PulseFlat (gratuita)":
        with st.spinner("Baixando posições em aberto..."):
            raw = load_pulseflat()
    else:
        if upload is None:
            st.info("Carregue um CSV ou CSV.GZ para continuar.")
            st.stop()
        raw = pd.read_csv(upload, compression="infer", low_memory=False)

except Exception as e:
    st.error("Não foi possível carregar a fonte.")
    st.code(str(e))
    st.info(
        "Se a internet estiver bloqueando o GitHub, baixe o CSV.GZ do PulseFlat "
        "e use a opção 'Carregar CSV/CSV.GZ'."
    )
    st.stop()

prepared, schema, missing = prepare(raw)

if missing:
    st.error(
        "O app não reconheceu todas as colunas essenciais: "
        + ", ".join(missing)
    )
    st.write("Colunas encontradas no arquivo:")
    st.code("\n".join(map(str, raw.columns)))
    st.write("Mapeamento detectado:")
    st.json({k: v for k, v in schema.items()})
    st.stop()

df = prepared.copy()
reference_date = latest_reference_date(df)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Séries carregadas", f"{len(df):,}".replace(",", "."))
m2.metric("Ativos", df["Ativo"].nunique())
m3.metric("Com IQ calculável", int(df["IQ"].notna().sum()))
m4.metric("Referência", str(reference_date) if reference_date else "não informada")

if reference_date:
    age = (datetime.now().date() - reference_date).days
    if age >= 4:
        st.warning(
            f"⚠️ A fonte informa referência em {reference_date} "
            f"({age} dias atrás). Não use o IQ como dado atual sem confirmar a atualização."
        )
    else:
        st.success(f"Fonte com referência em {reference_date}.")

st.subheader("Filtros")

c1, c2, c3, c4 = st.columns([1.4, 1, 1, 1])

with c1:
    ativo_input = st.text_input(
        "Ativo / raiz",
        value="ITSA4",
        placeholder="Ex.: ITSA4, PETR4, VALE3, COGN3",
    ).upper().strip()

root = ativo_input[:4] if ativo_input else ""

with c2:
    tipo = st.selectbox("Tipo", ["TODAS", "CALL", "PUT"])

with c3:
    iq_min = st.number_input(
        "IQ mínimo",
        min_value=0.0, max_value=100.0, value=0.0, step=0.10,
    )

with c4:
    so_descoberto = st.checkbox("Somente com Descoberto > 0", value=True)

filtered = df.copy()

if root:
    filtered = filtered[
        filtered["Ativo"].astype(str).str.startswith(root, na=False)
        | filtered["Opção"].astype(str).str.startswith(root, na=False)
    ]

if tipo != "TODAS":
    filtered = filtered[filtered["Tipo"] == tipo]

filtered = filtered[filtered["IQ"].fillna(-1) >= iq_min]

if so_descoberto:
    filtered = filtered[filtered["Descoberto"].fillna(0) > 0]

filtered["Destaque"] = filtered.apply(
    classify_barrier,
    axis=1,
    args=(iq_attention, iq_strong, pctl_attention),
)

filtered = add_previous_iq(filtered)

# Prioriza strike e depois IQ, sem transformar isso em recomendação.
sort_cols = [c for c in ["Tipo", "Strike", "IQ"] if c in filtered.columns]
if sort_cols:
    filtered = filtered.sort_values(sort_cols, ascending=[True, True, False][:len(sort_cols)])

st.subheader(f"Radar — {ativo_input or 'todos os ativos'}")

display_cols = [
    "Opção", "Tipo", "Strike", "Fechamento",
    "Coberto", "Trava", "Descoberto", "Total",
    "Titular", "Lançador", "IQ", "IQ anterior", "Δ IQ",
    "% Descoberto", "Perc. Descoberto", "Destaque",
]
display_cols = [c for c in display_cols if c in filtered.columns]

formatters = {
    "Strike": "{:.2f}",
    "Fechamento": "{:.2f}",
    "IQ": "{:.2f}",
    "IQ anterior": "{:.2f}",
    "Δ IQ": "{:+.2f}",
    "% Descoberto": "{:.1f}%",
    "Perc. Descoberto": "{:.0f}%",
}

st.dataframe(
    filtered[display_cols].style.format(
        {k: v for k, v in formatters.items() if k in display_cols},
        na_rep="—",
    ),
    use_container_width=True,
    height=520,
)

b1, b2, b3 = st.columns([1, 1, 2])

with b1:
    csv_bytes = filtered[display_cols].to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "Baixar tabela filtrada",
        data=csv_bytes,
        file_name=f"radar_iq_{root or 'b3'}.csv",
        mime="text/csv",
    )

with b2:
    if st.button("Salvar snapshot local"):
        append_history(filtered, reference_date)
        st.success("Snapshot salvo em data/historico_iq.csv.")

with b3:
    st.caption(
        "Salvar snapshots permite comparar IQ atual x anterior nas próximas execuções."
    )

st.subheader("Possíveis barreiras destacadas")

highlights = filtered[
    filtered["Destaque"].isin(["BARREIRA FORTE", "ATENÇÃO"])
].copy()

if highlights.empty:
    st.info("Nenhuma série atingiu os critérios configurados.")
else:
    cols = [
        "Opção", "Tipo", "Strike", "IQ", "Descoberto",
        "% Descoberto", "Perc. Descoberto", "Destaque"
    ]
    st.dataframe(
        highlights[[c for c in cols if c in highlights.columns]],
        use_container_width=True,
        hide_index=True,
    )

with st.expander("Diagnóstico da fonte / colunas detectadas"):
    st.json({k: v for k, v in schema.items()})
    st.write("Colunas originais:")
    st.code("\n".join(map(str, raw.columns)))

with st.expander("Como interpretar este MVP"):
    st.markdown(
        """
- **IQ = Titulares ÷ Lançadores.**
- O app **não converte IQ em probabilidade de exercício**.
- A classificação de barreira combina IQ com a concentração de posição
  descoberta dentro do próprio ativo/tipo.
- **Travas não recebem o mesmo peso**, porque podem representar estruturas
  com sentidos diferentes.
- Uma série isolada e muito diferente das vizinhas pode ser uma distorção;
  a leitura deve ser feita no conjunto dos strikes.
- O próximo passo do projeto é acrescentar vencimento, preço do ativo,
  distância do strike, histórico diário e alertas Telegram.
        """
    )
