"""
Predictor del Campeón del Mundial 2026
Modelo: Regresión Logística (extensión lineal para probabilidades)
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import plotly.express as px
import plotly.graph_objects as go

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Predictor Mundial 2026 ⚽",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTES
# Nota: usamos FIFA_Score = 211 - FIFA_Ranking para que mayor valor = mejor equipo
# ─────────────────────────────────────────────────────────────────────────────
RAW_FEATURES = [
    "FIFA_Ranking", "Elo_Rating", "Average_Age", "Squad_Value_MEUR",
    "Win_Rate_Last4Years", "Goals_Per_Game", "Previous_WorldCup_Appearances",
]
MODEL_FEATURES = [
    "FIFA_Score", "Elo_Rating", "Average_Age", "Squad_Value_MEUR",
    "Win_Rate_Last4Years", "Goals_Per_Game", "Previous_WorldCup_Appearances",
]
FEATURE_ES = {
    "FIFA_Score":                    "Puntuación FIFA (211 − Ranking)",
    "Elo_Rating":                    "Rating Elo",
    "Average_Age":                   "Edad Promedio (años)",
    "Squad_Value_MEUR":              "Valor Plantilla (M€)",
    "Win_Rate_Last4Years":           "Tasa de Victorias (4 años)",
    "Goals_Per_Game":                "Goles por Partido",
    "Previous_WorldCup_Appearances": "Apariciones en Mundiales",
}
FEATURE_DESC = {
    "FIFA_Score": (
        "Se calcula como **211 − Ranking FIFA**, de modo que **mayor valor = mejor posición**. "
        "Un equipo rankeado #1 tiene puntuación 210; uno rankeado #50 tiene 161."
    ),
    "Elo_Rating": (
        "Sistema de puntuación Elo adaptado al fútbol. Captura la fortaleza histórica "
        "ponderando rivales y resultados recientes."
    ),
    "Average_Age": (
        "Edad promedio de los jugadores convocados. Plantillas muy jóvenes (< 25 años) "
        "o muy veteranas (> 31) suelen rendir menos en torneos largos."
    ),
    "Squad_Value_MEUR": (
        "Valor de mercado total de la plantilla en millones de euros. "
        "Refleja la calidad individual de los jugadores."
    ),
    "Win_Rate_Last4Years": (
        "Fracción de partidos ganados en los últimos 4 años (0 a 1). "
        "Mide el rendimiento reciente y la forma actual del equipo."
    ),
    "Goals_Per_Game": (
        "Promedio de goles marcados por partido. "
        "Indica el potencial ofensivo del equipo."
    ),
    "Previous_WorldCup_Appearances": (
        "Número de Mundiales en los que el equipo ha participado históricamente. "
        "Refleja tradición y experiencia a nivel de torneos."
    ),
}
FEATURE_UNIT = {
    "FIFA_Score":                    (1, 210, 1),
    "Elo_Rating":                    (1700, 2300, 1),
    "Average_Age":                   (22.0, 34.0, 0.1),
    "Squad_Value_MEUR":              (50, 2000, 10),
    "Win_Rate_Last4Years":           (0.30, 1.00, 0.01),
    "Goals_Per_Game":                (0.50, 3.50, 0.05),
    "Previous_WorldCup_Appearances": (1, 25, 1),
}
TEAM_FLAGS = {
    "Argentina": "🇦🇷", "Brazil": "🇧🇷", "France": "🇫🇷", "England": "🏴󠁧󠁢󠁥󠁮󠁧󠁿",
    "Spain": "🇪🇸", "Germany": "🇩🇪", "Portugal": "🇵🇹", "Netherlands": "🇳🇱",
    "Belgium": "🇧🇪", "Croatia": "🇭🇷", "Uruguay": "🇺🇾", "Colombia": "🇨🇴",
    "Morocco": "🇲🇦", "Mexico": "🇲🇽", "USA": "🇺🇸", "Switzerland": "🇨🇭",
    "Australia": "🇦🇺", "Turkey": "🇹🇷", "Ecuador": "🇪🇨", "South Korea": "🇰🇷",
    "Norway": "🇳🇴", "Japan": "🇯🇵", "Austria": "🇦🇹", "Sweden": "🇸🇪",
    "Senegal": "🇸🇳", "Ivory Coast": "🇨🇮", "Paraguay": "🇵🇾", "Czechia": "🇨🇿",
    "Scotland": "🏴󠁧󠁢󠁳󠁣󠁴󠁿", "Ghana": "🇬🇭", "Qatar": "🇶🇦", "Iran": "🇮🇷",
    "South Africa": "🇿🇦", "Bosnia and Herzegovina": "🇧🇦", "Egypt": "🇪🇬",
    "Algeria": "🇩🇿", "Tunisia": "🇹🇳", "New Zealand": "🇳🇿", "Haiti": "🇭🇹",
    "Curacao": "🇨🇼", "Cape Verde": "🇨🇻", "Jordan": "🇯🇴", "Uzbekistan": "🇺🇿",
    "DR Congo": "🇨🇩", "Iraq": "🇮🇶", "Saudi Arabia": "🇸🇦", "Canada": "🇨🇦",
    "Panama": "🇵🇦",
}

# ─────────────────────────────────────────────────────────────────────────────
# DATOS Y MODELO
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    hist = pd.read_csv("data_historic.txt.txt")
    test = pd.read_csv("test.txt.txt")
    # Convert ranking to score (higher = better)
    hist["FIFA_Score"] = 211 - hist["FIFA_Ranking"]
    test["FIFA_Score"] = 211 - test["FIFA_Ranking"]
    return hist, test


@st.cache_resource
def build_model():
    hist, _ = load_data()
    X = hist[MODEL_FEATURES].values
    y = hist["Champion"].values
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)
    model = LogisticRegression(random_state=42, max_iter=2000, C=0.3)
    model.fit(X_sc, y)
    return model, scaler, X_sc, y


def predict(df: pd.DataFrame) -> np.ndarray:
    X = df[MODEL_FEATURES].values
    X_sc = scaler.transform(X)
    return model.predict_proba(X_sc)[:, 1]


def ranked_results(df: pd.DataFrame) -> pd.DataFrame:
    probs = predict(df)
    out = df[["Team"] + MODEL_FEATURES].copy()
    out["Probabilidad (%)"] = (probs * 100).round(3)
    out["Flag"] = out["Team"].map(lambda t: TEAM_FLAGS.get(t, "🏳"))
    out["Equipo"] = out["Flag"] + " " + out["Team"]
    return out.sort_values("Probabilidad (%)", ascending=False).reset_index(drop=True)


hist_df, test_df_original = load_data()
model, scaler, X_train_sc, y_train = build_model()
coefs = model.coef_[0]

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        "<h2 style='text-align:center'>⚽ Mundial 2026</h2>",
        unsafe_allow_html=True,
    )
    st.caption("Predictor con Regresión Logística")
    st.divider()
    section = st.radio(
        "Navegar a:",
        [
            "🏠 Introducción",
            "📊 Datos Históricos",
            "🧠 El Modelo Paso a Paso",
            "⚖️ Pesos y Variables",
            "🔮 Predicciones 2026",
            "🎛️ Simulador Interactivo",
        ],
    )
    st.divider()
    acc = accuracy_score(y_train, model.predict(X_train_sc))
    st.metric("Precisión en entrenamiento", f"{acc*100:.0f}%")
    st.caption(
        f"Entrenado con **{len(hist_df)}** selecciones  \n"
        f"de **7** mundiales (1998 – 2022)"
    )
    st.divider()
    st.caption(
        "📌 **Nota:** el modelo aprende de tan solo 7 campeones históricos. "
        "Las probabilidades reflejan patrones estadísticos, no certezas."
    )


# ─────────────────────────────────────────────────────────────────────────────
# SECCIÓN 1 — INTRODUCCIÓN
# ─────────────────────────────────────────────────────────────────────────────
if section == "🏠 Introducción":
    st.title("🏆 Predictor del Campeón del Mundial 2026")

    st.markdown(
        """
        <div style='background:linear-gradient(135deg,#1a472a,#2d6a4f);
                    padding:1.8rem 2rem;border-radius:14px;color:white;margin-bottom:1.5rem'>
            <h3 style='color:#ffd700;margin-top:0'>¿Qué hace esta app?</h3>
            <p style='font-size:1.05rem;margin:0'>
            Tomamos los datos estadísticos de los mundiales <b>1998–2022</b> (35 selecciones,
            7 campeones) y entrenamos un modelo de <b>Regresión Logística</b> que aprende qué
            combinación de variables caracteriza a un campeón mundial. Luego aplicamos ese
            modelo a las <b>48 selecciones del Mundial 2026</b> para estimar la probabilidad
            de que cada una se consagre campeona.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🗓 Mundiales", "7", "1998 → 2022")
    c2.metric("🧩 Variables", "7", "por selección")
    c3.metric("🏆 Campeones usados", "7", "para entrenar")
    c4.metric("🌍 Equipos 2026", str(len(test_df_original)), "a predecir")

    st.divider()

    st.subheader("¿Por qué Regresión Logística y no Lineal clásica?")
    col_a, col_b = st.columns(2)
    with col_a:
        st.warning(
            "**Regresión Lineal** produce un número sin límites. "
            "Para predecir si un equipo *gana* (1) o *no gana* (0), "
            "podría dar valores absurdos como **1.7** o **−0.4**."
        )
    with col_b:
        st.success(
            "**Regresión Logística** es idéntica en su núcleo lineal, "
            "pero aplica la función **sigmoide** al final para aplastar "
            "el resultado entre **0 % y 100 %**. ¡Es la extensión natural!"
        )

    st.markdown("### La fórmula en tres líneas")
    st.markdown("**1. Combinación lineal** (igual que regresión lineal):")
    st.latex(
        r"z = \beta_0 + \beta_1\,x_1 + \beta_2\,x_2 + \cdots + \beta_7\,x_7"
    )
    st.markdown("**2. Función sigmoide** (convierte *z* en probabilidad):")
    st.latex(r"P = \sigma(z) = \frac{1}{1+e^{-z}}")
    st.markdown("**3. Decisión**:")
    st.latex(r"\text{Si } P > 0.5 \Rightarrow \text{Predicción: campeón}")

    st.divider()
    st.subheader("🗺 Flujo del proceso")
    st.markdown(
        """
        ```
        📁 Datos históricos 1998-2022            🧠 Entrenamiento            🔮 Predicción 2026

        • 35 selecciones × 7 variables    ──►   Regresión Logística   ──►  Prob. para 48 equipos
        • Etiqueta: campeón = 1 / 0             aprende los β óptimos       ranking de favoritos
        ```
        """
    )

    st.divider()
    st.subheader("🏅 Campeones históricos en el dataset")
    champs = hist_df[hist_df["Champion"] == 1][
        ["WorldCup", "Team", "FIFA_Ranking", "Elo_Rating", "Win_Rate_Last4Years", "Squad_Value_MEUR"]
    ].copy()
    champs["Campeón"] = champs["Team"].map(lambda t: TEAM_FLAGS.get(t, "") + " " + t)
    champs = champs.rename(columns={
        "WorldCup": "Año", "FIFA_Ranking": "Ranking FIFA",
        "Elo_Rating": "Elo", "Win_Rate_Last4Years": "Tasa Victorias",
        "Squad_Value_MEUR": "Valor (M€)"
    })
    st.dataframe(
        champs[["Año", "Campeón", "Ranking FIFA", "Elo", "Tasa Victorias", "Valor (M€)"]],
        use_container_width=True, hide_index=True,
    )
    st.caption(
        "Observa que los campeones **no siempre son el #1 del ranking FIFA**: "
        "Francia ganó en 1998 (rankeada 18°) e Italia en 2006 (rankeada 13°). "
        "El modelo captura estas sutilezas."
    )


# ─────────────────────────────────────────────────────────────────────────────
# SECCIÓN 2 — DATOS HISTÓRICOS
# ─────────────────────────────────────────────────────────────────────────────
elif section == "📊 Datos Históricos":
    st.title("📊 Datos Históricos (1998–2022)")

    display = hist_df.copy()
    display["Champion"] = display["Champion"].map({1: "🏆 Sí", 0: "No"})
    display["Team"] = display["Team"].map(lambda t: TEAM_FLAGS.get(t, "") + " " + t)
    display = display.drop(columns=["FIFA_Score"])
    st.dataframe(display, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Dispersión: ¿cómo se diferencian los campeones?")

    c1, c2 = st.columns(2)
    with c1:
        xf = st.selectbox("Eje X", MODEL_FEATURES, index=4, key="hx",
                          format_func=lambda f: FEATURE_ES[f])
    with c2:
        yf = st.selectbox("Eje Y", MODEL_FEATURES, index=1, key="hy",
                          format_func=lambda f: FEATURE_ES[f])

    plot_df = hist_df.copy()
    plot_df["Resultado"] = plot_df["Champion"].map({1: "🏆 Campeón", 0: "Eliminado"})
    plot_df["Equipo"] = plot_df["Team"].map(lambda t: TEAM_FLAGS.get(t, "") + " " + t)

    fig = px.scatter(
        plot_df, x=xf, y=yf, color="Resultado",
        hover_data=["Equipo", "WorldCup"],
        color_discrete_map={"🏆 Campeón": "#ffd700", "Eliminado": "#4a90d9"},
        labels={xf: FEATURE_ES[xf], yf: FEATURE_ES[yf]},
        template="plotly_dark",
        title=f"{FEATURE_ES[xf]}  vs  {FEATURE_ES[yf]}",
    )
    fig.update_traces(marker=dict(size=12, opacity=0.85))
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Comparar distribución por variable")
    bv = st.selectbox("Variable", MODEL_FEATURES, format_func=lambda f: FEATURE_ES[f], key="bv")
    fig2 = px.box(
        plot_df, x="Resultado", y=bv, color="Resultado", points="all",
        color_discrete_map={"🏆 Campeón": "#ffd700", "Eliminado": "#4a90d9"},
        labels={bv: FEATURE_ES[bv]},
        template="plotly_dark",
        title=f"Distribución de '{FEATURE_ES[bv]}'",
    )
    st.plotly_chart(fig2, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# SECCIÓN 3 — EL MODELO PASO A PASO
# ─────────────────────────────────────────────────────────────────────────────
elif section == "🧠 El Modelo Paso a Paso":
    st.title("🧠 El Modelo Explicado Paso a Paso")

    # ── Paso 1 ──────────────────────────────────────────────────────────────
    with st.expander("**Paso 1 — Preparación: invertir el Ranking FIFA**", expanded=True):
        st.markdown(
            """
            El Ranking FIFA es un número donde **1 = mejor** y **200 = peor**.
            Eso haría que el coeficiente del modelo tenga un signo contraintuitivo.
            Por eso creamos la variable:
            """
        )
        st.latex(r"\text{FIFA\_Score} = 211 - \text{FIFA\_Ranking}")
        st.markdown(
            "Ahora **mayor FIFA_Score = mejor posición**. "
            "El equipo #1 del mundo pasa de valor 1 → 210; el #50 pasa de 50 → 161."
        )
        ex_df = pd.DataFrame({
            "Ranking FIFA": [1, 3, 7, 14, 30],
            "FIFA_Score (nuevo)": [210, 208, 204, 197, 181],
            "Ejemplo": ["Brasil", "Francia", "Portugal", "Marruecos", "Uruguay"],
        })
        st.dataframe(ex_df, hide_index=True, use_container_width=True)

    # ── Paso 2 ──────────────────────────────────────────────────────────────
    with st.expander("**Paso 2 — Estandarización de variables**", expanded=True):
        st.markdown(
            """
            Las 7 variables tienen escalas muy distintas (ej. Elo ≈ 2000 vs Win_Rate ≈ 0.80).
            Si no las estandarizamos, el modelo daría más peso a las variables con números
            grandes simplemente por su magnitud. La solución:
            """
        )
        st.latex(r"x'_i = \frac{x_i - \mu_i}{\sigma_i}")
        st.markdown(
            "Tras estandarizar, cada variable tiene **media = 0** y **desv. std = 1**, "
            "lo que hace los coeficientes β directamente comparables entre sí."
        )
        sc_df = pd.DataFrame({
            "Variable": [FEATURE_ES[f] for f in MODEL_FEATURES],
            "Media (μ)": scaler.mean_.round(3),
            "Desv. Std (σ)": scaler.scale_.round(3),
        })
        st.dataframe(sc_df, hide_index=True, use_container_width=True)

    # ── Paso 3 ──────────────────────────────────────────────────────────────
    with st.expander("**Paso 3 — Combinación lineal (z)**", expanded=True):
        st.markdown(
            "El modelo combina linealmente las variables estandarizadas con sus pesos β:"
        )
        β0 = model.intercept_[0]
        terms = "  +  ".join(
            [f"({coefs[i]:.3f})\\cdot x'_{{{i+1}}}" for i in range(len(MODEL_FEATURES))]
        )
        st.latex(rf"z = {β0:.3f}  +  {terms}")
        st.markdown("Cuanto mayor sea *z*, más probable es que el equipo sea campeón.")

    # ── Paso 4 ──────────────────────────────────────────────────────────────
    with st.expander("**Paso 4 — Función Sigmoide → Probabilidad**", expanded=True):
        z_range = np.linspace(-8, 8, 300)
        sigma = 1 / (1 + np.exp(-z_range))
        fig_s = go.Figure()
        fig_s.add_trace(go.Scatter(
            x=z_range, y=sigma, mode="lines",
            line=dict(color="#ffd700", width=3), name="σ(z)"
        ))
        fig_s.add_hline(y=0.5, line_dash="dash", line_color="white", opacity=0.4,
                        annotation_text="50 %")
        fig_s.update_layout(
            template="plotly_dark",
            title="Función Sigmoide — convierte cualquier número en probabilidad (0–1)",
            xaxis_title="z (combinación lineal)", yaxis_title="P (probabilidad)",
            yaxis_tickformat=".0%", height=380,
        )
        st.plotly_chart(fig_s, use_container_width=True)
        st.markdown(
            "- Cuando *z* es muy grande → P → 100 %  \n"
            "- Cuando *z* = 0 → P = 50 %  \n"
            "- Cuando *z* es muy negativo → P → 0 %"
        )

    # ── Paso 5 ──────────────────────────────────────────────────────────────
    with st.expander("**Paso 5 — Entrenamiento: ¿cómo aprende el modelo?**", expanded=True):
        st.markdown(
            """
            El algoritmo ajusta los coeficientes β para **maximizar la log-verosimilitud**
            (log-likelihood): busca los valores que asignen la mayor probabilidad posible
            a los campeones reales y la menor posible al resto.
            """
        )
        st.latex(
            r"\mathcal{L}(\beta) = \sum_i \Bigl[ y_i \log P_i + (1-y_i)\log(1-P_i) \Bigr]"
        )
        acc = accuracy_score(y_train, model.predict(X_train_sc))
        st.success(
            f"Precisión del modelo sobre los datos de entrenamiento: **{acc*100:.0f}%** "
            f"({int(acc*len(y_train))}/{len(y_train)} predicciones correctas)"
        )
        st.info(
            "**Regularización:** usamos `C = 0.3` (penalización L2) para evitar que el modelo "
            "memorice los 35 ejemplos históricos y generalice mejor a los datos de 2026."
        )


# ─────────────────────────────────────────────────────────────────────────────
# SECCIÓN 4 — PESOS Y VARIABLES
# ─────────────────────────────────────────────────────────────────────────────
elif section == "⚖️ Pesos y Variables":
    st.title("⚖️ Pesos del Modelo y Relevancia de Variables")

    st.markdown(
        """
        Los **coeficientes β** son los pesos aprendidos. Como las variables están estandarizadas,
        son directamente comparables: un β mayor (en valor absoluto) = mayor influencia
        sobre la probabilidad de ser campeón.
        """
    )

    coef_df = pd.DataFrame({
        "Feature": MODEL_FEATURES,
        "Variable": [FEATURE_ES[f] for f in MODEL_FEATURES],
        "β": coefs,
        "|β|": np.abs(coefs),
        "Importancia (%)": (np.abs(coefs) / np.abs(coefs).sum() * 100).round(1),
    }).sort_values("|β|", ascending=True)

    # ── Gráfico de barras de coeficientes ──────────────────────────────────
    fig_c = go.Figure(go.Bar(
        x=coef_df["β"],
        y=coef_df["Variable"],
        orientation="h",
        marker_color=["#2ecc71" if v >= 0 else "#e74c3c" for v in coef_df["β"]],
        text=[f"{v:+.4f}" for v in coef_df["β"]],
        textposition="outside",
    ))
    fig_c.update_layout(
        template="plotly_dark",
        title="Coeficientes β del modelo (datos estandarizados)",
        xaxis_title="Coeficiente β",
        height=480,
        xaxis=dict(zeroline=True, zerolinecolor="white", zerolinewidth=1.5),
    )
    st.plotly_chart(fig_c, use_container_width=True)

    col1, col2 = st.columns(2)
    col1.success("**Verde (β > 0):** A mayor valor de esa variable, **más** probable ser campeón.")
    col2.error("**Rojo (β < 0):** A mayor valor, **menos** probable. Hay una relación inversa.")

    st.divider()

    # ── Análisis detallado ──────────────────────────────────────────────────
    st.subheader("🔍 Análisis detallado variable por variable")

    sorted_feats = coef_df.sort_values("|β|", ascending=False)["Feature"].tolist()

    SPECIAL_NOTES = {
        "FIFA_Score": (
            "⚠️ **Coeficiente negativo** (-0.209). ¿Sorprendente? El modelo aprendió "
            "que ser el equipo mejor ranqueado por la FIFA **no garantiza** ganar el Mundial. "
            "En los datos históricos, Francia ganó en 1998 (rankeada **18°**) e Italia en 2006 "
            "(rankeada **13°**). El #1 de la FIFA (Brasil) apareció en múltiples mundiales sin "
            "ganar todos. Por eso una **Puntuación FIFA más baja** (peor ranking) no penaliza tanto."
        ),
        "Squad_Value_MEUR": (
            "⚠️ **Coeficiente negativo** (-0.073). Los equipos más caros (como Francia 2022, "
            "valorada en 1050 M€) no siempre ganan el torneo. El valor del mercado no captura "
            "factores como cohesión táctica o experiencia en torneos bajo presión."
        ),
        "Previous_WorldCup_Appearances": (
            "⚠️ **Coeficiente negativo** (-0.114). Más mundiales en el historial no se traduce "
            "directamente en campeonatos. Brasil (22+ mundiales) tiene un solo título en el dataset. "
            "La experiencia ayuda, pero no determina el resultado por sí sola."
        ),
        "Average_Age": (
            "⚠️ **Coeficiente negativo** (-0.084). Los campeones tienden a ser ligeramente más "
            "jóvenes que el promedio del dataset. Plantillas muy veteranas pueden tener menos "
            "continuidad física a lo largo del torneo."
        ),
    }

    for feat in sorted_feats:
        idx = MODEL_FEATURES.index(feat)
        beta = coefs[idx]
        imp = np.abs(beta) / np.abs(coefs).sum() * 100
        icon = "🟢" if beta >= 0 else "🔴"

        with st.expander(f"{icon} {FEATURE_ES[feat]}  |  β = {beta:+.4f}  |  Importancia: {imp:.1f}%"):
            ca, cb = st.columns([3, 1])
            with ca:
                st.markdown(f"**¿Qué mide?** {FEATURE_DESC[feat]}")
                if feat in SPECIAL_NOTES:
                    st.info(SPECIAL_NOTES[feat])
            with cb:
                st.metric("Coeficiente β", f"{beta:+.4f}")
                st.metric("Importancia relativa", f"{imp:.1f}%")
                direction = "↑ = más prob." if beta >= 0 else "↑ = menos prob."
                st.caption(f"Dirección: {direction}")

    st.divider()
    st.subheader("📊 Importancia relativa de las variables")
    imp_df = coef_df[["Variable", "Importancia (%)"]].sort_values("Importancia (%)", ascending=False)
    fig_i = px.pie(
        imp_df, names="Variable", values="Importancia (%)",
        title="¿Cuánto contribuye cada variable al modelo?",
        template="plotly_dark",
        color_discrete_sequence=px.colors.sequential.Viridis,
    )
    fig_i.update_traces(textposition="inside", textinfo="percent+label")
    st.plotly_chart(fig_i, use_container_width=True)

    st.subheader("📋 Tabla resumen de coeficientes")
    table = coef_df[["Variable", "β", "Importancia (%)"]].sort_values("Importancia (%)", ascending=False)
    table.index = range(1, len(table) + 1)
    st.dataframe(table, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# SECCIÓN 5 — PREDICCIONES 2026
# ─────────────────────────────────────────────────────────────────────────────
elif section == "🔮 Predicciones 2026":
    st.title("🔮 Predicciones para el Mundial 2026")

    results = ranked_results(test_df_original)
    top10 = results.head(10)

    # ── Podio ───────────────────────────────────────────────────────────────
    st.subheader("🥇 Grandes favoritos")
    c1, c2, c3 = st.columns(3)
    c1.metric("🥇 Favorito #1", top10.iloc[0]["Equipo"], f"{top10.iloc[0]['Probabilidad (%)']:.2f}%")
    c2.metric("🥈 Favorito #2", top10.iloc[1]["Equipo"], f"{top10.iloc[1]['Probabilidad (%)']:.2f}%")
    c3.metric("🥉 Favorito #3", top10.iloc[2]["Equipo"], f"{top10.iloc[2]['Probabilidad (%)']:.2f}%")

    # ── Bar chart top 10 ────────────────────────────────────────────────────
    fig_bar = px.bar(
        top10, x="Probabilidad (%)", y="Equipo", orientation="h",
        color="Probabilidad (%)",
        color_continuous_scale=["#2c7bb6", "#ffd700"],
        text=top10["Probabilidad (%)"].map(lambda v: f"{v:.2f}%"),
        title="Top 10 favoritos según el modelo",
        template="plotly_dark", labels={"Equipo": ""},
    )
    fig_bar.update_layout(yaxis=dict(autorange="reversed"), coloraxis_showscale=False, height=520)
    fig_bar.update_traces(textposition="outside")
    st.plotly_chart(fig_bar, use_container_width=True)

    # ── Todos los equipos ───────────────────────────────────────────────────
    st.divider()
    st.subheader("🌍 Todos los equipos clasificados")
    fig_all = px.bar(
        results, x="Team", y="Probabilidad (%)",
        color="Probabilidad (%)",
        color_continuous_scale=["#1a1a2e", "#ffd700"],
        hover_data=["Equipo"],
        title="Probabilidad de campeonato — todos los clasificados para 2026",
        template="plotly_dark", labels={"Team": ""},
    )
    fig_all.update_layout(coloraxis_showscale=False, height=420, xaxis_tickangle=-45)
    st.plotly_chart(fig_all, use_container_width=True)

    # ── Tabla ───────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("📋 Tabla completa")
    table = results.rename(columns={f: FEATURE_ES[f] for f in MODEL_FEATURES})
    table.index = range(1, len(table) + 1)
    st.dataframe(
        table[["Equipo", "Probabilidad (%)"] + [FEATURE_ES[f] for f in MODEL_FEATURES]],
        use_container_width=True,
    )

    # ── Nota interpretativa ─────────────────────────────────────────────────
    st.divider()
    st.info(
        "**¿Por qué las probabilidades son cercanas entre sí?**  \n"
        "El dataset de entrenamiento tiene 7 campeones entre 35 equipos (~20%). "
        "El modelo es apropiadamente incierto: en el fútbol, el campeón no es siempre el favorito. "
        "Las diferencias entre el 1° y el 5° reflejan una ventaja real pero no abrumadora."
    )


# ─────────────────────────────────────────────────────────────────────────────
# SECCIÓN 6 — SIMULADOR INTERACTIVO
# ─────────────────────────────────────────────────────────────────────────────
elif section == "🎛️ Simulador Interactivo":
    st.title("🎛️ Simulador Interactivo")
    st.markdown(
        """
        Ajusta los valores estadísticos de cualquier selección con los controles de abajo
        y observa cómo cambia su probabilidad de ser campeón en **tiempo real**.
        El resto de los equipos mantiene sus valores originales.
        """
    )

    if "sim_df" not in st.session_state:
        st.session_state.sim_df = test_df_original.copy()

    # ── Selección de equipo ─────────────────────────────────────────────────
    teams = sorted(st.session_state.sim_df["Team"].tolist())
    team_labels = [TEAM_FLAGS.get(t, "") + " " + t for t in teams]
    label_to_team = dict(zip(team_labels, teams))

    col_sel, col_reset = st.columns([3, 1])
    selected_label = col_sel.selectbox("Selecciona el equipo a editar:", team_labels)
    selected_team = label_to_team[selected_label]

    row_idx = st.session_state.sim_df[
        st.session_state.sim_df["Team"] == selected_team
    ].index[0]
    cur_row = st.session_state.sim_df.loc[row_idx]
    orig_row = test_df_original[test_df_original["Team"] == selected_team].iloc[0]

    if col_reset.button("↩️ Resetear este equipo", use_container_width=True):
        for f in MODEL_FEATURES:
            st.session_state.sim_df.loc[row_idx, f] = orig_row[f]
        st.rerun()

    st.markdown(f"### {TEAM_FLAGS.get(selected_team, '')} {selected_team}")

    # ── Sliders ─────────────────────────────────────────────────────────────
    new_vals = {}
    col_l, col_r = st.columns(2)
    for i, feat in enumerate(MODEL_FEATURES):
        mn, mx, step = FEATURE_UNIT[feat]
        orig_v = float(orig_row[feat])
        cur_v = float(cur_row[feat])
        target_col = col_l if i % 2 == 0 else col_r
        with target_col:
            new_v = st.slider(
                FEATURE_ES[feat],
                min_value=float(mn), max_value=float(mx),
                value=cur_v, step=float(step),
                help=FEATURE_DESC[feat],
                key=f"sl_{feat}_{selected_team}",
            )
            new_vals[feat] = new_v
            delta = new_v - orig_v
            if abs(delta) > 1e-9:
                st.caption(f"Original: **{orig_v:.3g}** → Δ {delta:+.3g}")

    # ── Live preview ────────────────────────────────────────────────────────
    preview = st.session_state.sim_df.loc[[row_idx]].copy()
    for f, v in new_vals.items():
        preview.loc[row_idx, f] = v

    prob_new = predict(preview)[0] * 100
    prob_orig = predict(
        test_df_original[test_df_original["Team"] == selected_team]
    )[0] * 100
    delta_p = prob_new - prob_orig

    st.divider()
    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("Prob. actual (editada)", f"{prob_new:.3f}%",
                  delta=f"{delta_p:+.3f}% vs original")
    col_m2.metric("Prob. original", f"{prob_orig:.3f}%")

    # Rank in simulation
    sim_temp = st.session_state.sim_df.copy()
    for f, v in new_vals.items():
        sim_temp.loc[row_idx, f] = v
    sim_res = ranked_results(sim_temp)
    rank_new = sim_res[sim_res["Team"] == selected_team].index[0] + 1
    col_m3.metric("Posición en el ranking simulado", f"#{rank_new}", f"de {len(sim_res)}")

    # ── Botones guardar / resetear todo ────────────────────────────────────
    col_a, col_b = st.columns(2)
    if col_a.button("✅ Guardar cambios en simulación", type="primary", use_container_width=True):
        for f, v in new_vals.items():
            st.session_state.sim_df.loc[row_idx, f] = v
        st.success(f"Cambios guardados para {selected_team}.")
        st.rerun()

    if col_b.button("🔄 Resetear todos los equipos", use_container_width=True):
        st.session_state.sim_df = test_df_original.copy()
        st.info("Todos los valores restaurados.")
        st.rerun()

    # ── Ranking simulado ────────────────────────────────────────────────────
    st.divider()
    st.subheader("📊 Ranking completo con los valores actuales del simulador")

    changed = [
        t for t in test_df_original["Team"]
        if any(
            abs(test_df_original[test_df_original["Team"] == t].iloc[0][f]
                - st.session_state.sim_df[st.session_state.sim_df["Team"] == t].iloc[0][f]) > 1e-9
            for f in MODEL_FEATURES
        )
    ]
    if changed:
        st.info(f"⚡ Equipos con valores modificados: **{', '.join(changed)}**")

    sim_results = ranked_results(st.session_state.sim_df)
    top_sim = sim_results.head(10)
    fig_sim = px.bar(
        top_sim, x="Probabilidad (%)", y="Equipo", orientation="h",
        color="Probabilidad (%)",
        color_continuous_scale=["#2c7bb6", "#ffd700"],
        text=top_sim["Probabilidad (%)"].map(lambda v: f"{v:.2f}%"),
        title="Top 10 — simulación actual",
        template="plotly_dark",
    )
    fig_sim.update_layout(yaxis=dict(autorange="reversed"), coloraxis_showscale=False, height=500)
    fig_sim.update_traces(textposition="outside")
    st.plotly_chart(fig_sim, use_container_width=True)

    table_s = sim_results.rename(columns={f: FEATURE_ES[f] for f in MODEL_FEATURES})
    table_s.index = range(1, len(table_s) + 1)
    st.dataframe(
        table_s[["Equipo", "Probabilidad (%)"] + [FEATURE_ES[f] for f in MODEL_FEATURES]],
        use_container_width=True,
    )
