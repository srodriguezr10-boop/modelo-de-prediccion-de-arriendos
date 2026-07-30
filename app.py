from __future__ import annotations

import io
import time
from dataclasses import dataclass
from typing import Any

import pandas as pd
import requests
import streamlit as st


# ============================================================
# CONFIGURACIÓN DE LA APLICACIÓN
# ============================================================
st.set_page_config(
    page_title="Estimador de arriendo",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

POLL_INTERVAL_SECONDS = 2
DEFAULT_TIMEOUT_SECONDS = 600

FEATURE_COLUMNS = [
    "metros_cuadrados",
    "habitaciones",
    "banos",
    "estrato",
]


# ============================================================
# ESTILOS VISUALES
# ============================================================
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap');

        html, body, [class*="css"] {
            font-family: "Manrope", sans-serif;
        }

        .stApp {
            background:
                radial-gradient(circle at 10% 0%, rgba(37, 99, 235, 0.12), transparent 32%),
                radial-gradient(circle at 95% 5%, rgba(124, 58, 237, 0.12), transparent 30%),
                #f7f9fc;
        }

        .main .block-container {
            max-width: 1180px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        .hero {
            padding: 2.2rem;
            border-radius: 26px;
            background: linear-gradient(135deg, #0f172a 0%, #1d4ed8 55%, #6d28d9 100%);
            color: white;
            box-shadow: 0 22px 55px rgba(15, 23, 42, 0.18);
            margin-bottom: 1.5rem;
        }

        .hero-badge {
            display: inline-block;
            background: rgba(255,255,255,0.14);
            border: 1px solid rgba(255,255,255,0.22);
            border-radius: 999px;
            padding: 0.38rem 0.8rem;
            font-size: 0.78rem;
            font-weight: 700;
            margin-bottom: 0.9rem;
        }

        .hero h1 {
            font-size: clamp(2rem, 5vw, 3rem);
            line-height: 1.05;
            margin: 0 0 0.8rem 0;
            font-weight: 800;
        }

        .hero p {
            max-width: 760px;
            margin: 0;
            color: rgba(255,255,255,0.84);
            font-size: 1rem;
        }

        .panel {
            background: rgba(255,255,255,0.94);
            border: 1px solid rgba(148,163,184,0.24);
            border-radius: 22px;
            padding: 1.35rem;
            box-shadow: 0 12px 36px rgba(15,23,42,0.07);
        }

        .section-title {
            font-size: 1.18rem;
            font-weight: 800;
            color: #0f172a;
            margin-bottom: 0.25rem;
        }

        .section-subtitle {
            color: #64748b;
            font-size: 0.9rem;
            margin-bottom: 1rem;
        }

        .result-card {
            padding: 1.8rem;
            border-radius: 22px;
            color: white;
            background: linear-gradient(135deg, #0f172a 0%, #1e40af 58%, #7c3aed 100%);
            box-shadow: 0 18px 45px rgba(30,64,175,0.22);
            text-align: center;
        }

        .result-label {
            color: rgba(255,255,255,0.74);
            font-size: 0.82rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }

        .result-value {
            font-size: clamp(2rem, 5vw, 3.4rem);
            line-height: 1;
            font-weight: 800;
            margin: 0.7rem 0;
        }

        .result-note {
            color: rgba(255,255,255,0.76);
            font-size: 0.88rem;
        }

        .feature-card {
            background: white;
            border: 1px solid #e2e8f0;
            border-radius: 16px;
            padding: 1rem;
            text-align: center;
        }

        .feature-label {
            color: #64748b;
            font-size: 0.78rem;
            font-weight: 700;
        }

        .feature-value {
            color: #0f172a;
            font-size: 1.35rem;
            font-weight: 800;
            margin-top: 0.2rem;
        }

        .stButton > button {
            width: 100%;
            min-height: 52px;
            border: none;
            border-radius: 14px;
            color: white;
            font-size: 1rem;
            font-weight: 800;
            background: linear-gradient(135deg, #2563eb, #7c3aed);
            box-shadow: 0 10px 24px rgba(37,99,235,0.24);
            transition: transform 0.15s ease, box-shadow 0.15s ease;
        }

        .stButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 13px 28px rgba(37,99,235,0.3);
        }

        div[data-testid="stSlider"] {
            padding-bottom: 0.55rem;
        }

        div[data-testid="stMetric"] {
            background: white;
            border: 1px solid #e2e8f0;
            padding: 0.9rem;
            border-radius: 16px;
        }

        div[data-testid="stDataFrame"] {
            border-radius: 16px;
            overflow: hidden;
            border: 1px solid #e2e8f0;
        }

        .footer-note {
            color: #64748b;
            font-size: 0.82rem;
            text-align: center;
            margin-top: 2rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# CLIENTE DE DATAROBOT
# ============================================================
class DataRobotPredictionError(RuntimeError):
    """Error controlado durante el consumo de la API de DataRobot."""


@dataclass(frozen=True)
class DataRobotConfig:
    api_key: str
    deployment_id: str
    host: str
    timeout: int = DEFAULT_TIMEOUT_SECONDS

    @property
    def batch_predictions_url(self) -> str:
        return f"{self.host.rstrip('/')}/api/v2/batchPredictions/"


class DataRobotBatchClient:
    def __init__(self, config: DataRobotConfig) -> None:
        self.config = config
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Token {config.api_key}",
                "User-Agent": "Streamlit-Rental-Prediction-App",
            }
        )

    def _request(self, method: str, url: str, **kwargs: Any) -> requests.Response:
        try:
            response = self.session.request(
                method,
                url,
                timeout=self.config.timeout,
                **kwargs,
            )
        except requests.RequestException as exc:
            raise DataRobotPredictionError(
                f"No fue posible conectarse con DataRobot: {exc}"
            ) from exc

        if not response.ok:
            detail = response.text[:1800]
            raise DataRobotPredictionError(
                f"DataRobot respondió con HTTP {response.status_code}: {detail}"
            )

        return response

    def create_job(self) -> dict[str, Any]:
        payload = {
            "deploymentId": self.config.deployment_id,
            "passthroughColumnsSet": "all",
            "includePredictionStatus": True,
        }
        response = self._request(
            "POST",
            self.config.batch_predictions_url,
            json=payload,
        )
        return response.json()

    def upload_csv(self, upload_url: str, csv_bytes: bytes) -> None:
        self._request(
            "PUT",
            upload_url,
            data=csv_bytes,
            headers={
                "Content-Type": "text/csv; encoding=utf-8",
                "Content-Length": str(len(csv_bytes)),
            },
        )

    def get_job(self, job_url: str) -> dict[str, Any]:
        return self._request("GET", job_url).json()

    def download_results(self, download_url: str) -> bytes:
        return self._request("GET", download_url).content

    def predict(self, input_df: pd.DataFrame) -> pd.DataFrame:
        csv_bytes = input_df.to_csv(index=False).encode("utf-8")
        job = self.create_job()

        job_id = job.get("id", "sin-id")
        links = job.get("links", {})
        upload_url = links.get("csvUpload")
        job_url = links.get("self")

        if not upload_url or not job_url:
            raise DataRobotPredictionError(
                "La respuesta de DataRobot no contiene los enlaces requeridos."
            )

        self.upload_csv(upload_url, csv_bytes)

        status_placeholder = st.empty()
        progress_bar = st.progress(0)

        started_at = time.time()

        while True:
            if time.time() - started_at > self.config.timeout:
                raise DataRobotPredictionError(
                    "La predicción excedió el tiempo máximo de espera."
                )

            current_job = self.get_job(job_url)
            status = current_job.get("status", "UNKNOWN")
            percentage = float(current_job.get("percentageCompleted", 0) or 0)

            progress_bar.progress(min(max(int(percentage), 0), 100))
            status_placeholder.info(
                f"Procesando predicción · Estado: {status} · Trabajo: {job_id}"
            )

            if status == "COMPLETED":
                progress_bar.progress(100)
                status_placeholder.success("Predicción completada correctamente.")
                download_url = current_job.get("links", {}).get("download")

                if not download_url:
                    raise DataRobotPredictionError(
                        "El trabajo terminó, pero no se encontró el enlace de descarga."
                    )

                result_bytes = self.download_results(download_url)
                return pd.read_csv(io.BytesIO(result_bytes))

            if status in {"FAILED", "ABORTED"}:
                details = current_job.get("statusDetails", "Sin detalles adicionales.")
                raise DataRobotPredictionError(
                    f"El trabajo terminó con estado {status}: {details}"
                )

            time.sleep(POLL_INTERVAL_SECONDS)


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================
def read_secret(name: str, default: str | None = None) -> str:
    try:
        value = st.secrets[name]
    except (KeyError, FileNotFoundError):
        if default is not None:
            return default
        raise DataRobotPredictionError(
            f"Falta configurar el secreto '{name}' en Streamlit."
        )

    value = str(value).strip()
    if not value:
        raise DataRobotPredictionError(
            f"El secreto '{name}' está vacío."
        )
    return value


def get_client() -> DataRobotBatchClient:
    config = DataRobotConfig(
        api_key=read_secret("DATAROBOT_API_KEY"),
        deployment_id=read_secret("DATAROBOT_DEPLOYMENT_ID"),
        host=read_secret("DATAROBOT_HOST", "https://app.datarobot.com"),
    )
    return DataRobotBatchClient(config)


def find_prediction_column(result_df: pd.DataFrame) -> str:
    excluded = set(FEATURE_COLUMNS + ["prediction_status"])

    preferred_exact = [
        "precio_arriendo_cop_PREDICTION",
        "precio_arriendo_cop_prediction",
        "prediction",
        "PREDICTION",
    ]

    for column in preferred_exact:
        if column in result_df.columns:
            return column

    prediction_columns = [
        column
        for column in result_df.columns
        if "prediction" in column.lower()
        and "status" not in column.lower()
        and column not in excluded
    ]

    if prediction_columns:
        return prediction_columns[0]

    numeric_candidates = [
        column
        for column in result_df.select_dtypes(include="number").columns
        if column not in excluded
    ]

    if numeric_candidates:
        return numeric_candidates[-1]

    raise DataRobotPredictionError(
        "No fue posible identificar la columna de predicción en la respuesta."
    )


def format_cop(value: float) -> str:
    return f"${value:,.0f}".replace(",", ".")


# ============================================================
# ENCABEZADO
# ============================================================
st.markdown(
    """
    <div class="hero">
        <div class="hero-badge">MODELO DE MACHINE LEARNING</div>
        <h1>Estimador de precio de arriendo</h1>
        <p>
            Ajusta las características del inmueble y consulta el precio mensual
            estimado por el modelo desplegado en DataRobot.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# FORMULARIO DE ENTRADA
# ============================================================
left_col, right_col = st.columns([1.18, 0.82], gap="large")

with left_col:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-title">Características del inmueble</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-subtitle">Mueve los controles para construir el registro que será enviado al modelo.</div>',
        unsafe_allow_html=True,
    )

    with st.form("prediction_form"):
        metros_cuadrados = st.slider(
            "Área del inmueble (m²)",
            min_value=20,
            max_value=150,
            value=75,
            step=1,
            help="Área total aproximada del inmueble.",
        )

        control_col1, control_col2 = st.columns(2)

        with control_col1:
            habitaciones = st.slider(
                "Habitaciones",
                min_value=1,
                max_value=6,
                value=3,
                step=1,
            )

            estrato = st.slider(
                "Estrato",
                min_value=1,
                max_value=6,
                value=3,
                step=1,
            )

        with control_col2:
            banos = st.slider(
                "Baños",
                min_value=1,
                max_value=5,
                value=2,
                step=1,
            )

            st.number_input(
                "Valor enviado como área",
                min_value=20,
                max_value=150,
                value=metros_cuadrados,
                disabled=True,
                help="Este campo confirma el valor seleccionado en el control de área.",
            )

        submitted = st.form_submit_button(
            "Calcular precio estimado",
            use_container_width=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)

    input_df = pd.DataFrame(
        [
            {
                "metros_cuadrados": metros_cuadrados,
                "habitaciones": habitaciones,
                "banos": banos,
                "estrato": estrato,
            }
        ]
    )

    st.write("")
    with st.expander("Ver datos enviados al modelo"):
        st.dataframe(input_df, use_container_width=True, hide_index=True)

with right_col:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-title">Resumen del inmueble</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-subtitle">Los valores se actualizan con los controles del formulario.</div>',
        unsafe_allow_html=True,
    )

    row1_col1, row1_col2 = st.columns(2)
    row2_col1, row2_col2 = st.columns(2)

    with row1_col1:
        st.markdown(
            f"""
            <div class="feature-card">
                <div class="feature-label">ÁREA</div>
                <div class="feature-value">{metros_cuadrados} m²</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with row1_col2:
        st.markdown(
            f"""
            <div class="feature-card">
                <div class="feature-label">HABITACIONES</div>
                <div class="feature-value">{habitaciones}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with row2_col1:
        st.markdown(
            f"""
            <div class="feature-card">
                <div class="feature-label">BAÑOS</div>
                <div class="feature-value">{banos}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with row2_col2:
        st.markdown(
            f"""
            <div class="feature-card">
                <div class="feature-label">ESTRATO</div>
                <div class="feature-value">{estrato}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")

    if "last_prediction" not in st.session_state:
        st.info(
            "Configura las características y presiona **Calcular precio estimado**."
        )
    else:
        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-label">Arriendo mensual estimado</div>
                <div class="result-value">{format_cop(st.session_state.last_prediction)}</div>
                <div class="result-note">
                    Valor calculado por el deployment configurado en DataRobot.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# EJECUCIÓN DE LA PREDICCIÓN
# ============================================================
if submitted:
    try:
        client = get_client()

        with st.spinner("Enviando datos al modelo..."):
            result_df = client.predict(input_df)

        prediction_column = find_prediction_column(result_df)
        prediction_value = pd.to_numeric(
            result_df.loc[0, prediction_column],
            errors="coerce",
        )

        if pd.isna(prediction_value):
            raise DataRobotPredictionError(
                f"La columna '{prediction_column}' no contiene un valor numérico válido."
            )

        st.session_state.last_prediction = float(prediction_value)
        st.session_state.last_result_df = result_df
        st.rerun()

    except DataRobotPredictionError as exc:
        st.error(str(exc))
    except Exception as exc:
        st.error(f"Ocurrió un error inesperado: {exc}")


# ============================================================
# RESULTADO TÉCNICO
# ============================================================
if "last_result_df" in st.session_state:
    st.write("")
    with st.expander("Ver respuesta completa de DataRobot"):
        st.dataframe(
            st.session_state.last_result_df,
            use_container_width=True,
            hide_index=True,
        )

        csv_result = st.session_state.last_result_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Descargar resultado CSV",
            data=csv_result,
            file_name="prediccion_arriendo.csv",
            mime="text/csv",
            use_container_width=True,
        )


st.markdown(
    """
    <div class="footer-note">
        La estimación depende de los datos utilizados para entrenar el modelo.
        No representa una valoración inmobiliaria oficial.
    </div>
    """,
    unsafe_allow_html=True,
)
