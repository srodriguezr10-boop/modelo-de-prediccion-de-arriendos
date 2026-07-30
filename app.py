from __future__ import annotations

import io
import time
from dataclasses import dataclass
from typing import Any

import pandas as pd
import requests
import streamlit as st


# =========================================================
# CONFIGURACIÓN GENERAL
# =========================================================
st.set_page_config(
    page_title="Predicción con DataRobot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

POLL_INTERVAL_SECONDS = 3
DEFAULT_TIMEOUT_SECONDS = 600


# =========================================================
# ESTILOS
# =========================================================
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(59, 130, 246, 0.14), transparent 30%),
                radial-gradient(circle at top right, rgba(139, 92, 246, 0.12), transparent 28%),
                #f7f9fc;
        }

        .main .block-container {
            max-width: 1200px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        .hero {
            padding: 2.2rem;
            border-radius: 24px;
            background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 55%, #6d28d9 100%);
            color: white;
            box-shadow: 0 18px 45px rgba(15, 23, 42, 0.18);
            margin-bottom: 1.5rem;
        }

        .hero h1 {
            font-size: 2.25rem;
            line-height: 1.1;
            margin: 0 0 0.7rem 0;
            font-weight: 800;
        }

        .hero p {
            margin: 0;
            color: rgba(255,255,255,0.84);
            font-size: 1.02rem;
            max-width: 760px;
        }

        .info-card {
            background: rgba(255,255,255,0.92);
            border: 1px solid rgba(148,163,184,0.25);
            border-radius: 18px;
            padding: 1.15rem 1.25rem;
            box-shadow: 0 8px 30px rgba(15,23,42,0.06);
            height: 100%;
        }

        .info-label {
            color: #64748b;
            font-size: 0.78rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.06em;
        }

        .info-value {
            color: #0f172a;
            font-size: 1.4rem;
            font-weight: 800;
            margin-top: 0.25rem;
        }

        div[data-testid="stFileUploader"] {
            background: rgba(255,255,255,0.88);
            border: 1px dashed #93c5fd;
            border-radius: 18px;
            padding: 0.5rem;
        }

        div[data-testid="stDataFrame"] {
            border-radius: 16px;
            overflow: hidden;
            border: 1px solid #e2e8f0;
        }

        .stButton > button,
        .stDownloadButton > button {
            border: none;
            border-radius: 12px;
            font-weight: 700;
            min-height: 46px;
            transition: transform 0.15s ease, box-shadow 0.15s ease;
        }

        .stButton > button {
            background: linear-gradient(135deg, #2563eb, #7c3aed);
            color: white;
            box-shadow: 0 8px 20px rgba(37,99,235,0.22);
        }

        .stButton > button:hover,
        .stDownloadButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 10px 24px rgba(37,99,235,0.28);
        }

        section[data-testid="stSidebar"] {
            background: #0f172a;
        }

        section[data-testid="stSidebar"] * {
            color: #e2e8f0;
        }

        section[data-testid="stSidebar"] input {
            color: #0f172a !important;
        }

        .small-note {
            color: #64748b;
            font-size: 0.88rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# CLASES Y FUNCIONES DE DATAROBOT
# =========================================================
class DataRobotPredictionError(RuntimeError):
    """Error controlado durante el proceso de predicción."""


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
                "User-Agent": "Streamlit-DataRobot-Batch-Prediction-App",
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
            detail = response.text[:1500]
            raise DataRobotPredictionError(
                f"DataRobot respondió con HTTP {response.status_code}: {detail}"
            )
        return response

    def create_job(
        self,
        include_all_columns: bool,
        include_prediction_status: bool,
        max_explanations: int | None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "deploymentId": self.config.deployment_id,
        }

        if include_all_columns:
            payload["passthroughColumnsSet"] = "all"
        if include_prediction_status:
            payload["includePredictionStatus"] = True
        if max_explanations and max_explanations > 0:
            payload["maxExplanations"] = max_explanations

        response = self._request("POST", self.config.batch_predictions_url, json=payload)
        return response.json()

    def upload_csv(self, upload_url: str, csv_bytes: bytes) -> None:
        headers = {
            "Content-Type": "text/csv; encoding=utf-8",
            "Content-Length": str(len(csv_bytes)),
        }
        self._request("PUT", upload_url, data=csv_bytes, headers=headers)

    def get_job(self, job_url: str) -> dict[str, Any]:
        return self._request("GET", job_url).json()

    def download_results(self, download_url: str) -> bytes:
        return self._request("GET", download_url).content

    def abort_job(self, job_id: str) -> None:
        abort_url = f"{self.config.batch_predictions_url}{job_id}/"
        try:
            self._request("DELETE", abort_url)
        except DataRobotPredictionError:
            pass

    def predict(
        self,
        csv_bytes: bytes,
        include_all_columns: bool,
        include_prediction_status: bool,
        max_explanations: int | None,
        progress_bar: Any,
        status_box: Any,
    ) -> bytes:
        job = self.create_job(
            include_all_columns=include_all_columns,
            include_prediction_status=include_prediction_status,
            max_explanations=max_explanations,
        )

        job_id = job["id"]
        links = job["links"]
        job_url = links["self"]

        try:
            status_box.info("Trabajo creado. Cargando los datos en DataRobot...")
            progress_bar.progress(5)
            self.upload_csv(links["csvUpload"], csv_bytes)
            progress_bar.progress(12)

            while True:
                job = self.get_job(job_url)
                status = job.get("status", "UNKNOWN")

                if status == "INITIALIZING":
                    queue_position = job.get("queuePosition")
                    message = "DataRobot está preparando el trabajo."
                    if isinstance(queue_position, int) and queue_position > 0:
                        message += f" Posición en cola: {queue_position}."
                    status_box.info(message)
                    progress_bar.progress(15)

                elif status == "RUNNING":
                    percentage = float(job.get("percentageCompleted", 0))
                    scored_rows = int(job.get("scoredRows", 0))
                    failed_rows = int(job.get("failedRows", 0))
                    visible_progress = min(95, max(15, int(percentage)))
                    progress_bar.progress(visible_progress)
                    status_box.info(
                        f"Procesando predicciones: {percentage:.0f}% · "
                        f"Filas procesadas: {scored_rows:,} · Errores: {failed_rows:,}"
                    )

                elif status == "COMPLETED":
                    progress_bar.progress(98)
                    status_box.info("Predicciones completadas. Descargando resultados...")
                    final_job = self.get_job(job_url)
                    result = self.download_results(final_job["links"]["download"])
                    progress_bar.progress(100)
                    status_box.success("Predicciones generadas correctamente.")
                    return result

                elif status in {"FAILED", "ABORTED"}:
                    details = job.get("statusDetails") or job.get("logs") or "Sin detalles."
                    raise DataRobotPredictionError(
                        f"El trabajo terminó con estado {status}: {details}"
                    )

                else:
                    raise DataRobotPredictionError(
                        f"DataRobot devolvió un estado no reconocido: {status}"
                    )

                time.sleep(POLL_INTERVAL_SECONDS)

        except Exception:
            self.abort_job(job_id)
            raise


# =========================================================
# UTILIDADES
# =========================================================
def get_secret(name: str, default: str | None = None) -> str:
    try:
        value = st.secrets[name]
    except (KeyError, FileNotFoundError):
        value = default

    if value is None or not str(value).strip():
        raise DataRobotPredictionError(
            f"Falta configurar el secreto '{name}' en Streamlit."
        )
    return str(value).strip()


def read_csv_safely(file_bytes: bytes) -> tuple[pd.DataFrame, str]:
    encodings = ("utf-8", "utf-8-sig", "latin-1")
    last_error: Exception | None = None

    for encoding in encodings:
        try:
            dataframe = pd.read_csv(io.BytesIO(file_bytes), encoding=encoding)
            return dataframe, encoding
        except Exception as exc:
            last_error = exc

    raise DataRobotPredictionError(
        f"No fue posible leer el CSV. Verifica el separador, la codificación y su estructura. "
        f"Detalle: {last_error}"
    )


def dataframe_from_result(result_bytes: bytes) -> pd.DataFrame | None:
    try:
        return pd.read_csv(io.BytesIO(result_bytes))
    except Exception:
        return None


# =========================================================
# INTERFAZ
# =========================================================
st.markdown(
    """
    <div class="hero">
        <h1>Predicción inteligente con DataRobot</h1>
        <p>
            Carga un archivo CSV, envíalo al modelo desplegado en DataRobot y descarga
            los resultados de predicción desde una interfaz clara y segura.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.title("Configuración")
    st.caption("Las credenciales se obtienen desde los secretos de Streamlit.")

    include_all_columns = st.checkbox(
        "Conservar columnas originales",
        value=True,
        help="Agrega las columnas del archivo de entrada al resultado final.",
    )
    include_prediction_status = st.checkbox(
        "Incluir estado por fila",
        value=True,
        help="Agrega información de errores o estado para cada registro.",
    )
    request_explanations = st.checkbox(
        "Solicitar explicaciones",
        value=False,
        help="Puede aumentar el tiempo de procesamiento y depende de la configuración del deployment.",
    )
    max_explanations = st.slider(
        "Número de explicaciones",
        min_value=1,
        max_value=10,
        value=3,
        disabled=not request_explanations,
    )

    st.divider()
    st.markdown("**Secretos requeridos**")
    st.code(
        'DATAROBOT_API_KEY = "..."\n'
        'DATAROBOT_DEPLOYMENT_ID = "..."\n'
        'DATAROBOT_HOST = "https://app.datarobot.com"',
        language="toml",
    )

uploaded_file = st.file_uploader(
    "Carga el archivo que deseas analizar",
    type=["csv"],
    help="El archivo debe contener exactamente las variables esperadas por el deployment.",
)

if uploaded_file is None:
    st.info("Carga un archivo CSV para comenzar.")
    st.stop()

input_bytes = uploaded_file.getvalue()

try:
    input_df, detected_encoding = read_csv_safely(input_bytes)
except DataRobotPredictionError as exc:
    st.error(str(exc))
    st.stop()

if input_df.empty:
    st.warning("El archivo está vacío. Agrega al menos una fila de datos.")
    st.stop()

metric_cols = st.columns(3)
with metric_cols[0]:
    st.markdown(
        f'<div class="info-card"><div class="info-label">Registros</div>'
        f'<div class="info-value">{len(input_df):,}</div></div>',
        unsafe_allow_html=True,
    )
with metric_cols[1]:
    st.markdown(
        f'<div class="info-card"><div class="info-label">Variables</div>'
        f'<div class="info-value">{len(input_df.columns):,}</div></div>',
        unsafe_allow_html=True,
    )
with metric_cols[2]:
    st.markdown(
        f'<div class="info-card"><div class="info-label">Codificación</div>'
        f'<div class="info-value">{detected_encoding}</div></div>',
        unsafe_allow_html=True,
    )

st.subheader("Vista previa de los datos")
st.dataframe(input_df.head(100), use_container_width=True, hide_index=True)
st.caption(
    "La vista previa muestra como máximo 100 filas. El archivo completo será enviado a DataRobot."
)

with st.expander("Revisar nombres de variables"):
    st.write(list(input_df.columns))

predict_button = st.button(
    "Generar predicciones",
    type="primary",
    use_container_width=True,
)

if predict_button:
    try:
        config = DataRobotConfig(
            api_key=get_secret("DATAROBOT_API_KEY"),
            deployment_id=get_secret("DATAROBOT_DEPLOYMENT_ID"),
            host=get_secret("DATAROBOT_HOST", "https://app.datarobot.com"),
        )

        # Se normaliza a UTF-8 para evitar problemas de codificación al subirlo.
        normalized_csv = input_df.to_csv(index=False).encode("utf-8")

        progress_bar = st.progress(0)
        status_box = st.empty()
        client = DataRobotBatchClient(config)

        result_bytes = client.predict(
            csv_bytes=normalized_csv,
            include_all_columns=include_all_columns,
            include_prediction_status=include_prediction_status,
            max_explanations=max_explanations if request_explanations else None,
            progress_bar=progress_bar,
            status_box=status_box,
        )

        st.session_state["prediction_result"] = result_bytes
        st.session_state["prediction_filename"] = (
            f"predicciones_{uploaded_file.name.rsplit('.', 1)[0]}.csv"
        )

    except DataRobotPredictionError as exc:
        st.error(str(exc))
    except Exception as exc:
        st.error(f"Ocurrió un error inesperado: {exc}")

if "prediction_result" in st.session_state:
    result_bytes = st.session_state["prediction_result"]
    result_df = dataframe_from_result(result_bytes)

    st.divider()
    st.subheader("Resultados del modelo")

    if result_df is not None:
        result_metrics = st.columns(2)
        result_metrics[0].metric("Filas resultantes", f"{len(result_df):,}")
        result_metrics[1].metric("Columnas resultantes", f"{len(result_df.columns):,}")
        st.dataframe(result_df.head(100), use_container_width=True, hide_index=True)
    else:
        st.warning(
            "Las predicciones se generaron, pero no fue posible mostrar una vista previa. "
            "Puedes descargar el archivo completo."
        )

    st.download_button(
        label="Descargar predicciones en CSV",
        data=result_bytes,
        file_name=st.session_state.get("prediction_filename", "predicciones.csv"),
        mime="text/csv",
        use_container_width=True,
    )

st.markdown(
    '<p class="small-note">La aplicación no muestra ni almacena la API Key. '
    'Las credenciales se leen directamente desde <code>st.secrets</code>.</p>',
    unsafe_allow_html=True,
)
