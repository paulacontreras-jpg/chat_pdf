import os
import platform

import streamlit as st
from PIL import Image
from PyPDF2 import PdfReader

from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_community.vectorstores import FAISS


# ============================================================
# CONFIGURACIÓN DE LA PÁGINA
# ============================================================

st.set_page_config(
    page_title="La Biblioteca Mágica",
    page_icon="📜",
    layout="centered"
)


# ============================================================
# ESTILOS (TEMÁTICA: BIBLIOTECA MÁGICA)
# ============================================================

st.markdown("""
<style>

    @import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@500;700&family=Lora:ital,wght@0,400;0,600;1,400&display=swap');

    /* ---------- Paleta ---------- */
    :root {
        --pergamino: #F4E9D0;
        --pergamino-claro: #FBF5E6;
        --madera: #2E1F14;
        --madera-clara: #4A3222;
        --dorado: #C9A227;
        --dorado-claro: #E6C75A;
        --burdeos: #6B1E2E;
        --tinta: #3B2A1A;
    }

    /* Fondo general: pergamino con luz suave */
    .stApp {
        background:
            radial-gradient(circle at 20% 10%, #FBF5E6 0%, transparent 45%),
            radial-gradient(circle at 80% 90%, #E9D8B0 0%, transparent 50%),
            var(--pergamino);
        font-family: 'Lora', serif;
    }

    /* Contenedor principal */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 900px;
    }

    /* Título principal */
    h1 {
        font-family: 'Cinzel', serif !important;
        color: var(--burdeos) !important;
        text-align: center;
        font-size: 44px !important;
        font-weight: 700 !important;
        letter-spacing: 2px;
        margin-bottom: 5px;
        text-shadow: 1px 1px 0px rgba(201, 162, 39, 0.4);
    }

    /* Subtítulos */
    h2, h3 {
        font-family: 'Cinzel', serif !important;
        color: var(--madera-clara) !important;
        letter-spacing: 1px;
    }

    /* Texto */
    p, label, li {
        font-family: 'Lora', serif;
        color: var(--tinta);
    }

    /* Barra lateral: madera oscura */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, var(--madera) 0%, var(--madera-clara) 100%);
        border-right: 4px solid var(--dorado);
    }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: var(--dorado-claro) !important;
    }

    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] .stCaption {
        color: #F4E9D0 !important;
    }

    [data-testid="stSidebar"] hr {
        border-color: var(--dorado);
        opacity: 0.5;
    }

    /* Cargador de PDF: se ve como un pergamino */
    [data-testid="stFileUploader"] {
        background-color: var(--pergamino-claro);
        border: 2px dashed var(--dorado);
        border-radius: 15px;
        padding: 15px;
        box-shadow: inset 0 0 20px rgba(201, 162, 39, 0.15);
    }

    /* Campo de la clave */
    .stTextInput input {
        background-color: var(--pergamino-claro) !important;
        border: 2px solid var(--dorado) !important;
        border-radius: 10px !important;
        color: var(--tinta) !important;
    }

    /* Caja de pregunta */
    textarea {
        background-color: var(--pergamino-claro) !important;
        border: 2px solid var(--dorado) !important;
        border-radius: 12px !important;
        color: var(--tinta) !important;
        font-family: 'Lora', serif !important;
        font-style: italic;
    }

    /* Botones */
    .stButton > button {
        background: linear-gradient(135deg, var(--burdeos), #8B2A3F);
        color: var(--dorado-claro);
        font-family: 'Cinzel', serif;
        border-radius: 10px;
        border: 2px solid var(--dorado);
        padding: 10px 25px;
        font-weight: bold;
    }

    .stButton > button:hover {
        background: var(--dorado);
        color: var(--madera);
        border-color: var(--madera);
    }

    /* Alertas */
    [data-testid="stAlert"] {
        border-radius: 12px;
        border: 1px solid rgba(201, 162, 39, 0.5);
    }

    /* Divisores */
    hr {
        border-color: var(--dorado) !important;
        opacity: 0.6;
    }

    /* Caja de respuesta: página de grimorio */
    .respuesta {
        background: var(--pergamino-claro);
        padding: 25px 28px;
        border-radius: 6px 18px 6px 18px;
        border: 2px solid var(--dorado);
        border-left: 8px solid var(--burdeos);
        margin-top: 15px;
        color: var(--tinta);
        line-height: 1.8;
        font-size: 17px;
        box-shadow: 0px 6px 18px rgba(46, 31, 20, 0.25);
    }

    .respuesta::first-letter {
        font-family: 'Cinzel', serif;
        font-size: 36px;
        color: var(--burdeos);
        font-weight: 700;
    }

    /* Texto pequeño */
    .texto-pequeno {
        text-align: center;
        color: var(--madera-clara);
        font-size: 16px;
        font-style: italic;
    }

    /* Decoración */
    .adorno {
        text-align: center;
        color: var(--dorado);
        font-size: 24px;
        letter-spacing: 12px;
        margin: 8px 0 18px 0;
    }

    /* Pie de página */
    .pie {
        text-align: center;
        color: var(--madera-clara);
        font-size: 13px;
        margin-top: 40px;
        font-style: italic;
    }

</style>
""", unsafe_allow_html=True)


# ============================================================
# TÍTULO
# ============================================================

st.title("📜 La Biblioteca Mágica")

st.markdown(
    "<p class='texto-pequeno'>"
    "Entrega tu pergamino al bibliotecario y pregúntale lo que desees "
    "saber sobre su contenido."
    "</p>"
    "<div class='adorno'>✦ ❖ ✦</div>",
    unsafe_allow_html=True
)


# ============================================================
# IMAGEN
# ============================================================

try:

    image = Image.open("Chat_pdf.png")
    st.image(image, width=350)

except Exception as e:

    st.warning(
        f"No se pudo cargar la imagen: {e}"
    )


# ============================================================
# BARRA LATERAL
# ============================================================

with st.sidebar:

    st.header("🕯️ Sobre la Biblioteca")

    st.write(
        "El bibliotecario lee el pergamino (PDF) que le entregues y "
        "responde tus preguntas utilizando únicamente lo que está "
        "escrito en él."
    )

    st.info(
        "🗝️ Consejo\n\n"
        "Pregunta sobre el contenido del documento para obtener "
        "mejores respuestas."
    )

    st.divider()

    st.write("🐸 **Magia utilizada: RAG**")

    st.write(
        "Primero se buscan los pasajes más relevantes dentro "
        "del documento y después la inteligencia artificial "
        "redacta la respuesta."
    )

    st.divider()

    st.caption(
        f"Versión de Python: {platform.python_version()}"
    )


# ============================================================
# CLAVE DE OPENAI
# ============================================================

st.subheader("🔑 La llave de la biblioteca")

ke = st.text_input(
    "Ingresa tu Clave de OpenAI",
    type="password",
    placeholder="sk-..."
)

if ke:

    os.environ["OPENAI_API_KEY"] = ke

else:

    st.warning(
        "⚠️ Necesitas tu clave de API de OpenAI para abrir las puertas "
        "de la biblioteca."
    )


# ============================================================
# CARGAR PDF
# ============================================================

st.subheader("📜 Entrega tu pergamino")

pdf = st.file_uploader(
    "Selecciona un archivo PDF",
    type=["pdf"]
)


# ============================================================
# PROCESAMIENTO DEL PDF
# ============================================================

if pdf is not None and ke:

    try:

        # ========================================================
        # LEER PDF
        # ========================================================

        pdf_reader = PdfReader(pdf)

        text = ""

        for page in pdf_reader.pages:

            page_text = page.extract_text()

            if page_text:

                text += page_text + "\n"


        # ========================================================
        # COMPROBAR SI SE EXTRAJO TEXTO
        # ========================================================

        if not text.strip():

            st.error(
                "❌ No se pudo extraer texto de este PDF."
            )

            st.info(
                "El pergamino puede ser un documento escaneado "
                "o estar protegido con un hechizo."
            )

            st.stop()


        # ========================================================
        # INFORMACIÓN DEL DOCUMENTO
        # ========================================================

        st.success(
            f"✅ Pergamino recibido: **{pdf.name}**"
        )

        st.info(
            f"📊 Se descifraron {len(text):,} caracteres."
        )


        # ========================================================
        # DIVIDIR EL TEXTO
        # ========================================================

        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=500,
            chunk_overlap=20,
            length_function=len
        )

        chunks = text_splitter.split_text(text)

        st.success(
            f"🧩 El pergamino se dividió en {len(chunks)} fragmentos."
        )


        # ========================================================
        # CREAR EMBEDDINGS Y BASE DE CONOCIMIENTO
        # ========================================================

        with st.spinner(
            "🔮 El bibliotecario está leyendo el pergamino..."
        ):

            embeddings = OpenAIEmbeddings()

            knowledge_base = FAISS.from_texts(
                chunks,
                embeddings
            )

        st.success(
            "📚 El pergamino ya está en los estantes. ¡Puedes preguntar!"
        )


        # ========================================================
        # PREGUNTA
        # ========================================================

        st.markdown("<div class='adorno'>✦ ❖ ✦</div>", unsafe_allow_html=True)

        st.subheader("🪶 Haz tu pregunta al bibliotecario")

        user_question = st.text_area(
            "Escribe tu pregunta:",
            placeholder=(
                "Ejemplo: ¿Cuál es la conclusión principal "
                "del documento?"
            ),
            height=120
        )


        # ========================================================
        # PROCESAR PREGUNTA
        # ========================================================

        if user_question.strip():

            # ====================================================
            # BUSCAR INFORMACIÓN
            # ====================================================

            with st.spinner(
                "🕯️ Buscando entre los estantes del pergamino..."
            ):

                docs_with_scores = (
                    knowledge_base.similarity_search_with_score(
                        user_question,
                        k=4
                    )
                )


            # ====================================================
            # FILTRAR RESULTADOS
            # ====================================================

            docs = []

            for doc, score in docs_with_scores:

                # Menor distancia = mayor similitud
                if score < 0.8:

                    docs.append(doc)


            # ====================================================
            # NO SE ENCONTRÓ INFORMACIÓN
            # ====================================================

            if not docs:

                st.warning(
                    "⚠️ El bibliotecario no encontró pasajes "
                    "relacionados con esa pregunta en el pergamino."
                )

                st.info(
                    "💡 Intenta preguntar algo sobre el contenido "
                    "del documento."
                )


            # ====================================================
            # SE ENCONTRÓ INFORMACIÓN
            # ====================================================

            else:

                st.success(
                    f"🔎 Se encontraron {len(docs)} pasajes "
                    "relacionados con tu pregunta."
                )


                # =================================================
                # CREAR MODELO
                # =================================================

                llm = OpenAI(
                    temperature=0,
                    model_name="gpt-4o-mini-2024-07-18"
                )


                # =================================================
                # CREAR CONTEXTO
                # =================================================

                context = "\n\n".join(
                    [
                        doc.page_content
                        for doc in docs
                    ]
                )


                # =================================================
                # PROMPT
                # =================================================

                final_prompt = f"""
Responde la pregunta utilizando ÚNICAMENTE
la información proporcionada en el contexto.

REGLAS IMPORTANTES:

1. No utilices información externa al PDF.
2. No inventes información.
3. No completes información utilizando conocimientos propios.
4. Si la respuesta no aparece en el contexto,
   debes indicarlo.
5. Si la pregunta no está relacionada con el documento,
   no la respondas.

Si no existe información suficiente para responder,
responde exactamente:

"No puedo responder esta pregunta porque
la información necesaria no aparece en el PDF."

CONTEXTO DEL PDF:

{context}

PREGUNTA DEL USUARIO:

{user_question}

RESPUESTA:
"""


                # =================================================
                # GENERAR RESPUESTA
                # =================================================

                with st.spinner(
                    "✨ El bibliotecario está escribiendo la respuesta..."
                ):

                    response = llm.invoke(
                        final_prompt
                    )


                # =================================================
                # MOSTRAR RESPUESTA
                # =================================================

                st.markdown(
                    "### 📖 Respuesta del bibliotecario"
                )

                st.markdown(
                    f"""
                    <div class="respuesta">
                    {response}
                    </div>
                    """,
                    unsafe_allow_html=True
                )


    # ============================================================
    # MANEJO DE ERRORES
    # ============================================================

    except Exception as e:

        st.error(
            f"❌ Un hechizo falló al procesar el PDF: {str(e)}"
        )

        with st.expander(
            "🔧 Ver detalles técnicos"
        ):

            import traceback

            st.code(
                traceback.format_exc()
            )


# ============================================================
# PDF SIN API KEY
# ============================================================

elif pdf is not None and not ke:

    st.warning(
        "🔑 Para comenzar, primero ingresa tu clave "
        "de API de OpenAI."
    )


# ============================================================
# SIN PDF
# ============================================================

else:

    st.info(
        "📜 Entrega un archivo PDF para comenzar."
    )


# ============================================================
# PIE DE PÁGINA
# ============================================================

st.markdown(
    "<div class='adorno'>✦ ❖ ✦</div>"
    "<p class='pie'>La Biblioteca Mágica · donde cada PDF guarda un secreto</p>",
    unsafe_allow_html=True
)
