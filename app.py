import os
import platform

import streamlit as st
from PIL import Image
from PyPDF2 import PdfReader

from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain


# ============================================================
# CONFIGURACIÓN DE LA PÁGINA
# ============================================================

st.set_page_config(
    page_title="ChatPDF",
    page_icon="💬",
    layout="centered"
)


# ============================================================
# ESTILOS
# ============================================================

st.markdown("""
<style>

    /* Fondo general */
    .stApp {
        background-color: #EAF4FF;
    }

    /* Contenedor principal */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 900px;
    }

    /* Título */
    h1 {
        color: #0B4F8A;
        text-align: center;
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    /* Subtítulos */
    h2, h3 {
        color: #1261A0;
    }

    /* Texto */
    p, label {
        color: #183B56;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #D6ECFF;
    }

    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #0B4F8A;
    }

    /* Zona para subir PDF */
    [data-testid="stFileUploader"] {
        background-color: #FFFFFF;
        border: 2px dashed #4A90E2;
        border-radius: 15px;
        padding: 15px;
    }

    /* Caja de preguntas */
    textarea {
        border: 2px solid #4A90E2 !important;
        border-radius: 12px !important;
        background-color: white !important;
    }

    /* Botones */
    .stButton > button {
        background-color: #1261A0;
        color: white;
        border-radius: 10px;
        border: none;
        padding: 10px 25px;
        font-weight: bold;
    }

    .stButton > button:hover {
        background-color: #0B4F8A;
        color: white;
    }

    /* Mensajes */
    [data-testid="stAlert"] {
        border-radius: 12px;
    }

    /* Caja de respuesta */
    .respuesta {
        background-color: white;
        padding: 20px;
        border-radius: 15px;
        border-left: 5px solid #1261A0;
        margin-top: 15px;
        color: #183B56;
        line-height: 1.6;
    }

    /* Texto pequeño */
    .texto-pequeno {
        text-align: center;
        color: #55758D;
        font-size: 14px;
    }

</style>
""", unsafe_allow_html=True)


# ============================================================
# TÍTULO
# ============================================================

st.title("💬 ChatPDF")

st.markdown(
    "<p class='texto-pequeno'>"
    "Analiza un documento y realiza preguntas sobre su contenido."
    "</p>",
    unsafe_allow_html=True
)


# ============================================================
# IMAGEN
# ============================================================

try:
    image = Image.open("Chat_pdf.png")
    st.image(image, width=350)

except Exception as e:
    st.warning(f"No se pudo cargar la imagen: {e}")


# ============================================================
# BARRA LATERAL
# ============================================================

with st.sidebar:

    st.header("📘 Sobre ChatPDF")

    st.write(
        "Este asistente analiza el PDF que cargues y responde "
        "preguntas utilizando únicamente la información encontrada "
        "en el documento."
    )

    st.info(
        "💡 Consejo\n\n"
        "Realiza preguntas relacionadas con el contenido del PDF."
    )

    st.divider()

    st.write("🐸 **Tecnología:** RAG")

    st.write(
        "El sistema busca información relevante dentro del "
        "documento y utiliza inteligencia artificial para "
        "generar la respuesta."
    )

    st.divider()

    st.caption(
        f"Versión de Python: {platform.python_version()}"
    )


# ============================================================
# CLAVE DE OPENAI
# ============================================================

st.subheader("🔑 Conexión con OpenAI")

ke = st.text_input(
    "Ingresa tu Clave de OpenAI",
    type="password",
    placeholder="sk-..."
)

if ke:

    os.environ["OPENAI_API_KEY"] = ke

else:

    st.warning(
        "⚠️ Ingresa tu clave de API de OpenAI para continuar."
    )


# ============================================================
# CARGAR PDF
# ============================================================

st.subheader("📄 Carga tu documento")

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
        # COMPROBAR TEXTO
        # ========================================================

        if not text.strip():

            st.error(
                "❌ No se pudo extraer texto de este PDF."
            )

            st.info(
                "El archivo puede ser un documento escaneado "
                "o estar protegido."
            )

            st.stop()


        # ========================================================
        # INFORMACIÓN DEL DOCUMENTO
        # ========================================================

        st.success(
            f"✅ PDF cargado: **{pdf.name}**"
        )

        st.info(
            f"📊 Se extrajeron {len(text):,} caracteres."
        )


        # ========================================================
        # DIVIDIR TEXTO EN FRAGMENTOS
        # ========================================================

        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=500,
            chunk_overlap=20,
            length_function=len
        )

        chunks = text_splitter.split_text(text)

        st.success(
            f"🧩 Documento dividido en {len(chunks)} fragmentos."
        )


        # ========================================================
        # CREAR EMBEDDINGS Y BASE DE DATOS
        # ========================================================

        with st.spinner("🧠 Analizando el documento..."):

            embeddings = OpenAIEmbeddings()

            knowledge_base = FAISS.from_texts(
                chunks,
                embeddings
            )

        st.success(
            "📚 Documento listo para responder preguntas."
        )


        # ========================================================
        # PREGUNTA
        # ========================================================

        st.divider()

        st.subheader("🔎 Pregunta sobre tu documento")

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

            with st.spinner(
                "🔍 Buscando información en el PDF..."
            ):

                docs_with_scores = (
                    knowledge_base.similarity_search_with_score(
                        user_question,
                        k=4
                    )
                )


            # ====================================================
            # FILTRAR FRAGMENTOS
            # ====================================================

            docs = []

            for doc, score in docs_with_scores:

                # En FAISS, un valor menor significa
                # mayor similitud.
                if score < 0.8:

                    docs.append(doc)


            # ====================================================
            # NO SE ENCONTRÓ INFORMACIÓN SUFICIENTE
            # ====================================================

            if not docs:

                st.warning(
                    "⚠️ No encontré información relacionada "
                    "con esa pregunta dentro del PDF."
                )

                st.info(
                    "💡 Intenta formular una pregunta sobre "
                    "el contenido del documento."
                )


            # ====================================================
            # SE ENCONTRÓ INFORMACIÓN
            # ====================================================

            else:

                st.success(
                    f"🔎 Encontré {len(docs)} fragmentos "
                    "relacionados con tu pregunta."
                )


                # =================================================
                # MODELO DE OPENAI
                # =================================================

                llm = OpenAI(
                    temperature=0,
                    model_name="gpt-4o-mini-2024-07-18"
                )


                # =================================================
                # CADENA DE PREGUNTAS Y RESPUESTAS
                # =================================================

                chain = load_qa_chain(
                    llm,
                    chain_type="stuff"
                )


                # =================================================
                # INSTRUCCIÓN
                # =================================================

                prompt = f"""
Responde la pregunta del usuario utilizando ÚNICAMENTE
la información contenida en los fragmentos proporcionados
del PDF.

REGLAS:

1. No utilices información externa al PDF.
2. No inventes información.
3. No completes información utilizando conocimientos propios.
4. Si la respuesta no aparece o no puede determinarse
   a partir de los fragmentos proporcionados, responde:

"No puedo responder esta pregunta porque la información
necesaria no aparece en el PDF."

Pregunta del usuario:

{user_question}
"""


                # =================================================
                # GENERAR RESPUESTA
                # =================================================

                with st.spinner(
                    "🤖 Generando respuesta..."
                ):

                    response = chain.run(
                        input_documents=docs,
                        question=prompt
                    )


                # =================================================
                # MOSTRAR RESPUESTA
                # =================================================

                st.markdown(
                    "### 💡 Respuesta"
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
            f"❌ Ocurrió un error al procesar el PDF: {str(e)}"
        )

        with st.expander("🔧 Ver detalles técnicos"):

            import traceback

            st.code(
                traceback.format_exc()
            )


# ============================================================
# SI HAY PDF PERO NO HAY API KEY
# ============================================================

elif pdf is not None and not ke:

    st.warning(
        "🔑 Para comenzar, primero ingresa tu clave de API de OpenAI."
    )


# ============================================================
# SI TODAVÍA NO HAY PDF
# ============================================================

else:

    st.info(
        "📄 Carga un archivo PDF para comenzar."
    )
