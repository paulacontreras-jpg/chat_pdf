import os
import streamlit as st
from PIL import Image
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
import platform


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

    [data-testid="stSidebar"] h1,
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

st.write("")


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
        "Haz preguntas relacionadas con el contenido del PDF."
    )

    st.divider()

    st.write("🐸 **RAG:** Recuperación Aumentada por Generación")

    st.write(
        "El sistema busca primero información relevante dentro "
        "del documento y después utiliza la IA para construir "
        "la respuesta."
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
        "⚠️ Ingresa tu clave de API de OpenAI para utilizar el asistente."
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

        # --------------------------------------------------------
        # LEER PDF
        # --------------------------------------------------------

        pdf_reader = PdfReader(pdf)

        text = ""

        for page in pdf_reader.pages:

            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"


        # --------------------------------------------------------
        # INFORMACIÓN DEL DOCUMENTO
        # --------------------------------------------------------

        st.success(
            f"✅ PDF cargado correctamente: **{pdf.name}**"
        )

        st.info(
            f"📊 Texto extraído: {len(text):,} caracteres"
        )


        # --------------------------------------------------------
        # COMPROBAR SI EL PDF TIENE TEXTO
        # --------------------------------------------------------

        if not text.strip():

            st.error(
                "❌ No se pudo extraer texto de este PDF. "
                "Puede que sea un documento escaneado o esté protegido."
            )

            st.stop()


        # --------------------------------------------------------
        # DIVIDIR TEXTO
        # --------------------------------------------------------

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


        # --------------------------------------------------------
        # CREAR EMBEDDINGS
        # --------------------------------------------------------

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
        # PREGUNTAS
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

        if user_question:

            with st.spinner("🔍 Buscando información en el PDF..."):

                # Buscar fragmentos relacionados
                docs_with_scores = (
                    knowledge_base.similarity_search_with_score(
                        user_question,
                        k=4
                    )
                )


            # ----------------------------------------------------
            # FILTRAR RESULTADOS
            # ----------------------------------------------------

            docs = []

            for doc, score in docs_with_scores:

                # Menor distancia = mayor similitud
                if score < 0.8:
                    docs.append(doc)


            # ----------------------------------------------------
            # SI NO HAY INFORMACIÓN RELACIONADA
            # ----------------------------------------------------

            if not docs:

                st.warning(
                    "⚠️ Esta pregunta no parece estar relacionada "
                    "con el contenido del PDF."
                )

                st.info(
                    "💡 Intenta realizar una pregunta sobre la "
                    "información que aparece en el documento."
                )


            # ----------------------------------------------------
            # SI ENCUENTRA INFORMACIÓN
            # ----------------------------------------------------

            else:

                st.success(
                    f"🔎 Se encontraron {len(docs)} fragmentos "
                    "relacionados con tu pregunta."
                )


                # ------------------------------------------------
                # MODELO DE OPENAI
                # ------------------------------------------------

                llm = OpenAI(
                    temperature=0,
                    model_name="gpt-4o-mini-2024-07-18"
                )


                # ------------------------------------------------
                # CADENA DE PREGUNTAS Y RESPUESTAS
                # ------------------------------------------------

                chain = load_qa_chain(
                    llm,
                    chain_type="stuff"
                )


                # ------------------------------------------------
                # INSTRUCCIÓN PARA LA IA
                # ------------------------------------------------

                prompt = f"""
                Responde la siguiente pregunta utilizando
                EXCLUSIVAMENTE la información contenida en los
                fragmentos proporcionados del PDF.

                No utilices conocimiento externo.

                No inventes información.

                Si los fragmentos no contienen suficiente
                información para responder la pregunta, responde
                exactamente:

                "No puedo responder esta pregunta porque la
                información necesaria no aparece en el PDF."

                Pregunta del usuario:

                {user_question}
                """


                # ------------------------------------------------
                # GENERAR RESPUESTA
                # ------------------------------------------------

                with st.spinner("🤖 Generando respuesta..."):

                    response = chain.run(
                        input_documents=docs,
                        question=prompt
                    )


                # ------------------------------------------------
                # MOSTRAR RESPUESTA
                # ------------------------------------------------

                st.markdown("### 💡 Respuesta")

                st.markdown(
                    f"""
                    <div class="respuesta">
                    {response}
                    </div>
                    """,
                    unsafe_allow_html=True
                )


# ============================================================
# MENSAJES INICIALES
# ============================================================

elif pdf is not None and not ke:

    st.warning(
        "🔑 Para comenzar, primero ingresa tu clave de API de OpenAI."
    )

else:

    st.info(
        "📄 Carga un archivo PDF para comenzar."
    )
