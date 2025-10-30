import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import re
from nltk.stem import SnowballStemmer

# Configuración de la página
st.set_page_config(
    page_title="TF-IDF Español",
    page_icon="🔍",
    layout="wide"
)

# Estilos CSS personalizados
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 700;
    }
    .suggested-btn {
        background-color: #EFF6FF;
        border: 2px solid #3B82F6;
        color: #1E40AF;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        margin: 0.2rem 0;
        width: 100%;
        text-align: left;
    }
    .suggested-btn:hover {
        background-color: #DBEAFE;
        border-color: #1D4ED8;
    }
    .result-high {
        background-color: #D1FAE5;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #10B981;
        margin: 1rem 0;
    }
    .result-low {
        background-color: #FEF3C7;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #F59E0B;
        margin: 1rem 0;
    }
    .matrix-header {
        background-color: #1E3A8A;
        color: white;
        padding: 1rem;
        border-radius: 8px 8px 0 0;
    }
</style>
""", unsafe_allow_html=True)

# Header principal
st.markdown('<h1 class="main-title">🔍 Demo TF-IDF en Español</h1>', unsafe_allow_html=True)

# Documentos de ejemplo
default_docs = """El perro ladra fuerte en el parque.
El gato maúlla suavemente durante la noche.
El perro y el gato juegan juntos en el jardín.
Los niños corren y se divierten en el parque.
La música suena muy alta en la fiesta.
Los pájaros cantan hermosas melodías al amanecer."""

# Stemmer en español
stemmer = SnowballStemmer("spanish")

def tokenize_and_stem(text):
    # Minúsculas
    text = text.lower()
    # Solo letras españolas y espacios
    text = re.sub(r'[^a-záéíóúüñ\s]', ' ', text)
    # Tokenizar
    tokens = [t for t in text.split() if len(t) > 1]
    # Aplicar stemming
    stems = [stemmer.stem(t) for t in tokens]
    return stems

# Layout en dos columnas
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📝 Documentos de Entrada")
    text_input = st.text_area(
        "Escribe tus documentos (uno por línea):", 
        default_docs, 
        height=150,
        label_visibility="collapsed"
    )
    
    st.subheader("❓ Realiza tu Pregunta")
    question = st.text_input(
        "Escribe tu pregunta:", 
        "¿Dónde juegan el perro y el gato?",
        label_visibility="collapsed"
    )

with col2:
    st.subheader("💡 Preguntas Sugeridas")
    
    # Preguntas sugeridas con mejor estilo
    suggested_questions = [
        "¿Dónde juegan el perro y el gato?",
        "¿Qué hacen los niños en el parque?",
        "¿Cuándo cantan los pájaros?",
        "¿Dónde suena la música alta?",
        "¿Qué animal maúlla durante la noche?"
    ]
    
    for i, suggested_q in enumerate(suggested_questions):
        if st.button(suggested_q, key=f"btn_{i}", use_container_width=True):
            st.session_state.question = suggested_q
            st.rerun()

# Actualizar pregunta si se seleccionó una sugerida
if 'question' in st.session_state:
    question = st.session_state.question

# Botón de análisis
col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
with col_btn2:
    if st.button("🔍 Analizar Documentos", type="primary", use_container_width=True):
        documents = [d.strip() for d in text_input.split("\n") if d.strip()]
        
        if len(documents) < 1:
            st.error("⚠️ Ingresa al menos un documento.")
        elif not question.strip():
            st.error("⚠️ Escribe una pregunta.")
        else:
            # Crear vectorizador TF-IDF
            vectorizer = TfidfVectorizer(
                tokenizer=tokenize_and_stem,
                min_df=1  # Incluir todas las palabras
            )
            
            # Ajustar con documentos
            X = vectorizer.fit_transform(documents)
            
            # Mostrar matriz TF-IDF con mejor estilo
            st.markdown("### 📊 Matriz TF-IDF")
            st.markdown('<div class="matrix-header"><strong>Términos y Pesos TF-IDF</strong></div>', unsafe_allow_html=True)
            df_tfidf = pd.DataFrame(
                X.toarray(),
                columns=vectorizer.get_feature_names_out(),
                index=[f"📄 Doc {i+1}" for i in range(len(documents))]
            )
            st.dataframe(df_tfidf.round(3), use_container_width=True)
            
            # Calcular similitud con la pregunta
            question_vec = vectorizer.transform([question])
            similarities = cosine_similarity(question_vec, X).flatten()
            
            # Encontrar mejor respuesta
            best_idx = similarities.argmax()
            best_doc = documents[best_idx]
            best_score = similarities[best_idx]
            
            # Mostrar respuesta con estilos diferentes según confianza
            st.markdown("### 🎯 Resultado del Análisis")
            
            if best_score > 0.1:  # Umbral ajustado
                st.markdown(f"""
                <div class="result-high">
                <h4>❓ <strong>Tu pregunta:</strong> {question}</h4>
                <h4>✅ <strong>Respuesta encontrada:</strong> {best_doc}</h4>
                <h4>📈 <strong>Nivel de confianza:</strong> {best_score:.3f}</h4>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-low">
                <h4>❓ <strong>Tu pregunta:</strong> {question}</h4>
                <h4>⚠️ <strong>Respuesta (baja confianza):</strong> {best_doc}</h4>
                <h4>📉 <strong>Nivel de confianza:</strong> {best_score:.3f}</h4>
                </div>
                """, unsafe_allow_html=True)
            
            # Mostrar todas las similitudes
            st.markdown("### 📈 Similitudes con Todos los Documentos")
            sim_df = pd.DataFrame({
                "Documento": [f"Doc {i+1}" for i in range(len(documents))],
                "Texto": documents,
                "Similitud": similarities.round(3)
            }).sort_values("Similitud", ascending=False)
            
            st.dataframe(sim_df, use_container_width=True)

# Información en el sidebar
with st.sidebar:
    st.markdown("### ℹ️ Sobre el Análisis")
    st.info("""
    **TF-IDF en Español** analiza la similitud entre tu pregunta y los documentos usando:
    
    • **Stemming**: Reduce palabras a su raíz
    • **TF-IDF**: Mide importancia de términos
    • **Similitud coseno**: Calcula parecido entre textos
    
    **Ejemplo:** "juegan" → "jueg" (stem)
    """)
    
    st.markdown("### 💡 Consejos")
    st.write("""
    • Usa preguntas específicas
    • Incluye palabras clave
    • Revisa los stems en la matriz
    • Considera sinónimos
    """)
