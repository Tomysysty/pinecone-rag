# Pinecone: de CRUD a RAG

Tres ejercicios en Python con los que aprendí bases de datos vectoriales. Cada uno usa lo del anterior y el último es un chatbot que responde con documentos propios (RAG).

**Python · Pinecone · sentence-transformers (all-MiniLM-L6-v2) · OpenAI (opcional)**

| # | Archivo | Qué hace |
|---|---|---|
| 1 | [`ejercicio_1_crud_basico.py`](ejercicio_1_crud_basico.py) | Crea un índice, inserta vectores con metadatos y los busca, filtra, actualiza y borra |
| 2 | [`ejercicio_2_busqueda_semantica.py`](ejercicio_2_busqueda_semantica.py) | Convierte frases en embeddings de 384 dimensiones y busca por significado, no por palabras exactas |
| 3 | [`ejercicio_3_rag_chatbot.py`](ejercicio_3_rag_chatbot.py) | Parte el manual de una empresa ficticia en fragmentos con solapamiento, los indexa y responde preguntas con los fragmentos más parecidos |

El chatbot funciona sin OpenAI: en ese caso muestra los fragmentos que encontró en vez de redactar la respuesta.

## Correrlo

```bash
pip install -r requirements.txt
export PINECONE_API_KEY="tu_clave"       # en PowerShell: $env:PINECONE_API_KEY="tu_clave"
export OPENAI_API_KEY="tu_clave"         # opcional, solo para el ejercicio 3
python ejercicio_1_crud_basico.py
```

La clave de Pinecone es gratis en https://app.pinecone.io/.

---

Mayo de 2026
