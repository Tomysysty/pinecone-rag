"""
EJERCICIO 2: BUSQUEDA SEMANTICA CON PINECONE + SENTENCE TRANSFORMERS
=====================================================================
Objetivo: Construir un buscador semantico REAL que entienda el
SIGNIFICADO de las frases, no solo palabras clave.

Que aprenderas:
 - Generar embeddings reales con sentence-transformers
 - Indexar documentos de texto en Pinecone
 - Buscar por significado (no por palabras exactas)
 - Usar namespaces para organizar datos

INSTRUCCIONES:
  1. pip install pinecone sentence-transformers
  2. Guarda tu clave en la variable de entorno PINECONE_API_KEY
  3. python ejercicio_2_busqueda_semantica.py
"""

import os
import time
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

API_KEY = os.environ.get("PINECONE_API_KEY", "")
INDEX_NAME = "busqueda-semantica"

def sep(titulo):
    print(f"\n{'='*60}\n  {titulo}\n{'='*60}")


# =====================================================================
# BASE DE CONOCIMIENTO
# Estos son los documentos que vamos a indexar y buscar.
# =====================================================================
DOCUMENTOS = [
    {"id": "doc-1",  "categoria": "salud",
     "texto": "El ejercicio cardiovascular mejora la salud del corazon y reduce el riesgo de enfermedades"},
    {"id": "doc-2",  "categoria": "salud",
     "texto": "Dormir 8 horas diarias es esencial para la recuperacion muscular y la salud mental"},
    {"id": "doc-3",  "categoria": "tecnologia",
     "texto": "La inteligencia artificial esta transformando la forma en que las empresas procesan datos"},
    {"id": "doc-4",  "categoria": "tecnologia",
     "texto": "Python es el lenguaje de programacion mas popular para ciencia de datos y machine learning"},
    {"id": "doc-5",  "categoria": "finanzas",
     "texto": "Invertir en fondos indexados es una estrategia de bajo riesgo para el largo plazo"},
    {"id": "doc-6",  "categoria": "finanzas",
     "texto": "El ahorro sistematico y la diversificacion son claves para la libertad financiera"},
    {"id": "doc-7",  "categoria": "salud",
     "texto": "Una dieta balanceada rica en proteinas y vegetales fortalece el sistema inmunologico"},
    {"id": "doc-8",  "categoria": "tecnologia",
     "texto": "Las bases de datos vectoriales permiten buscar informacion por significado semantico"},
    {"id": "doc-9",  "categoria": "educacion",
     "texto": "El aprendizaje activo y la practica espaciada son las tecnicas de estudio mas efectivas"},
    {"id": "doc-10", "categoria": "educacion",
     "texto": "Los cursos en linea democratizan el acceso a educacion de calidad mundial"},
]


def main():
    # PASO 1: CARGAR MODELO DE EMBEDDINGS
    # Este modelo convierte texto -> vectores numericos de 384 dimensiones.
    # Textos con significado similar tendran vectores cercanos.
    sep("PASO 1: Cargando modelo de embeddings")
    modelo = SentenceTransformer("all-MiniLM-L6-v2")
    dimension = 384  # Este modelo genera vectores de 384 dimensiones
    print(f"Modelo cargado: all-MiniLM-L6-v2 (dim={dimension})")

    # Ejemplo: ver como se ve un embedding
    ejemplo = modelo.encode("Hola mundo")
    print(f"Ejemplo de embedding: [{ejemplo[0]:.4f}, {ejemplo[1]:.4f}, ...] ({len(ejemplo)} dims)")

    # PASO 2: CONECTAR Y CREAR INDICE
    sep("PASO 2: Conectando a Pinecone")
    pc = Pinecone(api_key=API_KEY)
    existentes = [idx.name for idx in pc.list_indexes()]
    if INDEX_NAME not in existentes:
        pc.create_index(
            name=INDEX_NAME, dimension=dimension, metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        print(f"Indice '{INDEX_NAME}' creado. Esperando...")
        time.sleep(10)
    else:
        print(f"Indice '{INDEX_NAME}' ya existe")
    index = pc.Index(INDEX_NAME)

    # PASO 3: GENERAR EMBEDDINGS E INDEXAR
    # Convertimos cada texto a su vector y lo subimos a Pinecone
    sep("PASO 3: Indexando documentos")
    vectores = []
    for doc in DOCUMENTOS:
        embedding = modelo.encode(doc["texto"]).tolist()
        vectores.append({
            "id": doc["id"],
            "values": embedding,
            "metadata": {
                "texto": doc["texto"],
                "categoria": doc["categoria"]
            }
        })
    index.upsert(vectors=vectores)
    print(f"{len(vectores)} documentos indexados")
    time.sleep(3)

    # PASO 4: BUSQUEDA SEMANTICA
    # Ahora buscamos por SIGNIFICADO, no por palabras exactas.
    # Nota: "como ganar dinero" NO aparece textualmente en ningun documento,
    # pero Pinecone encontrara documentos con significado similar.
    sep("PASO 4: Busqueda semantica")

    consultas = [
        "como ganar dinero a largo plazo",
        "quiero aprender a programar",
        "como mejorar mi bienestar fisico",
        "herramientas modernas de inteligencia artificial",
        "mejores formas de estudiar para examenes",
    ]

    for consulta in consultas:
        print(f"\n  Consulta: \"{consulta}\"")
        print(f"  {'-'*50}")

        # Convertir la consulta a embedding
        query_embedding = modelo.encode(consulta).tolist()

        # Buscar en Pinecone
        resultados = index.query(
            vector=query_embedding, top_k=2, include_metadata=True
        )

        for i, match in enumerate(resultados.matches, 1):
            print(f"  {i}. [{match.metadata['categoria']}] {match.metadata['texto']}")
            print(f"     Similitud: {match.score:.4f}")

    # PASO 5: BUSQUEDA CON FILTRO POR CATEGORIA
    sep("PASO 5: Busqueda filtrada por categoria")
    consulta = "como mejorar mi vida"
    query_emb = modelo.encode(consulta).tolist()

    categorias = ["salud", "finanzas", "tecnologia"]
    for cat in categorias:
        print(f"\n  Categoria: {cat}")
        res = index.query(
            vector=query_emb, top_k=1, include_metadata=True,
            filter={"categoria": {"$eq": cat}}
        )
        if res.matches:
            m = res.matches[0]
            print(f"  -> {m.metadata['texto']} (score: {m.score:.4f})")

    # LIMPIEZA
    sep("Limpieza")
    r = input("Eliminar indice? (s/n): ").strip().lower()
    if r == "s":
        pc.delete_index(INDEX_NAME)
        print("Indice eliminado")

    sep("EJERCICIO 2 COMPLETADO!")
    print("""
    Aprendiste:
    - Generar embeddings reales con sentence-transformers
    - Buscar por SIGNIFICADO, no por palabras clave
    - Filtrar resultados por categoria
    - La potencia de la busqueda semantica
    """)

if __name__ == "__main__":
    main()
