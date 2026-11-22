"""
EJERCICIO 1: CRUD BASICO CON PINECONE
======================================
Objetivo: Aprender las operaciones fundamentales de Pinecone
(Create, Read, Update, Delete) con vectores.

Que aprenderas:
 - Conectarte a Pinecone y crear un indice
 - Insertar (upsert) vectores con metadatos
 - Consultar vectores por similitud
 - Actualizar y eliminar vectores

INSTRUCCIONES:
  1. pip install pinecone
  2. Obten tu API Key en https://app.pinecone.io/
  3. Guarda tu clave en la variable de entorno PINECONE_API_KEY
  4. python ejercicio_1_crud_basico.py
"""

import os
import time
from pinecone import Pinecone, ServerlessSpec

API_KEY = os.environ.get("PINECONE_API_KEY", "")
INDEX_NAME = "ejercicio-crud"
DIMENSION = 8

def sep(titulo):
    print(f"\n{'='*55}\n  {titulo}\n{'='*55}")

def main():
    # PASO 1: CONECTAR
    sep("PASO 1: Conectando a Pinecone")
    pc = Pinecone(api_key=API_KEY)
    print("Conexion exitosa")

    # PASO 2: CREAR INDICE
    sep("PASO 2: Creando indice")
    existentes = [idx.name for idx in pc.list_indexes()]
    if INDEX_NAME not in existentes:
        pc.create_index(
            name=INDEX_NAME, dimension=DIMENSION, metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        print(f"Indice '{INDEX_NAME}' creado")
        time.sleep(10)
    else:
        print(f"Indice '{INDEX_NAME}' ya existe")

    index = pc.Index(INDEX_NAME)

    # PASO 3: INSERTAR VECTORES (UPSERT)
    # Upsert = Insert + Update. Si el ID existe, se actualiza.
    sep("PASO 3: Insertando vectores")
    peliculas = [
        {"id": "pel-1", "values": [0.9,0.1,0.0,0.8,0.2,0.1,0.7,0.3],
         "metadata": {"titulo":"Inception","genero":"scifi","year":2010,"rating":8.8}},
        {"id": "pel-2", "values": [0.1,0.9,0.8,0.0,0.3,0.7,0.2,0.1],
         "metadata": {"titulo":"Titanic","genero":"romance","year":1997,"rating":7.9}},
        {"id": "pel-3", "values": [0.8,0.2,0.1,0.9,0.1,0.0,0.8,0.4],
         "metadata": {"titulo":"Interstellar","genero":"scifi","year":2014,"rating":8.7}},
        {"id": "pel-4", "values": [0.2,0.8,0.9,0.1,0.4,0.8,0.1,0.0],
         "metadata": {"titulo":"The Notebook","genero":"romance","year":2004,"rating":7.8}},
        {"id": "pel-5", "values": [0.5,0.5,0.3,0.5,0.6,0.4,0.5,0.5],
         "metadata": {"titulo":"The Matrix","genero":"scifi","year":1999,"rating":8.7}},
    ]
    index.upsert(vectors=peliculas)
    print(f"{len(peliculas)} peliculas insertadas")
    time.sleep(3)

    # PASO 4: ESTADISTICAS
    sep("PASO 4: Estadisticas del indice")
    stats = index.describe_index_stats()
    print(f"Vectores totales: {stats.total_vector_count}")
    print(f"Dimension: {stats.dimension}")

    # PASO 5: BUSQUEDA POR SIMILITUD (QUERY)
    # Le pasas un vector y te devuelve los mas parecidos
    sep("PASO 5: Busqueda por similitud")
    query_vec = [0.85, 0.15, 0.05, 0.85, 0.15, 0.05, 0.75, 0.35]
    resultados = index.query(vector=query_vec, top_k=3, include_metadata=True)
    print("Top 3 peliculas mas similares:\n")
    for i, m in enumerate(resultados.matches, 1):
        md = m.metadata
        print(f"  {i}. {md['titulo']} | {md['genero']} | {md['year']} | Score: {m.score:.4f}")

    # PASO 6: FILTRO POR METADATOS
    # Combinas similitud + filtros. Muy poderoso.
    sep("PASO 6: Busqueda con filtro")
    res_filtrado = index.query(
        vector=query_vec, top_k=3, include_metadata=True,
        filter={"genero": {"$eq": "scifi"}, "rating": {"$gte": 8.5}}
    )
    print("Solo scifi con rating >= 8.5:\n")
    for i, m in enumerate(res_filtrado.matches, 1):
        print(f"  {i}. {m.metadata['titulo']} (Rating: {m.metadata['rating']})")

    # PASO 7: FETCH - obtener vector por ID
    sep("PASO 7: Fetch por ID")
    fetched = index.fetch(ids=["pel-3"])
    v = fetched.vectors["pel-3"]
    print(f"ID: {v.id}")
    print(f"Metadata: {v.metadata}")

    # PASO 8: UPDATE - actualizar metadatos
    sep("PASO 8: Actualizar metadatos")
    index.update(id="pel-3", set_metadata={"rating": 9.0, "director": "Nolan"})
    time.sleep(2)
    updated = index.fetch(ids=["pel-3"])
    print(f"Nuevos metadatos: {updated.vectors['pel-3'].metadata}")

    # PASO 9: DELETE - eliminar vectores
    sep("PASO 9: Eliminar vector")
    index.delete(ids=["pel-4"])
    time.sleep(2)
    stats2 = index.describe_index_stats()
    print(f"Vectores restantes: {stats2.total_vector_count}")

    # PASO 10: LIMPIEZA
    sep("PASO 10: Limpieza")
    r = input("Eliminar el indice completo? (s/n): ").strip().lower()
    if r == "s":
        pc.delete_index(INDEX_NAME)
        print("Indice eliminado")
    else:
        print("Indice conservado")

    sep("EJERCICIO 1 COMPLETADO!")
    print("""
    Aprendiste:
    - Conectarte a Pinecone
    - Crear un indice serverless
    - Upsert vectores con metadatos
    - Query por similitud
    - Filtrar con metadatos
    - Fetch por ID
    - Update metadatos
    - Delete vectores
    """)

if __name__ == "__main__":
    main()
