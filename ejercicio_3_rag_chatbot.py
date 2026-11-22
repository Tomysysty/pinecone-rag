"""
EJERCICIO 3: RAG (Retrieval-Augmented Generation) CON PINECONE
===============================================================
Objetivo: Construir un chatbot que responde preguntas usando TUS
propios documentos como fuente de conocimiento.

Que es RAG?
  1. El usuario hace una pregunta
  2. Se busca en Pinecone los documentos mas relevantes
  3. Se pasan esos documentos como contexto a un LLM (OpenAI)
  4. El LLM genera una respuesta basada en TU informacion

Que aprenderas:
 - Dividir documentos largos en chunks
 - Crear un pipeline RAG completo
 - Combinar Pinecone + OpenAI para responder preguntas
 - Manejar namespaces para multiples fuentes

INSTRUCCIONES:
  1. pip install pinecone sentence-transformers openai
  2. Reemplaza las API Keys
  3. python ejercicio_3_rag_chatbot.py
"""

import os
import time
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

# Intentar importar OpenAI; si no esta, usamos modo simulado
try:
    from openai import OpenAI
    OPENAI_DISPONIBLE = True
except ImportError:
    OPENAI_DISPONIBLE = False

# =====================================================================
# CONFIGURACION
# =====================================================================
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "TU_OPENAI_API_KEY")  # Opcional: si no tienes, funciona en modo simulado
INDEX_NAME = "rag-chatbot"

def sep(titulo):
    print(f"\n{'='*60}\n  {titulo}\n{'='*60}")

# =====================================================================
# BASE DE CONOCIMIENTO: Manual de una empresa ficticia
# En un caso real, estos serian tus PDFs, wikis, documentos, etc.
# =====================================================================
MANUAL_EMPRESA = [
    {
        "seccion": "Politica de Vacaciones",
        "contenido": "Los empleados tienen derecho a 15 dias habiles de vacaciones al ano. "
        "Las vacaciones se deben solicitar con al menos 2 semanas de anticipacion "
        "a traves del portal de recursos humanos. No se pueden acumular mas de 5 dias "
        "para el siguiente ano. Los dias no utilizados se pierden el 31 de diciembre."
    },
    {
        "seccion": "Trabajo Remoto",
        "contenido": "La empresa permite trabajo remoto hasta 3 dias por semana. "
        "Los martes y jueves son dias obligatorios en oficina. Para trabajar "
        "remoto se necesita una conexion estable de internet y un espacio adecuado. "
        "El equipo de TI proporciona laptop y monitor adicional para casa."
    },
    {
        "seccion": "Beneficios de Salud",
        "contenido": "Todos los empleados de tiempo completo tienen seguro medico mayor "
        "que cubre al empleado y dependientes directos (conyuge e hijos). "
        "El seguro incluye consultas, hospitalizacion, medicamentos y dental. "
        "La empresa cubre el 80% de la prima y el empleado el 20%."
    },
    {
        "seccion": "Horario Laboral",
        "contenido": "El horario es flexible con una ventana de 7am a 7pm. "
        "Se deben cumplir 8 horas diarias y 40 semanales. "
        "El horario core (donde todos deben estar disponibles) es de 10am a 3pm. "
        "Las horas extra requieren aprobacion del supervisor directo."
    },
    {
        "seccion": "Evaluacion de Desempeno",
        "contenido": "Las evaluaciones se realizan semestralmente en junio y diciembre. "
        "Se evaluan 5 competencias: productividad, trabajo en equipo, comunicacion, "
        "liderazgo e innovacion. Los resultados se discuten en una reunion 1-a-1 "
        "con el supervisor. Los bonos anuales dependen de la calificacion obtenida."
    },
    {
        "seccion": "Capacitacion",
        "contenido": "Cada empleado tiene un presupuesto anual de $2000 USD para "
        "capacitacion profesional. Esto incluye cursos, conferencias, certificaciones "
        "y libros. Las solicitudes se aprueban a traves del portal de RRHH. "
        "La empresa tambien ofrece cursos internos mensuales gratuitos."
    },
    {
        "seccion": "Codigo de Vestimenta",
        "contenido": "El codigo de vestimenta es business casual de lunes a jueves. "
        "Los viernes se permite ropa casual. En reuniones con clientes se requiere "
        "vestimenta formal. No se permiten shorts, sandalias ni ropa deportiva "
        "en la oficina entre lunes y jueves."
    },
    {
        "seccion": "Proceso de Contratacion",
        "contenido": "El proceso consta de 4 etapas: revision de CV, entrevista telefonica, "
        "prueba tecnica y entrevista final con el equipo. El proceso dura entre "
        "2 y 4 semanas. Se ofrece retroalimentacion a todos los candidatos. "
        "Los referidos de empleados actuales tienen prioridad."
    },
]


def dividir_en_chunks(texto, tamano=200, overlap=50):
    """
    Divide un texto largo en pedazos (chunks) mas pequenos.
    Overlap: cuantos caracteres se comparten entre chunks consecutivos.
    Esto es CLAVE en RAG para no perder contexto en los cortes.
    """
    palabras = texto.split()
    chunks = []
    inicio = 0
    while inicio < len(palabras):
        fin = inicio + tamano
        chunk = " ".join(palabras[inicio:fin])
        chunks.append(chunk)
        inicio += tamano - overlap
    return chunks


def generar_respuesta_openai(pregunta, contexto, client):
    """Genera respuesta usando OpenAI GPT con el contexto recuperado."""
    prompt = f"""Eres un asistente de Recursos Humanos. Responde la pregunta
usando UNICAMENTE la informacion del contexto proporcionado.
Si la informacion no esta en el contexto, di que no tienes esa informacion.

CONTEXTO:
{contexto}

PREGUNTA: {pregunta}

RESPUESTA:"""

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=300
    )
    return response.choices[0].message.content


def generar_respuesta_simulada(pregunta, contexto):
    """Modo simulado: muestra el contexto recuperado sin llamar a OpenAI."""
    print("\n  [MODO SIMULADO - Sin OpenAI]")
    print(f"  El LLM recibiria este contexto para responder:\n")
    print(f"  Contexto recuperado:")
    print(f"  {'-'*50}")
    for i, c in enumerate(contexto.split('\n---\n'), 1):
        print(f"  Fragmento {i}: {c[:120]}...")
    print(f"\n  Con esto, el LLM generaria una respuesta coherente")
    print(f"  a la pregunta: \"{pregunta}\"")


def main():
    # PASO 1: CARGAR MODELO
    sep("PASO 1: Cargando modelo de embeddings")
    modelo = SentenceTransformer("all-MiniLM-L6-v2")
    dimension = 384
    print(f"Modelo cargado (dim={dimension})")

    # PASO 2: CONECTAR A PINECONE
    sep("PASO 2: Conectando a Pinecone")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    existentes = [idx.name for idx in pc.list_indexes()]
    if INDEX_NAME not in existentes:
        pc.create_index(
            name=INDEX_NAME, dimension=dimension, metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        print("Indice creado. Esperando...")
        time.sleep(10)
    else:
        print("Indice ya existe")
    index = pc.Index(INDEX_NAME)

    # PASO 3: INDEXAR EL MANUAL
    # Dividimos cada seccion en chunks y la indexamos
    sep("PASO 3: Indexando manual de la empresa")
    vectores = []
    chunk_id = 0
    for doc in MANUAL_EMPRESA:
        chunks = dividir_en_chunks(doc["contenido"], tamano=50, overlap=10)
        for chunk in chunks:
            embedding = modelo.encode(chunk).tolist()
            vectores.append({
                "id": f"chunk-{chunk_id}",
                "values": embedding,
                "metadata": {
                    "texto": chunk,
                    "seccion": doc["seccion"]
                }
            })
            chunk_id += 1

    # Upsert en batches (buena practica para muchos vectores)
    batch_size = 50
    for i in range(0, len(vectores), batch_size):
        batch = vectores[i:i+batch_size]
        index.upsert(vectors=batch)

    print(f"{len(vectores)} chunks indexados de {len(MANUAL_EMPRESA)} secciones")
    time.sleep(3)

    # Configurar OpenAI si esta disponible
    openai_client = None
    if OPENAI_DISPONIBLE and OPENAI_API_KEY != "TU_OPENAI_API_KEY":
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
        print("OpenAI conectado - Modo completo RAG")
    else:
        print("Modo simulado (sin OpenAI) - Veras el contexto recuperado")

    # PASO 4: PIPELINE RAG - Preguntas y Respuestas
    sep("PASO 4: Chatbot RAG en accion")

    preguntas = [
        "Cuantos dias de vacaciones tengo?",
        "Puedo trabajar desde casa?",
        "Que cubre el seguro medico?",
        "Como es el proceso para ser contratado?",
        "Cuanto presupuesto tengo para cursos?",
    ]

    for pregunta in preguntas:
        print(f"\n  PREGUNTA: {pregunta}")
        print(f"  {'='*50}")

        # PASO RAG 1: Convertir pregunta a embedding
        query_emb = modelo.encode(pregunta).tolist()

        # PASO RAG 2: Buscar contexto relevante en Pinecone
        resultados = index.query(
            vector=query_emb, top_k=3, include_metadata=True
        )

        # PASO RAG 3: Armar contexto con los documentos encontrados
        contexto = "\n---\n".join([
            f"[{m.metadata['seccion']}] {m.metadata['texto']}"
            for m in resultados.matches
        ])

        # Mostrar fuentes encontradas
        print(f"\n  Fuentes encontradas:")
        for m in resultados.matches:
            print(f"    - {m.metadata['seccion']} (score: {m.score:.4f})")

        # PASO RAG 4: Generar respuesta
        if openai_client:
            respuesta = generar_respuesta_openai(pregunta, contexto, openai_client)
            print(f"\n  RESPUESTA: {respuesta}")
        else:
            generar_respuesta_simulada(pregunta, contexto)

    # PASO 5: MODO INTERACTIVO
    sep("PASO 5: Modo interactivo")
    print("  Escribe tus preguntas sobre el manual de la empresa.")
    print("  Escribe 'salir' para terminar.\n")

    while True:
        pregunta = input("  Tu pregunta: ").strip()
        if pregunta.lower() in ("salir", "exit", "q"):
            break
        if not pregunta:
            continue

        query_emb = modelo.encode(pregunta).tolist()
        resultados = index.query(
            vector=query_emb, top_k=3, include_metadata=True
        )
        contexto = "\n---\n".join([
            f"[{m.metadata['seccion']}] {m.metadata['texto']}"
            for m in resultados.matches
        ])

        print(f"\n  Fuentes:")
        for m in resultados.matches:
            print(f"    - {m.metadata['seccion']} (score: {m.score:.4f})")

        if openai_client:
            respuesta = generar_respuesta_openai(pregunta, contexto, openai_client)
            print(f"\n  RESPUESTA: {respuesta}\n")
        else:
            generar_respuesta_simulada(pregunta, contexto)
            print()

    # LIMPIEZA
    sep("Limpieza")
    r = input("Eliminar indice? (s/n): ").strip().lower()
    if r == "s":
        pc.delete_index(INDEX_NAME)
        print("Indice eliminado")

    sep("EJERCICIO 3 COMPLETADO!")
    print("""
    Aprendiste:
    - Dividir documentos en chunks con overlap
    - Construir un pipeline RAG completo
    - Buscar contexto relevante con Pinecone
    - Pasar contexto a un LLM para generar respuestas
    - Crear un chatbot interactivo con tus propios datos

    Este es el patron que usan ChatGPT + docs, Notion AI,
    y la mayoria de asistentes empresariales modernos.
    """)

if __name__ == "__main__":
    main()
