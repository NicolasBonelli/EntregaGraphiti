# TechNova Knowledge Graph Project

## Objetivo

Construir un **sistema de GraphRAG** que combine:

* **Graphiti** para ingestión y gestión de un *knowledge graph temporal*.
* **Neo4j Aura (Free)** como backend de grafo en la nube.
* **LangChain Agent** como interfaz conversacional que aprovecha el grafo para responder preguntas complejas.
* **Docling** para extracción de texto desde documentos PDF.
* **OpenAI API** como motor LLM para extracción de entidades/relaciones y generación de respuestas.

## Alcance

* Ingestar un documento narrativo (historia de la empresa ficticia *TechNova*) en episodios.
* Construir un **grafo bi-temporal** con entidades, hechos y relaciones que cambian a lo largo del tiempo.
* Implementar un **LangChain Agent con executor sencillo** que use Graphiti como capa de retrieval híbrido (semantic + BM25 + graph-based).
* Exponer consultas del tipo:

  * “¿Quién era el CEO en 2014?”
  * “¿Qué productos se descontinuaron antes de 2023?”
  * “¿Qué alianzas estratégicas firmó la empresa en Asia?”
  * “¿Qué hechos eran válidos en 2021 pero ya no lo son en 2025?”

## Tecnologías utilizadas

* **Python 3.10+**
* **Graphiti** (`graphiti-core`)
* **Neo4j Aura Free** (5.26+)
* **LangChain** + **LangGraph**
* **Docling** (para parsing de PDF)
* **Docker** (opcional para desarrollo local de Neo4j)
* **OpenAI API** (extracción de entidades/relaciones y generación de embeddings)

## Arquitectura de alto nivel

1. **Ingesta**: PDF → Docling → texto limpio.
2. **Procesamiento**: Texto → episodios → Graphiti (extracción entidades + hechos).
3. **Almacenamiento**: Nodos y relaciones en Neo4j (con atributos temporales).
4. **Consulta**: LangChain Agent → retrieval híbrido de Graphiti → contexto → LLM.
5. **Respuesta**: Generación en lenguaje natural con grounding en facts recuperados.

## Buenas prácticas aplicadas

* Uso de `.env` para credenciales (Neo4j, OpenAI).
* Episodios atómicos (máx. 1–2 hechos por episodio).
* Validación bi-temporal automática de Graphiti.
* Logging del pipeline y de queries.
* Separación clara entre **data pipeline** y **capa de agente conversacional**.
