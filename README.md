# Rec-Sys-Test: Sistema de Recomendación Turístico Híbrido (RAG & GraphRAG)

## 1. Visión General del Proyecto
Este repositorio forma parte de una investigación doctoral orientada al desarrollo, optimización y benchmarking formal de sistemas de recomendación en el dominio turístico. El objetivo central es investigar cómo transicionar desde técnicas basadas en recuperación densa y modelos de lenguaje locales (*RAG*) hacia representaciones estructuradas sobre grafos de conocimiento enriquecidos (*GraphRAG*).

El sistema se enfoca en resolver dos retos clave:
* **Cold Start declarativo:** Mapeo de perfiles explícitos estáticos hacia representaciones vectoriales densas.
* **Personalización dinámica explicable:** Recomendación de ítems novedosos combinando filtrado colaborativo semántico, modelos de lenguaje para razonamiento/reranking y justificaciones causales fundamentadas.

---

## 2. Pila Tecnológica y Entorno
* **Lenguaje:** Python 3.11+
* **Gestor de Paquetes / Entorno:** `uv`
* **Base de Datos Vectorial:** ChromaDB (con métrica de similitud de coseno sobre espacio HNSW)
* **Inferencia Local:** Ollama / vLLM
* **Modelos Actuales:**
  * Embeddings: `qwen3-embedding:8b` (vectorización densa de usuarios, ítems y narrativas)
  * Generación y Reranking: `qwen2.5:7b` (baseline), `llama3.2:3b`, `gemma2:2b`, `qwen2.5-coder:7b`
* **UI de Experimentación:** Streamlit (`app.py`)

---

## 3. Estado Actual del Desarrollo (Lo que YA está implementado)

### 3.1. Arquitectura y Módulos
```text
Rec-Sys-Test/
├── data/
│   └── kadir_data/               # Datasets de prueba inicial (usuarios_perfiles.csv, historial_interacciones.csv)
├── src/
│   ├── narrative_builder.py      # Transforma datos tabulares/interacciones en narrativas textuales continuas
│   ├── vector_store.py           # Cliente ChromaDB (colecciones para catálogo de sitios y perfiles de usuarios)
│   ├── ollama_client.py          # Wrapper de inferencia con esquemas JSON estructurados y control de prompts
│   └── rag_engine.py             # Orquestador del pipeline RAG (Retrieval + Generación + Métricas de latencia)
├── seed_users.py                 # Script batch para ingesta e indexación vectorial de perfiles de usuario
├── app.py                        # Dashboard interactivo en Streamlit con desacoplamiento de parámetros y métricas
└── pyproject.toml / requirements # Dependencias gestionadas con uv

```

### 3.2. Funcionalidades Operativas

* **Pipeline RAG Directo (B2C):**
1. Genera la narrativa semántica del usuario (explícita o implícita agregada).
2. Recupera candidatos en ChromaDB con `qwen3-embedding:8b` excluyendo ítems ya consumidos (`consumed_urls`).
3. Pasa candidatos a un LLM local (`qwen2.5:7b` u otro) vía Ollama para filtrar al Top-$N$ y generar justificaciones explicativas causales en formato JSON estricto.


* **Pipeline Inverso / Segmentación de Audiencias (B2B):**
1. Recibe un recurso o sitio turístico y vectoriza sus atributos.
2. Consulta la colección `perfiles_usuarios` en ChromaDB para identificar usuarios afines que aún no han interactuado con el sitio.
3. El LLM justifica por qué cada usuario seleccionado encaja como público objetivo idóneo.


* **Instrumentación:**
* Medición precisa de latencias desacopladas con `time.perf_counter()` (tiempo de *retrieval* vs. tiempo de generación del LLM vs. tiempo total).


* **Interfaz de Control (`app.py`):**
* Pestañas independientes para recomendación directa, recomendación inversa, editor de prompts y visualizador de datos.
* Selector dinámico de modelo generativo y sliders de control para $K$ (candidatos recuperados) y $N$ (recomendaciones finales).



---

## 4. Reglas Críticas de Implementación para Asistentes de IA (Claude Code)

1. **Prioridad por Eficiencia y Tipado:** Usar Python idiomático, modular y tipado estricto (`type hints` completos con `typing` y notaciones modernas).
2. **Entornos y Comandos:** Todas las operaciones de entorno o instalación deben utilizar `uv` (`uv pip install`, `uv run ...`). No usar `pip` directamente ni instalar dependencias globales sin aislar.
3. **No romper contratos de datos:** Mantener el aislamiento entre componentes. Los módulos en `src/` no deben acoplarse a `Streamlit`; toda interacción con la UI debe limitarse a `app.py`.
4. **Formato JSON Estricto:** Cualquier interacción generativa con el LLM debe forzarse mediante `format="json"` en Ollama y contar con validación/sanitización defensiva del schema devuelto.
5. **Separación de Lógica Experimental:** La evaluación cuantitativa masiva debe residir en scripts de consola independientes, nunca dentro del bucle de renderizado de Streamlit.

---

## 5. Hoja de Ruta Experimental (Next Steps)

### Fase 1: Transición hacia Datasets de Referencia (A corto plazo)

* **Integración de Datasets Reales:**
* `TripAdvisor` / `Booking`: Sustituir datos de prueba por transacciones y reseñas reales de usuarios para sintetizar perfiles implícitos basados en NLP.
* `Wikivoyage`: Indexación del corpus geográfico/enciclopédico para dotar a los ítems turísticos de contexto externo denso.
* `SemEval-2014` (Task 4 - ABSA): Utilizar el dataset para validar y evaluar la extracción estructurada de aspectos (servicio, precio, ambiente) a partir del texto de las reviews.


* **Refactorización de Ingesta:** Adaptar `narrative_builder.py` para operar en escenarios sin datos explícitos (*implicit-only*), extrayendo vectores de usuario puramente a partir de sus comentarios históricos.

### Fase 2: Módulo Autónomo de Benchmarking y Evaluación Formal (`benchmark.py`)

* Implementar un script CLI independiente para evaluar el rendimiento cuantitativo offline:
* **Protocolo:** División temporal (*Temporal Train/Test Split*) y protocolo *Leave-One-Out*.
* **Métricas de Ranking (IR):** Hit Rate@$N$ (HR@$N$), Normalized Discounted Cumulative Gain (NDCG@$N$), Mean Reciprocal Rank (MRR), Precision@$N$ y Recall@$N$.
* **Métricas de Catálogo:** Catalog Coverage, Intra-List Diversity (ILD) mediante distancia coseno y Novedad.
* **Métricas RAG / NLG:** Fidelidad Factual (*Faithfulness* o ausencia de alucinación frente al contexto) y tiempo de respuesta / consumo de memoria por modelo.


* Comparar empíricamente:
* *Baseline 1:* Filtrado colaborativo / similitud puramente vectorial (Top-$K$ directo de ChromaDB sin LLM).
* *Baseline 2:* Pipeline RAG híbrido actual (Vectorial + Reranker generativo LLM).



### Fase 3: Evolución a GraphRAG (A mediano/largo plazo)

* **Modelado del Grafo de Conocimiento (KG):**
* Nodos: `Usuario`, `Sitio`, `Ciudad`, `Región`, `Categoria`, `Aspecto`.
* Aristas: `[:VISITO {rating, timestamp}]`, `[:UBICADO_EN]`, `[:PERTENECE_A]`, `[:OFRECE_ASPECTO {polaridad}]`.


* **Recuperación Híbrida (Vector + Grafo):**
* Localización de subgrafos locales por similitud semántica y expansión de caminos multired (*multi-hop paths*).
* Inyección de subgrafos y caminos causales en el prompt de contexto para que el LLM genere justificaciones con trazabilidad formal estricta.


