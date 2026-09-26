# Informe Técnico de Estado: Sistema de Recomendación Basado en RAG Turístico

---

## 1. Estado Actual del Sistema 

El sistema implementado aborda la recomendación turística personalizada integrando dos dimensiones de información del usuario:

- **Perfil Explícito:** Datos estáticos y declarativos provistos durante el registro (identidad y preferencias directas).
    
- **Perfil Implícito:** Datos dinámicos derivados de las acciones observadas (historial de consumo, reservas y valoraciones).
    

En lugar de codificar variables tabulares en vectores dispersos mediante esquemas tradicionales, la solución se basa en la **construcción de narrativas en lenguaje natural**. Estas descripciones densas capturan el contexto semántico completo del usuario y de los sitios turísticos, permitiendo proyectar perfiles y catálogo en un mismo espacio vectorial mediante embeddings densos.

## 2. Estructura de los Datos de Entrada

El pipeline se alimenta directamente de dos fuentes estructuradas:

### 2.1. Perfil Explícito (`usuarios_perfiles.csv`)

Define la identidad declarada del usuario a través de 21 atributos demográficos, contextuales y de preferencia:

- **Identificadores y Demografía:** `id_usuario`, `nombre`, `edad`, `genero`, `nacionalidad`, `idioma_principal`, `nivel_estudios`, `ocupacion`, `estado_civil`, `tiene_hijos`.
- **Estilo y Parámetros de Viaje:** `presupuesto_viaje`, `tipo_viajero`, `duracion_estadia_promedio`, `epoca_preferida`, `compania_viaje`, `destino_frecuente`, `alojamiento_preferido`, `nivel_actividad`, `dispositivo_principal`.
- **Intereses y Actividades (multivalorados con separador `|`):** `intereses_principales`, `actividades_preferidas`.
### 2.2. Historial de Interacciones (`historial_interacciones.csv`)

Registra el comportamiento dinámico observable mediante 13 atributos por interacción:

- **Metadatos del Evento:** `id_interaccion`, `id_usuario`, `fecha_interaccion`, `hora_dia`, `dispositivo_usado`.
- **Entidad Turística Interactuada:** `url_sitio` (identificador único del ítem), `nombre_sitio`, `categoria_sitio`, `ubicacion`.
			- **Comportamiento y Retroalimentación:** `tipo_interaccion` (_Click, Comentario, Reserva, Compartir, Favorito, Calificación_), `calificacion` (escala 1 a 5), `comentario` (texto libre), `tiempo_visita_segundos`.

## 3. Estado Actual de la Implementación (Pipeline RAG)

El prototipo se ejecuta íntegramente en local combinando **Ollama**, **ChromaDB** y una interfaz interactiva en **Streamlit**.

### Arquitectura Técnica Implementada

```text
[usuarios_perfiles.csv] ───► Perfil Explícito ──┐
                                                 ├─► [Narrative Builder] ─► Contexto Usuario
[historial_interacciones.csv] ─► Perfil Implícito ─┘                                │
                                                                                   ▼
[Catálogo de Lugares] ──► qwen3-embedding:8b ──► ChromaDB (HNSW Cosine) ◄── [Retrieval: Top-K]
                                                                                   │
                                                         ┌─────────────────────────┘
                                                         ▼
                                            [Augmented Prompting]
                                                         │
                                                         ▼
                                                qwen2.5:7b (Ollama)
                                                         │
                                                         ▼
                                          Top-N Recomendaciones (JSON)
                                          (Selección + Justificación Causal)

```

* **Pipeline de Datos y Narrativas (`src/narrative_builder.py`):**
* **Perfil Explícito:** Transforma datos sociodemográficos y preferencias declaradas en descripciones densas en lenguaje natural. Permite resolver el escenario de arranque en frío (*Cold Start*).
* **Perfil Implícito:** Agrega el comportamiento histórico (reservas, favoritos, calificaciones $\ge 4.0$, categorías recurrentes) para modelar las preferencias reales observadas.
* **Catálogo de Ítems:** Convierte los lugares turísticos únicos en documentos semánticos contextualizados (nombre, categoría, ubicación).


* **Motor Vectorial e Indexación (`src/vector_store.py`):**
* Base de datos vectorial persistente mediante **ChromaDB**.
* Generación de embeddings  con **`qwen3-embedding:8b`**.
* Filtrado estricto post-búsqueda para excluir lugares previamente consumidos/visitados por el usuario.


* **Módulo Generativo y Razonamiento (`src/ollama_client.py`):**
* Inferencia generativa local con **`qwen2.5:7b`**.
* Restricción estricta de salida a formato JSON estructurado (`format="json"`).
* Generación de justificaciones explicables (*reasoning*) que vinculan explícitamente los atributos del ítem con el perfil del usuario.


* **Entorno de Experimentación y Sandbox (`app.py` en Streamlit):**
* Ajuste dinámico en caliente de hiperparámetros de recuperación ($K$) y generación ($N$).
* Modificación del *System Prompt* en tiempo de ejecución sin reinicios de servicio.
* Monitoreo y desglose de latencias en tiempo real (tiempo de retrieval, tiempo de generación y tiempo total).
* Inspección de datos crudos (perfiles, interacciones y estado de la colección en ChromaDB).


---

## 4. Problemas Encontrados y Soluciones Implementadas

Durante el desarrollo e integración de los componentes surgieron varios retos técnicos críticos:

### A. Degradación por sobrecarga de contexto (*Lost in the Middle*)

* **Problema:** Al elevar el parámetro $K$ de recuperación por encima de 20 candidatos, la calidad de las recomendaciones caía drásticamente. El modelo tendía a ignorar candidatos intermedios o a degradar su razonamiento.
* **Solución:** Se estableció la cota empírica $K \approx 2 \times N$ a $2.5 \times N$ (ej. $K=10$ para $N=3\text{--}5$), podando además metadatos irrelevantes del JSON de candidatos inyectado al prompt.

### B. Alucinación de preferencias geográficas por sesgo de catálogo

* **Problema:** Al contener el catálogo una alta concentración de destinos en España, el LLM justificaba sus elecciones asumiendo erróneamente que el usuario "deseaba viajar a España", aunque dicha preferencia no figurase en el perfil.
* **Solución:** Incorporación de directrices estrictas en el prompt prohibiendo atribuir intenciones geográficas no declaradas y forzando a justificar por categoría de actividad o estilo de viaje cuando el destino sea novedoso.

---

## 5. Planes a Corto Plazo (Fase de Consolidación RAG)


1. **Evaluación Cuantitativa y Cualitativa:**
* Definir un conjunto de prueba (*benchmark*).
* Medir precisión de recuperación (Recall@K, Hit Rate).

2. **Calibración y Temperatura de Inferencia:**
* Ajustar los parámetros de muestreo (`temperature`, `top_p`, `repeat_penalty`) de `qwen2.5:7b` para garantizar determinismo en el formato JSON minimizando alucinaciones.


1. **Optimización de Entrada y Contexto:**
- Probar ajustes en la longitud y síntesis de las narrativas generadas para equilibrar riqueza semántica frente a velocidad de inferencia en los embeddings.


---

## 6. Planes a Largo Plazo (Evolución hacia GraphRAG)

- **Modelado del Conocimiento Relacional:**
    
    - Representar las entidades y relaciones clave presentes en los datos (`Usuario`, `Sitio`, `Categoría`, `Ubicación`, `TipoViajero`) mediante un grafo de conocimiento estructurado.
        
- **Recuperación Aumentada por Grafo (Graph Retrieval):**
    
    - Explotar conexiones de orden superior y caminos relacionales que la búsqueda semántica plana no captura de forma directa (patrones cruzados entre tipos de interacción, valoraciones y afinidad contextual compartida).

