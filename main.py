# import streamlit as st
# import pandas as pd
# import json

# from src.ollama_client import OllamaService
# from src.vector_store import ChromaStore
# from src.rag_engine import RAGRecommendationEngine
# from src.narrative_builder import (
#     build_user_explicit_narrative,
#     build_user_implicit_narrative,
# )

# # 1. Configuración de página
# st.set_page_config(page_title="Demo RAG Turismo", layout="wide")
# st.title("🌍 Sistema de Recomendación Turística (RAG)")


# # 2. Caché para evitar recargar Chroma y modelos en cada interacción
# @st.cache_resource
# def load_system():
#     df_profiles = pd.read_csv(
#         "./data/kadir_data/usuarios_perfiles.csv", encoding="latin1"
#     )
#     df_interactions = pd.read_csv(
#         "./data/kadir_data/historial_interacciones.csv", encoding="latin1"
#     )

#     ollama_svc = OllamaService(embed_model="qwen3-embedding:8b", gen_model="qwen2.5:7b")
#     store = ChromaStore(ollama_svc=ollama_svc)

#     engine = RAGRecommendationEngine(
#         df_profiles=df_profiles,
#         df_interactions=df_interactions,
#         chroma_store=store,
#         ollama_svc=ollama_svc,
#     )
#     return engine, df_profiles, df_interactions


# engine, df_profiles, df_interactions = load_system()

# # 3. Panel Lateral: Selección de Usuario
# st.sidebar.header("Control de Usuario")
# # Formatear la lista para mostrar "ID - Nombre"
# lista_usuarios = df_profiles.apply(
#     lambda x: f"{x['id_usuario']} - {x['nombre']}", axis=1
# ).tolist()
# usuario_seleccionado = st.sidebar.selectbox("Selecciona un usuario:", lista_usuarios)

# # Extraer el ID (el primer valor antes del guion)
# target_id = usuario_seleccionado.split(" - ")[0].strip()

# # 4. Mostrar Perfil del Usuario
# st.subheader(f"👤 Perfil del Usuario: {usuario_seleccionado}")

# user_row = df_profiles[df_profiles["id_usuario"].astype(str).str.strip() == target_id]
# col1, col2 = st.columns(2)

# with col1:
#     st.markdown("**Contexto Explícito (Declarado):**")
#     st.info(build_user_explicit_narrative(user_row.iloc[0]))

# with col2:
#     st.markdown("**Contexto Implícito (Historial Observado):**")
#     historial_texto = build_user_implicit_narrative(target_id, df_interactions)
#     if historial_texto:
#         st.success(historial_texto)
#     else:
#         st.warning("Usuario Nuevo (Cold Start) - Sin historial previo.")

# st.divider()

# # 5. Generación de Recomendaciones
# if st.button("🚀 Generar Recomendaciones (RAG)", type="primary"):
#     with st.spinner(f"Consultando a qwen2.5:7b para el usuario {target_id}..."):
#         try:
#             # Ejecutar el motor
#             resultado = engine.recommend_for_user(
#                 user_id=target_id, top_k_retrieval=6, top_n_final=3
#             )

#             # Mostrar resultados
#             st.subheader("🎯 Recomendaciones Personalizadas")

#             recomendaciones = resultado.get("resultado_rag", {}).get(
#                 "recomendaciones", []
#             )

#             if not recomendaciones:
#                 st.error(
#                     "El LLM no devolvió el formato JSON esperado o no hay candidatos."
#                 )
#                 st.json(resultado)
#             else:
#                 for idx, rec in enumerate(recomendaciones, 1):
#                     with st.expander(
#                         f"{idx}. {rec.get('nombre_sitio', 'Desconocido')}",
#                         expanded=True,
#                     ):
#                         st.markdown(
#                             f"**Categoría:** {rec.get('categoria_sitio', 'N/A')} | **Ubicación:** {rec.get('ubicacion', 'N/A')}"
#                         )
#                         st.markdown(
#                             f"**Justificación:** {rec.get('justificacion', 'N/A')}"
#                         )

#             # Debugging (opcional, para ver qué recuperó ChromaDB)
#             with st.expander("🛠️ Ver metadatos (Retrieval vs Generation)"):
#                 st.markdown("**Contexto inyectado al LLM:**")
#                 st.code(resultado.get("contexto_usado"))
#                 st.markdown("**Candidatos recuperados de ChromaDB:**")
#                 st.json(resultado.get("candidatos_recuperados"))

#         except Exception as e:
#             st.error(f"Error durante la inferencia: {str(e)}")

import json
import pandas as pd
import streamlit as st

from src.narrative_builder import (
    build_user_explicit_narrative,
    build_user_implicit_narrative,
)
from src.ollama_client import DEFAULT_PROMPT_TEMPLATE, OllamaService
from src.rag_engine import RAGRecommendationEngine
from src.vector_store import ChromaStore

AVAILABLE_GEN_MODELS = [
    "qwen2.5:7b",
    "llama3.2:3b",
    "gemma2:2b",
    "qwen2.5-coder:7b-instruct-q4_K_M",
    "gemma4:e4b",
]

selected_model = st.sidebar.selectbox(
    "Modelo Generativo (LLM):",
    AVAILABLE_GEN_MODELS,
    index=0,
    help="Modelo encargado de filtrar candidatos, estructurar el JSON y redactar justificaciones.",
)

st.set_page_config(page_title="RAG RecSys Lab", layout="wide")
st.title("🔬 RAG RecSys: Laboratorio de Recomendación Turística")


@st.cache_resource
def load_system():
    df_profiles = pd.read_csv(
        "data/kadir_data/usuarios_perfiles.csv", encoding="latin1"
    )
    df_interactions = pd.read_csv(
        "data/kadir_data/historial_interacciones.csv", encoding="latin1"
    )

    # Limpieza básica de columnas
    df_profiles.columns = df_profiles.columns.str.strip()
    df_interactions.columns = df_interactions.columns.str.strip()

    ollama_svc = OllamaService(embed_model="qwen3-embedding:8b", gen_model="qwen2.5:7b")
    store = ChromaStore(ollama_svc=ollama_svc)

    engine = RAGRecommendationEngine(
        df_profiles=df_profiles,
        df_interactions=df_interactions,
        chroma_store=store,
        ollama_svc=ollama_svc,
    )
    return engine, df_profiles, df_interactions, store


engine, df_profiles, df_interactions, store = load_system()

# --- SIDEBAR: CONTROLES DE EXPERIMENTACIÓN ---
st.sidebar.header("⚙️ Parámetros de Recuperación y Generación")

top_k = st.sidebar.slider(
    "Top-K Candidatos (Retrieval ChromaDB)",
    min_value=3,
    max_value=30,
    value=10,
    help="Lugares más cercanos recuperados vectorialmente.",
)

top_n = st.sidebar.slider(
    "Top-N Salida (Generación LLM)",
    min_value=1,
    max_value=15,
    value=3,
    help="Lugares finales que el LLM filtrará, ordenará y justificará.",
)

if top_k < top_n:
    st.sidebar.warning("⚠️ Se recomienda que Top-K sea mayor o igual a Top-N.")

# st.sidebar.divider()
# st.sidebar.subheader("Selección de Sujeto de Prueba")
# lista_usuarios = df_profiles.apply(
#     lambda x: f"{x['id_usuario']} - {x.get('nombre', 'Sin Nombre')}", axis=1
# ).tolist()
# usuario_seleccionado = st.sidebar.selectbox("Usuario activo:", lista_usuarios)
# target_id = usuario_seleccionado.split(" - ")[0].strip()

# --- ESTRUCTURA POR PESTAÑAS ---
tab_rec, tab_inverse, tab_prompt, tab_data = st.tabs(
    ["🎯 Recomendar a Usuario", "📍 Recomendar a Sitio (Inverso)", "📝 Editor de Prompt", "🗄️ Visualizador de Datos"]
)

# --- TAB 1: RECOMENDADOR ---
with tab_rec:
    st.subheader("Selección de Sujeto de Prueba")
    lista_usuarios = df_profiles.apply(
        lambda x: f"{x['id_usuario']} - {x.get('nombre', 'Sin Nombre')}", axis=1
    ).tolist()
    usuario_seleccionado = st.selectbox("Usuario activo:", lista_usuarios)
    target_id = usuario_seleccionado.split(" - ")[0].strip()


    user_row = df_profiles[
        df_profiles["id_usuario"].astype(str).str.strip() == target_id
    ]

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("##### 👤 Perfil Explícito")
        st.info(build_user_explicit_narrative(user_row.iloc[0]))
    with col2:
        st.markdown("##### 🕒 Historial Implícito")
        historial_texto = build_user_implicit_narrative(target_id, df_interactions)
        if historial_texto:
            st.success(historial_texto)
        else:
            st.warning("Usuario en Cold Start (sin interacciones previas).")

    st.write("")
    if st.button("🚀 Ejecutar Pipeline RAG", type="primary"):
        active_prompt = st.session_state.get("custom_prompt", DEFAULT_PROMPT_TEMPLATE)

        with st.spinner(
            f"Retrieval de {top_k} candidatos y razonamiento de {top_n} con qwen2.5:7b..."
        ):
            try:
                res = engine.recommend_for_user(
                user_id=target_id,
                top_k_retrieval=top_k,
                top_n_final=top_n,
                prompt_template=active_prompt,
                model_name=selected_model,
            )

                # --- MÉTRICAS DE LATENCIA ---
                tiempos = res.get("tiempos", {})
                m1, m2, m3 = st.columns(3)
                m1.metric("⏱️ Tiempo Total", f"{tiempos.get('total_s', 0.0)} s")
                m2.metric(
                    "🔍 Retrieval (Embeddings)", f"{tiempos.get('retrieval_s', 0.0)} s"
                )
                m3.metric(
                    "🧠 Generación (Qwen 2.5)", f"{tiempos.get('generacion_s', 0.0)} s"
                )
                st.divider()

                # --- RESULTADOS ---
                st.subheader("Resultados de la Recomendación")
                recomendaciones = res.get("resultado_rag", {}).get(
                    "recomendaciones", []
                )

                if not recomendaciones:
                    st.error("No se generaron recomendaciones válidas.")
                    st.json(res)
                else:
                    for idx, rec in enumerate(recomendaciones, 1):
                        nombre_sitio = rec.get("nombre_sitio", "Lugar")
                        with st.expander(f"{idx}. {nombre_sitio}", expanded=True):
                            st.markdown(
                                f"**Categoría:** `{rec.get('categoria_sitio', 'N/A')}` | **Ubicación:** `{rec.get('ubicacion', 'N/A')}`"
                            )
                            st.markdown(
                                f"**Justificación:** {rec.get('justificacion', 'N/A')}"
                            )

                with st.expander("🔍 Inspeccionar Metadatos del Pipeline (Debug)"):
                    st.markdown("**1. Contexto inyectado en el prompt:**")
                    st.code(res.get("contexto_usado"))
                    st.markdown(
                        f"**2. Candidatos recuperados de ChromaDB ({len(res.get('candidatos_recuperados', []))} ítems):**"
                    )
                    st.dataframe(pd.DataFrame(res.get("candidatos_recuperados", [])))

            except Exception as e:
                st.error(f"Error en inferencia: {str(e)}")

# --- TAB 2: EDITOR DE PROMPT ON-THE-FLY ---
with tab_prompt:
    st.markdown("### Modificar Plantilla del Prompt en Caliente")
    st.caption(
        "Variables disponibles obligatorias: `{top_n}`, `{user_context}`, `{candidates_json}`."
    )

    if "custom_prompt" not in st.session_state:
        st.session_state["custom_prompt"] = DEFAULT_PROMPT_TEMPLATE

    prompt_input = st.text_area(
        "Plantilla del Prompt",
        value=st.session_state["custom_prompt"],
        height=400,
    )

    col_btn1, col_btn2 = st.columns([1, 5])
    with col_btn1:
        if st.button("Guardar Cambios"):
            st.session_state["custom_prompt"] = prompt_input
            st.success("Prompt actualizado para las siguientes ejecuciones.")
    with col_btn2:
        if st.button("Restablecer Prompt Base"):
            st.session_state["custom_prompt"] = DEFAULT_PROMPT_TEMPLATE
            st.rerun()

# --- TAB 3: VISUALIZADOR DE LA BASE DE DATOS ---
with tab_data:
    st.markdown("### Exploración de Datos")

    subtab_users, subtab_inter, subtab_chroma = st.tabs(
        [
            "Usuarios (Perfiles)",
            "Historial de Interacciones",
            "Catálogo en ChromaDB",
        ]
    )

    with subtab_users:
        st.markdown(f"**Total usuarios:** {len(df_profiles)}")
        st.dataframe(df_profiles, use_container_width=True)

    with subtab_inter:
        st.markdown(f"**Total interacciones:** {len(df_interactions)}")
        st.dataframe(df_interactions, use_container_width=True)

    with subtab_chroma:
        chroma_data = store.places_col.get(include=["metadatas", "documents"])
        total_chroma = len(chroma_data["ids"])
        st.markdown(f"**Lugares indexados vectorialmente:** {total_chroma}")

        if total_chroma > 0:
            df_chroma = pd.DataFrame(chroma_data["metadatas"])
            df_chroma["documento_indexado"] = chroma_data["documents"]
            st.dataframe(df_chroma, use_container_width=True)


# --- PESTAÑA INVERSA ---
with tab_inverse:
    st.markdown("### 🎯 Identificación de Audiencia Objetivo para un Sitio")
    st.caption("Selecciona un lugar del catálogo para encontrar qué usuarios sin interacción previa son los clientes ideales.")
    
    # Lista única de sitios disponibles
    catalogo_sitios = df_interactions.drop_duplicates(subset=["url_sitio"])[["url_sitio", "nombre_sitio", "categoria_sitio", "ubicacion"]]
    opciones_sitios = catalogo_sitios.apply(
        lambda x: f"{x['nombre_sitio']} ({x['categoria_sitio']} - {x['ubicacion']}) | ID: {x['url_sitio']}", axis=1
    ).tolist()
    
    sitio_seleccionado = st.selectbox("Seleccionar Sitio Turístico:", opciones_sitios)
    selected_url = sitio_seleccionado.split(" | ID: ")[-1].strip()
    
    if st.button("🔍 Buscar Audiencia Objetivo", type="primary"):
        with st.spinner("Buscando usuarios afines en ChromaDB y analizando perfiles con el LLM..."):
            try:
                res_inv = engine.recommend_users_for_place(
                    url_sitio=selected_url,
                    top_k_retrieval=top_k,
                    top_n_final=top_n,
                    model_name=selected_model
                )
                
                # Tiempos
                tiempos = res_inv.get("tiempos", {})
                m1, m2, m3 = st.columns(3)
                m1.metric("⏱️ Tiempo Total", f"{tiempos.get('total_s', 0.0)} s")
                m2.metric("🔍 Retrieval Usuarios", f"{tiempos.get('retrieval_s', 0.0)} s")
                m3.metric(f"🧠 Generación ({selected_model})", f"{tiempos.get('generacion_s', 0.0)} s")
                st.divider()
                
                users_rec = res_inv.get("resultado_rag", {}).get("usuarios_recomendados", [])
                
                if not users_rec:
                    st.warning("No se obtuvieron recomendaciones de usuarios.")
                    st.json(res_inv)
                else:
                    for idx, u in enumerate(users_rec, 1):
                        with st.expander(f"{idx}. {u.get('nombre', 'Usuario')} (ID: {u.get('id_usuario', 'N/A')}) — {u.get('tipo_viajero', 'Viajero')}", expanded=True):
                            st.markdown(f"**Justificación de Segmento:** {u.get('justificacion_target', 'N/A')}")
                            
                with st.expander("🔍 Usuarios recuperados de ChromaDB (Candidatos)"):
                    st.dataframe(pd.DataFrame(res_inv.get("usuarios_recuperados", [])))
                    
            except Exception as e:
                st.error(f"Error en recomendación inversa: {str(e)}")