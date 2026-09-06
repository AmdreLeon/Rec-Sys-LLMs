import json
from numpy import tan
import pandas as pd
from src.narrative_builder import (
    build_user_explicit_narrative,
    build_user_implicit_narrative,
)
from src.ollama_client import OllamaService
from src.rag_engine import RAGRecommendationEngine
from src.vector_store import ChromaStore


def main():
    user_test_id = 3
    print("1. Cargando datos CSV...")
    df_profiles = pd.read_csv(
        "data/kadir_data/usuarios_perfiles.csv", encoding="latin-1"
    )
    print()
    print(df_profiles.iloc[user_test_id])
    print()
    df_interactions = pd.read_csv(
        "data/kadir_data/historial_interacciones.csv", encoding="latin-1"
    )

    print("2. Inicializando servicios locales de Ollama y ChromaDB...")
    ollama_svc = OllamaService(embed_model="qwen3-embedding:8b", gen_model="qwen2.5:7b")
    store = ChromaStore(ollama_svc=ollama_svc)

    print("3. Indexando catálogo de lugares...")
    store.ingest_places_from_df(df_interactions)

    engine = RAGRecommendationEngine(
        df_profiles=df_profiles,
        df_interactions=df_interactions,
        chroma_store=store,
        ollama_svc=ollama_svc,
    )

    # Probar con el primer usuario del dataset
    target_user_id = str(df_profiles.iloc[user_test_id]["id_usuario"])
    # print(target_user_id, type(target_user_id))

    print(f"\n4. Generando recomendación RAG para usuario: {target_user_id}...")

    result = engine.recommend_for_user(
        user_id=target_user_id, top_k_retrieval=20, top_n_final=10
    )

    print("\n" + "=" * 50)
    print("RESULTADO FINAL (JSON LLM):")
    print("=" * 50)
    print(json.dumps(result["resultado_rag"], indent=2, ensure_ascii=False))


def list_available_users(df_profiles: pd.DataFrame, n_preview: int = 10):
    """Muestra información sobre los IDs y nombres de los usuarios disponibles."""
    print("\n" + "=" * 60)
    print("RESUMEN DE USUARIOS EN EL DATASET")
    print("=" * 60)
    print(f"Total de usuarios encontrados: {len(df_profiles)}")
    print(f"Tipo de dato de la columna 'id_usuario': {df_profiles['id_usuario'].dtype}")

    # Columnas relevantes para visualizar
    preview_cols = [
        col
        for col in ["id_usuario", "nombre", "edad", "nacionalidad", "tipo_viajero"]
        if col in df_profiles.columns
    ]

    print(f"\nPrimeros {min(n_preview, len(df_profiles))} usuarios:")
    print(df_profiles[preview_cols].head(n_preview).to_string(index=False))
    print("=" * 60 + "\n")


if __name__ == "__main__":
    # list_available_users()
    main()
