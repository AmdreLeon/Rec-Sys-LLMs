import pandas as pd
from tqdm import tqdm

from src.narrative_builder import build_user_full_narrative
from src.ollama_client import OllamaService
from src.vector_store import ChromaStore


def seed_database():
    print("1. Leyendo archivos CSV...")
    df_profiles = pd.read_csv(
        "data/kadir_data/usuarios_perfiles.csv", encoding="latin1"
    )
    df_interactions = pd.read_csv(
        "data/kadir_data/historial_interacciones.csv", encoding="latin1"
    )

    df_profiles.columns = df_profiles.columns.str.strip()
    df_interactions.columns = df_interactions.columns.str.strip()
    df_profiles["id_usuario"] = (
        df_profiles["id_usuario"].astype(str).str.strip()
    )
    df_interactions["id_usuario"] = (
        df_interactions["id_usuario"].astype(str).str.strip()
    )

    print("2. Inicializando conexión con Ollama y ChromaDB...")
    ollama_svc = OllamaService(
        embed_model="qwen3-embedding:8b", gen_model="qwen2.5:7b"
    )
    store = ChromaStore(ollama_svc=ollama_svc)
    users_col = store.get_or_create_users_collection()

    count_actual = users_col.count()
    if count_actual > 0:
        print(
            f"ℹ️ La colección 'perfiles_usuarios' ya contiene {count_actual} usuarios indexados."
        )
        reset = (
            input("¿Quieres vaciarla y reindexar desde cero? (s/N): ")
            .strip()
            .lower()
        )
        if reset == "s":
            store.client.delete_collection("perfiles_usuarios")
            users_col = store.get_or_create_users_collection()
            print("Colección reiniciada.")
        else:
            print("Operación cancelada. Usando datos existentes.")
            return

    print(
        f"3. Generando embeddings con qwen3-embedding:8b para {len(df_profiles)} usuarios..."
    )

    ids, docs, embeddings, metadatas = [], [], [], []

    for _, row in tqdm(
        df_profiles.iterrows(), total=len(df_profiles), desc="Vectorizando"
    ):
        u_id = str(row["id_usuario"]).strip()
        narrativa = build_user_full_narrative(u_id, row, df_interactions)

        if not narrativa.strip():
            continue

        emb = ollama_svc.get_embedding(narrativa)

        ids.append(u_id)
        docs.append(narrativa)
        embeddings.append(emb)
        metadatas.append(
            {
                "id_usuario": u_id,
                "nombre": str(row.get("nombre", "Sin Nombre")),
                "nacionalidad": str(row.get("nacionalidad", "N/A")),
                "tipo_viajero": str(row.get("tipo_viajero", "N/A")),
                "presupuesto_viaje": str(row.get("presupuesto_viaje", "N/A")),
            }
        )

    # Inserción en lotes en ChromaDB
    print("4. Guardando en ChromaDB...")
    users_col.upsert(
        ids=ids,
        documents=docs,
        embeddings=embeddings,
        metadatas=metadatas,
    )
    print(
        f"✅ Completado: {users_col.count()} usuarios indexados exitosamente."
    )


if __name__ == "__main__":
    seed_database()