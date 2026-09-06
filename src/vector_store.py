import chromadb
import pandas as pd
from tqdm import tqdm
from src.ollama_client import OllamaService


class ChromaStore:

    def __init__(
        self, ollama_svc: OllamaService, persist_path: str = "./chroma_db"
    ):
        self.ollama = ollama_svc
        self.client = chromadb.PersistentClient(path=persist_path)
        self.places_col = self.client.get_or_create_collection(
            name="lugares_turisticos", metadata={"hnsw:space": "cosine"}
        )

    def ingest_places_from_df(
        self, df_interactions: pd.DataFrame, force_reindex: bool = False
    ):
        """Extrae el catálogo único de lugares e indexa sus embeddings con barra de progreso."""
        # Evitar reindexar si ya existen registros en ChromaDB
        existing_count = self.places_col.count()
        if existing_count > 0 and not force_reindex:
            print(
                f"ChromaDB ya contiene {existing_count} lugares indexados. Saltando ingesta."
            )
            return

        places_df = df_interactions.drop_duplicates(
            subset=["url_sitio"]
        ).dropna(subset=["url_sitio", "nombre_sitio"])

        total_places = len(places_df)
        print(f"Indexando {total_places} lugares únicos con Ollama...")

        ids, docs, embeddings, metadatas = [], [], [], []

        for _, row in tqdm(
            places_df.iterrows(), total=total_places, desc="Generando embeddings"
        ):
            place_id = str(row["url_sitio"])
            text_doc = (
                f"Lugar: {row.get('nombre_sitio', '')}. "
                f"Categoría: {row.get('categoria_sitio', '')}. "
                f"Ubicación: {row.get('ubicacion', '')}."
            )
            emb = self.ollama.get_embedding(text_doc)

            ids.append(place_id)
            docs.append(text_doc)
            embeddings.append(emb)
            metadatas.append(
                {
                    "nombre_sitio": str(row.get("nombre_sitio", "")),
                    "categoria_sitio": str(row.get("categoria_sitio", "")),
                    "ubicacion": str(row.get("ubicacion", "")),
                    "url_sitio": place_id,
                }
            )

        if ids:
            self.places_col.upsert(
                ids=ids,
                documents=docs,
                embeddings=embeddings,
                metadatas=metadatas,
            )
            print(f"\nIndexados exitosamente {len(ids)} lugares en ChromaDB.")

    def retrieve_candidates(
        self,
        query_text: str,
        n_results: int = 6,
        exclude_urls: set[str] | None = None,
    ) -> list[dict]:
        """Recupera candidatos por similitud semántica excluyendo ítems ya consumidos."""
        query_emb = self.ollama.get_embedding(query_text)
        results = self.places_col.query(
            query_embeddings=[query_emb],
            n_results=n_results + (len(exclude_urls) if exclude_urls else 0),
        )

        candidates = []
        if results and results["metadatas"]:
            for meta, dist in zip(results["metadatas"][0], results["distances"][0]):
                if exclude_urls and meta["url_sitio"] in exclude_urls:
                    continue
                c = dict(meta)
                c["similitud_vectorial"] = round(1.0 - dist, 4)
                candidates.append(c)
                if len(candidates) == n_results:
                    break

        return candidates


if __name__ == "__main__":
    print("Hello World!")
