import time
import pandas as pd
from src.narrative_builder import (
    build_user_explicit_narrative,
    build_user_implicit_narrative,
)
from src.ollama_client import OllamaService
from src.vector_store import ChromaStore


class RAGRecommendationEngine:

    def __init__(
        self,
        df_profiles: pd.DataFrame,
        df_interactions: pd.DataFrame,
        chroma_store: ChromaStore,
        ollama_svc: OllamaService,
    ):
        self.profiles = df_profiles.copy()
        self.interactions = df_interactions.copy()
        self.store = chroma_store
        self.ollama = ollama_svc

        self.profiles.columns = self.profiles.columns.str.strip()
        self.interactions.columns = self.interactions.columns.str.strip()

        self.profiles["id_usuario"] = (
            self.profiles["id_usuario"].astype(str).str.strip()
        )
        self.interactions["id_usuario"] = (
            self.interactions["id_usuario"].astype(str).str.strip()
        )

    def recommend_for_user(
        self,
        user_id: str | int,
        top_k_retrieval: int = 6,
        top_n_final: int = 3,
        prompt_template: str | None = None,
    ) -> dict:
        t_total_start = time.perf_counter()
        target_id_str = str(user_id).strip()

        user_row = self.profiles[self.profiles["id_usuario"] == target_id_str]
        if user_row.empty:
            raise ValueError(
                f"Usuario '{target_id_str}' no encontrado en usuarios_perfiles.csv"
            )

        # 1. Construir contexto
        explicit_text = build_user_explicit_narrative(user_row.iloc[0])
        implicit_text = build_user_implicit_narrative(target_id_str, self.interactions)

        consumed_urls = set(
            self.interactions[self.interactions["id_usuario"] == target_id_str][
                "url_sitio"
            ]
            .dropna()
            .astype(str)
        )

        if implicit_text:
            combined_context = (
                f"{explicit_text}\n[COMPORTAMIENTO RECIENTE] {implicit_text}"
            )
        else:
            combined_context = f"[COLD START - PERFIL DECLARADO] {explicit_text}"

        # 2. Retrieval y medición
        t_retrieval_start = time.perf_counter()
        candidates = self.store.retrieve_candidates(
            query_text=combined_context,
            n_results=top_k_retrieval,
            exclude_urls=consumed_urls,
        )
        latencia_retrieval = round(time.perf_counter() - t_retrieval_start, 3)

        if not candidates:
            return {
                "user_id": target_id_str,
                "mensaje": "No se encontraron nuevos candidatos disponibles.",
                "tiempos": {
                    "retrieval_s": latencia_retrieval,
                    "generacion_s": 0.0,
                    "total_s": round(time.perf_counter() - t_total_start, 3),
                },
            }

        # 3. Inferencia LLM y medición
        t_gen_start = time.perf_counter()
        recommendations = self.ollama.generate_recommendations(
            user_context=combined_context,
            candidates=candidates,
            top_n=top_n_final,
            prompt_template=prompt_template,
        )
        latencia_generacion = round(time.perf_counter() - t_gen_start, 3)
        latencia_total = round(time.perf_counter() - t_total_start, 3)

        return {
            "user_id": target_id_str,
            "contexto_usado": combined_context,
            "candidatos_recuperados": candidates,
            "resultado_rag": recommendations,
            "tiempos": {
                "retrieval_s": latencia_retrieval,
                "generacion_s": latencia_generacion,
                "total_s": latencia_total,
            },
        }


if __name__ == "__main__":
    print("Hello World!")
