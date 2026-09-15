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
        model_name: str | None = None,
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
            model_override=model_name,
        )
        latencia_generacion = round(time.perf_counter() - t_gen_start, 3)
        latencia_total = round(time.perf_counter() - t_total_start, 3)

        return {
            "user_id": target_id_str,
            "modelo_usado": model_name or self.ollama.gen_model,
            "contexto_usado": combined_context,
            "candidatos_recuperados": candidates,
            "resultado_rag": recommendations,
            "tiempos": {
                "retrieval_s": latencia_retrieval,
                "generacion_s": latencia_generacion,
                "total_s": latencia_total,
            },
        }

    def recommend_users_for_place(
        self,
        url_sitio: str,
        top_k_retrieval: int = 10,
        top_n_final: int = 3,
        model_name: str | None = None
    ) -> dict:
        t_total_start = time.perf_counter()
        
        # 1. Obtener datos del sitio desde el catálogo de interacciones
        place_matches = self.interactions[self.interactions["url_sitio"].astype(str).str.strip() == str(url_sitio).strip()]
        if place_matches.empty:
            raise ValueError(f"Sitio con url '{url_sitio}' no encontrado.")
            
        place_row = place_matches.iloc[0]
        place_context = (
            f"Nombre: {place_row.get('nombre_sitio', '')}. "
            f"Categoría: {place_row.get('categoria_sitio', '')}. "
            f"Ubicación: {place_row.get('ubicacion', '')}."
        )
        
        # Excluir usuarios que ya han interactuado con este sitio
        visited_user_ids = set(place_matches["id_usuario"].dropna().astype(str))
        
        # 2. Retrieval: Búsqueda vectorial de usuarios afines
        t_ret_start = time.perf_counter()
        user_candidates = self.store.retrieve_user_candidates(
            query_text=place_context,
            n_results=top_k_retrieval,
            exclude_user_ids=visited_user_ids
        )
        t_ret_end = time.perf_counter()
        
        if not user_candidates:
            return {
                "url_sitio": url_sitio,
                "mensaje": "No se encontraron usuarios potenciales nuevos.",
                "tiempos": {"retrieval_s": round(t_ret_end - t_ret_start, 3), "generacion_s": 0.0, "total_s": round(time.perf_counter() - t_total_start, 3)}
            }
            
        # 3. Inferencia Generativa: Ranking y justificación de target
        t_gen_start = time.perf_counter()
        result = self.ollama.generate_inverse_recommendations(
            place_context=place_context,
            user_candidates=user_candidates,
            top_n=top_n_final,
            model_override=model_name
        )
        t_gen_end = time.perf_counter()
        
        return {
            "url_sitio": url_sitio,
            "sitio_analizado": place_context,
            "usuarios_recuperados": user_candidates,
            "resultado_rag": result,
            "tiempos": {
                "retrieval_s": round(t_ret_end - t_ret_start, 3),
                "generacion_s": round(t_gen_end - t_gen_start, 3),
                "total_s": round(time.perf_counter() - t_total_start, 3)
            }
        }


if __name__ == "__main__":
    print("Hello World!")
