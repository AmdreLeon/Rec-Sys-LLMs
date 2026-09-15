# import json
# import ollama


# class OllamaService:

#     def __init__(
#         self,
#         embed_model: str = "qwen3-embedding:8b",
#         gen_model: str = "qwen2.5:7b",
#     ):
#         self.embed_model = embed_model
#         self.gen_model = gen_model

#     def get_embedding(self, text: str) -> list[float]:
#         """Obtiene el vector usando qwen3-embedding:8b."""
#         res = ollama.embeddings(model=self.embed_model, prompt=text)
#         return res["embedding"]

#     def generate_recommendations(
#         self, user_context: str, candidates: list[dict], top_n: int = 10
#     ) -> dict:
#         """Pasa el contexto de usuario y candidatos recuperados al LLM para rankeo y justificación."""
#         prompt = f"""Eres un sistema de recomendación turística altamente preciso.
# Analiza el perfil y necesidades del usuario frente a los siguientes lugares candidatos recuperados.
# Selecciona los mejores {top_n} lugares y redacta una justificación concisa, personalizada y convincente para cada uno 

# [CONTEXTO DEL USUARIO]
# {user_context}

# [LUGARES CANDIDATOS RECUPERADOS]
# {json.dumps(candidates, ensure_ascii=False, indent=2)}

# Instrucción de salida: Devuelve ÚNICAMENTE un objeto JSON válido con este formato:
# {{
#   "recomendaciones": [
#     {{
#       "nombre_sitio": "Nombre del sitio",
#       "categoria_sitio": "Categoría",
#       "ubicacion": "Ubicación",
#       "justificacion": "Por qué este lugar encaja exactamente con su perfil/historial"
#     }}
#   ]
# }}"""

#         response = ollama.generate(
#             model=self.gen_model,
#             prompt=prompt,
#             format="json",
#             options={"temperature": 0.1},
#         )
#         return json.loads(response["response"])


# if __name__ == "__main__":
#     # Ejemplo de uso
#     print("Hello World")
#     # service = OllamaService()

#     # user_context = "El usuario es un amante de la naturaleza y busca experiencias al aire libre."
#     # candidates = [
#     #     {"nombre_sitio": "Parque Nacional A", "categoria_sitio": "Parque", "ubicacion": "Ciudad X"},
#     #     {"nombre_sitio": "Museo B", "categoria_sitio": "Museo", "ubicacion": "Ciudad Y"},
#     #     {"nombre_sitio": "Playa C", "categoria_sitio": "Playa", "ubicacion": "Ciudad Z"},
#     # ]

#     # recommendations = service.generate_recommendations(user_context, candidates, top_n=2)
#     # print(json.dumps(recommendations, ensure_ascii=False, indent=2))

import json
import ollama

DEFAULT_PROMPT_TEMPLATE = """Eres un recomendador turístico experto, riguroso y objetivo.
Analiza el perfil y contexto del usuario frente a los lugares candidatos recuperados.

REGLAS ESTRICTAS:
1. Basar tus justificaciones ÚNICAMENTE en los intereses, presupuesto, compañía y actividades del perfil del usuario.
2. NO inventes preferencias geográficas (países/ciudades) no declaradas. Si recomiendas un lugar en otra región, justifícalo por el TIPO de actividad o categoría, reconociendo con transparencia que es un nuevo destino.
3. Selecciona los mejores {top_n} lugares y asigna un nivel de afinidad cualitativo ("Muy Alta", "Alta", "Exploratoria").

[CONTEXTO DEL USUARIO]
{user_context}

[LUGARES CANDIDATOS RECUPERADOS]
{candidates_json}

Devuelve ÚNICAMENTE un JSON con esta estructura exacta:
{{
  "recomendaciones": [
    {{
      "nombre_sitio": "Nombre del sitio",
      "categoria_sitio": "Categoría",
      "ubicacion": "Ciudad, País",
      "nivel_afinidad": "Muy Alta" | "Alta" | "Exploratoria",
      "justificacion": "Explicación ceñida a los datos reales"
    }}
  ]
}}"""


class OllamaService:

    def __init__(
        self,
        embed_model: str = "qwen3-embedding:8b",
        gen_model: str = "qwen2.5:7b",
    ):
        self.embed_model = embed_model
        self.gen_model = gen_model

    def get_embedding(self, text: str) -> list[float]:
        res = ollama.embeddings(model=self.embed_model, prompt=text)
        return res["embedding"]

    def generate_recommendations(
        self,
        user_context: str,
        candidates: list[dict],
        top_n: int = 3,
        prompt_template: str | None = None,
        model_override: str | None = None,
    ) -> dict:
        template = (
            prompt_template if prompt_template else DEFAULT_PROMPT_TEMPLATE
        )
        active_gen_model = model_override if model_override else self.gen_model

        clean_candidates = [
            {
                "nombre_sitio": c.get("nombre_sitio", ""),
                "categoria_sitio": c.get("categoria_sitio", ""),
                "ubicacion": c.get("ubicacion", ""),
            }
            for c in candidates
        ]

        formatted_prompt = template.format(
            top_n=top_n,
            user_context=user_context,
            candidates_json=json.dumps(
                clean_candidates, ensure_ascii=False, indent=2
            ),
        )

        response = ollama.generate(
            model=active_gen_model,
            prompt=formatted_prompt,
            format="json",
            options={"temperature": 0.1},
        )
        return json.loads(response["response"])