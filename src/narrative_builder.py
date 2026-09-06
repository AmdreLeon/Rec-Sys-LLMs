import pandas as pd


def build_user_explicit_narrative(row: pd.Series) -> str:
    """Construye la narrativa del perfil explícito (Cold Start / Demografía)."""
    intereses = str(row.get("intereses_principales", "")).replace("|", ", ").strip(", ")
    actividades = (
        str(row.get("actividades_preferidas", "")).replace("|", ", ").strip(", ")
    )

    return (
        f"Usuario {row.get('genero', '')} de {row.get('edad', '')} años, nacionalidad {row.get('nacionalidad', '')}, "
        f"idioma principal {row.get('idioma_principal', '')}, ocupación {row.get('ocupacion', '')}. "
        f"Tipo de viajero: {row.get('tipo_viajero', '')} con presupuesto {row.get('presupuesto_viaje', '')}. "
        f"Suele viajar en {row.get('compania_viaje', '')}, estadía promedio {row.get('duracion_estadia_promedio', '')} días, "
        f"en época {row.get('epoca_preferida', '')}. Destinos frecuentes: {row.get('destino_frecuente', '')}, "
        f"alojamiento preferido: {row.get('alojamiento_preferido', '')}, nivel de actividad: {row.get('nivel_actividad', '')}. "
        f"Intereses principales: {intereses}. Actividades preferidas: {actividades}."
    ).strip()


def build_place_narrative(row: pd.Series) -> str:
    """Construye la descripción semántica de un sitio turístico para el catálogo."""
    return (
        f"Lugar turístico: {row.get('nombre_sitio', '')}. "
        f"Categoría: {row.get('categoria_sitio', '')}. "
        f"Ubicación: {row.get('ubicacion', '')}."
    ).strip()


def build_user_implicit_narrative(
    user_id: str | int, df_interactions: pd.DataFrame
) -> str:
    target_id = str(user_id).strip()
    user_data = df_interactions[
        df_interactions["id_usuario"].astype(str).str.strip() == target_id
    ]
    if user_data.empty:
        return ""

    top_cats = (
        user_data["categoria_sitio"].dropna().value_counts().head(3).index.tolist()
    )
    cats_str = ", ".join(top_cats) if top_cats else "diversas"

    top_lugares = user_data["ubicacion"].dropna().value_counts().head(2).index.tolist()
    lugares_str = ", ".join(top_lugares) if top_lugares else "varias"

    reservas = user_data[
        user_data["tipo_interaccion"].astype(str).str.lower() == "reserva"
    ]["nombre_sitio"].tolist()
    favoritos = user_data[
        user_data["tipo_interaccion"].astype(str).str.lower() == "favorito"
    ]["nombre_sitio"].tolist()

    if "calificacion" in user_data.columns:
        top_rated = (
            user_data[pd.to_numeric(user_data["calificacion"], errors="coerce") >= 4.0][
                "nombre_sitio"
            ]
            .dropna()
            .tolist()
        )
    else:
        top_rated = []

    parts = [
        f"Historial del usuario: interactúa con categorías {cats_str} en {lugares_str}."
    ]
    if reservas:
        parts.append(f"Reservó en: {', '.join(reservas[:3])}.")
    if favoritos:
        parts.append(f"Marcó como favoritos: {', '.join(favoritos[:3])}.")
    if top_rated:
        parts.append(f"Valoró positivamente: {', '.join(top_rated[:3])}.")

    return " ".join(parts)
