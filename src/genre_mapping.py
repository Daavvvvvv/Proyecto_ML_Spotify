"""Mapeo de sub-géneros a macro-géneros.

Definido y justificado en E1 (`notebooks/01_eda_baseline.ipynb`, sección 2)
mediante análisis de perfiles de audio (medianas de energy, acousticness,
loudness, danceability, valence). Este módulo lo centraliza para reutilización
en E2 y E3.

Notas:
- Sub-géneros no listados aquí caen en "other" (vía `.fillna("other")`).
- "sad" y "romance" se mantienen en "other" intencionalmente por perfil
  acústico ambiguo; ver E1 para justificación.
"""
from __future__ import annotations

GENRE_MAPPING: dict[str, str] = {
    # ROCK — núcleo, alternativo, punk, casos especiales
    "rock": "rock",
    "rock-n-roll": "rock",
    "hard-rock": "rock",
    "alt-rock": "rock",
    "alternative": "rock",
    "grunge": "rock",
    "psych-rock": "rock",
    "punk": "rock",
    "punk-rock": "rock",
    "rockabilly": "rock",
    "emo": "rock",
    "garage": "rock",
    "indie": "rock",
    "j-rock": "rock",
    "goth": "rock",

    # METAL — núcleo, extremo, agresivos
    "metal": "metal",
    "heavy-metal": "metal",
    "black-metal": "metal",
    "death-metal": "metal",
    "grindcore": "metal",
    "metalcore": "metal",
    "hardcore": "metal",
    "industrial": "metal",
    "happy": "metal",

    # POP (occidental)
    "pop": "pop",
    "power-pop": "pop",
    "indie-pop": "pop",
    "synth-pop": "pop",
    "pop-film": "pop",

    # ASIAN-POP
    "j-pop": "asian-pop",
    "k-pop": "asian-pop",
    "cantopop": "asian-pop",
    "mandopop": "asian-pop",
    "j-idol": "asian-pop",
    "j-dance": "asian-pop",
    "anime": "asian-pop",

    # ELECTRONIC
    "electronic": "electronic",
    "edm": "electronic",
    "house": "electronic",
    "deep-house": "electronic",
    "techno": "electronic",
    "trance": "electronic",
    "dubstep": "electronic",
    "drum-and-bass": "electronic",
    "minimal-techno": "electronic",
    "idm": "electronic",
    "electro": "electronic",
    "progressive-house": "electronic",
    "chicago-house": "electronic",
    "detroit-techno": "electronic",
    "breakbeat": "electronic",
    "hardstyle": "electronic",
    "club": "electronic",
    "dance": "electronic",
    "party": "electronic",

    # HIP-HOP — rap, R&B, trip-hop
    "hip-hop": "hip-hop",
    "r-n-b": "hip-hop",
    "trip-hop": "hip-hop",

    # LATIN — iberoamericana
    "latin": "latin",
    "latino": "latin",
    "reggaeton": "latin",
    "salsa": "latin",
    "samba": "latin",
    "tango": "latin",
    "brazil": "latin",
    "pagode": "latin",
    "forro": "latin",
    "mpb": "latin",
    "sertanejo": "latin",
    "spanish": "latin",

    # JAZZ
    "jazz": "jazz",

    # CLASSICAL
    "classical": "classical",
    "opera": "classical",
    "piano": "classical",

    # FOLK — folk, country, blues, acústico
    "folk": "folk",
    "country": "folk",
    "bluegrass": "folk",
    "honky-tonk": "folk",
    "acoustic": "folk",
    "singer-songwriter": "folk",
    "songwriter": "folk",
    "blues": "folk",
    "guitar": "folk",

    # REGGAE
    "reggae": "reggae",
    "dub": "reggae",
    "ska": "reggae",
    "dancehall": "reggae",

    # AMBIENT — atmosférica/funcional
    "ambient": "ambient",
    "chill": "ambient",
    "sleep": "ambient",
    "study": "ambient",
    "new-age": "ambient",

    # WORLD — étnica y por región
    "afrobeat": "world",
    "indian": "world",
    "iranian": "world",
    "turkish": "world",
    "malay": "world",
    "british": "world",
    "french": "world",
    "german": "world",
    "swedish": "world",
    "world-music": "world",

    # KIDS-COMEDY
    "children": "kids-comedy",
    "kids": "kids-comedy",
    "disney": "kids-comedy",
    "comedy": "kids-comedy",
    "show-tunes": "kids-comedy",

    # SOUL-FUNK
    "soul": "soul-funk",
    "funk": "soul-funk",
    "gospel": "soul-funk",
    "groove": "soul-funk",
    "disco": "soul-funk",
}

MACRO_GENRES: list[str] = sorted(set(GENRE_MAPPING.values()) | {"other"})

NUMERIC_FEATURES: list[str] = [
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "duration_ms",
]

CATEGORICAL_FEATURES: list[str] = ["key", "mode", "time_signature", "explicit"]
