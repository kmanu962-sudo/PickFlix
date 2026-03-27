import re
from datetime import date, datetime

import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(
    page_title="PickFlix",
    page_icon="🎬",
    layout="wide"
)


# ---------------------------
# SESSION STATE
# ---------------------------
if "watchlist" not in st.session_state:
    st.session_state.watchlist = []

if "show_filters" not in st.session_state:
    st.session_state.show_filters = False

if "show_search_details" not in st.session_state:
    st.session_state.show_search_details = False


# ---------------------------
# STYLING
# ---------------------------
def set_custom_style():
    st.markdown(
        """
        <style>
        .stApp {
            background: #120b08;
        }

        .block-container {
            max-width: 1200px !important;
            padding-top: 0.2rem !important;
            padding-bottom: 2rem !important;
        }

        html, body, [class*="css"], p, div, span, label {
            font-family: "Times New Roman", serif !important;
            color: #f3eadf !important;
        }

        h1, h2, h3, h4, h5, h6 {
            font-family: "Times New Roman", serif !important;
            color: #f7efe4 !important;
            letter-spacing: 0.2px;
        }

        /* remove default Streamlit white top bar */
        header[data-testid="stHeader"] {
            background: transparent !important;
            height: 0rem !important;
        }

        div[data-testid="stToolbar"] {
            right: 1rem;
        }

        main {
            padding-top: 0rem !important;
        }

        /* hide sidebar */
        section[data-testid="stSidebar"] {
            display: none !important;
        }

        /* text input */
        .stTextInput input {
            color: #111111 !important;
            background-color: #f3eee8 !important;
            border-radius: 8px !important;
            border: none !important;
            -webkit-text-fill-color: #111111 !important;
        }

        .stTextInput input::placeholder {
            color: #6b6b6b !important;
            opacity: 1 !important;
        }

        textarea {
            color: #111111 !important;
            background-color: #f3eee8 !important;
            -webkit-text-fill-color: #111111 !important;
        }

        /* selectbox styling */
        div[data-baseweb="select"] > div {
            background-color: #f3eee8 !important;
            color: #111111 !important;
            border-radius: 8px !important;
            border: none !important;
        }

        div[data-baseweb="select"] span {
            color: #111111 !important;
            -webkit-text-fill-color: #111111 !important;
        }

        div[data-baseweb="select"] input {
            color: #111111 !important;
            -webkit-text-fill-color: #111111 !important;
        }

        div[data-baseweb="select"] svg {
            fill: #111111 !important;
        }

        /* multiselect tags */
        span[data-baseweb="tag"] {
            background-color: #e8ddd0 !important;
            color: #111111 !important;
        }

        /* date input */
        .stDateInput input {
            color: #111111 !important;
            background-color: #f3eee8 !important;
            -webkit-text-fill-color: #111111 !important;
        }

        /* number-like widgets */
        .stSlider [data-baseweb="slider"] {
            color: #f3eadf !important;
        }

        /* buttons */
        .stButton > button {
            background: transparent !important;
            color: #e7d3bc !important;
            border: 1px solid rgba(231, 211, 188, 0.45) !important;
            border-radius: 999px !important;
            padding: 0.45rem 1.1rem !important;
            font-family: "Times New Roman", serif !important;
            font-size: 0.95rem !important;
        }

        .stButton > button:hover {
            border-color: rgba(231, 211, 188, 0.9) !important;
            color: #fff6ea !important;
        }

        .element-container {
            margin-bottom: 0.35rem !important;
        }

        .hero-wrap {
            position: relative;
            width: 100%;
            height: 420px;
            border-radius: 22px;
            overflow: hidden;
            margin-bottom: 1.25rem;
            background:
                linear-gradient(rgba(12,8,6,0.50), rgba(12,8,6,0.68)),
                url("https://images.unsplash.com/photo-1502134249126-9f3755a50d78?auto=format&fit=crop&w=1600&q=80");
            background-size: cover;
            background-position: center;
            display: flex;
            align-items: center;
            justify-content: center;
            text-align: center;
        }

        .hero-inner {
            max-width: 760px;
            padding: 2rem;
        }

        .hero-brand {
            font-size: 0.95rem;
            letter-spacing: 3px;
            text-transform: uppercase;
            color: #d8c0a5;
            margin-bottom: 1rem;
        }

        .hero-title {
            font-size: 4.4rem;
            line-height: 1.02;
            margin: 0 0 0.9rem 0;
            color: #fff6ea;
            font-weight: normal;
        }

        .hero-subtitle {
            font-size: 1.18rem;
            line-height: 1.6;
            color: #f0e2d2;
            max-width: 700px;
            margin: 0 auto;
        }

        button[data-baseweb="tab"] {
            font-family: "Times New Roman", serif !important;
            color: #e7d8c6 !important;
            font-size: 1rem !important;
        }

        .section-panel {
            border: 1px solid rgba(231, 211, 188, 0.18);
            border-radius: 16px;
            padding: 1rem 1rem 0.25rem 1rem;
            background: rgba(255,255,255,0.02);
            margin-bottom: 1rem;
        }

        .movie-card {
            border: 1px solid rgba(231, 211, 188, 0.16);
            border-radius: 18px;
            padding: 1rem;
            background: rgba(255,255,255,0.02);
            margin-bottom: 1rem;
        }

        hr {
            border: none;
            border-top: 1px solid rgba(231, 211, 188, 0.15);
            margin: 1rem 0;
        }
        
        /* FIX DROPDOWN MENU TEXT (the actual list items) */
        div[data-baseweb="menu"] {
            background-color: #f3eee8 !important;
        }

        div[data-baseweb="menu"] div {
            color: #111111 !important;
        }

        /* selected item text */
        div[data-baseweb="select"] span {
            color: #111111 !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )


set_custom_style()


# ---------------------------
# CONFIG
# ---------------------------
API_KEY = "f0baf4291ae67be04e7d59b95c295194"
BASE_URL = "https://api.themoviedb.org/3"
IMAGE_BASE = "https://image.tmdb.org/t/p/w500"

if API_KEY == "PASTE_YOUR_API_KEY_HERE":
    st.error("Add your TMDB API key first.")
    st.stop()


# ---------------------------
# NLP SETUP
# ---------------------------
@st.cache_resource
def get_sentiment_analyzer():
    try:
        return SentimentIntensityAnalyzer()
    except LookupError:
        nltk.download("vader_lexicon", quiet=True)
        return SentimentIntensityAnalyzer()


sia = get_sentiment_analyzer()


# ---------------------------
# API HELPERS
# ---------------------------
def tmdb_get(endpoint, params=None):
    if params is None:
        params = {}
    params["api_key"] = API_KEY
    response = requests.get(f"{BASE_URL}{endpoint}", params=params, timeout=20)
    response.raise_for_status()
    return response.json()


@st.cache_data(show_spinner=False)
def get_genres():
    data = tmdb_get("/genre/movie/list")
    return {g["name"]: g["id"] for g in data.get("genres", [])}


@st.cache_data(show_spinner=False)
def get_languages():
    try:
        data = tmdb_get("/configuration/languages")
        mapping = {}
        for item in data:
            code = item.get("iso_639_1")
            name = item.get("english_name") or item.get("name")
            if code and name:
                mapping[name] = code
        return dict(sorted(mapping.items()))
    except Exception:
        return {
            "English": "en",
            "Spanish": "es",
            "French": "fr",
            "German": "de",
            "Italian": "it",
            "Japanese": "ja",
            "Korean": "ko",
            "Hindi": "hi",
            "Chinese": "zh",
            "Portuguese": "pt",
        }


@st.cache_data(show_spinner=False)
def search_movies(query):
    if not query.strip():
        return []
    data = tmdb_get("/search/movie", {"query": query, "include_adult": "false"})
    return data.get("results", [])


@st.cache_data(show_spinner=False)
def get_trending_movies():
    data = tmdb_get("/trending/movie/week")
    return data.get("results", [])


@st.cache_data(show_spinner=False)
def discover_movies(
    page=1,
    min_rating=0.0,
    min_votes=0,
    start_release_date=None,
    end_release_date=None,
    sort_by="popularity.desc"
):
    params = {
        "page": page,
        "include_adult": "false",
        "vote_average.gte": min_rating,
        "vote_count.gte": min_votes,
        "sort_by": sort_by,
    }

    if start_release_date:
        params["primary_release_date.gte"] = start_release_date
    if end_release_date:
        params["primary_release_date.lte"] = end_release_date

    data = tmdb_get("/discover/movie", params)
    return data.get("results", [])


def extract_certification(release_dates_payload):
    results = release_dates_payload.get("results", [])
    preferred = ["US", "CA", "GB", "AU"]

    for country_code in preferred:
        for country in results:
            if country.get("iso_3166_1") == country_code:
                for rel in country.get("release_dates", []):
                    cert = rel.get("certification")
                    if cert:
                        return cert

    for country in results:
        for rel in country.get("release_dates", []):
            cert = rel.get("certification")
            if cert:
                return cert

    return "Not Rated"


def extract_director(crew):
    for person in crew:
        if person.get("job") == "Director":
            return person.get("name")
    return "Not available"


def extract_writers(crew):
    writers = []
    for person in crew:
        if person.get("job") in {"Writer", "Screenplay", "Story"}:
            writers.append(person.get("name"))
    return list(dict.fromkeys([w for w in writers if w]))[:5]


@st.cache_data(show_spinner=False)
def get_movie_metadata(movie_id):
    details = tmdb_get(
        f"/movie/{movie_id}",
        {"append_to_response": "credits,keywords,release_dates"}
    )

    release_date = details.get("release_date") or ""
    year = int(release_date[:4]) if release_date and release_date[:4].isdigit() else None

    credits = details.get("credits", {})
    cast_list = [c.get("name") for c in credits.get("cast", [])[:8] if c.get("name")]
    crew = credits.get("crew", [])
    director = extract_director(crew)
    writers = extract_writers(crew)

    keyword_block = details.get("keywords", {})
    keyword_items = keyword_block.get("keywords", []) or keyword_block.get("results", [])
    keywords = [k.get("name") for k in keyword_items if k.get("name")]

    genres = [g.get("name") for g in details.get("genres", []) if g.get("name")]
    certification = extract_certification(details.get("release_dates", {}))
    overview = details.get("overview") or "No plot summary available."

    original_language = details.get("original_language", "")
    runtime = details.get("runtime") or 0
    vote_average = details.get("vote_average") or 0.0
    vote_count = details.get("vote_count") or 0
    revenue = details.get("revenue") or 0
    budget = details.get("budget") or 0
    popularity = details.get("popularity") or 0.0

    content_text = " ".join(
        genres
        + keywords
        + cast_list
        + ([director] if director and director != "Not available" else [])
        + writers
        + [overview]
    )

    sentiment_score = sia.polarity_scores(overview)["compound"]

    return {
        "id": details.get("id"),
        "title": details.get("title", "No title"),
        "year": year,
        "release_date": release_date,
        "runtime": runtime,
        "certification": certification,
        "genres": genres,
        "plot_summary": overview,
        "keywords": keywords,
        "cast": cast_list,
        "director": director,
        "writers": writers,
        "rating": vote_average,
        "vote_count": vote_count,
        "box_office": revenue,
        "budget": budget,
        "popularity": popularity,
        "language_code": original_language,
        "poster_path": details.get("poster_path"),
        "backdrop_path": details.get("backdrop_path"),
        "content_text": content_text.strip(),
        "sentiment_score": sentiment_score,
    }


# ---------------------------
# NATURAL LANGUAGE PARSER
# ---------------------------
def parse_natural_language(text, genre_dict, language_dict):
    text_lower = text.lower()

    parsed = {
        "query_terms": [],
        "genres": [],
        "min_rating": None,
        "min_votes": None,
        "year_min": None,
        "year_max": None,
        "release_start": None,
        "release_end": None,
        "runtime_min": None,
        "runtime_max": None,
        "certifications": [],
        "languages": [],
        "actor_text": "",
        "director_text": "",
        "writer_text": "",
        "genre_logic": None,
    }

    mood_to_genres = {
        "funny": ["Comedy"],
        "romantic": ["Romance"],
        "sad": ["Drama"],
        "emotional": ["Drama"],
        "scary": ["Horror"],
        "dark": ["Thriller"],
        "space": ["Science Fiction"],
        "sci-fi": ["Science Fiction"],
        "mystery": ["Mystery"],
        "crime": ["Crime"],
        "family": ["Family"],
        "animated": ["Animation"],
        "adventure": ["Adventure"],
        "fantasy": ["Fantasy"],
        "war": ["War"],
        "history": ["History"],
        "action": ["Action"],
    }

    certifications = ["G", "PG", "PG-13", "R", "NC-17", "NR", "TV-MA"]

    for genre_name in genre_dict.keys():
        if genre_name.lower() in text_lower and genre_name not in parsed["genres"]:
            parsed["genres"].append(genre_name)

    for mood_word, mapped_genres in mood_to_genres.items():
        if mood_word in text_lower:
            for g in mapped_genres:
                if g in genre_dict and g not in parsed["genres"]:
                    parsed["genres"].append(g)

    for cert in certifications:
        if cert.lower() in text_lower:
            parsed["certifications"].append(cert)

    for language_name, language_code in language_dict.items():
        if language_name.lower() in text_lower:
            parsed["languages"].append(language_code)

    year_match = re.search(r"\b(19\d{2}|20\d{2})\b", text_lower)
    if year_match:
        parsed["year_min"] = int(year_match.group(1))
        parsed["year_max"] = int(year_match.group(1))

    decade_match = re.search(r"\b(19\d0|20\d0)s\b", text_lower)
    if decade_match:
        decade_start = int(decade_match.group(1))
        parsed["year_min"] = decade_start
        parsed["year_max"] = decade_start + 9

    after_match = re.search(r"(after|since)\s+(19\d{2}|20\d{2})", text_lower)
    if after_match:
        parsed["year_min"] = int(after_match.group(2))

    before_match = re.search(r"(before)\s+(19\d{2}|20\d{2})", text_lower)
    if before_match:
        parsed["year_max"] = int(before_match.group(2))

    rating_match = re.search(r"(above|over|at least|min(?:imum)?)\s*(\d+(\.\d+)?)", text_lower)
    if rating_match:
        parsed["min_rating"] = float(rating_match.group(2))

    votes_match = re.search(r"(at least|min(?:imum)?|over|above)\s*(\d+)\s*(votes|vote)", text_lower)
    if votes_match:
        parsed["min_votes"] = int(votes_match.group(2))

    runtime_under = re.search(r"(under|less than)\s*(\d+)\s*(minutes|min)", text_lower)
    if runtime_under:
        parsed["runtime_max"] = int(runtime_under.group(2))

    runtime_over = re.search(r"(over|more than)\s*(\d+)\s*(minutes|min)", text_lower)
    if runtime_over:
        parsed["runtime_min"] = int(runtime_over.group(2))

    between_runtime = re.search(r"between\s*(\d+)\s*and\s*(\d+)\s*(minutes|min)", text_lower)
    if between_runtime:
        parsed["runtime_min"] = int(between_runtime.group(1))
        parsed["runtime_max"] = int(between_runtime.group(2))

    actor_match = re.search(r"(starring|with actor|actor)\s+([a-zA-Z\s\-']+)", text_lower)
    if actor_match:
        parsed["actor_text"] = actor_match.group(2).strip()

    director_match = re.search(r"(directed by|director)\s+([a-zA-Z\s\-']+)", text_lower)
    if director_match:
        parsed["director_text"] = director_match.group(2).strip()

    writer_match = re.search(r"(written by|writer)\s+([a-zA-Z\s\-']+)", text_lower)
    if writer_match:
        parsed["writer_text"] = writer_match.group(2).strip()

    if "all genres" in text_lower or "must include all" in text_lower:
        parsed["genre_logic"] = "AND"
    elif "any genre" in text_lower or "either genre" in text_lower:
        parsed["genre_logic"] = "OR"

    stop_words = {
        "movie", "movies", "film", "films", "find", "show", "me", "something",
        "with", "and", "or", "the", "a", "an", "from", "after", "before",
        "under", "over", "less", "than", "more", "minutes", "minute", "min"
    }

    blocked = set(word.lower() for word in mood_to_genres.keys())

    words = re.findall(r"[a-zA-Z0-9\-']+", text_lower)
    for word in words:
        if (
            word not in stop_words
            and word not in blocked
            and not re.fullmatch(r"\d+", word)
            and word not in [c.lower() for c in certifications]
        ):
            parsed["query_terms"].append(word)

    parsed["query_terms"] = list(dict.fromkeys(parsed["query_terms"]))
    parsed["genres"] = list(dict.fromkeys(parsed["genres"]))
    parsed["certifications"] = list(dict.fromkeys(parsed["certifications"]))
    parsed["languages"] = list(dict.fromkeys(parsed["languages"]))

    return parsed


# ---------------------------
# FILTERING / POOL BUILDING
# ---------------------------
def build_candidate_ids(
    search_query,
    min_rating,
    min_votes,
    start_release_date,
    end_release_date
):
    ids = []

    if search_query.strip():
        for item in search_movies(search_query)[:15]:
            if item.get("id"):
                ids.append(item["id"])

    for page in [1, 2]:
        discovered = discover_movies(
            page=page,
            min_rating=min_rating,
            min_votes=min_votes,
            start_release_date=start_release_date,
            end_release_date=end_release_date
        )
        for item in discovered[:12]:
            if item.get("id"):
                ids.append(item["id"])

    for item in get_trending_movies()[:12]:
        if item.get("id"):
            ids.append(item["id"])

    unique_ids = []
    seen = set()
    for movie_id in ids:
        if movie_id not in seen:
            seen.add(movie_id)
            unique_ids.append(movie_id)

    return unique_ids[:40]


def runtime_bracket_pass(runtime, bracket):
    if bracket == "Any":
        return True
    if runtime is None:
        return False
    if bracket == "Under 90":
        return runtime < 90
    if bracket == "90 to 120":
        return 90 <= runtime <= 120
    if bracket == "121 to 150":
        return 121 <= runtime <= 150
    if bracket == "150+":
        return runtime >= 150
    return True


def year_pass(movie_year, year_min, year_max, decades):
    if movie_year is None:
        return False

    if year_min and movie_year < year_min:
        return False
    if year_max and movie_year > year_max:
        return False

    if decades:
        in_any_decade = any(dec <= movie_year <= dec + 9 for dec in decades)
        if not in_any_decade:
            return False

    return True


def release_period_pass(release_date_str, start_date, end_date):
    if not start_date and not end_date:
        return True
    if not release_date_str:
        return False

    try:
        movie_date = datetime.strptime(release_date_str, "%Y-%m-%d").date()
    except ValueError:
        return False

    if start_date and movie_date < start_date:
        return False
    if end_date and movie_date > end_date:
        return False

    return True


def personnel_pass(movie, actor_text, director_text, writer_text):
    actor_text = actor_text.strip().lower()
    director_text = director_text.strip().lower()
    writer_text = writer_text.strip().lower()

    if actor_text:
        joined_cast = " ".join(movie["cast"]).lower()
        if actor_text not in joined_cast:
            return False

    if director_text:
        if director_text not in movie["director"].lower():
            return False

    if writer_text:
        joined_writers = " ".join(movie["writers"]).lower()
        if writer_text not in joined_writers:
            return False

    return True


def genre_pass(movie_genres, selected_genres, logic):
    if not selected_genres:
        return True

    movie_set = set(movie_genres)
    selected_set = set(selected_genres)

    if logic == "AND":
        return selected_set.issubset(movie_set)
    return len(movie_set.intersection(selected_set)) > 0


def apply_filters(movies, filters):
    filtered = []

    for movie in movies:
        if movie["rating"] < filters["min_rating"]:
            continue
        if movie["vote_count"] < filters["min_votes"]:
            continue
        if not year_pass(movie["year"], filters["year_min"], filters["year_max"], filters["decades"]):
            continue
        if not release_period_pass(movie["release_date"], filters["release_start"], filters["release_end"]):
            continue
        if not runtime_bracket_pass(movie["runtime"], filters["runtime_bracket"]):
            continue
        if filters["certifications"]:
            if movie["certification"] not in filters["certifications"]:
                continue
        if filters["languages"]:
            if movie["language_code"] not in filters["languages"]:
                continue
        if not personnel_pass(movie, filters["actor_text"], filters["director_text"], filters["writer_text"]):
            continue
        if not genre_pass(movie["genres"], filters["genres"], filters["genre_logic"]):
            continue

        filtered.append(movie)

    return filtered


# ---------------------------
# FEATURE ENGINEERING / RECOMMENDERS
# ---------------------------
def prepare_feature_space(movies):
    if not movies:
        return None, None, None, None, None

    df = pd.DataFrame(movies)
    df = df.drop_duplicates(subset="id").reset_index(drop=True)

    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
    content_matrix = vectorizer.fit_transform(df["content_text"].fillna(""))

    sentiments = df["sentiment_score"].fillna(0).astype(float).to_numpy()
    ratings = df["rating"].fillna(0).astype(float).to_numpy()
    votes = df["vote_count"].fillna(0).astype(float).to_numpy()

    max_votes = votes.max() if len(votes) else 1
    vote_norm = votes / max_votes if max_votes > 0 else np.zeros_like(votes)
    rating_norm = ratings / 10.0

    popularity_component = 0.6 * rating_norm + 0.4 * vote_norm

    return df, vectorizer, content_matrix, sentiments, popularity_component


def get_content_scores(seed_idx, content_matrix):
    return cosine_similarity(content_matrix[seed_idx], content_matrix).flatten()


def get_sentiment_scores(seed_sentiment, sentiments):
    return 1 - (np.abs(sentiments - seed_sentiment) / 2)


def hybrid_scores(content_scores, sentiment_scores, popularity_component):
    return 0.55 * content_scores + 0.25 * sentiment_scores + 0.20 * popularity_component


def explain_similarity(seed, candidate):
    reasons = []

    shared_genres = list(set(seed["genres"]).intersection(set(candidate["genres"])))
    shared_keywords = list(set(seed["keywords"]).intersection(set(candidate["keywords"])))
    shared_cast = list(set(seed["cast"]).intersection(set(candidate["cast"])))

    if shared_genres:
        reasons.append(f"shared genres ({', '.join(shared_genres[:3])})")
    if seed["director"] != "Not available" and seed["director"] == candidate["director"]:
        reasons.append(f"the same director ({seed['director']})")
    if shared_cast:
        reasons.append(f"overlapping cast ({', '.join(shared_cast[:2])})")
    if shared_keywords:
        reasons.append(f"similar keywords ({', '.join(shared_keywords[:3])})")
    if abs(seed["sentiment_score"] - candidate["sentiment_score"]) <= 0.15:
        reasons.append("a similar plot tone")

    if not reasons:
        return "Recommended because its metadata profile and plot tone are close to your selected movie."

    return "Recommended because of " + ", ".join(reasons[:3]) + "."


def recommend_for_seed(seed_id, movies, approach="Hybrid", top_n=8):
    prepared = prepare_feature_space(movies)
    if prepared[0] is None:
        return []

    df, _, content_matrix, sentiments, popularity_component = prepared
    id_to_idx = {movie_id: idx for idx, movie_id in enumerate(df["id"].tolist())}

    if seed_id not in id_to_idx:
        return []

    seed_idx = id_to_idx[seed_id]
    content_scores = get_content_scores(seed_idx, content_matrix)
    sentiment_scores = get_sentiment_scores(sentiments[seed_idx], sentiments)
    hybrid = hybrid_scores(content_scores, sentiment_scores, popularity_component)

    if approach == "Content-Based":
        scores = content_scores
    elif approach == "Sentiment-Based":
        scores = sentiment_scores
    else:
        scores = hybrid

    ranked = []
    seed_movie = df.iloc[seed_idx].to_dict()

    for idx, score in enumerate(scores):
        if idx == seed_idx:
            continue
        candidate = df.iloc[idx].to_dict()
        candidate["recommendation_score"] = float(score)
        candidate["explanation"] = explain_similarity(seed_movie, candidate)
        ranked.append(candidate)

    ranked.sort(key=lambda x: x["recommendation_score"], reverse=True)
    return ranked[:top_n]


def recommend_from_watchlist_profile(watchlist_ids, movies, top_n=8):
    prepared = prepare_feature_space(movies)
    if prepared[0] is None:
        return []

    df, _, content_matrix, sentiments, popularity_component = prepared
    id_to_idx = {movie_id: idx for idx, movie_id in enumerate(df["id"].tolist())}

    watch_idxs = [id_to_idx[mid] for mid in watchlist_ids if mid in id_to_idx]
    if not watch_idxs:
        return []

    profile_vector = np.asarray(content_matrix[watch_idxs].mean(axis=0)).reshape(1, -1)
    profile_content_scores = cosine_similarity(profile_vector, content_matrix).flatten()

    profile_sentiment = np.mean(sentiments[watch_idxs])
    profile_sentiment_scores = get_sentiment_scores(profile_sentiment, sentiments)

    scores = hybrid_scores(profile_content_scores, profile_sentiment_scores, popularity_component)

    watchlist_movies = df.iloc[watch_idxs].to_dict("records")
    top_watchlist_genres = []
    for movie in watchlist_movies:
        top_watchlist_genres.extend(movie["genres"])
    genre_counts = pd.Series(top_watchlist_genres).value_counts().index.tolist()[:3] if top_watchlist_genres else []

    ranked = []
    watchlist_set = set(watchlist_ids)

    for idx, score in enumerate(scores):
        candidate = df.iloc[idx].to_dict()
        if candidate["id"] in watchlist_set:
            continue

        candidate["recommendation_score"] = float(score)
        overlap = list(set(candidate["genres"]).intersection(set(genre_counts)))
        if overlap:
            candidate["explanation"] = (
                f"Recommended because your watchlist leans toward {', '.join(genre_counts[:3])}, "
                f"and this movie matches {', '.join(overlap[:2])} with a similar tone."
            )
        else:
            candidate["explanation"] = "Recommended because it is close to your watchlist’s overall content and sentiment profile."

        ranked.append(candidate)

    ranked.sort(key=lambda x: x["recommendation_score"], reverse=True)
    return ranked[:top_n]


# ---------------------------
# UI HELPERS
# ---------------------------
def language_name_from_code(code, language_map):
    reverse = {v: k for k, v in language_map.items()}
    return reverse.get(code, code.upper() if code else "Unknown")


def poster_url(path):
    return f"{IMAGE_BASE}{path}" if path else None


def render_movie_card(movie, show_watchlist_button=True, context="default"):
    st.markdown('<div class="movie-card">', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2.2], gap="large")

    with col1:
        if movie.get("poster_path"):
            st.image(poster_url(movie["poster_path"]), use_container_width=True)
        else:
            st.markdown("*(No poster available)*")

    with col2:
        st.subheader(f"{movie['title']} ({movie['year'] if movie['year'] else 'Unknown'})")
        st.write(f"**Release Date:** {movie['release_date'] or 'Unknown'}")
        st.write(f"**Runtime:** {movie['runtime']} min")
        st.write(f"**Rating:** {movie['rating']}")
        st.write(f"**Genres:** {', '.join(movie['genres']) if movie['genres'] else 'Not available'}")
        st.write(f"**Director:** {movie['director']}")
        st.write(f"**Plot Summary:** {movie['plot_summary']}")

        meta_key = f"show_meta_{context}_{movie['id']}"

        if meta_key not in st.session_state:
            st.session_state[meta_key] = False

        button_label = "Show Full Metadata" if not st.session_state[meta_key] else "Hide Full Metadata"

        if st.button(button_label, key=f"btn_{meta_key}"):
            st.session_state[meta_key] = not st.session_state[meta_key]

        if st.session_state[meta_key]:
            st.write(f"Certification: {movie['certification']}")
            st.write(f"Language: {language_name_from_code(movie['language_code'], language_dict)}")
            st.write(f"Vote Count: {movie['vote_count']}")
            st.write(f"Box Office: ${movie['box_office']:,}" if movie['box_office'] else "Box Office: Not available")
            st.write(f"Popularity: {movie['popularity']}")
            st.write(f"Writers: {', '.join(movie['writers']) if movie['writers'] else 'Not available'}")
            st.write(f"Cast: {', '.join(movie['cast']) if movie['cast'] else 'Not available'}")
            st.write(f"Keywords: {', '.join(movie['keywords']) if movie['keywords'] else 'Not available'}")

        if show_watchlist_button:
            already_saved = any(saved["id"] == movie["id"] for saved in st.session_state.watchlist)
            if already_saved:
                st.write("Already in watchlist")
            else:
                if st.button(
                    f"Add {movie['title']} to Watchlist",
                    key=f"add_watch_{context}_{movie['id']}"
                ):
                    st.session_state.watchlist.append(
                        {
                            "id": movie["id"],
                            "title": movie["title"],
                            "year": movie["year"],
                            "poster_path": movie["poster_path"],
                            "rating": movie["rating"],
                        }
                    )
                    st.success(f"{movie['title']} added to your watchlist.")
                    st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


def render_recommendations(recommendations, context="recommendations"):
    if not recommendations:
        st.info("No recommendations available.")
        return

    for rec in recommendations:
        render_movie_card(rec, show_watchlist_button=True, context=context)
        st.write(f"**Why this matches:** {rec['explanation']}")
        st.write(f"**Recommendation Score:** {rec['recommendation_score']:.3f}")


def build_comparison_table(movie_a, movie_b, language_map):
    rows = [
        ("Title", movie_a["title"], movie_b["title"]),
        ("Year", movie_a["year"], movie_b["year"]),
        ("Release Date", movie_a["release_date"], movie_b["release_date"]),
        ("Runtime", movie_a["runtime"], movie_b["runtime"]),
        ("Certification", movie_a["certification"], movie_b["certification"]),
        ("Language", language_name_from_code(movie_a["language_code"], language_map), language_name_from_code(movie_b["language_code"], language_map)),
        ("Rating", movie_a["rating"], movie_b["rating"]),
        ("Vote Count", movie_a["vote_count"], movie_b["vote_count"]),
        ("Box Office", movie_a["box_office"], movie_b["box_office"]),
        ("Popularity", movie_a["popularity"], movie_b["popularity"]),
        ("Director", movie_a["director"], movie_b["director"]),
        ("Genres", ", ".join(movie_a["genres"]), ", ".join(movie_b["genres"])),
    ]
    return pd.DataFrame(rows, columns=["Metric", movie_a["title"], movie_b["title"]])


# ---------------------------
# HERO
# ---------------------------
st.markdown(
    """
    <div class="hero-wrap">
        <div class="hero-inner">
            <div class="hero-brand">PickFlix</div>
            <div class="hero-title">Curated cinema,<br>intelligently matched.</div>
            <div class="hero-subtitle">
                A refined movie discovery experience combining natural language search,
                advanced filtering, recommendation intelligence, and personalized watchlists.
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# ---------------------------
# LOOKUPS
# ---------------------------
genre_dict = get_genres()
language_dict = get_languages()


# ---------------------------
# FILTER TOGGLE
# ---------------------------
toggle_col1, toggle_col2 = st.columns([1, 5])

with toggle_col1:
    if st.button("Show Filters" if not st.session_state.show_filters else "Hide Filters", key="toggle_filters_btn"):
        st.session_state.show_filters = not st.session_state.show_filters
        st.rerun()

if st.session_state.show_filters:
    st.markdown('<div class="section-panel">', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        search_mode = st.radio(
            "Recommendation approach",
            ["Hybrid", "Content-Based", "Sentiment-Based"],
            key="search_mode_top"
        )

        min_rating = st.slider(
            "Minimum rating",
            0.0, 10.0, 0.0, 0.5,
            key="min_rating_top"
        )

        min_votes = st.slider(
            "Minimum vote count",
            0, 5000, 50, 50,
            key="min_votes_top"
        )

        year_min, year_max = st.slider(
            "Year range",
            1950, date.today().year, (1990, date.today().year),
            key="year_range_top"
        )

        decade_options = [1950, 1960, 1970, 1980, 1990, 2000, 2010, 2020]
        selected_decades = st.multiselect(
            "Decade filters",
            options=decade_options,
            format_func=lambda x: f"{x}s",
            key="decades_top"
        )

    with c2:
        release_start = st.date_input("Release period start", value=None, key="release_start_top")
        release_end = st.date_input("Release period end", value=None, key="release_end_top")

        runtime_bracket = st.selectbox(
            "Runtime bracket",
            ["Any", "Under 90", "90 to 120", "121 to 150", "150+"],
            key="runtime_top"
        )

        cert_options = ["G", "PG", "PG-13", "R", "NC-17", "NR", "TV-MA"]
        selected_certs = st.multiselect(
            "Certification levels",
            cert_options,
            key="certs_top"
        )

        selected_languages_names = st.multiselect(
            "Languages",
            options=list(language_dict.keys()),
            key="langs_top"
        )
        selected_language_codes = [language_dict[name] for name in selected_languages_names]

    with c3:
        selected_genres = st.multiselect(
            "Genres",
            options=list(genre_dict.keys()),
            key="genres_top"
        )

        genre_logic = st.radio(
            "Genre logic",
            ["OR", "AND"],
            horizontal=True,
            key="genre_logic_top"
        )

        actor_text = st.text_input("Actor preference", key="actor_top")
        director_text = st.text_input("Director preference", key="director_top")
        writer_text = st.text_input("Writer preference", key="writer_top")

        max_results = st.slider(
            "Maximum results to display",
            5, 25, 10,
            key="max_results_top"
        )

    st.markdown("</div>", unsafe_allow_html=True)
else:
    search_mode = "Hybrid"
    min_rating = 0.0
    min_votes = 50
    year_min, year_max = (1990, date.today().year)
    selected_decades = []
    release_start = None
    release_end = None
    runtime_bracket = "Any"
    selected_certs = []
    selected_languages_names = []
    selected_language_codes = []
    selected_genres = []
    genre_logic = "OR"
    actor_text = ""
    director_text = ""
    writer_text = ""
    max_results = 10


# ---------------------------
# MAIN QUERY
# ---------------------------
user_input = st.text_input(
    "Search for a movie or describe what you want:",
    placeholder="Try: dark sci-fi after 2010 over 7 with at least 500 votes directed by denis villeneuve"
)

parsed = parse_natural_language(user_input, genre_dict, language_dict) if user_input else {
    "query_terms": [],
    "genres": [],
    "min_rating": None,
    "min_votes": None,
    "year_min": None,
    "year_max": None,
    "release_start": None,
    "release_end": None,
    "runtime_min": None,
    "runtime_max": None,
    "certifications": [],
    "languages": [],
    "actor_text": "",
    "director_text": "",
    "writer_text": "",
    "genre_logic": None,
}

final_filters = {
    "min_rating": parsed["min_rating"] if parsed["min_rating"] is not None else min_rating,
    "min_votes": parsed["min_votes"] if parsed["min_votes"] is not None else min_votes,
    "year_min": parsed["year_min"] if parsed["year_min"] is not None else year_min,
    "year_max": parsed["year_max"] if parsed["year_max"] is not None else year_max,
    "release_start": parsed["release_start"] if parsed["release_start"] else release_start,
    "release_end": parsed["release_end"] if parsed["release_end"] else release_end,
    "runtime_bracket": runtime_bracket,
    "certifications": list(dict.fromkeys(selected_certs + parsed["certifications"])),
    "languages": list(dict.fromkeys(selected_language_codes + parsed["languages"])),
    "genres": list(dict.fromkeys(selected_genres + parsed["genres"])),
    "genre_logic": parsed["genre_logic"] if parsed["genre_logic"] else genre_logic,
    "actor_text": parsed["actor_text"] if parsed["actor_text"] else actor_text,
    "director_text": parsed["director_text"] if parsed["director_text"] else director_text,
    "writer_text": parsed["writer_text"] if parsed["writer_text"] else writer_text,
    "decades": selected_decades,
}

search_query = " ".join(parsed["query_terms"]).strip() if user_input else ""
if not search_query and user_input:
    search_query = user_input.strip()


# ---------------------------
# SEARCH DETAILS TOGGLE
# ---------------------------
detail_col1, detail_col2 = st.columns([1, 5])

with detail_col1:
    if st.button("Show Search Details" if not st.session_state.show_search_details else "Hide Search Details", key="toggle_search_details_btn"):
        st.session_state.show_search_details = not st.session_state.show_search_details
        st.rerun()

if st.session_state.show_search_details:
    st.markdown('<div class="section-panel">', unsafe_allow_html=True)
    st.write(f"**Search terms:** {search_query if search_query else 'Trending / filtered discovery'}")
    st.write(f"**Approach:** {search_mode}")
    st.write(f"**Genres:** {final_filters['genres'] if final_filters['genres'] else 'Any'}")
    st.write(f"**Year range:** {final_filters['year_min']} to {final_filters['year_max']}")
    st.write(f"**Minimum rating:** {final_filters['min_rating']}")
    st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------
# BUILD MOVIE POOL
# ---------------------------
with st.spinner("Building your movie pool..."):
    candidate_ids = build_candidate_ids(
        search_query=search_query,
        min_rating=final_filters["min_rating"],
        min_votes=final_filters["min_votes"],
        start_release_date=final_filters["release_start"].isoformat() if isinstance(final_filters["release_start"], date) else None,
        end_release_date=final_filters["release_end"].isoformat() if isinstance(final_filters["release_end"], date) else None,
    )

    for saved in st.session_state.watchlist:
        if saved["id"] not in candidate_ids:
            candidate_ids.append(saved["id"])

    movie_pool = []
    for movie_id in candidate_ids:
        try:
            movie_pool.append(get_movie_metadata(movie_id))
        except Exception:
            continue

filtered_movies = apply_filters(movie_pool, final_filters)
filtered_movies = sorted(
    filtered_movies,
    key=lambda m: (m["rating"], m["vote_count"], m["popularity"]),
    reverse=True
)


# ---------------------------
# TABS
# ---------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Discover", "Recommendations", "Watchlist", "Insights", "Trending"]
)


# ---------------------------
# TAB 1: DISCOVER
# ---------------------------
with tab1:
    st.subheader("Search Results")
    if not filtered_movies:
        st.warning("No movies matched your full filter set.")
    else:
        for movie in filtered_movies[:max_results]:
            render_movie_card(movie, show_watchlist_button=True, context="search")


# ---------------------------
# TAB 2: RECOMMENDATIONS
# ---------------------------
with tab2:
    st.subheader("Movie Recommendations")
    if len(filtered_movies) < 2:
        st.info("You need at least two filtered movies to generate recommendations.")
    else:
        seed_labels = {
            f"{m['title']} ({m['year'] if m['year'] else 'Unknown'})": m["id"]
            for m in filtered_movies
        }
        selected_seed_label = st.selectbox(
            "Choose a movie",
            list(seed_labels.keys()),
            key="choose_movie_recommendation"
        )
        seed_id = seed_labels[selected_seed_label]

        if st.button("Generate Recommendations"):
            recs = recommend_for_seed(seed_id, filtered_movies, approach=search_mode, top_n=8)
            render_recommendations(recs, context="recommendations")


# ---------------------------
# TAB 3: WATCHLIST
# ---------------------------
with tab3:
    st.subheader("My Watchlist")

    if not st.session_state.watchlist:
        st.info("Add movies to your watchlist first.")
    else:
        remove_title = st.selectbox(
            "Remove one movie",
            ["None"] + [movie["title"] for movie in st.session_state.watchlist],
            key="remove_watch_tab"
        )

        col_a, col_b = st.columns(2)
        with col_a:
            if remove_title != "None" and st.button("Remove Selected", key="remove_selected_tab"):
                st.session_state.watchlist = [m for m in st.session_state.watchlist if m["title"] != remove_title]
                st.rerun()
        with col_b:
            if st.button("Clear Watchlist", key="clear_watch_tab"):
                st.session_state.watchlist = []
                st.rerun()

        st.markdown("---")
        st.write("### Saved Movies")

        for saved in st.session_state.watchlist:
            saved_movie = next((m for m in movie_pool if m["id"] == saved["id"]), None)

            if saved_movie:
                render_movie_card(saved_movie, show_watchlist_button=False, context=f"saved_{saved['id']}")
            else:
                st.write(f"**{saved['title']} ({saved['year']})**")

        st.markdown("---")
        st.subheader("Watchlist Profile Recommendations")

        watchlist_ids = [movie["id"] for movie in st.session_state.watchlist]
        watchlist_recs = recommend_from_watchlist_profile(watchlist_ids, movie_pool, top_n=8)

        if not watchlist_recs:
            st.info("Not enough watchlist overlap with the current pool yet. Try a broader search or fewer filters.")
        else:
            st.write("These are based on the overall profile of your saved movies.")
            render_recommendations(watchlist_recs, context="watchlist_profile")


# ---------------------------
# TAB 4: INSIGHTS
# ---------------------------
with tab4:
    st.subheader("Insights and Comparison")

    if filtered_movies:
        df = pd.DataFrame(filtered_movies)

        ratings_fig = px.histogram(
            df,
            x="rating",
            nbins=12,
            title="Ratings Distribution"
        )
        st.plotly_chart(ratings_fig, use_container_width=True)

        genre_rows = []
        for movie in filtered_movies:
            for g in movie["genres"]:
                genre_rows.append({"genre": g})
        if genre_rows:
            genre_df = pd.DataFrame(genre_rows)
            top_genres = genre_df["genre"].value_counts().reset_index()
            top_genres.columns = ["Genre", "Count"]
            genre_fig = px.bar(
                top_genres.head(10),
                x="Genre",
                y="Count",
                title="Top Genres in Current Results"
            )
            st.plotly_chart(genre_fig, use_container_width=True)

        compare_pool = {f"{m['title']} ({m['year'] if m['year'] else 'Unknown'})": m for m in filtered_movies}
        if len(compare_pool) >= 2:
            c1, c2 = st.columns(2)
            with c1:
                label_a = st.selectbox("Movie A", list(compare_pool.keys()), key="compare_a")
            with c2:
                label_b = st.selectbox("Movie B", list(compare_pool.keys()), index=min(1, len(compare_pool) - 1), key="compare_b")

            movie_a = compare_pool[label_a]
            movie_b = compare_pool[label_b]

            comparison_df = build_comparison_table(movie_a, movie_b, language_dict)
            st.dataframe(comparison_df, use_container_width=True, hide_index=True)

            compare_chart_df = pd.DataFrame(
                {
                    "Metric": ["Rating", "Runtime", "Vote Count", "Box Office"],
                    movie_a["title"]: [movie_a["rating"], movie_a["runtime"], movie_a["vote_count"], movie_a["box_office"]],
                    movie_b["title"]: [movie_b["rating"], movie_b["runtime"], movie_b["vote_count"], movie_b["box_office"]],
                }
            )
            compare_chart_long = compare_chart_df.melt(id_vars="Metric", var_name="Movie", value_name="Value")
            compare_fig = px.bar(compare_chart_long, x="Metric", y="Value", color="Movie", barmode="group", title="Movie Comparison")
            st.plotly_chart(compare_fig, use_container_width=True)
    else:
        st.info("No filtered results available yet for insights or comparison.")


# ---------------------------
# TAB 5: TRENDING
# ---------------------------
with tab5:
    st.subheader("Trending Movies")
    trending_raw = get_trending_movies()
    trending_ids = [item["id"] for item in trending_raw[:10] if item.get("id")]

    trending_movies = []
    for movie_id in trending_ids:
        try:
            trending_movies.append(get_movie_metadata(movie_id))
        except Exception:
            continue

    if not trending_movies:
        st.info("Trending content is unavailable right now.")
    else:
        for movie in trending_movies[:8]:
            render_movie_card(movie, show_watchlist_button=True, context="trending")