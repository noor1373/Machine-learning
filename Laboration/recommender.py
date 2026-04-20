import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

_movies = None
_tfidf_matrix = None
_model = None


def load_data(movies_path='../data/movies.csv', tags_path='../data/tags.csv'):
    """Läser in och förbehandlar movies.csv och tags.csv."""
    try:
        movies = pd.read_csv(movies_path)
        tags = pd.read_csv(tags_path)
        print(f'Datasetet laddat: {len(movies)} filmer hittades.')
    except FileNotFoundError:
        print('Fel: Hittade inte movies.csv eller tags.csv.')
        raise

    movies['genre_text'] = movies['genres'].str.replace('|', '', regex=False)

    tags['tag'] = tags['tag'].astype('str').str.lower().str.strip()
    tags_grouped = (
        tags.groupby('movieId')['tag']
        .apply(lambda x: ' '.join(x.unique()))
        .reset_index()
        .rename(columns={'tag': 'tag_text'})
    )

    movies = movies.merge(tags_grouped, on='movieId', how='left')
    movies['tag_text'] = movies['tag_text'].fillna('')
    movies['combined_text'] = movies['genre_text'] + ' ' + movies['tag_text']
    return movies


def build_model(movies):
    """Bygger TF-IDF-matris och tränar en NearestNeighbors-modell."""
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies['combined_text'])

    model = NearestNeighbors(metric='cosine', algorithm='brute')
    model.fit(tfidf_matrix)
    return tfidf_matrix, model


def initialize(movies_path='../data/movies.csv', tags_path='../data/tags.csv'):
    """Laddar data och bygger modellen. Måste anropas före get_recommendations()."""
    global _movies, _tfidf_matrix, _model
    _movies = load_data(movies_path, tags_path)
    _tfidf_matrix, _model = build_model(_movies)
    print('Modellen är kalr.')


def get_recommendations(movie_title, n=5):
    """Returnerar de n mest liknande filmerna för en given titel."""
    if _movies is None or _model is None:
        raise RuntimeError('Anropa initialize() innan get_recommendations().')
    
    matches = _movies[_movies['title'].str.contains(movie_title, case=False, na=False, regex=False)]
    if matches.empty:
        print(f'Ingen film hittades för "{movie_title}".')
        return None
    
    index = matches.index[0]
    distances, indices = _model.kneighbors(_tfidf_matrix[index], n_neighbors=n + 1)

    return (
        _movies.iloc[indices[0][1:]][['title', 'genres']]
        .assign(
            somilarity=1 - distances[0][1:],
            genres=lambda df: df['genres'].str.replace('|', ', ', regex=False)

        )
        .reset_index(drop=True)
    )