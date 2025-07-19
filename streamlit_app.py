import streamlit as st
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# # ---------- PAGE SETUP: Dark/Light Mode (native) -------------
# st.set_page_config(
#     page_title="Netflix Movie Recommender",
#     layout="wide",  # Use wide mode to take advantage of side space!
#     initial_sidebar_state="expanded"
# )
#
# # --- (Optional) Show theme toggle in sidebar ---
# theme_mode = st.sidebar.radio("Theme", ["Light", "Dark"], index=1)
# if theme_mode == "Dark":
#     st.markdown(
#         """
#         <style>
#             body, .stApp { background-color: #181818; color: #eee; }
#             .st-cd, .st-b2, .st-cc { background: #242424; }
#         </style>
#         """, unsafe_allow_html=True
#     )

# ----------- DATA LOADING AND PREP ------------
@st.cache_data
def load_data():
    df = pd.read_csv("mymoviedb.csv", lineterminator="\n")
    df['Release_Date'] = pd.to_datetime(df['Release_Date'], errors='coerce')
    df['Release_Year'] = df['Release_Date'].dt.year
    df['Vote_Label'] = pd.qcut(df['Vote_Average'], 4, labels=['Not Popular', 'Below Average', 'Average', 'Popular'])
    df['Genre'] = df['Genre'].str.split(', ')
    df = df.explode('Genre').reset_index(drop=True)
    df['Genre'] = df['Genre'].str.strip()
    df['Genre'] = df['Genre'].astype('category')
    return df

@st.cache_data
def load_base_and_exploded():
    df_base = pd.read_csv("mymoviedb.csv", lineterminator="\n")
    df_base['Release_Date'] = pd.to_datetime(df_base['Release_Date'], errors='coerce')
    df_base['Release_Year'] = df_base['Release_Date'].dt.year
    df_base['Genre'] = df_base['Genre'].str.split(', ')
    df_base['Genre'] = df_base['Genre'].apply(lambda x: [g.strip() for g in x])
    df_base['Overview'] = df_base['Overview'].fillna('')
    df1 = pd.read_csv("mymoviedb.csv", lineterminator="\n")
    df1['Release_Date'] = pd.to_datetime(df1['Release_Date'], errors='coerce')
    df1['Release_Year'] = df1['Release_Date'].dt.year
    df1['Genre'] = df1['Genre'].str.split(', ')
    df1 = df1.explode('Genre').reset_index(drop=True)
    df1['Genre'] = df1['Genre'].str.strip()
    return df_base, df1

df = load_data()
df_base, df1 = load_base_and_exploded()

# ----------- SIDEBAR FILTERS ------------
st.sidebar.header("🔍 Filter Movies")

min_year = int(df['Release_Year'].min())
max_year = int(df['Release_Year'].max())
year_range = st.sidebar.slider("Release Year", min_year, max_year, (min_year, max_year), step=1)

all_genres = sorted(list(df['Genre'].cat.categories))
genre_options = ["All"] + all_genres
default_genres = ["All"]
selected_genres = st.sidebar.multiselect("Genre(s)", genre_options, default=default_genres)
if "All" in selected_genres:
    filtered_genres = all_genres
else:
    filtered_genres = selected_genres

vote_min, vote_max = float(df['Vote_Average'].min()), float(df['Vote_Average'].max())
vote_range = st.sidebar.slider("Vote Average", vote_min, vote_max, (vote_min, vote_max), step=0.1)

pop_min, pop_max = float(df['Popularity'].min()), float(df['Popularity'].max())
pop_range = st.sidebar.slider("Popularity", pop_min, pop_max, (pop_min, pop_max))

# ----------- FILTERED MOVIES FOR SELECTBOX ------------
filtered_df = df[
    (df['Release_Year'] >= year_range[0]) &
    (df['Release_Year'] <= year_range[1]) &
    (df['Genre'].isin(filtered_genres)) &
    (df['Vote_Average'] >= vote_range[0]) &
    (df['Vote_Average'] <= vote_range[1]) &
    (df['Popularity'] >= pop_range[0]) &
    (df['Popularity'] <= pop_range[1])
]
titles_genre = df['Title'].drop_duplicates().tolist()
titles_overview = df_base['Title'].tolist()
filtered_titles = sorted(set(filtered_df['Title']) & set(titles_overview))

# ----------- GENRE-BASED RECOMMENDER SETUP ------------
mlb = MultiLabelBinarizer()
genre_matrix = mlb.fit_transform(df.groupby('Title')['Genre'].apply(list))
genre_sim = cosine_similarity(genre_matrix, genre_matrix)
titles_genre = df['Title'].drop_duplicates().tolist()
movie_indices_genre = pd.Series(range(len(titles_genre)), index=titles_genre)

def recommend_movies_by_genre(title, top_n=5):
    if title not in movie_indices_genre:
        raise ValueError(f"Movie title '{title}' not found in dataset. Check spelling/case.")
    idx = movie_indices_genre[title]
    sim_row = genre_sim[idx]
    sim_scores = list(enumerate(sim_row))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]
    movie_indices_list = [i[0] for i in sim_scores]
    recommended_titles = [titles_genre[i] for i in movie_indices_list]
    recommendations = df[df['Title'].isin(recommended_titles)][['Title', 'Genre', 'Vote_Average', 'Popularity', 'Release_Year', 'Poster_Url']]
    recommendations = recommendations.drop_duplicates('Title')
    recommendations = recommendations.set_index('Title').loc[recommended_titles].reset_index()
    return recommendations

# ----------- OVERVIEW-BASED RECOMMENDER SETUP ------------
tfidf = TfidfVectorizer(stop_words='english')
overview_matrix = tfidf.fit_transform(df_base['Overview'])
overview_sim = cosine_similarity(overview_matrix, overview_matrix)
titles_overview = df_base['Title'].tolist()
movie_indices_overview = pd.Series(range(len(titles_overview)), index=titles_overview)

def recommend_movies_by_overview(title, top_n=5):
    idx = movie_indices_overview[title]
    sim_row = overview_sim[idx]
    sim_scores = list(enumerate(sim_row))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]
    movie_indices_list = [i[0] for i in sim_scores]
    recommended_titles = [titles_overview[i] for i in movie_indices_list]
    recommendations = df1[df1['Title'].isin(recommended_titles)][['Title', 'Genre', 'Vote_Average', 'Popularity', 'Release_Year', 'Poster_Url']]
    recommendations = recommendations.drop_duplicates('Title')
    recommendations = recommendations.set_index('Title').loc[recommended_titles].reset_index()
    return recommendations

# ----------------- MAIN PAGE LAYOUT ----------------

# ------- TOP ROW: App Title and Summary Dashboard (right panel) -------
st.markdown("""
    <style>
    /* Override Streamlit's block container max-width and padding */
    .block-container {
        max-width: 1400px !important;    /* Make it much wider */
        padding-top: 2rem !important;    /* Less vertical gap */
        padding-bottom: 2rem !important;
        padding-left: 2.5rem !important; /* Tweak as needed */
        padding-right: 2.5rem !important;
    }
    </style>
""", unsafe_allow_html=True)
colL, colR = st.columns([2, 1])
with colL:
    st.title("🎬 Netflix Movie Recommendation System")
    st.caption("Built with Streamlit · Content-based recommender (Genre & Overview)")

    # -------- FEATURED/TRENDING MOVIES POSTERS BLOCK (Put here!) ---------
    st.markdown(
        f"<h3 style='margin-bottom:0.6em; color:#ff4b4b'>🔥 Featured & Trending Movies</h3>",
        unsafe_allow_html=True
    )
    N_FEATURED = 4
    TOP_POOL = 30
    trending_pool = df.sort_values("Popularity", ascending=False).drop_duplicates("Title").head(TOP_POOL)
    sampled_trending = trending_pool.sample(n=N_FEATURED, random_state=None)
    cols = st.columns(N_FEATURED)
    for i, (_, row) in enumerate(sampled_trending.iterrows()):
        with cols[i]:
            if pd.notnull(row['Poster_Url']) and row['Poster_Url']:
                st.image(row['Poster_Url'], use_container_width=True, caption=row['Title'])
            st.markdown(
                f"**{row['Title']}** ({int(row['Release_Year']) if pd.notnull(row['Release_Year']) else 'N/A'})",
                unsafe_allow_html=True)
            # Genres
            genre_text = ", ".join(row['Genre']) if isinstance(row['Genre'], list) else str(row['Genre'])
            st.markdown(f"<span style='color:#ff4b4b;font-size:0.97em'>{genre_text}</span>", unsafe_allow_html=True)
            st.caption(f"⭐️ {row['Vote_Average']} | 🔥 {row['Popularity']:.1f}")

    st.markdown(
        """
        <style>
        .stImage img { transition: transform .2s; }
        .stImage img:hover { transform: scale(1.09); box-shadow: 0 0 14px #ff4b4b55; }
        </style>
        """, unsafe_allow_html=True
    )
stats_df = filtered_df if not filtered_df.empty else df

# Calculate Quick Stats
total_movies = stats_df['Title'].nunique()
all_genres_list = stats_df['Genre'].dropna().tolist()
n_genres = len(set(all_genres_list))

from collections import Counter
genre_counts = Counter(all_genres_list)
top3 = genre_counts.most_common(3)
top_genres = [g for g, _ in top3] + ['–'] * (3 - len(genre_counts))
top_genre_counts = [c for _, c in top3] + ['–'] * (3 - len(genre_counts))

min_year = int(stats_df['Release_Year'].min())
max_year = int(stats_df['Release_Year'].max())

# if theme_mode == "Dark":
card_bg = "#23272f"
card_text = "#fafafa"
card_shadow = "0 2px 12px 0 rgba(20,20,20,.3)"
# else:
#     card_bg = "#fff"
#     card_text = "#18181b"
#     card_shadow = "0 2px 12px 0 rgba(20,20,20,.06)"



with colR:
    st.markdown("""
            <style>
            .quick-stats-card {
                background: #23272f;
                color: #fafafa;
                border-radius: 24px;
                box-shadow: 0 2px 12px 0 rgba(20,20,20,.3);
                padding: 25px 25px 24px 25px;
                margin-top : 150px;
                margin-bottom: 20px;
                min-width: 320px;
                max-width: 420px;
                transition: 
                    box-shadow 0.25s, 
                    transform 0.18s,
                    background 0.18s;
            }
            .quick-stats-card:hover {
                box-shadow: 0 8px 24px 0 rgba(255,76,76,0.25), 0 2px 12px 0 rgba(20,20,20,.45);
                background: #272f3c;
                transform: scale(1.025) translateY(-2px);
            }
            </style>
        """, unsafe_allow_html=True)
    st.markdown(
        f"""
            <div class="quick-stats-card">
                <h3 style="margin-top:0">
                    <span style="font-size: 1.7em;">👨‍💻</span> <span style="vertical-align: middle;">Quick Stats</span>
                </h3>
                <p style="font-size: 1.6em; margin: 12px 0 0 0;">Total Movies<br><b>{total_movies:,}</b></p>
                <p style="font-size: 1.2em; margin: 12px 0 0 0;">Unique Genres<br><b>{n_genres}</b></p>
                <p style="margin-top: 24px; font-weight: bold;">Top Genres:</p>
                <ul>
                    <li>{top_genres[0]} ({top_genre_counts[0]})</li>
                    <li>{top_genres[1]} ({top_genre_counts[1]})</li>
                    <li>{top_genres[2]} ({top_genre_counts[2]})</li>
                </ul>
                <span style="font-size:0.95em;">Year Range: {min_year} - {max_year}</span>
            </div>
            """,
        unsafe_allow_html=True,
    )


# --- Let user pick a movie from filtered list
st.markdown("""
<style>
/* Target Streamlit selectbox component */
div[data-baseweb="select"] {
    transition: box-shadow 0.2s, border-color 0.2s, transform 0.13s;
    border-radius: 8px !important;
}
div[data-baseweb="select"]:hover {
    box-shadow: 0 0 8px 0 rgba(255,76,76,0.25), 0 2px 8px 0 rgba(20,20,20,.18);
    border: 1.8px solid #ff4b4b !important;
    transform: scale(1.022);
    background: #22252c14;
}
</style>
""", unsafe_allow_html=True)
st.markdown("""
    <style>
    .stSelectbox > div[data-baseweb="select"] { margin-top: -20px !important; }
    </style>
""", unsafe_allow_html=True)

# Then render the label and selectbox
st.markdown(
    "<div style='font-size:2em; font-weight:700; color:#ff4b4b; margin-bottom:0.1em;'>Select a movie</div>",
    unsafe_allow_html=True
)
selected_movie = st.selectbox("", filtered_titles, index=0 if filtered_titles else None)

# --- Extra toggles for details ---
# st.sidebar.markdown("---")
show_details = st.sidebar.checkbox("Show all movie details", value=True)

# -- Style for the label and to collapse the gap above/below radio buttons --
st.markdown("""
    <style>
    .fancy-radio-label {
        font-size: 1.45em;
        font-weight: 800;
        color: #ff4b4b;
        margin-bottom: -3.6em !important;    /* Tighter! */
        margin-top: 0 !important;
        line-height: 1.1 !important;
        padding-bottom: 0 !important;
    }
    /* Remove Streamlit radio group top margin and padding */
    div[data-baseweb="radio"] > div:first-child {
        margin-top: -1.2em !important;
        padding-top: 0 !important;
    }
    /* Optional: Slightly tighten gap below radios */
    div[data-baseweb="radio"] {
        margin-bottom: 0.1em !important;
    }
    </style>
""", unsafe_allow_html=True)

# Render the label and radio
st.markdown('<div class="fancy-radio-label">How would you like to get recommendations?</div>', unsafe_allow_html=True)

rec_type = st.radio(
    label="",
    options=["Genre Similarity", "Overview Similarity"],
    horizontal=True,
    key="rec_type_fancy"
)


st.markdown(
    '''
    <div style="font-size:1.7em; font-weight:800; color:#ff4b4b; margin-bottom:-30px;">
        Number of Recommendations
    </div>
    ''',
    unsafe_allow_html=True
)
top_n = st.slider("", 2, 10, 5)



st.markdown("""
<style>
/* --- Selectbox hover --- */
div[data-baseweb="select"] {
    transition: box-shadow 0.2s, border-color 0.2s, transform 0.13s;
    border-radius: 8px !important;
}
div[data-baseweb="select"]:hover {
    box-shadow: 0 0 8px 0 rgba(255,76,76,0.25), 0 2px 8px 0 rgba(20,20,20,.18);
    border: 1.8px solid #ff4b4b !important;
    transform: scale(1.022);
    background: #22252c14;
}

/* --- Recommend Button hover effect (robust) --- */
.stButton > button {
    background: linear-gradient(90deg,#ff4b4b 60%, #ff884b 120%) !important;
    border: none !important;
    color: #fff !important;
    border-radius: 7px !important;
    box-shadow: 0 2px 16px 0 rgba(255,76,76,0.09);
    transition: box-shadow 0.18s, transform 0.12s, background 0.22s;
    font-weight: 600;
    font-size: 1.09em;
    padding: 0.60em 2em;
}
.stButton > button:hover {
    box-shadow: 0 0 18px 0 rgba(255,76,76,0.18), 0 2px 18px 0 rgba(20,20,20,.22);
    background: linear-gradient(90deg,#ff884b 60%, #ff4b4b 120%) !important;
    transform: scale(1.035);
    color: #fff !important;
    border: none !important;
}
</style>
""", unsafe_allow_html=True)
if st.button("Recommend!"):
    with st.spinner("Finding your movies..."):
        try:
            if rec_type == "Genre Similarity":
                recs = recommend_movies_by_genre(selected_movie, top_n)
            else:
                recs = recommend_movies_by_overview(selected_movie, top_n)
            # Apply same sidebar filters
            recs = recs[
                (recs['Release_Year'] >= year_range[0]) &
                (recs['Release_Year'] <= year_range[1]) &
                (recs['Genre'].isin(filtered_genres)) &
                (recs['Vote_Average'] >= vote_range[0]) &
                (recs['Vote_Average'] <= vote_range[1]) &
                (recs['Popularity'] >= pop_range[0]) &
                (recs['Popularity'] <= pop_range[1])
            ]
            if recs.empty:
                st.info("No recommendations found for current filters. Try other settings.")
            else:
                st.success(f"Recommended movies based on **{rec_type}** for: {selected_movie}")
                for i, row in recs.iterrows():
                    # ------- COLLAPSIBLE MOVIE CARD -----
                    with st.expander(f"{row['Title']} ({int(row['Release_Year']) if pd.notnull(row['Release_Year']) else 'N/A'})", expanded=show_details):
                        cols = st.columns([1, 4])
                        with cols[0]:
                            poster_url = row['Poster_Url']
                            if pd.notnull(poster_url) and poster_url.strip():
                                st.image(poster_url, width=120)
                            else:
                                st.write("No image")
                        with cols[1]:
                            st.markdown(
                                f"**Genre:** {row['Genre']}  \n"
                                f"**Vote Avg:** {row['Vote_Average']} &nbsp;|&nbsp; **Popularity:** {row['Popularity']}"
                            )
        except Exception as e:
            st.error(f"Error: {e}")

st.markdown("""
---
*Tip: Use the sidebar to filter movies by year , rating , poularity and show/hide details. Click any movie title to expand or collapse details!*
""")
# st.markdown("""
#     <style>
#     /* Hide Streamlit's default menu and footer */
#     #MainMenu {visibility: hidden;}
#     footer {visibility: hidden;}
#     header {visibility: hidden;}
#     </style>
# """, unsafe_allow_html=True)
