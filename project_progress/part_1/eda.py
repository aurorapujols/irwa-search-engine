import os, sys
import numpy as np
import re
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from collections import Counter
from wordcloud import WordCloud

from dotenv import load_dotenv

try:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if repo_root not in sys.path:
        sys.path.append(repo_root)
except Exception as e:
    print(f"Could not adjust sys.path: {e}")

import nltk

# Setup
nltk.download('stopwords')
sns.set_style("whitegrid")

st.set_page_config(page_title="Product Data Dashboard", layout="wide")

# --- Sidebar ---
st.sidebar.title("⚙️ Controls")

load_dotenv()
json_path = os.getenv("DATA_FILE_PATH")

TOP = st.sidebar.slider("Top N words for WordCloud", 10, 100, 20)
bins_wordcount = st.sidebar.slider("Histogram bins (word count)", 10, 100, 20)
bins_sentence = st.sidebar.slider("Histogram bins (sentence length)", 10, 100, 30)

# --- Data Loading ---
@st.cache_data
def load_data(path):
    data = pd.read_json(path)
    return data

if json_path:
    data = load_data(json_path)
else:
    st.error("Please upload or configure a JSON data file.")
    st.stop()

# --- Helper functions ---
def build_terms(text):
    words = re.findall(r'\b\w+\b', str(text).lower())
    stop_words = set(nltk.corpus.stopwords.words('english'))
    return [w for w in words if w not in stop_words]

def get_avg_sentence_length(text):
    sentences = re.split(r'[.!?]+', str(text))
    sentences = [s.strip() for s in sentences if s.strip()]
    lengths = [len(s.split()) for s in sentences]
    return np.mean(lengths) if lengths else 0

def join_build_terms(texts):
    combined = " ".join(texts)
    return build_terms(combined)

# --- Data Preprocessing ---
titles = data['title'].apply(build_terms)
descriptions = data['description'].apply(build_terms)

titles_dist = titles.apply(len)
descriptions_dist = descriptions.apply(len)
avg_description_sentence_lengths = data['description'].apply(get_avg_sentence_length)

# --- Tabs ---
tabs = st.tabs(["📊 Overview", "📈 Distributions", "☁️ Word Cloud", "📦 Stock & Ratings"])

# --- Overview ---
with tabs[0]:
    st.header("Dataset Overview")
    st.write("### Sample Data")
    st.dataframe(data.head())
    st.write("### Word Count Statistics")
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Titles**")
        st.write(titles_dist.describe())
    with col2:
        st.write("**Descriptions**")
        st.write(descriptions_dist.describe())

# --- Distributions ---
with tabs[1]:
    st.header("Word Count & Sentence Length Distributions")
    fig, ax = plt.subplots(2, 1, figsize=(12, 10))
    
    sns.histplot(titles_dist, bins=bins_wordcount, color='skyblue', ax=ax[0])
    ax[0].axvline(titles_dist.mean(), linestyle='--', color='darkblue', label=f"Mean={titles_dist.mean():.2f}")
    ax[0].legend()
    ax[0].set_title("Title Word Count Distribution")

    sns.histplot(descriptions_dist, bins=bins_wordcount, color='gold', ax=ax[1])
    ax[1].axvline(descriptions_dist.mean(), linestyle='--', color='red', label=f"Mean={descriptions_dist.mean():.2f}")
    ax[1].legend()
    ax[1].set_title("Description Word Count Distribution")

    st.pyplot(fig)

    st.subheader("Average Sentence Length Distribution")
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    sns.histplot(avg_description_sentence_lengths, bins=bins_sentence, color='lightgreen', ax=ax2)
    ax2.axvline(avg_description_sentence_lengths.mean(), linestyle='--', color='red', label=f"Mean={avg_description_sentence_lengths.mean():.2f}")
    ax2.legend()
    st.pyplot(fig2)

# --- Word Cloud ---
with tabs[2]:
    st.header("Top Words WordCloud")
    words = join_build_terms([
        " ".join(data['title'].astype(str)),
        " ".join(data['description'].astype(str)),
        " ".join(data['brand'].astype(str)),
        " ".join(data['category'].astype(str)),
        " ".join(data['sub_category'].astype(str)),
        " ".join(data['seller'].astype(str))
    ])
    word_counts = Counter(words)
    top_words = word_counts.most_common(TOP)

    st.write(f"**Top {TOP} Words:**")
    st.table(pd.DataFrame(top_words, columns=["Word", "Count"]))

    wc = WordCloud(width=800, height=400, background_color="white", colormap="plasma").generate_from_frequencies(dict(top_words))
    fig_wc, ax_wc = plt.subplots(figsize=(12, 6))
    ax_wc.imshow(wc, interpolation='bilinear')
    ax_wc.axis("off")
    st.pyplot(fig_wc)

# --- Stock & Ratings ---
with tabs[3]:
    st.header("Stock and Ratings Insights")

    # --- Stock Distribution ---
    out_of_stock_dist = data['out_of_stock']
    st.write("### Stock Distribution")
    fig3, ax3 = plt.subplots()
    ax3.pie(out_of_stock_dist.value_counts(), labels=["In Stock", "Out of Stock"], autopct='%1.2f')
    ax3.set_title("Stock Availability")
    st.pyplot(fig3)

    # --- Preprocess numeric fields once ---
    @st.cache_data
    def preprocess_numeric_fields(df):
        df = df.copy()
        df['average_rating'] = df['average_rating'].replace('', '0').astype(float)
        df['actual_price'] = df['actual_price'].replace(r'[^\d.]', '', regex=True).replace('', '0').astype(float)
        df['selling_price'] = df['selling_price'].replace(r'[^\d.]', '', regex=True).replace('', '0').astype(float)
        df['discount'] = df['discount'].replace(r'[^\d.]', '', regex=True).replace('', '0').astype(float) / 100
        return df

    clean_data = preprocess_numeric_fields(data)

    # --- Precompute all sorted DataFrames once ---
    @st.cache_data
    def precompute_sorted_versions(df, top_n=20):
        sorted_dfs = {}
        sort_columns = ["average_rating", "actual_price", "discount", "selling_price"]
        for col in sort_columns:
            sorted_dfs[(col, "asc")] = df.sort_values(by=col, ascending=True).head(top_n)
            sorted_dfs[(col, "desc")] = df.sort_values(by=col, ascending=False).head(top_n)
        return sorted_dfs

    precomputed_rankings = precompute_sorted_versions(clean_data, TOP)

    # --- Sorting controls ---
    st.write("### Product Rankings")

    sort_option = st.selectbox(
        "Sort products by:",
        ["average_rating", "actual_price", "discount", "selling_price"]
    )

    sort_ascending = st.radio(
        "Sort order:",
        ["Descending (highest first)", "Ascending (lowest first)"],
        horizontal=True
    )

    order_key = "asc" if "Ascending" in sort_ascending else "desc"
    ranked = precomputed_rankings[(sort_option, order_key)]

    # --- Display only relevant columns ---
    st.dataframe(ranked[["pid", "title", sort_option, "url"]])



# --- Footer ---
st.markdown("---")
st.caption("Built with ❤️ using Streamlit and Python")
