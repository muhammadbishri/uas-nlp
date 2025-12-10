import sys
import os
sys.path.append(os.path.dirname(__file__))

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
import plotly.express as px
import plotly.graph_objects as go
import io

from preprocessing import preprocess_reviews, get_statistics, get_sentiment_statistics

st.set_page_config(
    page_title="Analisis Topik Ulasan E-commerce",
    layout="wide"
)

st.title("Analisis Topik Ulasan Produk E-commerce")
st.markdown("**Preprocessing Text menggunakan NLP**")
st.divider()

# ===========================
# SIDEBAR
# ===========================
with st.sidebar:
    st.header("Pengaturan")
    input_method = st.radio(
        "Pilih metode input:",
        ["Upload CSV", "Input Manual"]
    )
    
    st.divider()
    st.info("**Dibuat oleh:** Muhammad Bishri Annas\n\n**Mata Kuliah:** NLP")

# ===========================
# INPUT DATA
# ===========================
if input_method == "Upload CSV":
    st.subheader("Upload File CSV")
    
    uploaded_file = st.file_uploader(
        "Pilih file CSV",
        type=['csv']
    )
    
    df = None
    reviews = None

    if uploaded_file:
        try:
            content = uploaded_file.getvalue().decode("utf-8", errors="ignore")

            df = pd.read_csv(
                io.StringIO(content),
                engine="python",
                on_bad_lines="skip"
            )

            df = df.fillna("-")

            st.success(f"Berhasil memuat dataset dengan {len(df)} baris.")
            st.write("Preview dataset:")
            st.dataframe(df.head(), use_container_width=True)

            # ======================================
            # AUTO-DETECT kolom review
            # ======================================
            possible_review_cols = [
                "review", "reviewContent", "content", "comment",
                "ulasan", "text", "review_text"
            ]
            review_col = None
            for col in possible_review_cols:
                if col in df.columns:
                    review_col = col
                    break

            if review_col is None:
                st.error("Tidak ditemukan kolom ulasan! Gunakan salah satu nama: review, content, comment, ulasan, reviewContent.")
                st.stop()

            reviews = df[review_col].astype(str).tolist()

        except Exception as e:
            st.error(f"Error membaca file: {str(e)}")

else:
    # INPUT MANUAL
    st.subheader("Input Ulasan Manual")
    
    num_reviews = st.number_input(
        "Jumlah ulasan:",
        min_value=1,
        max_value=10,
        value=3
    )
    
    reviews = []
    for i in range(num_reviews):
        review = st.text_area(
            f"Ulasan {i+1}:",
            key=f"review_{i}",
            height=80
        )
        if review:
            reviews.append(review)


# ===========================
# PROSES NLP
# ===========================
if reviews and len(reviews) > 0:
    
    st.divider()
    
    if st.button("Proses Data", type="primary"):

        with st.spinner("Memproses data..."):
            results = preprocess_reviews(reviews)
            stats = get_statistics(results)
            sentiment_stats = get_sentiment_statistics(results)

        st.success("Preprocessing selesai!")

        # ===========================
        # TABS
        # ===========================
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Statistik",
            "Detail Preprocessing",
            "Word Cloud",
            "Analisis Topik",
            "Sentiment Analysis"
        ])

        # ===========================
        # TAB 1: STATISTIK
        # ===========================
        with tab1:
            st.subheader("Statistik Preprocessing")

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Ulasan", stats['total_reviews'])
            with col2:
                st.metric("Rata-rata Token Awal", stats['avg_tokens_original'])
            with col3:
                st.metric("Rata-rata Token Akhir", stats['avg_tokens_final'])
            with col4:
                st.metric("Efisiensi", f"{stats['reduction_rate']}%")

            comparison_data = []
            for i, result in enumerate(results):
                comparison_data.append({
                    "Ulasan": f"Ulasan {i+1}",
                    "Original": len(result["tokens"]),
                    "Filtered": len(result["filtered"]),
                    "Stemmed": len(result["stemmed"])
                })

            df_comparison = pd.DataFrame(comparison_data)

            fig = px.bar(
                df_comparison,
                x="Ulasan",
                y=["Original", "Filtered", "Stemmed"],
                title="Perbandingan Jumlah Token per Tahap",
                barmode="group"
            )
            st.plotly_chart(fig, use_container_width=True)

        # ===========================
        # TAB 2: DETAIL
        # ===========================
        with tab2:
            st.subheader("Detail Preprocessing")

            for i, result in enumerate(results):
                with st.expander(f"Ulasan {i+1}"):

                    st.write("### 1. Original")
                    st.info(result["original"])

                    st.write("### 2. Tokenization")
                    st.code(result["tokens"])

                    st.write("### 3. Stopword Removal")
                    st.code(result["filtered"])

                    st.write("### 4. Stemming")
                    st.code(result["stemmed"])

                    st.write("### 5. Final Text")
                    st.success(result["final_text"])

        # ===========================
        # TAB 3: WORD CLOUD
        # ===========================
        with tab3:
            st.subheader("Word Cloud")

            all_words = []
            for r in results:
                all_words.extend(r["stemmed"])

            if all_words:
                wordcloud = WordCloud(
                    width=800,
                    height=400,
                    background_color="white"
                ).generate(" ".join(all_words))

                fig, ax = plt.subplots(figsize=(12, 6))
                ax.imshow(wordcloud, interpolation="bilinear")
                ax.axis("off")
                st.pyplot(fig)

        # ===========================
        # TAB 4: ANALISIS TOPIK
        # ===========================
        with tab4:
            st.subheader("Analisis Topik")

            all_words = []
            for r in results:
                all_words.extend(r["stemmed"])

            freq = Counter(all_words).most_common(20)

            df_freq = pd.DataFrame(freq, columns=["Kata", "Frekuensi"])
            st.dataframe(df_freq, use_container_width=True)

        # ===========================
        # TAB 5: SENTIMENT + TABEL LENGKAP
        # ===========================
        with tab5:
            st.subheader("Sentiment Analysis")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Positif", sentiment_stats["positive"])
            with col2:
                st.metric("Netral", sentiment_stats["neutral"])
            with col3:
                st.metric("Negatif", sentiment_stats["negative"])

            st.divider()
            st.subheader("Tabel Lengkap (Username + Score + Review + Preprocessing + Sentiment)")

            # deteksi kolom userName
            possible_username_cols = ["userName", "username", "author", "user"]
            username_col = None
            if "df" in locals():
                for col in possible_username_cols:
                    if col in df.columns:
                        username_col = col
                        break

            # deteksi kolom score
            possible_score_cols = ["score", "rating", "stars"]
            score_col = None
            if "df" in locals():
                for col in possible_score_cols:
                    if col in df.columns:
                        score_col = col
                        break

            # deteksi kolom review lagi
            review_col_final = None
            if "df" in locals():
                for col in ["review", "reviewContent", "content", "comment", "ulasan", "text"]:
                    if col in df.columns:
                        review_col_final = col
                        break

            # build table output
            df_output = pd.DataFrame({
                "Username": df[username_col] if username_col else "-",
                "Score": df[score_col] if score_col else "-",
                "Review Original": df[review_col_final] if review_col_final else reviews,
                "Case Folding": [r["final_text"] for r in results],
                "Sentiment Score": [r["sentiment_score"] for r in results],
                "Sentiment Label": [r["sentiment_label"] for r in results]
            })

            st.dataframe(df_output, use_container_width=True)

            csv_output = df_output.to_csv(index=False)
            st.download_button(
                "Download Tabel Sentiment (CSV)",
                data=csv_output,
                file_name="sentiment_table.csv",
                mime="text/csv"
            )

else:
    st.info("Silakan upload file CSV atau input ulasan manual terlebih dahulu.")
