"""
Bu dosya, hem lemmatized hem de stemmed TF-IDF çıktılarından cümleler arası cosine similarity matrisini hesaplar.
Bir giriş metni ile veri setindeki en benzer 5 metni sıralar.
"""

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
import matplotlib.pyplot as plt

# Dosya yolları
tfidf_lemma_path = "data/tfidf_lemmatized_results.csv"
tfidf_stem_path = "data/tfidf_stemmed_results.csv"
lemmatized_content_path = "data/lemmatized.csv"
stemmed_content_path = "data/stemmed.csv"

# Giriş metni
input_text = ("mix sugar cornstarch cinnamon nutmeg large pan add water salt mix "
              "well bring boil cook thick bubbly remove heat add lemon juice food coloring sterilize canning "
              "jar lid ring boiling large pot water peel core slice apple pack sliced apple hot canning jar "
              "leaving headspace fill jar hot syrup gently remove air bubble knife put lid process water bath canner minute"
              )

# TF-IDF matrislerini yükle
tfidf_lemma = pd.read_csv(tfidf_lemma_path)
tfidf_stem = pd.read_csv(tfidf_stem_path)

# Veri setindeki içerikleri yükle
lemmatized_content = pd.read_csv(lemmatized_content_path)
stemmed_content = pd.read_csv(stemmed_content_path)

# İçeriklerdeki tekrarları kaldır
lemmatized_content = lemmatized_content.drop_duplicates(subset="content")
stemmed_content = stemmed_content.drop_duplicates(subset="content")

# Giriş metnini TF-IDF vektörüne dönüştürmek için bir satır ekleyin
def get_tfidf_vector(input_text, tfidf_matrix):
    vector = np.zeros((1, tfidf_matrix.shape[1]))
    for word in input_text.split():
        if word in tfidf_matrix.columns:
            vector[0, tfidf_matrix.columns.get_loc(word)] = 1
    return vector

# Giriş metni için TF-IDF vektörlerini oluştur
input_vector_lemma = get_tfidf_vector(input_text, tfidf_lemma)
input_vector_stem = get_tfidf_vector(input_text, tfidf_stem)

# Cosine similarity hesapla
cosine_scores_lemma = cosine_similarity(input_vector_lemma, tfidf_lemma.values).flatten()
cosine_scores_stem = cosine_similarity(input_vector_stem, tfidf_stem.values).flatten()

# En benzer 5 metni bul
top_5_indices_lemma = cosine_scores_lemma.argsort()[-5:][::-1]
top_5_indices_stem = cosine_scores_stem.argsort()[-5:][::-1]

# Sonuçları yazdır
print("\nLemmatized için en benzer 5 metin:")
for idx in top_5_indices_lemma:
    print(f"Metin ID: {lemmatized_content.iloc[idx]['document_id']}, Benzerlik Skoru: {cosine_scores_lemma[idx]:.4f}")
    print(f"Metin: {lemmatized_content.iloc[idx]['content']}\n")

print("\nStemmed için en benzer 5 metin:")
for idx in top_5_indices_stem:
    print(f"Metin ID: {stemmed_content.iloc[idx]['document_id']}, Benzerlik Skoru: {cosine_scores_stem[idx]:.4f}")
    print(f"Metin: {stemmed_content.iloc[idx]['content']}\n")

# Lemmatized sonuçları tabloya ekle
lemmatized_results = [
    {"Metin ID": "doc79", "Benzerlik Skoru": 0.8961},
    {"Metin ID": "doc17", "Benzerlik Skoru": 0.8961},
    {"Metin ID": "doc110", "Benzerlik Skoru": 0.8961},
    {"Metin ID": "doc48", "Benzerlik Skoru": 0.8961},
    {"Metin ID": "doc639", "Benzerlik Skoru": 0.4586},
]

# Stemmed sonuçları tabloya ekle
stemmed_results = [
    {"Metin ID": "doc110", "Benzerlik Skoru": 0.7819},
    {"Metin ID": "doc17", "Benzerlik Skoru": 0.7819},
    {"Metin ID": "doc48", "Benzerlik Skoru": 0.7819},
    {"Metin ID": "doc79", "Benzerlik Skoru": 0.7819},
    {"Metin ID": "doc348", "Benzerlik Skoru": 0.4385},
]

# DataFrame oluştur
lemmatized_df = pd.DataFrame(lemmatized_results)
stemmed_df = pd.DataFrame(stemmed_results)

# Tabloyu görselleştir ve kaydet
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

lemmatized_df.plot.bar(x="Metin ID", y="Benzerlik Skoru", ax=axes[0], color="skyblue", legend=False)
axes[0].set_title("Lemmatized Benzerlik Skorları")
axes[0].set_ylim(0, 1)

stemmed_df.plot.bar(x="Metin ID", y="Benzerlik Skoru", ax=axes[1], color="lightgreen", legend=False)
axes[1].set_title("Stemmed Benzerlik Skorları")
axes[1].set_ylim(0, 1)

plt.tight_layout()

# Görseli kaydetmek için klasör oluştur
output_dir = "gorsel/cosine"
os.makedirs(output_dir, exist_ok=True)

# Görseli kaydet
output_path = os.path.join(output_dir, "cosine_similarity.png")
plt.savefig(output_path)
print(f"Görsel kaydedildi: {output_path}")
