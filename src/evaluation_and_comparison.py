import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# --- 1. Model çıktıları (Word2Vec - Lemmatized ve Stemmed CBOW ve SkipGram) ---
model_top5 = {
    "lemmatized_cbow_w2_d100": {"doc517", "doc833", "doc452", "doc863", "doc773"},
    "lemmatized_cbow_w2_d300": {"doc517", "doc833", "doc452", "doc748", "doc555"},
    "lemmatized_cbow_w4_d100": {"doc833", "doc452", "doc863", "doc517", "doc773"},
    "lemmatized_cbow_w4_d300": {"doc452", "doc517", "doc863", "doc833", "doc1031"},
    "lemmatized_skipgram_w2_d100": {"doc5", "doc36", "doc67", "doc98", "doc453"},
    "lemmatized_skipgram_w2_d300": {"doc5", "doc36", "doc67", "doc98", "doc453"},
    "lemmatized_skipgram_w4_d100": {"doc5", "doc36", "doc67", "doc98", "doc453"},
    "lemmatized_skipgram_w4_d300": {"doc5", "doc36", "doc67", "doc98", "doc453"},
    "stemmed_cbow_w2_d100": {"doc453", "doc721", "doc317", "doc3", "doc34"},
    "stemmed_cbow_w2_d300": {"doc721", "doc317", "doc448", "doc453", "doc3"},
    "stemmed_cbow_w4_d100": {"doc721", "doc453", "doc519", "doc448", "doc317"},
    "stemmed_cbow_w4_d300": {"doc721", "doc453", "doc448", "doc519", "doc317"},
    "stemmed_skipgram_w2_d100": {"doc542", "doc869", "doc511", "doc1020", "doc400"},
    "stemmed_skipgram_w2_d300": {"doc869", "doc542", "doc400", "doc1020", "doc317"},
    "stemmed_skipgram_w4_d100": {"doc542", "doc511", "doc869", "doc1020", "doc458"},
    "stemmed_skipgram_w4_d300": {"doc542", "doc511", "doc1020", "doc869", "doc400"},
    # TF-IDF sonuçları
    "tfidf_lemmatized": {"doc172", "doc17", "doc203", "doc141", "doc747"},
    "tfidf_stemmed": {"doc203", "doc17", "doc141", "doc172", "doc444"},
}

# --- 2. Ortalama benzerlik skorları ---
avg_scores = {
    "lemmatized_cbow_w2_d100": 0.9986,
    "lemmatized_cbow_w2_d300": 0.9993,
    "lemmatized_cbow_w4_d100": 0.9987,
    "lemmatized_cbow_w4_d300": 0.9991,
    "lemmatized_skipgram_w2_d100": 0.9961,
    "lemmatized_skipgram_w2_d300": 0.9968,
    "lemmatized_skipgram_w4_d100": 0.9960,
    "lemmatized_skipgram_w4_d300": 0.9968,
    "stemmed_cbow_w2_d100": 0.9988,
    "stemmed_cbow_w2_d300": 0.9993,
    "stemmed_cbow_w4_d100": 0.9984,
    "stemmed_cbow_w4_d300": 0.9990,
    "stemmed_skipgram_w2_d100": 0.9902,
    "stemmed_skipgram_w2_d300": 0.9932,
    "stemmed_skipgram_w4_d100": 0.9886,
    "stemmed_skipgram_w4_d300": 0.9908,
    # TF-IDF sonuçları
    "tfidf_lemmatized": 0.8961,
    "tfidf_stemmed": 0.7819,
}

# --- 3. Anlamsal değerlendirme puanları ---
subjective_evals = {
    "lemmatized_cbow_w2_d100": [5, 5, 4, 4, 4],
    "lemmatized_cbow_w2_d300": [5, 5, 5, 4, 4],
    "lemmatized_cbow_w4_d100": [5, 5, 4, 4, 4],
    "lemmatized_cbow_w4_d300": [5, 5, 5, 4, 4],
    "lemmatized_skipgram_w2_d100": [4, 4, 4, 4, 3],
    "lemmatized_skipgram_w2_d300": [4, 4, 4, 4, 3],
    "lemmatized_skipgram_w4_d100": [4, 4, 4, 4, 3],
    "lemmatized_skipgram_w4_d300": [4, 4, 4, 4, 3],
    "stemmed_cbow_w2_d100": [5, 5, 4, 4, 4],
    "stemmed_cbow_w2_d300": [5, 5, 5, 4, 4],
    "stemmed_cbow_w4_d100": [4, 4, 4, 4, 3],
    "stemmed_cbow_w4_d300": [5, 5, 4, 4, 4],
    "stemmed_skipgram_w2_d100": [3, 3, 3, 3, 2],
    "stemmed_skipgram_w2_d300": [4, 4, 4, 4, 3],
    "stemmed_skipgram_w4_d100": [3, 3, 3, 3, 2],
    "stemmed_skipgram_w4_d300": [4, 4, 4, 4, 3],
    # TF-IDF sonuçları
    "tfidf_lemmatized": [4, 4, 4, 4, 3],
    "tfidf_stemmed": [3, 3, 3, 3, 2],
}

# --- 4. Ortalama anlamsal skor hesaplamaları ---
subjective_avg = {k: np.mean(v) for k, v in subjective_evals.items()}
df_subjective = pd.DataFrame(subjective_avg.items(), columns=["Model", "Anlamsal Skor"])
df_similarity = pd.DataFrame(avg_scores.items(), columns=["Model", "Benzerlik Skoru"])

# --- 5. Jaccard benzerlik matrisi ---
def jaccard(a, b):
    return len(a & b) / len(a | b) if a and b else 0.0

models = list(model_top5.keys())
jaccard_matrix = pd.DataFrame(index=models, columns=models)

for m1 in models:
    for m2 in models:
        jaccard_matrix.loc[m1, m2] = round(jaccard(model_top5[m1], model_top5[m2]), 2)

# --- 6. Görselleştirme ve kayıt ---
os.makedirs("rapor_cikti", exist_ok=True)

plt.figure(figsize=(12, 10))
plt.imshow(jaccard_matrix.astype(float), cmap="YlGnBu")
plt.colorbar(label="Jaccard Skoru")
plt.xticks(range(len(models)), models, rotation=45, ha="right")
plt.yticks(range(len(models)), models)
plt.title("Jaccard Benzerlik Matrisi - Word2Vec ve TF-IDF Modelleri")
plt.tight_layout()
plt.savefig("rapor_cikti/jaccard_matrix_word2vec_tfidf_models.png")
plt.close()

# CSV çıktılar
df_subjective.to_csv("rapor_cikti/anlamsal_skorlar_word2vec_tfidf_models.csv", index=False)
df_similarity.to_csv("rapor_cikti/benzerlik_skorlari_word2vec_tfidf_models.csv", index=False)
jaccard_matrix.to_csv("rapor_cikti/jaccard_matrix_word2vec_tfidf_models.csv")


print("Tüm modeller için analizler başarıyla kaydedildi.")

