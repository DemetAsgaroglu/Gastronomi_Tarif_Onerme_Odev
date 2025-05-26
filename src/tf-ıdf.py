import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import time  # Eğitim süresini ölçmek için time modülü
import matplotlib.pyplot as plt

# Veri setini yükle (temizlenmiş ve işlenmiş veriler)
df = pd.read_csv("data/recipes_cleaned.csv")


# Lemmatize edilmiş ve Stemlenmiş metinlerden TF-IDF hesaplama
def calculate_tfidf(df, column_name):
    vectorizer = TfidfVectorizer(stop_words='english')

    # Model eğitim süresi ölçümü
    start_time = time.time()

    X = vectorizer.fit_transform(df[column_name])

    # Eğitim süresini hesapla
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"TF-IDF hesaplama süresi ({column_name}): {elapsed_time:.4f} saniye")

    return X, vectorizer.get_feature_names_out(), elapsed_time


# Lemmatize edilmiş metinler üzerinde TF-IDF hesaplama
X_lemmatized, features_lemmatized, lemmatized_time = calculate_tfidf(df, 'lemmatized_directions')

# Stemlenmiş metinler üzerinde TF-IDF hesaplama
X_stemmed, features_stemmed, stemmed_time = calculate_tfidf(df, 'stemmed_directions')

# Sonuçları CSV dosyasına kaydetme
output_dir = "data"
os.makedirs(output_dir, exist_ok=True)

# Lemmatize edilmiş verilerin TF-IDF sonuçlarını DataFrame olarak kaydedelim
tfidf_df_lemmatized = pd.DataFrame(X_lemmatized.toarray(), columns=features_lemmatized)
tfidf_df_lemmatized.to_csv(os.path.join(output_dir, "tfidf_lemmatized_results.csv"), index=False)

# Stemlenmiş verilerin TF-IDF sonuçlarını DataFrame olarak kaydedelim
tfidf_df_stemmed = pd.DataFrame(X_stemmed.toarray(), columns=features_stemmed)
tfidf_df_stemmed.to_csv(os.path.join(output_dir, "tfidf_stemmed_results.csv"), index=False)

# Eğitim sürelerini yazdıralım
print(f"Lemmatize edilmiş veriler için TF-IDF hesaplama süresi: {lemmatized_time:.4f} saniye")
print(f"Stemlenmiş veriler için TF-IDF hesaplama süresi: {stemmed_time:.4f} saniye")


# 10 en yüksek TF-IDF kelimesini terminale yazdırma ve görseli kaydetme
def print_top_10_and_plot(tfidf_df, features, name):
    # Kelimelerin toplam TF-IDF değerlerini hesapla
    tfidf_sum = tfidf_df.sum(axis=0)
    top_10_words = tfidf_sum.sort_values(ascending=False).head(10)

    # Terminale en yüksek 10 kelimeyi yazdır
    print(f"\nEn yüksek 10 kelime ({name}):")
    print(top_10_words)

    # Görseli kaydetmek için klasör oluştur
    output_dir_visuals = "gorsel/tf-ıdf"
    os.makedirs(output_dir_visuals, exist_ok=True)

    # Bar grafik çizimi
    plt.figure(figsize=(10, 6))
    top_10_words.plot(kind='bar', color='salmon')
    plt.title(f"En Yüksek 10 TF-IDF Kelimesi ({name})")
    plt.xlabel('Kelime')
    plt.ylabel('TF-IDF Değeri')
    plt.xticks(rotation=45, ha='right')

    # Görseli kaydet
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir_visuals, f"top_10_tfidf_{name}.png"))
    plt.close()  # Görseli kapat


# Lemmatize edilmiş veriler için en yüksek 10 kelimeyi yazdır ve görseli kaydet
print_top_10_and_plot(tfidf_df_lemmatized, features_lemmatized, "lemmatized")

# Stemlenmiş veriler için en yüksek 10 kelimeyi yazdır ve görseli kaydet
print_top_10_and_plot(tfidf_df_stemmed, features_stemmed, "stemmed")

print("Lemmatize edilmiş ve Stemlenmiş veriler için TF-IDF sonuçları başarıyla kaydedildi.")
