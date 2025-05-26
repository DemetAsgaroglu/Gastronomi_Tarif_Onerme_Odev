"""
Bu dosya, Word2Vec modellerini kullanarak giriş metnine göre,
veri setindeki metinlerin cosine similarity'sini hesaplar.
Her model için en benzer 5 metin detaylı olarak listelenir ve
sonuçlar lemmatized ve stemmed olarak iki gruba ayrılarak ortalama
benzerlik skorları ekrana yazdırılır ve görsel olarak kaydedilir.
"""

import os
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# Word2Vec model yolları
model_dir = "model"
model_paths = [os.path.join(model_dir, model) for model in os.listdir(model_dir) if model.endswith(".model")]

# Veri seti dosya yolları
lemmatized_content_path = "data/lemmatized.csv"
stemmed_content_path = "data/stemmed.csv"
lemmatized_content = pd.read_csv(lemmatized_content_path)
stemmed_content = pd.read_csv(stemmed_content_path)

# Giriş metni (bangiriş metni)
input_text_in = (
    "toss appl lemon juic larg bowl set asid pour water dutch oven medium heat combin sugar cornstarch cinnamon salt "
    "nutmeg bowl add water stir well bring boil boil minut stir constantli add appl return boil reduc heat cover "
    "simmer appl tender minut cool minut ladl freezer contain leav headspac cool room temperatur longer hour seal freez "
    "store month dotdash meredith food studio"
)

# Ortalama vektör hesaplama fonksiyonu
def calculate_average_vector(model, text):
    vectors = []
    for word in text.split():
        if word in model.wv:
            vectors.append(model.wv[word])
    if not vectors:
        return None
    return np.mean(vectors, axis=0)

# Veri setindeki her cümle için ortalama vektör hesaplama
def calculate_dataset_vectors(model, content):
    dataset_vectors = []
    for text in content["content"]:
        avg_vector = calculate_average_vector(model, text)
        if avg_vector is not None:
            dataset_vectors.append(avg_vector)
        else:
            dataset_vectors.append(np.zeros(model.vector_size))
    return np.array(dataset_vectors)

# Sonuçların toplanacağı listeler
lemmatized_results = []
stemmed_results = []

for model_path in model_paths:
    print(f"\nModel yükleniyor: {model_path}")
    model = Word2Vec.load(model_path)
    
    # Giriş metni için ortalama vektörü hesapla
    input_vector = calculate_average_vector(model, input_text_in)
    if input_vector is None:
        print(f"Giriş metni için modelde vektör bulunamadı: {model_path}")
        continue
    
    # Model türüne göre uygun veri setini seçelim.
    if "lemmatized" in model_path:
        content = lemmatized_content
    else:
        content = stemmed_content
        
    dataset_vectors = calculate_dataset_vectors(model, content)
    
    # Cosine similarity hesapla
    cosine_scores = cosine_similarity([input_vector], dataset_vectors).flatten()
    
    # Giriş metniyle tam eşleşen metinleri hariç tutarak en benzer 5 metni seçelim.
    valid_indices = [i for i in range(len(content)) if content.iloc[i]["content"] != input_text_in]
    if not valid_indices:
        print("Filtre sonrası geçerli metin kalmadı, model atlanıyor.")
        continue
    filtered_scores = [(i, cosine_scores[i]) for i in valid_indices]
    top_5 = sorted(filtered_scores, key=lambda x: x[1], reverse=True)[:5]
    
    # Detaylı listeleme:
    base_name = os.path.basename(model_path)
    print(f"\nModel: {base_name} - input_text_in için en benzer 5 metin:")
    for idx, score in top_5:
        # Varsayalım veri setinde 'document_id' sütunu mevcut,
        # aksi halde, örnek çıktı için alternatif olarak sabit değer kullanılabilir.
        doc_id = content.iloc[idx]["document_id"] if "document_id" in content.columns else f"doc{idx}"
        metin = content.iloc[idx]["content"]
        print(f"Metin ID: {doc_id}, Benzerlik Skoru: {score:.4f}")
        print(f"Metin: {metin}\n")
    
    # Ortalama benzerlik hesapla
    avg_similarity = np.mean([score for _, score in top_5])
    print(f"Model: {base_name} için ortalama top-5 benzerlik: {avg_similarity:.4f}\n")
    
    # Sonuçları gruplara ekleyelim:
    if "lemmatized" in base_name:
        lemmatized_results.append({"Model": base_name, "Ortalama Benzerlik": avg_similarity})
    else:
        stemmed_results.append({"Model": base_name, "Ortalama Benzerlik": avg_similarity})

# Görselleştirme için çıkış klasörü
output_dir = "gorsel/calculate_wordvec"
os.makedirs(output_dir, exist_ok=True)

# Lemmatized modeller için çubuk grafik
if lemmatized_results:
    lemmatized_df = pd.DataFrame(lemmatized_results)
    plt.figure(figsize=(8,6))
    plt.bar(lemmatized_df["Model"], lemmatized_df["Ortalama Benzerlik"], color="skyblue")
    plt.title("Lemmatized Modeller için Ortalama Top-5 Benzerlik", fontsize=14)
    plt.xlabel("Model")
    plt.ylabel("Ortalama Benzerlik")
    plt.ylim(0, 1)
    plt.tight_layout()
    output_path = os.path.join(output_dir, "lemmatized_models_similarity.png")
    plt.savefig(output_path)
    print(f"Lemmatized görsel kaydedildi: {output_path}")
    plt.close()
else:
    print("Lemmatized sonuç yok.")

# Stemmed modeller için çubuk grafik
if stemmed_results:
    stemmed_df = pd.DataFrame(stemmed_results)
    plt.figure(figsize=(8,6))
    plt.bar(stemmed_df["Model"], stemmed_df["Ortalama Benzerlik"], color="lightgreen")
    plt.title("Stemmed Modeller için Ortalama Top-5 Benzerlik", fontsize=14)
    plt.xlabel("Model")
    plt.ylabel("Ortalama Benzerlik")
    plt.ylim(0, 1)
    plt.tight_layout()
    output_path = os.path.join(output_dir, "stemmed_models_similarity.png")
    plt.savefig(output_path)
    print(f"Stemmed görsel kaydedildi: {output_path}")
    plt.close()
else:
    print("Stemmed sonuç yok.")
