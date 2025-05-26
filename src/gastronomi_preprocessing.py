import pandas as pd
import re
import matplotlib.pyplot as plt
import numpy as np
import os
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import nltk

# Veri setini yükle
df = pd.read_csv("data/veriseti/recipes.csv")
df.info()

# Gerekli NLTK modüllerini indir
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Stopwords ve lemmatizer, stemmer'ı başlat
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

# Gereksiz sütunları çıkartma
columns_to_keep = [
    'recipe_name', 'prep_time', 'cook_time', 'total_time', 'servings',
    'ingredients', 'directions'
]
df = df[columns_to_keep]

# Birimler listesi
units = ['cups', 'cup', 'tablespoons', 'tablespoon', 'tbsp', 'teaspoons', 'teaspoon', 'tsp',
         'kg', 'grams', 'gram', 'g', 'liters', 'liter', 'ml', 'l', 'pounds', 'pound', 'ounces',
         'ounce', 'inches', 'inch', '½', '⅓', '¼', '⅔', '¾']

# Sıcaklık birimleri (°C, °F gibi)
temperature_units = ['°C', 'C', '°F', 'F', 'Fahrenheit', 'Celsius', 'degrees F', 'degrees C']

# Sayı, birim ve sıcaklıkları tespit etmek için regex desenleri
quantity_pattern = r'(\d+(\.\d+)?)|([¼½⅓⅔¾⅔⅝]+|[0-9]+[\/\d]+)'  # Sayıları ve kesirleri yakalar
unit_pattern = r'(\b(?:' + '|'.join(units) + r')\b)'  # Birimleri yakalar
temperature_pattern = r'(\d{1,3}(\.\d+)?)\s*(°?[CF]|Fahrenheit|Celsius|degrees?\s?[CF])'  # Sıcaklıkları yakalar

# Sayısal değerleri, birimleri ve sıcaklıkları ayıran fonksiyon
def extract_quantities_units_and_temperatures(directions):
    quantities = []
    units_found = []
    temperatures = []

    # Her bir cümleye bakarak sayısal değerleri, birimleri ve sıcaklıkları ayır
    for direction in directions:
        # Sayıları ve kesirleri yakalama
        found_quantities = re.findall(quantity_pattern, direction)
        quantities.extend([item[0] for item in found_quantities])

        # Birimleri yakalama
        found_units = re.findall(unit_pattern, direction)
        units_found.extend(found_units)

        # Sıcaklıkları yakalama
        found_temperatures = re.findall(temperature_pattern, direction)
        for temp in found_temperatures:
            temp_value, unit = temp[0], temp[2]
            temperatures.append(f"{temp_value}{unit}")

    return quantities, units_found, temperatures

# 'directions' sütununu işlemeyi başlatıyoruz
df['directions'] = df['directions'].apply(lambda x: x.split('.') if isinstance(x, str) else [])

# 'quantities', 'units' ve 'temperatures' sütunlarını oluştur
df['quantities'], df['units'], df['temperatures'] = zip(
    *df['directions'].apply(extract_quantities_units_and_temperatures))

# İlk 10 satırı yazdır
print("\n--- İlk 10 Satır ---")
print(df[['quantities', 'units', 'temperatures']].head(10))

# directions sütununu tekrar birleştir (nokta ile)
df['directions'] = df['directions'].apply(lambda x: '.'.join(x).strip())

# Sonucu recipes.csv'ye tekrar kaydet
df.to_csv("data/veriseti/recipes.csv", index=False)

print("Sayısal veriler ve sıcaklık bilgileri recipes.csv dosyasına kaydedildi.")

# Sayısal değerleri, birimleri ve sıcaklıkları ayıran fonksiyon
def remove_quantities_units_and_temperatures(directions):
    # Sayıları, birimleri ve sıcaklıkları silmek için regex kullan
    directions = re.sub(quantity_pattern, '', directions)
    directions = re.sub(unit_pattern, '', directions)
    directions = re.sub(temperature_pattern, '', directions)

    # Fazla boşlukları temizle
    directions = re.sub(r'\s+', ' ', directions)  # Birden fazla boşluğu tek boşluk yap
    directions = directions.strip()  # Baş ve sonlardaki boşlukları kaldır

    return directions


# Kelimeleri tokenleştirip, lemmatize etme ve stemleme
def preprocess_sentence(sentence):
    # Cümleyi kelimelere ayır
    tokens = word_tokenize(sentence)
    # Sadece harf olan kelimeleri al ve stopword'leri çıkar
    filtered_tokens = [token.lower() for token in tokens if token.isalpha() and token.lower() not in stop_words]
    # Lemmatize etme
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
    # Stemleme
    stemmed_tokens = [stemmer.stem(token) for token in filtered_tokens]

    return lemmatized_tokens, stemmed_tokens


# Tariflerin 'directions' sütununu işleme
def clean_and_process_directions(df):
    # Sayısal ifadeleri, birimleri ve sıcaklıkları temizle
    df['cleaned_directions'] = df['directions'].apply(remove_quantities_units_and_temperatures)

    # Lemmatize ve Stemleme işlemlerini yap
    tokenized_corpus_lemmatized = []
    tokenized_corpus_stemmed = []
    for sentence in df['cleaned_directions']:
        lemmatized_tokens, stemmed_tokens = preprocess_sentence(sentence)
        tokenized_corpus_lemmatized.append(lemmatized_tokens)
        tokenized_corpus_stemmed.append(stemmed_tokens)

    # Yeni sütunları ekleyelim
    df['lemmatized_directions'] = tokenized_corpus_lemmatized
    df['stemmed_directions'] = tokenized_corpus_stemmed

    return df

# 'directions' sütununda işlem yapalım
df = clean_and_process_directions(df)

# Temizlenmiş ve işlenmiş metinleri görmek için ilk 5 satır
print(df[['directions', 'cleaned_directions', 'lemmatized_directions', 'stemmed_directions']].head())

# Temizlenmiş veriyi kaydet
output_dir = "data"
os.makedirs(output_dir, exist_ok=True)

df.to_csv(os.path.join(output_dir, "recipes_cleaned.csv"), index=False)

print("Temizlenmiş veri kaydedildi.")

# Temizlenmiş ve işlenmiş metinleri görmek için ilk 5 satırı yazdıralım
for index, row in df.head().iterrows():
    print(f"Original: {row['directions']}")
    print(f"Cleaned: {row['cleaned_directions']}")
    print(f"Lemmatized: {row['lemmatized_directions']}")
    print(f"Stemmed: {row['stemmed_directions']}")
    print("-" * 80)

# Temizlenmiş veriyi yeni formatta kaydet: dokuman_id ve tarifler (lemmatized ve stemmed)
df_lemmatized = pd.DataFrame({
    "dokuman_id": df.index,
    "tarifler": df['lemmatized_directions'].apply(lambda x: ' '.join(x))
})

df_stemmed = pd.DataFrame({
    "dokuman_id": df.index,
    "tarifler": df['stemmed_directions'].apply(lambda x: ' '.join(x))
})

output_dir = "data"
os.makedirs(output_dir, exist_ok=True)

df_lemmatized.to_csv(os.path.join(output_dir, "recipes_lemmatized.csv"), index=False)
df_stemmed.to_csv(os.path.join(output_dir, "recipes_stemmed.csv"), index=False)

print("Lemmatized ve stemmed tarifler dokuman_id ile birlikte kaydedildi.")


# Temel veri setini tekrar oku
df = pd.read_csv("data/recipes_cleaned.csv")

# Listeyi düz metne çevir: ['heat', 'butter'] → 'heat butter'
df["document_id"] = ["doc" + str(i+1) for i in range(len(df))]
df["content"] = df["stemmed_directions"].apply(lambda x: ' '.join(eval(x)) if isinstance(x, str) else "")

# Sadece gerekli sütunlar
stemmed_df = df[["document_id", "content"]]
stemmed_df.to_csv("data/stemmed.csv", index=False)

# Liste içeriğini düz metne çevir (örneğin: ['heat', 'butter'] → 'heat butter')
df["document_id"] = ["doc" + str(i+1) for i in range(len(df))]
df["content"] = df["lemmatized_directions"].apply(lambda x: ' '.join(eval(x)) if isinstance(x, str) else "")

# Sadece gerekli sütunları al ve kaydet
lemmatized_df = df[["document_id", "content"]]
lemmatized_df.to_csv("data/lemmatized.csv", index=False)

