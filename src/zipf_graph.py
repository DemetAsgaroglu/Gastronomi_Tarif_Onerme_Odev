import pandas as pd
import os
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np


# Zipf grafiği çizen fonksiyon
def plot_zipf(word_freq, title, filename):
    sorted_freq = sorted(word_freq.values(), reverse=True)
    ranks = np.arange(1, len(sorted_freq) + 1)
    plt.figure(figsize=(10, 6))
    plt.loglog(ranks, sorted_freq, marker=".")
    plt.title(f"Zipf Plot - {title}")
    plt.xlabel("Kelime Sıralaması (Rank)")
    plt.ylabel("Frekans")
    plt.grid(True)

    os.makedirs("Zipf Yasası", exist_ok=True)
    plt.savefig(f"Zipf Yasası/{filename}.png")
    plt.close()


# Ham veri
raw_df = pd.read_csv("data/veriseti/recipes.csv")
raw_words = " ".join(raw_df["directions"].dropna()).split()
raw_freq = Counter(raw_words)
plot_zipf(raw_freq, "Ham Veri", "zipf_raw")

# Cleaned veri
cleaned_df = pd.read_csv("data/recipes_cleaned.csv")

# Lemmatized
if "lemmatized_directions" in cleaned_df.columns:
    lemma_words = " ".join(cleaned_df["lemmatized_directions"].dropna()).split()
    lemma_freq = Counter(lemma_words)
    plot_zipf(lemma_freq, "Lemmatize Edilmiş Veri", "zipf_lemma")
else:
    print("HATA: 'lemmatized_directions' sütunu yok.")

# Stemmed
if "stemmed_directions" in cleaned_df.columns:
    stem_words = " ".join(cleaned_df["stemmed_directions"].dropna()).split()
    stem_freq = Counter(stem_words)
    plot_zipf(stem_freq, "Stem Edilmiş Veri", "zipf_stem")
else:
    print("HATA: 'stemmed_directions' sütunu yok.")
