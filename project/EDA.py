import pandas as pd
import os
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud


output_dir = "figures"
os.makedirs(output_dir, exist_ok=True)


df = pd.read_csv(r"reviews_preprocessed.csv")

print("Nombre total de reviews :", len(df))
print("Colonnes disponibles :", df.columns.tolist())
print("\nDistribution des sentiments :")
print(df['sentiment'].value_counts())

df_original = pd.read_csv(r"archive\phone_user_review_file_1.csv", encoding='latin1')
print("\nExemples de reviews brutes :")
print(df_original[['extract']].sample(5, random_state=42))

print("\nExemples de reviews nettoyées :")
print(df[['cleaned_text', 'sentiment']].sample(5, random_state=42))


# 1. Word Cloud global
text = ' '.join(df['cleaned_text'])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

plt.figure(figsize=(15,7))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("Word Cloud global des reviews")
plt.savefig(os.path.join(output_dir, "wordcloud_global.png"))
plt.close()


# 2. Word Cloud par sentiment
for sentiment_label in ['positive', 'neutral', 'negative']:
    text_sentiment = ' '.join(df[df['sentiment'] == sentiment_label]['cleaned_text'])
    if len(text_sentiment.strip()) == 0:
        print(f"Pas de texte pour le sentiment '{sentiment_label}', skipping wordcloud.")
        continue
    wordcloud_sent = WordCloud(width=800, height=400, background_color='white').generate(text_sentiment)
    plt.figure(figsize=(15,7))
    plt.imshow(wordcloud_sent, interpolation='bilinear')
    plt.axis('off')
    plt.title(f"Word Cloud des reviews {sentiment_label}")
    plt.savefig(os.path.join(output_dir, f"wordcloud_{sentiment_label}.png"))
    plt.close()


# 3. Distribution des sentiments
plt.figure(figsize=(8,5))
sns.countplot(data=df, x='sentiment', order=['positive', 'neutral', 'negative'])
plt.title("Distribution des sentiments")
plt.savefig(os.path.join(output_dir, "distribution_sentiments.png"))
plt.close()


# 4. Top 20 mots fréquents
all_words = ' '.join(df['cleaned_text']).split()
word_freq = Counter(all_words)

most_common = word_freq.most_common(20)
words, counts = zip(*most_common)

plt.figure(figsize=(12,6))
sns.barplot(x=list(words), y=list(counts), palette="viridis")
plt.title("Top 20 mots les plus fréquents")
plt.xticks(rotation=45)
plt.savefig(os.path.join(output_dir, "top20_mots_frequents.png"))
plt.close()


# 5. Distribution de la longueur des reviews
df['review_length'] = df['cleaned_text'].apply(lambda x: len(x.split()))

plt.figure(figsize=(12,6))
sns.histplot(df['review_length'], bins=30, kde=True, color='skyblue')
plt.title("Distribution de la longueur des reviews (en mots)")
plt.xlabel("Nombre de mots")
plt.ylabel("Nombre de reviews")
plt.savefig(os.path.join(output_dir, "distribution_longueur_reviews.png"))
plt.close()

# 6. Top 20 marques les plus mentionnées
if 'product' in df.columns:
    top_brands = df['product'].value_counts().head(20)

    plt.figure(figsize=(12,6))
    sns.barplot(x=top_brands.values, y=top_brands.index, palette="coolwarm")
    plt.title("Top 20 des marques les plus mentionnées dans les reviews")
    plt.xlabel("Nombre d'avis")
    plt.ylabel("Marques")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "top20_marques.png"))
    plt.close()
else:
    print("La colonne 'product' n'existe pas dans le dataframe.")




print(f"\nLes graphes ont été sauvegardés dans le dossier '{output_dir}'.")
