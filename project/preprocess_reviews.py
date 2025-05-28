import pandas as pd
import glob
import nltk
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer

# Télécharger les ressources nécessaires de NLTK
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')
nltk.download('vader_lexicon')

# Étape 1 : Chargement et agrégation des fichiers CSV
# Récupère tous les fichiers de reviews dans le dossier "archive"
files = glob.glob("archive/phone_user_review_file_*.csv")
df = pd.concat([pd.read_csv(f, encoding="latin1") for f in files], ignore_index=True)

# Étape 2 : Filtrage des données
# Conserve uniquement les reviews rédigées en anglais et provenant des États-Unis
df = df[(df['lang'] == 'en') & (df['country'] == 'us')]

# Étape 3 : Prétraitement textuel
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess(text):
    """Nettoie et lemmatise un texte brut."""
    if not isinstance(text, str):
        return ''
    text = text.lower()
    text = text.replace('"', '').replace("'", '')
    text = re.sub(r'<.*?>|http\S+|www\S+', ' ', text)  # Supprime HTML et URLs
    text = re.sub(r'[^a-z\s]', ' ', text)              # Supprime les caractères non-alphabétiques
    words = text.split()
    # Lemmatise les mots, supprime les stopwords et les mots trop courts
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words and len(w) > 2]
    return ' '.join(words)

# Applique le nettoyage aux extraits textuels
df['cleaned_text'] = df['extract'].apply(preprocess)

# Supprime les avis trop courts (moins de 4 mots)
df = df[df['cleaned_text'].str.split().str.len() >= 4]

# Étape 4 : Analyse de sentiment
sia = SentimentIntensityAnalyzer()

def get_sentiment(text):
    # Retourne le label de sentiment basé sur le score compound de VADER
    score = sia.polarity_scores(text)['compound']
    if score >= 0.05:
        return 'positive'
    elif score <= -0.05:
        return 'negative'
    else:
        return 'neutral'

# Applique le modèle VADER à chaque texte nettoyé
df['sentiment'] = df['cleaned_text'].apply(get_sentiment)

# Étape 5 : Nettoyage des noms de produits
# Standardise les noms de produits pour éviter les doublons (ex : guillemets)
df['product'] = df['product'].str.strip('"').str.strip("'").str.lower()

# Étape 6 : Sauvegarde du jeu de données prétraité
# Conserve uniquement les colonnes essentielles pour la suite du pipeline
df[['product', 'cleaned_text', 'sentiment']].to_csv("reviews_preprocessed.csv", index=False)

print("Fichier reviews_preprocessed.csv généré avec succès.")
