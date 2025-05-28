from flask import Flask, request, render_template
import pandas as pd
import re
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

app = Flask(__name__)


# Définition d'une liste étendue de mots à ignorer (stopwords)

# Cette liste est adaptée spécifiquement aux avis produits (reviews)
# et contient des mots fréquents mais peu informatifs pour l'analyse de sujets
base_words_to_remove = {
    # Mots fréquents peu informatifs
    'amazon', 'amazing', 'arrived', 'awesome', 'back', 'bad', 'best', 'block', 'bought', 'buy',
    'came', 'come', 'contacted', 'customer', 'day', 'delivered', 'device', 'even', 'ever', 'excellent',
    'fantastic', 'five', 'four', 'get', 'good', 'got', 'great', 'hat', 'hour', 'issue', 'item', 'just',
    'later', 'like', 'minute', 'month', 'more', 'much', 'new', 'nice', 'object', 'only', 'order',
    'ordered', 'perfect', 'phone', 'problem', 'product', 'purchase', 'purchased', 'rating', 'really',
    'refund', 'return', 'review', 'sent', 'series', 'seller', 'service', 'shipping', 'star', 'still',
    'stop', 'stopped', 'stuff', 'support', 'thank', 'thing', 'things', 'time', 'unit', 'use', 'used',
    'using', 'very', 'week', 'well', 'within', 'with', 'without', 'work', 'working', 'worst', 'year',
    'mother', 'give', 'would', 'daughter', 'love', 'christmas', 'thanks',
    
    # Formes contractées ou familières
    'dont', 'never', 'ive', 'say', 'said', 'one', 'also', 'lot', 'lots', 'make', 'makes', 'made',
    'go', 'going', 'went', 'right', 'first', 'second', 'last', 'days', 'better', 'old', 'try', 'trying',
    'tried', 'yes', 'no', 'ok', 'okay',
    
    # Marques (peu utiles pour l’analyse sémantique des sujets)
    'samsung', 'apple', 'google', 'iphone', 'asus', 'pixel', 'nexus', 'moto', 'galaxy', 'sony', 'lg',
    'huawei', 'xiaomi', 'lenovo', 'motorola', 'nokia', 'htc', 'dell', 'hp', 'acer', 'asus', 'kindle',

    # Divers
    'everything', 'ago', 'far', 'yet', 'could', 'two', 'many', 'take', 'taken', 'see', 'seen',
    'know', 'known', 'need', 'needed'
}


# Fonction de tokenisation simple : extrait les mots alphanumériques
def tokenize(text):
    if not isinstance(text, str):
        return []
    text = text.lower()
    words = re.findall(r'\b[a-z0-9]+\b', text)
    return words


# Chargement du modèle d'embedding pour BERTopic (optimisé pour rapidité & qualité)

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')


# Fonction d'extraction des sujets via BERTopic

# Elle entraîne un modèle BERTopic sur les textes filtrés par sentiment
# et retourne les mots-clés les plus représentatifs (en excluant les stopwords)
def extract_topics(df_sentiment, sentiment_label, words_to_remove):
    if df_sentiment.empty:
        return []

    # Initialisation du modèle BERTopic
    topic_model = BERTopic(
        language="english",
        embedding_model=embedding_model,
        min_topic_size=30,
        nr_topics=10  # On demande plus de topics pour pouvoir mieux filtrer
    )

    # Entraînement sur les textes
    topics, _ = topic_model.fit_transform(df_sentiment['cleaned_text'])

    # Récupération des informations sur tous les topics
    topic_info = topic_model.get_topic_info()

    # On ignore le topic -1 (outliers), puis on garde les 5 plus gros topics
    main_topics = topic_info[topic_info['Topic'] != -1].sort_values(by='Count', ascending=False).head(5)

    result = []
    for topic_id in main_topics['Topic']:
        keywords = topic_model.get_topic(topic_id)
        filtered = [word for word, _ in keywords if word not in words_to_remove]
        if len(filtered) > 1:
            result.append(filtered)

    # Affichage pour vérification
    print(f"\n--- {sentiment_label.upper()} TOPICS ---")
    for i, topic in enumerate(result):
        print(f"Topic {i+1}: {topic}")
    print("--------------------------\n")

    return result



# Route principale de l'application (GET & POST)

@app.route('/', methods=['GET', 'POST'])
def index():
    topics_pos, topics_neg = [], []

    if request.method == 'POST':
        keyword = request.form['keyword'].lower()

        # Vérifier que le fichier a bien été généré au préalable
        if not pd.io.common.file_exists("reviews_preprocessed.csv"):
            return "Erreur : reviews_preprocessed.csv non trouvé. Veuillez lancer preprocess_reviews.py."

        df = pd.read_csv("reviews_preprocessed.csv")

        # Filtrer les avis contenant le mot-clé produit
        df_filtered = df[df['product'].str.contains(keyword, case=False, na=False)]

        if df_filtered.empty:
            return f"Aucun avis trouvé pour le mot-clé : {keyword}"

        # Séparer les avis positifs et négatifs (max 300 pour limiter la charge)
        words_to_remove = set(base_words_to_remove)
        df_pos = df_filtered[df_filtered['sentiment'] == 'positive'].head(300)
        df_neg = df_filtered[df_filtered['sentiment'] == 'negative'].head(300)

        # Extraction des topics
        topics_pos = extract_topics(df_pos, "positive", words_to_remove)
        topics_neg = extract_topics(df_neg, "negative", words_to_remove)

    return render_template("index.html", topics_pos=topics_pos, topics_neg=topics_neg)


# Lancement de l'application Flask

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
