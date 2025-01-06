import streamlit as st
import pandas as pd
import asyncio
import aiohttp
from bs4 import BeautifulSoup
from textblob import TextBlob
import matplotlib.pyplot as plt
from io import BytesIO
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm
import nltk

# Setup NLTK
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

# Fungsi scraping
async def fetch(session, url):
    async with session.get(url) as response:
        return await response.text()

async def scrape_reviews(base_url):
    headers = {'User-Agent': 'Mozilla/5.0'}
    reviews = []
    async with aiohttp.ClientSession(headers=headers) as session:
        content = await fetch(session, base_url)
        soup = BeautifulSoup(content, 'html.parser')
        total_reviews = int(soup.find('div', {'data-testid': 'tturv-total-reviews'}).text.split()[0].replace(',', ''))

        page, review_urls = 1, []
        while len(review_urls) < total_reviews:
            content = await fetch(session, f"{base_url}?paginationKey={page}")
            soup = BeautifulSoup(content, 'html.parser')
            review_cards = soup.find_all('article', class_='sc-d99cd751-1 kzUfxa user-review-item')
            if not review_cards:
                break

            for card in review_cards:
                if len(review_urls) >= total_reviews:
                    break
                try:
                    user = card.find('ul', class_='ipc-inline-list').li.text.strip()
                    date = card.find('li', class_='ipc-inline-list__item review-date').text.strip()
                    link = card.find('a', class_='ipc-title-link-wrapper').get('href')
                    review_urls.append({'user': user, 'date': date, 'url': f"https://www.imdb.com{link}"})
                except:
                    continue

            page += 1
            await asyncio.sleep(1)

        tasks = [fetch(session, item['url']) for item in review_urls]
        reviews_text = await asyncio.gather(*tasks)

        for i, item in enumerate(review_urls):
            soup = BeautifulSoup(reviews_text[i], 'html.parser')
            review = soup.find('div', {'class': 'text show-more__control'})
            reviews.append({'id_user': item['user'], 'date': item['date'], 'review': review.text.strip() if review else "Review not found"})

    return pd.DataFrame(reviews)

# Fungsi preprocessing
def decontracted(phrase):
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase

def preprocess_text(text_data):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    preprocessed_text = []

    for sentence in tqdm(text_data, desc="Processing Text"):
        sentence = decontracted(sentence)
        sentence = re.sub(r"[^A-Za-z0-9]+", " ", sentence)
        sentence = re.sub(r"\b\w{1,2}\b", " ", sentence)
        sentence = re.sub(r"\d+", "", sentence)

        tokens = word_tokenize(sentence.lower())
        tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
        preprocessed_text.append(" ".join(tokens).strip())

    return preprocessed_text

def process_scraped_data(scraped):
    scraped['Processed_Review'] = preprocess_text(scraped['review'].dropna())
    return scraped

# Fungsi labeling sentiment menggunakan Textblob
def labeling_sentiment(data, text_column='Processed_Review'):
    def classify_review(review):
        polarity = TextBlob(review).sentiment.polarity
        label = 'Positive' if polarity > 0.1 else 'Negative' if polarity < -0.1 else 'Neutral'
        return label, polarity

    data[['Sentiment_Label', 'Polarity']] = data[text_column].apply(lambda x: pd.Series(classify_review(x)))
    return data

# Fungsi visualisasi pie chart
def create_pie_chart(data):
    sentiment_counts = data['Sentiment_Label'].value_counts()
    fig, ax = plt.subplots()
    ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=90, colors=['#66c2a5', '#fc8d62', '#8da0cb'])
    ax.axis('equal')
    return fig

# Streamlit App
st.title("Analisis Sentiment Review Film Pada IMDb")

url = st.text_input("Masukkan link URL review IMDb")
if st.button("Mulai Analisis"):
    if url:
        with st.spinner("Tunggu ya, data masih di scraping..."):
            try:
                scraped_reviews = asyncio.run(scrape_reviews(url))
                if len(scraped_reviews) == 0:
                    st.error("Tidak ada ulasan ditemukan. Pastikan URL yang dimasukkan benar.")
                else:
                    data = pd.DataFrame(scraped_reviews)
                    processed_data = process_scraped_data(data)
                    labeled_data = labeling_sentiment(processed_data)

                    st.success("Analisis komplit!")
                    st.write("Berikut adalah hasil tabel data:")
                    st.dataframe(labeled_data)

                    # Tombol unduh tabel
                    csv_data = labeled_data.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Unduh Tabel sebagai CSV",
                        data=csv_data,
                        file_name="hasil_analisis.csv",
                        mime="text/csv",
                    )

                    # Visualisasi pie chart
                    st.write("Visualisasi distribusi sentimen:")
                    pie_chart = create_pie_chart(labeled_data)
                    st.pyplot(pie_chart)

                    # Tombol unduh gambar pie chart
                    buf = BytesIO()
                    pie_chart.savefig(buf, format="png")
                    buf.seek(0)
                    st.download_button(
                        label="Unduh Pie Chart sebagai PNG",
                        data=buf,
                        file_name="pie_chart.png",
                        mime="image/png",
                    )
            except Exception as e:
                st.error(f"Terjadi kesalahan: {e}")
    else:
        st.warning("Masukkan URL terlebih dahulu.")
