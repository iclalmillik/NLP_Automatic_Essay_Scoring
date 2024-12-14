import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk import pos_tag
from textstat import textstat
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import language_tool_python
import tensorflow as tf

# NLTK veri yolunu ayarlayın
nltk.data.path.append('C:\\Users\\ZEHRA\\Desktop\\zf4\\nltk')

# Gerekli NLTK verilerinin mevcut olup olmadığını kontrol et ve indir
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

# Duygu analizi için eğitimli bir Keras modeli 
def duygu_analizi_modeli_olustur(vocab_size, embedding_dim, input_length):
    model = Sequential([
        Embedding(vocab_size, embedding_dim, input_length=input_length),
        Bidirectional(LSTM(64, return_sequences=True)),
        Bidirectional(LSTM(32)),
        Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Metin uzunluklarının hesaplanması
def uzunluklari_hesapla(text):
    kelimeler = word_tokenize(text)
    cumleler = sent_tokenize(text)
    kelime_sayisi = len(kelimeler)
    cumle_sayisi = len(cumleler)
    ortalama_kelime_uzunlugu = sum(len(kelime) for kelime in kelimeler) / kelime_sayisi
    ortalama_cumle_uzunlugu = kelime_sayisi / cumle_sayisi
    return kelime_sayisi, cumle_sayisi, ortalama_kelime_uzunlugu, ortalama_cumle_uzunlugu

# Dilbilgisi denetimi
def dilbilgisi_kontrolu(text):
    tool = language_tool_python.LanguageTool('en-US')
    eslesmeler = tool.check(text)
    dilbilgisi_hatalari = len(eslesmeler)
    return dilbilgisi_hatalari

# Leksik Metrikler
def leksik_metrikler(text):
    kelimeler = word_tokenize(text)
    benzersiz_kelimeler = set(kelimeler)
    ttr = len(benzersiz_kelimeler) / len(kelimeler)
    leksik_yogunluk = len([kelime for kelime in kelimeler if kelime.isalpha()]) / len(kelimeler)
    return ttr, leksik_yogunluk

# Okunabilirlik puanı
def okunabilirlik_puani(text):
    flesch_kincaid = textstat.flesch_kincaid_grade(text)
    gunning_fog = textstat.gunning_fog(text)
    smog_index = textstat.smog_index(text)
    coleman_liau = textstat.coleman_liau_index(text)
    
    puanlar = {
        'Flesch-Kincaid Seviye': flesch_kincaid,
        'Gunning Fog Indeksi': gunning_fog,
        'SMOG Indeksi': smog_index,
        'Coleman-Liau Indeksi': coleman_liau
    }
    
    return puanlar

# Konusal modelleme 
def konusal_modelleme(texts, n_topics=1):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf = vectorizer.fit_transform(texts)
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(tfidf)
    konular = lda.transform(tfidf)
    return konular

# Metin önişleme
def metin_onisleme(text, tokenizer, max_length):
    diziler = tokenizer.texts_to_sequences([text])
    padded_sequences = pad_sequences(diziler, maxlen=max_length, padding='post')
    return padded_sequences

# Olağanüstü kelime kullanımı
def olaganustu_kelime_kullanimi(text):
    kelimeler = word_tokenize(text)
    durak_kelimeler = set(stopwords.words('english'))
    filtrelenmis_kelimeler = [kelime for kelime in kelimeler if kelime.isalnum() and kelime.lower() not in durak_kelimeler]
    
    # Kelime çeşitliliği ve frekansı
    fdist = FreqDist(filtrelenmis_kelimeler)
    cesitli_kelimeler = len(fdist)
    toplam_kelimeler = len(filtrelenmis_kelimeler)
    cesitlilik_orani = cesitli_kelimeler / toplam_kelimeler if toplam_kelimeler > 0 else 0
    
    # Eylem kelimeleri ve değiştiriciler
    pos_etiketler = pos_tag(filtrelenmis_kelimeler)
    eylem_kelimeleri = [kelime for kelime, etiket in pos_etiketler if etiket.startswith('VB')]
    degistiriciler = [kelime for kelime, etiket in pos_etiketler if etiket in ['JJ', 'RB']]
    
    # Betimlemeler ve duyusal ayrıntılar
    betimleyici_kelimeler = len(eylem_kelimeleri) + len(degistiriciler)
    
    return cesitlilik_orani, betimleyici_kelimeler

# Olağanüstü yazma tekniği
def olaganustu_yazma_teknigi(text):
    cumleler = sent_tokenize(text)
    kelimeler = word_tokenize(text)
    
    # Akıcılık
    ortalama_cumle_uzunlugu = len(kelimeler) / len(cumleler) if len(cumleler) > 0 else 0
    
    # Cümle çeşitliliği
    pos_etiketler = [pos_tag(word_tokenize(cumle)) for cumle in cumleler]
    karmasik_cumleler = sum(1 for etiketler in pos_etiketler if any(etiket.startswith('VB') for kelime, etiket in etiketler) and any(etiket in ['IN', 'CC'] for kelime, etiket in etiketler))
    
    # Yazma teknikleri
    edebi_teknikler = sum(1 for cumle in cumleler if any(kelime in cumle.lower() for kelime in ['imagery', 'dialogue', 'humor', 'suspense']))
    
    return ortalama_cumle_uzunlugu, karmasik_cumleler, edebi_teknikler

'''# Göreve uygun kayıt, izleyici duygusu ve orijinal bakış açısını değerlendirme
def ton_ve_izleyici_analizi(text):
    # Göreve uygun kayıt (resmi, kişisel veya lehçe)
    ton_modeli = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    ton_gommesi = ton_modeli.encode([text], convert_to_tensor=True)
    
    # Güçlü bir izleyici duygusu
    izleyici_modeli = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    izleyici_gommesi = izleyici_modeli.encode([text], convert_to_tensor=True)
    
    # Orijinal bakış açısı
    orijinallik_modeli = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    orijinallik_gommesi = orijinallik_modeli.encode([text], convert_to_tensor=True)
    
    return ton_gommesi, izleyici_gommesi, orijinallik_gommesi
'''
# Ek özellikleri kontrol eden fonksiyonlar
def buyuk_harf_kullanimi_kontrolu(text):
    cumleler = sent_tokenize(text)
    dogru_buyuk_harf_kullanimi = sum(1 for cumle in cumleler if cumle[0].isupper())
    return dogru_buyuk_harf_kullanimi / len(cumleler) if cumleler else 0

def noktalama_isaretleri_kontrolu(text):
    kelimeler = word_tokenize(text)
    noktalama_isaretleri = [".", ",", "!", "?", ":", ";", "'", '"', "(", ")", "-", "—"]
    noktalama_sayisi = sum(1 for kelime in kelimeler if kelime in noktalama_isaretleri)
    return noktalama_sayisi / len(kelimeler) if kelimeler else 0

def yazim_hatalari_kontrolu(text):
    tool = language_tool_python.LanguageTool('en-US')
    eslesmeler = tool.check(text)
    yazim_hatalari = sum(1 for eslesme in eslesmeler if eslesme.ruleId == 'MORFOLOGIK_RULE_EN_US')
    return yazim_hatalari

def paragraf_yapisi_kontrolu(text):
    paragraflar = text.split('\n\n')
    yapili_paragraflar = sum(1 for paragraf in paragraflar if len(paragraf) > 0)
    return yapili_paragraflar / len(paragraflar) if paragraflar else 0

def cumle_yapisi_kontrolu(text):
    cumleler = sent_tokenize(text)
    tamamlanmis_cumleler = sum(1 for cumle in cumleler if cumle[-1] in ".!?")
    return tamamlanmis_cumleler / len(cumleler) if cumleler else 0

# Yazının genel analizi ve puanlama
def ozellikleri_cikar_ve_puanla(text, model, tokenizer, max_length):
    kelime_sayisi, cumle_sayisi, ortalama_kelime_uzunlugu, ortalama_cumle_uzunlugu = uzunluklari_hesapla(text)
    dilbilgisi_hatalari = dilbilgisi_kontrolu(text)
    ttr, leksik_yogunluk = leksik_metrikler(text)
    okunabilirlik = okunabilirlik_puani(text)
    konu = konusal_modelleme([text])
    cesitlilik_orani, betimleyici_kelimeler = olaganustu_kelime_kullanimi(text)
    ortalama_cumle_uzunlugu, karmasik_cumleler, edebi_teknikler = olaganustu_yazma_teknigi(text)
   # ton_gommesi, izleyici_gommesi, orijinallik_gommesi = ton_ve_izleyici_analizi(text)
    
    buyuk_harf_puani = buyuk_harf_kullanimi_kontrolu(text)
    noktalama_puani = noktalama_isaretleri_kontrolu(text)
    yazim_hatalari = yazim_hatalari_kontrolu(text)
    paragraf_yapisi_puani = paragraf_yapisi_kontrolu(text)
    cumle_yapisi_puani = cumle_yapisi_kontrolu(text)

    # Duygu analizi
    onislenmis_text = metin_onisleme(text, tokenizer, max_length)
    polarite = model.predict(onislenmis_text)[0][0]

    # Puanlama kuralları
    def puanlama_metodu(metric, thresholds):
        for i, threshold in enumerate(thresholds):
            if metric <= threshold:
                return i + 1
        return len(thresholds) + 1

    # Alan 1 (Yazma Uygulamaları) Nihai Puan Aralığı: 1-6
    # Alan 1 Kriterleri
    kelime_sayisi_puani = puanlama_metodu(kelime_sayisi, [50, 100, 150, 200, 250])
    ortalama_cumle_uzunlugu_puani = puanlama_metodu(ortalama_cumle_uzunlugu, [10, 15, 20, 25, 30])
    leksik_yogunluk_puani = puanlama_metodu(leksik_yogunluk, [0.2, 0.3, 0.4, 0.5, 0.6])
    betimleyici_kelimeler_puani = puanlama_metodu(betimleyici_kelimeler, [5, 10, 15, 20, 25])
    karmasik_cumleler_puani = puanlama_metodu(karmasik_cumleler, [1, 2, 3, 4, 5])
    edebi_teknikler_puani = puanlama_metodu(edebi_teknikler, [1, 2, 3, 4, 5])

    Alan1_puani = (
        kelime_sayisi_puani + 
        ortalama_cumle_uzunlugu_puani + 
        leksik_yogunluk_puani + 
        betimleyici_kelimeler_puani + 
        karmasik_cumleler_puani + 
        edebi_teknikler_puani
    ) / 6

    # Alan 2 (Dil Kuralları) Nihai Puan Aralığı: 1-4
    # Alan 2 Kriterleri
    dilbilgisi_hatalari_puani = puanlama_metodu(dilbilgisi_hatalari, [2, 4, 6, 8, 10])
    buyuk_harf_puani = puanlama_metodu(buyuk_harf_puani, [0.5, 0.6, 0.7, 0.8, 0.9])
    noktalama_puani = puanlama_metodu(noktalama_puani, [0.25, 0.2, 0.15, 0.1, 0.05])
    yazim_hatalari_puani = puanlama_metodu(yazim_hatalari, [2, 4, 6, 8, 10])
    cumle_yapisi_puani = puanlama_metodu(cumle_yapisi_puani, [0.5, 0.6, 0.7, 0.8, 0.9])

    Alan2_puani = (
        dilbilgisi_hatalari_puani + 
        buyuk_harf_puani + 
        noktalama_puani + 
        yazim_hatalari_puani + 
        cumle_yapisi_puani
    ) / 5

    # Alan 1 ve Alan 2 Puanları'nın ağırlıklı ortalaması
    final_puan = 0.6 * Alan1_puani + 0.4 * Alan2_puani
    final_puan = max(1, min(6, round(final_puan)))

    ozellikler = {
        'Kelime Sayisi': kelime_sayisi,
        'Cumle Sayisi': cumle_sayisi,
        'Ortalama Kelime Uzunlugu': ortalama_kelime_uzunlugu,
        'Ortalama Cumle Uzunlugu': ortalama_cumle_uzunlugu,
        'Dilbilgisi Hatalari': dilbilgisi_hatalari,
        'Polarite': polarite,
        'Kelime Turu Orani': ttr,
        'Leksik Yogunluk': leksik_yogunluk,
        'Okunabilirlik Puani': okunabilirlik,
        'Konu': konu,
        'Cesitlilik Orani': cesitlilik_orani,
        'Betimleyici Kelimeler': betimleyici_kelimeler,
        'Karmasik Cumleler': karmasik_cumleler,
        'Edebi Teknikler': edebi_teknikler,
        #'Ton Gommesi': ton_gommesi.numpy(),  # Tensor'u numpy'a dönüştürüyoruz
        #'Izleyici Gommesi': izleyici_gommesi.numpy(),  # Tensor'u numpy'a dönüştürüyoruz
        #'Orijinallik Gommesi': orijinallik_gommesi.numpy(),  # Tensor'u numpy'a dönüştürüyoruz
        'Buyuk Harf Kullanimi Puani': buyuk_harf_puani,
        'Noktalama Isaretleri Puani': noktalama_puani,
        'Yazim Hatalari': yazim_hatalari,
        'Paragraf Yapisi Puani': paragraf_yapisi_puani,
        'Cumle Yapisi Puani': cumle_yapisi_puani,
        'Alan1 Puani': Alan1_puani,
        'Alan2 Puani': Alan2_puani,
        'Nihai Puan': final_puan
    }
    
    return ozellikler

# Örnek metin
metin = """
I not agreed the people shuld d in the computer a lot and I going to said way. My first reason is because if you are a old man or your not going to d wit your kids and you going to have eyes problem so thas way you shuld in d in the computer a lot. My secon reason is the you going to have problem wit your wife if she found out the you are talking to another woman online because a lot of people they wife found out the they cant d wit other woman. My teard reason is the if you are a kid like me you not going to have fun and you not going to have friends because the computer problem came and go and not going to d wit your family and you lil sister out said playing wit her. So thas why you shulding d a lot in the computer.
"""

# Tokenizer ve modelin oluşturulması
vocab_size = 10000
embedding_dim = 100
max_length = 100

tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts([metin])
model = duygu_analizi_modeli_olustur(vocab_size, embedding_dim, max_length)

# Özelliklerin çıkarılması ve puanlama
ozellikler_ve_puan = ozellikleri_cikar_ve_puanla(metin, model, tokenizer, max_length)

# Sonuçları yazdırın
print("Yazı Özellikleri ve Puan:")
for ozellik, deger in ozellikler_ve_puan.items():
    print(f'{ozellik}: {deger}')