import string
import re
import emoji
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

slang_dict = {
    "kaga": "tidak",
    "ngga": "tidak",
    "gada": "tidak",
    "gak": "tidak",
    "ga": "tidak",
    "yg": "yang",
    "udah": "sudah",
    "gue": "saya",
    "lu": "kamu",
    "lo": "kamu",
    "gw": "saya",
    "tuh": "itu",
    "nih": "ini",
    "tp": "tapi",
    "dr": "dari",
    "pd": "pada",
    "gw": "saya",
    "kalo": "kalau",
    "si": "",
    "btw": "ngomong-ngomong",
    "dapet": "dapat",
    "sampe": "sampai",
    "mending": "lebih baik",
    "amp": "dan",
    "aja": "saja",
    "pas": "saat",
    "kayak": "seperti",
    "bilang": "mengatakan",
    "duit": "uang",
    "pinter": "pintar",
    "sih": "",
    "oke": "baik",
    "bikin": "membuat",
    "boong": "bohong",
    "pinter": "pintar",
    "no": "tidak",
    "males": "malas", 
    "episiensi": "efisiensi",
    "anjir": "anjing",
    "bgst": "bangsat"
    
}

def text_cleaning(text):
    text = text.lower()
    text = "".join([char for char in text if char not in string.punctuation])
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = emoji.demojize(text)
    text = re.sub(r":\w+:", "", text)
    text = re.sub(r'²', '', text)
    return text

def preprocess(text, slang_dict):
    # Cleaning
    text = text_cleaning(text)
    
    # Tokenization + Slang Conversion + Stopwords Removal + Stemming
    stop_words = set(stopwords.words("indonesian"))
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()

    tokens = word_tokenize(text)
    tokens = [slang_dict.get(word, word) for word in tokens]
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [stemmer.stem(word) for word in tokens]

    return tokens