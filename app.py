import streamlit as st
import pickle
from preprocessing_module import preprocess, slang_dict

# Load model dan vectorizer
with open('DecisionTreeModel.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Judul aplikasi
st.title("Analisis Sentimen: Program Makan Bergizi Gratis")
st.write("Masukkan teks untuk melihat prediksi sentimennya (positif/negatif/netral).")

# Input teks dari pengguna
input_text = st.text_area("Masukkan teks di sini:", "")

if st.button("Prediksi Sentimen"):
    if not input_text.strip():
        st.warning("Teks kosong. Silakan masukkan teks.")
    else:
        try:
            # Preprocessing
            preprocess_tokens = preprocess(input_text, slang_dict)
            preprocess_tokens = [token for token in preprocess_tokens if token != ""]
            preprocess_input = " ".join(preprocess_tokens)

            # Vektorisasi dan prediksi
            vectorize_input = vectorizer.transform([preprocess_input])
            pred_input = model.predict(vectorize_input)
            sentiment = pred_input[0]

            # Tampilkan hasil
            st.success(f"Sentimen: **{sentiment}**")
        except Exception as e:
            st.error(f"Terjadi kesalahan: {str(e)}")
