from flask import Flask, request, render_template
import pickle
from preprocessing_module import preprocess, slang_dict, text_cleaning

app = Flask(__name__, template_folder='Page')

# Load model dan vectorizer dari folder saat ini
with open('DecisionTreeModel.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

@app.route('/')
def home():
    return "Welcome to the Sentiment Analysis Makan Bergizi Gratis."

@app.route('/form', methods=['GET', 'POST'])
def form_predict():
    if request.method == 'POST':
        try:
            input_text = request.form['text']
            if not input_text.strip():
                return render_template('home.html', input_text="", sentiment="Teks kosong.")

            preprocess_tokens = preprocess(input_text, slang_dict)
            preprocess_tokens = [token for token in preprocess_tokens if token != ""]  # skip empty tokens
            preprocess_input = " ".join(preprocess_tokens)

            vectorize_input = vectorizer.transform([preprocess_input])
            pred_input = model.predict(vectorize_input)

            sentiment = pred_input[0]

            return render_template('home.html', input_text=input_text, sentiment=sentiment)
        except Exception as e:
            return render_template('home.html', input_text=input_text, sentiment=f"Error: {str(e)}")

    return render_template('home.html')

if __name__ == '__main__':
    app.run(debug=True)