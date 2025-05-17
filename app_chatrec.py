import streamlit as st
import nltk
import speech_recognition as sr
import pyttsx3
from sentence_transformers import SentenceTransformer, util

# Télécharger les ressources NLTK nécessaires
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Corpus enrichi prêt bancaire
corpus = [
    "Bonjour",
    "Bonjour ! Comment puis-je vous aider aujourd'hui ?",
    "Je voudrais obtenir un prêt bancaire.",
    "Très bien, quel type de prêt souhaitez-vous ?",
    "Un prêt immobilier.",
    "Quel est le montant que vous souhaitez emprunter ?",
    "150 000 euros.",
    "Sur quelle durée souhaitez-vous rembourser ce prêt ?",
    "Sur 20 ans.",
    "Avez-vous déjà un apport personnel ?",
    "Oui, j’ai 30 000 euros d’apport.",
    "Parfait, avez-vous des revenus réguliers ?",
    "Oui, je travaille en CDI avec un salaire mensuel stable.",
    "Très bien, nous allons étudier votre dossier. Souhaitez-vous prendre rendez-vous avec un conseiller ?",
    "Oui, ce serait parfait.",
    "Je vous remercie. Un conseiller vous contactera sous peu. Avez-vous d'autres questions ?",
    "Non, merci beaucoup.",
    "Je vous en prie, bonne journée !"
]

# Charger le modèle sentence-transformers (multilingue, léger)
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# Encoder le corpus une fois
corpus_embeddings = model.encode(corpus, convert_to_tensor=True)

def chatbot_response(user_input):
    user_embedding = model.encode(user_input, convert_to_tensor=True)
    cosine_scores = util.pytorch_cos_sim(user_embedding, corpus_embeddings)[0]
    top_result = cosine_scores.argmax().item()
    top_score = cosine_scores[top_result].item()

    if top_score < 0.5:
        return "Désolé, je n'ai pas compris. Pouvez-vous reformuler ?"
    else:
        return corpus[top_result]

def recognize_speech():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    with mic as source:
        st.info("🎤 Parlez maintenant...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio, language="fr-FR")
            st.success(f"Vous avez dit : {text}")
            return text
        except sr.UnknownValueError:
            st.error("Je n'ai pas compris.")
        except sr.RequestError:
            st.error("Erreur de connexion.")
    return None

def speak(text):
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    for voice in voices:
        if hasattr(voice, 'languages'):
            if any("fr" in (lang.decode('utf-8') if isinstance(lang, bytes) else lang) for lang in voice.languages):
                engine.setProperty('voice', voice.id)
                break
        elif "french" in voice.name.lower():
            engine.setProperty('voice', voice.id)
            break
    engine.setProperty('rate', 150)
    engine.say(text)
    engine.runAndWait()

# Interface Streamlit
st.set_page_config(page_title="Chatbot Vocal Français", page_icon="🗣️")
st.title("🤖 Chatbot Vocal en Français 🇫🇷")

mode = st.radio("Choisissez le mode d'entrée :", ("Texte", "Voix"))

if "history" not in st.session_state:
    st.session_state.history = []

if mode == "Texte":
    user_input = st.text_input("💬 Entrez votre message :")
    if st.button("Envoyer"):
        if user_input:
            response = chatbot_response(user_input)
            st.session_state.history.append(("👤 Vous", user_input))
            st.session_state.history.append(("🤖 Chatbot", response))
            speak(response)

elif mode == "Voix":
    if st.button("🎙️ Parler"):
        user_input = recognize_speech()
        if user_input:
            response = chatbot_response(user_input)
            st.session_state.history.append(("👤 Vous", user_input))
            st.session_state.history.append(("🤖 Chatbot", response))
            speak(response)

st.subheader("📝 Historique de la conversation")
for speaker, message in st.session_state.history:
    st.markdown(f"**{speaker} :** {message}")
