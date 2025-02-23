import asyncio

# ✅ Fix for asyncio event loop issue
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# ✅ Set Page Config
st.set_page_config(page_title="Movie Review Sentiment Analysis", layout="wide")

# ✅ Title
st.title("🎬 Movie Review Sentiment Analysis using BERT")

# ✅ Load Pre-trained BERT Model & Tokenizer
model_path = r"E:\IMDB_MOVIES_REVIEWS_BERT\fine_tuned_bert_model"  # Replace with your local path or HuggingFace repo

@st.cache_resource
def load_model_and_tokenizer(model_path):
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path)
    return tokenizer, model

tokenizer, model = load_model_and_tokenizer(model_path)

# ✅ Set Device (GPU/CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ✅ User Input
st.subheader("📥 Enter a Movie Review:")
user_input = st.text_area("Type your review here...", height=150)

if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a review to analyze.")
    else:
        # Tokenize the input
        inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding='max_length', max_length=128)
        inputs = {key: val.to(device) for key, val in inputs.items()}

        # Get Prediction
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            predicted_class = torch.argmax(logits, dim=1).item()

        # Map Prediction
        labels_map = {0: "Negative", 1: "Positive"}
        sentiment = labels_map[predicted_class]

        # ✅ Display Result
        if sentiment == "Positive":
            st.success(f"🎉 **Predicted Sentiment:** {sentiment}")
        else:
            st.error(f"💔 **Predicted Sentiment:** {sentiment}")

# ✅ Batch Testing Section
st.subheader("📊 Batch Review Testing")

batch_reviews = st.text_area("Paste multiple reviews (one per line):", height=200)
if st.button("Run Batch Analysis"):
    if batch_reviews.strip() == "":
        st.warning("⚠️ Please add some reviews for batch testing.")
    else:
        review_list = batch_reviews.strip().split('\n')
        inputs = tokenizer(review_list, truncation=True, padding='max_length', max_length=128, return_tensors='pt')
        inputs = {key: val.to(device) for key, val in inputs.items()}

        # Predictions
        with torch.no_grad():
            outputs = model(**inputs)
            predicted_labels = torch.argmax(outputs.logits, dim=1)

        # ✅ Display Predictions
        for review, label in zip(review_list, predicted_labels):
            sentiment = "Positive" if label == 1 else "Negative"
            st.write(f"**Review:** {review[:100]} ➔ **Sentiment:** {sentiment}")

# ✅ Evaluation Metrics (Optional)
st.subheader("📈 Model Evaluation (Optional)")
if st.button("Show Sample Evaluation"):
    # Dummy data for evaluation
    true_labels = [1, 0, 1, 0, 1]  # Replace with actual test labels
    predicted_labels = [1, 0, 1, 1, 0]  # Replace with model predictions

    # Classification Report
    report = classification_report(true_labels, predicted_labels, target_names=["Negative", "Positive"], output_dict=True)
    st.write("### 📊 Classification Report")
    st.json(report)

    # Confusion Matrix
    cm = confusion_matrix(true_labels, predicted_labels)
    st.write("### 📉 Confusion Matrix")
    st.write(cm)

# ✅ Footer
st.markdown("---")
st.markdown("Built with ❤️ using **BERT** and **Streamlit**")

