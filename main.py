import streamlit as st
import PyPDF2
from io import BytesIO
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def analyze_profile(content):
    # Load the model and tokenizer
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Preprocess the text
    inputs = tokenizer(content, return_tensors="pt")

    # Make predictions
    outputs = model(**inputs)
    scores = torch.softmax(outputs.logits, dim=1)

    # Extract the top 5 action items
    action_items = []
    for idx, score in enumerate(scores[0]):
        if score > 0.5:
            action_items.append(f"{tokenizer.decode(idx, skip_special_tokens=True)} ({score:.2f})")
    return action_items

def extract_text_from_pdf(pdf_file):
    # Read the PDF content
    reader = PyPDF2.PdfReader(pdf_file)
    extracted_text = ""
    for page_num in range(len(reader.pages)):
        page = reader.pages[page_num]
        extracted_text += page.extract_text()
    return extracted_text

def main():
    st.title("LinkedIn Profile Feedback Tool")

    # Step 1: Ask the user to upload the LinkedIn Profile PDF
    # st.image("helper_screenshot.png", caption="How to get export the profile as PDF?")
    uploaded_file = st.file_uploader("Upload your LinkedIn profile PDF", type="pdf")
    
    if uploaded_file:
        # Step 2: Extract and Parse LinkedIn Content
        with st.spinner('Extracting content from PDF...'):
            profile_content = extract_text_from_pdf(uploaded_file)
            st.success("Profile content extracted successfully!")

        st.subheader("Profile Content")
        st.write(profile_content)

        # Step 3: Provide feedback (Top 5 action items)
        st.subheader("Top 5 Action Items to Improve Your Profile")
        action_items = analyze_profile(profile_content)
        for idx, item in enumerate(action_items, 1):
            st.write(f"{idx}. {item}")

if __name__ == "__main__":
    main()
