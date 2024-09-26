import streamlit as st
import PyPDF2
from io import BytesIO
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

def analyze_profile(content):
    # Initialize the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
    
    # Create a pipeline for text generation
    pipe = pipeline("text2text-generation", model="google/flan-t5-small")

    # Summarize the profile
    summary_prompt = f"Please summarize the following resume in a concise manner:\n\n{content}\n\nSummary:"
    summary = pipe(summary_prompt, max_length=150, num_return_sequences=1)
    summary_text = summary[0]['generated_text'].strip() if summary else "No summary generated."
    print("Summary:\n", summary_text)
    
    # Top 5 action suggestions and action items to improve the profile
    suggestions_prompt = f"Based on the following resume, suggest up to 5 actionable items to improve it:\n\n{content}\n\nSuggestions:"
    suggestions = pipe(suggestions_prompt, max_length=150, num_return_sequences=1)
    suggestions_text = suggestions[0]['generated_text'].strip() if suggestions else "No suggestions generated."
    print("\nSuggestions:", suggestions_text)

    return summary_text, suggestions_text

def extract_text_from_pdf(pdf_file):
    # Read the PDF content
    reader = PyPDF2.PdfReader(pdf_file)
    extracted_text = ""
    for page_num in range(len(reader.pages)):
        page = reader.pages[page_num]
        extracted_text += page.extract_text()
    return extracted_text

def main():
    st.set_page_config(page_title="CV / LinkedIn Feedback Tool", page_icon="icon.png", layout="wide")
    st.title("CV / LinkedIn Profile Feedback Tool")
    hf_token = st.secrets["HUNGGINGFACE_API_TOKEN"]
    print("DEBUG", hf_token)

    # Step 1: Ask the user to upload the LinkedIn Profile PDF
    # st.image("helper_screenshot.png", caption="How to get export the profile as PDF?")
    uploaded_file = st.file_uploader("Upload your CV or LinkedIn profile PDF", type="pdf")
    
    if uploaded_file:
        # Step 2: Extract and Parse LinkedIn Content
        with st.spinner('Extracting content from PDF...'):
            profile_content = extract_text_from_pdf(uploaded_file)
            st.success("Profile content extracted successfully!")

        # Step 3: Provide feedback (Top 5 action items)
        summary, suggestions = analyze_profile(profile_content)

        st.subheader("Profile Summary")
        st.write(summary)

        st.subheader("Action Items to Improve Your Profile")
        for idx, item in enumerate(action_items, 1):
            st.write(f"{idx}. {item}")

if __name__ == "__main__":
    main()
