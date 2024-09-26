import streamlit as st
import PyPDF2
from io import BytesIO
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import plotly.express as px

def summary(content):
    # Create a summary pipeline
    summarizer = pipeline("summarization")
    # Summarize the profile
    summary = summarizer(content, max_length=150, min_length=40, do_sample=False)
    summary_text = summary[0]['summary_text'].strip() if summary else "No summary generated."
    return summary_text

def suggestions(content):
    # Initialize the tokenizer and model for text generation
    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-large-uncased-whole-word-masking-finetuned-squad")
    model = AutoModelForQuestionAnswering.from_pretrained("google-bert/bert-large-uncased-whole-word-masking-finetuned-squad")
    # Create a pipeline for question answering
    qa_pipe = pipeline("question-answering", model=model, tokenizer=tokenizer)
    # Define a question for actionable suggestions
    suggestions_question = "This is a resume / cv of a person. Return a list of important keywords (ignore the personal details like email phone nubmer)"
    suggestions = qa_pipe(question=suggestions_question, context=content)
    suggestions_text = suggestions['answer'].strip() if suggestions else "No suggestions generated."
    return suggestions_text

def bag_of_words(content, st):
    # Split input into individual documents
    documents = content.splitlines()
    # Create the CountVectorizer object
    vectorizer = CountVectorizer()
    # Fit the model and transform the documents into Bag of Words
    bag_of_words = vectorizer.fit_transform(documents)
    # Create a DataFrame to display the results
    bow_df = pd.DataFrame(bag_of_words.toarray(), columns=vectorizer.get_feature_names_out())
    # Calculate word frequencies
    word_frequencies = bow_df.sum(axis=0).reset_index()
    word_frequencies.columns = ['Word', 'Frequency']
    # Create a bubble chart using Plotly
    fig = px.scatter(
        word_frequencies,
        x='Word',
        y='Frequency',
        size='Frequency',
        hover_name='Word',
        title='Word Frequencies in Bag of Words',
        size_max=40
    )
    # Customize the layout
    fig.update_layout(
        xaxis_title='Words',
        yaxis_title='Frequency',
        xaxis=dict(tickangle=-45),  # Rotate x-axis labels for better visibility
        showlegend=False
    )
    # Display the bubble chart in Streamlit
    st.subheader("Bubble Chart of Word Frequencies")
    st.plotly_chart(fig, use_container_width=True)

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

    # st.image("helper_screenshot.png", caption="How to get export the profile as PDF?")
    uploaded_file = st.file_uploader("Upload your CV or LinkedIn profile PDF", type="pdf")
    
    if uploaded_file:
        with st.spinner('Extracting content from PDF...'):
            profile_content = extract_text_from_pdf(uploaded_file)
            st.success("Profile content extracted successfully!")
        
        bag_of_words(profile_content, st)

        st.subheader("Profile Summary")
        st.write(summary(profile_content))

        st.subheader("Key highlight(s)")
        for idx, item in enumerate(suggestions(profile_content), 1):
            st.write(f"{idx}. {item}")
        
        st.subheader("Bag of words")
        st.write(vectorizer.get_feature_names_out())

if __name__ == "__main__":
    main()
