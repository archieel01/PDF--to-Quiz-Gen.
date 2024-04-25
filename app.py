import streamlit as st
import os
import random
from PyPDF2 import PdfReader
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
import torch
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def extract_text_from_pdf(uploaded_file):
    text = ""
    pdf_reader = PdfReader(uploaded_file)
    num_pages = len(pdf_reader.pages)
    for page_number in range(num_pages):
        page = pdf_reader.pages[page_number]
        text += page.extract_text()
    return text

def generate_quiz(text, num_questions=10):
    # Tokenize text into sentences
    sentences = sent_tokenize(text)
    
    # Use a pre-trained sentence transformer model
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    
    # Encode sentences into embeddings
    sentence_embeddings = model.encode(sentences, convert_to_tensor=True)
    
    # Find the most informative sentences based on cosine similarity
    query_embedding = model.encode(['generate questions'], convert_to_tensor=True)
    cosine_scores = torch.nn.functional.cosine_similarity(query_embedding, sentence_embeddings)
    top_sentences_indices = cosine_scores.argsort(descending=True)[:num_questions]
    selected_sentences = [sentences[i] for i in top_sentences_indices]
    
    # Define stop words
    stop_words = set(stopwords.words('english'))
    
    # Generate questions from selected sentences
    questions = []
    for sentence in selected_sentences:
        # Tokenize the sentence into words
        words = nltk.word_tokenize(sentence)
        
        # Remove stopwords and punctuation
        filtered_words = [word.lower() for word in words if word.isalpha() and word.lower() not in stop_words]
        
        # Ensure the sentence contains enough meaningful words for a question
        if len(filtered_words) >= 5:
            # Select a random word from the sentence
            blank_word = random.choice(filtered_words)
            
            # Replace the word with a blank
            blank_index = filtered_words.index(blank_word)
            filtered_words[blank_index] = '__________'
            
            # Generate multiple-choice options
            choices = [blank_word.capitalize()]  # Correct answer
            for word in filtered_words:
                if word != blank_word and word not in choices:
                    choices.append(word.capitalize())
                    if len(choices) == 4:
                        break
            random.shuffle(choices)
            
            # Ensure there are exactly 4 choices
            while len(choices) < 4:
                additional_choice = random.choice(filtered_words)
                if additional_choice.capitalize() not in choices:
                    choices.append(additional_choice.capitalize())
            random.shuffle(choices)
            
            # Construct the question
            question = ' '.join(filtered_words)
            
            # Add the question and choices to the list
            questions.append((sentence.strip(), question, choices, blank_word))
    
    return questions



def grade_quiz(questions, selected_answers):
    num_correct = 0
    num_questions = len(questions)
    if num_questions == 0:
        return 0  # Return score of 0 if there are no questions
    for question, selected_option in zip(questions, selected_answers):
        correct_option = question[-1]  # Get the last element of the tuple as the correct option
        if selected_option[0].capitalize() == correct_option.capitalize():  # Access the first element of the tuple and capitalize it
            num_correct += 1
    score = (num_correct / num_questions) * 100
    return score

def main():
    # Add CSS for black background
    st.markdown(
        """
        <style>
        body {
            background-color: #000000;
            color: white;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("PDF to Quiz Generator")

    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

    if uploaded_file is not None:
        # Extract text from uploaded PDF
        extracted_text = extract_text_from_pdf(uploaded_file)
        # Generate quiz questions
        multiple_choice_questions = generate_quiz(extracted_text, num_questions=10)  # Generate multiple-choice questions
        
        # Display multiple-choice questions
        st.subheader("Multiple Choice Questions")
        selected_answers_mcq = []
        for i, (original_sentence, question, choices, _) in enumerate(multiple_choice_questions, start=1):
            st.write(f"Question {i}:")
            st.write(f"Question: {question}")
            selected_option = st.radio("Select the correct word for the blank:", choices)
            selected_answers_mcq.append((question, selected_option, choices))
        
        
        # Grade the multiple-choice quiz
        score_mcq = grade_quiz(multiple_choice_questions, selected_answers_mcq)
        st.write(f"Multiple Choice Quiz Score: {score_mcq}%")

if __name__ == "__main__":
    main()
