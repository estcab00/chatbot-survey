import os
import json
import pandas as pd
import streamlit as st
import openai
from dotenv import load_dotenv
import re  
from openai import OpenAI

# Streamlit configuration
st.set_page_config(page_title="Chatbot 💬", layout="centered")

with st.sidebar:
    st.title('Chatbot con la ENAHO (2022)')
    st.markdown('''
    ## Acerca de
    Este chatbot está diseñado para interactuar con los datos de la Encuesta Nacional de Hogares (ENAHO) de Perú del año 2022. Puedes hacer preguntas sobre los datos de la encuesta y el chatbot tratará de proporcionarte respuestas basadas en la información disponible.''')

load_dotenv()

# Define load_data_from_csv function
def load_data_from_csv(file_path='data/df_enaho.csv'):
    df = pd.read_csv(file_path, encoding='latin-1')
    return df

# Load the survey data from the CSV file
data = load_data_from_csv('data/df_enaho.csv')

def main():
    st.header("Chat with Survey Data 💬")

# Define the system_prompt
system_prompt = """
You are an expert in analyzing survey data from Peru. You will answer user questions based on the data provided. """

if "messages" not in st.session_state:
    st.session_state["messages"] = []

prompt = st.text_input("Your inquiry:", "")

openai.api_key = st.secrets['openai_key']

# Define the extract_relevant_info function

def extract_relevant_info(data, question):
    relevant_info = []
    question_keywords = set(re.findall(r'\w+', question.lower()))

    for index, row in data.iterrows():
        row_text = ' '.join(map(str, row.values)).lower()
        row_keywords = set(re.findall(r'\w+', row_text))
        common_keywords = question_keywords.intersection(row_keywords)
        
        if common_keywords:
            relevant_info.append({
                "index": index,
                "content": row_text,
                "common_keywords": common_keywords
            })

    return relevant_info

docs_chunks = extract_relevant_info(data, prompt)

def find_relevant_chunks(question, docs_chunks, max_chunks=5):
    # Tokenize the question and extract keywords
    question_keywords = set(re.findall(r'\w+', question.lower()))
    relevance_scores = []

    # Calculate the relevance score for each chunk
    for chunk in docs_chunks:
        chunk_text = chunk["content"].lower()
        chunk_keywords = set(re.findall(r'\w+', chunk_text))
        common_keywords = question_keywords.intersection(chunk_keywords)
        relevance_scores.append((len(common_keywords), chunk))

    # Order the chunks by relevance score
    relevant_chunks = [chunk for _, chunk in sorted(relevance_scores, key=lambda x: x[0], reverse=True)]

    # Return only the top max_chunks
    return relevant_chunks[:max_chunks]

def send_question_to_openai(question, docs_chunks):
    # Find the most relevant chunks of text based on the question
    relevant_chunks = find_relevant_chunks(question, docs_chunks)
    
    # Build the system prompt with the relevant chunks and the question
    prompt_text = system_prompt + "\n\n" + "\n\n".join([chunk["content"] for chunk in relevant_chunks]) + "\n\nQuestion: " + question

    # Call the OpenAI API to get a response
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]
    )
    
    # Return the message content directly
    return response.choices[0].message.content

if st.button("Send"):
    if prompt:  # Check if the prompt is not empty
        user_message = {"role": "user", "content": prompt}
        st.session_state.messages.append(user_message)

        with st.spinner("Generating answer..."):
            response_text = send_question_to_openai(prompt, docs_chunks)
            if response_text:  # Check if the response_text is not None or empty
                assistant_message = {"role": "assistant", "content": response_text}
                st.session_state.messages.append(assistant_message)
            else:
                st.error("Failed to get a response.")  # Display an error if no response was received

# Display only the latest messages
if st.session_state.messages:
    # Get the last user message
    latest_user_message = None
    latest_assistant_message = None

    for message in reversed(st.session_state.messages):
        if message["role"] == "user" and latest_user_message is None:
            latest_user_message = message
        elif message["role"] == "assistant" and latest_assistant_message is None:
            latest_assistant_message = message
        
        if latest_user_message and latest_assistant_message:
            break
    
    # Display the last user question
    if latest_user_message:
        st.text_area("Question", value=latest_user_message["content"], height=75, disabled=True)
    
    # Display the last assistant answer
    if latest_assistant_message:
        st.text_area("Answer", value=latest_assistant_message["content"], height=100, disabled=True)

if __name__ == "__main__":
    main()