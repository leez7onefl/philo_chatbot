import streamlit as st
import os
from dotenv import load_dotenv, find_dotenv
import openai
import base64
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"./myenv/Tesseract-OCR/tesseract.exe"
from pdf2image import convert_from_bytes
from github import Github
from pinecone import Pinecone

# Set page configuration first
st.set_page_config(page_title="philo_chatbot", page_icon="🤖", layout="wide")

#______________________________________________________________

def load_multiple_envs(env_paths):
    for folder, filename in env_paths:
        dotenv_path = find_dotenv(os.path.join(folder, filename), raise_error_if_not_found=True, usecwd=True)
        load_dotenv(dotenv_path, override=True)

#______________________________________________________________

def openai_response(system_prompt, user_prompt, selected_model, temperature, top_p, frequency_penalty, max_length, env_variables):
    openai.api_key = env_variables["openai_api_key"]

    try:
        response = openai.chat.completions.create(
            model=selected_model,
            max_tokens=max_length,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        )
    
        return response.choices[0].message.content if response.choices else "No response found."
    except Exception as e:
        return f"An error occurred: {str(e)}"

#______________________________________________________________

def get_embedding(text, model="text-embedding-3-small"):
    response = openai.embeddings.create(
        model=model,
        input=text,
        encoding_format="float"
    )
    return response.data[0].embedding

#______________________________________________________________

def summarize_text(text, model="gpt-4o", max_length=300):
    try:
        response = openai.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": f"Summarize the following text:\n\n{text}"}],
            max_tokens=max_length,
            temperature=0.5
        )
        summary = response.choices[0].message.content
        return summary
    except Exception as e:
        return f"Summary error: {str(e)}"

def refine_response(text, prompt, model="gpt-4", max_length=1024):
    try:
        response = openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Tu résumes à partir des informations afin de répondre aux questions. Tu as le droit d'ajouter des informations externes"},
                {"role": "user", "content": f"Refine and summarize the relevant information in order to answer this question : {prompt}\n\n{text}"}
            ],
            max_tokens=max_length,
            temperature=1
        )
        refined_summary = response.choices[0].message.content
        return refined_summary
    except Exception as e:
        return f"Refinement error: {str(e)}"


#______________________________________________________________

def extract_text_from_pdf(file_content):
    images = convert_from_bytes(file_content, poppler_path=r"./myenv/poppler-24.08.0/Library/bin")
    text = ""
    for image in images:
        text += pytesseract.image_to_string(image)
    return text

def chunk_text(text, max_tokens=8191):
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0
    
    for word in words:
        word_length = len(word) + 1  # Account for space
        if current_length + word_length <= max_tokens:
            current_chunk.append(word)
            current_length += word_length
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_length = word_length

    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks

def process_and_store_documents(repo_docs):
    vectors = []
    progress_bar = st.progress(0)

    for doc_id, document in enumerate(repo_docs):
        extracted_text = extract_text_from_pdf(document['file_content'])
        text_chunks = chunk_text(extracted_text, max_tokens=8191)
        
        for i, chunk in enumerate(text_chunks):
            summary = summarize_text(chunk)
            vector = get_embedding(chunk)
            vectors.append({
                "id": f"{doc_id}_{i}",
                "values": vector,
                "metadata": {
                    "file_path": document['file_path'],
                    "summary": summary  # Store the summary in metadata
                }
            })
        
        progress = (doc_id + 1) / len(repo_docs)
        progress_bar.progress(progress)

    return vectors

#______________________________________________________________

def retrieve_and_process_github_documents(github_token, repo_name, branch='documents', path=None):
    g = Github(github_token)
    repo = g.get_repo(repo_name)
    contents = repo.get_contents(path or "", ref=branch)

    documents = []
    while contents:
        file_content = contents.pop(0)
        if file_content.type == 'file' and file_content.path.endswith('.pdf'):
            content_file = repo.get_contents(file_content.path, ref=branch)
            file_content_data = base64.b64decode(content_file.content)
            
            documents.append({"file_path": file_content.path, "file_content": file_content_data})
            
        elif file_content.type == 'dir':
            contents.extend(repo.get_contents(file_content.path, ref=branch))

    return documents

#______________________________________________________________

def initialize_pinecone(api_key, index_name):
    pc = Pinecone(api_key=api_key)
    index = pc.Index(index_name)
    return index

def is_pinecone_index_empty(index):
    response = index.describe_index_stats()
    namespaces = response.get('namespaces', {})
    ns2_stats = namespaces.get('ns2', {})
    vector_count = ns2_stats.get('vector_count', 0)
    return vector_count == 0

def store_vectors_in_pinecone(index, vectors):
    index.upsert(vectors=vectors, namespace="ns2")

def query_pinecone_index(index, query_vector, top_k=5):
    response = index.query(
        namespace="ns2",
        vector=query_vector,
        top_k=top_k,
        include_values=False,
        include_metadata=True
    )
    return response['matches']

#______________________________________________________________

load_multiple_envs([
    ['env', '.env']
])
env_variables = {
    'openai_api_key': os.getenv('OPENAI_API_KEY'),
    'pinecone_key': os.getenv('PINECONE_API_KEY'),
    'pinecone_index': os.getenv('PINECONE_INDEX_NAME'),
    'github_token': os.getenv('GITHUB_TOKEN'),
    'github_repo': os.getenv('GITHUB_REPO'),
    'github_branch': os.getenv('GITHUB_BRANCH', 'documents')
}

# Initialize Pinecone
pinecone_index = initialize_pinecone(env_variables['pinecone_key'], env_variables['pinecone_index'])

# Check if Pinecone index is empty and process if necessary
if is_pinecone_index_empty(pinecone_index):
    st.info("Waking up instance and loading documents...")

    with st.spinner("Retrieving and processing documents..."):
        repo_docs = retrieve_and_process_github_documents(
            env_variables['github_token'],
            env_variables['github_repo'],
            env_variables['github_branch']
        )
        st.write("Documents retrieved.")

    with st.spinner("Chunking, embedding, vectorizing, and adding metadata..."):
        vectors = process_and_store_documents(repo_docs)
        store_vectors_in_pinecone(pinecone_index, vectors)
        st.write("Documents processed and stored.")
else:
    st.info("Documents are already processed and stored in vectorial database.")

# Clear UI before displaying chat
st.empty()

#______________________________________________________________

st.markdown(
    r"""
    <style>
    .stDeployButton {
            visibility: hidden;
        }
    </style>
    """, unsafe_allow_html=True
)

with st.sidebar:

    st.title("Leonard's AI profile 📱")
    
    with st.expander("Parameters"):
        selected_model = st.selectbox('Model', ['GPT-4o', 'o1-mini','gpt-3.5-turbo'], key='selected_model')
        temperature = st.slider('Creativity -/+', min_value=0.01, max_value=1.0, value=0.8, step=0.01)
        top_p = st.slider('Words randomness -/+', min_value=0.01, max_value=1.0, value=0.95, step=0.01)
        frequency_penalty = st.slider('Frequence Penalty -/+', min_value=-1.99, max_value=1.99, value=0.00, step=0.01)
        max_length = st.slider('Max Length', min_value=256, max_value=8192, value=4224, step=2)

#______________________________________________________________

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Ask me anything regarding philo ING3 course ! 🔮"}]

for message in st.session_state['messages']:
    with st.chat_message(message["role"]):
        st.write(message["content"])

def clear_chat_history():
    st.session_state['messages'] = [{"role": "assistant", "content": "Ask me anything regarding philo ING3 course ! 🔮"}]

st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

#______________________________________________________________
st.session_state['messages'] = [{"role": "assistant", "content": "Ask me anything regarding philo ING3 course ! 🔮"}]
if prompt := st.chat_input(placeholder="Enter your message"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.write(f"**User:** {prompt}")
    
    with st.spinner("Thinking . . . "):
        query_vector = get_embedding(prompt)
        results = query_pinecone_index(pinecone_index, query_vector)
        
        # Concatenate all summaries
        all_summaries = "\n".join([
            f"File: {item.metadata['file_path']} - Summary: {item.metadata['summary']} - Score: {item['score']}"
            for item in results if item.metadata
        ]) if results else "No relevant information found."
        
        # Refine the response
        refined_response = refine_response(all_summaries, prompt)
        
        # Directly display the refined response
        st.write(f"**Assistant:** {refined_response}")

    message = {"role": "assistant", "content": refined_response}
    st.session_state.messages.append(message)
