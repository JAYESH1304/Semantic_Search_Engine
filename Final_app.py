import streamlit as st
import os
import time
import numpy as np
import pandas as pd

def add_custom_css():
    st.markdown("""
    <style>
    .container {
        text-align: center;
        background-color: #f0f0f0;
        padding: 20px;
    }
    .big-font {
        font-size: 50px;
        color: #4CAF50;
    }
    .progress-bar {
        margin-top: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

if 'packages_installed' not in st.session_state:
    st.info("Installing required packages...")
    os.system("pip install -U sentence-transformers")
    os.system("pip install pinecone-client")
    st.session_state['packages_installed'] = True

    from sentence_transformers import SentenceTransformer
    from pinecone import Pinecone, ServerlessSpec, PodSpec

if 'pc' not in st.session_state:
    use_serverless = False
    # Configure Pinecone client
    api_key = os.environ.get('PINECONE_API_KEY', '28b0fd5a-fdfb-422d-9a44-c0ec09a25074')
    environment = os.environ.get('PINECONE_ENVIRONMENT', 'gcp-starter')
    st.session_state['pc'] = Pinecone(api_key=api_key)

    if use_serverless:
        spec = ServerlessSpec(cloud='gcp', region='asia-southeast1-gcp')
    else:
        spec = PodSpec(environment=environment)
      
    if 'model' not in st.session_state:
      st.session_state['model'] = SentenceTransformer('intfloat/e5-small')

index_name = 'dataset'

if index_name not in st.session_state.pc.list_indexes().names():
  dimensions = 384
  st.session_state.pc.create_index(
            name=index_name,
            dimension=dimensions,
            metric='cosine',
            spec=spec
        )
    # Wait until index is ready
  while not st.session_state.pc.describe_index(index_name).status['ready']:
      time.sleep(1)
    
if 'index' not in st.session_state:
  st.session_state['index'] = st.session_state.pc.Index(index_name)


# Function to process data and insert into Pinecone index
def process_data(data, namespace):
    input_texts = data['Query']
    
    progress_bar = st.progress(0)
    total_chunks = len(data) // 1000 + 1

    for chunk_start in range(0, len(data), 1000):
        chunk_end = min(chunk_start + 1000, len(data))
        chunk = data.iloc[chunk_start:chunk_end]
        
        # Generate embeddings for the current chunk
        chunk_embeddings = [st.session_state.model.encode(query, normalize_embeddings=True) for query in chunk['Query']]
        chunk['embedding'] = chunk_embeddings
        
        # Upsert embeddings
        st.session_state.index.upsert(vectors=zip(chunk['id'], chunk['embedding']), namespace=namespace)
        
        # Update progress bar
        progress = (chunk_end / len(data)) * 100
        progress_bar.progress(int(progress))



def load_and_process_data(file):
    data = pd.read_csv(file)
    data = data[0:500]
    data['id'] = data.index.astype(str)
    namespace = file.name[:15]  # Use first 15 characters of file name as namespace
    if 'embeddings_done' not in st.session_state:
        process_data(data, namespace)
        st.session_state['embeddings_done'] = True
    return data, namespace

def main():
    add_custom_css()

    st.markdown("""
    <div class='container'>
        <h1 class='big-font'>Semantic Search Engine</h1>
    </div>
    """, unsafe_allow_html=True)
    
    # Use session state to retain information across interactions
    if 'namespace' not in st.session_state:
        st.session_state.namespace = None
    if 'df' not in st.session_state:
        st.session_state.df = None

    uploaded_file = st.file_uploader("Upload dataset (CSV format)", type=["csv"])
    
    if uploaded_file is not None:
        filename = uploaded_file.name
        namespace = filename.split('.')[0]  
        st.info("Dataset Processing Started...")
        st.session_state.df, st.session_state.namespace = load_and_process_data(uploaded_file)
        st.info("Dataset Processing Completed...")

    if st.session_state.namespace:
        query = st.text_input("Enter your query about the data (or type 'exit' to quit):")
        
        if query.lower() != 'exit':
            vec = st.session_state.model.encode(query)
            result = None
            result = st.session_state.index.query(
                namespace=st.session_state.namespace,
                vector=vec.tolist(),
                top_k=5,
                include_values=False
            )
            
            st.subheader("Query Results:")
            if result:
                id = result['matches'][0]['id']
                data = st.session_state.df
                answer = data[data['id'] == id]['Answer'].values[0]
                st.write(answer)
            
        if st.button("Delete Stored Data"):
            st.session_state.index.delete(deleteAll=True, namespace =st.session_state.namespace)
            st.stop()
            
if __name__ == "__main__":
    main()
