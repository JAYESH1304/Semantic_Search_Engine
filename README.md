# Semantic Search Engine

The aim of this project is to develop a custom semantic search engine tailored for specific use cases.

## Technologies Used

1. Pinecone Vector Databases: Utilized for storing and querying vector embeddings of text data.
2. Sentence Transformer: Used to generate high-quality embeddings from input texts.
3. Streamlit: Deployed for creating a user-friendly interface where users can interact with the search engine.
   
## Progress of Task
1. Data Input: Text data with accompanying metadata can be uploaded through a Streamlit interface hosted on Huggingface. The data must conform to predefined column names.

2. Indexing: Pinecone is employed to establish indexes where vector embeddings of the input texts are stored. These embeddings are organized under specific namespaces, facilitating efficient retrieval.

3. Querying: Users can input queries through the Streamlit interface. The search engine utilizes Pinecone to fetch the top 5 most relevant results based on the vector similarity of the query.

## Installation and Usage

- Run the Streamlit app locally using streamlit run Final_app.py.
