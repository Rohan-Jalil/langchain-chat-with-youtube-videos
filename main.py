import streamlit as st
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import OpenAIWhisperParser
from langchain.document_loaders.blob_loaders.youtube_audio import YoutubeAudioLoader
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

# st.title("Chat with Youtube")

# with st.form("my_form"):

    # video_url = st.text_input(label="Please insert Youtube video URL to Chat with:", placeholder="URL")

    # Every form must have a submit button.
    # submitted = st.form_submit_button("Submit")

    # if submitted and video_url != "":

        save_dir = "./videos/"

        loader = GenericLoader(YoutubeAudioLoader(['https://www.youtube.com/watch?v=YK8GZmuf8_0'], save_dir), OpenAIWhisperParser())
        docs = loader.load()

        # Combine doc
        combined_docs = [doc.page_content for doc in docs]
        text = " ".join(combined_docs)

        # Split them
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
        splits = text_splitter.split_text(text)

        # Build an index
        embeddings = OpenAIEmbeddings()
        vectordb = Chroma.from_texts(splits, embeddings)

        # Build a QA chain
        qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0),
        chain_type="stuff",
        retriever=vectordb.as_retriever(),
        )

        # question = st.text_input("Enter your query:")
        
        question = "What is this video about?"

        result = qa_chain.run(question)
        
        print(result)

        # st.write(result["result"])
       
       
       
       
       
       
