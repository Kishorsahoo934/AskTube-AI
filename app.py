import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os
import re

# Load API key
# load_dotenv()

# UI
st.title("🎥 Ask Questions About a YouTube Video")
video_url = st.text_input("Paste YouTube video URL:")

# State for transcript, vector store, and question
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if video_url:
    # Show video in Streamlit
    st.video(video_url)

    # Extract video ID
    match = re.search(r"(?:v=|youtu\.be/)([a-zA-Z0-9_-]{11})", video_url)
    if not match:
        st.error("❌ Invalid YouTube URL.")
        st.stop()
    video_id = match.group(1)

    # Load transcript and vector store once
    if st.session_state.vector_store is None:
        with st.spinner("⏳ Fetching and processing transcript..."):
            try:
                ytt_api = YouTubeTranscriptApi()
                transcript_list = ytt_api.fetch(video_id, languages=["en"])
                transcript = " ".join(chunk.text for chunk in transcript_list)

                # Split
                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                chunks = splitter.create_documents([transcript])

                # Embedding + FAISS
                embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
                vector_store = FAISS.from_documents(chunks, embeddings)
                st.session_state.vector_store = vector_store

                st.success("✅ Transcript loaded! Ask your question below.")

            except TranscriptsDisabled:
                st.error("❌ This video has no transcript.")
            except Exception as e:
                st.error(f"Error: {e}")

# Question input and answer
if st.session_state.vector_store:
    question = st.text_input("❓ Ask a question about the video:")
    if question:
        with st.spinner("💡 Thinking..."):
            retriever = st.session_state.vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

            def format_docs(docs):
                return "\n\n".join(doc.page_content for doc in docs)

            prompt = PromptTemplate(
                template="""
                You are a helpful assistant.
                Answer ONLY from the provided transcript context.
                If the context is insufficient, just say you don't know.

                {context}
                Question: {question}
                """,
                input_variables=["context", "question"]
            )

            parallel_chain = RunnableParallel({
                "context": retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough()
            })

            llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash", temperature=0.1,api_key="AIzaSyBz80LVIZEwFDppx8s0Ii8AEJ3CQYTEfsc")
            parser = StrOutputParser()
            main_chain = parallel_chain | prompt | llm | parser

            answer = main_chain.invoke(question)
            st.success("✅ Answer:")
            st.write(answer)
