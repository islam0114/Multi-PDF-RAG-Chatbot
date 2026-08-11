import streamlit as st
import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_cohere import ChatCohere, CohereRerank
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

st.set_page_config(page_title="Multi-PDF RAG Chatbot", layout="wide")
st.title("Multi-PDF RAG Chatbot")

# Retrieve Cohere API Key
with st.expander("⚙️ Configuration & Document Upload", expanded=not st.session_state.vectorstore):
    cohere_api_key = st.text_input("Enter Cohere API Key", type="password")
    st.markdown("💡 **Tip:** Get your free API key from [Cohere's Dashboard](https://dashboard.cohere.com/api-keys).")

    if not cohere_api_key:
        st.warning("⚠️ Please enter your Cohere API key above to unlock the chatbot.")
        st.stop()

    st.markdown("---")
    uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    
    col1, col2 = st.columns(2)
    with col1:
        k_val = st.slider("Retriever 'k' (Number of chunks)", min_value=1, max_value=20, value=5)
    with col2:
        st.markdown("<br>", unsafe_allow_html=True) # spacing
        use_rerank = st.checkbox("Use Cohere Rerank (Hybrid-like retrieval)", value=True)

    if uploaded_files and st.button("🚀 Process PDFs", use_container_width=True):
        with st.spinner("Processing documents..."):
            st.session_state.all_documents = []
            for file in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(file.getvalue())
                    tmp_path = tmp.name
                
                loader = PyPDFLoader(tmp_path)
                docs = loader.load()
                for d in docs:
                    d.metadata["source"] = file.name
                
                st.session_state.all_documents.extend(docs)
                os.remove(tmp_path)
            
            total_pages = len(st.session_state.all_documents)
            if total_pages < 30:
                c_size, c_overlap = 800, 100
            elif total_pages < 200:
                c_size, c_overlap = 1500, 200
            else:
                c_size, c_overlap = 3000, 400
                
            st.info(f"Auto-adjusted chunk size to {c_size} (Total pages: {total_pages})")
            
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=c_size, chunk_overlap=c_overlap)
            chunks = text_splitter.split_documents(st.session_state.all_documents)
            
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            
            with st.spinner("Embedding documents locally (No API limits!)..."):
                st.session_state.vectorstore = Chroma.from_documents(chunks, embeddings)
                    
            st.success(f"✅ Processed {len(uploaded_files)} files into {len(chunks)} chunks successfully!")

# Filtering Section
if st.session_state.all_documents:
    available_pdfs = list(set([doc.metadata.get('source') for doc in st.session_state.all_documents]))
    selected_pdfs = st.multiselect("📑 Select PDFs to search in:", options=available_pdfs, default=available_pdfs)
else:
    selected_pdfs = []

# Main Chat Interface
for msg in st.session_state.chat_history:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)

user_query = st.chat_input("Ask a question about the PDFs...")

if user_query:
    if not st.session_state.vectorstore:
        st.error("Please upload and process at least one PDF first.")
        st.stop()
    if not selected_pdfs:
        st.warning("Please select at least one PDF to search from the sidebar.")
        st.stop()
        
    # Append user message
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    with st.chat_message("user"):
        st.markdown(user_query)
        
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # 1. Filter documents context for the retriever (Chroma doesn't easily support dynamic source filtering natively without metadata filters, so we recreate a temp retriever or use search_kwargs filter)
                # We can use a metadata filter in Chroma
                filter_dict = {"source": {"$in": selected_pdfs}} if len(selected_pdfs) > 0 else None
                
                base_retriever = st.session_state.vectorstore.as_retriever(
                    search_kwargs={"k": k_val, "filter": filter_dict} if filter_dict else {"k": k_val}
                )
                
                # 3. Build QA Chain
                llm = ChatCohere(model="command-a-03-2025", temperature=0, cohere_api_key=cohere_api_key)
                
                qa_prompt = ChatPromptTemplate.from_messages([
                    ("system", "Answer the user's question using ONLY the context below. Please answer in the same language as the user's question. If you don't know, say 'I don't know.'\n\nContext:\n{context}"),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"),
                ])
                
                # Retrieve documents
                docs = base_retriever.invoke(user_query)
                
                # 2. Add Reranker if selected
                if use_rerank and len(docs) > 0:
                    compressor = CohereRerank(cohere_api_key=cohere_api_key, model="rerank-multilingual-v3.0", top_n=3)
                    docs = compressor.compress_documents(docs, user_query)
                
                # 4. Invoke Chain manually without langchain.chains
                context_str = "\n\n".join([doc.page_content for doc in docs])
                chain = qa_prompt | llm
                
                response = chain.invoke({
                    "input": user_query,
                    "context": context_str,
                    "chat_history": st.session_state.chat_history[:-1]  # Exclude current question
                })
                
                answer = response.content
                sources = docs
                
                # Format Response
                output_md = answer + "\n\n"
                if sources:
                    output_md += "**Sources:**\n"
                    for i, doc in enumerate(sources):
                        source_file = doc.metadata.get("source", "Unknown")
                        page = doc.metadata.get("page", "Unknown")
                        output_md += f"- {source_file} (Page {page})\n"
                
                st.markdown(output_md)
                
                # Append assistant message
                st.session_state.chat_history.append(AIMessage(content=output_md))
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
