# HR Policy Assistant Bot

A conversational AI agent built using LangGraph and Retrieval-Augmented Generation (RAG) to help employees quickly access and understand company HR policies.

## Live Demo
🔗 https://hr-policy.streamlit.app/

---

## Overview

This project is an intelligent HR assistant that answers employee queries related to company policies such as leave, salary, reimbursements, work hours, and more.

The system uses a combination of:
- Retrieval-Augmented Generation (RAG)
- LangGraph workflow orchestration
- ChromaDB vector database
- LLM-based reasoning (Groq)

---

## Features

- Accurate HR policy Q&A using document retrieval
- Conversation memory across multiple turns
- Tool integration (date/time queries)
- Self-evaluation loop to improve answer faithfulness
- Streamlit-based interactive UI
- Handles out-of-scope queries gracefully

---

## Tech Stack

- Python
- LangChain
- LangGraph
- ChromaDB
- Sentence Transformers
- Streamlit
- Groq LLM API

---

## Project Structure

---

## How It Works

1. User enters a query via Streamlit UI  
2. Query is routed using LangGraph  
3. Relevant documents are retrieved from ChromaDB  
4. LLM generates an answer strictly grounded in context  
5. Evaluation node checks faithfulness and retries if needed  
6. Final answer is returned to the user  

---

## Example Queries

- "How many paid leaves do employees get?"
- "What is the reimbursement policy?"
- "What did I just ask?"
- "What is today's date?"

---

## Deployment

The application is deployed using **Streamlit Cloud**.

To run locally:

```bash
streamlit run capstone_streamlit.py

AUTHOR - GUARAV DAS

---

# What this does for you

- Looks professional on GitHub  
- Helps in evaluation  
- Can be used in resume/projects section  
- Matches exactly what you built  

---

If you want next, I can give:
- **resume bullet points**
- **viva answers (very likely questions)**
