# JURIS_AI: A Retrieval-Augmented Generation System for Legal Documents

JURIS_AI is a sophisticated **Retrieval-Augmented Generation (RAG)** system designed to assist with legal document analysis. It leverages advanced semantic search, re-ranking, and language models to generate precise and context-aware responses from legal texts.

---

## 🚀 Features

- **Dataset**: Utilizes the **BSARD** dataset containing Belgian legal articles.  
- **Semantic Chunking + Recursive chunking** to optimize source length for efficient processing.  
- **Milvus Integration**: A vector database with partitions created per legal code for efficient storage and retrieval.  
- **Search Options**:  
  - Simple search  
  - Multi-query search  
  - HyDE (Hypothetical Document Embedding)  
- **Re-ranking**: Employs **BGE M3** for scoring articles using dense, sparse, and ColBERT embeddings.  
- **Regeneration**: A final stage for generating refined responses based on the top-ranked articles.

---

## 🖥️ Usage 
### Start the Milvus Server: Follow the Milvus documentation to set up and run the Milvus server. 
### Run the Gradio App:
```bash
python app/main.py
```

---
## 🎥 Demo Video Check out the demo of JURIS_AI in action: [Lien Vidéo]

---


## 📊 System Architecture The system architecture diagram illustrates the flow of data from legal document ingestion to response generation. [lien_architecture]

--- 


## 🤝 Contributing Contributions are welcome! 


