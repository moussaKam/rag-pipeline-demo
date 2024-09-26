# **Wikipedia-based RAG System**

This repository demonstrates a **Retrieval-Augmented Generation (RAG) system** built using Wikipedia as a source of factual information. The system retrieves relevant content from Wikipedia and combines it with a language model to provide contextually accurate and enriched responses to user queries. This project is a powerful example of how RAG can mitigate hallucinations in language models, especially when dealing with languages with limited coverage, such as Arabic.

Checkout my my blog [here!](https://moussakam.github.io/demo/2024/09/26/arabic-rag.html)

## **Table of Contents**

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Files Description](#files-description)


## **Introduction**

This RAG system leverages Wikipedia content and a hybrid retrieval model, [BGE-M3](https://huggingface.co/BAAI/bge-m3), to enhance the factuality of responses produced by a language model. The project is designed to extract, chunk, encode, and retrieve relevant content dynamically based on user queries. The entire pipeline uses a combination of multithreaded web scraping, text chunking, document encoding, and large language models (LLMs).

The goal is to enable smaller LLMs to deliver more accurate results by augmenting them with external knowledge, thereby minimizing hallucinations and enhancing output quality.

## **Features**

- Extracts and retrieves relevant information from Wikipedia pages.
- Utilizes multithreaded scraping for efficient content retrieval.
- Implements chunk-based text processing for optimal embedding and retrieval.
- Supports hybrid document retrieval with BGE-M3 (lexical matching, dense embeddings, and multi-vector retrieval).
- Offers a simple, interactive [Gradio](https://gradio.app/) interface for querying and receiving responses.
- Works with both Arabic and English Wikipedia content.

## **Installation**

1. Clone the repository:
    ```bash
    git clone https://github.com/moussaKam/rag-pipeline-demo.git
    cd rag-pipeline-demo
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Verify installation:
    Ensure all packages are correctly installed and that the necessary models (e.g., `Qwen/Qwen2-7B-Instruct-awq` and `BAAI/bge-m3`) are downloaded.

## **Usage**

Run the `demo.py` file to see the RAG system in action.

### **Command-Line Arguments**

- `--title`: The title of the Wikipedia page to extract links from (default: `"الحرب العالمية الثانية"`).
- `--max_workers`: The maximum number of workers to use for multithreading (default: `5`).
- `--model_name`: The name of the model to use for generation (default: `Qwen/Qwen2-7B-Instruct-awq`).
- `--device`: The device to load the model onto (`cpu` or `cuda:0`) (default: `"cuda:0"`).
- `--wikipedia_language`: The language code for Wikipedia content (`ar` for Arabic, `en` for English) (default: `"ar"`).
- `--embedding_model`: The name of the embedding model used for retrieval (default: `"BAAI/bge-m3"`).
- `--chunk_size`: The size of text chunks for embedding (default: `300`).
- `--overlap`: Overlap between chunks to maintain context (default: `50`).
- `--num_chunks`: Number of top chunks to retrieve for context (default: `3`).

### **Example Command**

To run the RAG system using the default settings, execute:

```bash
python demo.py --title "الحرب العالمية الثانية"
```

## **Files Description**

### **1. `demo.py`**
- The main file for running the RAG system demo.


### **2. `retriever.py`**
- Implements the `Retriever` class for managing the document retrieval process.
- Uses a hybrid retrieval approach (lexical and dense similarity) based on the BGE-M3 model.
- Manages the embedding of text chunks and the ranking of results.

### **3. `requirements.txt`**
- Lists all the necessary libraries and dependencies for running the project.

