# Chat With Your Docs

## Introduction

---

The ChatWithYourDocs Chat App is a Python application that allows you to chat with multiple Docs formats like PDF, WEB pages and YouTube videos. You can ask questions about the PDFs using natural language, and the application will provide relevant responses based on the content of the documents. This app utilizes a language model to generate accurate answers to your queries. Please note that the app will only respond to questions related to the loaded Docs.

## How It Works

---

![ChatWithYourDocs Chat App Diagram](./docs/images/cwd_flow.png)

The application follows these steps to provide responses to your questions:

1. Doc Loading: The app reads multiple PDF Docs types and extracts their text content.

2. Text Chunking: The extracted text is divided into smaller chunks that can be processed effectively.

3. Language Model: The application utilizes a language model to generate vector representations (embeddings) of the text chunks.

4. Similarity Matching: When you ask a question, the app compares it with the text chunks and identifies the most semantically similar ones.

5. Response Generation: The selected chunks are passed to the language model, which generates a response based on the relevant content of the PDFs.

## Dependencies and Installation

---

To install the Chat With PDF App, please follow these steps:

1. Download [Ollama library](https://github.com/jmorganca/ollama)
   ```
   curl https://ollama.ai/install.sh | sh
   ```
2. pull the chat models we will use, in this case we will use [LLAMA2](https://ollama.ai/library/llama2), [MISTRAL](https://ollama.ai/library/mistral) and [GEMMA](https://ollama.ai/library/gemma)
   ```
   ollama pull llama2
   ```
   ```
   ollama pull mistral
   ```
   ```
   ollama pull gemma
   ```
3. Create new environment with python 3.9 and activate it, in this case we will use [conda](https://www.anaconda.com/download)

   ```
   conda create -n cwp python=3.9
   ```

   ```
   conda activate cwp
   ```

4. Clone the repository to your local machine.

   ```
   git clone https://github.com/jorge-armando-navarro-flores/chat_with_your_docs.git
   ```

   ```
   cd chat_with_your_docs
   ```

5. Install the required dependencies by running the following command:
   ```
   pip install -r requirements.txt
   ```

## Usage

---

To use the Chat With Your Docs app, follow these steps:

1. Run the `main.py` file using the Streamlit CLI. Execute the following command:

   ```
   python3 main.py
   ```

2. The application will launch in your default web browser, displaying the user interface.
   ![ChatWithYourDocs Interface](./docs/images/cwd_interface.png)
