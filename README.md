# 📝🗨️ DocTalk

Welcome to DocTalk! This project leverages advanced language models to create a powerful chatbot that can interact with PDF documents. The application is built using Streamlit, LangChain, and other modern technologies to provide an intuitive and interactive user experience.

## 🌟 Features

- **Upload and Chat with PDFs**: Easily upload your PDF documents and ask questions to extract relevant information.
- **Advanced Language Model**: Powered by the Llama3-8b-8192 model from Groq.
- **Efficient Text Processing**: Utilizes FAISS for similarity search and vector indexing.
- **Google Generative AI Embeddings**: Enhanced document understanding with Google's state-of-the-art embeddings.
- **Streamlit Integration**: Simple and interactive user interface built with Streamlit.
- **WhatsApp Integration**: Seamlessly integrate the chatbot with WhatsApp for enhanced accessibility (future work).

## 🚀 Quick Start

### Prerequisites

Ensure you have Python installed. Recommended version: Python 3.8 or higher.

### Installation

1. **Clone the repository:**

    ```sh
    git clone https://github.com/your-username/doctalk.git
    cd doctalk
    ```

2. **Create virtual environment:**

    ```sh
    conda create -p venv python==3.12
    conda activate venv/
    ```

3. **Install the required dependencies:**

    ```sh
    pip install -r requirements.txt
    ```

4. **Set up environment variables:**

    Create a `.env` file in the root directory and add your API keys:

    ```plaintext
    GROQ_API_KEY=your_groq_api_key
    GOOGLE_API_KEY=your_google_api_key
    ```

5. **Run the application:**

    ```sh
    streamlit run Main.py
    ```

6. **Upload your PDF and start chatting!**

## 📚 Project Structure

- `Main.py`: Main application file containing the Streamlit code and chatbot logic.
- `requirements.txt`: List of dependencies required for the project.
- `.env.example`: Example of the environment variables file.
- `data/`: Directory to store uploaded PDF files.

## 📖 How It Works

1. **Upload a PDF**: Use the file uploader to select and upload a PDF document.
2. **Embed the Document**: The PDF is processed, and text is extracted and split into chunks for efficient querying.
3. **Ask Questions**: Enter your question related to the document.
4. **Get Answers**: The chatbot processes your question using the LLM and retrieves relevant information from the PDF.

## 🛠️ Technologies Used

- **Streamlit**: For building the web interface.
- **PyPDF2**: For PDF processing.
- **LangChain**: For handling language model interactions.
- **FAISS**: For efficient similarity search and vector indexing.
- **Google Generative AI**: For advanced text embeddings.
- **dotenv**: For managing environment variables.

## 🔧 Future Work

- **WhatsApp Integration**: Extend the chatbot to work with WhatsApp for more accessible interactions.
- **Enhanced PDF Processing**: Improve the text extraction and processing capabilities for better accuracy.

## 🌟 Contributions

We welcome contributions! Please fork the repository and submit a pull request to contribute to the project.

## 📫 Contact

For any inquiries, please contact [harshitwaldia112@gmail.com](mailto:harshitwaldia112@gmail.com).

## 📚 About RAG (Retrieval-Augmented Generation)

The chatbot in DocTalk utilizes a Retrieval-Augmented Generation (RAG) approach, which combines retrieval-based and generation-based techniques to enhance performance, especially for tasks involving large-scale or complex documents like PDFs.

### Key Aspects of RAG in DocTalk

- **Document Retrieval**: Utilizes FAISS for efficient similarity search and vector indexing to retrieve relevant text passages from PDF documents.
- **Generation with Context**: Employs the Groq-powered language model to generate contextually relevant answers based on the retrieved text.
- **Enhanced Understanding**: Integrates Google Generative AI Embeddings for semantic understanding and processing of document text, improving the accuracy and relevance of responses.

### Advantages of RAG in DocTalk

- **Improved Accuracy**: By combining retrieval with generation, the chatbot provides more accurate and contextually relevant responses.
- **Efficient Processing**: The RAG approach allows for efficient handling of large documents without requiring the entire document to be processed at once.
- **Enhanced User Experience**: Users can interact with PDF documents seamlessly, extracting information quickly and effectively.

DocTalk effectively leverages RAG principles to provide an efficient and powerful chatbot experience for interacting with PDF documents.
