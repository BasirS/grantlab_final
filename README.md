# AI-Powered Grant Application Assistant

Building intelligent grant discovery and application generation for nonprofit organizations using local AI models and retrieval-augmented generation.

## Overview

This system helps Cambio Labs automatically discover relevant grant opportunities and generate draft applications using their organizational voice and past successful grant examples. The entire system runs locally using open-source models, maintaining data privacy while providing powerful AI capabilities.

## Key Features

**Document Processing**: Analyzes 47+ past grant applications to extract Cambio Labs' organizational voice, mission statements, and successful application patterns

**Grant Discovery**: Searches federal databases and foundation websites to identify relevant funding opportunities matching organizational priorities

**Application Generation**: Uses RAG (Retrieval-Augmented Generation) to draft grant applications maintaining authentic organizational voice and incorporating relevant past examples

**Application Refinement**: Iterative improvement system allowing review and enhancement of generated content

## Technical Architecture

**LLM**: Ollama with Llama 3.1 8B (local inference, zero API costs)

**Vector Database**: ChromaDB for persistent storage of document embeddings

**Embeddings**: HuggingFace BAAI/bge-small-en-v1.5 (runs locally on CPU)

**Web Interface**: Streamlit for rapid prototyping and user interaction

**Web Scraping**: BeautifulSoup and Selenium for grant discovery automation

## System Requirements

**Minimum Requirements**:
- 8GB RAM (16GB recommended)
- 10GB free disk space for models and data
- Modern CPU (Intel i5 or equivalent)

**Your System** (more than sufficient):
- Intel i7-8850H @ 2.60GHz
- 64GB RAM
- 4GB Graphics

## Installation

### Prerequisites

Install Ollama for local LLM inference:
```bash
# Download from https://ollama.ai
# After installation, pull the model:
ollama pull llama3.1:8b
```

### Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd grant-assistant

# Create environment file
cp .env.example .env

# Install Python dependencies
pip install -r requirements.txt
```

### Configuration

Edit `.env` file with your settings:
```
OLLAMA_MODEL=llama3.1:8b
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
CHROMA_PERSIST_DIR=./data/chroma
```

## Running the Application

```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501`

## Project Structure

```
grant-assistant/
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── .env.example               # Environment template
├── README.md                  # This file
│
├── src/
│   ├── processing/
│   │   └── document_processor.py    # Grant document analysis
│   ├── rag/
│   │   ├── vector_store.py          # ChromaDB operations
│   │   └── retriever.py             # Semantic search
│   ├── discovery/
│   │   └── grant_scraper.py         # Web scraping for grants
│   └── generation/
│       └── application_generator.py  # LLM-powered writing
│
├── config/
│   └── settings.py            # Configuration management
│
├── examples/
│   └── grants/                # Cambio Labs grant examples (47+ files)
│       ├── AWS/
│       ├── BRL/
│       └── AI_Economic/
│
└── data/                      # Created automatically
    ├── chroma/                # Vector database
    └── processed/             # Cached embeddings
```

## Usage Guide

### 1. Document Processing

Upload or select from 47+ existing Cambio Labs grant applications. The system extracts:
- Organizational mission and values
- Program descriptions and impact metrics
- Writing style and tone patterns
- Successful application structures

### 2. Grant Discovery

Search for relevant opportunities by:
- Keywords (e.g., "education technology", "workforce development")
- Funding amount ranges
- Geographic focus
- Application deadlines

### 3. Application Generation

Generate draft applications by:
- Selecting discovered grant opportunity
- Choosing relevant organizational programs to highlight
- Specifying grant-specific requirements
- Reviewing generated draft with citations to source documents

### 4. Application Refinement

Iteratively improve drafts through:
- Feedback on tone, structure, or content
- Adding specific details or metrics
- Adjusting emphasis on different programs
- Final review and export

## Development Timeline

**Phase 1 (Weeks 1-4)**: Local infrastructure setup, document processing pipeline, basic RAG implementation

**Phase 2 (Weeks 5-8)**: Grant discovery automation, application generation with organizational voice

**Phase 3 (Weeks 9-12)**: Refinement system, user interface polish, deployment preparation

## Key Technical Decisions

**Why Local Models?**
- Zero ongoing API costs
- Complete data privacy for sensitive grant documents
- Educational value in understanding RAG fundamentals
- Works offline without internet dependency

**Why ChromaDB?**
- Persistent vector storage across sessions
- Efficient similarity search on modest hardware
- Native Python integration with LlamaIndex
- Easy backup and version control

**Why Streamlit?**
- Rapid prototyping and iteration
- Python-native development
- Built-in state management
- Easy deployment options

## Troubleshooting

**Ollama connection failed**: Ensure Ollama service is running: `ollama serve`

**Out of memory errors**: Reduce batch size in `config/settings.py` or use smaller embedding model

**Slow document processing**: First run downloads embedding model (~400MB), subsequent runs are faster

**Missing grant examples**: Ensure `examples/grants/` directory contains sample documents

## Future Enhancements

- Mobile-responsive interface for grant review
- Email notifications for new grant opportunities
- Integration with Cambio Labs' existing CRM
- Multi-language support for international grants
- Advanced grant matching using machine learning

## Contributing

This is a student project for Cambio Labs. For questions or contributions, contact the development team.

## License

Educational project for Cambio Labs. Not licensed for commercial use.

## Acknowledgments

Built for Cambio Labs' mission to empower underestimated BIPOC youth and adults through technology and innovation.

**Technical Guidance**: Angelo Orciuoli (Cambio Labs)

**Academic Support**: Break Through Tech AI Studio Program

**Tools**: Ollama, LlamaIndex, ChromaDB, Streamlit, HuggingFace
