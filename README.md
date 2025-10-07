# Grant Discovery and Application Assistant

This project is designed to help nonprofit organizations identify relevant grant opportunities and generate draft applications that sound natural and reflect their mission and voice. The system learns from past grant materials and helps streamline the overall grant-seeking process.

## Overview

We built this system for Cambio Labs, a nonprofit focused on educational technology and social entrepreneurship. The project works by processing organizational documents, finding matching grants, and generating drafts that sound consistent with how the organization usually writes. The goal is to make the grant application process faster, more focused, and more accessible for nonprofit teams.

## Features

- **Document Analysis**: Reads and summarizes existing grant applications to understand tone, mission statements, and impact areas.
- **Grant Discovery**: Searches publicly available databases like Grants.gov and foundation listings to find matching opportunities.
- **Draft Generation**: Creates first-pass drafts using the organization’s voice, which can then be refined by the team.
- **Refinement Cycle**: Allows us to improve drafts over multiple rounds of feedback.
- **Local Execution**: Works entirely on local infrastructure without relying on external services.

## Tech Stack

- **Frontend**: Streamlit
- **Language Model**: Ollama running LLaMA 3.1 (8B)
- **Vector Storage**: ChromaDB
- **Embeddings**: BGE-small-en-v1.5 from HuggingFace
- **Web Scraping**: BeautifulSoup and Selenium

## Requirements

- Python 3.9 or higher
- At least 16GB of RAM (64GB recommended for heavier workloads)
- Ollama installed ([installation guide](https://ollama.ai))

## Quick Start

1. Clone the repository:

```bash
git clone https://github.com/yourusername/grant-assistant.git
cd grant-assistant
```

2. Install Ollama and pull the model:

```bash
ollama pull llama3.1:8b
```

3. Set up the Python environment:

```bash
pip install -r requirements.txt
```

4. Configure your environment variables:

```bash
cp .env.example .env
```
Then update the `.env` file with your local settings.

5. Run the application:

```bash
streamlit run app.py
```

Once launched, open the browser and visit [http://localhost:8501](http://localhost:8501).

## Project Structure

```
grant-assistant/
├── app.py                        # Main Streamlit application
├── requirements.txt              # Dependencies
├── .env.example                  # Environment template
├── README.md                     # Project documentation
│
├── src/
│   ├── processing/               # Document analysis
│   │   └── document_processor.py
│   ├── rag/                      # Retrieval and generation modules
│   │   ├── vector_store.py
│   │   └── retriever.py
│   ├── discovery/                # Grant search system
│   │   └── grant_scraper.py
│   └── generation/               # Draft generation logic
│       └── generator.py
│
├── config/                       # Configuration and settings
│   └── settings.py
│
└── examples/                     # Sample grant materials
    └── cambio_grants/
```

## How to Use

### Processing Documents
The system starts by analyzing grant applications in the `examples/` folder. This helps it learn the writing style and tone used by the organization.

### Discovering Grants
In the Grant Discovery tab, we can search for new opportunities based on focus areas or mission keywords.

### Generating Applications
After selecting a grant, the Application Generation tab lets us create a draft application using the voice patterns learned earlier.

### Refining Results
Through the Refinement tab, we can adjust and improve the drafts based on team feedback.

## Configuration

Main parameters can be adjusted in `config/settings.py`:
- Embedding model type
- Model generation parameters
- Vector database configuration
- Search behavior

## Contribution and Credits

This project was created as part of the Break Through Tech AI Studio program in collaboration with Cambio Labs.

- Cambio Labs provided organizational data and feedback.
- Break Through Tech AI Studio supported the project framework.
- Angelo Orciuoli served as a technical advisor.

## License

This project is intended for educational and nonprofit use.

## Contact

For questions, please reach out through the Break Through Tech AI Studio program.
