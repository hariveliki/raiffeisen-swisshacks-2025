# AI-Augmented Financial Advisor System

A multi-agent system for post-meeting analysis of financial advisor-client conversations. This system analyzes meeting transcripts, client data, and product portfolios to generate structured summaries, detect emotional patterns, identify unmet needs, and recommend products.

## Project Structure

```
.
├── app/                    # Main application code
│   ├── agents/            # Agent implementations
│   └── utils/             # Utility functions
├── config/                # Configuration files
├── data/                  # Data directory for input files
├── docs/                  # Documentation
├── misc/                  # Miscellaneous files
├── output/                # Generated reports and analysis
├── .env                   # Environment variables (not in git)
├── .env.example          # Example environment variables
├── main.py               # Main entry point
└── requirements.txt      # Python dependencies
```

## Overview

The system is composed of several specialized agents working together:

1. **Data Retrieval Agent**: Fetches and indexes relevant client and product information
2. **Dialogue Analysis Agent**: Analyzes the conversation transcript for topics, questions, and emotional cues
3. **Summarization Agent**: Creates structured summaries of the meeting
4. **Recommendation Agent**: Identifies unmet needs and suggests relevant products
5. **Orchestrator**: Coordinates the agents and generates the final report

## Requirements

- Python 3.8+
- OpenAI API key
- Dependencies listed in `requirements.txt`:
  - openai>=1.0.0
  - langchain>=0.1.0
  - langchain-openai>=0.0.1
  - langchain-community>=0.0.1
  - langchain-experimental>=0.0.1
  - python-dotenv>=1.0.0
  - docx2txt>=0.8
  - pandas>=2.0.0
  - openpyxl>=3.1.2
  - python-docx>=0.8.11
  - pydub>=0.25.1
  - numpy>=1.24.0
  - faiss-cpu>=1.7.4
  - chromadb>=0.4.18

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/financial-advisor-system.git
   cd financial-advisor-system
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file with your OpenAI API key:
   ```bash
   cp .env.example .env
   # Edit .env with your OpenAI API key
   ```

## Data Preparation

Place your data files in the `data/` directory:
- `client_state.xlsx`: Client financial information
- `product_portfolio.docx`: Product catalog
- `transcript.m4a`: Audio recording of the meeting (or text transcript)

For demo purposes, the system can run without these files by using simulated data.

## Usage

Run the system:
```bash
python main.py
```

The system will:
1. Load and index client and product data
2. Analyze the meeting transcript
3. Generate a structured summary
4. Identify unmet needs and recommend products
5. Create a comprehensive report

Reports and results are saved in the `output/` directory.

## Output

The system generates two types of output files:
1. **JSON analysis file**: Complete analysis results in structured format
2. **Text report file**: Human-readable report with key sections:
   - Client's Goals & Questions
   - Advisor's Analysis & Recommendations
   - Action Items & Next Steps
   - Client's Reactions/Concerns
   - Unmet Financial Needs
   - Product Recommendations
   - Suggested Next Steps

## Implementation Details

This system uses:
- LangChain for agent and chain orchestration
- OpenAI GPT models for language processing
- FAISS and ChromaDB for vector search and semantic retrieval
- Chain-of-Thought prompting for emotional analysis
- Structured summarization with prompt engineering

## Limitations

- Real audio transcription requires OpenAI Whisper (not implemented in this prototype)
- Analysis quality depends on the OpenAI model used (GPT-4 recommended)
- Simulated data is used when real data files are not available

## Future Improvements

- Implement real-time audio transcription with Whisper
- Add more sophisticated emotion detection
- Improve parsing of Excel and Word documents
- Add a web interface for viewing reports
- Fine-tune models on financial domain data
- Add support for multiple languages
- Implement real-time analysis during meetings
