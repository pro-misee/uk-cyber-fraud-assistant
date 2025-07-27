# UK Cyber Fraud Chatbot - Data Collection & Fine-Tuning Pipeline

## Project Overview

Developed a comprehensive data collection and Q&A generation pipeline for fine-tuning LLaMA 2 to support UK cyber fraud victims. The approach combines automated web scraping with AI-powered Q&A generation to create training datasets from authoritative UK fraud guidance sources.

## Data Sources & Results

### Successfully Scraped and Processed Sources
- **Action Fraud**: 47 Q&A pairs from UK's national fraud reporting centre
- **GetSafeOnline**: 23 Q&A pairs from UK's leading internet safety resource  
- **FCA (Financial Conduct Authority)**: 34 Q&A pairs covering financial scams and consumer protection
- **UK Finance**: 7 Q&A pairs from industry fraud prevention strategies
- **Which**: Consumer protection guidance and fraud prevention advice

**Total Dataset**: 111 high-quality, victim-focused Q&A pairs ready for fine-tuning

### Quality Characteristics
- **Victim-focused questions**: All questions written from fraud victim perspective
- **Empathetic responses**: Supportive, non-judgmental tone throughout
- **UK-specific guidance**: Includes proper UK contact numbers and procedures
- **Source attribution**: Every Q&A pair linked to authoritative source material
- **Training-ready format**: Structured in Alpaca format for LLaMA 2 fine-tuning

## Technical Implementation

### 1. Web Scraping Framework (`scripts/`)

Built a scraping framework that works across different UK fraud guidance websites:

**Key Features:**
- **Configurable source management**: Easy addition of new fraud guidance websites
- **Intelligent content extraction**: Automatically extracts meaningful content from various site structures
- **Smart link following**: Discovers and scrapes relevant linked pages using fraud-related keywords
- **Respectful scraping**: 3-second delays between requests, respects robots.txt
- **Structured output**: Individual JSON files per page plus compiled datasets

**Site Configurations:**
```python
SITE_CONFIGS = {
    'actionfraud': {...},    # UK's national fraud reporting centre
    'getsafeonline': {...},  # UK's leading internet safety resource
    'fca': {...},           # Financial Conduct Authority 
    'ukfinance': {...},     # UK Finance industry body
    'which': {...},         # Consumer protection guidance
    'citizensadvice': {...} # Citizens Advice consumer guidance
}
```

### 2. Q&A Generation Process

Used Gemini AI with carefully crafted prompts to generate victim-focused Q&A pairs:

**Generation Pipeline:**
```
Compiled dataset → Gemini AI → Victim-focused Q&A pairs → JSON validation → Training dataset
```

**Process Details:**
- **Chunking strategy**: Split large datasets into 2-3 documents per chunk for optimal quality
- **Victim perspective**: Generated questions from fraud victim's point of view
- **Empathetic responses**: 100-300 word supportive answers grounded in source material
- **UK-specific guidance**: Embedded proper contact numbers and UK procedures
- **Quality validation**: Ensured proper JSON format for downstream processing

**Output Format:**
```json
[
  {
    "instruction": "I think I've been scammed online — what should I do immediately?",
    "input": "",
    "output": "I understand how distressing this must be for you. If you believe you've been scammed, here are the immediate steps you should take...",
    "source_document": "Action Fraud - What to do if you've been scammed",
    "source_url": "https://www.actionfraud.police.uk/...",
    "chunk_number": 1,
    "document_index": 1,
    "generated_by": "gemini"
  }
]
```

### 3. Model Training Implementation

I successfully fine-tuned LLaMA 2-7B using the generated dataset:

**Training Configuration:**
- **Base model**: NousResearch/Llama-2-7b-chat-hf
- **Method**: QLoRA (4-bit quantization with LoRA adapters)
- **Dataset**: 111 Q&A pairs from UK fraud guidance sources
- **Training approach**: Instruction tuning with victim support focus
- **Output**: Fine-tuned adapter (misee/uk-fraud-chatbot-llama2)

## Project Structure

```
uk-fraud-chatbot/
├── data_sources/                    # Organised by source website
│   ├── actionfraud/
│   │   ├── scraped/                # Raw scraped content
│   │   └── processed/gemini_qa_pairs/  # Generated Q&A pairs
│   ├── getsafeonline/
│   ├── fca/
│   ├── ukfinance/
│   ├── which/
│   └── citizensadvice/
├── scripts/
│   ├── scraper.py                  # Main scraping orchestrator
│   ├── site_scrapers.py            # Generic scraping framework + configurations
│   ├── combine_datasets.py         # Dataset combination utilities
├── model_training/
│   ├── master_fraud_qa_dataset.json    # Combined training dataset (111 Q&A pairs)
│   ├── train_fraud_qa_dataset.json     # Training split (80%)
│   ├── val_fraud_qa_dataset.json       # Validation split (20%)
│   └── dataset_info.json              # Dataset metadata
├── notebooks/
│   ├── LLaMA2_Fraud_Chatbot_FineTuning.ipynb  # Complete training pipeline
│   ├── Demo_Chatbot_Colab.ipynb              # Interactive demo notebook
├── QA_prompt_engineering.txt      # Gemini Q&A generation instructions
├── requirements.txt                # Python dependencies
└── README.md                      # This documentation
```

## Usage Instructions

### 1. Environment Setup
```bash
# Create and activate virtual environment
python3 -m venv fraud_chatbot_env
source fraud_chatbot_env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Collection
```bash
cd scripts

# Scrape specific source (modify scraper.py line 81 to change source)
python scraper.py  # Currently configured for UK Finance

# Available sources: actionfraud, getsafeonline, fca, ukfinance, which, citizensadvice
```

### 3. Q&A Generation
```bash
# Use the Gemini prompt with your scraped data
# 1. Copy content from model_processing_prompt.txt
# 2. Input scraped dataset into Gemini AI
# 3. Save generated Q&A pairs to data_sources/[source]/processed/gemini_qa_pairs/
```

### 4. Model Training
```bash
# Use the provided Colab notebook for training
# notebooks/LLaMA2_Fraud_Chatbot_FineTuning.ipynb contains the complete pipeline
```

### 5. Demo and Testing
```bash
# Run the interactive demo using the Colab notebook
# notebooks/Demo_Chatbot_Colab.ipynb provides a working chatbot interface
```

## Key Design Decisions

### 1. Generic Scraping Framework
I built a reusable scraper that works across different fraud guidance sites because fraud information comes from multiple authoritative sources. This framework allows easy expansion to new sources without rewriting scraping logic.

### 2. Gemini for Q&A Generation
I chose Gemini AI with carefully crafted prompts for Q&A generation because it produces more empathetic, victim-focused responses compared to template-based approaches. The model understands the emotional nuance needed for fraud victim support.

### 3. Source-Specific Organisation
I organized data by source website rather than topic to maintain proper attribution and allow for source-specific processing while enabling easy combination for training.

### 4. UK-Specific Focus
I ensured all responses include proper UK contact numbers, procedures, and legal context because generic responses are unhelpful in crisis situations. Fraud victims need specific, actionable guidance.

## Results and Achievements

### Dataset Quality
- **111 total Q&A pairs** across 5 authoritative UK sources
- **100% victim-focused questions** written from fraud victim perspective  
- **UK-specific guidance** with proper contact numbers and procedures
- **Source attribution** linking every Q&A pair to authoritative material
- **Training-ready format** structured for LLaMA 2 fine-tuning

### Model Performance
- Successfully fine-tuned LLaMA 2-7B for UK fraud victim support
- Model demonstrates understanding of UK-specific fraud procedures
- Responses include appropriate UK contact information and guidance
- Maintains empathetic, supportive tone consistent with training data

### Technical Infrastructure
- Scalable scraping framework supporting multiple UK fraud guidance sites
- Automated Q&A generation pipeline using state-of-the-art AI
- Complete training pipeline from data collection to model deployment
- Comprehensive documentation and reproducible implementation

## Data Sources & Attribution

### Authoritative UK Fraud Guidance Sources
- **Action Fraud**: UK's national fraud and cybercrime reporting centre (actionfraud.police.uk)
- **GetSafeOnline**: UK's leading internet safety website (getsafeonline.org)
- **FCA**: Financial Conduct Authority - UK financial regulator (fca.org.uk)
- **UK Finance**: Industry body representing UK financial services (ukfinance.org.uk)
- **Which**: UK consumer protection organization (which.co.uk)
- **Citizens Advice**: UK consumer guidance organization (citizensadvice.org.uk)

## Future Enhancements

**Expand data sources**: Process Citizens Advice content and identify additional UK fraud guidance sites

---