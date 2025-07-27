# Methodology

## 3.1 Research Design and Methodology Overview

This chapter presents a systematic approach for developing a conversational AI system to support UK cyber fraud victims through domain-specific fine-tuning of Large Language Models. The methodology employs a pragmatic research paradigm that focuses on technical implementation and performance evaluation, acknowledging that AI systems for vulnerable populations require both technical rigour and practical effectiveness.

The research follows a design science approach, combining experimental development with empirical evaluation to create and validate a novel artefact—a fine-tuned LLaMA 2 chatbot designed to address the current inadequacy of support mechanisms for UK cyber fraud victims. This approach responds directly to gaps identified in the literature regarding the adaptation of general-purpose language models for specialised applications requiring both factual precision and emotional sensitivity.

The methodology is structured around six core phases:
1. **Multi-source data collection framework** using automated web scraping
2. **AI-powered Q&A generation** using Gemini AI for dataset creation
3. **Model selection and configuration** with NousResearch LLaMA 2 variant
4. **Parameter-efficient fine-tuning implementation** using QLoRA methodology
5. **Interactive demonstration system** deployment via Google Colab
6. **Comprehensive evaluation framework** for UK-specific fraud support effectiveness

This systematic approach ensures reproducibility whilst addressing the specific requirements of creating empathetic, contextually-aware conversational AI for crisis support scenarios, demonstrated through a complete pipeline from data collection to working chatbot deployment.

---

## 3.2 Model Selection and Justification

### 3.2.1 Base Model Selection

**NousResearch/Llama-2-7b-chat-hf** is selected as the foundation model for this research, representing an optimised variant of the original Meta LLaMA 2 architecture.

**Technical Justification:**  
The NousResearch variant provides enhanced stability and compatibility for fine-tuning workflows compared to the original Meta implementation. This variant incorporates improved tokenizer configurations and training optimisations specifically designed for instruction-following tasks. The model demonstrates superior performance in conversational applications whilst maintaining computational efficiency essential for academic research constraints.

**Fine-tuning Compatibility:**  
Unlike the standard Meta variant, the NousResearch implementation includes preprocessing optimisations that facilitate more stable Parameter-Efficient Fine-Tuning (PEFT) implementations. The model's attention mechanism and layer structure are specifically configured to support effective adaptation through QLoRA techniques whilst preserving the broad knowledge acquired during pre-training.

**Community Validation:**  
The NousResearch variant has been extensively validated within the open-source machine learning community, demonstrating consistent performance across diverse fine-tuning scenarios. This community validation provides confidence in the model's reliability for research applications and ensures reproducibility of results.

**Accessibility and Implementation:**  
The model is available through the Hugging Face Transformers library with optimised configurations for academic use. Unlike proprietary alternatives, this open-source approach enables complete transparency in model behaviour and facilitates future extensions by other researchers.

### 3.2.2 Model Variant Selection

This research employs the **7B parameter** variant, which represents the optimal balance between computational feasibility and performance for this application. The Chat variant is specifically optimised for conversational applications, incorporating safety measures and dialogue optimisations essential for user-facing applications. Comparative analysis demonstrates that 7B parameter models achieve substantial performance gains over smaller variants whilst remaining deployable on standard GPU configurations available through Google Colab.

---

## 3.3 Data Collection Framework

### 3.3.1 Multi-Source Data Collection Strategy

**Authoritative Source Selection:**  
The data collection strategy targets five primary authoritative UK fraud guidance sources, selected based on their official status, comprehensive coverage, and regular updates:

1. **Action Fraud** (actionfraud.police.uk) - UK's national fraud reporting centre
2. **GetSafeOnline** (getsafeonline.org) - UK's leading internet safety resource
3. **Financial Conduct Authority** (fca.org.uk) - UK financial services regulator
4. **UK Finance** (ukfinance.org.uk) - Financial services industry body
5. **Which** (which.co.uk) - Consumer protection organisation

### 3.3.2 Automated Web Scraping Framework

**Generic Scraping Architecture:**  
A configurable scraping framework is implemented to accommodate diverse website structures whilst maintaining consistent data extraction quality. The framework employs a site-specific configuration approach that allows rapid adaptation to new sources without fundamental code modifications.

**Technical Implementation:**
```python
SITE_CONFIGS = {
    'actionfraud': {
        'base_url': 'https://www.actionfraud.police.uk',
        'test_urls': [
            'https://www.actionfraud.police.uk/what-is-action-fraud',
            'https://www.actionfraud.police.uk/types-of-fraud'
        ]
    },
    # Additional configurations for each source
}
```

**Content Extraction Methodology:**  
The scraping framework implements intelligent content extraction that:
- Removes navigation elements, headers, and footers automatically
- Identifies and extracts main content using CSS selectors and content analysis
- Follows relevant internal links using fraud-related keyword filtering
- Maintains source attribution and URL tracking for each extracted segment

**Compliance and Ethics:**  
All data collection adheres to ethical scraping practices:
- Rate limiting with 3-second delays between requests
- Respect for robots.txt directives
- Compliance with the Open Government Licence framework
- Full adherence to the Computer Misuse Act 1990
- Academic research user agent identification

### 3.3.3 Data Output and Compilation

**Structured Data Organization:**  
Scraped content is systematically organised by source with comprehensive metadata:
```
data_sources/[source_name]/
├── scraped/
│   ├── [source]_page_*.json      # Individual page content
│   ├── [source]_linked_*.json    # Linked page content
│   ├── compiled_dataset.json     # Combined source content
│   └── compiled_dataset.csv      # Alternative format
```

**Content Quality Assurance:**  
Each extracted document includes:
- Source URL and retrieval timestamp
- Content type and structure metadata
- Relevance scoring based on fraud-related keywords
- Duplicate detection and removal

---

## 3.4 AI-Powered Q&A Generation Pipeline

### 3.4.1 Gemini AI Integration for Dataset Creation

**Strategic Selection of Gemini AI:**  
Gemini AI is selected for Q&A generation based on its superior capability to produce empathetic, contextually-aware responses compared to template-based approaches. Unlike automated extraction tools, Gemini demonstrates understanding of emotional nuance essential for fraud victim support scenarios.

**Prompt Engineering Methodology:**  
A comprehensive prompt is developed to guide Gemini in generating victim-focused Q&A pairs:

```
You are tasked with creating question-answer pairs for training a UK cyber fraud victim support chatbot. Generate realistic questions that fraud victims might ask, with empathetic, practical answers based strictly on the provided UK fraud guidance documents.

Requirements:
- Questions from fraud victim perspective
- Answers 100-300 words, empathetic yet practical
- Include UK-specific contact information and procedures
- Maintain supportive, non-judgmental tone
- Ensure JSON format compatibility
```

### 3.4.2 Chunking Strategy for Optimal Quality

**Document Processing Approach:**  
Large compiled datasets are strategically divided into 2-3 document chunks to optimise Gemini's processing quality. This chunking approach prevents information overload whilst ensuring comprehensive coverage of source material.

**Quality Control Methodology:**  
Each generated Q&A pair undergoes validation for:
- Factual accuracy against source material
- Appropriate empathetic tone
- UK-specific guidance inclusion
- Proper JSON formatting
- Victim perspective authenticity

### 3.4.3 Output Format and Structure

**Alpaca-Compatible Format:**  
Generated Q&A pairs conform to the Alpaca instruction-tuning format for seamless integration with fine-tuning pipelines:

```json
[
  {
    "instruction": "I think I've been scammed online — what should I do immediately?",
    "input": "",
    "output": "I understand how distressing this must be for you. If you believe you've been scammed, here are the immediate steps you should take: First, don't panic - you're taking the right step by seeking help...",
    "source_document": "Action Fraud - What to do if you've been scammed",
    "source_url": "https://www.actionfraud.police.uk/...",
    "chunk_number": 1,
    "document_index": 1,
    "generated_by": "gemini"
  }
]
```

**Dataset Compilation Results:**  
The generation process produces:
- **Action Fraud**: 47 Q&A pairs
- **GetSafeOnline**: 23 Q&A pairs  
- **FCA**: 34 Q&A pairs
- **UK Finance**: 7 Q&A pairs
- **Which**: (additional source coverage)
- **Total**: 111 comprehensive Q&A pairs

---

## 3.5 Dataset Preparation and Preprocessing

### 3.5.1 Multi-Source Dataset Integration

**Source-Specific Processing:**  
Each data source undergoes individual processing to maintain attribution and enable source-specific analysis:
```
data_sources/[source]/processed/gemini_qa_pairs/
├── chunk_*_qa_pairs.json        # Individual chunk outputs
└── final_combined_qa_pairs.json # Source-specific compilation
```

**Master Dataset Creation:**  
Individual source datasets are systematically combined into training-ready formats:
- `master_fraud_qa_dataset.json` - Complete 111 Q&A pairs
- `train_fraud_qa_dataset.json` - Training split (80% - 89 pairs)
- `val_fraud_qa_dataset.json` - Validation split (20% - 22 pairs)

### 3.5.2 Data Quality Characteristics

**Victim-Focused Question Design:**  
All questions are crafted from the fraud victim's perspective, reflecting authentic scenarios:
- Immediate post-incident confusion and distress
- Practical concerns about financial impact
- Procedural questions about reporting and recovery
- Emotional support needs during crisis situations

**UK-Specific Response Integration:**  
Responses consistently include:
- Action Fraud contact information (0300 123 2040)
- Emergency services guidance (999)
- UK-specific legal and procedural context
- Appropriate escalation pathways
- Regional variations where applicable

### 3.5.3 LLaMA 2 Chat Template Implementation

**Instruction Formatting:**  
Each Q&A pair is formatted according to LLaMA 2's chat template requirements:

```
<s>[INST] <<SYS>>
You are a specialized UK fraud victim support assistant. Provide immediate, practical guidance to cyber fraud victims. Maintain an empathetic, supportive tone whilst ensuring accuracy of UK-specific procedures and contact information.
<</SYS>>

{user_question} [/INST] {assistant_response} </s>
```

**System Prompt Optimization:**  
The system prompt is carefully crafted to:
- Establish UK-specific context
- Emphasise empathetic communication
- Ensure practical guidance provision
- Maintain appropriate professional boundaries

---

## 3.6 Fine-Tuning Methodology

### 3.6.1 Parameter-Efficient Fine-Tuning Implementation

**QLoRA Configuration:**  
The fine-tuning implementation employs Quantised Low-Rank Adaptation (QLoRA) for memory-efficient training:

```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)
```

**LoRA Parameters:**  
- **Rank (r)**: 16 - Optimal balance between expressivity and efficiency
- **Alpha (α)**: 32 - Learning rate scaling factor
- **Dropout**: 0.1 - Regularisation to prevent overfitting
- **Target Modules**: Query, key, value, and output projection layers

### 3.6.2 Training Configuration and Implementation

**Optimisation Strategy:**  
```python
training_arguments = TrainingArguments(
    output_dir="./uk-fraud-chatbot-llama2-lora",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    warmup_ratio=0.1,
    logging_steps=5,
    evaluation_strategy="steps",
    eval_steps=10,
    save_strategy="epoch",
    fp16=True,
    gradient_checkpointing=True,
)
```

**Hardware Infrastructure:**  
- **Platform**: Google Colab with GPU acceleration
- **GPU Types**: T4, A100, or V100 depending on availability
- **Memory Management**: 4-bit quantisation enables training on 12-16GB VRAM
- **Session Management**: Checkpoint saving to handle Colab session limitations

### 3.6.3 Model Integration and Deployment

**Adapter Integration:**  
The fine-tuned LoRA adapter is integrated with the base model for inference:

```python
base_model = AutoModelForCausalLM.from_pretrained(
    "NousResearch/Llama-2-7b-chat-hf",
    quantization_config=bnb_config,
    device_map="auto"
)

model = PeftModel.from_pretrained(base_model, "misee/uk-fraud-chatbot-llama2")
```

**Hugging Face Hub Integration:**  
The trained adapter is published to Hugging Face Hub (`misee/uk-fraud-chatbot-llama2`) for:
- Reproducibility and sharing
- Version control and documentation
- Easy integration in demonstration systems

---

## 3.7 Interactive Demonstration System

### 3.7.1 Google Colab Deployment Architecture

**Gradio Interface Implementation:**  
A comprehensive demonstration interface is developed using Gradio to provide interactive access to the fine-tuned model:

```python
demo = gr.ChatInterface(
    chat_with_model,
    title="UK Fraud Support Chatbot Demo",
    description="Fine-tuned LLaMA 2 for fraud victim support",
    additional_inputs=[
        gr.Textbox(value="System message", label="System message"),
        gr.Slider(minimum=100, maximum=800, value=400, label="Max tokens"),
        gr.Slider(minimum=0.1, maximum=1.0, value=0.7, label="Temperature"),
        gr.Slider(minimum=0.1, maximum=1.0, value=0.9, label="Top-p"),
    ]
)
```

**Generation Parameter Optimisation:**  
Careful tuning of generation parameters ensures optimal response quality:
- **Max tokens**: 400 (sufficient for comprehensive responses)
- **Temperature**: 0.7 (balanced creativity and consistency)
- **Top-p**: 0.9 (nucleus sampling for quality control)
- **Repetition penalty**: 1.1 (reduces repetitive content)

### 3.7.2 Response Quality Optimisation

**Text Generation Pipeline:**  
The inference pipeline includes several optimisations to ensure complete, coherent responses:

```python
outputs = model.generate(
    **inputs,
    max_new_tokens=max_tokens,
    do_sample=True,
    temperature=temperature,
    top_p=top_p,
    repetition_penalty=1.1,
    no_repeat_ngram_size=3,
    pad_token_id=tokenizer.eos_token_id,
    eos_token_id=tokenizer.eos_token_id,
)
```

**Post-Processing Quality Control:**  
Automated post-processing addresses common generation issues:
- Incomplete sentence detection and removal
- Response length validation
- Content appropriateness verification

---

## 3.8 Evaluation Framework

### 3.8.1 Functional Evaluation Methodology

**UK-Specific Accuracy Assessment:**  
Responses are evaluated for accurate inclusion of:
- Correct contact numbers (Action Fraud: 0300 123 2040)
- Appropriate emergency guidance (999 for immediate threats)
- UK-specific legal and procedural context
- Accurate reporting and escalation pathways

**Empathetic Response Evaluation:**  
Assessment of emotional intelligence and support quality:
- Acknowledgment of victim distress
- Supportive, non-judgmental language
- Appropriate reassurance without minimising concerns
- Balanced emotional support with practical guidance

### 3.8.2 Comparative Analysis Framework

**Baseline Comparison:**  
Systematic comparison with general-purpose models:
- GPT-3.5-turbo responses to identical scenarios
- Assessment of UK-specific knowledge gaps
- Evaluation of empathetic response quality
- Analysis of practical guidance effectiveness

**Scenario-Based Testing:**  
Comprehensive testing across fraud categories:
- Online shopping scams
- Romance fraud scenarios  
- Investment and pension scams
- Identity theft and data breaches
- Phishing and social engineering

---

## 3.9 Technical Implementation Environment

### 3.9.1 Development Stack and Dependencies

**Core Machine Learning Framework:**
```python
torch>=2.0.0                    # PyTorch deep learning framework
transformers>=4.35.0            # Hugging Face transformers
peft>=0.6.0                     # Parameter-efficient fine-tuning
bitsandbytes>=0.41.0            # Quantisation support
accelerate>=0.24.1              # Training acceleration
```

**Data Collection and Processing:**
```python
requests>=2.31.0                # HTTP requests for web scraping
beautifulsoup4>=4.12.0          # HTML parsing and extraction
pandas>=2.0.0                   # Data manipulation and analysis
datasets>=2.12.0                # Hugging Face datasets integration
```

**Interactive Interface:**
```python
gradio>=4.0.0                   # Web-based demonstration interface
spaces>=0.28.0                  # Hugging Face Spaces integration
```

### 3.9.2 Infrastructure and Deployment

**Google Colab Configuration:**  
- **GPU Access**: T4, A100, or V100 depending on availability
- **Memory Management**: 4-bit quantisation enables 12-16GB operations
- **Storage**: Google Drive integration for persistent model storage
- **Session Management**: Automatic checkpoint saving and recovery

**Reproducibility Framework:**  
- Fixed random seeds for deterministic training
- Comprehensive logging of hyperparameters and configurations
- Version control integration with Git
- Complete documentation of environment configurations

---

## 3.10 Ethical Considerations and Compliance

### 3.10.1 Data Protection and Privacy

**UK GDPR Compliance:**  
All data collection and processing adheres to UK GDPR requirements:
- Lawful basis established under legitimate interest for academic research
- Data minimisation principles applied throughout collection process
- No personal data collection or retention beyond public sources
- Clear documentation of data sources and processing purposes

**Content Anonymisation:**  
Systematic removal of any personal information from scraped content:
- Automated scanning for personal identifiers
- Manual review of generated Q&A pairs
- Exclusion of case studies containing personal details

### 3.10.2 Ethical AI Development

**Bias Mitigation Strategies:**  
- Multi-source data collection to avoid single-perspective bias
- Evaluation across diverse fraud scenarios and victim demographics
- Regular assessment of response appropriateness across different contexts
- Transparency about model limitations and appropriate use cases

**Responsible Deployment:**  
- Clear labelling of AI-generated content
- Explicit guidance about limitations and appropriate use
- Emphasis on human oversight and professional support
- Regular monitoring for potential misuse or harmful outputs

---

## 3.11 Limitations and Methodological Considerations

### 3.11.1 Data Collection Limitations

**Source Coverage Constraints:**  
Focus on official UK sources provides authoritative guidance but may miss:
- Victim perspective narratives and lived experiences
- Emerging fraud types not yet covered by official sources
- Regional variations in support services and procedures
- Community-based support resources and informal guidance

**Temporal Limitations:**  
- Dataset represents snapshot at time of collection
- Fraud landscape evolves rapidly with new techniques
- Contact information and procedures may change over time
- Requires regular updates to maintain accuracy

### 3.11.2 Technical and Resource Constraints

**Model Size Considerations:**  
The 7B parameter model represents a compromise between:
- Computational feasibility within academic resource constraints
- Performance quality sufficient for demonstration purposes
- Potential for enhanced capability with larger model variants
- Memory and processing limitations of available infrastructure

**Evaluation Scope:**  
Limited evaluation framework due to:
- Resource constraints preventing large-scale user studies
- Ethical considerations around testing with actual fraud victims
- Reliance on proxy measures rather than real-world effectiveness
- Need for longitudinal studies to assess long-term impact

### 3.11.3 Generalisation and Scalability

**Geographic Specificity:**  
UK-focused approach limits:
- Transferability to other jurisdictions with different legal frameworks
- Applicability to international fraud scenarios
- Cultural adaptation requirements for different contexts
- Language and terminology variations across regions

**Scale Considerations:**  
Current implementation addresses:
- Proof-of-concept development rather than production deployment
- Limited concurrent user capacity
- Manual quality control processes requiring automation
- Infrastructure requirements for large-scale deployment

---

## 3.12 Conclusion

This methodology presents a comprehensive framework for developing a domain-specific conversational AI system for UK cyber fraud victim support. The approach successfully demonstrates the integration of automated data collection, AI-powered dataset generation, parameter-efficient fine-tuning, and interactive deployment within academic research constraints.

**Key Methodological Contributions:**
1. **Scalable data collection framework** enabling systematic acquisition from multiple authoritative sources
2. **AI-powered Q&A generation** using Gemini to create empathetic, victim-focused training data
3. **Optimised fine-tuning pipeline** balancing performance with computational efficiency
4. **Interactive demonstration system** providing practical validation of approach effectiveness
5. **Comprehensive evaluation framework** addressing both technical and ethical considerations

**Reproducibility and Extension:**  
The methodology is designed for reproducibility and extension by future researchers:
- Complete documentation of all implementation details
- Open-source codebase with modular architecture
- Clear guidelines for adapting to new sources or domains
- Systematic approach enabling replication with different model variants

**Academic and Practical Impact:**  
This methodology bridges the gap between theoretical AI research and practical application in crisis support scenarios. The systematic approach demonstrates how domain-specific fine-tuning can create more effective, empathetic AI systems whilst maintaining ethical standards and technical rigour essential for academic research.

The resulting framework provides both a functioning prototype and a replicable methodology for developing conversational AI systems for vulnerable populations, contributing to the growing field of AI for social good whilst addressing critical gaps in cyber fraud victim support infrastructure.