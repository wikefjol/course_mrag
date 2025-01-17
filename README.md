# Multi-Source Conversational QA System

## Overview
This project provides a **multi-agent conversational question-answering (QA) system** that integrates multiple knowledge sources and tools. The system processes user queries, generates answers from individual agents (retrieval-augmented generation, or RAG), and aggregates these responses using an advanced language model.

Key features include:
- **Multi-source retrieval**: Processes data from diverse domains such as books, lectures, and other structured sources.
- **Dynamic question answering**: Generates answers, follow-up questions, and aggregated responses.
- **Cross-referencing**: Verifies responses by cross-checking agents' outputs.
- **Extensibility**: Easily add new knowledge sources or improve components.

---

## Project Structure
```plaintext
.
├── multi_source_conversation.json         # Output JSON of the QA system
├── README.md                              # Project documentation
├── requirements.txt                       # Python dependencies
├── question_logs/                         # Logs for processed questions
│   ├── easier_questions/                  # Logs for easier questions
│   └── harder_questions/                  # Logs for more complex queries
├── sources/                               # Knowledge sources
│   ├── book/                              # Books and academic references
│   └── lectures/                          # Lecture notes and related PDFs
├── src/                                   # Core Python modules
│   ├── dialogue_tools.py                  # Main logic for RAG workflow
│   ├── vectorstore.py                     # Handles embeddings and vector storage
│   ├── preprocessing.py                   # Prepares data for processing
│   ├── rag_tools.py                       # RAG implementation
│   ├── utils.py                           # Helper functions
├── venv/                                  # Virtual environment for dependencies
```

---

## Core Workflow
1. **Configuration**:
   - Set up RAG agents for each knowledge source under `sources/`.
   - Define language models and embeddings in the configuration.

2. **Initial Response**:
   - Each RAG agent answers the original question based on its knowledge base.

3. **Follow-Up Questions**:
   - The aggregation LLM generates specific follow-up queries for each agent.

4. **Cross-Verification**:
   - Each agent will cross-reference another agents answers against its own sources and provide corrections/clarifications or just expand on the answers

5. **Final Aggregation**:
   - The aggregation LLM consolidates all responses into a cohesive final answer, which could be returned to user.

---

## Usage

### Setup
1. Clone the repository and navigate to the project directory:
   ```bash
   git clone https://github.com/wikefjol/course_mrag.git
   cd course_mrag
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. Populate the `sources/` directory with knowledge bases (e.g., PDFs, JSON files).
   One RAG will be created for each subfolder of sources. 
   For example: 
   sources/
   ├── book/
   │   └── book.pdf
   ├── lectures/
   │   ├── lecturenotes_1.pdf
   │   ├── lecturenotes_2.pdf
   │   ...
   │   └── lecturenotes_16.pdf
   └── presentations/
       ├── presentations_1.pdf
       ├── presentations_2.pdf
       ...
       └── presentations_16.pdf

4. Run through the notebook from the top. Vectorstores will only be updated if changing embedding models or making changes in data structure. 
   If the vectorstore for a specific combination of data and embedding model has been constructed earlier, it is loaded. First construction might take a couple of minutes. 

   How much time it takes for the system to process a question depends on which model you use. In my experience it takse approx one minute to run throguh 5 questions when using the following config: 

        "llm_name": "Llama3.2-3b",
        "emb_name": "hf-minilm-l6-v2",
        "k": 10,


### Running the System
- Use the notebook or scripts to:
  - Process questions.
  - Aggregate responses.
  - Save outputs in `question_logs/`.

- Example workflow:
  ```python
  from src.dialogue_tools import run_all_questions

  questions = [
      "How does synaptic plasticity contribute to memory?",
      "What role does the prefrontal cortex play in decision-making?",
  ]
  run_all_questions(questions)
  ```

### Outputs
- Final responses are stored as JSON in `multi_source_conversation.json`.
- Logs and intermediate responses are saved in the `question_logs/` directory.
