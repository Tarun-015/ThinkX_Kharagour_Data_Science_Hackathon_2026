Story Consistency Checker
Overview
The Story Consistency Checker is an NLP-based application designed to evaluate whether a character
specific event or action remains consistent with the broader narrative context of a story or novel. The
system compares global story information with localized character behavior and produces a clear
consistency verdict.
The project is structured with an emphasis on modularity, clarity, and separation of concerns, ensuring that
each component has a well-defined role within the overall pipeline.
System Architecture
At a high level, the application follows the pipeline below:
1. 
2. 
3. 
4. 
User provides story and character-related inputs through the UI
Text inputs are transformed into numerical embeddings
A trained classifier evaluates semantic compatibility
A consistency result is returned to the UI
User Input
   ↓
Text Embedding Generation
   ↓
Compatibility Classification
   ↓
Consistency Verdict
Each folder in the repository corresponds to one logical layer in this workflow.
Directory Structure
├── README.md
├── requirements.txt
├── environment.yml
├── main.py
├── streamlit_app.py
├── generate_results.py
├── results.csv
1
│
├── outputs/
│   ├── classifier.pth
│   └── memory.json
│
├── bdh/
│   ├── state.py
│   ├── update.py
│   └── __pycache__/
│
├── embedding/
│   ├── encoder.py
│   └── __pycache__/
│
├── model/
│   ├── compatibility_classifier.py
│   └── __pycache__/
│
└── input/
    ├── train.csv
    └── test.csv
UI Layer (
streamlit_app.py )
The Streamlit application serves as the user interface for the system. It allows users to input:
• 
• 
• 
• 
Full story or novel text
Character name
Character context or description
A specific character event or snippet
Upon user interaction, the UI forwards the collected inputs to the backend logic and displays the final
consistency result. No machine learning logic is implemented directly in this layer.
Core Controller (
main.py )
main.py functions as the central orchestrator of the application. Its responsibilities include:
• 
• 
• 
• 
• 
Receiving inputs from the UI
Coordinating embedding generation
Loading the trained classifier
Running inference
Returning results to the UI
2
This file connects independent components without embedding domain-specific logic within itself.
Embedding Layer (
embedding/encoder.py )
The embedding module converts raw textual inputs into numerical vector representations suitable for
machine learning models. This process captures semantic information such as meaning, tone, and
character traits.
The generated embeddings act as feature inputs for the classifier.
Model Layer (
model/compatibility_classifier.py )
This module defines the classification logic used to assess story consistency. It includes:
• 
• 
• 
Model architecture definition
Loading of trained weights from 
classifier.pth
Inference and decision threshold handling
The classifier outputs a consistency label based on the compatibility between story context and character
behavior.
State and Memory Management (
bdh/state.py , 
These modules handle internal state tracking and persistence:
• 
• 
state.py defines runtime data structures
bdh/update.py )
update.py records inference results and metadata into 
memory.json
This layer supports reproducibility, debugging, and future extensions such as history tracking or analytics.
Input Data (
input/ )
The input directory contains datasets and sample texts used for training, evaluation, and demonstration
purposes:
• 
• 
• 
train.csv for model training
test.csv for evaluation
Text files containing sample literary works
3
Outputs (
outputs/ )
This directory stores generated artifacts:
• 
• 
classifier.pth containing trained model weights
memory.json containing logged inference data
These files are produced programmatically and should not be manually modified.
End-to-End Execution Flow
User Input (UI)
      ↓
streamlit_app.py
      ↓
main.py
      ↓
encoder.py
      ↓
compatibility_classifier.py
      ↓
state/update
      ↓
Result Displayed in UI
Environment and Dependencies
Two environment configuration options are provided:
• 
• 
requirements.txt for pip-based installations
environment.yml for Conda-based reproducible environments
Both ensure consistent dependency management across systems.
Conclusion
The Story Consistency Checker is designed to be modular, interpretable, and extensible. By clearly
separating UI logic, text representation, model inference, and state management, the project remains
maintainable and suitable for both academic and applied use cases.
4