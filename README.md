Story Consistency Checker

Overview

The Story Consistency Checker is an NLP-based application designed to evaluate whether a character
specific event or action remains consistent with the broader narrative context of a story or novel. The
system compares global story information with localized character behavior and produces a clear
consistency verdict.

The project is structured with an emphasis on modularity, clarity, and separation of concerns, ensuring that
each component has a well-defined role within the overall pipeline

------------------------------------
System Architecture
At a high level, the application follows the pipeline below:
1.  User provides story and character-related inputs through the UI
2.  Text inputs are transformed into numerical embeddings
3.  A trained classifier evaluates semantic compatibility
4.  A consistency result is returned to the UI
<img width="673" height="175" alt="image" src="https://github.com/user-attachments/assets/ff82a0c3-0385-460b-9fc7-ca057cb28a03" />

Each folder in the repository corresponds to one logical layer in this workflow.

---------------------------------------
Directory Structure

<img width="282" height="598" alt="image" src="https://github.com/user-attachments/assets/9cfeb927-dcb3-4274-bb65-53c15c22565b" />

---------------------------------------

UI Layer (streamlit_app.py )

The Streamlit application serves as the user interface for the system. It allows users to input:

>Full story or novel text

>Character name

>Character context or description

>A specific character event or snippet

Upon user interaction, the UI forwards the collected inputs to the backend logic and displays the final
consistency result. No machine learning logic is implemented directly in this layer

---------------------------------------
Core Controller (main.py )

main.py functions as the central orchestrator of the application. Its responsibilities include:

>Receiving inputs from the UI

>Coordinating embedding generation

>Loading the trained classifier

>Running inference

>Returning results to the UI

This file connects independent components without embedding domain-specific logic within itself.
