# ThinkX_Kharagour_Data_Science_Hackathon_2026


project/
│
├── data/
│   ├── train/
│   ├── test/
│   └── README.md
│
├── input/                       # 🔹 Live inputs for inference
│   ├── novel.txt                # narrative or story text
│   ├── question.txt             # backstory / character query
│
├── bdh/                         # 🔹 Persistent reasoning memory layer
│   ├── state.py                 # memory store + caching
│   ├── update.py                # manages incremental updates
│   └── graph_memory.py          # (optional) stores relation graphs
│
├── embedding/                   # 🔹 Semantic text encoding
│   ├── encoder.py               # converts text → embeddings
│   ├── preprocessor.py          # text cleaning / tokenization
│   └── __init__.py
│
├── model/                       # 🔹 Core Track B reasoning models
│   ├── compatibility_classifier.py   # predicts consistency
│   ├── conflict_detector.py          # finds contradictions
│   ├── missing_info_checker.py       # finds gaps in backstory
│   └── __init__.py
│
├── api/                         # 🔹 optional UI / evaluation layer
│   ├── app.py                   # FastAPI/Flask server
│   └── routes/
│       └── inference_route.py
│
├── utils/                       # 🔹 small helpers
│   ├── file_io.py               # read/write functions
│   ├── config.py                # model paths, constants
│   └── logger.py
│
├── outputs/                     # 🔹 results & logs
│   ├── results.json
│   ├── logs/
│   └── checkpoints/
│
├── requirements.txt
├── setup.sh
└── main.py                      # orchestrator (entry point)

 ## Contributor Notes 
 EDA Branch initialized by Sanskriti
