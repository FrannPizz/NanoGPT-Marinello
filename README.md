# NanoGPT Marinello

Prototype of a char-level LLM based on a Transformer decoder-only architecture, with multimodal pipeline (text and voice) and web interface.

This repository accompanies the Bachelor’s thesis:  
**“CONVERSING WITH ROBOTS: BUILDING LLM ASSISTANTS TO UNDERSTAND AND UTILIZE AUTONOMOUS SYSTEMS”** – Francesco Pizzato, Università degli Studi di Padova, 2025.  

---

## Features

- Train a Transformer-based language model **from scratch** on a custom Q&A dataset.  
- Chat with the model via **text** or **voice** (speech-to-text with Whisper, text-to-speech with gTTS).  
- Simple **web interface** built with Gradio.  
- Support for **GPU with CUDA** or **CPU-only** environments.  

---

## Requirements

- Python 3.10 or newer  
- PIP (updated)  
- Recommended: NVIDIA GPU with CUDA 11.8+  

Install dependencies from `requirements.txt`:  
```bash
pip install -r requirements.txt
```

---

## Quick Start

1. Clone the repository:  
   ```bash
   git clone https://github.com/[your-username]/NanoGPT-Marinello.git
   cd NanoGPT-Marinello
   ```

2. Launch the application:  
   ```bash
   python NanoGPT_Marinello.py
   ```

3. Open Gradio in your browser (default: [http://localhost:7860](http://localhost:7860)).

If a checkpoint file (e.g., `nanoGPT_checkpoint.pth`) is present in the same folder, the program will automatically load it at startup.  

---

## Dataset

The model is trained on a Q&A dataset with the following format:  

```
###Prompt: What is BlueBoat?
###Response: BlueBoat is an autonomous surface vessel developed as part of the NTNU projects...
```

The dataset can be extended with new domain-specific documents.  
A sample dataset is included in the repository.  

---

## Checkpoints

During training, checkpoints are automatically saved (model state, optimizer state, vocabulary).  
Placing the checkpoint file in the same directory as the main script is sufficient for it to be loaded automatically at program startup.  

---

## Technical details

- Transformer decoder-only  
- 8 layers, 8 attention heads, 512 embedding dimension  
- Context size: 128 tokens (~500–700 characters)  
- Char-level tokenization  
- Whisper for speech-to-text  
- gTTS for text-to-speech  
- Gradio for web interface  

---

## Reference

If you use this code or build upon it, please cite the thesis:

**Francesco Pizzato (2025)**  
*CONVERSING WITH ROBOTS: BUILDING LLM ASSISTANTS TO UNDERSTAND AND UTILIZE AUTONOMOUS SYSTEMS*  
Bachelor’s thesis, Università degli Studi di Padova.  

---

## Author

- **Francesco Pizzato** – [francesco.pizzato.1@studenti.unipd.it](mailto:francesco.pizzato.1@studenti.unipd.it)

---
