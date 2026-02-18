# H-MAPS: Hierarchical Memory-Augmented Proactive Search Assistant

This repository contains the demonstration code for the paper:
**"H-MAPS: Hierarchical Memory-Augmented Proactive Search Assistant for Scientific Literature"** (Submitted to SIGIR '26).

H-MAPS is a proactive literature exploration assistant that resolves context ambiguity by leveraging a three-layered hierarchical memory. 

## 📂 Repository Structure

- `demo_main.py`: The main entry point for the H-MAPS system.
- `demo_main_reproduct.py`: Same as demo_main.py, but includes arguments and logic for reproducing the demo scenario.
- `demo_backend.py`: Backend logic for OCR, Retrieval (S2ORC), and Memory Management.
- `demo_gui.py`: User interface implementation (PySide6).
- `PROMPTS.md`: The specific LLM prompts used for question generation and memory management, as described in the paper.
- `data/`: Directory containing demo logs (JSON, JSONL).

## 🚀 Getting Started

### Prerequisites
- Python 3.12+
- Required packages: `PySide6`, `mss`, `pyocr`, `numpy`, `openai`, `pynput`, `Pillow`
- (Optional for full local mode) `llama-cpp-python`, `torch`, `transformers`, `faiss-cpu`

### Running the Demo (Mock Mode)
To reproduce the scenarios shown in the demonstration video without loading the full S2ORC index (approx. 70GB), use the `--demo` flag with the provided logs. This mode bypasses heavy initialization and simulates the retrieval results.

**Scenario 1: NLP Researcher**
```bash
python demo_main_reproduct.py --demo data/nlp_search_log.jsonl
```

**Scenario 2: HCI Researcher**
```bash
python demo_main_reproduct.py --demo data/hci_search_log.jsonl
```
