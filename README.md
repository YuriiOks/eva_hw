# 🏥 **JARVIS AI - NHS Navigator**
## *Revolutionizing Healthcare Communication with AI*

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25+-red.svg)](https://streamlit.io)
[![LLaMA](https://img.shields.io/badge/LLaMA-2_7B-green.svg)](https://llama.meta.com) <!-- Placeholder for actual model used -->
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE) <!-- Assuming MIT License -->

---

## 🎯 **Project Overview**

**JARVIS AI - NHS Navigator** is an intelligent healthcare assistant designed to revolutionize patient-NHS interactions. This project demonstrates how AI can address systemic healthcare inefficiencies, specifically targeting the common issue where patients face repetitive questioning and communication barriers during hospital visits. The system aims to reduce patient stress, improve staff efficiency, and enhance overall healthcare quality.

### **The Problem Addressed**
- Significant inefficiencies within large healthcare systems like the NHS.
- Repetitive questioning leading to patient anxiety and staff burnout.
- Language barriers hindering effective communication for a segment of patients.
- Long wait times and complex navigation in emergency departments.

### **The Solution: JARVIS AI**
An AI-powered virtual NHS assistant that:
- 🤖 **Predicts and prepares** patients for NHS interactions by familiarizing them with common questions.
- 🗣️ **Offers a multi-modal interface**, including text chat, voice recognition (simulated), and an avatar interface (simulated).
- 🛡️ **Emphasizes resilience and patient safety** through robust error handling, fallbacks to verified NHS guidance, and safety disclaimers.
- 📊 **Aims to reduce inefficiency** and improve the patient journey.

---

## ✨ **Key Features**

*   **AI-Powered NHS Guidance:** Utilizes a (simulated) fine-tuned LLaMA 2 model for providing information about NHS processes and answering patient queries.
*   **Multi-Modal Interface:**
    *   **Text Chat:** For direct interaction.
    *   **Voice Assistant (Simulated):** Placeholder for voice input and output capabilities.
    *   **Avatar Simulation (Simulated):** A virtual NHS receptionist ("Sarah") to guide patients through common questions.
*   **Progressive Milestones:** Demonstrates functionality through three integrated Streamlit applications:
    1.  Basic Text-Based Chatbot
    2.  Voice-Enabled Assistant (Conceptual)
    3.  Full Avatar Simulation (Conceptual)
*   **Comprehensive Error Handling & Resilience:**
    *   **Circuit Breaker:** Prevents repeated calls to a failing AI model.
    *   **Offline Mode:** Falls back to cached, verified NHS guidance if the AI model is unavailable.
    *   **User-Friendly Error Messages:** Designed to inform without causing patient anxiety.
    *   **Input Sanitization & Disclaimers:** Basic safety measures for inputs and outputs.
    *   **Emergency Keyword Detection:** Identifies critical symptoms and advises users to seek immediate medical help.
*   **Audit Logging:** Records key interactions for review and compliance (PII-conscious).
*   **Modular Design:** Built with a clear project structure for better maintainability.

---

## 🏗️ **Project Architecture**

The system is built around a (simulated) fine-tuned LLaMA 2 model, accessed via a Python backend. Streamlit is used for the web interface.

```
🧠 Fine-tuned LLaMA 2 (7B) ← Core Intelligence (Simulated)
    ↓
🔧 LoRA Adapter (NHS Knowledge) ← Specialized Training (Simulated)
    ↓
🎤 Google Speech API (Conceptual) ← Voice Input/Output
    ↓
👤 Avatar Interface (Pillow-based) ← Visual Interaction
    ↓
📱 Streamlit Web App ← User Interface
```
Key Python libraries include `transformers`, `streamlit`, `Pillow`. For a full list, see `requirements.txt`.

---

## 📁 **Project Structure**

```
nhs-navigator-ai/
├── 📄 README.md                     # This file
├── 📄 requirements.txt              # Python dependencies
├── 📄 .env.example                  # Environment variables template
│
├── 📁 data/                         # Datasets (NHS Q&A, scenarios, etc.)
├── 📁 src/                          # Core source code
│   ├── 📁 avatar/                   # Avatar simulation logic
│   ├── 📁 inference/                # AI model interaction (assistant, loader)
│   ├── 📁 training/                 # Model training scripts (placeholders)
│   ├── 📁 utils/                    # Error handling and other utilities
│   └── 📁 voice/                    # Voice processing (placeholders)
│
├── 📁 app/                          # Streamlit applications (main, milestones)
├── 📁 assets/                       # Static files (images, etc.)
├── 📁 logs/                         # Application and audit logs
├── 📁 models/                       # Model storage (placeholders)
├── 📁 scripts/                      # Utility and test scripts
└── 📁 tests/                        # Test suites (e.g., for resilience)
```

---

## ⚙️ **Setup and Installation**

1.  **Clone the repository (if applicable):**
    ```bash
    # git clone <repository_url>
    # cd nhs-navigator-ai
    ```

2.  **Create a Python virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *Note: Some dependencies related to specific hardware (like CUDA for GPU, or PyAudio for microphone) might require additional system libraries. The application uses placeholders for these functionalities, so direct hardware access is not strictly required to run the UI in a simulated mode.*

4.  **Set up environment variables:**
    Copy `.env.example` to `.env` and fill in any necessary values (though most are placeholders for this simulation):
    ```bash
    cp .env.example .env
    ```
    Key placeholders in `.env`:
    *   `GOOGLE_APPLICATION_CREDENTIALS`: Path to Google Cloud service account JSON (for voice features - conceptual).
    *   `WANDB_API_KEY`: For Weights & Biases logging during model training (conceptual).
    *   `MODEL_NAME`: Specifies the base language model (defaults to a LLaMA 2 placeholder).

---

## 🚀 **Running the Application**

The main application integrates all milestones and can be launched via Streamlit:

```bash
streamlit run app/main.py
```

This will start the web server, and you can access the application in your browser (usually at `http://localhost:8501`).

You can also launch specific milestone applications (though `app/main.py` is recommended):
```bash
# Example for Milestone 1 (Basic Chat)
streamlit run app/milestone1_basic.py

# Example for Milestone 3 (Avatar Simulation)
streamlit run app/milestone3_avatar.py
```

*(Note: Actual AI model responses, voice interactions, and advanced avatar animations are simulated with placeholders in the current version.)*

---

## 🧪 **Testing**

### **Resilience Tests**
A dedicated test suite (`tests/test_healthcare_resilience.py`) using Python's `unittest` module is provided to test error handling, fallbacks, circuit breaker logic, and patient safety features.

To run these tests:
```bash
python -m unittest tests/test_healthcare_resilience.py
```

### **Validation Scripts**
The `scripts/` directory contains various scripts for validation and testing:
*   `scripts/final_validation.py`: Performs a series of checks on system health, core functionality, fallbacks, and audit logging (uses simulated interactions).
    ```bash
    python scripts/final_validation.py
    ```
*   `scripts/health_monitor.py`: Simulates a health monitoring service for the application.
    ```bash
    python scripts/health_monitor.py
    ```

---

##🛡️ **Error Handling & Resilience Features**

This project places a strong emphasis on reliability and patient safety:

*   **Custom Error Handling:** Specific exceptions and an `ErrorHandler` class manage issues gracefully.
*   **Circuit Breaker:** Prevents the system from repeatedly trying a failing AI model, switching to fallbacks.
*   **Offline Mode:** Provides basic, verified NHS guidance from a local cache if the AI model is unavailable.
*   **Retry Mechanisms:** Decorators like `@retry_on_failure` are used for transient errors (e.g., conceptual network issues during model resource loading).
*   **Safe Execution:** Utilities like `safe_execute` ensure parts of the system can fail without crashing the entire application.
*   **User-Friendly Messages:** Errors are translated into messages that aim to inform users clearly without causing undue anxiety.
*   **Patient Safety:** Includes input sanitization, medical advice disclaimers, and emergency keyword detection to guide users appropriately.
*   **Audit Trails:** Key system events and interactions are logged for review in `logs/nhs_navigator_audit.log`.

---

## 📜 **License**

This project is licensed under the MIT License. See the `LICENSE` file for details (assuming a `LICENSE` file would be added, typically containing the MIT License text).
```
