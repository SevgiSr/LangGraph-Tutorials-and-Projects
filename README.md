# LangGraph Agents Collection

This repository contains a collection of various AI and non-AI agents written as jupyter notebooks for learning innerworkings of LangGraph. The project is structured to use a single, shared Python virtual environment to manage dependencies for all agents.

## Project Structure

The repository is organized as follows:

```
.
├── AI-agents/
│   ├── cli-agents/
│       └── ...
├── non-AI/
│   └── ...
├── venv/
├── .env
├── .gitignore
└── requirements.txt
```

- **`AI-agents/`**: Jupyter notebooks explaining and implementing LangGraph agents that use LLM's.
- **`cli-agents/`**: Unlike the rest of the files these are not jupyter notebooks but regular .py files, therefore they need to be run from the terminal. These are AI agents that I built as example projects.
- **`non-AI/`**: Contains simple LangGraph graphs without integration of LLM's, for the purpose of grasping LangGraph concepts clearly.
- **`venv/`**: The shared Python virtual environment for all projects in this repository. This folder is not tracked by Git.
- **`.env`**: A file to store secret keys and environment variables (e.g., `OPENAI_API_KEY`). This file is not tracked by Git.
- **`requirements.txt`**: A list of all Python packages required to run the projects in this repository.

---

## Prerequisites

Before you begin, ensure you have Python 3.8 or newer installed on your system. You can check your Python version by running:

```bash
python3 --version
```

---

## Setup and Installation

Follow these steps to set up the project environment on your local machine.

### 1. Clone the Repository

If you haven't already, clone the project to your local machine:

```bash
git clone <your-repository-url>
cd <repository-name>
```

### 2. Create and Activate the Virtual Environment

This project uses a single virtual environment named `venv` located in the root directory.

**To create the environment (only needs to be done once):**

```bash
python3 -m venv venv
```

**To activate the environment:**

- **On macOS / Linux:**
  ```bash
  source venv/bin/activate
  ```
- **On Windows (Command Prompt):**
  ```bash
  .\venv\Scripts\activate
  ```
- **On Windows (PowerShell):**
  ```bash
  .\venv\Scripts\Activate.ps1
  ```

Your terminal prompt should now be prefixed with `(venv)`, indicating the environment is active. You must activate the environment every time you work on this project in a new terminal session.

### 3. Install Dependencies

With the virtual environment active, install all the required Python packages using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables

The AI agents require API keys to function. These are stored in a `.env` file.

1.  Create a file named `.env` in the root of the project directory.
2.  Open the `.env` file and add the necessary key-value pairs. For example:

    ```ini
    # .env file
    OPENAI_API_KEY="your_openai_api_key_here"
    # Add other keys as needed
    # TAVILY_API_KEY="your_tavily_api_key_here"
    ```

The Python scripts are configured to automatically load these variables from this file.

---

## How to Run Agents

### Running agents in `cli-agents` folder

Each script or agent within the `cli-agents` are designed to be run independently.

1.  **Ensure your virtual environment is active.** If you see `(venv)` in your terminal prompt, you are ready. If not, follow step 2 in the installation guide.

2.  **Navigate to the script's directory** or run it from the root. For example, to run a chatbot script located at `AI-agents/cli-agents/Chatbot.py`, you would run:

    ```bash
    python AI-agents/cli-agents/Chatbot.py
    ```

### Running Jupyter Notebooks (`.ipynb` files):

- Make sure you have Jupyter installed (`pip install jupyter`).
- Start the Jupyter server from the project root:

```bash
jupyter notebook
```

- This will open a new tab in your browser. Navigate to the notebook file you wish to run and open it. Ensure the notebook is using the project's `venv` kernel.
