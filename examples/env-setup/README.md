# CMPS 460 Python Environment Setup

The recommended way to run notebooks for this course is using **Google Colab in VS Code**.

## Option 1: Google Colab in VS Code (Recommended)

This option gives you the power of cloud computing ☁️ with the convenience of your local editor 💻.

1.  **Install Extensions:** In VS Code, install the following extensions:
    *   **Python** (by Microsoft)
    *   **Jupyter** (by Microsoft)
    *   **Google Colab** (by Google)
2.  **Open Notebook:** Open your `.ipynb` file.
3.  **Connect:** Click **Select Kernel** (top-right of the notebook editor) > **Colab** > **Select Google Account** (sign in if prompted).

---

## Option 2: Google Colab on the Web

You can also run notebooks directly in your browser without VS Code.

1.  Go to [colab.research.google.com](https://colab.research.google.com).
2.  **Upload** your notebook (`.ipynb` file).
3.  **Install Libraries:** Libraries are pre-installed in Colab. If you need to install libraries, add and run this code cell at the top of your notebook:
    ```python
    !pip install -r https://raw.githubusercontent.com/mls26/ml-content/refs/heads/main/examples/env-setup/requirements.txt
    ```

---

## Option 3: Local Python Installation

If you prefer running everything locally on your machine, follow these steps:

### 1. Install Python
Download and install the latest Python from [python.org](https://www.python.org/downloads/).
*   **Important:** Check the box **"Add Python to path"** during installation.

### 2. Setup Environment

## Prerequisites (All Platforms)

- Python **3.10+** installed
- `requirements.txt` file available
- Setup scripts included in the project:
  - `setup-env.ps1` for Windows
  - `setup-env.sh` for macOS / Linux

## Windows Setup

### Step 1: Open the terminal
- Navigate to the folder containing `setup-env.ps1`
- Open a terminal in that folder

### Step 2: Allow script execution (one time only)
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned

### Step 3: Run the setup script
.\setup-env.ps1

- Virtual environment is created at:
C:\Users<username>.venvs\ml-env

## macOS / Linux Setup

### Step 1: Open the terminal
- Navigate to the folder containing `setup-env.sh`

### Step 2: Make the script executable (one time only)
chmod +x setup-env.sh

### Step 3: Run the setup script
./setup-env.sh

- Virtual environment is created at:
~/.venvs/ml-env


### 3. Run Notebooks
*   **VS Code:** Install the **Jupyter** extension. Open a `.ipynb` file, click **Select Kernel** > **Python Environments** > `ml-env`.
*   **Browser:** Run `jupyter lab` in your activated terminal.

---

## Updating Libraries
To update your environment with the latest course requirements, run:
```bash
pip install --upgrade -r requirements.txt
```
