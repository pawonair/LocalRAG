## Local RAG System with Ollama + Deepseek R1
### Install & Run
- Initialize a `venv`
    ```bash
        python -m venv .venv
        source .venv/bin/activate
    ```
- Install requirements
    ```bash
        pip install -r requirements.txt
    ```
- Run app
    ```bash
        streamlit run ./src/home.py
    ```

### Generate executable with `PyInstaller`
- Create `pyinstaller` spec file
    ```bash
        pyi-makespec --onefile --additional-hooks-dir=./hooks app.py
    ```
- Run
    ```bash
        pyinstaller app.spec --clean
    ```