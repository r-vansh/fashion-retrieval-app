$ErrorActionPreference = "Stop"

$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $python)) {
    throw "Project virtual environment not found at $python"
}

& $python -c "import faiss" 2>$null

if ($LASTEXITCODE -ne 0) {
    throw "FAISS is missing from the project environment. Run: .\.venv\Scripts\python.exe -m pip install -r requirements.txt"
}

& $python -m streamlit run (Join-Path $PSScriptRoot "app.py")
