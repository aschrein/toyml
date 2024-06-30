echo "Setting up windows environment"
if not exist bindeps mkdir bindeps
cd bindeps
if not exist Winpython64-3.12.4.1.exe (
    echo "Downloading Python 3.12.4.1"
    curl -L https://github.com/winpython/winpython/releases/download/8.2.20240618final/Winpython64-3.12.4.1.exe -o Winpython64-3.12.4.1.exe
    @rem del Winpython64-3.12.4.1.exe
)
if not exist WPy64-31241 (
    echo "Extracting Python"
    Winpython64-3.12.4.1.exe -o WPy64-31241 -y
)
cd ..
echo "Setting up Python environment and installing dependencies"
if not exist venv (
    bindeps\WPy64-31241\python-3.12.4.amd64\python.exe -m pip install --upgrade pip
    bindeps\WPy64-31241\python-3.12.4.amd64\python.exe -m venv venv
)
@REM Try if our project is installed
python -c "import py.dsl"
@REM Fetch bindeps
python.exe scripts/fetch_bindeps.py
@REM echo "ADDING %CD%\bindeps\renderdoc\RenderDoc_1.33_64"
SET PATH=%PATH%;%CD%\bindeps\renderdoc\RenderDoc_1.33_64;
if %errorlevel% neq 0 (
    echo "Installing project"
    call venv\Scripts\activate.bat
    python.exe -m pip install -r requirements.txt
    pip install -e .
)

echo "Environment setup complete"
echo "------------------------"
echo "------ SUCCESS ---------"
echo "------------------------"
powershell

