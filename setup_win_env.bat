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
bindeps\WPy64-31241\python-3.12.4.amd64\python.exe -m pip install --upgrade pip
if not exist venv (
    bindeps\WPy64-31241\python-3.12.4.amd64\python.exe -m venv venv
)
call venv\Scripts\activate.bat
python.exe -m pip install -r requirements.txt
powershell

