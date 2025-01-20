@echo off
setlocal

:: Determine the script location and project directory
set "PROJECT_DIR=%~dp0"

:: Define the virtual environment directory name
set "VENV_DIR=%PROJECT_DIR%myenv"

:: Define the path to the requirements file
set "REQUIREMENTS_FILE=%PROJECT_DIR%requirements.txt"

:: Check if the virtual environment already exists
IF NOT EXIST "%VENV_DIR%" (
    echo Creating virtual environment...
    python -m venv "%VENV_DIR%"
    
    :: Activate the virtual environment
    call "%VENV_DIR%\Scripts\activate"

    :: Upgrade pip to the latest version
    pip install --upgrade pip

    :: Install required packages from requirements.txt
    pip install -r "%REQUIREMENTS_FILE%"

    :: Explicitly install any additional packages needed
    pip install PyGithub fitz openai pinecone-client numpy streamlit python-dotenv

    :: Deactivate after installation
    call deactivate
) ELSE (
    echo Virtual environment already exists.
)

:: Pause to keep the window open
echo Script execution finished. Press any key to exit.
pause