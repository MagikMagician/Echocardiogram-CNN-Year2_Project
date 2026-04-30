@echo off
echo Setting up Echocardiogram CNN environment...
echo.

REM Check if venv exists
if exist venv (
    echo Virtual environment already exists.
    echo To recreate, delete the venv folder first.
    echo.
) else (
    echo Creating virtual environment...
    python -m venv venv
    echo.
)

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Installing dependencies...
python -m pip install --upgrade pip
pip install -r "%~dp0requirements.txt"

echo.
echo Setup complete!
echo.
echo To activate the environment in the future, run:
echo   venv\Scripts\activate
echo.
pause
