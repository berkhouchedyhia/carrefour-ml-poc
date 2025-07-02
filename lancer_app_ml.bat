@echo off
echo ===============================
echo   Lancement de l'app ML Dyhia
echo ===============================
echo.

REM === Étape 1 : Installer les dépendances ===
echo [1/2] Vérification / installation des modules nécessaires...
C:\Python311\python.exe -m pip install --upgrade pip >nul 2>&1
C:\Python311\python.exe -m pip install -r "C:\Users\MOHAND KACI\Downloads\ML Dyhia\requirements.txt" >nul 2>&1

REM === Étape 2 : Lancer l'application Streamlit ===
echo [2/2] Lancement de l'application Streamlit...
start "" http://localhost:8501
C:\Python311\python.exe -m streamlit run "C:\Users\MOHAND KACI\Downloads\ML Dyhia\app.py"

echo.
pause
