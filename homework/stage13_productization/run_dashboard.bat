@echo off
echo Starting Options Pricing Model Dashboard...
echo.
echo Activating Stage 13 environment...
call stage13_env\Scripts\activate.bat
echo.
echo Dashboard will open in your browser automatically
echo.
echo Features:
echo   - Interactive price prediction
echo   - Sensitivity analysis charts  
echo   - Model performance metrics
echo   - Input validation
echo.
echo Press Ctrl+C to stop the dashboard
echo.
streamlit run app_streamlit.py
pause
