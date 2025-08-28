@echo off
echo Starting Options Pricing Model API...
echo.
echo Activating Stage 13 environment...
call stage13_env\Scripts\activate.bat
echo.
echo API will be available at: http://127.0.0.1:5000
echo.
echo Endpoints:
echo   GET  /predict/{vol}/{moneyness}/{vol_time}
echo   POST /predict (with JSON body)
echo   GET  /plot
echo   GET  /health
echo.
echo Press Ctrl+C to stop the server
echo.
python app.py
pause
