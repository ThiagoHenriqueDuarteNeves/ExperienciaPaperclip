@echo off
setlocal

set SHARE=lmstudioapi
set TARGET=http://localhost:8001
set LOGFILE=%~dp0zrok-tunnel.log

echo [%date% %time%] Verificando processo zrok existente...
taskkill /F /IM zrok.exe /T >nul 2>&1
if %errorlevel% equ 0 (
    echo [%date% %time%] Processo zrok encerrado.
    timeout /t 2 /nobreak >nul
) else (
    echo [%date% %time%] Nenhum processo zrok ativo.
)

echo [%date% %time%] Iniciando tunel: %SHARE% -^> %TARGET%
start "" /B zrok share reserved %SHARE% --override-endpoint %TARGET% --headless > "%LOGFILE%" 2>&1

timeout /t 4 /nobreak >nul

echo.
echo URL publica: https://%SHARE%.share.zrok.io
echo Log:         %LOGFILE%
echo.

curl -s https://%SHARE%.share.zrok.io/health -H "skip_zrok_interstitial: true" --max-time 8
if %errorlevel% equ 0 (
    echo.
    echo [OK] Tunel online e memory-api respondendo.
) else (
    echo.
    echo [AVISO] Tunel pode ainda estar iniciando. Verifique %LOGFILE%
)

endlocal
