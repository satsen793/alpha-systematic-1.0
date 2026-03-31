@echo off
setlocal

set REQ=requirements.txt
if exist %REQ% (
  echo Installing from %REQ%...
  python -m pip install --user -r %REQ%
) else (
  echo requirements.txt not found, installing core packages...
  python -m pip install --user numpy pandas pyarrow pandas-market-calendars
)

echo Done.
pause
