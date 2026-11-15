@echo off
rem Double-click: starts the game and opens its page in the browser.
cd /d "%~dp0"
py game.py
if errorlevel 1 pause
