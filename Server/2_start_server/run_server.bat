REM LAUNCHING THE OPENSCORING SERVER ON http://localhost:8080/openscoring
start "openscoring" java -jar ../lib/openscoring-server-executable-2.1.1.jar

REM CREATING ML MODELS
py ../1_py_code/main.py

REM STARTING CLIENT INSTANCE IN GOOGLE CHROME AND DISABLING CHROME SECURITY
cd ..\..
cd Client
"C:\Program Files\Google\Chrome\Application\chrome.exe" --disable-web-security --disable-gpu --user-data-dir=%LOCALAPPDATA%\Google\chromeTemp %CD%\homePage.html