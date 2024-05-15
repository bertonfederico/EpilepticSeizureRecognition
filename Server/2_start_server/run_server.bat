REM CREATING ML MODELS IN .PMML FORMAT
py ../1_py_code/main.py


REM LAUNCHING THE OPENSCORING SERVER ON http://localhost:8080/openscoring
start "openscoring" java -jar ../lib/openscoring-server-executable-2.1.1.jar


REM WAITING FOR COMPLETION OF OPENSCORING LAUNCH
@echo off
set timeout=10
set /a "loops=%timeout%+1"
for /l %%i in (1,1,%loops%) do (
    ping 127.0.0.1 -n 2 > nul
)


REM LAUNCHING .PMML ON OPENSCORING
for %%i in (
    "NeuralNetwork" "SupportVectorClassification"
) do (
    java -cp ../lib/openscoring-client-executable-2.1.1.jar org.openscoring.client.Deployer --model http://localhost:8080/openscoring/model/%%i --file ../pmml/%%i.pmml
)


REM STARTING CLIENT INSTANCE IN GOOGLE CHROME AND DISABLING CHROME SECURITY
cd ..\..\Client
"C:\Program Files\Google\Chrome\Application\chrome.exe" --disable-web-security --disable-gpu --user-data-dir=%LOCALAPPDATA%\Google\chromeTemp %CD%\homePage.html


REM STOPPING OPENSCORING SERVER
taskkill /fi "windowtitle eq openscoring"