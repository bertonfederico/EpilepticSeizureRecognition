#!/bin/bash

#LAUNCHING THE OPENSCORING SERVER ON http://localhost:8080/openscoring
java.exe -jar ../lib/openscoring-server-executable-2.1.1.jar &

# CREATING ML MODELS
py ../1_py_code/main.py

#STARTING CLIENT INSTANCE IN GOOGLE CHROME AND DISABLING CHROME SECURITY
google-chrome --disable-web-security --allow-file-access-from-files --user-data-dir=~/chromeTemp ../../Client/homePage.html &