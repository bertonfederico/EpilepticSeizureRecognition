#!/bin/bash


# CREATING ML MODELS
py ../1_py_code/main.py


# LAUNCHING THE OPENSCORING SERVER ON http://localhost:8080/openscoring
java.exe -jar ../lib/openscoring-server-executable-2.1.1.jar &


# WAITING FOR COMPLETION OF OPENSCORING LAUNCH
sleep 10


# LAUNCHING .PMML ON OPENSCORING
pmmlnames=("DecisionTreeClassifier" "ExtraTreeClassifier" "GaussianNaiveBayes" "NeuralNetwork" "SupportVectorClassification")
for pmmlname in "${pmmlnames[@]}"; do
  java -cp ../lib/openscoring-client-executable-2.1.1.jar org.openscoring.client.Deployer --model http://localhost:8080/openscoring/model/"$pmmlname" --file ../pmml/"$pmmlname".pmml
done


# STARTING CLIENT INSTANCE IN GOOGLE CHROME AND DISABLING CHROME SECURITY
google-chrome --disable-web-security --allow-file-access-from-files --user-data-dir=~/chromeTemp ../../Client/homePage.html &


# STOPPING OPENSCORING SERVER
pkill -f ../lib/openscoring-server-executable-2.1.1.jar