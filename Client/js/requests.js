function getPrediction() {
    const csvFile = document.getElementById("csvFile");
    const input = csvFile.files[0];
    const reader = new FileReader();

    reader.onload = e => {
        const text = e.target.result;
        const data = text.split(";");
        const split_data = {};
        let i = 0;
        while (i < data.length) {
            split_data["x" + (i+1).toString()] = data[i]
            i++;
        }
        const postData = {
            id: "input",
            arguments: split_data
        }
        sendRequest(postData)
    };

    reader.readAsText(input);
}

function sendRequest(jsonReq) {
    const xhr = new XMLHttpRequest();
    const url = "http://localhost:8080/openscoring/model/" + actualML;
    xhr.open("POST", url, true);
    xhr.setRequestHeader("Content-type", "application/json");
    xhr.setRequestHeader('Accept', 'application/json');
    xhr.setRequestHeader('Access-Control-Allow-Origin', '*');
    xhr.onreadystatechange = function () {
        if (xhr.readyState === 4 && xhr.status === 200) {
            const json = JSON.parse(xhr.responseText);
            const prediction = json.results.y === 0 ? "Non-epileptic" : "Epileptic";
            document.getElementById("prediction").innerHTML = "Model prediction: " + prediction;
            let probability = json.results["probability(" + json.results.y + ")"];
            probability = probability !== undefined ? ("Probability: " + ((probability*100).toFixed(2)) + "%") : "";
            document.getElementById("probability").innerHTML =  probability;
        }
    }
    const data = JSON.stringify(jsonReq);
    xhr.send(data);
}