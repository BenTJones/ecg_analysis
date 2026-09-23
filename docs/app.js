const API_URL =
    "https://https://ecg-analysis-ws6t.onrender.com/predict";


const heaFileInput =
    document.getElementById("heaFile");

const datFileInput =
    document.getElementById("datFile");

const analyseButton =
    document.getElementById("analyseButton");

const statusMessage =
    document.getElementById("statusMessage");

const resultCard =
    document.getElementById("resultCard");

const predictionText =
    document.getElementById("prediction");

const probabilityText =
    document.getElementById("probability");


analyseButton.addEventListener(
    "click",
    async function () {

        const heaFile =
            heaFileInput.files[0];

        const datFile =
            datFileInput.files[0];


        if (!heaFile || !datFile) {

            statusMessage.textContent =
                "Please select both a .hea file and a .dat file.";

            return;
        }


        statusMessage.textContent =
            "Analysing ECG...";

        analyseButton.disabled = true;

        resultCard.hidden = true;


        const formData =
            new FormData();

        formData.append(
            "files",
            heaFile
        );

        formData.append(
            "files",
            datFile
        );


        try {

            const response =
                await fetch(
                    API_URL,
                    {
                        method: "POST",
                        body: formData
                    }
                );


            const result =
                await response.json();


            if (!response.ok) {

                throw new Error(
                    result.detail ||
                    "The API could not analyse this ECG."
                );

            }


            predictionText.textContent =
                result.prediction;


            const percentage =
                (
                    result.sttc_probability * 100
                ).toFixed(2);


            probabilityText.textContent =
                percentage + "%";


            resultCard.hidden = false;


            statusMessage.textContent =
                "Analysis complete.";

        }

        catch (error) {

            statusMessage.textContent =
                "Error: " + error.message;

        }

        finally {

            analyseButton.disabled = false;

        }

    }
);