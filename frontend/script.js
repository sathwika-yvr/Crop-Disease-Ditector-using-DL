async function uploadImage() {
    const file = document.getElementById("imageUpload").files[0];
    let formData = new FormData();
    formData.append("file", file);

    let response = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        body: formData
    });

    let result = await response.json();
    document.getElementById("result").innerHTML =
        `<b>Disease:</b> ${result.disease} <br>
         <b>Confidence:</b> ${result.confidence}% <br>
         <b>Remedy:</b> ${result.remedy}`;
}
