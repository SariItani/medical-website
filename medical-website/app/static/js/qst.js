document.addEventListener('DOMContentLoaded', function() {
    var addSymptomButton = document.getElementById('add-symptom');
    addSymptomButton.addEventListener('click', addSymptom);

    var symptomsContainer = document.getElementById('symptoms-container');
    symptomsContainer.addEventListener('click', removeSymptom);

    function addSymptom() {
        var symptomDropdown = document.querySelector('.symptom-group');
        var clonedSymptomDropdown = symptomDropdown.cloneNode(true);
        symptomsContainer.appendChild(clonedSymptomDropdown);
    }

    function removeSymptom(event) {
        if (event.target.classList.contains('remove-symptom')) {
            var symptomDropdown = event.target.parentElement;
            symptomsContainer.removeChild(symptomDropdown);
        }
    }
});
