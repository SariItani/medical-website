document.addEventListener('DOMContentLoaded', function() {
    var addSymptomButton = document.getElementById('add-symptom');
    addSymptomButton.addEventListener('click', addSymptom);

    var symptomsContainer = document.getElementById('symptoms-container');
    symptomsContainer.addEventListener('click', removeSymptom);

    var symptomCount = 1;

    // Check initial visibility of "-" button
    updateRemoveButtonVisibility();

    function addSymptom() {
        var symptomGroup = document.querySelector('.symptom-group');
        var clonedSymptomGroup = symptomGroup.cloneNode(true);
        symptomCount++;

        var symptomLabel = clonedSymptomGroup.querySelector('label');
        var symptomSelect = clonedSymptomGroup.querySelector('select');

        symptomLabel.setAttribute('for', 'symptom-' + symptomCount);
        symptomLabel.textContent = 'Symptom ' + symptomCount + ':';
        symptomSelect.setAttribute('id', 'symptom-' + symptomCount);
        symptomSelect.selectedIndex = 0;

        symptomsContainer.appendChild(clonedSymptomGroup);

        updateRemoveButtonVisibility();
    }

    function removeSymptom(event) {
        if (event.target.classList.contains('remove-symptom')) {
            var symptomGroup = event.target.parentElement;
            symptomsContainer.removeChild(symptomGroup);
            symptomCount--;

            updateRemoveButtonVisibility();
        }
    }

    function updateRemoveButtonVisibility() {
        var removeButtons = document.querySelectorAll('.remove-symptom');
        removeButtons.forEach(function(button) {
            button.style.display = (symptomCount > 1) ? 'inline-block' : 'none';
        });
    }
});