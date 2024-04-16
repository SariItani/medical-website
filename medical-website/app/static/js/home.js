document.addEventListener('DOMContentLoaded', function () {
    const toggleButton1 = document.getElementById('toggleButton1');
    const toggleButton2 = document.getElementById('toggleButton2');
    const file1 = document.getElementById('file1');
    const file2 = document.getElementById('file2');

    toggleButton1.addEventListener('click', function () {
        file1.classList.toggle('hidden-file');
    });

    toggleButton2.addEventListener('click', function () {
        file2.classList.toggle('hidden-file');
    });
});
