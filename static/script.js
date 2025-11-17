document.addEventListener('DOMContentLoaded', () => {
    const uploadForm = document.getElementById('upload-form');
    const fileInput = document.getElementById('file-input');
    const startBtn = document.getElementById('start-btn');
    const stopBtn = document.getElementById('stop-btn');
    const downloadBtn = document.getElementById('download-btn');
    const statusSpan = document.getElementById('status');
    const filenameSpan = document.getElementById('filename');
    const logsPre = document.getElementById('logs');

    let statusInterval;

    // Update UI based on state
    function updateUI(state) {
        statusSpan.textContent = state.status;
        filenameSpan.textContent = state.filename || 'none';
        logsPre.textContent = state.log ? state.log.join('\n') : '';
        logsPre.scrollTop = logsPre.scrollHeight;

        if (state.status === 'running') {
            startBtn.disabled = true;
            stopBtn.disabled = false;
            uploadForm.querySelector('button').disabled = true;
            fileInput.disabled = true;
        } else { // stopped, error, or initial
            stopBtn.disabled = true;
            uploadForm.querySelector('button').disabled = false;
            fileInput.disabled = false;
            if (state.filename) {
                startBtn.disabled = false;
            } else {
                startBtn.disabled = true;
            }
        }

        if (state.enriched_filename) {
            downloadBtn.disabled = false;
        } else {
            downloadBtn.disabled = true;
        }
    }

    // Fetch status from the backend
    async function fetchStatus() {
        try {
            const response = await fetch('/status');
            const state = await response.json();
            updateUI(state);
        } catch (error) {
            console.error('Error fetching status:', error);
            statusSpan.textContent = 'error';
        }
    }

    // File upload
    uploadForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const formData = new FormData();
        formData.append('file', fileInput.files[0]);

        try {
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData,
            });
            const result = await response.json();
            if (response.ok) {
                fetchStatus();
            } else {
                alert(`Error: ${result.error}`);
            }
        } catch (error) {
            console.error('Error uploading file:', error);
            alert('An error occurred during upload.');
        }
    });

    // Start process
    startBtn.addEventListener('click', async () => {
        try {
            await fetch('/start', { method: 'POST' });
            fetchStatus();
        } catch (error) {
            console.error('Error starting process:', error);
        }
    });

    // Stop process
    stopBtn.addEventListener('click', async () => {
        try {
            await fetch('/stop', { method: 'POST' });
            fetchStatus();
        } catch (error) {
            console.error('Error stopping process:', error);
        }
    });

    // Download file
    downloadBtn.addEventListener('click', () => {
        window.location.href = '/download';
    });

    // Initial status fetch and polling
    fetchStatus();
    statusInterval = setInterval(fetchStatus, 3000); // Poll every 3 seconds
});