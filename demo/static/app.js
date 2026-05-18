const metricsMeta = document.getElementById("metrics-meta");
const metricsTable = document.querySelector("#metrics-table tbody");
const fileInput = document.getElementById("file-input");
const transcribeBtn = document.getElementById("transcribe-btn");
const recordBtn = document.getElementById("record-btn");
const statusEl = document.getElementById("status");
const outputEl = document.getElementById("output");
const summaryOutputEl = document.getElementById("summary-output");
const scrollBtn = document.getElementById("scroll-cta");
const audioPreview = document.getElementById("audio-preview");
const searchInput = document.getElementById("search-input");
const searchBtn = document.getElementById("search-btn");
const searchResults = document.getElementById("search-results");
const mambaToggle = document.getElementById("mamba-toggle");
const mambaOutputEl = document.getElementById("mamba-output");

const fmt = (value) => value === null || value === undefined ? "N/A" : value.toFixed(4);

let mediaRecorder = null;
let audioChunks = [];
let isRecording = false;
let recordedBlob = null;
let wavBlob = null;

async function blobToWav(blob) {
    const arrayBuffer = await blob.arrayBuffer();
    const audioContext = new (window.AudioContext || window.webkitAudioContext)();
    const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
    
    const numOfChan = audioBuffer.numberOfChannels;
    const length = audioBuffer.length * numOfChan * 2;
    const buffer = new ArrayBuffer(44 + length);
    const view = new DataView(buffer);
    const channels = [], sampleRate = audioBuffer.sampleRate;
    let offset = 0, pos = 0;
    
    const setUint32 = (data) => { view.setUint32(pos, data, true); pos += 4; }
    const setUint16 = (data) => { view.setUint16(pos, data, true); pos += 2; }
    const writeString = (s) => { for (let i = 0; i < s.length; i++) { view.setUint8(pos, s.charCodeAt(i)); pos++; } }
    
    writeString('RIFF'); setUint32(36 + length); writeString('WAVE');
    writeString('fmt '); setUint32(16); setUint16(1); setUint16(numOfChan);
    setUint32(sampleRate); setUint32(sampleRate * 2 * numOfChan); setUint16(numOfChan * 2); setUint16(16);
    writeString('data'); setUint32(length);
    
    for (let i = 0; i < audioBuffer.numberOfChannels; i++) channels.push(audioBuffer.getChannelData(i));
    
    while(pos < length + 44) {
        for (let i = 0; i < numOfChan; i++) {
            let sample = Math.max(-1, Math.min(1, channels[i][offset]));
            sample = sample < 0 ? sample * 0x8000 : sample * 0x7FFF;
            view.setInt16(pos, sample, true);
            pos += 2;
        }
        offset++;
    }
    return new Blob([buffer], { type: "audio/wav" });
}

async function loadMetrics() {
    try {
        const res = await fetch("/api/metrics");
        if (!res.ok) {
            metricsMeta.textContent = "Metrics not available. Run evaluation.";
            return;
        }

        const data = await res.json();
        const dataset = data.dataset || {};
        const whisper = data.whisper || {};

        const samplesTotal = dataset.samples_total;
        const samples = dataset.samples;
        const limit = dataset.limit ?? "all";
        const generated = dataset.generated_at || data.generated_at || "unknown";

        const samplesLine = samplesTotal && samples !== undefined
            ? `Samples: ${samples}/${samplesTotal}`
            : `Samples: ${samples ?? "N/A"}`;

        metricsMeta.textContent = `${samplesLine} | Limit: ${limit} | Generated: ${generated}`;

        const calcAcc = (wer) => {
            if (wer === null || wer === undefined) return "N/A";
            const acc = Math.max(0, 100 - (wer * 100));
            return acc.toFixed(2) + "%";
        };

        metricsTable.innerHTML = `
      <tr>
        <td>Whisper (${whisper.model || "large"})</td>
        <td>${fmt(whisper.wer)}</td>
        <td>${fmt(whisper.cer)}</td>
        <td>${calcAcc(whisper.wer)}</td>
      </tr>
    `;

    } catch (err) {
        metricsMeta.textContent = "Metrics unavailable.";
    }
}

async function transcribe() {
    if (!fileInput.files.length && !recordedBlob) {
        statusEl.textContent = "Please select an audio file or record audio.";
        return;
    }

    statusEl.textContent = "Transcribing & Summarizing (this may take a moment)...";
    transcribeBtn.disabled = true;
    outputEl.value = "";
    summaryOutputEl.value = "";
    if (mambaOutputEl) mambaOutputEl.value = "";

    const formData = new FormData();
    if (wavBlob) {
        formData.append("file", wavBlob, "recording.wav");
    } else if (recordedBlob) {
        formData.append("file", recordedBlob, "recording.webm");
    } else if (fileInput.files.length > 0) {
        formData.append("file", fileInput.files[0]);
    } else {
        statusEl.textContent = "Please select an audio file or record audio.";
        transcribeBtn.disabled = false;
        return;
    }

    if (mambaToggle && mambaToggle.checked) {
        formData.append("mamba_analysis", "true");
    }

    try {
        const res = await fetch("/api/transcribe", {
            method: "POST",
            body: formData,
        });
        const data = await res.json();

        if (!res.ok) {
            statusEl.textContent = data.error || "Transcription failed.";
            return;
        }

        outputEl.value = data.text || "";
        summaryOutputEl.value = data.summary || "No summary generated.";
        if (mambaOutputEl) mambaOutputEl.value = data.mamba_story || "";
        statusEl.textContent = "Done.";
        loadHistory();
    } catch (err) {
        statusEl.textContent = "Transcription failed.";
    } finally {
        transcribeBtn.disabled = false;
    }
}

async function loadHistory(query = "") {
    try {
        const res = await fetch(`/api/search?q=${encodeURIComponent(query)}`);
        const data = await res.json();
        
        searchResults.innerHTML = "";
        
        if (!data.results || data.results.length === 0) {
            searchResults.innerHTML = `<div class="status">No results found.</div>`;
            return;
        }
        
        data.results.forEach(record => {
            const date = new Date(record.timestamp).toLocaleString();
            searchResults.innerHTML += `
                <div style="border: 1px solid var(--border); border-radius: 8px; padding: 12px; background: rgba(0,0,0,0.2);">
                    <div style="font-size: 0.8rem; color: var(--text-secondary); margin-bottom: 8px;">${date}</div>
                    <div style="margin-bottom: 8px; font-family: 'Cairo', sans-serif;"><strong>Transcript:</strong> ${record.text}</div>
                    <div style="font-family: 'Cairo', sans-serif; color: var(--text-secondary);"><strong>Summary:</strong> ${record.summary || 'N/A'}</div>
                </div>
            `;
        });
    } catch (err) {
        searchResults.innerHTML = `<div class="status" style="color: var(--danger);">Failed to load history.</div>`;
    }
}

async function toggleRecord() {
    if (!isRecording) {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            mediaRecorder = new MediaRecorder(stream);
            
            mediaRecorder.ondataavailable = e => {
                if (e.data.size > 0) audioChunks.push(e.data);
            };
            
            mediaRecorder.onstop = async () => {
                transcribeBtn.disabled = true;
                statusEl.textContent = "Processing audio...";
                recordedBlob = new Blob(audioChunks, { type: mediaRecorder.mimeType || "audio/webm" });
                try {
                    wavBlob = await blobToWav(recordedBlob);
                    const url = URL.createObjectURL(wavBlob);
                    audioPreview.src = url;
                    audioPreview.style.display = "block";
                    fileInput.value = ""; // Clear file input
                    statusEl.textContent = "Recording ready. Click Transcribe.";
                } catch (err) {
                    console.error("Audio processing failed:", err);
                    statusEl.textContent = "Audio processing failed.";
                } finally {
                    transcribeBtn.disabled = false;
                }
                
                // Stop all tracks to release mic
                stream.getTracks().forEach(track => track.stop());
            };
            
            audioChunks = [];
            mediaRecorder.start();
            isRecording = true;
            
            recordBtn.textContent = "Stop";
            recordBtn.classList.add("recording");
            statusEl.textContent = "Recording...";
            recordedBlob = null;
            wavBlob = null;
        } catch (err) {
            statusEl.textContent = "Microphone access denied.";
        }
    } else {
        mediaRecorder.stop();
        isRecording = false;
        recordBtn.textContent = "Record";
        recordBtn.classList.remove("recording");
        statusEl.textContent = "Recording stopped. Ready to transcribe.";
    }
}

if (scrollBtn) {
    scrollBtn.addEventListener("click", () => {
        const section = document.getElementById("demo");
        if (section) section.scrollIntoView({ behavior: "smooth" });
    });
}

if (fileInput) {
    fileInput.addEventListener("change", (e) => {
        recordedBlob = null; // Clear any recording
        wavBlob = null;
        if (fileInput.files.length > 0) {
            const file = fileInput.files[0];
            const url = URL.createObjectURL(file);
            audioPreview.src = url;
            audioPreview.style.display = "block";
        } else {
            audioPreview.style.display = "none";
            audioPreview.src = "";
        }
    });
}

transcribeBtn.addEventListener("click", transcribe);
if (recordBtn) recordBtn.addEventListener("click", toggleRecord);
if (searchBtn) searchBtn.addEventListener("click", () => loadHistory(searchInput.value));
if (searchInput) searchInput.addEventListener("keyup", (e) => {
    if (e.key === "Enter") loadHistory(searchInput.value);
});

loadMetrics();
loadHistory();
