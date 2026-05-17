const metricsMeta = document.getElementById("metrics-meta");
const metricsTable = document.querySelector("#metrics-table tbody");
const modelChip = document.getElementById("model-chip");
const fileInput = document.getElementById("file-input");
const transcribeBtn = document.getElementById("transcribe-btn");
const recordBtn = document.getElementById("record-btn");
const statusEl = document.getElementById("status");
const outputEl = document.getElementById("output");
const scrollBtn = document.getElementById("scroll-cta");
const audioPreview = document.getElementById("audio-preview");

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

    statusEl.textContent = "Transcribing (this may take a moment if using large model)...";
    transcribeBtn.disabled = true;
    outputEl.value = "";

    const formData = new FormData();
    if (wavBlob) {
        formData.append("file", wavBlob, "recording.wav");
    } else {
        formData.append("file", fileInput.files[0]);
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
        statusEl.textContent = "Done.";
    } catch (err) {
        statusEl.textContent = "Transcription failed.";
    } finally {
        transcribeBtn.disabled = false;
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
loadMetrics();
