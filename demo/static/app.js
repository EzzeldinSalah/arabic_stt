const metricsMeta = document.getElementById("metrics-meta");
const metricsTable = document.querySelector("#metrics-table tbody");
const metricsCompare = document.getElementById("metrics-compare");
const modelChip = document.getElementById("model-chip");
const fileInput = document.getElementById("file-input");
const transcribeBtn = document.getElementById("transcribe-btn");
const statusEl = document.getElementById("status");
const outputEl = document.getElementById("output");
const scrollBtn = document.getElementById("scroll-cta");
const audioPreview = document.getElementById("audio-preview");

const fmt = (value) =>
    value === null || value === undefined ? "N/A" : value.toFixed(4);
const fmtDelta = (value) =>
    value === null || value === undefined
        ? "N/A"
        : value.toFixed(4).replace(/^/, value >= 0 ? "+" : "");

async function loadMetrics() {
    try {
        const res = await fetch("/api/metrics");
        if (!res.ok) {
            metricsMeta.textContent =
                "Metrics not available. Run whisper_baseline/whisper_eval.py";
            return;
        }

        const data = await res.json();
        const dataset = data.dataset || {};
        const whisper = data.whisper || {};
        const cnn = data.cnn_lstm || {};
        const comparison = data.comparison || {};

        const samplesTotal = dataset.samples_total;
        const samples = dataset.samples;
        const limit = dataset.limit ?? "all";
        const generated =
            dataset.generated_at || data.generated_at || "unknown";

        const samplesLine =
            samplesTotal && samples !== undefined
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
        <td>Whisper (${whisper.model || ""})</td>
        <td>${fmt(whisper.wer)}</td>
        <td>${fmt(whisper.cer)}</td>
        <td>${calcAcc(whisper.wer)}</td>
      </tr>
      <tr>
        <td>CNN-LSTM</td>
        <td>${fmt(cnn.wer)}</td>
        <td>${fmt(cnn.cer)}</td>
        <td>${calcAcc(cnn.wer)}</td>
      </tr>
    `;

        metricsCompare.textContent = `Delta (CNN-LSTM - Whisper): WER ${fmtDelta(
            comparison.wer_delta,
        )}, CER ${fmtDelta(comparison.cer_delta)}`;

        if (whisper.model) {
            modelChip.textContent = `Whisper ${whisper.model}`;
        }
    } catch (err) {
        metricsMeta.textContent = "Metrics unavailable.";
    }
}

async function transcribe() {
    if (!fileInput.files.length) {
        statusEl.textContent = "Please select an audio file.";
        return;
    }

    statusEl.textContent = "Transcribing...";
    transcribeBtn.disabled = true;
    outputEl.value = "";

    const formData = new FormData();
    formData.append("file", fileInput.files[0]);

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

if (scrollBtn) {
    scrollBtn.addEventListener("click", () => {
        const section = document.getElementById("demo");
        if (section) {
            section.scrollIntoView({ behavior: "smooth" });
        }
    });
}

if (fileInput) {
    fileInput.addEventListener("change", (e) => {
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
loadMetrics();
