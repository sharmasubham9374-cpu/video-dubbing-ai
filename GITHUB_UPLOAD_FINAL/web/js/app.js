document.addEventListener("DOMContentLoaded", () => {
    // State
    const DEFAULT_UPI_ID = "subham.088@fam";
    let selectedPlan = 20;
    let selectedPlanName = "28-Days Monthly VIP Pass";
    let currentConfig = {};
    let uploadedVideoData = null;
    let activeJobId = null;
    let pollInterval = null;

    // DOM Elements - Nav & Badges
    const btnOpenPricing = document.getElementById("btnOpenPricing");
    const passStatusText = document.getElementById("passStatusText");
    const apiStatusBadge = document.getElementById("apiStatusBadge");
    const apiStatusText = document.getElementById("apiStatusText");
    const btnOpenSettings = document.getElementById("btnOpenSettings");
    const btnCloseSettings = document.getElementById("btnCloseSettings");
    const btnCancelSettings = document.getElementById("btnCancelSettings");
    const btnSaveSettings = document.getElementById("btnSaveSettings");
    const settingsModal = document.getElementById("settingsModal");
    const settingsFeedback = document.getElementById("settingsFeedback");

    // Pricing Modal Elements
    const pricingModal = document.getElementById("pricingModal");
    const btnClosePricing = document.getElementById("btnClosePricing");
    const btnCancelPricing = document.getElementById("btnCancelPricing");
    const planCards = document.querySelectorAll(".plan-card");
    const upiQrImage = document.getElementById("upiQrImage");
    const selectedPlanAmountDisplay = document.getElementById("selectedPlanAmountDisplay");
    const intentPayAmount = document.getElementById("intentPayAmount");
    const btnUpiIntentLink = document.getElementById("btnUpiIntentLink");
    const textUpiId = document.getElementById("textUpiId");
    const btnCopyUpi = document.getElementById("btnCopyUpi");
    const inputUtrNumber = document.getElementById("inputUtrNumber");
    const btnVerifyUtr = document.getElementById("btnVerifyUtr");
    const paymentFeedback = document.getElementById("paymentFeedback");

    // Settings Inputs
    const inputElevenLabsKey = document.getElementById("inputElevenLabsKey");
    const inputVoiceId = document.getElementById("inputVoiceId");
    const inputGeminiKey = document.getElementById("inputGeminiKey");

    // Upload & Form Elements
    const dropzone = document.getElementById("dropzone");
    const videoFileInput = document.getElementById("videoFileInput");
    const dropzonePrompt = document.getElementById("dropzonePrompt");
    const selectedVideoPreview = document.getElementById("selectedVideoPreview");
    const inputVideoPreview = document.getElementById("inputVideoPreview");
    const previewFileName = document.getElementById("previewFileName");
    const previewSpecs = document.getElementById("previewSpecs");
    const btnRemoveVideo = document.getElementById("btnRemoveVideo");

    const voiceSelect = document.getElementById("voiceSelect");
    const voiceHelperText = document.getElementById("voiceHelperText");
    const btnTestVoice = document.getElementById("btnTestVoice");
    const voiceTestAudio = document.getElementById("voiceTestAudio");

    const bgVolume = document.getElementById("bgVolume");
    const bgVolumeVal = document.getElementById("bgVolumeVal");
    const btnStartDubbing = document.getElementById("btnStartDubbing");

    const jobStatusBadge = document.getElementById("jobStatusBadge");
    const progressBarFill = document.getElementById("progressBarFill");
    const statusMessage = document.getElementById("statusMessage");

    const stepNodes = {
        extract: document.getElementById("step-extract"),
        transcribe: document.getElementById("step-transcribe"),
        translate: document.getElementById("step-translate"),
        synthesize: document.getElementById("step-synthesize"),
        merge: document.getElementById("step-merge")
    };

    const tabBtns = document.querySelectorAll(".tab-btn");
    const tabContents = document.querySelectorAll(".tab-content");

    const playerEmptyState = document.getElementById("playerEmptyState");
    const playerWrapper = document.getElementById("playerWrapper");
    const outputVideoPlayer = document.getElementById("outputVideoPlayer");
    const btnDownloadVideo = document.getElementById("btnDownloadVideo");
    const btnDownloadSrt = document.getElementById("btnDownloadSrt");
    const transcriptSegmentsList = document.getElementById("transcriptSegmentsList");

    // Initialize App
    initApp();

    async function initApp() {
        checkPassStatus();
        await loadConfig();
        setupEventListeners();
        setupPricingEvents();
        loadVoices();
    }

    // Check if user already unlocked pass
    function checkPassStatus() {
        const isUnlocked = localStorage.getItem("videodubber_unlocked_pass") === "true";
        if (isUnlocked) {
            passStatusText.textContent = "👑 Pro Member (Active)";
            btnOpenPricing.classList.add("unlocked-pro");
        } else {
            passStatusText.textContent = "👑 Unlock Pass (₹10/₹15/₹20)";
            btnOpenPricing.classList.remove("unlocked-pro");
        }
    }

    async function loadConfig() {
        try {
            // Check LocalStorage first for BYOK (Bring Your Own Key)
            const localGemini = localStorage.getItem("videodubber_gemini_key");
            const localEleven = localStorage.getItem("videodubber_elevenlabs_key");
            const localVoice = localStorage.getItem("videodubber_voice_id");

            const res = await fetch("/api/config");
            currentConfig = await res.json();

            // LocalStorage takes priority for visitors
            if (localGemini) currentConfig.gemini_api_key = localGemini;
            if (localEleven) currentConfig.elevenlabs_api_key = localEleven;
            if (localVoice) currentConfig.selected_voice_id = localVoice;

            inputElevenLabsKey.value = currentConfig.elevenlabs_api_key || "";
            inputGeminiKey.value = currentConfig.gemini_api_key || "";
            inputVoiceId.value = currentConfig.selected_voice_id || "";

            if (currentConfig.bg_music_volume !== undefined) {
                bgVolume.value = currentConfig.bg_music_volume;
                bgVolumeVal.textContent = `${Math.round(currentConfig.bg_music_volume * 100)}%`;
            }

            if (currentConfig.audio_mode) {
                const radio = document.querySelector(`input[name="audioMode"][value="${currentConfig.audio_mode}"]`);
                if (radio) radio.checked = true;
            }

            updateApiStatusBadge();
        } catch (err) {
            console.error("Failed to load config:", err);
        }
    }

    function updateApiStatusBadge() {
        const hasGemini = !!(currentConfig.gemini_api_key || localStorage.getItem("videodubber_gemini_key"));
        const hasEleven = !!(currentConfig.elevenlabs_api_key || localStorage.getItem("videodubber_elevenlabs_key"));

        if (hasGemini) {
            apiStatusBadge.className = "status-pill status-success";
            apiStatusText.textContent = hasEleven ? "AI Ready (Gemini + ElevenLabs/Bunty)" : "AI Ready (Gemini + Bunty Voice)";
        } else {
            apiStatusBadge.className = "status-pill status-warning";
            apiStatusText.textContent = "Gemini Key Needed";
        }
    }

    async function loadVoices() {
        try {
            voiceSelect.innerHTML = '<option value="" disabled selected>Loading ElevenLabs & Indian voices...</option>';
            const key = currentConfig.elevenlabs_api_key || localStorage.getItem("videodubber_elevenlabs_key") || "";
            const res = await fetch(`/api/voices?api_key=${encodeURIComponent(key)}`);
            const data = await res.json();

            if (data.voices && data.voices.length > 0) {
                voiceSelect.innerHTML = "";
                let buntyVoice = null;

                data.voices.forEach(v => {
                    const opt = document.createElement("option");
                    opt.value = v.voice_id;
                    opt.textContent = `${v.name} ${v.is_bunty ? "⭐ (Bunty)" : `(${v.category || "Voice"})`}`;
                    if (v.is_bunty) {
                        opt.classList.add("bunty-opt");
                        buntyVoice = v;
                    }
                    voiceSelect.appendChild(opt);
                });

                // Auto-select Bunty if available or previously saved voice
                const savedVoice = currentConfig.selected_voice_id || localStorage.getItem("videodubber_voice_id");
                if (savedVoice) {
                    voiceSelect.value = savedVoice;
                } else if (buntyVoice) {
                    voiceSelect.value = buntyVoice.voice_id;
                    voiceHelperText.textContent = `Selected: ${buntyVoice.name} (Auto-detected Bunty)`;
                } else if (data.voices.length > 0) {
                    voiceSelect.value = data.voices[0].voice_id;
                }

                inputVoiceId.value = voiceSelect.value;
            } else {
                const errMsg = data.error || data.message || "No voices found.";
                voiceSelect.innerHTML = `<option value="" disabled selected>${errMsg}</option>`;
                voiceHelperText.textContent = errMsg;
            }
        } catch (err) {
            voiceSelect.innerHTML = '<option value="" disabled selected>Default: Bunty Voice Ready</option>';
            voiceHelperText.textContent = "Bunty Hindi Voice Enabled";
        }
    }

    function updatePricingCheckout() {
        selectedPlanAmountDisplay.textContent = selectedPlan;
        intentPayAmount.textContent = selectedPlan;
        
        const upiUri = `upi://pay?pa=${DEFAULT_UPI_ID}&pn=VideoDubber%20AI&am=${selectedPlan}&cu=INR&tn=VideoDubber%20${encodeURIComponent(selectedPlanName)}`;
        
        // Update QR Code
        upiQrImage.src = `https://api.qrserver.com/v1/create-qr-code/?size=220x220&data=${encodeURIComponent(upiUri)}`;
        
        // Update Intent Button
        btnUpiIntentLink.href = upiUri;
    }

    function setupPricingEvents() {
        // Modal toggles
        btnOpenPricing.addEventListener("click", () => {
            pricingModal.classList.remove("hidden");
            paymentFeedback.textContent = "";
            updatePricingCheckout();
        });

        btnClosePricing.addEventListener("click", () => pricingModal.classList.add("hidden"));
        btnCancelPricing.addEventListener("click", () => pricingModal.classList.add("hidden"));

        // Plan Selection
        planCards.forEach(card => {
            card.addEventListener("click", () => {
                planCards.forEach(c => {
                    c.classList.remove("selected");
                    const b = c.querySelector(".btn-plan-select");
                    if (b) b.textContent = `Select ₹${c.dataset.plan}`;
                });

                card.classList.add("selected");
                const btn = card.querySelector(".btn-plan-select");
                if (btn) btn.textContent = `Selected (₹${card.dataset.plan})`;

                selectedPlan = parseInt(card.dataset.plan, 10);
                selectedPlanName = card.dataset.name;
                updatePricingCheckout();
            });
        });

        // Copy UPI ID
        btnCopyUpi.addEventListener("click", () => {
            navigator.clipboard.writeText(DEFAULT_UPI_ID).then(() => {
                btnCopyUpi.textContent = "✓ Copied!";
                setTimeout(() => { btnCopyUpi.textContent = "📋 Copy"; }, 2000);
            }).catch(() => {
                prompt("Copy UPI ID:", DEFAULT_UPI_ID);
            });
        });

        // Verify UTR / Passcode
        btnVerifyUtr.addEventListener("click", async () => {
            const utr = inputUtrNumber.value.trim();
            if (!utr) {
                paymentFeedback.className = "settings-feedback error";
                paymentFeedback.textContent = "Kripya payment ke baad apna 12-digit UTR No. ya VIP code dalein.";
                return;
            }

            btnVerifyUtr.disabled = true;
            btnVerifyUtr.textContent = "Verifying...";
            paymentFeedback.className = "settings-feedback";
            paymentFeedback.textContent = "Verifying transaction reference...";

            try {
                const res = await fetch("/api/verify-pass", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        utr_or_code: utr,
                        plan_price: selectedPlan
                    })
                });

                const data = await res.json();
                if (res.ok && data.unlocked) {
                    localStorage.setItem("videodubber_unlocked_pass", "true");
                    localStorage.setItem("videodubber_pass_type", selectedPlanName);
                    checkPassStatus();

                    paymentFeedback.className = "settings-feedback success";
                    paymentFeedback.innerHTML = `🎉 <strong>Mubarak Ho!</strong> Studio Access Unlocked. (Plan: ${selectedPlanName})`;

                    setTimeout(() => {
                        pricingModal.classList.add("hidden");
                        btnVerifyUtr.disabled = false;
                        btnVerifyUtr.textContent = "Unlock Now 🔓";
                    }, 1500);
                } else {
                    throw new Error(data.detail || "Invalid transaction reference.");
                }
            } catch (err) {
                paymentFeedback.className = "settings-feedback error";
                paymentFeedback.textContent = `Verification Error: ${err.message}`;
                btnVerifyUtr.disabled = false;
                btnVerifyUtr.textContent = "Unlock Now 🔓";
            }
        });
    }

    function setupEventListeners() {
        // Modal toggles
        btnOpenSettings.addEventListener("click", () => {
            settingsModal.classList.remove("hidden");
            settingsFeedback.textContent = "";
        });
        btnCloseSettings.addEventListener("click", () => settingsModal.classList.add("hidden"));
        btnCancelSettings.addEventListener("click", () => settingsModal.classList.add("hidden"));

        // Save Settings (Save to LocalStorage & Config)
        btnSaveSettings.addEventListener("click", async () => {
            const elKey = inputElevenLabsKey.value.trim();
            const gemKey = inputGeminiKey.value.trim();
            const vId = inputVoiceId.value.trim();

            // Store in user's browser localStorage for BYOK
            if (gemKey) localStorage.setItem("videodubber_gemini_key", gemKey);
            if (elKey) localStorage.setItem("videodubber_elevenlabs_key", elKey);
            if (vId) localStorage.setItem("videodubber_voice_id", vId);

            currentConfig.gemini_api_key = gemKey || currentConfig.gemini_api_key;
            currentConfig.elevenlabs_api_key = elKey || currentConfig.elevenlabs_api_key;
            currentConfig.selected_voice_id = vId || currentConfig.selected_voice_id;

            settingsFeedback.className = "settings-feedback";
            settingsFeedback.textContent = "Saving keys securely in browser...";

            try {
                await fetch("/api/config", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        elevenlabs_api_key: elKey,
                        gemini_api_key: gemKey,
                        selected_voice_id: vId
                    })
                });

                settingsFeedback.className = "settings-feedback success";
                settingsFeedback.textContent = "✓ API Keys saved securely in browser!";
                updateApiStatusBadge();
                if (elKey) loadVoices();
                setTimeout(() => {
                    settingsModal.classList.add("hidden");
                }, 1000);
            } catch (err) {
                settingsFeedback.className = "settings-feedback success";
                settingsFeedback.textContent = "✓ API Keys saved locally!";
                updateApiStatusBadge();
                setTimeout(() => { settingsModal.classList.add("hidden"); }, 1000);
            }
        });

        // Voice Select Change
        voiceSelect.addEventListener("change", () => {
            inputVoiceId.value = voiceSelect.value;
            localStorage.setItem("videodubber_voice_id", voiceSelect.value);
            const selectedText = voiceSelect.options[voiceSelect.selectedIndex].text;
            voiceHelperText.textContent = `Selected: ${selectedText}`;
        });

        // Test Voice button
        btnTestVoice.addEventListener("click", async () => {
            const voiceId = voiceSelect.value || inputVoiceId.value.trim();
            if (!voiceId) {
                alert("Please select or enter a Voice ID first.");
                return;
            }
            btnTestVoice.textContent = "⏳ Generating sample...";
            btnTestVoice.disabled = true;

            const elKey = currentConfig.elevenlabs_api_key || localStorage.getItem("videodubber_elevenlabs_key") || "";

            try {
                const res = await fetch("/api/test-voice", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        voice_id: voiceId,
                        text: "नमस्ते दोस्तों! मैं बंटी हूँ, और आपकी इंग्लिश वीडियो को हिंदी में डब करूँगा।",
                        elevenlabs_api_key: elKey
                    })
                });
                const data = await res.json();
                if (data.status === "success" && data.audio_url) {
                    voiceTestAudio.src = data.audio_url;
                    voiceTestAudio.play();
                } else {
                    alert(`Voice test error: ${data.detail || "Unknown error"}`);
                }
            } catch (err) {
                alert(`Could not play sample: ${err.message}`);
            } finally {
                btnTestVoice.textContent = '🔊 Test "Bunty" Voice';
                btnTestVoice.disabled = false;
            }
        });

        // Drag & Drop Upload
        dropzone.addEventListener("click", (e) => {
            if (e.target !== btnRemoveVideo && !selectedVideoPreview.contains(e.target)) {
                videoFileInput.click();
            }
        });

        dropzone.addEventListener("dragover", (e) => {
            e.preventDefault();
            dropzone.classList.add("dragover");
        });

        dropzone.addEventListener("dragleave", () => {
            dropzone.classList.remove("dragover");
        });

        dropzone.addEventListener("drop", (e) => {
            e.preventDefault();
            dropzone.classList.remove("dragover");
            if (e.dataTransfer.files.length > 0) {
                handleVideoFile(e.dataTransfer.files[0]);
            }
        });

        videoFileInput.addEventListener("change", () => {
            if (videoFileInput.files.length > 0) {
                handleVideoFile(videoFileInput.files[0]);
            }
        });

        btnRemoveVideo.addEventListener("click", (e) => {
            e.stopPropagation();
            resetVideoUpload();
        });

        // Volume Slider
        bgVolume.addEventListener("input", () => {
            bgVolumeVal.textContent = `${Math.round(bgVolume.value * 100)}%`;
        });

        // Tabs
        tabBtns.forEach(btn => {
            btn.addEventListener("click", () => {
                tabBtns.forEach(b => b.classList.remove("active"));
                tabContents.forEach(c => c.classList.remove("active"));
                btn.classList.add("active");
                const target = document.getElementById(btn.dataset.tab);
                if (target) target.classList.add("active");
            });
        });

        // Start Dubbing
        btnStartDubbing.addEventListener("click", startDubbingProcess);
    }

    async function handleVideoFile(file) {
        if (!file) return;
        
        statusMessage.textContent = "Uploading video...";
        const formData = new FormData();
        formData.append("file", file);

        try {
            const res = await fetch("/api/upload", {
                method: "POST",
                body: formData
            });

            if (!res.ok) {
                const err = await res.json();
                throw new Error(err.detail || "Upload failed");
            }

            uploadedVideoData = await res.json();
            
            // Show preview
            dropzonePrompt.classList.add("hidden");
            selectedVideoPreview.classList.remove("hidden");
            inputVideoPreview.src = uploadedVideoData.video_url;
            previewFileName.textContent = uploadedVideoData.filename;
            
            const mins = Math.floor(uploadedVideoData.duration / 60);
            const secs = Math.floor(uploadedVideoData.duration % 60);
            previewSpecs.textContent = `${mins}:${secs < 10 ? '0' : ''}${secs} • ${uploadedVideoData.width}x${uploadedVideoData.height}`;

            btnStartDubbing.disabled = false;
            statusMessage.textContent = `Video loaded (${uploadedVideoData.filename}). Ready to dub to Hindi.`;
        } catch (err) {
            alert(`Upload failed: ${err.message}`);
            resetVideoUpload();
        }
    }

    function resetVideoUpload() {
        uploadedVideoData = null;
        videoFileInput.value = "";
        inputVideoPreview.src = "";
        selectedVideoPreview.classList.add("hidden");
        dropzonePrompt.classList.remove("hidden");
        btnStartDubbing.disabled = true;
        statusMessage.textContent = "Upload a video to begin translation and dubbing.";
    }

    async function startDubbingProcess() {
        if (!uploadedVideoData) return;

        // Check if pass unlocked (If not unlocked, ask for ₹10/15/20 pass)
        const isUnlocked = localStorage.getItem("videodubber_unlocked_pass") === "true";
        if (!isUnlocked) {
            pricingModal.classList.remove("hidden");
            paymentFeedback.className = "settings-feedback";
            paymentFeedback.textContent = "👑 Unlimited Video Dubbing ke liye apna ₹10, ₹15, ya ₹20 ka Pass choose karein.";
            return;
        }

        const gemKey = currentConfig.gemini_api_key || localStorage.getItem("videodubber_gemini_key");
        const elKey = currentConfig.elevenlabs_api_key || localStorage.getItem("videodubber_elevenlabs_key");

        if (!gemKey) {
            settingsModal.classList.remove("hidden");
            settingsFeedback.className = "settings-feedback error";
            settingsFeedback.textContent = "Please provide your Free Google Gemini API Key in Settings to proceed.";
            return;
        }

        const voiceId = voiceSelect.value || inputVoiceId.value.trim();
        const audioMode = document.querySelector('input[name="audioMode"]:checked').value;
        const bgVol = parseFloat(bgVolume.value);

        btnStartDubbing.disabled = true;
        btnStartDubbing.innerHTML = '<span class="btn-icon">⏳</span><span>Dubbing in Progress...</span>';

        resetPipelineVisualizer();

        try {
            const res = await fetch("/api/start-dub", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    video_id: uploadedVideoData.video_id,
                    voice_id: voiceId,
                    audio_mode: audioMode,
                    bg_music_volume: bgVol,
                    voice_volume: 1.0,
                    gemini_api_key: gemKey,
                    elevenlabs_api_key: elKey
                })
            });

            const data = await res.json();
            activeJobId = data.job_id;
            jobStatusBadge.textContent = "Processing";
            jobStatusBadge.style.backgroundColor = "rgba(99, 102, 241, 0.3)";

            // Start polling
            pollInterval = setInterval(pollJobStatus, 1500);
        } catch (err) {
            alert(`Failed to start dubbing: ${err.message}`);
            btnStartDubbing.disabled = false;
            btnStartDubbing.innerHTML = '<span class="btn-icon">⚡</span><span>Start English to Hindi Dubbing</span>';
        }
    }

    async function pollJobStatus() {
        if (!activeJobId) return;

        try {
            const res = await fetch(`/api/job/${activeJobId}`);
            const job = await res.json();

            // Update Progress Bar & Message
            progressBarFill.style.width = `${job.progress}%`;
            statusMessage.textContent = job.message;

            // Update Step Nodes
            updateStepVisualizer(job.step);

            // Update Transcripts if present
            if (job.segments && job.segments.length > 0) {
                renderTranscript(job.segments);
            }

            if (job.status === "completed") {
                clearInterval(pollInterval);
                jobStatusBadge.textContent = "Completed";
                jobStatusBadge.style.backgroundColor = "rgba(16, 185, 129, 0.3)";
                btnStartDubbing.disabled = false;
                btnStartDubbing.innerHTML = '<span class="btn-icon">⚡</span><span>Start English to Hindi Dubbing</span>';

                // Display finished video
                playerEmptyState.classList.add("hidden");
                playerWrapper.classList.remove("hidden");
                outputVideoPlayer.src = job.dubbed_video_url;
                btnDownloadVideo.href = job.dubbed_video_url;
                btnDownloadSrt.href = job.subtitles_srt_url;

                // Auto switch to player tab
                document.querySelector('.tab-btn[data-tab="tab-player"]').click();
            } else if (job.status === "failed") {
                clearInterval(pollInterval);
                jobStatusBadge.textContent = "Failed";
                jobStatusBadge.style.backgroundColor = "rgba(244, 63, 94, 0.3)";
                btnStartDubbing.disabled = false;
                btnStartDubbing.innerHTML = '<span class="btn-icon">⚡</span><span>Start English to Hindi Dubbing</span>';
                alert(`Dubbing failed: ${job.error || "Unknown error"}`);
            }
        } catch (err) {
            console.error("Polling error:", err);
        }
    }

    function resetPipelineVisualizer() {
        Object.values(stepNodes).forEach(node => {
            node.className = "step-node";
        });
        progressBarFill.style.width = "0%";
    }

    function updateStepVisualizer(step) {
        const order = ["extract", "transcribe", "translate", "synthesize", "merge"];
        const stepMap = {
            "extracting_audio": 0,
            "transcribing": 1,
            "translating": 2,
            "synthesizing": 3,
            "merging": 4,
            "completed": 5
        };

        const activeIndex = stepMap[step] !== undefined ? stepMap[step] : -1;

        order.forEach((key, idx) => {
            const node = stepNodes[key];
            if (idx < activeIndex) {
                node.className = "step-node completed";
            } else if (idx === activeIndex) {
                node.className = "step-node active";
            } else {
                node.className = "step-node";
            }
        });
    }

    function renderTranscript(segments) {
        transcriptSegmentsList.innerHTML = "";
        segments.forEach((seg, idx) => {
            const div = document.createElement("div");
            div.className = "segment-item";
            div.innerHTML = `
                <div class="segment-time">#${idx + 1} [${seg.start.toFixed(1)}s - ${seg.end.toFixed(1)}s]</div>
                <div class="segment-en"><strong>EN:</strong> ${seg.english_text || seg.text || ""}</div>
                <div class="segment-hi"><strong>HI:</strong> ${seg.hindi_text || "Translating..."}</div>
            `;
            transcriptSegmentsList.appendChild(div);
        });
    }
});

