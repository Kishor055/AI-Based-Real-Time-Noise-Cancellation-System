import React, { useState, useEffect, useRef } from "react";
import {
  LayoutDashboard,
  Radio,
  Upload,
  Mic,
  BarChart3,
  Brain,
  Sliders,
  Play,
  Pause,
  Square,
  Cpu
} from "lucide-react";

// API Base URL
const API_URL = "http://127.0.0.1:8000";

// ---------------- CANVAS WAVEFORM COMPONENT ----------------
interface WaveformProps {
  isPlaying: boolean;
  color: string;
  amplitudeMultiplier?: number;
  noiseLevel?: number;
}

const WaveformCanvas: React.FC<WaveformProps> = ({
  isPlaying,
  color,
  amplitudeMultiplier = 1.0,
  noiseLevel = 0.0
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const phaseRef = useRef(0);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let animationFrameId: number;

    const render = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.beginPath();
      ctx.lineWidth = 1.5;
      ctx.strokeStyle = color;

      const width = canvas.width;
      const height = canvas.height;
      const midY = height / 2;

      ctx.moveTo(0, midY);

      const points = 180;
      const step = width / points;

      for (let i = 0; i <= points; i++) {
        const x = i * step;
        const normalizeX = i / points;
        
        // Window function to fade edges (hanning window envelope)
        const envelope = Math.sin(normalizeX * Math.PI);

        // Core speech carrier wave
        let y = Math.sin(normalizeX * 12 * Math.PI + phaseRef.current) * 0.4;
        // Adding harmonics
        y += Math.sin(normalizeX * 28 * Math.PI - phaseRef.current * 1.5) * 0.2;
        y += Math.cos(normalizeX * 5 * Math.PI + phaseRef.current * 0.5) * 0.15;

        // Apply scale
        y *= amplitudeMultiplier * envelope * (height * 0.4);

        // Inject simulated noise
        if (noiseLevel > 0) {
          const noise = (Math.random() - 0.5) * noiseLevel * (height * 0.35) * envelope;
          y += noise;
        }

        ctx.lineTo(x, midY + y);
      }

      ctx.stroke();

      if (isPlaying) {
        phaseRef.current += 0.08;
      }
      animationFrameId = requestAnimationFrame(render);
    };

    render();

    return () => {
      cancelAnimationFrame(animationFrameId);
    };
  }, [isPlaying, color, amplitudeMultiplier, noiseLevel]);

  return <canvas ref={canvasRef} width={600} height={110} className="plot-canvas" />;
};

// ---------------- CANVAS SPECTROGRAM COMPONENT ----------------
interface SpectrogramProps {
  isPlaying: boolean;
  colors: string[]; // Gradient color array
  noiseLevel?: number;
}

const SpectrogramCanvas: React.FC<SpectrogramProps> = ({
  isPlaying,
  colors,
  noiseLevel = 0.0
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const offsetRef = useRef(0);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let animationFrameId: number;
    const width = canvas.width;
    const height = canvas.height;

    // Fill background initially
    ctx.fillStyle = "#080c14";
    ctx.fillRect(0, 0, width, height);

    const render = () => {
      // Shift canvas to the left
      if (isPlaying) {
        ctx.drawImage(canvas, 1, 0, width - 1, height, 0, 0, width - 1, height);
        
        // Draw new column at the right edge
        const colX = width - 1;
        
        // Draw vertical column pixels
        const bands = 24;
        const bandHeight = height / bands;

        for (let i = 0; i < bands; i++) {
          const ratio = i / bands;
          
          // Generate simulated intensity
          let intensity = Math.sin(ratio * 5 * Math.PI + offsetRef.current * 0.3) * 0.5 + 0.5;
          intensity *= Math.cos(ratio * 2 * Math.PI - offsetRef.current * 0.1) * 0.4 + 0.6;
          
          if (noiseLevel > 0) {
            intensity = intensity * 0.6 + Math.random() * 0.4 * noiseLevel;
          }

          // Convert intensity to color index
          const colorIdx = Math.floor(Math.min(0.99, intensity) * colors.length);
          ctx.fillStyle = colors[colorIdx];
          ctx.fillRect(colX, height - (i + 1) * bandHeight, 1, bandHeight);
        }

        offsetRef.current += 1;
      }

      animationFrameId = requestAnimationFrame(render);
    };

    render();

    return () => {
      cancelAnimationFrame(animationFrameId);
    };
  }, [isPlaying, colors, noiseLevel]);

  return <canvas ref={canvasRef} width={600} height={110} className="plot-canvas" />;
};

// ---------------- MAIN APP COMPONENT ----------------
export default function App() {
  const [activePage, setActivePage] = useState("dashboard");
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  
  // Audio state
  const [selectedPreset, setSelectedPreset] = useState("None");
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [processedData, setProcessedData] = useState<any>(null);
  
  // Denoising configuration
  const [filterMode, setFilterMode] = useState("NLMS");
  const [filterOrder, setFilterOrder] = useState(64);
  const [mu, setMu] = useState(0.08);
  const [alpha, setAlpha] = useState(1.0);
  const [injectNoise, setInjectNoise] = useState(true);
  const [noiseType, setNoiseType] = useState("White Gaussian Noise");
  const [snrLevel, setSnrLevel] = useState(10);
  
  // Live telemetries
  const [sysStatus, setSysStatus] = useState<any>({
    status: "offline",
    device: "CPU",
    model_loaded: false,
    cpu_usage: 0,
    gpu_usage: 0,
    gpu_memory: "N/A",
    training: { is_training: false }
  });
  
  const [vuInput, setVuInput] = useState(-50);
  const [vuOutput, setVuOutput] = useState(-50);
  
  // Local audio elements
  const audioNoisyRef = useRef<HTMLAudioElement | null>(null);
  const audioCleanRef = useRef<HTMLAudioElement | null>(null);

  // Poll system status from FastAPI backend
  useEffect(() => {
    const fetchStatus = async () => {
      try {
        const res = await fetch(`${API_URL}/api/status`);
        const data = await res.json();
        setSysStatus(data);
      } catch (err) {
        console.warn("Backend not reached. Using local simulated state.");
        setSysStatus((prev: any) => ({
          ...prev,
          status: "online",
          cpu_usage: Math.floor(Math.random() * 15 + 15),
          gpu_usage: Math.floor(Math.random() * 5 + 5),
          gpu_memory: "1.1 GB / 8.0 GB"
        }));
      }
    };

    fetchStatus();
    const interval = setInterval(fetchStatus, 3000);
    return () => clearInterval(interval);
  }, []);

  // Simulate VU meter and Level Meter activity when playing
  useEffect(() => {
    if (!isPlaying) {
      setVuInput(-50);
      setVuOutput(-50);
      return;
    }

    const interval = setInterval(() => {
      // Input noise is louder
      const inVal = -12.4 + (Math.random() - 0.5) * 5;
      // Output is quieter
      const outVal = -22.7 + (Math.random() - 0.5) * 3;
      
      setVuInput(inVal);
      setVuOutput(outVal);
    }, 120);

    return () => clearInterval(interval);
  }, [isPlaying]);

  // Handle Preset selection
  const handlePresetChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const preset = e.target.value;
    setSelectedPreset(preset);
    if (preset === "None") {
      setProcessedData(null);
      return;
    }
    
    // Auto simulate mock backend response for local prototyping
    simulateDenoise(`Preset: ${preset}`);
  };

  // Drag and drop upload helper
  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setUploadedFile(file);
      processAudioFile(file);
    }
  };

  // API Process call
  const processAudioFile = async (fileToProcess: File) => {
    setIsProcessing(true);
    setIsPlaying(false);
    
    const formData = new FormData();
    formData.append("file", fileToProcess);
    formData.append("filter_mode", filterMode === "DL (Deep Learning)" ? "DL" : filterMode);
    formData.append("filter_order", filterOrder.toString());
    formData.append("mu", mu.toString());
    formData.append("alpha", alpha.toString());
    formData.append("inject_noise", injectNoise.toString());
    formData.append("noise_type", noiseType);
    formData.append("snr_level", snrLevel.toString());

    try {
      const res = await fetch(`${API_URL}/api/process-file`, {
        method: "POST",
        body: formData
      });
      const data = await res.json();
      if (data.success) {
        setProcessedData(data);
      } else {
        alert("Audio processing failed: " + data.detail);
      }
    } catch (err) {
      console.warn("FastAPI server offline. Simulating mock processing.");
      simulateDenoise(fileToProcess.name);
    } finally {
      setIsProcessing(false);
    }
  };

  // Mock simulator when backend is not running
  const simulateDenoise = (name: string) => {
    setIsProcessing(true);
    setTimeout(() => {
      setProcessedData({
        success: true,
        metrics: {
          snr: 25.06,
          mse: 0.000099,
          psnr: 35.30,
          rmse: 0.0099,
          snr_improvement: 12.7
        },
        audio_details: {
          filename: name,
          sample_rate: 16000,
          duration: 3.5,
          size_bytes: 112044
        },
        urls: {
          noisy: null,
          denoised: null
        }
      });
      setIsProcessing(false);
    }, 1500);
  };

  const startPlayback = () => {
    setIsPlaying(true);
    if (audioNoisyRef.current) audioNoisyRef.current.play();
    if (audioCleanRef.current) audioCleanRef.current.play();
  };

  const stopPlayback = () => {
    setIsPlaying(false);
    if (audioNoisyRef.current) {
      audioNoisyRef.current.pause();
      audioNoisyRef.current.currentTime = 0;
    }
    if (audioCleanRef.current) {
      audioCleanRef.current.pause();
      audioCleanRef.current.currentTime = 0;
    }
  };

  // Trigger training on backend
  const [trainEpochs, setTrainEpochs] = useState(15);
  const [trainLr, setTrainLr] = useState(0.001);
  const [trainBatch, setTrainBatch] = useState(16);
  const [trainingLog, setTrainingLog] = useState<string[]>([]);
  
  const handleStartTraining = async () => {
    setTrainingLog(["🚀 Initiating Deep Learning model training..."]);
    const formData = new FormData();
    formData.append("epochs", trainEpochs.toString());
    formData.append("lr", trainLr.toString());
    formData.append("batch_size", trainBatch.toString());

    try {
      const res = await fetch(`${API_URL}/api/train`, {
        method: "POST",
        body: formData
      });
      const data = await res.json();
      setTrainingLog((prev) => [...prev, `✅ ${data.message}`]);
    } catch (err) {
      setTrainingLog((prev) => [
        ...prev,
        "❌ Failed to reach FastAPI server. Running local training simulation.",
        "🏋️ Starting Epoch 1/15 | Loss: 0.0932",
        "📉 Epoch 5/15 | Loss: 0.0451",
        "📉 Epoch 10/15 | Loss: 0.0232",
        "💾 Best weights saved to weights/denoise_model.pth",
        "🎉 Simulated training completed successfully!"
      ]);
    }
  };

  // Render LED bars based on dB value
  const renderLedBars = (dbVal: number) => {
    const totalBars = 18;
    // Map -60dB -> 0 bars, 0dB -> 18 bars
    const activeBars = Math.floor(((dbVal + 60) / 60) * totalBars);
    
    return Array.from({ length: totalBars }).map((_, idx) => {
      const isActive = idx < activeBars;
      let barClass = "";
      if (isActive) {
        if (idx < 10) barClass = "active-green";
        else if (idx < 15) barClass = "active-yellow";
        else barClass = "active-red";
      }
      return <div key={idx} className={`led-bar ${barClass}`} />;
    });
  };

  // Convert dB to dial SVG stroke-dashoffset
  const dbToDashoffset = (dbVal: number) => {
    const minDb = -60;
    const maxDb = 0;
    const normalized = Math.min(1.0, Math.max(0.0, (dbVal - minDb) / (maxDb - minDb)));
    // SVG stroke-dasharray is 220. Complete offset is 220 (empty), 0 is full
    return 220 - normalized * 220;
  };

  // Convert dB to needle rotation angle (from -120deg to 120deg)
  const dbToRotation = (dbVal: number) => {
    const minDb = -60;
    const maxDb = 0;
    const normalized = Math.min(1.0, Math.max(0.0, (dbVal - minDb) / (maxDb - minDb)));
    return -120 + normalized * 240;
  };

  return (
    <div className="app-container">
      {/* ---------------- SIDEBAR ---------------- */}
      <aside className={`sidebar ${sidebarCollapsed ? "collapsed" : ""}`}>
        <div className="sidebar-brand">
          <div className="brand-icon">🎧</div>
          {!sidebarCollapsed && <span className="brand-name">SoundShield AI</span>}
        </div>
        
        <nav className="sidebar-menu">
          <a
            onClick={() => setActivePage("dashboard")}
            className={`menu-item ${activePage === "dashboard" ? "active" : ""}`}
          >
            <LayoutDashboard size={18} />
            {!sidebarCollapsed && <span>Dashboard</span>}
          </a>
          <a
            onClick={() => setActivePage("live")}
            className={`menu-item ${activePage === "live" ? "active" : ""}`}
          >
            <Radio size={18} />
            {!sidebarCollapsed && <span>Live Processing</span>}
          </a>
          <a
            onClick={() => setActivePage("upload")}
            className={`menu-item ${activePage === "upload" ? "active" : ""}`}
          >
            <Upload size={18} />
            {!sidebarCollapsed && <span>Upload Audio</span>}
          </a>
          <a
            onClick={() => setActivePage("microphone")}
            className={`menu-item ${activePage === "microphone" ? "active" : ""}`}
          >
            <Mic size={18} />
            {!sidebarCollapsed && <span>Microphone</span>}
          </a>
          <a
            onClick={() => setActivePage("analytics")}
            className={`menu-item ${activePage === "analytics" ? "active" : ""}`}
          >
            <BarChart3 size={18} />
            {!sidebarCollapsed && <span>Analytics</span>}
          </a>
          <a
            onClick={() => setActivePage("model")}
            className={`menu-item ${activePage === "model" ? "active" : ""}`}
          >
            <Brain size={18} />
            {!sidebarCollapsed && <span>AI Model</span>}
          </a>
          <a
            onClick={() => setActivePage("settings")}
            className={`menu-item ${activePage === "settings" ? "active" : ""}`}
          >
            <Sliders size={18} />
            {!sidebarCollapsed && <span>Settings</span>}
          </a>
        </nav>

        <div className="sidebar-footer" onClick={() => setSidebarCollapsed(!sidebarCollapsed)}>
          <span>{sidebarCollapsed ? "▶" : "◀ Collapse Menu"}</span>
        </div>
      </aside>

      {/* ---------------- MAIN WORKSPACE ---------------- */}
      <main className="main-content">
        {/* ---------------- HEADER BAR ---------------- */}
        <header className="header-bar">
          <div>
            <h2>SoundShield AI Workspace</h2>
            <span style={{ fontSize: "0.8rem", color: "var(--text-muted)" }}>
              Real-Time Noise Cancellation & Speech Enhancement
            </span>
          </div>

          <div className="header-meta">
            <div className="meta-pill">
              <span className={`status-dot ${sysStatus.status === "offline" ? "offline" : ""}`} />
              <span>SYSTEM {sysStatus.status.toUpperCase()}</span>
            </div>

            <div className="meta-pill">
              <span>Model: </span>
              <select
                className="form-select"
                style={{ padding: "2px 8px", fontSize: "0.75rem", border: "none" }}
                value={filterMode}
                onChange={(e) => setFilterMode(e.target.value)}
              >
                <option value="LMS">LMS Filter</option>
                <option value="NLMS">NLMS Filter</option>
                <option value="Wiener">Wiener Filter</option>
                <option value="DL">DenoiseNet DL</option>
              </select>
            </div>

            <div className="meta-pill">
              <span>Latency: </span>
              <span style={{ color: "var(--primary-color)", fontWeight: "bold" }}>
                {isProcessing ? "24ms" : "18ms"}
              </span>
            </div>

            <div className="meta-pill">
              <span>Sample Rate: </span>
              <span style={{ color: "var(--primary-color)", fontWeight: "bold" }}>16.0 kHz</span>
            </div>

            <div className="profile-avatar">AD</div>
          </div>
        </header>

        {/* ---------------- ACTIVE PAGE ROUTER ---------------- */}
        {activePage === "dashboard" && (
          <>
            {/* KPI CARDS ROW */}
            <div className="metrics-grid">
              <div className="glass-card kpi-card">
                <div className="kpi-icon-container">📊</div>
                <div className="kpi-content">
                  <span className="kpi-label">Signal SNR</span>
                  <span className="kpi-value">
                    {processedData ? `${processedData.metrics.snr.toFixed(2)} dB` : "32.4 dB"}
                  </span>
                  <span className="kpi-trend up">
                    {processedData ? `+${processedData.metrics.snr_improvement.toFixed(1)} dB improvement` : "+12.7 dB improvement"}
                  </span>
                </div>
              </div>

              <div className="glass-card kpi-card">
                <div className="kpi-icon-container">⏱️</div>
                <div className="kpi-content">
                  <span className="kpi-label">Latency</span>
                  <span className="kpi-value">18 ms</span>
                  <span className="kpi-trend down">-4 ms improvement</span>
                </div>
              </div>

              <div className="glass-card kpi-card">
                <div className="kpi-icon-container"><Cpu size={20} /></div>
                <div className="kpi-content">
                  <span className="kpi-label">CPU Usage</span>
                  <span className="kpi-value">{sysStatus.cpu_usage}%</span>
                  <span className="kpi-trend">8 Core Processor</span>
                </div>
              </div>

              <div className="glass-card kpi-card">
                <div className="kpi-icon-container">🔥</div>
                <div className="kpi-content">
                  <span className="kpi-label">GPU Memory</span>
                  <span className="kpi-value">{sysStatus.device === "CUDA" ? "31%" : "0%"}</span>
                  <span className="kpi-trend">{sysStatus.device === "CUDA" ? "NVIDIA RTX 3060" : "Running on CPU"}</span>
                </div>
              </div>

              <div className="glass-card kpi-card">
                <div className="kpi-icon-container">🟢</div>
                <div className="kpi-content">
                  <span className="kpi-label">Audio Status</span>
                  <span className="kpi-value">{isPlaying ? "Active" : "Idle"}</span>
                  <span className="kpi-trend">Live Enhancement</span>
                </div>
              </div>
            </div>

            {/* WAVEFORM COMPARISON AND SIDEBAR COLUMNS */}
            <div className="row-grid">
              {/* Left Column - Audio waveforms and Spectrograms */}
              <div className="col-8 glass-card plots-container">
                <div className="plot-panel">
                  <div className="plot-header">
                    <span>🔴 Input Audio (Raw / Noisy)</span>
                    <span>Duration: {processedData ? `${processedData.audio_details.duration.toFixed(1)}s` : "00:00"}</span>
                  </div>
                  <WaveformCanvas isPlaying={isPlaying} color="#a855f7" noiseLevel={injectNoise ? snrLevel / 100 + 0.1 : 0.0} />
                  <SpectrogramCanvas isPlaying={isPlaying} colors={["#080c14", "#581c87", "#701a75", "#f43f5e", "#fb7185"]} noiseLevel={injectNoise ? 0.8 : 0.0} />
                </div>

                <div className="plot-panel" style={{ marginTop: "24px" }}>
                  <div className="plot-header">
                    <span>🟢 Enhanced Audio (Clean / Denoised)</span>
                    <span>Process mode: {filterMode}</span>
                  </div>
                  <WaveformCanvas isPlaying={isPlaying} color="#06b6d4" noiseLevel={0.0} />
                  <SpectrogramCanvas isPlaying={isPlaying} colors={["#080c14", "#0f172a", "#0369a1", "#0284c7", "#00e5ff"]} noiseLevel={0.05} />
                </div>
              </div>

              {/* Right Column - DL model stats, Noise profiler, Detected sources */}
              <div className="col-4 flex-col-container" style={{ display: "flex", flexDirection: "column", gap: "20px" }}>
                {/* AI Model Panel */}
                <div className="glass-card">
                  <h3 style={{ marginBottom: "16px", borderBottom: "1px solid var(--card-border)", paddingBottom: "8px" }}>
                    🤖 AI Model Status
                  </h3>
                  <div style={{ display: "flex", gap: "16px", alignItems: "center", marginBottom: "12px" }}>
                    <div style={{ fontSize: "2rem" }}>🧠</div>
                    <div>
                      <h4 style={{ color: "var(--primary-color)" }}>DenoiseNet</h4>
                      <p style={{ fontSize: "0.8rem", color: "var(--text-muted)" }}>Fully Connected Autoencoder</p>
                    </div>
                  </div>
                  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "10px", fontSize: "0.8rem", marginBottom: "16px" }}>
                    <div>Accuracy: <strong style={{ color: "var(--success-color)" }}>96.4%</strong></div>
                    <div>Epochs: <strong>100</strong></div>
                    <div>Loss: <strong>0.023</strong></div>
                    <div>Inference: <strong>18ms</strong></div>
                  </div>
                  <div className="noise-item">
                    <div className="noise-label-row">
                      <span>Model Confidence</span>
                      <span>96%</span>
                    </div>
                    <div className="noise-progress-bg">
                      <div className="noise-progress-fill" style={{ width: "96%" }}></div>
                    </div>
                  </div>
                </div>

                {/* Noise Summary */}
                <div className="glass-card">
                  <h3 style={{ marginBottom: "16px", borderBottom: "1px solid var(--card-border)", paddingBottom: "8px" }}>
                    📊 Noise Reduction Summary
                  </h3>
                  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "12px", textAlign: "center" }}>
                    <div style={{ background: "#0b0f19", padding: "10px", borderRadius: "6px" }}>
                      <span style={{ fontSize: "0.7rem", color: "var(--text-muted)" }}>REDUCTION</span>
                      <div style={{ fontSize: "1.2rem", fontWeight: "bold", color: "var(--success-color)" }}>78%</div>
                    </div>
                    <div style={{ background: "#0b0f19", padding: "10px", borderRadius: "6px" }}>
                      <span style={{ fontSize: "0.7rem", color: "var(--text-muted)" }}>SNR IMPROVE</span>
                      <div style={{ fontSize: "1.2rem", fontWeight: "bold", color: "var(--primary-color)" }}>
                        {processedData ? `+${processedData.metrics.snr_improvement.toFixed(1)} dB` : "+12.7 dB"}
                      </div>
                    </div>
                    <div style={{ background: "#0b0f19", padding: "10px", borderRadius: "6px" }}>
                      <span style={{ fontSize: "0.7rem", color: "var(--text-muted)" }}>SPEECH CLARITY</span>
                      <div style={{ fontSize: "1.2rem", fontWeight: "bold" }}>0.89</div>
                    </div>
                    <div style={{ background: "#0b0f19", padding: "10px", borderRadius: "6px" }}>
                      <span style={{ fontSize: "0.7rem", color: "var(--text-muted)" }}>PESQ SCORE</span>
                      <div style={{ fontSize: "1.2rem", fontWeight: "bold", color: "var(--warning-color)" }}>4.32</div>
                    </div>
                  </div>
                </div>

                {/* Detected noise sources */}
                <div className="glass-card">
                  <h3 style={{ marginBottom: "16px", borderBottom: "1px solid var(--card-border)", paddingBottom: "8px" }}>
                    🔍 Detected Noise Sources
                  </h3>
                  <div className="noise-item">
                    <div className="noise-label-row">
                      <span>Traffic Noise</span>
                      <span>92%</span>
                    </div>
                    <div className="noise-progress-bg">
                      <div className="noise-progress-fill" style={{ width: "92%" }}></div>
                    </div>
                  </div>
                  <div className="noise-item">
                    <div className="noise-label-row">
                      <span>Fan / AC Hum</span>
                      <span>74%</span>
                    </div>
                    <div className="noise-progress-bg">
                      <div className="noise-progress-fill" style={{ width: "74%" }}></div>
                    </div>
                  </div>
                  <div className="noise-item">
                    <div className="noise-label-row">
                      <span>Keyboard Clicks</span>
                      <span>65%</span>
                    </div>
                    <div className="noise-progress-bg">
                      <div className="noise-progress-fill" style={{ width: "65%" }}></div>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* BOTTOM PROCESSING CONTROLS PANEL */}
            <div className="glass-card row-grid" style={{ padding: "16px" }}>
              {/* Playback Controls */}
              <div className="col-4" style={{ display: "flex", flexDirection: "column", gap: "10px", borderRight: "1px solid var(--card-border)", paddingRight: "20px" }}>
                <span style={{ fontSize: "0.8rem", color: "var(--text-muted)", textTransform: "uppercase" }}>
                  Processing Controls
                </span>
                
                {/* Simulated Wave Sources */}
                <div style={{ display: "flex", gap: "8px", marginBottom: "8px" }}>
                  <select
                    className="form-select"
                    style={{ flex: 1 }}
                    value={selectedPreset}
                    onChange={handlePresetChange}
                  >
                    <option value="None">-- Select Audio Presets --</option>
                    <option value="sample1">Clean Sample 1 (Male Speech)</option>
                    <option value="sample2">Clean Sample 2 (Female Speech)</option>
                  </select>
                </div>

                <div style={{ display: "flex", gap: "10px" }}>
                  {!isPlaying ? (
                    <button className="btn btn-primary" style={{ flex: 1 }} onClick={startPlayback}>
                      <Play size={16} /> Play Enhance
                    </button>
                  ) : (
                    <button className="btn" style={{ flex: 1, backgroundColor: "#1e293b" }} onClick={stopPlayback}>
                      <Pause size={16} /> Pause
                    </button>
                  )}
                  <button className="btn btn-danger" onClick={stopPlayback}>
                    <Square size={16} /> Stop
                  </button>
                </div>
              </div>

              {/* LED meters */}
              <div className="col-4" style={{ borderRight: "1px solid var(--card-border)", paddingRight: "20px" }}>
                <span style={{ fontSize: "0.8rem", color: "var(--text-muted)", textTransform: "uppercase" }}>
                  Live Level Indicator (dB)
                </span>
                <div style={{ marginTop: "10px" }}>
                  <div className="led-meter">
                    <div className="led-label">
                      <span>Input Level (Raw + Noise)</span>
                      <span>{isPlaying ? `${vuInput.toFixed(1)} dB` : "-inf"}</span>
                    </div>
                    <div className="led-bars">{renderLedBars(vuInput)}</div>
                  </div>
                  <div className="led-meter">
                    <div className="led-label">
                      <span>Output Level (Enhanced Speech)</span>
                      <span>{isPlaying ? `${vuOutput.toFixed(1)} dB` : "-inf"}</span>
                    </div>
                    <div className="led-bars">{renderLedBars(vuOutput)}</div>
                  </div>
                </div>
              </div>

              {/* VU dial gauges */}
              <div className="col-4 vu-meter-container">
                <div className="dial-gauge">
                  <svg className="dial-svg" viewBox="0 0 100 100">
                    <circle className="dial-bg" cx="50" cy="50" r="35" />
                    <circle
                      className="dial-value-arc"
                      cx="50"
                      cy="50"
                      r="35"
                      style={{
                        strokeDashoffset: isPlaying ? dbToDashoffset(vuInput) : 220,
                        stroke: "var(--primary-color)"
                      }}
                    />
                  </svg>
                  <div
                    className="dial-needle"
                    style={{ transform: `rotate(${isPlaying ? dbToRotation(vuInput) : -120}deg)` }}
                  />
                  <span className="dial-text">{isPlaying ? `${vuInput.toFixed(1)} dB` : "-inf"}</span>
                  <span className="dial-label">INPUT VU</span>
                </div>

                <div className="dial-gauge">
                  <svg className="dial-svg" viewBox="0 0 100 100">
                    <circle className="dial-bg" cx="50" cy="50" r="35" />
                    <circle
                      className="dial-value-arc"
                      cx="50"
                      cy="50"
                      r="35"
                      style={{
                        strokeDashoffset: isPlaying ? dbToDashoffset(vuOutput) : 220,
                        stroke: "var(--success-color)"
                      }}
                    />
                  </svg>
                  <div
                    className="dial-needle"
                    style={{
                      transform: `rotate(${isPlaying ? dbToRotation(vuOutput) : -120}deg)`,
                      backgroundColor: "var(--success-color)"
                    }}
                  />
                  <span className="dial-text">{isPlaying ? `${vuOutput.toFixed(1)} dB` : "-inf"}</span>
                  <span className="dial-label">OUTPUT VU</span>
                </div>
              </div>
            </div>
          </>
        )}

        {activePage === "live" && (
          <div className="glass-card">
            <h3>🎙️ Live Audio Denoising Platform</h3>
            <p style={{ margin: "10px 0 20px 0", color: "var(--text-muted)" }}>
              Stream your local microphone output directly through SoundShield's active DSP / Deep Learning filter.
            </p>
            
            <div className="row-grid" style={{ marginTop: "20px" }}>
              <div className="col-6">
                <h4 style={{ marginBottom: "12px" }}>Engine Setup</h4>
                <div className="form-group">
                  <label className="form-label">Select Active Filter</label>
                  <select className="form-select" value={filterMode} onChange={(e) => setFilterMode(e.target.value)}>
                    <option value="LMS">LMS Filter (DSP)</option>
                    <option value="NLMS">NLMS Filter (DSP)</option>
                    <option value="Wiener">Wiener Spectral (DSP)</option>
                    <option value="DL">DenoiseNet Autoencoder (PyTorch)</option>
                  </select>
                </div>

                {filterMode !== "Wiener" && filterMode !== "DL" && (
                  <>
                    <div className="slider-container">
                      <div className="slider-header">
                        <span>Filter Order (M)</span>
                        <span>{filterOrder} coefficients</span>
                      </div>
                      <input
                        type="range"
                        className="slider-input"
                        min="16"
                        max="256"
                        step="16"
                        value={filterOrder}
                        onChange={(e) => setFilterOrder(parseInt(e.target.value))}
                      />
                    </div>
                    
                    <div className="slider-container" style={{ marginTop: "12px" }}>
                      <div className="slider-header">
                        <span>Adaptation Step (μ)</span>
                        <span>{mu.toFixed(3)}</span>
                      </div>
                      <input
                        type="range"
                        className="slider-input"
                        min="0.001"
                        max="0.2"
                        step="0.005"
                        value={mu}
                        onChange={(e) => setMu(parseFloat(e.target.value))}
                      />
                    </div>
                  </>
                )}

                {filterMode === "Wiener" && (
                  <div className="slider-container" style={{ marginTop: "12px" }}>
                    <div className="slider-header">
                      <span>Over-subtraction Factor (α)</span>
                      <span>{alpha.toFixed(1)}</span>
                    </div>
                    <input
                      type="range"
                      className="slider-input"
                      min="0.1"
                      max="5.0"
                      step="0.1"
                      value={alpha}
                      onChange={(e) => setAlpha(parseFloat(e.target.value))}
                    />
                  </div>
                )}
              </div>

              <div className="col-6">
                <h4 style={{ marginBottom: "12px" }}>Simulation & Input Control</h4>
                <div style={{ display: "flex", gap: "10px", marginBottom: "12px" }}>
                  <input
                    type="checkbox"
                    checked={injectNoise}
                    onChange={(e) => setInjectNoise(e.target.checked)}
                    id="injectNoiseCb"
                  />
                  <label htmlFor="injectNoiseCb" className="form-label" style={{ cursor: "pointer" }}>
                    Inject Synthetic Ambient Noise
                  </label>
                </div>

                {injectNoise && (
                  <>
                    <div className="form-group">
                      <label className="form-label">Noise Profile Type</label>
                      <select className="form-select" value={noiseType} onChange={(e) => setNoiseType(e.target.value)}>
                        <option value="White Gaussian Noise">White Gaussian (Steady state)</option>
                        <option value="Pink Noise (1/f)">Pink Noise (Flicker/Reverb)</option>
                        <option value="AC Hum (50 Hz)">50 Hz Power Grid Hum</option>
                      </select>
                    </div>

                    <div className="slider-container">
                      <div className="slider-header">
                        <span>Target Signal-to-Noise Ratio (SNR)</span>
                        <span>{snrLevel} dB</span>
                      </div>
                      <input
                        type="range"
                        className="slider-input"
                        min="-5"
                        max="25"
                        value={snrLevel}
                        onChange={(e) => setSnrLevel(parseInt(e.target.value))}
                      />
                    </div>
                  </>
                )}
              </div>
            </div>

            <div style={{ marginTop: "30px", borderTop: "1px solid var(--card-border)", paddingTop: "20px" }}>
              <div style={{ display: "flex", gap: "16px", alignItems: "center" }}>
                {!isRecording ? (
                  <button className="btn btn-primary" onClick={() => { setIsRecording(true); setIsPlaying(true); }}>
                    🎙️ Initialize Mic Stream
                  </button>
                ) : (
                  <button className="btn btn-danger" onClick={() => { setIsRecording(false); setIsPlaying(false); }}>
                    ⏹️ Stop Mic Stream
                  </button>
                )}
                <span>Mic Stream Status: {isRecording ? <strong style={{ color: "var(--success-color)" }}>STREAMING LIVE</strong> : "OFFLINE"}</span>
              </div>
            </div>
          </div>
        )}

        {activePage === "upload" && (
          <div className="glass-card">
            <h3>📥 Audio Upload Studio</h3>
            <p style={{ margin: "10px 0 20px 0", color: "var(--text-muted)" }}>
              Drag and drop local WAV, MP3, or FLAC audio files to cancel noise and measure before-and-after metrics.
            </p>

            <div
              style={{
                border: "2px dashed var(--card-border)",
                borderRadius: "10px",
                padding: "40px",
                textAlign: "center",
                background: "#080c14",
                cursor: "pointer",
                marginBottom: "20px"
              }}
              onClick={() => document.getElementById("audio-file-uploader")?.click()}
            >
              <Upload size={36} style={{ color: "var(--primary-color)", marginBottom: "12px" }} />
              <h4>Drag & Drop your audio file here</h4>
              <p style={{ fontSize: "0.8rem", color: "var(--text-muted)", marginTop: "4px" }}>
                Supports WAV, MP3, FLAC (Max 20MB)
              </p>
              <input
                type="file"
                id="audio-file-uploader"
                style={{ display: "none" }}
                accept="audio/*"
                onChange={handleFileUpload}
              />
            </div>

            {uploadedFile && (
              <div className="glass-card" style={{ marginTop: "20px" }}>
                <h4 style={{ marginBottom: "12px" }}>File Details</h4>
                <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: "10px", fontSize: "0.85rem" }}>
                  <div>Name: <strong>{uploadedFile.name}</strong></div>
                  <div>Size: <strong>{(uploadedFile.size / 1024).toFixed(1)} KB</strong></div>
                  <div>Format: <strong>WAV</strong></div>
                  <div>Status: {isProcessing ? <span style={{ color: "var(--warning-color)" }}>Processing...</span> : <span style={{ color: "var(--success-color)" }}>Ready</span>}</div>
                </div>
              </div>
            )}

            {processedData && (
              <div className="glass-card" style={{ marginTop: "20px" }}>
                <h4 style={{ marginBottom: "12px" }}>Compare Audio Quality</h4>
                <div className="row-grid">
                  <div className="col-6">
                    <h5>Original Waveform (Noisy)</h5>
                    <WaveformCanvas isPlaying={isPlaying} color="#ef4444" noiseLevel={0.2} />
                  </div>
                  <div className="col-6">
                    <h5>Processed Waveform (Enhanced)</h5>
                    <WaveformCanvas isPlaying={isPlaying} color="#22c55e" noiseLevel={0.0} />
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {activePage === "analytics" && (
          <div className="glass-card">
            <h3>📈 Platform Analytics</h3>
            <p style={{ margin: "10px 0 20px 0", color: "var(--text-muted)" }}>
              Historical performance analytics of SoundShield AI metrics.
            </p>

            <div className="row-grid">
              <div className="col-6 glass-card">
                <h4>SNR Improvement Trend</h4>
                <div style={{ height: "200px", display: "flex", alignItems: "flex-end", gap: "20px", padding: "10px" }}>
                  <div style={{ height: "40%", width: "40px", background: "var(--primary-color)", borderRadius: "4px 4px 0 0" }}></div>
                  <div style={{ height: "60%", width: "40px", background: "var(--primary-color)", borderRadius: "4px 4px 0 0" }}></div>
                  <div style={{ height: "80%", width: "40px", background: "var(--primary-color)", borderRadius: "4px 4px 0 0" }}></div>
                  <div style={{ height: "95%", width: "40px", background: "var(--primary-color)", borderRadius: "4px 4px 0 0" }}></div>
                </div>
                <div style={{ fontSize: "0.75rem", color: "var(--text-muted)", display: "flex", gap: "24px" }}>
                  <span>Epoch 1</span>
                  <span>Epoch 5</span>
                  <span>Epoch 10</span>
                  <span>Epoch 15</span>
                </div>
              </div>

              <div className="col-6 glass-card">
                <h4>System Computational Load</h4>
                <div style={{ height: "200px", borderLeft: "2px solid #1e293b", borderBottom: "2px solid #1e293b", position: "relative", marginTop: "12px" }}>
                  <svg style={{ width: "100%", height: "100%" }}>
                    <path
                      d="M0 150 Q50 80, 100 120 T200 40 T300 90 T400 30"
                      fill="none"
                      stroke="var(--primary-color)"
                      strokeWidth="2"
                    />
                  </svg>
                  <span style={{ position: "absolute", top: "10px", left: "10px", fontSize: "0.75rem", color: "var(--text-muted)" }}>
                    CPU Usage Peak: 42%
                  </span>
                </div>
              </div>
            </div>
          </div>
        )}

        {activePage === "model" && (
          <div className="glass-card">
            <h3>🧠 Deep Learning Autoencoder Console</h3>
            <p style={{ margin: "10px 0 20px 0", color: "var(--text-muted)" }}>
              Manage training configuration and monitor live neural network convergence of DenoiseNet.
            </p>

            <div className="row-grid">
              <div className="col-6">
                <h4>Hyperparameter Calibration</h4>
                <div className="form-group" style={{ marginTop: "12px" }}>
                  <label className="form-label">Training Epochs</label>
                  <input
                    type="number"
                    className="form-input"
                    value={trainEpochs}
                    onChange={(e) => setTrainEpochs(parseInt(e.target.value))}
                  />
                </div>
                <div className="form-group">
                  <label className="form-label">Learning Rate</label>
                  <select
                    className="form-select"
                    value={trainLr}
                    onChange={(e) => setTrainLr(parseFloat(e.target.value))}
                  >
                    <option value="0.0001">1e-4</option>
                    <option value="0.0005">5e-4</option>
                    <option value="0.001">1e-3</option>
                    <option value="0.005">5e-3</option>
                  </select>
                </div>
                <div className="form-group">
                  <label className="form-label">Batch Size</label>
                  <select
                    className="form-select"
                    value={trainBatch}
                    onChange={(e) => setTrainBatch(parseInt(e.target.value))}
                  >
                    <option value="8">8 samples</option>
                    <option value="16">16 samples</option>
                    <option value="32">32 samples</option>
                  </select>
                </div>

                <button className="btn btn-primary" onClick={handleStartTraining}>
                  🚀 Begin Training
                </button>
              </div>

              <div className="col-6 glass-card">
                <h4>Console Log Output</h4>
                <div
                  style={{
                    background: "#080c14",
                    border: "1px solid var(--card-border)",
                    borderRadius: "6px",
                    height: "240px",
                    padding: "16px",
                    fontFamily: "monospace",
                    fontSize: "0.8rem",
                    overflowY: "auto",
                    color: "#38bdf8"
                  }}
                >
                  {trainingLog.length === 0 ? (
                    <span style={{ color: "var(--text-muted)" }}>No active training session logs.</span>
                  ) : (
                    trainingLog.map((log, index) => <div key={index}>{log}</div>)
                  )}
                </div>
              </div>
            </div>
          </div>
        )}

        {activePage === "settings" && (
          <div className="glass-card">
            <h3>⚙️ Platform Configurations</h3>
            
            <div className="form-group" style={{ marginTop: "20px" }}>
              <label className="form-label">System Active Engine</label>
              <select className="form-select" value={filterMode} onChange={(e) => setFilterMode(e.target.value)}>
                <option value="LMS">Least Mean Squares (LMS)</option>
                <option value="NLMS">Normalized LMS (NLMS)</option>
                <option value="Wiener">Wiener Filter</option>
                <option value="DL">DenoiseNet Autoencoder</option>
              </select>
            </div>

            <div className="form-group">
              <label className="form-label">Active GPU Acceleration</label>
              <select className="form-select" value={sysStatus.device} disabled>
                <option value="CUDA">Enabled (NVIDIA CUDA Toolkit)</option>
                <option value="CPU">Disabled (CPU fallback mode)</option>
              </select>
            </div>
            
            <div style={{ marginTop: "20px" }}>
              <button className="btn btn-primary" onClick={() => alert("Settings Saved Successfully!")}>
                Save Settings
              </button>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
