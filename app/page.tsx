"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import { useDropzone } from "react-dropzone";

type FaceRegion = { x: number; y: number; width: number; height: number; imgSrc: string };

// --- MINIMALIST SVG ICONS ---
const UploadIcon = () => (
  <svg fill="none" stroke="currentColor" strokeWidth="1" viewBox="0 0 24 24" className="w-16 h-16 mb-6 text-gray-500 transition-transform duration-500 group-hover:scale-110 group-hover:-translate-y-2">
    <path strokeLinecap="round" strokeLinejoin="round" d="M12 16.5V9.75m0 0l3 3m-3-3l-3 3M6.75 19.5a4.5 4.5 0 01-1.41-8.775 5.25 5.25 0 0110.233-2.33 3 3 0 013.758 3.848A3.752 3.752 0 0118 19.5H6.75z" />
  </svg>
);

const SparklesIcon = () => (
  <svg fill="none" stroke="currentColor" strokeWidth="1.5" viewBox="0 0 24 24" className="w-5 h-5 text-gray-400">
    <path strokeLinecap="round" strokeLinejoin="round" d="M9.813 15.904L9 18.75l-.813-2.846a4.5 4.5 0 00-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 003.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 003.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 00-3.09 3.09zM18.259 8.715L18 9.75l-.259-1.035a3.375 3.375 0 00-2.455-2.456L14.25 6l1.036-.259a3.375 3.375 0 002.455-2.456L18 2.25l.259 1.035a3.375 3.375 0 002.456 2.456L21.75 6l-1.035.259a3.375 3.375 0 00-2.456 2.456z" />
  </svg>
);

const DownloadIcon = () => (
  <svg fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24" className="w-4 h-4 inline-block mr-2">
    <path strokeLinecap="round" strokeLinejoin="round" d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5M16.5 12L12 16.5m0 0L7.5 12m4.5 4.5V3" />
  </svg>
);

const ResetIcon = () => (
  <svg fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24" className="w-4 h-4 inline-block mr-2">
    <path strokeLinecap="round" strokeLinejoin="round" d="M16.023 9.348h4.992v-.001M2.985 19.644v-4.992m0 0h4.992m-4.993 0l3.181 3.183a8.25 8.25 0 0013.803-3.7M4.031 9.865a8.25 8.25 0 0113.803-3.7l3.181 3.182m0-4.991v4.99" />
  </svg>
);

const SliderArrowsIcon = () => (
  <svg fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24" className="w-5 h-5 text-gray-900">
    <path strokeLinecap="round" strokeLinejoin="round" d="M8 7h8M8 17h8M5 12l3-3v6zM19 12l-3 3V9z" />
  </svg>
);

export default function UpscalerApp() {
  const [status, setStatus] = useState<string>("Waiting for image...");
  const [isProcessing, setIsProcessing] = useState<boolean>(false);
  const [progress, setProgress] = useState<number>(0);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [dimensions, setDimensions] = useState<{ w: number; h: number } | null>(null);
  const [hasResult, setHasResult] = useState<boolean>(false);
  const [sliderPos, setSliderPos] = useState<number>(50);
  const [enableFaceRestoration, setEnableFaceRestoration] = useState<boolean>(true);
  const [detectedFaces, setDetectedFaces] = useState<FaceRegion[]>([]);
  const [selectedFaces, setSelectedFaces] = useState<Set<number>>(new Set());
  const [showFaceSelection, setShowFaceSelection] = useState<boolean>(false);
  const [isFaceModelLoaded, setIsFaceModelLoaded] = useState<boolean>(false);

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const workerRef = useRef<Worker | null>(null);
  const isLoadingModel = useRef<boolean>(false);
  const enableFaceRestorationRef = useRef(enableFaceRestoration);
  
  // Create a ref to hold the dynamically imported library
  const faceapiRef = useRef<typeof import("@vladmandic/face-api") | null>(null);

  useEffect(() => {
    enableFaceRestorationRef.current = enableFaceRestoration;
  }, [enableFaceRestoration]);

  // --- INITIALIZE FACE AI MODEL (Dynamically Imported to fix SSR crashes) ---
  useEffect(() => {
    if (isLoadingModel.current) return;
    isLoadingModel.current = true;

    const loadFaceModel = async () => {
      try {
        // Dynamically import the library only on the browser
        const api = await import("@vladmandic/face-api");
        faceapiRef.current = api;
        
        await api.nets.tinyFaceDetector.loadFromUri('/models');
        setIsFaceModelLoaded(true);
      } catch (error) {
        console.error("Failed to load face detection model:", error);
      }
    };
    loadFaceModel();
  }, []);

  const completeProcess = useCallback((msg: string = "Enhancement Complete") => {
    setStatus(msg);
    setHasResult(true);
    setProgress(100);
    setIsProcessing(false);
  }, []);

  // --- AI FACE DETECTION w/ THUMBNAILS & BOUNDARY CLAMPING ---
  const detectFaces = useCallback(async () => {
    const canvas = canvasRef.current;
    const api = faceapiRef.current;
    if (!canvas || !api) return completeProcess();

    try {
      const detections = await api.detectAllFaces(
        canvas,
        new api.TinyFaceDetectorOptions({ inputSize: 960, scoreThreshold: 0.3 })
      );

      const faces: FaceRegion[] = [];
      const tempCanvas = document.createElement("canvas");
      tempCanvas.width = 120;
      tempCanvas.height = 120;
      const tCtx = tempCanvas.getContext("2d");

      detections.forEach(detection => {
        const pad = detection.box.width * 0.6; 
        
        const safeX = Math.max(0, Math.floor(detection.box.x - pad / 2));
        const safeY = Math.max(0, Math.floor(detection.box.y - pad / 2));
        const safeWidth = Math.min(canvas.width - safeX, Math.floor(detection.box.width + pad));
        const safeHeight = Math.min(canvas.height - safeY, Math.floor(detection.box.height + pad));

        if (tCtx) {
          tCtx.clearRect(0, 0, 120, 120);
          tCtx.drawImage(canvas, safeX, safeY, safeWidth, safeHeight, 0, 0, 120, 120);
          faces.push({ 
            x: safeX, y: safeY, width: safeWidth, height: safeHeight, 
            imgSrc: tempCanvas.toDataURL("image/jpeg", 0.8) 
          });
        }
      });

      if (faces.length > 0) {
        setDetectedFaces(faces);
        setSelectedFaces(new Set(faces.map((_, i) => i)));
        setIsProcessing(false); 
        setShowFaceSelection(true); 
        setStatus(`Detected ${faces.length} face region(s).`);
      } else {
        completeProcess("No faces detected.");
      }
    } catch (error) {
      console.error("Face detection failed:", error);
      completeProcess("Face detection failed.");
    }
  }, [completeProcess]);

  // --- WORKER LIFECYCLE ---
  useEffect(() => {
    workerRef.current = new Worker("/upscaleWorker.js");

    workerRef.current.onmessage = (event) => {
      const { type, progress: workerProgress, status: workerStatus, imageData, error } = event.data;

      if (type === "progress") {
        setProgress(workerProgress);
        setStatus(workerStatus);
      } else if (type === "faceProgress") {
        setProgress(90 + Math.round((workerProgress / 100) * 10));
        setStatus(`Restoring facial region ${event.data.index + 1}...`);
      } else if (type === "complete" || type === "facesRestored") {
        const canvas = canvasRef.current;
        if (canvas && imageData) {
          const ctx = canvas.getContext("2d");
          if (ctx) {
            canvas.width = imageData.width;
            canvas.height = imageData.height;
            ctx.putImageData(imageData, 0, 0);
          }
        }

        if (type === "complete" && enableFaceRestorationRef.current) {
          setStatus("Analyzing image for faces...");
          detectFaces();
        } else {
          completeProcess();
        }
      } else if (type === "error") {
        console.error("Worker error:", error);
        setStatus(`Processing Error`);
        setIsProcessing(false);
      }
    };

    return () => {
      workerRef.current?.terminate();
    };
  }, [detectFaces, completeProcess]);

  // --- REACT-DROPZONE HANDLER ---
  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    const file = acceptedFiles[0];
    if (!file) return;
    
    setIsProcessing(true);
    setHasResult(false);
    setProgress(0);
    setSliderPos(50);
    setStatus("Initializing neural engine...");
    
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    
    const img = new Image();
    img.src = url;
    await new Promise((resolve) => (img.onload = resolve));
    
    setDimensions({ w: img.width, h: img.height });
    
    const canvas = document.createElement("canvas");
    canvas.width = img.width;
    canvas.height = img.height;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    
    ctx.drawImage(img, 0, 0);
    const imageData = ctx.getImageData(0, 0, img.width, img.height);
    
    try {
      setStatus("Loading weights...");
      if (workerRef.current) {
        workerRef.current.postMessage({
          imageData,
          originalWidth: img.width,
          originalHeight: img.height,
        }, [imageData.data.buffer]);
      }
    } catch (error) {
      console.error(error);
      setStatus("Initialization failed.");
      setIsProcessing(false);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({ 
    onDrop, 
    accept: { 'image/png': [], 'image/jpeg': [], 'image/webp': [] },
    maxFiles: 1,
    disabled: isProcessing 
  });

  const toggleFaceSelection = (index: number) => {
    const newSelected = new Set(selectedFaces);
    if (newSelected.has(index)) newSelected.delete(index);
    else newSelected.add(index);
    setSelectedFaces(newSelected);
  };

  const applyFaceRestoration = () => {
    const canvas = canvasRef.current;
    if (!canvas || selectedFaces.size === 0) return completeProcess();

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const selectedRegions = detectedFaces.filter((_, i) => selectedFaces.has(i));

    setShowFaceSelection(false);
    setIsProcessing(true);
    setStatus("Restoring selected features...");

    if (workerRef.current) {
      workerRef.current.postMessage({
        type: "restoreFaces",
        regions: selectedRegions,
        outputImageData: imageData,
      }, [imageData.data.buffer]); 
    }
  };

  const handleDownload = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    canvas.toBlob((blob) => {
      if (!blob) return;
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.download = `CrispLocal-${dimensions?.w ? dimensions.w * 4 : "output"}.png`;
      link.href = url;
      link.click();
      URL.revokeObjectURL(url);
    }, "image/png");
  };

  // SVG Circular Progress math
  const progressRadius = 44;
  const progressCircumference = 2 * Math.PI * progressRadius;
  const progressOffset = progressCircumference - (progress / 100) * progressCircumference;

  return (
    <main className="flex flex-col items-center justify-center min-h-screen p-4 md:p-8 bg-gray-950 text-gray-100 font-sans selection:bg-blue-500/30">
      <div className="max-w-4xl w-full p-6 md:p-10 rounded-3xl border border-gray-800 bg-gray-900 shadow-2xl">
        
        {/* HEADER & TUTORIAL */}
        <header className="mb-8">
          <h1 className="text-4xl md:text-5xl font-black tracking-tighter mb-4 text-white">
            Crisp Local
          </h1>
          <p className="text-sm md:text-base text-gray-400 mb-6 max-w-2xl leading-relaxed">
            A zero-trust AI upscaler. Enhance your low-resolution images directly in your browser using local machine learning models. No cloud uploads, absolute privacy.
          </p>
          
          <div className="flex flex-wrap gap-4 text-xs font-medium text-gray-400 bg-gray-950 p-4 rounded-xl border border-gray-800 w-fit">
            <div className="flex items-center gap-2">
              <span className="flex items-center justify-center w-5 h-5 rounded bg-gray-800 text-gray-200">1</span> 
              <span>Upload File</span>
            </div>
            <div className="w-px h-5 bg-gray-800 hidden sm:block"></div>
            <div className="flex items-center gap-2">
              <span className="flex items-center justify-center w-5 h-5 rounded bg-gray-800 text-gray-200">2</span> 
              <span>Process Locally</span>
            </div>
            <div className="w-px h-5 bg-gray-800 hidden sm:block"></div>
            <div className="flex items-center gap-2">
              <span className="flex items-center justify-center w-5 h-5 rounded bg-gray-800 text-gray-200">3</span> 
              <span>Review & Save</span>
            </div>
          </div>
        </header>

        {/* SETTINGS TOGGLE */}
        <div className="mb-6 p-4 rounded-2xl border border-gray-800 bg-gray-950/50 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div className="flex items-center justify-center w-10 h-10 rounded-xl bg-gray-900 border border-gray-800 shadow-inner">
              <SparklesIcon />
            </div>
            <div>
              <div className="font-semibold text-sm text-gray-200 tracking-wide">Face Restoration</div>
              <div className="text-xs text-gray-500 mt-0.5">Automatically detect and reconstruct facial details</div>
            </div>
          </div>
          <button
            onClick={() => setEnableFaceRestoration(!enableFaceRestoration)}
            disabled={isProcessing || !isFaceModelLoaded}
            className={`relative inline-flex h-6 w-11 items-center rounded-full transition-all duration-300 ${
              enableFaceRestoration ? 'bg-blue-600' : 'bg-gray-700'
            } ${isProcessing || !isFaceModelLoaded ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer hover:ring-2 ring-blue-500/30 ring-offset-2 ring-offset-gray-900'}`}
          >
            <span className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform duration-300 shadow-sm ${enableFaceRestoration ? 'translate-x-6' : 'translate-x-1'}`} />
          </button>
        </div>

        {/* MAIN WORKSPACE */}
        <div className="relative border border-gray-800 rounded-2xl overflow-hidden h-150 bg-gray-950">
          
          {/* DRAG AND DROP ZONE */}
          {!previewUrl && (
            <div {...getRootProps()} className={`absolute inset-0 z-50 flex flex-col items-center justify-center p-8 text-center cursor-pointer transition-all duration-300 group ${isDragActive ? 'bg-gray-900' : 'bg-gray-950 hover:bg-gray-900'}`}>
              <input {...getInputProps()} />
              <UploadIcon />
              <div className="text-xl font-semibold mb-2 text-gray-200 tracking-tight">
                {isDragActive ? "Drop image to upscale" : "Click or drag file to this area"}
              </div>
              <div className="text-xs text-gray-500">Supports PNG, JPEG, WEBP up to 2048x2048</div>
            </div>
          )}

          {/* STATIC PREVIEW & SLIDER */}
          {previewUrl && (
            <div className="relative w-full h-full flex items-center justify-center overflow-hidden bg-gray-950">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img src={previewUrl} className="max-w-full max-h-full object-contain pointer-events-none" style={{ opacity: (!hasResult && !isProcessing) ? 0.3 : 1 }} alt="Original" />
              <canvas ref={canvasRef} className={`absolute inset-0 w-full h-full object-contain pointer-events-none ${hasResult ? "z-10" : "hidden"}`} style={hasResult ? { clipPath: `inset(0 ${100 - sliderPos}% 0 0)` } : {}} />
              
              {hasResult && (
                <div className="absolute inset-0 w-full h-full z-30 group">
                  <div className="absolute top-0 bottom-0 w-px bg-white/50 pointer-events-none transition-all" style={{ left: `${sliderPos}%`, transform: 'translateX(-50%)' }}>
                      <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-8 h-8 bg-white rounded-full shadow-lg flex items-center justify-center text-gray-900 group-hover:scale-105 transition-transform border border-gray-200">
                        <SliderArrowsIcon />
                      </div>
                  </div>
                  <input 
                    type="range" min="0" max="100" value={sliderPos} 
                    onChange={(e) => setSliderPos(Number(e.target.value))} 
                    className="absolute inset-0 w-full h-full opacity-0 cursor-ew-resize" 
                  />
                  <div className="absolute bottom-6 left-6 z-20 bg-gray-900/90 backdrop-blur-md px-4 py-2 rounded text-[10px] font-bold tracking-widest uppercase text-gray-100 border border-gray-800 shadow-sm pointer-events-none" style={{ opacity: sliderPos > 15 ? 1 : 0, transition: 'opacity 0.2s' }}>Enhanced</div>
                  <div className="absolute bottom-6 right-6 z-20 bg-gray-900/90 backdrop-blur-md px-4 py-2 rounded text-[10px] font-bold tracking-widest uppercase text-gray-400 border border-gray-800 shadow-sm pointer-events-none" style={{ opacity: sliderPos < 85 ? 1 : 0, transition: 'opacity 0.2s' }}>Original</div>
                </div>
              )}
            </div>
          )}

          {/* LOADING OVERLAY w/ CIRCULAR PROGRESS */}
          {isProcessing && (
            <div className="absolute inset-0 flex flex-col items-center justify-center bg-gray-950/90 backdrop-blur-md z-40">
              <div className="relative flex items-center justify-center mb-6">
                <svg className="w-24 h-24 transform -rotate-90">
                  <circle cx="48" cy="48" r="44" stroke="currentColor" strokeWidth="3" fill="transparent" className="text-gray-800" />
                  <circle 
                    cx="48" 
                    cy="48" 
                    r="44" 
                    stroke="currentColor" 
                    strokeWidth="3" 
                    fill="transparent" 
                    strokeDasharray={progressCircumference} 
                    strokeDashoffset={progressOffset} 
                    className="text-blue-500 transition-all duration-300 ease-out" 
                    strokeLinecap="round"
                  />
                </svg>
                <div className="absolute flex items-center justify-center text-gray-100 text-lg font-light">
                  {progress}%
                </div>
              </div>
              <p className="text-xs text-gray-400 font-mono tracking-wider uppercase">{status}</p>
            </div>
          )}

          {/* FACE SELECTION MODAL */}
          {showFaceSelection && (
            <div className="absolute inset-0 bg-gray-950/95 backdrop-blur-xl z-50 flex flex-col items-center justify-center p-6">
              <div className="w-full max-w-xl flex flex-col max-h-full">
                <div className="text-center mb-8">
                  <h3 className="text-xl font-semibold text-gray-100 tracking-tight">Select regions to enhance</h3>
                  <p className="text-sm text-gray-500 mt-2">{status}</p>
                </div>
                
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 mb-8 overflow-y-auto custom-scrollbar flex-1 min-h-0 px-2">
                  {detectedFaces.map((face, index) => (
                    <button
                      key={index}
                      onClick={() => toggleFaceSelection(index)}
                      className={`flex items-center gap-4 p-3 rounded-xl border transition-all text-left ${
                        selectedFaces.has(index) ? "bg-blue-900/20 border-blue-500/50" : "bg-gray-900 border-gray-800 hover:border-gray-700"
                      }`}
                    >
                      {/* eslint-disable-next-line @next/next/no-img-element */}
                      <img src={face.imgSrc} alt={`Face ${index}`} className="w-12 h-12 rounded object-cover bg-gray-950" />
                      <div className="flex-1">
                        <div className={`text-sm font-medium ${selectedFaces.has(index) ? 'text-gray-100' : 'text-gray-400'}`}>Region {index + 1}</div>
                        <div className="flex items-center gap-1.5 mt-1">
                          <div className={`w-1.5 h-1.5 rounded-full ${selectedFaces.has(index) ? 'bg-blue-500' : 'bg-gray-600'}`}></div>
                          <span className="text-[10px] uppercase tracking-wider text-gray-500">{selectedFaces.has(index) ? "Selected" : "Ignored"}</span>
                        </div>
                      </div>
                    </button>
                  ))}
                </div>

                <div className="flex gap-3 mt-auto">
                  <button onClick={() => completeProcess()} className="flex-1 py-3 px-4 bg-gray-900 hover:bg-gray-800 text-gray-300 rounded-xl text-sm font-medium transition-all border border-gray-800">
                    Skip
                  </button>
                  <button onClick={applyFaceRestoration} disabled={selectedFaces.size === 0} className="flex-2 py-3 px-4 bg-blue-600 hover:bg-blue-500 text-white rounded-xl text-sm font-medium transition-all disabled:opacity-50 disabled:cursor-not-allowed">
                    Enhance Selection
                  </button>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* BOTTOM METRICS & ACTIONS */}
        {dimensions && hasResult && (
          <div className="mt-6 flex flex-col sm:flex-row gap-3 animate-in fade-in slide-in-from-bottom-4 duration-500">
            <button onClick={() => window.location.reload()} className="flex-1 py-3 px-6 rounded-xl font-medium text-sm transition-all bg-gray-900 hover:bg-gray-800 text-gray-300 border border-gray-800">
              <ResetIcon /> Start Over
            </button>
            <button onClick={handleDownload} className="flex-[2] py-3 px-6 bg-gray-100 hover:bg-white text-gray-900 rounded-xl font-medium text-sm transition-all shadow-sm">
              <DownloadIcon /> Save Output ({dimensions.w * 4}x{dimensions.h * 4})
            </button>
          </div>
        )}
      </div>
      
      {/* Scrollbar styling for modal */}
      <style dangerouslySetInnerHTML={{__html: `
        .custom-scrollbar::-webkit-scrollbar {
          width: 4px;
        }
        .custom-scrollbar::-webkit-scrollbar-track {
          background: transparent;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb {
          background: #374151;
          border-radius: 4px;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb:hover {
          background: #4B5563;
        }
      `}} />
    </main>
  );
}