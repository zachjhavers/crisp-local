"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import { useDropzone } from "react-dropzone";

// --- MINIMALIST SVG ICONS ---
const UploadIcon = () => (
  <svg fill="none" stroke="currentColor" strokeWidth="1" viewBox="0 0 24 24" className="w-16 h-16 mb-6 text-gray-500 transition-transform duration-500 group-hover:scale-110 group-hover:-translate-y-2">
    <path strokeLinecap="round" strokeLinejoin="round" d="M12 16.5V9.75m0 0l3 3m-3-3l-3 3M6.75 19.5a4.5 4.5 0 01-1.41-8.775 5.25 5.25 0 0110.233-2.33 3 3 0 013.758 3.848A3.752 3.752 0 0118 19.5H6.75z" />
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

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const workerRef = useRef<Worker | null>(null);

  const completeProcess = useCallback((msg: string = "Enhancement Complete") => {
    setStatus(msg);
    setHasResult(true);
    setProgress(100);
    setIsProcessing(false);
  }, []);

  // --- WORKER LIFECYCLE ---
  useEffect(() => {
    workerRef.current = new Worker("/upscaleWorker.js");

    workerRef.current.onmessage = (event) => {
      const { type, progress: workerProgress, status: workerStatus, imageData, error } = event.data;

      if (type === "progress") {
        setProgress(workerProgress);
        setStatus(workerStatus);
      } else if (type === "complete") {
        const canvas = canvasRef.current;
        if (canvas && imageData) {
          const ctx = canvas.getContext("2d");
          if (ctx) {
            canvas.width = imageData.width;
            canvas.height = imageData.height;
            ctx.putImageData(imageData, 0, 0);
          }
        }
        completeProcess();
      } else if (type === "error") {
        console.error("Worker error:", error);
        setStatus(`Processing Error`);
        setIsProcessing(false);
      }
    };

    return () => {
      workerRef.current?.terminate();
    };
  }, [completeProcess]);

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
    </main>
  );
}