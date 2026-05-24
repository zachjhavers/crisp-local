"use client";

import { useState, useRef } from "react";
import * as ort from "onnxruntime-web";

// Force the engine to fetch WebAssembly files from a public CDN
ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/";
ort.env.wasm.numThreads = 1;

// --- TENSOR HELPERS ---
function imageDataToTensor(imageData: ImageData): ort.Tensor {
  const { data, width, height } = imageData;
  const imagePixelsCount = width * height;
  const float32Array = new Float32Array(3 * imagePixelsCount);
  for (let i = 0; i < imagePixelsCount; i++) {
    const rgbaIndex = i * 4;
    float32Array[i] = data[rgbaIndex] / 255.0;
    float32Array[imagePixelsCount + i] = data[rgbaIndex + 1] / 255.0;
    float32Array[2 * imagePixelsCount + i] = data[rgbaIndex + 2] / 255.0;
  }
  return new ort.Tensor("float32", float32Array, [1, 3, height, width]);
}

function tensorToImageData(tensor: ort.Tensor, width: number, height: number): ImageData {
  const floatData = tensor.data as Float32Array;
  const imagePixelsCount = width * height;
  const rgbaArray = new Uint8ClampedArray(4 * imagePixelsCount);
  for (let i = 0; i < imagePixelsCount; i++) {
    const r = Math.min(Math.max(floatData[i] * 255, 0), 255);
    const g = Math.min(Math.max(floatData[imagePixelsCount + i] * 255, 0), 255);
    const b = Math.min(Math.max(floatData[2 * imagePixelsCount + i] * 255, 0), 255);
    const rgbaIndex = i * 4;
    rgbaArray[rgbaIndex] = r;
    rgbaArray[rgbaIndex + 1] = g;
    rgbaArray[rgbaIndex + 2] = b;
    rgbaArray[rgbaIndex + 3] = 255;
  }
  return new ImageData(rgbaArray, width, height);
}

export default function UpscalerApp() {
  const [isDarkMode, setIsDarkMode] = useState<boolean>(true);
  const [status, setStatus] = useState<string>("Waiting for image...");
  const [isProcessing, setIsProcessing] = useState<boolean>(false);
  const [progress, setProgress] = useState<number>(0);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [dimensions, setDimensions] = useState<{w: number, h: number} | null>(null);
  const [hasResult, setHasResult] = useState<boolean>(false);
  const [sliderPos, setSliderPos] = useState<number>(50);
  
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const handleImageUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setIsProcessing(true);
    setHasResult(false);
    setProgress(0);
    setSliderPos(50);
    setStatus("Warming up AI engine...");
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    const img = new Image();
    img.src = url;
    await new Promise((resolve) => (img.onload = resolve));
    setDimensions({ w: img.width, h: img.height });
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    if (!ctx) return;
    canvas.width = img.width;
    canvas.height = img.height;
    ctx.drawImage(img, 0, 0);

    const scale = 4;
    const upscaledCanvas = document.createElement("canvas");
    upscaledCanvas.width = canvas.width * scale;
    upscaledCanvas.height = canvas.height * scale;
    const upscaledCtx = upscaledCanvas.getContext("2d");
    const tempCanvas = document.createElement("canvas");
    const tempCtx = tempCanvas.getContext("2d");
    const fixedDimension = 128;
    const padCanvas = document.createElement("canvas");
    padCanvas.width = fixedDimension;
    padCanvas.height = fixedDimension;
    const padCtx = padCanvas.getContext("2d", { willReadFrequently: true });

    try {
      setStatus("Loading Neural Weights...");
      // Using fallback providers
      const enhanceSession = await ort.InferenceSession.create("/models/color_enhance.onnx", { 
        executionProviders: ["webgl", "wasm"] 
      });
      
      const inputName = enhanceSession.inputNames[0];
      const outputName = enhanceSession.outputNames[0];

      const padding = 16;
      const coreTileSize = fixedDimension - (padding * 2);
      const totalCols = Math.ceil(canvas.width / coreTileSize);
      const totalRows = Math.ceil(canvas.height / coreTileSize);
      const totalTiles = totalCols * totalRows;
      let tileCount = 0;

      for (let row = 0; row < totalRows; row++) {
        for (let col = 0; col < totalCols; col++) {
          tileCount++;
          const currentProgress = Math.round((tileCount / totalTiles) * 100);
          setProgress(currentProgress);
          setStatus(`Enhancing & Upscaling... ${currentProgress}%`);
          const x = col * coreTileSize;
          const y = row * coreTileSize;
          const currentTileWidth = Math.min(coreTileSize, canvas.width - x);
          const currentTileHeight = Math.min(coreTileSize, canvas.height - y);
          const startX = Math.max(0, x - padding);
          const startY = Math.max(0, y - padding);
          const endX = Math.min(canvas.width, x + currentTileWidth + padding);
          const endY = Math.min(canvas.height, y + currentTileHeight + padding);
          const paddedWidth = endX - startX;
          const paddedHeight = endY - startY;
          const rawTileImageData = ctx.getImageData(startX, startY, paddedWidth, paddedHeight);
          padCtx?.clearRect(0, 0, fixedDimension, fixedDimension);
          padCtx?.putImageData(rawTileImageData, 0, 0);
          const safeImageData = padCtx?.getImageData(0, 0, fixedDimension, fixedDimension);
          const currentTensor = imageDataToTensor(safeImageData!);
          const outputs = await enhanceSession.run({ [inputName]: currentTensor });
          const finalOutputTensor = outputs[outputName];
          const upscaledTileData = tensorToImageData(finalOutputTensor, fixedDimension * scale, fixedDimension * scale);
          tempCanvas.width = fixedDimension * scale;
          tempCanvas.height = fixedDimension * scale;
          tempCtx?.putImageData(upscaledTileData, 0, 0);
          upscaledCtx?.drawImage(
            tempCanvas,
            (x - startX) * scale, (y - startY) * scale,
            currentTileWidth * scale, currentTileHeight * scale,
            x * scale, y * scale,
            currentTileWidth * scale, currentTileHeight * scale
          );
          await new Promise((resolve) => setTimeout(resolve, 0));
        }
      }
      canvas.width = upscaledCanvas.width;
      canvas.height = upscaledCanvas.height;
      ctx.drawImage(upscaledCanvas, 0, 0);
      setStatus("Enhancement Complete!");
      setHasResult(true);
      setProgress(100);
    } catch (error) {
      console.error(error);
      setStatus("Error processing image.");
    } finally {
      setIsProcessing(false);
    }
  };

  const handleDownload = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    // Using blob to prevent mobile memory crashes
    canvas.toBlob((blob) => {
      if (!blob) return;
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.download = `CrispLocal-${dimensions?.w}x${dimensions?.h}.png`;
      link.href = url;
      link.click();
      URL.revokeObjectURL(url);
    }, "image/png");
  };

  const bgMain = isDarkMode ? "bg-gray-950 text-white" : "bg-gray-50 text-gray-900";
  const bgCard = isDarkMode ? "bg-gray-900 border-gray-800" : "bg-white border-gray-200 shadow-xl";
  const textMuted = isDarkMode ? "text-gray-400" : "text-gray-500";
  const bgDropzone = isDarkMode ? "bg-gray-800/50 border-gray-700" : "bg-gray-50 border-gray-300";
  const dropzoneHover = isDarkMode ? "hover:border-blue-500 hover:bg-gray-800" : "hover:border-blue-500 hover:bg-gray-100";
  const bgMetric = isDarkMode ? "bg-gray-800 border-gray-700" : "bg-gray-50 border-gray-200";
  const bgProgress = isDarkMode ? "bg-gray-800 border-gray-700" : "bg-gray-200 border-gray-300";
  const btnReset = isDarkMode ? "bg-gray-800 hover:bg-gray-700 text-white" : "bg-gray-200 hover:bg-gray-300 text-gray-900";

  return (
    <main className={`flex flex-col items-center justify-center min-h-screen p-4 md:p-8 transition-colors duration-300 ${bgMain}`}>
      <div className={`max-w-3xl w-full p-6 md:p-10 rounded-2xl transition-colors duration-300 border ${bgCard}`}>
        <div className="flex justify-between items-start mb-8">
          <header>
            <h1 className="text-4xl font-black tracking-tighter mb-3 bg-gradient-to-r from-blue-500 to-indigo-500 bg-clip-text text-transparent">Crisp Local</h1>
            <p className={`text-base font-medium max-w-md ${textMuted}`}>A complete zero-trust upscale and enhancement app.</p>
          </header>
          <button onClick={() => setIsDarkMode(!isDarkMode)} className={`p-2.5 rounded-full transition-all duration-300 ${isDarkMode ? 'bg-gray-800 hover:bg-gray-700 text-yellow-400' : 'bg-gray-100 hover:bg-gray-200 text-gray-800'}`}>
            {isDarkMode ? "🌙" : "☀️"}
          </button>
        </div>

        <div className={`relative border-2 rounded-xl overflow-hidden shadow-inner min-h-[400px] transition-colors duration-300 ${isDarkMode ? 'border-gray-700 bg-gray-800/50' : 'border-gray-300 bg-gray-100/50'}`}>
          {!previewUrl && (
            <div className={`absolute inset-0 z-50 flex flex-col items-center justify-center p-8 text-center cursor-pointer transition-all border-dashed border-2 ${bgDropzone} ${dropzoneHover}`}>
              <input type="file" accept="image/png, image/jpeg" onChange={handleImageUpload} className="absolute inset-0 opacity-0 cursor-pointer z-50" />
              <div className="text-4xl mb-4">🖼️</div>
              <div className="text-blue-500 font-bold text-lg">Click or drop to enhance</div>
            </div>
          )}

          {previewUrl && (
            <div className="relative w-full h-[400px] flex items-center justify-center bg-black overflow-hidden">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img src={previewUrl} className="max-w-full max-h-full object-contain" style={{ opacity: (!hasResult && !isProcessing) ? 0.5 : 1 }} alt="Original Preview" />
                <canvas ref={canvasRef} className={`absolute inset-0 w-full h-full object-contain ${hasResult ? "z-10" : "hidden"}`} style={hasResult ? { clipPath: `inset(0 ${100 - sliderPos}% 0 0)` } : {}} />
                
                {hasResult && (
                  <>
                    <div className="absolute top-0 bottom-0 w-1 bg-white shadow-[0_0_10px_rgba(0,0,0,0.8)] z-20 pointer-events-none" style={{ left: `${sliderPos}%`, transform: 'translateX(-50%)' }}>
                        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-8 h-8 bg-white rounded-full shadow-lg flex items-center justify-center text-gray-900 ring-4 ring-black/10">↔</div>
                    </div>
                    <input type="range" min="0" max="100" value={sliderPos} onChange={(e) => setSliderPos(Number(e.target.value))} className="absolute inset-0 w-full h-full opacity-0 cursor-ew-resize z-30" />
                    <div className="absolute bottom-4 left-4 z-20 bg-black/60 backdrop-blur-md px-3 py-1.5 rounded-md text-xs font-bold text-white shadow-lg pointer-events-none" style={{ opacity: sliderPos > 20 ? 1 : 0 }}>Enhanced Output</div>
                    <div className="absolute bottom-4 right-4 z-20 bg-black/60 backdrop-blur-md px-3 py-1.5 rounded-md text-xs font-bold text-white shadow-lg pointer-events-none" style={{ opacity: sliderPos < 80 ? 1 : 0 }}>Original</div>
                  </>
                )}

                {isProcessing && (
                  <div className="absolute inset-0 flex flex-col items-center justify-center bg-black/60 backdrop-blur-sm z-50">
                    <div className="animate-spin rounded-full h-10 w-10 border-b-2 border-blue-500 mb-4"></div>
                    <p className="text-sm text-white font-mono">{status}</p>
                  </div>
                )}
            </div>
          )}
        </div>

        {dimensions && (
          <div className="mt-8 space-y-6">
            <div className="grid grid-cols-2 gap-4">
              <div className={`p-4 rounded-lg border ${bgMetric}`}>
                <div className={`text-xs uppercase font-bold mb-1 ${textMuted}`}>Input Size</div>
                <div className="text-xl font-mono">{dimensions.w} × {dimensions.h}</div>
              </div>
              <div className={`p-4 rounded-lg border ${bgMetric}`}>
                <div className="text-xs text-blue-500 uppercase font-bold mb-1">Enhanced Target</div>
                <div className="text-xl font-mono text-blue-500">{dimensions.w * 4} × {dimensions.h * 4}</div>
              </div>
            </div>

            <div className={`w-full h-3 rounded-full overflow-hidden border ${bgProgress}`}>
              <div className="h-full bg-blue-600 transition-all duration-300 ease-out" style={{ width: `${progress}%` }} />
            </div>

            <div className="flex gap-3">
               <button onClick={() => window.location.reload()} className={`flex-1 py-3 px-6 rounded-lg font-bold text-sm ${btnReset}`}>Reset</button>
              {hasResult && <button onClick={handleDownload} className="flex-2 py-3 px-6 bg-blue-600 hover:bg-blue-500 text-white rounded-lg font-bold text-sm">Download High-Res PNG</button>}
            </div>
          </div>
        )}
      </div>
    </main>
  );
}