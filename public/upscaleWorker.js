importScripts("https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.all.min.js");

let enhanceSession = null;

// Configure ONNX Runtime for stable, single-threaded web execution
self.ort = self.ort || {};
self.ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/";
self.ort.env.wasm.numThreads = 1; 

const sessionOptions = {
  executionProviders: ["wasm"], // CPU fallback for maximum stability
  graphOptimizationLevel: "all",
  executionMode: "sequential"
};

function imageDataToTensor(imageData) {
  const { data, width, height } = imageData;
  const imagePixelsCount = width * height;
  const float32Array = new Float32Array(3 * imagePixelsCount);
  for (let i = 0; i < imagePixelsCount; i++) {
    const rgbaIndex = i * 4;
    float32Array[i] = data[rgbaIndex] / 255.0;
    float32Array[imagePixelsCount + i] = data[rgbaIndex + 1] / 255.0;
    float32Array[2 * imagePixelsCount + i] = data[rgbaIndex + 2] / 255.0;
  }
  return new self.ort.Tensor("float32", float32Array, [1, 3, height, width]);
}

function tensorToImageData(tensor, width, height) {
  const floatData = tensor.data;
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

async function initializeSession() {
  if (!enhanceSession) {
    try {
      const upscaleModelUrl = "https://huggingface.co/zacharyhavers/crisp-local-models/resolve/main/upscale_enhance.onnx";
      enhanceSession = await self.ort.InferenceSession.create(upscaleModelUrl, sessionOptions);
    } catch (error) {
      console.error("Failed to initialize enhancement session:", error);
      throw error;
    }
  }
}

self.onmessage = async (event) => {
  const { imageData, originalWidth, originalHeight } = event.data;

  try {
    await initializeSession();

    const scale = 4;
    const upscaledWidth = originalWidth * scale;
    const upscaledHeight = originalHeight * scale;

    const offscreenCanvas = new OffscreenCanvas(upscaledWidth, upscaledHeight);
    const upscaledCtx = offscreenCanvas.getContext("2d");

    const tempCanvas = new OffscreenCanvas(128 * scale, 128 * scale);
    const tempCtx = tempCanvas.getContext("2d", { willReadFrequently: true });

    const padCanvas = new OffscreenCanvas(128, 128);
    const padCtx = padCanvas.getContext("2d", { willReadFrequently: true });

    const fixedDimension = 128;
    const padding = 16;
    const coreTileSize = fixedDimension - padding * 2;
    const totalCols = Math.ceil(originalWidth / coreTileSize);
    const totalRows = Math.ceil(originalHeight / coreTileSize);
    const totalTiles = totalCols * totalRows;

    let tileCount = 0;

    const enhanceInputName = enhanceSession.inputNames[0];
    const enhanceOutputName = enhanceSession.outputNames[0];

    const sourceImageCanvas = new OffscreenCanvas(originalWidth, originalHeight);
    const sourceImageCtx = sourceImageCanvas.getContext("2d");
    sourceImageCtx.putImageData(imageData, 0, 0);

    for (let row = 0; row < totalRows; row++) {
      for (let col = 0; col < totalCols; col++) {
        tileCount++;

        const x = col * coreTileSize;
        const y = row * coreTileSize;
        const currentTileWidth = Math.min(coreTileSize, originalWidth - x);
        const currentTileHeight = Math.min(coreTileSize, originalHeight - y);
        
        const startX = Math.max(0, x - padding);
        const startY = Math.max(0, y - padding);
        const endX = Math.min(originalWidth, x + currentTileWidth + padding);
        const endY = Math.min(originalHeight, y + currentTileHeight + padding);
        const paddedWidth = endX - startX;
        const paddedHeight = endY - startY;

        padCtx.clearRect(0, 0, fixedDimension, fixedDimension);
        padCtx.drawImage(sourceImageCanvas, startX, startY, paddedWidth, paddedHeight, 0, 0, paddedWidth, paddedHeight);
        const safeImageData = padCtx.getImageData(0, 0, fixedDimension, fixedDimension);

        const inputTensor = imageDataToTensor(safeImageData);
        const enhanceOutputs = await enhanceSession.run({ [enhanceInputName]: inputTensor });
        const enhancedTensor = enhanceOutputs[enhanceOutputName];
        const enhancedTileData = tensorToImageData(enhancedTensor, fixedDimension * scale, fixedDimension * scale);

        tempCtx.putImageData(enhancedTileData, 0, 0);
        
        upscaledCtx.drawImage(
          tempCanvas,
          (x - startX) * scale,
          (y - startY) * scale,
          currentTileWidth * scale,
          currentTileHeight * scale,
          x * scale,
          y * scale,
          currentTileWidth * scale,
          currentTileHeight * scale
        );

        const progressPercent = Math.round((tileCount / totalTiles) * 90);
        self.postMessage({
          type: "progress",
          progress: progressPercent,
          status: `Super-Resolution processing... ${progressPercent}%`,
        });

        if (tileCount % 5 === 0) {
          await new Promise((resolve) => setTimeout(resolve, 0));
        }
      }
    }

    // Maintained hardware-accelerated micro-contrast to reduce AI haze
    const finalCanvas = new OffscreenCanvas(upscaledWidth, upscaledHeight);
    const finalCtx = finalCanvas.getContext("2d");
    finalCtx.filter = "contrast(1.05) saturate(1.10) brightness(1.02)";
    finalCtx.drawImage(offscreenCanvas, 0, 0);

    const finalImageData = finalCtx.getImageData(0, 0, upscaledWidth, upscaledHeight);
    
    self.postMessage({
      type: "complete",
      imageData: finalImageData,
    }, [finalImageData.data.buffer]);

  } catch (error) {
    self.postMessage({ type: "error", error: error.message });
  }
};