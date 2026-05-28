# Crisp Local

A browser-based AI photo upscaler. Drop in any image and get a 4x resolution enhancement — processed entirely on your device using a neural network model running via ONNX Runtime Web. No uploads, no accounts, no data ever leaves your machine.

## How it works

1. You drop an image into the app (PNG, JPEG, or WEBP, up to 2048×2048)
2. A Web Worker loads the ONNX upscaling model and runs inference locally
3. The enhanced image is rendered to a canvas — you can compare before/after with a drag slider
4. Download the result at 4× the original resolution

All processing happens in the browser. The app never makes a network request with your image data.

## Tech stack

- **Next.js 16** + TypeScript + Tailwind CSS
- **ONNX Runtime Web** — runs the upscaling neural network in-browser
- **Web Worker** — keeps the UI responsive during inference
- **react-dropzone** — drag and drop upload
- **react-zoom-pan-pinch** — before/after comparison slider

## Running locally

```bash
# Install dependencies
pnpm install

# Start the dev server
pnpm dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

```bash
# Production build
pnpm build && pnpm start
```

## Supported formats

| Format | Input | Output |
|--------|-------|--------|
| PNG | ✓ | ✓ |
| JPEG | ✓ | PNG |
| WEBP | ✓ | PNG |

Max input size: 2048 × 2048 px. Output is always 4× the input dimensions.
