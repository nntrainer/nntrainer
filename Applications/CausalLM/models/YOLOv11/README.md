# YOLOv11m detection inference (nntrainer)

Inference-only YOLOv11m (832×832, nc=1) built from nntrainer's existing layers
(only the C2PSA spatial attention is a custom layer). Matches PyTorch
(ultralytics) end-to-end.

## 1. Convert weights (once)
Needs a Python env with `ultralytics` + `torch`:
```bash
python PyTorch/convert_weights.py \
  --weights /path/to/v11m_832rect_best.pt \
  --out res/yolov11m.safetensors
```
This writes a single nntrainer safetensors whose tensor names match the model.

## 2. (Optional) enable direct image input
The example reads a raw `[1,3,832,832]` float32 `.bin` by default. To also accept
`.jpg/.png` directly, drop in the public-domain `stb_image.h` (NOT committed):
```bash
curl -fsSL https://raw.githubusercontent.com/nothings/stb/master/stb_image.h \
  -o jni/stb_image.h
```
The build auto-detects it (meson `fs.exists`) and enables image decoding +
letterbox. Without it the example still builds and runs on `.bin` input.

## 3. Build
```bash
# from the nntrainer project root
meson setup build -Denable-app=true        # or: meson setup --reconfigure build ...
ninja -C build "Applications/CausalLM/models/YOLOv11/jni/yolov11_infer"
```

## 4. Run
```bash
BIN=./build/Applications/CausalLM/models/YOLOv11/jni/yolov11_infer
RES=Applications/CausalLM/models/YOLOv11/res

# image (requires stb_image.h at build time)
$BIN $RES some_image.jpg
# raw .bin input
$BIN $RES $RES/input_832.bin
# verify vs PyTorch references (ref_*.bin in res/)
YOLO_VERIFY=1 $BIN $RES
```
Output: `Detections (conf>=0.25, xyxy @832): N` followed by `[i] (x1,y1,x2,y2) conf= cls=`.

## Compare with PyTorch
Local helper scripts (not committed): `PyTorch/run_pytorch.py` writes the exact
letterboxed input as a `.bin` and prints PyTorch detections; feed that same
`.bin` to `yolov11_infer` for a bit-exact comparison (`compare.sh` runs both).

> Note: the in-process image letterbox uses a plain bilinear resampler, so
> detections from a `.jpg` differ from the OpenCV/PyTorch path by sub-pixel
> amounts. For an exact match, run on the `.bin` produced by `run_pytorch.py`.
