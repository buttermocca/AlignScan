# Changelog

## 1.0.0
- Initial Windows desktop application.
- Multi-document outline detection and perspective correction.
- Manual four-point selection and draggable corner adjustment.
- Adjustable brightness, contrast, saturation, denoising and dust cleanup.
- PNG/JPG individual export and JPG quality control.
- Optional YOLOv8 Detection ONNX inference with DirectML-first/CPU fallback.
- Animated mascot splash screen and Windows EXE build workflow.

## 1.1.0
- Faster document detection with a bounded working image and a three-level edge sweep.
- Adjustable 0–100 automatic detection sensitivity.
- Drag-and-drop loading from the application and scan canvas.
- Mouse-wheel zoom preview (35–500%).
- Sharpness slider using an unsharp mask.
- Cached downscaled preview source and lighter live-preview denoising/dust cleanup.
- Updated pastel lavender interface and cute receipt/girl/cat mascot banner.
- Windows packaging switched to onedir for faster cold startup (keep the complete folder together).

## 1.2.0
- Added mouse-wheel zoom to the main scan canvas (10–1200%) and panning with middle/right mouse buttons.
- Added a live 3× magnifier around the pointer while manually placing crop corners and dragging corner handles.
- Added horizontal and vertical ruler drawing; apply the guide to deskew the complete image and remap all detected document corners.
- Added exact 90°, 180°, 270° clockwise rotations, 90° counter-clockwise rotation, and horizontal/vertical flips.
- Added manual arbitrary-angle rotation with synchronized slider and numeric input.
- Added affine point transforms so existing crop regions follow rotate/flip/deskew changes.
- Added unit tests for rotation, flip, arbitrary-angle canvas expansion, and ruler-based deskew.

## 1.2.1
- Corrected yolo_doc_v1.onnx handling to match upstream 960px letterbox preprocessing.
- Changed the YOLO document confidence default from 0.30 to the upstream 0.25 threshold.
- Added a confidence adjustment control (5–80%) and an explicit AI re-run button.
- Added runtime diagnostics: model input size, output tensor shape, maximum confidence, candidate and kept detection counts.
- Clarified that yolo_doc_v1 is a single-class coarse axis-aligned document-box detector, not a four-corner or multi-receipt flatbed scan model.
- Added ONNX output-layout and 960px letterbox regression tests.

## 1.2.2
- Rejected YOLO predictions that cover at least 88% of the current image/window to prevent the full scanner page from being reported as one receipt.
- Added overlapping left/right and top/bottom tile inference when the full image yields fewer than two valid document boxes.
- Merged duplicate detections from overlapping windows with global NMS.
- Removed stale AI proposal rectangles before each AI rerun.
- AI status now reports tile inference count and full-frame false-positive rejection count.
- Added tests for multi-window coverage on the previously reported 2527×3369 scan dimensions.
