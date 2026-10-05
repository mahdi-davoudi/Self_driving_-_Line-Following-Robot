import cv2
import numpy as np
import os
import onnxruntime as ort

try:
    from base_config import BASE_DIR
except ImportError:
    BASE_DIR =  os.path.dirname(os.path.abspath(__file__))

class TrafficSignDetector:
    def __init__(self, model_path=None, save_crops=False):
        self.SIGNS = ["ERROR", "STOP", "TURN RIGHT", "TURN LEFT", "STRAIGHT", "PARK"]
        self.count = 0 
        # Saving crops to disk is slow on the Pi's SD card, so it is OFF by default.
        # Set save_crops=True only when you need to collect debug images.
        self.save_crops = save_crops
        
        # Default to the ONNX model
        if model_path is None:
            self.model_file = os.path.join(BASE_DIR, 'assets', 'best416.onnx')
        else:
            self.model_file = model_path
            
        print(f"Loading ONNX model from {self.model_file}...")
        
        # Initialize ONNX session (CPU execution only, perfectly safe for Pi)
        self.session = ort.InferenceSession(self.model_file, providers=['CPUExecutionProvider'])
        
        # Get model input shape automatically
        model_inputs = self.session.get_inputs()
        self.input_name = model_inputs[0].name
        input_shape = model_inputs[0].shape 
        
        # Handle dynamic shapes; fallback to 416x416
        self.input_height = input_shape[2] if isinstance(input_shape[2], int) else 416
        self.input_width = input_shape[3] if isinstance(input_shape[3], int) else 416
        
        if self.save_crops:
            os.makedirs("signs_output", exist_ok=True)

    def letterbox(self, img):
        """
        Manually recreate YOLOv8's image padding technique (letterboxing)
        using NumPy/OpenCV so we don't need PyTorch/Ultralytics installed.
        """
        shape = img.shape[:2]  # [height, width]
        new_shape = (self.input_height, self.input_width)
        
        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        
        # Compute padding
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  
        
        dw /= 2  # Divide padding equally on both sides
        dh /= 2
        
        if shape[::-1] != new_unpad:  # Resize if needed
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
            
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        
        # Add gray borders
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        
        # Format for ONNX: BGR -> RGB -> normalize -> HWC to CHW -> add batch dimension
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_normalized = img_rgb.astype(np.float32) / 255.0
        img_chw = np.transpose(img_normalized, (2, 0, 1))
        img_batch = np.expand_dims(img_chw, axis=0)
        
        return img_batch, r, dw, dh

    def process_frame(self, original_image, debug_frame=None, confidence_threshold=0.35):
        # NOTE ON CLASS ORDER: `self.SIGNS[class_id + 1]` assumes the model was
        # trained with classes in exactly this order (id 0->STOP, 1->TURN
        # RIGHT, 2->TURN LEFT, 3->STRAIGHT, 4->PARK). Double-check this against
        # your training data.yaml / dataset.yaml `names:` list.
        #
        # NOTE ON ROI: no ROI filtering is done here. The caller (city.py)
        # already crops the frame to the sign ROI before calling this method.
        self.count += 1
        
        full_binary = np.zeros(original_image.shape[:2], dtype=np.uint8)
        
        # 1. Preprocess image
        img_tensor, r, dw, dh = self.letterbox(original_image)
        
        # 2. Run pure ONNX inference
        outputs = self.session.run(None, {self.input_name: img_tensor})
        
        # 3. Parse outputs (ONNX returns shape [1, 4+classes, num_boxes])
        preds = np.squeeze(outputs[0], axis=0).T  # -> [num_boxes, 4+classes]
        
        best_coordinate = None
        best_sign_type = -1
        text = ""
        orig_h, orig_w = original_image.shape[:2]
        
        # 4. Vectorized filtering (replaces the per-row Python loop)
        scores = preds[:, 4:]
        class_ids = np.argmax(scores, axis=1)
        confs = scores[np.arange(scores.shape[0]), class_ids]
        
        keep = confs >= confidence_threshold
        if np.any(keep):
            boxes = preds[keep, :4]
            k_confs = confs[keep]
            k_ids = class_ids[keep]
            
            # Map [cx, cy, w, h] on the padded image back to the original image scale
            cx = (boxes[:, 0] - dw) / r
            cy = (boxes[:, 1] - dh) / r
            w = boxes[:, 2] / r
            h = boxes[:, 3] / r
            
            # Clamp boundaries so boxes don't overflow image size
            x1 = np.clip((cx - w / 2).astype(int), 0, orig_w)
            y1 = np.clip((cy - h / 2).astype(int), 0, orig_h)
            x2 = np.clip((cx + w / 2).astype(int), 0, orig_w)
            y2 = np.clip((cy + h / 2).astype(int), 0, orig_h)
            
            # Reject tiny/degenerate boxes (smaller than ~4x4 px)
            areas = (x2 - x1) * (y2 - y1)
            valid = areas >= 16
            
            if np.any(valid):
                # Among valid boxes, pick the most CONFIDENT one
                best = int(np.argmax(np.where(valid, k_confs, -1.0)))
                best_coordinate = [(int(x1[best]), int(y1[best])), (int(x2[best]), int(y2[best]))]
                # Align YOLO zero-indexed IDs to the 1-indexed SIGNS array
                best_sign_type = int(k_ids[best]) + 1

        # 5. Format outputs exactly like the old codebase expects
        if best_sign_type != -1:
            text = f"{self.SIGNS[best_sign_type]}"
            (x1_, y1_), (x2_, y2_) = best_coordinate
            
            # Save the cropped sign (debug only)
            if self.save_crops:
                sign_crop = original_image[y1_:y2_, x1_:x2_]
                if sign_crop.size > 0:
                    cv2.imwrite(f"signs_output/{self.count}_{text}.png", sign_crop)
            
            # Draw overlay
            if debug_frame is not None:
                cv2.rectangle(debug_frame, (x1_, y1_), (x2_, y2_), (0, 255, 0), 2)
                cv2.putText(debug_frame, text, (x1_, y1_ - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

        if text:
            print(text)
        else:
            print("nothing")

        return {
            "coordinate": best_coordinate,
            "binary_mask": full_binary,
            "sign_type": best_sign_type,
            "text": text,
            "debug_frame": debug_frame 
        }