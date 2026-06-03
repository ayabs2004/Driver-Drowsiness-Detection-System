"""
app.py – Driver Drowsiness Detection — Photo Upload GUI
=========================================================
Fix: all CNN crops now taken from clean `img` (no drawings).
     Landmark dots and boxes are drawn only on `annotated`.
"""

import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import dlib

MODEL_PATH          = r"c:\Users\aya\Documents\driver ai project\drowsiness_model.keras"
PLACEHOLDER_PATH    = r"c:\Users\aya\Documents\driver ai project\upload_placeholder.png"
DLIB_PREDICTOR_PATH = r"c:\Users\aya\Documents\driver ai project\shape_predictor_68_face_landmarks.dat"

CLASSES  = ['closed_eye', 'closed_mouth', 'open_eye', 'open_mouth']
IMG_SIZE = (96, 96)
DISPLAY_MAX = 600
EYE_PAD   = 0.25
MOUTH_PAD = 0.20


# ── Loaders ───────────────────────────────────────────────────────────────

def load_detectors():
    if not os.path.exists(DLIB_PREDICTOR_PATH):
        raise FileNotFoundError(
            f"dlib predictor not found at:\n{DLIB_PREDICTOR_PATH}\n\n"
            "Download from:\nhttps://github.com/italojs/facial-landmarks-recognition"
            "/raw/master/shape_predictor_68_face_landmarks.dat"
        )
    return dlib.get_frontal_face_detector(), dlib.shape_predictor(DLIB_PREDICTOR_PATH)


def load_dnn_face_model():
    proto = "deploy.prototxt.txt"
    model = "res10_300x300_ssd_iter_140000.caffemodel"
    if os.path.exists(proto) and os.path.exists(model):
        try:
            return cv2.dnn.readNetFromCaffe(proto, model)
        except Exception as e:
            print(f"DNN face model load error: {e}")
    return None


# ── Image helpers ─────────────────────────────────────────────────────────

def resize_for_display(img, max_size=DISPLAY_MAX):
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return img
    scale = min(max_size / w, max_size / h, 1.0)
    return cv2.resize(img, (int(w * scale), int(h * scale)))


def cv2_to_tk(cv_img):
    return ImageTk.PhotoImage(
        Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)))


def prepare_patch(bgr_crop):
    """BGR crop → (display RGB uint8, normalized float32 model input)."""
    rgb     = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, IMG_SIZE)
    return resized, resized.reshape(1, *IMG_SIZE, 3) / 255.0


def landmarks_bbox(points, pad_frac, img_h, img_w):
    """Return padded bounding box (x1,y1,x2,y2) around a set of landmarks."""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    bx1, by1, bx2, by2 = min(xs), min(ys), max(xs), max(ys)
    bw, bh = bx2 - bx1, by2 - by1
    px = int(bw * pad_frac)
    py = int(bh * pad_frac)
    return (max(0, bx1 - px), max(0, by1 - py),
            min(img_w, bx2 + px), min(img_h, by2 + py))


# ── Core detection ────────────────────────────────────────────────────────

def run_detection(image_path, model, detectors, dnn_net=None):
    dlib_detector, dlib_predictor = detectors

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    h, w = img.shape[:2]
    scale = 800 / max(h, w)
    img = cv2.resize(img, (int(w * scale), int(h * scale)))
    h, w = img.shape[:2]

    # FIX: annotated is the only image we draw on.
    #      img stays completely clean — all CNN crops come from img.
    annotated = img.copy()

    results, patches = [], []

    # ── Face detection ────────────────────────────────────────────────
    face_rects = []
    if dnn_net is not None:
        blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), (104.0, 177.0, 123.0))
        dnn_net.setInput(blob)
        dets = dnn_net.forward()
        for i in range(dets.shape[2]):
            conf = float(dets[0, 0, i, 2])
            if conf < 0.7:
                continue
            box = (dets[0, 0, i, 3:7] * np.array([w, h, w, h])).astype(int)
            x1, y1, x2, y2 = (max(0, box[0]), max(0, box[1]),
                               min(w, box[2]), min(h, box[3]))
            if (x2 - x1) < 80 or (y2 - y1) < 80:
                continue
            face_rects.append(dlib.rectangle(x1, y1, x2, y2))

    if not face_rects:
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        face_rects = dlib_detector(rgb_img, 1)

    if not face_rects:
        results.append("⚠  No face detected in the image.")
        return annotated, results, patches

    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    for face_idx, rect in enumerate(face_rects):
        fx1, fy1 = rect.left(), rect.top()
        fx2, fy2 = rect.right(), rect.bottom()

        # Draw face box on annotated only
        cv2.rectangle(annotated, (fx1, fy1), (fx2, fy2), (255, 100, 0), 2)
        cv2.putText(annotated, "Driver detected", (fx1, fy1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2)

        # 68 landmarks
        shape = dlib_predictor(rgb_img, rect)
        pts   = [(shape.part(i).x, shape.part(i).y) for i in range(68)]

        # Draw landmark dots on annotated only
        for px, py in pts:
            cv2.circle(annotated, (px, py), 1, (0, 255, 255), -1)

        # ── Eyes ──────────────────────────────────────────────────────
        for eye_idx, eye_pts in enumerate([pts[36:42], pts[42:48]]):
            eye_name = "Left eye" if eye_idx == 0 else "Right eye"
            x1, y1, x2, y2 = landmarks_bbox(eye_pts, EYE_PAD, h, w)

            # FIX: crop from clean img, not annotated
            crop_bgr = img[y1:y2, x1:x2]
            if crop_bgr.size == 0:
                results.append(f"⚠ {eye_name}: crop failed")
                continue

            patch_rgb, eye_input = prepare_patch(crop_bgr)
            patches.append((patch_rgb, eye_name))

            pred      = model.predict(eye_input, verbose=0)[0]
            eye_probs = np.array([pred[0], pred[2]])  # closed_eye, open_eye
            eye_probs /= eye_probs.sum()

            label = 'closed_eye' if np.argmax(eye_probs) == 0 else 'open_eye'
            conf  = eye_probs.max()

            color = (0, 220, 0) if label == 'open_eye' else (0, 0, 220)
            # Draw box and label on annotated only
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated, f"{label} ({conf:.0%})",
                        (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
            results.append(f"{eye_name}: {label} ({conf:.1%})")
        # ── Mouth ─────────────────────────────────────────────────────
        mouth_pts = pts[48:68]
        x1, y1, x2, y2 = landmarks_bbox(mouth_pts, MOUTH_PAD, h, w)

        # FIX: crop from clean img, not annotated
        crop_bgr = img[y1:y2, x1:x2]
        if crop_bgr.size == 0:
            results.append("Mouth: crop failed")
        else:
            patch_rgb, m_input = prepare_patch(crop_bgr)
            patches.append((patch_rgb, "Mouth"))

            pred        = model.predict(m_input, verbose=0)[0]
            mouth_probs = np.array([pred[1], pred[3]])  # closed_mouth, open_mouth
            mouth_probs /= mouth_probs.sum()

            m_label   = 'open_mouth' if np.argmax(mouth_probs) == 1 else 'closed_mouth'
            m_conf    = mouth_probs.max()
            display_m = "yawning" if m_label == 'open_mouth' else "closed"

            m_color = (0, 165, 255) if display_m == "yawning" else (0, 255, 0)
            # Draw on annotated only
            cv2.rectangle(annotated, (x1, y1), (x2, y2), m_color, 2)
            cv2.putText(annotated, f"Mouth: {display_m}",
                        (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, m_color, 1)
            results.append(f"Mouth: {display_m} ({m_conf:.1%})")

    return annotated, results, patches


# ── Status verdict ────────────────────────────────────────────────────────

def determine_overall_status(results):
    closed  = sum(1 for r in results if 'closed_eye' in r)
    yawning = any('yawning' in r for r in results)
    if closed >= 2:
        return "🚨 CRITICAL: SLEEP DETECTED", "#c0392b"
    if closed == 1:
        return "⚠ ALERT: ONE EYE CLOSED", "#e67e22"
    if yawning:
        return "⚠ WARNING: YAWNING", "#f39c12"
    if any('open_eye' in r for r in results):
        return "✔ NORMAL", "#27ae60"
    return "— Status Uncertain", "#7f8c8d"


# ── GUI ───────────────────────────────────────────────────────────────────

class DrowsinessApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Driver Drowsiness Detection")
        self.root.configure(bg="#1a1a2e")
        self.root.geometry("1100x900")
        self.model     = None
        self.detectors = None
        self.dnn_net   = None
        self._image_cache = {"placeholder": None, "patches": []}
        self._build_ui()
        self._load_resources()

    def _build_ui(self):
        header = tk.Frame(self.root, bg="#16213e", pady=15)
        header.pack(fill=tk.X)
        tk.Label(header, text="🚗 Driver Drowsiness Detection",
                 font=("Segoe UI", 22, "bold"), fg="#e94560", bg="#16213e").pack()
        tk.Label(header, text="Analyze images to ensure driver safety",
                 font=("Segoe UI", 11), fg="#a8dadc", bg="#16213e").pack()

        paned = tk.PanedWindow(self.root, orient=tk.HORIZONTAL,
                               bg="#1a1a2e", sashwidth=4)
        paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        left = tk.Frame(paned, bg="#1a1a2e")
        paned.add(left, stretch="always")
        left.columnconfigure(0, weight=1)
        left.columnconfigure(1, weight=1)
        left.rowconfigure(0, weight=1)

        for col, title, attr, ph_text, fg in [
            (0, "Original Image",   "orig_label", "Drop image here",     "#a8dadc"),
            (1, "Detection Result", "res_label",  "Awaiting analysis...", "#555577"),
        ]:
            f = tk.Frame(left, bg="#16213e", bd=2, relief=tk.GROOVE)
            f.grid(row=0, column=col, padx=10, sticky="nsew")
            tk.Label(f, text=title, font=("Segoe UI", 12, "bold"),
                     fg="#a8dadc", bg="#16213e").pack(pady=5)
            lbl = tk.Label(f, bg="#0f3460", text=ph_text, fg=fg)
            lbl.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            setattr(self, attr, lbl)
        self.orig_label.bind("<Button-1>", lambda e: self._on_new_analysis())

        sidebar = tk.Frame(paned, bg="#1a1a2e", width=350)
        paned.add(sidebar, stretch="never")

        self.verdict_frame = tk.Frame(sidebar, bg="#2c3e50", pady=10)
        self.verdict_frame.pack(fill=tk.X, pady=(0, 10))
        self.verdict_label = tk.Label(self.verdict_frame, text="READY",
                                      font=("Segoe UI", 14, "bold"),
                                      fg="white", bg="#2c3e50")
        self.verdict_label.pack()

        tk.Label(sidebar, text="Detected Features",
                 font=("Segoe UI", 11, "bold"),
                 fg="#a8dadc", bg="#1a1a2e").pack(anchor=tk.W, pady=(10, 5))
        self.gallery = tk.Frame(sidebar, bg="#16213e", pady=10)
        self.gallery.pack(fill=tk.X)

        tk.Label(sidebar, text="Analysis Details",
                 font=("Segoe UI", 11, "bold"),
                 fg="#a8dadc", bg="#1a1a2e").pack(anchor=tk.W, pady=(15, 5))
        self.details = tk.Text(sidebar, height=10, bg="#16213e", fg="#e0e0e0",
                               font=("Consolas", 9), padx=10, pady=10,
                               relief=tk.FLAT)
        self.details.pack(fill=tk.BOTH, expand=True)
        self.details.config(state=tk.DISABLED)

        bottom = tk.Frame(self.root, bg="#16213e", pady=15)
        bottom.pack(fill=tk.X, side=tk.BOTTOM)

        self.status_var = tk.StringVar(value="Initializing models...")
        self.status_banner = tk.Label(bottom, textvariable=self.status_var,
                                      font=("Segoe UI", 12, "bold"),
                                      fg="white", bg="#27ae60", padx=20, pady=8)
        self.status_banner.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=20)

        self.btn_new = tk.Button(bottom, text="📂 Analyze New Photo",
                                 command=self._on_new_analysis,
                                 font=("Segoe UI", 12, "bold"),
                                 bg="#e94560", fg="white", relief=tk.FLAT,
                                 padx=25, pady=10, cursor="hand2")
        self.btn_new.pack(side=tk.RIGHT, padx=20)

        self.load_lbl = tk.Label(bottom, text="⏳ Loading...",
                                 font=("Segoe UI", 10), fg="#f39c12", bg="#16213e")
        self.load_lbl.pack(side=tk.RIGHT)

        if os.path.exists(PLACEHOLDER_PATH):
            try:
                ph = cv2_to_tk(resize_for_display(cv2.imread(PLACEHOLDER_PATH)))
                self.orig_label.config(image=ph, text="")
                self._image_cache["placeholder"] = ph
            except Exception:
                pass

    def _load_resources(self):
        self.root.after(200, self._do_load)

    def _do_load(self):
        try:
            from keras.models import load_model
            self.model     = load_model(MODEL_PATH, compile=False)
            self.detectors = load_detectors()
            self.dnn_net   = load_dnn_face_model()
            self.load_lbl.pack_forget()
            self.status_var.set("System Ready — Select a photo to begin")
        except Exception as e:
            self.status_var.set(f"Error: {e}")

    def _reset_ui(self):
        self.orig_label.config(image=self._image_cache.get("placeholder"), text="")
        self.res_label.config(image="", text="Awaiting analysis...")
        self.verdict_label.config(text="READY", bg="#2c3e50")
        self.verdict_frame.config(bg="#2c3e50")
        for child in self.gallery.winfo_children():
            child.destroy()
        self.details.config(state=tk.NORMAL)
        self.details.delete("1.0", tk.END)
        self.details.config(state=tk.DISABLED)
        self.status_var.set("Opening file...")
        self.root.update()

    def _on_new_analysis(self):
        if self.model is None:
            messagebox.showinfo("Loading", "Model is still loading, please wait...")
            return
        path = filedialog.askopenfilename(
            title="Select Photo",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.webp"),
                       ("All", "*.*")])
        if not path:
            return

        self._reset_ui()
        try:
            orig_cv = cv2.imread(path)
            orig_tk = cv2_to_tk(resize_for_display(orig_cv))
            self.orig_label.config(image=orig_tk)
            self._image_cache["current_orig"] = orig_tk
            self.status_var.set("Analyzing image...")
            self.root.update()

            annotated, results, patches = run_detection(
                path, self.model, self.detectors, self.dnn_net)

            res_tk = cv2_to_tk(resize_for_display(annotated))
            self.res_label.config(image=res_tk, text="")
            self._image_cache["current_res"] = res_tk

            self.details.config(state=tk.NORMAL)
            self.details.insert(tk.END,
                                "\n".join(results) if results else "No detections.")
            self.details.config(state=tk.DISABLED)

            self._image_cache["patches"] = []
            for p_img, p_label in patches:
                p_tk = cv2_to_tk(p_img)
                self._image_cache["patches"].append(p_tk)
                pc = tk.Frame(self.gallery, bg="#16213e")
                pc.pack(side=tk.LEFT, padx=5)
                tk.Label(pc, image=p_tk, bg="#0f3460").pack()
                tk.Label(pc, text=p_label, font=("Segoe UI", 8),
                         fg="#a8dadc", bg="#16213e").pack()

            status, color = determine_overall_status(results)
            self.verdict_label.config(text=status, bg=color)
            self.verdict_frame.config(bg=color)
            self.status_banner.config(bg=color)
            self.status_var.set("Analysis Complete")

        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.status_var.set("Ready")


if __name__ == "__main__":
    root = tk.Tk()
    DrowsinessApp(root)
    root.mainloop()