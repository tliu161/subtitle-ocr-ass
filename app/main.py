# app/main.py
import os
import json
import threading
import traceback
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk

from .video_utils import get_video_info, extract_preview_frame
from .ocr_engine import process_video_to_ass


# -------------------------
# Config persistence
# -------------------------
def get_config_dir():
    base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
    d = os.path.join(base, "SubtitleOCR_ASS")
    os.makedirs(d, exist_ok=True)
    return d


CONFIG_PATH = os.path.join(get_config_dir(), "config.json")

DEFAULT_CONFIG = {
    "last_roi": None,  # [x, y, w, h]
    "sample_fps": 3.0,
    "change_threshold": 0.18,
    "hold_gap": 0.25,
    "fill_gaps": 2.0,
    "preview_time_sec": 30,
}


def load_config():
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        out = dict(DEFAULT_CONFIG)
        out.update(cfg or {})
        return out
    except Exception:
        return dict(DEFAULT_CONFIG)


def save_config(cfg):
    try:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


class ROISelectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("SubtitleOCR_ASS — Select ROI")

        self.cfg = load_config()

        self.video_path = None
        self.preview_png = "_preview_frame.png"
        self.duration = None

        # display scaling
        self.scale = 1.0
        self.display_max_w = 1000
        self.display_max_h = 560

        # ROI drawing
        self.start_x = None
        self.start_y = None
        self.rect_id = None
        self.roi_original = None  # (x,y,w,h)

        # Debounce for slider preview updates
        self._preview_job = None
        self._last_preview_t = None

        # Load last ROI from config
        if isinstance(self.cfg.get("last_roi"), list) and len(self.cfg["last_roi"]) == 4:
            self.roi_original = tuple(int(v) for v in self.cfg["last_roi"])

        # -------------------------
        # Top controls
        # -------------------------
        top = tk.Frame(root)
        top.pack(fill="x", padx=10, pady=8)

        self.btn_select = tk.Button(top, text="Select Video", command=self.select_video)
        self.btn_select.pack(side="left")

        self.btn_preview = tk.Button(top, text="Load Preview", command=self.load_preview)
        self.btn_preview.pack(side="left", padx=8)

        self.btn_run = tk.Button(top, text="Run OCR → ASS", command=self.run_ocr)
        self.btn_run.pack(side="left", padx=8)

        # -------------------------
        # Time slider + time entry
        # -------------------------
        time_row = tk.Frame(root)
        time_row.pack(fill="x", padx=10, pady=(0, 6))

        tk.Label(time_row, text="Preview time:").pack(side="left")

        self.preview_t_var = tk.DoubleVar(value=float(self.cfg.get("preview_time_sec", 30)))
        self.preview_label = tk.Label(time_row, text="(no video yet)")
        self.preview_label.pack(side="left", padx=(8, 10))

        self.time_slider = tk.Scale(
            time_row,
            from_=0,
            to=100,  # placeholder; set after video selection
            orient="horizontal",
            variable=self.preview_t_var,
            showvalue=False,
            length=520,
            resolution=0.5,
            command=self.on_slider_change,
        )
        self.time_slider.pack(side="left", fill="x", expand=True)

        tk.Label(time_row, text="sec").pack(side="left", padx=(10, 2))
        self.time_entry_var = tk.StringVar(value=str(int(self.preview_t_var.get())))
        tk.Entry(time_row, textvariable=self.time_entry_var, width=6).pack(side="left")
        tk.Button(time_row, text="Go", command=self.on_time_go).pack(side="left", padx=6)

        # -------------------------
        # Params (editable)
        # -------------------------
        params = tk.LabelFrame(root, text="Parameters (editable)")
        params.pack(fill="x", padx=10, pady=(0, 6))

        self.fps_var = tk.StringVar(value=str(self.cfg.get("sample_fps", 3.0)))
        self.change_var = tk.StringVar(value=str(self.cfg.get("change_threshold", 0.18)))
        self.hold_var = tk.StringVar(value=str(self.cfg.get("hold_gap", 0.25)))
        self.fill_var = tk.StringVar(value=str(self.cfg.get("fill_gaps", 2.0)))

        def add_field(label, var):
            row = tk.Frame(params)
            row.pack(side="left", padx=10, pady=6)
            tk.Label(row, text=label).pack(anchor="w")
            tk.Entry(row, textvariable=var, width=10).pack()

        add_field("sample_fps", self.fps_var)
        add_field("change_threshold", self.change_var)
        add_field("hold_gap", self.hold_var)
        add_field("fill_gaps", self.fill_var)

        # -------------------------
        # Canvas (preview + ROI selection)
        # -------------------------
        self.canvas = tk.Canvas(root, bg="#222")
        self.canvas.pack(fill="both", expand=True, padx=10, pady=10)

        # Bind mouse for ROI
        self.canvas.bind("<ButtonPress-1>", self.on_down)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_up)

        self.tk_img = None

        # -------------------------
        # Progress + Status
        # -------------------------
        bottom = tk.Frame(root)
        bottom.pack(fill="x", padx=10, pady=(0, 6))

        self.progress_text = tk.Label(bottom, text="Idle")
        self.progress_text.pack(side="left")

        self.progress_var = tk.DoubleVar(value=0.0)
        self.progressbar = ttk.Progressbar(bottom, variable=self.progress_var, maximum=100.0)
        self.progressbar.pack(side="left", fill="x", expand=True, padx=10)

        self.status = tk.Label(root, text=self._status_text())
        self.status.pack(fill="x", padx=10, pady=(0, 10))

        # Update preview label initially
        self._refresh_preview_label()

    # -------------------------
    # UI helpers
    # -------------------------
    def _status_text(self):
        if self.roi_original:
            return f"Last ROI loaded: {self.roi_original}  (Load preview & drag to override)"
        return "Select a video → Load preview → Drag to select subtitle ROI"

    def _refresh_preview_label(self):
        t = float(self.preview_t_var.get())
        if self.duration is None:
            self.preview_label.config(text=f"{t:.1f}s")
        else:
            self.preview_label.config(text=f"{t:.1f}s / {self.duration:.1f}s")
        self.time_entry_var.set(str(int(round(t))))

    def _schedule_preview_update(self, delay_ms=350):
        if self._preview_job is not None:
            try:
                self.root.after_cancel(self._preview_job)
            except Exception:
                pass
            self._preview_job = None
        self._preview_job = self.root.after(delay_ms, self.load_preview)

    def _set_controls_enabled(self, enabled: bool):
        state = "normal" if enabled else "disabled"
        # keep it simple: disable the main actions while running
        try:
            self.btn_select.configure(state=state)
            self.btn_preview.configure(state=state)
            self.btn_run.configure(state=state)
        except Exception:
            pass

    # -------------------------
    # Actions
    # -------------------------
    def select_video(self):
        path = filedialog.askopenfilename(
            title="Select a video",
            filetypes=[("Video files", "*.mp4 *.mov *.mkv *.avi"), ("All files", "*.*")],
        )
        if not path:
            return

        try:
            _, _, dur = get_video_info(path)
        except Exception as e:
            messagebox.showerror("ffprobe error", str(e))
            return

        self.video_path = path
        self.duration = dur

        self.time_slider.config(from_=0, to=max(0.0, dur))
        saved_t = float(self.cfg.get("preview_time_sec", 30))
        saved_t = min(max(0.0, saved_t), max(0.0, dur - 0.1))
        self.preview_t_var.set(saved_t)
        self._refresh_preview_label()

        self.status.config(text=f"Selected: {os.path.basename(path)}  ({dur:.1f}s)  |  ROI: {self.roi_original}")

        # Auto load preview once
        self.load_preview()

    def on_slider_change(self, _value):
        self._refresh_preview_label()
        if self.video_path:
            self._schedule_preview_update(350)

    def on_time_go(self):
        if not self.video_path or self.duration is None:
            return
        try:
            t = float(self.time_entry_var.get().strip())
        except Exception:
            return
        t = min(max(0.0, t), max(0.0, self.duration - 0.1))
        self.preview_t_var.set(t)
        self._refresh_preview_label()
        self.load_preview()

    def load_preview(self):
        if not self.video_path:
            messagebox.showwarning("No video", "Please select a video first.")
            return

        t = float(self.preview_t_var.get())
        if self._last_preview_t is not None and abs(self._last_preview_t - t) < 0.01 and self.tk_img is not None:
            return
        self._last_preview_t = t

        try:
            extract_preview_frame(self.video_path, self.preview_png, t)
        except Exception as e:
            messagebox.showerror("ffmpeg error", str(e))
            return

        # persist preview time
        self.cfg["preview_time_sec"] = float(t)
        save_config(self.cfg)

        img = Image.open(self.preview_png).convert("RGB")
        ow, oh = img.size

        self.scale = min(self.display_max_w / ow, self.display_max_h / oh, 1.0)
        dw, dh = int(ow * self.scale), int(oh * self.scale)
        img_disp = img.resize((dw, dh), Image.LANCZOS)

        self.tk_img = ImageTk.PhotoImage(img_disp)
        self.canvas.delete("all")
        self.canvas.config(width=dw, height=dh)
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_img)

        # draw existing ROI as reference
        self.rect_id = None
        if self.roi_original:
            x, y, w, h = self.roi_original
            dx1 = int(x * self.scale)
            dy1 = int(y * self.scale)
            dx2 = int((x + w) * self.scale)
            dy2 = int((y + h) * self.scale)
            self.rect_id = self.canvas.create_rectangle(dx1, dy1, dx2, dy2, outline="red", width=2)

        self.status.config(text="Preview updated. Drag to set a NEW ROI, or keep saved ROI and press Run.")

    # -------------------------
    # ROI drawing
    # -------------------------
    def on_down(self, event):
        if not self.tk_img:
            return
        self.start_x, self.start_y = event.x, event.y
        if self.rect_id:
            self.canvas.delete(self.rect_id)
        self.rect_id = self.canvas.create_rectangle(
            self.start_x, self.start_y, self.start_x, self.start_y, outline="red", width=2
        )

    def on_drag(self, event):
        if not self.tk_img or self.rect_id is None:
            return
        self.canvas.coords(self.rect_id, self.start_x, self.start_y, event.x, event.y)

    def on_up(self, event):
        if not self.tk_img or self.rect_id is None:
            return
        x1, y1, x2, y2 = self.canvas.coords(self.rect_id)
        x1, x2 = sorted([int(x1), int(x2)])
        y1, y2 = sorted([int(y1), int(y2)])

        cw = int(self.canvas.cget("width"))
        ch = int(self.canvas.cget("height"))
        x1, x2 = max(0, min(cw, x1)), max(0, min(cw, x2))
        y1, y2 = max(0, min(ch, y1)), max(0, min(ch, y2))

        if (x2 - x1) < 5 or (y2 - y1) < 5:
            self.status.config(text="ROI too small. Drag a bigger rectangle.")
            return

        ox1, oy1 = int(x1 / self.scale), int(y1 / self.scale)
        ox2, oy2 = int(x2 / self.scale), int(y2 / self.scale)
        x, y = ox1, oy1
        w, h = max(1, ox2 - ox1), max(1, oy2 - oy1)

        self.roi_original = (x, y, w, h)

        # persist ROI
        self.cfg["last_roi"] = [x, y, w, h]
        save_config(self.cfg)

        self.status.config(text=f"ROI saved: {self.roi_original}")

    # -------------------------
    # Run OCR (background thread + progress)
    # -------------------------
    def run_ocr(self):
        if not self.video_path:
            messagebox.showwarning("No video", "Please select a video first.")
            return
        if not self.roi_original:
            messagebox.showwarning("No ROI", "No saved ROI. Load preview and drag to select subtitle region.")
            return

        def parse_float(var, fallback):
            try:
                return float(var.get().strip())
            except Exception:
                return fallback

        fps = parse_float(self.fps_var, 3.0)
        change = parse_float(self.change_var, 0.18)
        hold = parse_float(self.hold_var, 0.25)
        fill = parse_float(self.fill_var, 2.0)

        # persist params
        self.cfg["sample_fps"] = fps
        self.cfg["change_threshold"] = change
        self.cfg["hold_gap"] = hold
        self.cfg["fill_gaps"] = fill
        save_config(self.cfg)

        # UI: reset progress
        self.progress_var.set(0.0)
        self.progress_text.config(text="Starting…")
        self.status.config(text="Running OCR… please keep this window open.")
        self._set_controls_enabled(False)

        # progress callback from worker thread
        def progress_cb(_kind: str, frac: float, msg: str):
            def _update():
                self.progress_var.set(max(0.0, min(100.0, frac * 100.0)))
                self.progress_text.config(text=msg)
            self.root.after(0, _update)

        def worker():
            try:
                out_ass = process_video_to_ass(
                    video_path=self.video_path,
                    roi_xywh=self.roi_original,
                    sample_fps=fps,
                    change_threshold=change,
                    hold_gap=hold,
                    fill_gaps=fill,
                    progress_cb=progress_cb,
                )
                self.root.after(0, lambda: self._on_ocr_success(out_ass))
            except Exception as e:
                tb = traceback.format_exc()
                self.root.after(0, lambda: self._on_ocr_error(str(e), tb))

        threading.Thread(target=worker, daemon=True).start()

    def _on_ocr_success(self, out_ass: str):
        self._set_controls_enabled(True)
        self.progress_var.set(100.0)
        self.progress_text.config(text="Done ✅")
        self.status.config(text="✅ Finished.")
        messagebox.showinfo("Done", f"✅ ASS generated:\n{out_ass}")

    def _on_ocr_error(self, msg: str, tb: str):
        self._set_controls_enabled(True)
        self.progress_text.config(text="Error ❌")
        self.status.config(text="❌ Error occurred.")
        messagebox.showerror("OCR error", msg)
        print(tb)


def main():
    root = tk.Tk()
    app = ROISelectorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
