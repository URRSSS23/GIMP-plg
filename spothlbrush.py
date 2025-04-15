#!/usr/bin/env python
import os
import cv2
import numpy as np
import dlib
from gimpfu import *
import logging
import json
from datetime import datetime
from gi.repository import Gtk, GdkPixbuf

class AIHealingBrushProPlus:
    def __init__(self):
        # Configuration with defaults
        self.config = {
            "brush": {
                "size": 35,
                "hardness": 0.9,
                "texture": None,
                "texture_scale": 1.0,
                "spacing": 25
            },
            "ai": {
                "method": "ns_fsr",
                "face_detection": True,
                "patch_size": 15
            },
            "performance": {
                "use_gpu": True,
                "tile_size": 512,
                "history_steps": 10
            }
        }
        
        # Runtime data
        self.history_stack = []
        self.brush_texture_cache = None
        self.preferences_dialog = None
        
        # Initialize subsystems
        self.load_config()
        self.init_face_detection()
        self.setup_logging()
        
    def load_config(self):
        """Load saved preferences"""
        config_path = os.path.expanduser("~/.gimp_ai_heal_config.json")
        try:
            if os.path.exists(config_path):
                with open(config_path) as f:
                    self.config.update(json.load(f))
        except Exception as e:
            self.log(f"Config load failed: {str(e)}", "error")

    def save_config(self):
        """Save current preferences"""
        config_path = os.path.expanduser("~/.gimp_ai_heal_config.json")
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)

    def init_face_detection(self):
        """Initialize face detection models"""
        try:
            self.face_detector = dlib.get_frontal_face_detector()
            model_path = os.path.expanduser("~/.gimp/shape_predictor_68_face_landmarks.dat")
            if os.path.exists(model_path):
                self.shape_predictor = dlib.shape_predictor(model_path)
            else:
                self.config["ai"]["face_detection"] = False
                self.log("Face model not found - disabling face detection", "warning")
        except Exception as e:
            self.config["ai"]["face_detection"] = False
            self.log(f"Face detection init failed: {str(e)}", "error")

    def setup_logging(self):
        """Configure advanced logging"""
        self.log_file = os.path.expanduser("~/.gimp_ai_heal_pro.log")
        logging.basicConfig(
            filename=self.log_file,
            level=logging.DEBUG,
            format='%(asctime)s [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        self.log("Plugin initialized")

    def log(self, message, level="info"):
        """Unified logging with user feedback"""
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        gimp.message(f"[{timestamp}] {message}")
        
        if level == "info":
            logging.info(message)
        elif level == "warning":
            logging.warning(message)
        elif level == "error":
            logging.error(message, exc_info=True)

    def load_brush_texture(self, path):
        """Load and cache brush texture"""
        try:
            if path and os.path.exists(path):
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    self.brush_texture_cache = img
                    self.config["brush"]["texture"] = path
                    self.log(f"Loaded brush texture: {path}")
                    return True
        except Exception as e:
            self.log(f"Texture load failed: {str(e)}", "error")
        return False

    def create_textured_brush(self, width, height, x, y):
        """Generate brush with custom texture"""
        # Base circle mask
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.circle(
            mask, (x, y), 
            int(self.config["brush"]["size"] * self.config["brush"]["texture_scale"]),
            255, -1
        )
        
        # Apply texture if available
        if self.brush_texture_cache is not None:
            try:
                # Scale texture to brush size
                tex_size = int(self.config["brush"]["size"] * 2 * self.config["brush"]["texture_scale"])
                texture = cv2.resize(self.brush_texture_cache, (tex_size, tex_size))
                
                # Position texture under brush
                y1, y2 = max(0, y-tex_size//2), min(height, y+tex_size//2)
                x1, x2 = max(0, x-tex_size//2), min(width, x+tex_size//2)
                
                # Blend texture with brush
                mask[y1:y2, x1:x2] = cv2.multiply(
                    mask[y1:y2, x1:x2], 
                    texture[:y2-y1, :x2-x1] / 255.0,
                    scale=1.0
                )
            except Exception as e:
                self.log(f"Texture application failed: {str(e)}", "warning")
        
        # Apply hardness
        if self.config["brush"]["hardness"] < 0.95:
            kernel_size = int((1 - self.config["brush"]["hardness"]) * self.config["brush"]["size"] * 2) | 1
            mask = cv2.GaussianBlur(mask, (kernel_size, kernel_size), 0)
        
        return mask

    def show_preferences(self):
        """Create interactive preferences dialog"""
        dialog = Gtk.Dialog(title="AI Healing Brush Preferences")
        dialog.set_default_size(400, 500)
        
        # Notebook for tabs
        notebook = Gtk.Notebook()
        dialog.get_content_area().pack_start(notebook, True, True, 0)
        
        # Brush Settings Tab
        brush_tab = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        notebook.append_page(brush_tab, Gtk.Label(label="Brush"))
        
        # Brush Size
        size_adj = Gtk.Adjustment(
            value=self.config["brush"]["size"],
            lower=1, upper=500,
            step_increment=1, page_increment=10
        )
        size_scale = Gtk.Scale(
            orientation=Gtk.Orientation.HORIZONTAL,
            adjustment=size_adj
        )
        size_scale.set_digits(0)
        size_scale.connect("value-changed", lambda w: self.config["brush"].update({"size": w.get_value()}))
        brush_tab.pack_start(Gtk.Label(label="Brush Size:"), False, False, 0)
        brush_tab.pack_start(size_scale, False, False, 0)
        
        # Texture Selection
        texture_box = Gtk.Box(spacing=10)
        texture_entry = Gtk.Entry(text=self.config["brush"]["texture"] or "")
        texture_btn = Gtk.Button(label="Browse...")
        
        def on_texture_select(_):
            chooser = Gtk.FileChooserDialog(
                title="Select Brush Texture",
                action=Gtk.FileChooserAction.OPEN
            )
            chooser.add_buttons(
                Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL,
                Gtk.STOCK_OPEN, Gtk.ResponseType.OK
            )
            
            filter_img = Gtk.FileFilter()
            filter_img.set_name("Image files")
            filter_img.add_mime_type("image/png")
            filter_img.add_mime_type("image/jpeg")
            chooser.add_filter(filter_img)
            
            if chooser.run() == Gtk.ResponseType.OK:
                path = chooser.get_filename()
                if self.load_brush_texture(path):
                    texture_entry.set_text(path)
            chooser.destroy()
        
        texture_btn.connect("clicked", on_texture_select)
        texture_box.pack_start(Gtk.Label(label="Texture:"), False, False, 0)
        texture_box.pack_start(texture_entry, True, True, 0)
        texture_box.pack_start(texture_btn, False, False, 0)
        brush_tab.pack_start(texture_box, False, False, 0)
        
        # AI Settings Tab
        ai_tab = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        notebook.append_page(ai_tab, Gtk.Label(label="AI"))
        
        # Method Selection
        method_combo = Gtk.ComboBoxText()
        for method in ["ns_fsr", "telea", "navier_stokes"]:
            method_combo.append_text(method)
        method_combo.set_active(list(self.config["ai"]["methods"].keys()).index(self.config["ai"]["method"]))
        method_combo.connect("changed", lambda w: self.config["ai"].update({"method": w.get_active_text()}))
        ai_tab.pack_start(Gtk.Label(label="Inpainting Method:"), False, False, 0)
        ai_tab.pack_start(method_combo, False, False, 0)
        
        # Face Detection Toggle
        face_toggle = Gtk.Switch(active=self.config["ai"]["face_detection"])
        face_toggle.connect("state-set", lambda w, s: self.config["ai"].update({"face_detection": s}))
        ai_tab.pack_start(Gtk.Label(label="Face Detection:"), False, False, 0)
        ai_tab.pack_start(face_toggle, False, False, 0)
        
        # Save/Cancel Buttons
        dialog.add_button("Cancel", Gtk.ResponseType.CANCEL)
        dialog.add_button("Save", Gtk.ResponseType.OK)
        
        if dialog.run() == Gtk.ResponseType.OK:
            self.save_config()
            self.log("Preferences saved")
        
        dialog.destroy()

# Register tools and menu entries
register(
    "python_fu_ai_heal_pro_plus",
    "AI Healing Brush Pro+",
    "Advanced healing with textures and preferences",
    "AI Developer", "AI Developer", "2024",
    "<Image>/Tools/Paint Tools/AI Healing Pro+",
    "RGB*", [
        (PF_IMAGE, "image", "Input image", None),
        (PF_DRAWABLE, "drawable", "Layer to heal", None),
        (PF_INT, "x", "Brush X", 0),
        (PF_INT, "y", "Brush Y", 0),
        (PF_FLOAT, "pressure", "Pressure", 1.0)
    ], [],
    AIHealingBrushProPlus().apply_healing,
    menu="<Image>/Filters/Enhance"
)

register(
    "python_fu_ai_heal_preferences",
    "AI Healing Preferences...",
    "Configure plugin settings",
    "AI Developer", "AI Developer", "2024",
    "<Image>/Edit/Preferences/AI Healing Brush",
    "", [], [],
    AIHealingBrushProPlus().show_preferences
)

main()
