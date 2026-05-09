"""
Run this once to set your box coordinates.
Saves to config.json which color.py will use to get references for the colors, buttons etc.

Controls:
  1 -> draw BOX1 (the drawing canvas)
  2 -> draw BOX2 (the "TO COPY" thumbnail)
  3 -> draw BUTTONS (the color palette grid)
  4 -> draw EXCLUDE (zone to never click, e.g. STOP button) — optional
  S -> save config.json and quit
  R -> redo current box
  F5 -> take a fresh screenshot
  ESC -> quit without saving
"""

import json
import mss
import cv2
import os
import keyboard
import numpy as np


CONFIG_FILE = "config.json"
SCREENSHOT_FILE = "calibration_screenshot.png"

boxes = {
    "box1": None,
    "box2": None,
    "buttons": None,
    "exclude": None,
}

current_mode = None
drawing = False
start_pt = None
img_display = None
img_original = None

BOX_COLORS = {
    "box1": (255, 80, 0),
    "box2": (0, 220, 0),
    "buttons": (0, 80, 255),
    "exclude": (0, 0, 200),
}

BOX_LABELS = {
    "box1": "1: BOX1 : Drawing canvas",
    "box2": "2: BOX2 : Reference thumbnail",
    "buttons": "3: BUTTONS : Color palette",
    "exclude": "4: EXCLUDE : Never click here",
}

def take_screenshot():
    with mss.MSS() as sct:
        sct.shot(output=SCREENSHOT_FILE)

def load_screenshot():
    global img_original, img_display
    img_original = cv2.imread(SCREENSHOT_FILE)

    if img_original is None:
        raise FileNotFoundError(f"No screenshot at {SCREENSHOT_FILE}")

    img_display = img_original.copy()

def redraw():
    global img_display
    img_display = img_original.copy()

    for key, box in boxes.items():
        if box is None:
            continue

        x, y, w, h = box
        color = BOX_COLORS[key]

        if key == "exclude":
            overlay = img_display.copy()
            cv2.rectangle(overlay, (x, y), (x+w, y+h), (0, 0, 180), -1)
            cv2.addWeighted(overlay, 0.35, img_display, 0.65, 0, img_display)

            for i in range(0, w+h, 12):
                x1 = x + min(i, w);  y1 = y + max(0, i-w)
                x2 = x + max(0, i-h); y2 = y + min(i, h)
                cv2.line(img_display, (x1,y1), (x2,y2), color, 1)

        cv2.rectangle(img_display, (x,y), (x+w,y+h), color, 2)
        cv2.putText(img_display, BOX_LABELS[key], (x+4, y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 3)
        cv2.putText(img_display, BOX_LABELS[key], (x+4, y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)

    for i, line in enumerate(["1=BOX1  2=BOX2  3=BUTTONS  4=EXCLUDE(optional)", "S=save  R=redo  F5=screenshot  ESC=quit"]):
        y = 22 + i*22
        cv2.putText(img_display, line, (8,y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 3)
        cv2.putText(img_display, line, (8,y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)

    done = [k for k,v in boxes.items() if v is not None]

    status = f"Done: {', '.join(done) if done else 'none'}  |  Active: {current_mode or 'none'}"

    cv2.putText(img_display, status, (8, img_display.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 3)
    cv2.putText(img_display, status, (8, img_display.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)

    cv2.imshow("Calibration", img_display)

def mouse_cb(event, x, y, flags, param):
    global drawing, start_pt, img_display

    if current_mode is None:
        return

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_pt = (x, y)

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        tmp = img_display.copy()
        cv2.rectangle(tmp, start_pt, (x, y), BOX_COLORS[current_mode], 2)
        w = abs(x - start_pt[0])
        h = abs(y - start_pt[1])
        cv2.putText(tmp, f"{w}x{h}",
                    (min(start_pt[0], x), min(start_pt[1], y) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, BOX_COLORS[current_mode], 2)
        cv2.imshow("Calibration", tmp)

    elif event == cv2.EVENT_LBUTTONUP and drawing:
        drawing = False
        x0, y0 = min(start_pt[0], x), min(start_pt[1], y)
        x1, y1 = max(start_pt[0], x), max(start_pt[1], y)
        w, h = x1 - x0, y1 - y0
        if w > 5 and h > 5:
            boxes[current_mode] = (x0, y0, w, h)
            print(f"-[{current_mode}] x={x0} y={y0} w={w} h={h}")
        redraw()

def save_config():
    cfg = {}
    if boxes["box1"]:
        x,y,w,h = boxes["box1"]
        cfg.update({"box1_x":x,"box1_y":y,"box1_w":w,"box1_h":h})

    if boxes["box2"]:
        x,y,w,h = boxes["box2"]
        cfg.update({"box2_x":x,"box2_y":y,"box2_w":w,"box2_h":h})

    if boxes["buttons"]:
        x,y,w,h = boxes["buttons"]
        cfg.update({"button_box_x":x,"button_box_y":y,"button_box_w":w,"button_box_h":h})

    if boxes["exclude"]:
        x,y,w,h = boxes["exclude"]
        cfg.update({"exclude_x":x,"exclude_y":y,"exclude_w":w,"exclude_h":h})
        print(f"-Exclusion zone saved: x={x} y={y} w={w} h={h}")
    else:
        print("-No exclusion zone (optional)")

    with open(CONFIG_FILE, "w") as f:
        json.dump(cfg, f, indent=2)

def main():
    global current_mode

    if not os.path.exists(SCREENSHOT_FILE):
        print("Taking screenshot...")
        take_screenshot()

    load_screenshot()

    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE) as f:
                cfg = json.load(f)

            if all(k in cfg for k in ["box1_x","box1_y","box1_w","box1_h"]):
                boxes["box1"] = (cfg["box1_x"],cfg["box1_y"],cfg["box1_w"],cfg["box1_h"])

            if all(k in cfg for k in ["box2_x","box2_y","box2_w","box2_h"]):
                boxes["box2"] = (cfg["box2_x"],cfg["box2_y"],cfg["box2_w"],cfg["box2_h"])

            if all(k in cfg for k in ["button_box_x","button_box_y","button_box_w","button_box_h"]):
                boxes["buttons"] = (cfg["button_box_x"],cfg["button_box_y"], cfg["button_box_w"],cfg["button_box_h"])

            if all(k in cfg for k in ["exclude_x","exclude_y","exclude_w","exclude_h"]):
                boxes["exclude"] = (cfg["exclude_x"],cfg["exclude_y"], cfg["exclude_w"],cfg["exclude_h"])

            print("Loaded existing config.json")

        except Exception:
            pass

    cv2.namedWindow("Calibration", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Calibration", 1280, 720)
    cv2.setMouseCallback("Calibration", mouse_cb)
    redraw()

    print("\nPress 1/2/3 to draw boxes, 4 for exclusion zone (optional).")

    # Zone selection loop (not very efficient)
    while True:
        key = cv2.waitKey(20) & 0xFF

        if   key == ord('1'):
            current_mode = "box1"
            redraw()

        elif key == ord('2'):
            current_mode = "box2";
            redraw()

        elif key == ord('3'):
            current_mode = "buttons";
            redraw()

        elif key == ord('4'):
            current_mode = "exclude";
            redraw()

        elif key in (ord('r'), ord('R')):
            if current_mode:
                boxes[current_mode] = None
                print(f"Reset {current_mode}")
                redraw()
                
        elif key == ord('f'):
            take_screenshot()
            load_screenshot()
            redraw()

        elif key in (ord('s'), ord('S')):
            save_config(); break

        elif key == 27:
            print("Quit without saving."); break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    keyboard.add_hotkey("F7", main)
    keyboard.wait("esc") # Closes the app / keep alive