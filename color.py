import os
import cv2
import mss
import time
import json
import keyboard
import ctypes
import ctypes.wintypes
import traceback
import numpy as np
from collections import Counter, defaultdict


DEBUG_DISPLAY = False
ENABLE_MOUSE = True

# Config
CLICK_DELAY = 0.01
cols, rows = 3, 5

def load_config():
    cfg_path = os.path.join(os.path.dirname(__file__), "config.json")

    if not os.path.exists(cfg_path):
        raise FileNotFoundError("config.json not found — run calibrate.py first.")

    with open(cfg_path) as f:
        cfg = json.load(f)

    required = ["box1_x","box1_y","box1_w","box1_h",
                "box2_x","box2_y","box2_w","box2_h",
                "button_box_x","button_box_y",
                "button_box_w","button_box_h"]

    missing = [k for k in required if k not in cfg]

    if missing:
        raise ValueError(f"config.json missing keys: {missing}: run calibrate.py")

    print(f"[config] box1: x={cfg['box1_x']} y={cfg['box1_y']} w={cfg['box1_w']} h={cfg['box1_h']}")
    print(f"[config] box2: x={cfg['box2_x']} y={cfg['box2_y']} w={cfg['box2_w']} h={cfg['box2_h']}")
    print(f"[config] buttons: x={cfg['button_box_x']} y={cfg['button_box_y']} w={cfg['button_box_w']} h={cfg['button_box_h']}")

    if all(k in cfg for k in ["exclude_x","exclude_y","exclude_w","exclude_h"]):
        print(f"[config] exclude: x={cfg['exclude_x']} y={cfg['exclude_y']} w={cfg['exclude_w']} h={cfg['exclude_h']}")
    else:
        print("[config] exclude: none")

    return cfg


# Utils / Helpers
def pixel_is_colored(bgr):
    pixel = np.uint8([[list(bgr[:3])]])
    hsv = cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV)

    return int(hsv[0][0][1]) > 30 and int(hsv[0][0][2]) > 15

def snap_to_palette(rgb, palette):
    return min(palette, key=lambda c: sum((a-b)**2 for a,b in zip(rgb,c)))

def get_drawing_bbox(crop):
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask_black = cv2.inRange(hsv, (0,0,0), (180,100,80))
    ys, xs = np.where(mask_black > 0)

    if len(xs) == 0:
        return 0, 0, crop.shape[1], crop.shape[0]

    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())

def get_click_point(zone_mask):
    mask_u8 = zone_mask.astype(np.uint8) * 255
    kernel = np.ones((5,5), np.uint8)
    eroded = mask_u8.copy()

    for _ in range(20):
        candidate = cv2.erode(eroded, kernel)

        if np.any(candidate > 0):
            eroded = candidate
        else:
            break

    ys, xs = np.where(eroded > 0)

    return int(np.mean(xs)), int(np.mean(ys))

def point_in_rect(px, py, rx, ry, rw, rh):
    return rx <= px <= rx+rw and ry <= py <= ry+rh

def label_bgr(img, text, pos, scale=0.45, color=(0,0,0), thickness=1):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (255,255,255), thickness+2)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)


# Analyse
def analyse(screenshot_path, cfg):
    box1_x = cfg["box1_x"]; box1_y = cfg["box1_y"]
    box1_w = cfg["box1_w"]; box1_h = cfg["box1_h"]
    box2_x = cfg["box2_x"]; box2_y = cfg["box2_y"]
    box2_w = cfg["box2_w"]; box2_h = cfg["box2_h"]
    button_box_x = cfg["button_box_x"]; button_box_y = cfg["button_box_y"]
    button_box_w = cfg["button_box_w"]; button_box_h = cfg["button_box_h"]

    # Exclusion zone in screen coordinates (optional)
    exclude = None
    if all(k in cfg for k in ["exclude_x","exclude_y","exclude_w","exclude_h"]):
        exclude = (cfg["exclude_x"], cfg["exclude_y"], cfg["exclude_w"], cfg["exclude_h"])

    img = cv2.imread(screenshot_path)

    if img is None:
        raise FileNotFoundError(f"Cannot read: {screenshot_path}")

    # Palette / buttons
    palette, button_clicks = [], {}
    bws, bhs = button_box_w / cols, button_box_h / rows
    btn_debug = img[button_box_y:button_box_y+button_box_h, button_box_x:button_box_x+button_box_w].copy()

    for row in range(rows):
        for col in range(cols):
            cx = int(button_box_x + col*bws + bws/2)
            cy = int(button_box_y + row*bhs + bhs/2)

            bgr = img[cy, cx]

            lx = int(col*bws + bws/2)
            ly = int(row*bhs + bhs/2)

            hsv_px = cv2.cvtColor(np.uint8([[list(bgr[:3])]]), cv2.COLOR_BGR2HSV)[0][0]

            s, v = int(hsv_px[1]), int(hsv_px[2])

            if pixel_is_colored(bgr):
                rgb = tuple(int(c) for c in bgr[::-1])
                palette.append(rgb)
                button_clicks[rgb] = (cx, cy)
                cv2.circle(btn_debug, (lx,ly), 8, (0,255,0), 2)
                label_bgr(btn_debug, f"OK s={s} v={v}", (lx-25,ly-12), 0.3, (0,200,0))
            else:
                cv2.circle(btn_debug, (lx,ly), 8, (0,0,255), 2)
                label_bgr(btn_debug, f"SKIP s={s} v={v}", (lx-30,ly-12), 0.3, (0,0,200))

    print(f"[1] Palette: {len(palette)} colors")

    if not palette:
        raise ValueError(f"Buttons region not valid, run calibrate again, make sure all buttons are in the selected zone")

    # Crops
    box1 = img[box1_y:box1_y+box1_h, box1_x:box1_x+box1_w]
    box2 = img[box2_y:box2_y+box2_h, box2_x:box2_x+box2_w]
    h1, w1 = box1.shape[:2]
    h2, w2 = box2.shape[:2]

    # Content bboxes
    b1x0,b1y0,b1x1,b1y1 = get_drawing_bbox(box1)
    b2x0,b2y0,b2x1,b2y1 = get_drawing_bbox(box2)
    b1cw,b1ch = b1x1-b1x0, b1y1-b1y0
    b2cw,b2ch = b2x1-b2x0, b2y1-b2y0

    print(f"[2] box1 content: ({b1x0},{b1y0})->({b1x1},{b1y1})")
    print(f"[2] box2 content: ({b2x0},{b2y0})->({b2x1},{b2y1})")

    # Per pixel snapped box2 (color matching from the reference)
    hsv2 = cv2.cvtColor(box2, cv2.COLOR_BGR2HSV)
    colored_mask2 = (hsv2[:,:,1] > 30) & (hsv2[:,:,2] > 15)
    snapped_box2 = np.zeros_like(box2)
    
    for y, x in zip(*np.where(colored_mask2)):
        rgb = tuple(int(c) for c in box2[y, x, ::-1])
        snapped_box2[y, x] = snap_to_palette(rgb, palette)[::-1]

    print(f"[3] box2 colored pixels: {int(np.sum(colored_mask2))} / {h2*w2}")

    # Zones to color
    hsv1 = cv2.cvtColor(box1, cv2.COLOR_BGR2HSV)
    mask_black1 = cv2.inRange(hsv1, (0,0,0), (180,100,80))
    mask_black1 = cv2.morphologyEx(mask_black1, cv2.MORPH_CLOSE, np.ones((3,3),np.uint8))
    num1, labels1, stats1, _ = cv2.connectedComponentsWithStats(cv2.bitwise_not(mask_black1))
    zones = {l: (labels1==l) for l in range(1,num1) if stats1[l][4] >= 80}

    print(f"[4] Zones detected: {len(zones)}")

    # Coords map
    gy, gx = np.mgrid[0:h1, 0:w1]
    mx = np.clip(((gx-b1x0)/b1cw*b2cw+b2x0).astype(int), 0, w2-1)
    my = np.clip(((gy-b1y0)/b1ch*b2ch+b2y0).astype(int), 0, h2-1)

    mapped_colors = snapped_box2[my, mx]
    mapped_is_colored = colored_mask2[my, mx]

    # Build actions
    actions = []
    
    reconstructed = np.ones((h1,w1,3), dtype=np.uint8) * 240
    
    fallback_count = 0
    excluded_count = 0

    zone_debug = box1.copy()

    for zone_label, zone_mask in zones.items():
        lx, ly = get_click_point(zone_mask)
        screen_x = box1_x + lx
        screen_y = box1_y + ly

        # Exclusion check
        if exclude and point_in_rect(screen_x, screen_y, *exclude):
            excluded_count += 1
            
            # Painted dark red in debug so its obvious
            reconstructed[zone_mask] = (30, 30, 150)
            label_bgr(reconstructed, "EXCL", (lx-8, ly+4), 0.3, (0,0,200))
            
            cv2.circle(zone_debug, (lx,ly), 5, (0,0,200), -1)
            contours, _ = cv2.findContours(zone_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(zone_debug, contours, -1, (0,0,200), 1)
            
            continue

        zone_colored = zone_mask & mapped_is_colored
        
        if np.any(zone_colored):
            votes = [tuple(p) for p in mapped_colors[zone_colored]]
            
            dominant_bgr = Counter(votes).most_common(1)[0][0]
            dominant_rgb = tuple(int(c) for c in dominant_bgr[::-1])
            
            source = "map"
        else:
            fallback_count += 1

            px_list = [p for p in box1[zone_mask] if pixel_is_colored(p)]
            med = tuple(int(np.median([p[i] for p in px_list])) for i in range(3)) if px_list else (0,0,0)
            
            dominant_rgb = snap_to_palette(med[::-1], palette)
            source = "fallback"

        actions.append((dominant_rgb, button_clicks[dominant_rgb], (screen_x, screen_y)))

        bgr = tuple(int(c) for c in dominant_rgb[::-1])
        reconstructed[zone_mask] = bgr
        color_idx = palette.index(dominant_rgb) if dominant_rgb in palette else -1
        label_bgr(reconstructed, f"#{color_idx}{'FB' if source=='fallback' else ''}", (lx-8, ly+4), 0.3)

        contours, _ = cv2.findContours(zone_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cv2.drawContours(zone_debug, contours, -1, (0,200,0), 1)
        cv2.circle(zone_debug, (lx,ly), 3, (0,0,255), -1)

    print(f"[5] Actions: {len(actions)}  fallbacks: {fallback_count}  excluded: {excluded_count}")

    # Draw exclusion zone overlay on zone_debug
    if exclude:
        ex, ey, ew, eh = exclude
        # Convert screen coords to box1-local
        lx0, ly0 = ex - box1_x, ey - box1_y
        cv2.rectangle(zone_debug, (lx0,ly0), (lx0+ew,ly0+eh), (0,0,200), 2)
        label_bgr(zone_debug, "EXCLUDE", (lx0+2, ly0+16), 0.5, (0,0,200))

    gray = cv2.cvtColor(reconstructed, cv2.COLOR_BGR2GRAY)
    reconstructed[cv2.Canny(gray, 50, 150) > 0] = (0,0,0)

    # Debug panel
    SCALE = 0.55

    def rs(im):
        return cv2.resize(im, None, fx=SCALE, fy=SCALE)

    regions_debug = img.copy()
    cv2.rectangle(regions_debug, (box1_x,box1_y), (box1_x+box1_w,box1_y+box1_h), (255,80,0), 3)
    cv2.rectangle(regions_debug, (box2_x,box2_y), (box2_x+box2_w,box2_y+box2_h), (0,255,0), 3)
    cv2.rectangle(regions_debug, (button_box_x,button_box_y), (button_box_x+button_box_w,button_box_y+button_box_h), (0,80,255), 3)

    if exclude:
        ex, ey, ew, eh = exclude
        cv2.rectangle(regions_debug, (ex,ey), (ex+ew,ey+eh), (0,0,200), 3)
        label_bgr(regions_debug, "EXCLUDE", (ex+4,ey+20), 0.7, (0,0,200), 2)

    label_bgr(regions_debug, "BOX1", (box1_x+5, box1_y+25), 0.8, (255,80,0), 2)
    label_bgr(regions_debug, "BOX2", (box2_x+5, box2_y+25), 0.8, (0,200,0),  2)
    label_bgr(regions_debug, "BUTTONS", (button_box_x+5, button_box_y+25), 0.8, (0,80,255), 2)

    row1 = rs(regions_debug)

    box2_big = cv2.resize(snapped_box2, (w1,h1), interpolation=cv2.INTER_NEAREST)

    gap = np.ones((h1,6,3), dtype=np.uint8)*80
    row2 = rs(np.hstack([box2_big, gap, zone_debug, gap, reconstructed]))
    
    label_bgr(row2, "BOX2 snapped", (5, 18), 0.5, (0,200,0))
    label_bgr(row2, "ZONES (red=excl)", (int(w1*SCALE)+10, 18), 0.5, (0,150,200))
    label_bgr(row2, "RECONSTRUCTED", (int(w1*SCALE*2)+15, 18), 0.5, (200,100,0))

    swatch_w = 60

    pal_strip = np.zeros((50, max(1,swatch_w*len(palette)), 3), dtype=np.uint8)

    for i, rgb in enumerate(palette):
        pal_strip[:, i*swatch_w:(i+1)*swatch_w] = rgb[::-1]
        label_bgr(pal_strip, f"#{i}", (i*swatch_w+2, 30), 0.4)

    btn_big = cv2.resize(btn_debug, None, fx=1.5, fy=1.5)
    rw = row1.shape[1]

    panel = np.vstack([
        row1,
        np.ones((4,rw,3),dtype=np.uint8)*60,
        cv2.resize(row2, (rw, row2.shape[0])),
        np.ones((4,rw,3),dtype=np.uint8)*60,
        cv2.resize(pal_strip, (rw,50)),
        np.ones((4,rw,3),dtype=np.uint8)*60,
        cv2.resize(btn_big, (rw, int(btn_big.shape[0]*rw/btn_big.shape[1]))),
    ])

    return actions, panel


# MOUSE : ctypes to bypasses Roblox SendInput filter
MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_ABSOLUTE = 0x8000

def _to_absolute(x, y):
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
    sw = ctypes.windll.user32.GetSystemMetrics(0)
    sh = ctypes.windll.user32.GetSystemMetrics(1)
    return int(x * 65535 / sw), int(y * 65535 / sh)

def _move(x, y):
    ax, ay = _to_absolute(x, y)
    ctypes.windll.user32.mouse_event(MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE, ax, ay, 0, 0)

def _click(x, y):
    ax, ay = _to_absolute(x, y)
    ctypes.windll.user32.mouse_event(MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE, ax, ay, 0, 0)
    time.sleep(0.05)
    ctypes.windll.user32.mouse_event(MOUSEEVENTF_LEFTDOWN | MOUSEEVENTF_ABSOLUTE, ax, ay, 0, 0)
    time.sleep(0.05)
    ctypes.windll.user32.mouse_event(MOUSEEVENTF_LEFTUP | MOUSEEVENTF_ABSOLUTE, ax, ay, 0, 0)
    time.sleep(0.05)


# Do the actions
def execute(actions):
    if not ENABLE_MOUSE:
        print("Mouse disabled (ENABLE_MOUSE=False)")
        return

    by_color = defaultdict(list)
    for color, btn_xy, zone_xy in actions:
        by_color[color].append((btn_xy, zone_xy))

    total, done = len(actions), 0

    for color, clicks in by_color.items():
        _click(*clicks[0][0])
        time.sleep(CLICK_DELAY * 3)

        for _, zone_xy in clicks:
            _click(*zone_xy)
            time.sleep(CLICK_DELAY)
            done += 1
            print(f"\r  {done}/{total} zones", end="", flush=True)

    print("\nDone.")


# Main
def run_bot(cfg):
    with mss.MSS() as sct:
        sct.shot(output="screenshot.png")

    try:
        actions, panel = analyse("screenshot.png", cfg)
    except Exception as e:
        print(f"ERROR: {e}")
        traceback.print_exc()
        return

    if DEBUG_DISPLAY:
        cv2.imshow("DEBUG, press any key to continue", panel)
        cv2.imwrite("debug_panel.png", panel)
        
        print("debug_panel.png saved")
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    print(f"Ready: {len(actions)} zones, {len(set(c for c,_,_ in actions))} colors")
    
    if ENABLE_MOUSE:
        print("Starting in 2s, switch to game")
        time.sleep(2)
        execute(actions)

if __name__ == "__main__":
    cfg = load_config()
    keyboard.add_hotkey("F7", lambda: run_bot(cfg))

    print("Bot ready, make sure you ran calibrate.py before and switch to the game window (IMPORTANT, if you ran calibrate in full screen, the game must still be full screen)." +
    " Currently not really reliable at the click level, it won't fill everything (and there probably will be mistakes)."+
    " So, run this twice (wait for the movements to stop to press f7 again) to try again. \nF7=run ESC=quit")

    keyboard.wait("esc") # Quit app / keep alive