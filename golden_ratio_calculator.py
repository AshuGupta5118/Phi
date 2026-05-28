"""
Golden Ratio (φ) Image Analyzer
Uses MediaPipe Face Mesh for exact anatomical landmarks.
Computes true physiological φ ratios and generates an advanced dashboard.
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import math, os
import sys

print(f"\n[DEBUG] Running on Python version: {sys.version}\n")

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ─────────────────────────────────────────────────────────────────────────────
PHI = (1 + math.sqrt(5)) / 2          # 1.6180339887…
PHI_LABEL = "φ = 1.6180339887…"

# ─────────────────────────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def phi_closeness(r: float) -> float:
    if r == 0: return 0.0
    if r < 1.0: 
        r = 1.0 / r
    return max(0.0, 100.0 - (abs(r - PHI) / PHI) * 100.0)

def dist(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

def midpt(a, b):
    return (int((a[0]+b[0])/2), int((a[1]+b[1])/2))

def closeness_color(c):
    if c >= 90: return "#69FF47"
    if c >= 70: return "#FFD740"
    return "#FF5252"

# ─────────────────────────────────────────────────────────────────────────────
# STRUCTURAL DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def get_structural_keypoints(gray):
    orb = cv2.ORB_create(nfeatures=150)
    kps = orb.detect(gray, None)
    return [(int(k.pt[0]), int(k.pt[1])) for k in kps]

def detect_edges(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    return cv2.Canny(blur, 50, 150)

# ─────────────────────────────────────────────────────────────────────────────
# RESULT CONTAINER
# ─────────────────────────────────────────────────────────────────────────────
class PhiResult:
    def __init__(self, mode="structural"):
        self.mode = mode
        self.measurements = []
        self.mesh_points = []
        self.spiral_center = None
        self.spiral_radius = 60
        self.golden_rect = None

    def add(self, name, pt_a=None, pt_b=None, ratio=0.0, pts=None):
        norm_ratio = ratio if ratio >= 1.0 else (1.0 / ratio if ratio > 0 else 0)
        
        # Safe coordinates extraction to guarantee keys always exist
        safe_a = tuple(int(v) for v in pt_a) if pt_a is not None else (0, 0)
        safe_b = tuple(int(v) for v in pt_b) if pt_b is not None else (0, 0)
        
        self.measurements.append({
            "name": name,
            "a": safe_a,
            "b": safe_b,
            "ratio": norm_ratio,
            "deviation": abs(norm_ratio - PHI),
            "closeness": max(0.0, 100.0 - (abs(norm_ratio - PHI) / PHI) * 100.0),
            "pts": [tuple(int(v) for v in p) for p in (pts or [])],
        })

    @property
    def best(self):
        if not self.measurements: 
            return None
        return max(self.measurements, key=lambda m: m["closeness"])

    @property
    def advanced_score(self):
        if not self.measurements: return 0.0
        closeness_scores = [m["closeness"] for m in self.measurements]
        deviations = [m["deviation"] for m in self.measurements]
        
        # High-order statistical aggregation
        geom_mean = np.exp(np.log([max(x, 0.1) for x in closeness_scores]).mean())
        rmse_dev = np.sqrt(np.mean(np.square(deviations)))
        rmse_closeness = max(0.0, 100.0 - (rmse_dev / PHI) * 100.0)
        std_dev = np.std(closeness_scores)
        
        # Master ensemble math mapping
        ensemble_score = (0.5 * geom_mean) + (0.5 * rmse_closeness) - (0.25 * std_dev)
        return max(0.0, min(100.0, ensemble_score))
# ─────────────────────────────────────────────────────────────────────────────
# HIGH-PRECISION MEDIAPIPE FACE ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def compute_face_phi(img_bgr):
    h, w = img_bgr.shape[:2]
    
    # Target absolute path setup for the required binary model asset
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_landmarker.task")
    if not os.path.exists(model_path):
        model_path = "face_landmarker.task" # Fallback to working directory check

    if not os.path.exists(model_path):
        print(f"  ⚠  Missing 'face_landmarker.task' compilation asset in directory paths.")
        return None

    # Modern Options configuration framework
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1
    )

    # Initialize processor via context management context
    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        # Load object matrix layout structure inside native MediaPipe image shells
        rgb_frame = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Execute processing sequence execution block
        detection_result = landmarker.detect(mp_image)
        
        # Validation checks on array bounds
        if not detection_result.face_landmarks:
            return None

        res = PhiResult(mode="facial anatomy")
        
        # Pull layout map indexes from index offset reference maps
        face_landmarks = detection_result.face_landmarks[0]
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in face_landmarks]
        res.mesh_points = pts

        # --- KEY ANATOMICAL POINTS ---
        top_head = pts[10]       # Top of forehead
        chin = pts[152]          # Bottom of chin
        left_cheek = pts[234]    # Leftmost edge
        right_cheek = pts[454]   # Rightmost edge
        
        nose_bottom = pts[2]
        nose_L = pts[129]        # Left nose wing
        nose_R = pts[358]        # Right nose wing
        
        mouth_L = pts[61]        # Left lip corner
        mouth_R = pts[291]       # Right lip corner
        mouth_top = pts[0]
        mouth_btm = pts[17]
        
        eye_out_L = pts[33]
        eye_out_R = pts[263]
        
        pupil_L = pts[468]       # Exact left iris center
        pupil_R = pts[473]       # Exact right iris center

        # --- 1. Face Height vs Face Width ---
        face_h = dist(top_head, chin)
        face_w = dist(left_cheek, right_cheek)
        if face_w > 0:
            res.add("Face Height / Width", top_head, chin, face_h / face_w, pts=[left_cheek, right_cheek])

        # --- 2. Mouth Width vs Nose Width ---
        mouth_w = dist(mouth_L, mouth_R)
        nose_w = dist(nose_L, nose_R)
        if nose_w > 0:
            res.add("Mouth W / Nose W", mouth_L, mouth_R, mouth_w / nose_w, pts=[nose_L, nose_R])

        # --- 3. Pupil Span vs Mouth Width ---
        pupil_span = dist(pupil_L, pupil_R)
        if mouth_w > 0:
            res.add("Pupil Span / Mouth W", pupil_L, pupil_R, pupil_span / mouth_w, pts=[mouth_L, mouth_R])

        # --- 4. Forehead to Nose vs Nose to Chin ---
        top_to_nose = dist(top_head, nose_bottom)
        nose_to_chin = dist(nose_bottom, chin)
        if nose_to_chin > 0:
            res.add("Top→Nose / Nose→Chin", top_head, nose_bottom, top_to_nose / nose_to_chin, pts=[nose_bottom, chin])

        # --- 5. Face Width vs Outer Eye Span ---
        eye_span_out = dist(eye_out_L, eye_out_R)
        if eye_span_out > 0:
            res.add("Face W / Outer Eye Span", left_cheek, right_cheek, face_w / eye_span_out, pts=[eye_out_L, eye_out_R])

        # --- 6. Lower Lip to Chin vs Nose to Mouth ---
        nose_to_mouth = dist(nose_bottom, mouth_top)
        lip_to_chin = dist(mouth_btm, chin)
        if nose_to_mouth > 0:
            res.add("Lip→Chin / Nose→Mouth", mouth_btm, chin, lip_to_chin / nose_to_mouth, pts=[nose_bottom, mouth_top])

        # Create bounding coordinates frame metrics
        min_x = min([top_head[0], chin[0], left_cheek[0], right_cheek[0]])
        max_x = max([top_head[0], chin[0], left_cheek[0], right_cheek[0]])
        min_y = top_head[1]
        max_y = chin[1]
        res.golden_rect = (min_x, min_y, max_x - min_x, max_y - min_y)

        res.spiral_center = midpt(pupil_L, pupil_R)
        res.spiral_radius = int(face_w * 0.35)

        return res

# ─────────────────────────────────────────────────────────────────────────────
# STRUCTURAL PHI ANALYSIS (FALLBACK)
# ─────────────────────────────────────────────────────────────────────────────
def compute_structural_phi(img_bgr, edges, struct_pts):
    h, w = img_bgr.shape[:2]
    res = PhiResult(mode="structural chaos test")

    # 1. Image Canvas Proportions (Legitimate framing check)
    res.add("Image Canvas W / H", (0, h//2), (w, h//2), w/h if h else 0)

    # 2. Track Real Contours instead of hardcoded math variables
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        lc = max(contours, key=cv2.contourArea)
        if cv2.contourArea(lc) > 200:
            x, y, cw, ch = cv2.boundingRect(lc)
            if cw > 0 and ch > 0:
                res.add("Main Subject W/H", (x, y+ch//2), (x+cw, y+ch//2), cw/ch)

    # 3. CRUCIAL FIX: Measure internal chaos/asymmetry of structural points
    # Real structural harmony requires key points to be balanced across the center line.
    if len(struct_pts) > 4:
        left_side_points = sum(1 for pt in struct_pts if pt[0] < w//2)
        right_side_points = sum(1 for pt in struct_pts if pt[0] >= w//2)
        
        # In a chaotic painting or a potato, edge point distribution is wildly asymmetric
        if right_side_points > 0:
            balance_ratio = left_side_points / right_side_points
            res.add("Point Distribution Symmetry", (0, h//2), (w, h//2), balance_ratio)
        else:
            res.add("Point Distribution Symmetry", (0, h//2), (w, h//2), 0.0)
    else:
        # If there are no clear features or edges, assign a definitive failure metric
        res.add("Point Distribution Symmetry", (0,0), (0,0), 0.0)

    # Re-map standard visual display parameters
    res.spiral_center = (w//2, h//2)
    res.spiral_radius = min(w, h)//6
    res.golden_rect = (int(w*0.1), int(h*0.1), int(w*0.8), int(h*0.8))

    return res
# ─────────────────────────────────────────────────────────────────────────────
# GOLDEN SPIRAL DRAW
# ─────────────────────────────────────────────────────────────────────────────

def draw_spiral(ax, cx, cy, base_r, color="#FFD700", alpha=0.75):
    t = np.linspace(0, 4*math.pi, 800)
    r = base_r * np.exp(0.3063 * t) / np.exp(0.3063 * 4*math.pi)
    xs = cx + r * np.cos(t)
    ys = cy + r * np.sin(t)
    ax.plot(xs, ys, color=color, lw=1.6, alpha=alpha)

# ─────────────────────────────────────────────────────────────────────────────
# GENERATE OUTPUT IMAGE
# ─────────────────────────────────────────────────────────────────────────────

CMAP = {
    "phi_grid": "#00E5FF",
    "rect":     "#FF6D00",
    "spiral":   "#FFD700",
    "kp":       "#B388FF",
    "mesh":     "#FFFFFF",
}

def generate_output(img_bgr, result: PhiResult, edges, struct_pts, out_path):
    h, w = img_bgr.shape[:2]
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    fig = plt.figure(figsize=(20, 12), facecolor="#060606")
    gs = fig.add_gridspec(
        2, 3,
        left=0.01, right=0.995, top=0.945, bottom=0.03,
        hspace=0.28, wspace=0.07,
        height_ratios=[3, 1],
        width_ratios=[2.5, 1, 0.95],
    )
    ax_img   = fig.add_subplot(gs[0, 0])
    ax_stat  = fig.add_subplot(gs[0, 1])
    ax_edge  = fig.add_subplot(gs[0, 2])
    ax_bar   = fig.add_subplot(gs[1, :2])
    ax_phi   = fig.add_subplot(gs[1, 2])

    for ax in [ax_img, ax_stat, ax_edge, ax_bar, ax_phi]:
        ax.set_facecolor("#060606")
        for s in ax.spines.values(): s.set_edgecolor("#2a2a2a")

    # Image overlay
    ax_img.imshow(rgb)
    ax_img.axis("off")
    ax_img.set_title("True Anatomical φ Overlay", color="#FFD700", fontsize=11, pad=7, fontfamily="monospace", fontweight="bold")

    # Draw face mesh point arrays
    if result.mesh_points:
        mx = [p[0] for p in result.mesh_points]
        my = [p[1] for p in result.mesh_points]
        ax_img.plot(mx, my, ".", color=CMAP["mesh"], ms=0.8, alpha=0.3)

    # Golden Rectangle based on anatomy bounding box
    if result.golden_rect:
        rx,ry,rw,rh = result.golden_rect
        rect = plt.Rectangle((rx,ry), rw, rh, lw=1.2, edgecolor=CMAP["rect"], facecolor="none", ls="--", alpha=0.6)
        ax_img.add_patch(rect)
        phi_ry = ry + int(rh/PHI)
        ax_img.plot([rx,rx+rw],[phi_ry,phi_ry], color=CMAP["rect"], lw=0.8, ls="--", alpha=0.4)

    # Spiral plotting
    if result.spiral_center:
        cx,cy = result.spiral_center
        draw_spiral(ax_img, cx, cy, result.spiral_radius, color=CMAP["spiral"], alpha=0.7)
        ax_img.plot(cx, cy, "*", color=CMAP["spiral"], ms=10, alpha=0.8)

    # Draw measurements vectors
    for i, m in enumerate(result.measurements[:10]):
        a, b = m["a"], m["b"]
        col = closeness_color(m["closeness"])
        ax_img.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="<->", color=col, lw=1.8))
        ax_img.plot(a[0],a[1],"o",color=col,ms=5,alpha=0.9,zorder=5)
        ax_img.plot(b[0],b[1],"o",color=col,ms=5,alpha=0.9,zorder=5)
        mx2=(a[0]+b[0])/2; my2=(a[1]+b[1])/2
        ax_img.text(mx2, my2, f"{m['ratio']:.3f}", color=col, fontsize=6.5, fontfamily="monospace", ha="center", va="bottom", bbox=dict(fc="#000000AA",ec="none",pad=0.5))
        
        for p in m["pts"]: 
            ax_img.plot(p[0],p[1],"s",color=col,ms=3.5,alpha=0.7,zorder=4)
            ax_img.plot([a[0], p[0]], [a[1], p[1]], color=col, lw=0.5, ls=":", alpha=0.5)

    # Statistics Text compilation panel
    ax_stat.axis("off")
    ax_stat.set_title("  φ  Anatomical Stats", color="#FFD700", fontsize=11, pad=7, fontfamily="monospace", fontweight="bold", loc="left")

    avg_c = result.advanced_score
    ratios = [m["ratio"] for m in result.measurements]
    avg_r = float(np.mean(ratios)) if ratios else 0.0
    best  = result.best

    rows = [
        ("Golden Ratio φ",    f"{PHI:.10f}",       "#FFD700"),
        ("",                  "",                   ""),
        ("Analysis Mode",     result.mode.upper(),  "#90CAF9"),
        ("Measurements",      str(len(result.measurements)), "#DDD"),
        ("",                  "",                   ""),
        ("Calculated Ratio",  f"{avg_r:.6f}",       closeness_color(avg_c)),
        ("Closeness to φ",    f"{avg_c:.2f}%",      closeness_color(avg_c)),
        ("φ Deviation",       f"{abs(avg_r-PHI):.6f}", closeness_color(avg_c)),
        ("φ Error %",         f"{abs(avg_r-PHI)/PHI*100:.4f}%", closeness_color(avg_c)),
        ("",                  "",                   ""),
    ]
    if best:
        rows += [
            ("Best Measurement",  best["name"][:24],      "#B388FF"),
            ("  Ratio",           f"{best['ratio']:.6f}", closeness_color(best["closeness"])),
            ("  Closeness",       f"{best['closeness']:.2f}%", closeness_color(best["closeness"])),
        ]

    ty = 0.97
    for lbl, val, col in rows:
        if not lbl: ty -= 0.018; continue
        ax_stat.text(0.04, ty, lbl+":", color="#777", fontsize=7.8, fontfamily="monospace", transform=ax_stat.transAxes, va="top")
        ax_stat.text(0.97, ty, val, color=col, fontsize=7.8, fontfamily="monospace", transform=ax_stat.transAxes, va="top", ha="right")
        ty -= 0.048

    # Performance Gauge layout engine
    arc_y = 0.16
    theta_bg = np.linspace(0, math.pi, 200)
    xs_bg = 0.5 + 0.38*np.cos(theta_bg[::-1])
    ys_bg = arc_y + 0.12*np.sin(theta_bg)
    ax_stat.plot(xs_bg, ys_bg, color="#2a2a2a", lw=7, solid_capstyle="round", transform=ax_stat.transAxes, clip_on=False)
    theta_v = np.linspace(0, avg_c/100*math.pi, 200)
    xs_v = 0.5 + 0.38*np.cos(theta_v[::-1])
    ys_v = arc_y + 0.12*np.sin(theta_v)
    ax_stat.plot(xs_v, ys_v, color=closeness_color(avg_c), lw=7, solid_capstyle="round", transform=ax_stat.transAxes, clip_on=False)
    ax_stat.text(0.5, arc_y-0.045, f"{avg_c:.1f}%", color=closeness_color(avg_c), fontsize=14, fontweight="bold", ha="center", fontfamily="monospace", transform=ax_stat.transAxes)
    ax_stat.text(0.5, arc_y-0.10, "Overall Closeness", color="#555", fontsize=7, ha="center", fontfamily="monospace", transform=ax_stat.transAxes)

    # Horizontal metrics summary subbars
    tm_y = 0.15
    for m in result.measurements[:3]:  # Avoid overfill bounds inside chart allocations
        col = closeness_color(m["closeness"])
        ax_stat.text(0.04, tm_y-0.02, m["name"][:24], color="#888", fontsize=5.8, fontfamily="monospace", transform=ax_stat.transAxes)
        bar_w = m["closeness"]/100 * 0.55
        ax_stat.add_patch(plt.Rectangle((0.04, tm_y-0.038), 0.55, 0.012, facecolor="#1a1a1a", transform=ax_stat.transAxes, clip_on=False))
        ax_stat.add_patch(plt.Rectangle((0.04, tm_y-0.038), bar_w, 0.012, facecolor=col, alpha=0.7, transform=ax_stat.transAxes, clip_on=False))
        ax_stat.text(0.97, tm_y-0.025, f"{m['ratio']:.3f}  {m['closeness']:.1f}%", color=col, fontsize=5.8, fontfamily="monospace", transform=ax_stat.transAxes, ha="right")
        tm_y -= 0.055

    # Edge Map Panel rendering
    ax_edge.imshow(edges, cmap="inferno", aspect="auto")
    ax_edge.axis("off")
    ax_edge.set_title("Structural Edges", color="#FF9800", fontsize=9, pad=5, fontfamily="monospace")
    for (kx,ky) in struct_pts[:60]: ax_edge.plot(kx, ky, ".", color="#00E5FF", ms=1.8, alpha=0.6)

    # Main Bar Chart visualization matrix
    if result.measurements:
        names = [m["name"] for m in result.measurements]
        cv_list = [m["closeness"] for m in result.measurements]
        ratios  = [m["ratio"]    for m in result.measurements]
        bcolors = [closeness_color(c) for c in cv_list]
        x = np.arange(len(names))
        bars = ax_bar.bar(x, cv_list, color=bcolors, width=0.6, edgecolor="#111", lw=0.5, alpha=0.88)
        ax_bar.axhline(100, color="#FFD700", lw=1.2, ls="--", alpha=0.55, label="Perfect φ (100%)")
        ax_bar.axhline(avg_c, color="#FF4081", lw=1.3, ls="-", alpha=0.7, label=f"Average {avg_c:.1f}%")
        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels(names, rotation=12, ha="right", color="#AAA", fontsize=8, fontfamily="monospace")
        ax_bar.set_yticks([0,25,50,75,100])
        ax_bar.set_yticklabels(["0","25","50","75","100%"], color="#666", fontsize=7)
        ax_bar.set_ylabel("Closeness %", color="#777", fontsize=8, fontfamily="monospace")
        ax_bar.set_ylim(0, 118)
        ax_bar.set_title("Anatomical Ratios to φ", color="#FFD700", fontsize=9, pad=5, fontfamily="monospace")
        ax_bar.legend(fontsize=7, loc="upper right", labelcolor="#CCC", facecolor="#111", edgecolor="#333")
        ax_bar.grid(axis="y", color="#1a1a1a", lw=0.8)
        for bar, r in zip(bars, ratios): 
            ax_bar.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1.5, f"{r:.3f}", ha="center", va="bottom", color="#CCC", fontsize=6.5, fontfamily="monospace")

    # Phi Mathematical Diagram compilation
    ax_phi.set_xlim(0, 1); ax_phi.set_ylim(0, 1)
    ax_phi.set_aspect("equal"); ax_phi.axis("off")
    ax_phi.set_title("φ Diagram", color="#FFD700", fontsize=9, pad=5, fontfamily="monospace")

    seq_colors = ["#FF4081","#FF9800","#FFD740","#69FF47","#00E5FF","#7C4DFF","#F06292"]
    fibs = [1, 1, 2, 3, 5, 8, 13]
    scale = 0.038
    positions = [(0.12,0.38),(0.12+scale,0.38),(0.12,0.38+scale),(0.12-2*scale,0.38),(0.12-2*scale,0.38-3*scale),(0.12+scale,0.38-3*scale),(0.12-2*scale,0.38+scale)]
    for (px,py),fs,col in zip(positions, fibs, seq_colors):
        s = fs*scale
        ax_phi.add_patch(plt.Rectangle((px,py),s,s, lw=1.2, edgecolor=col, facecolor=col+"22"))

    t_s = np.linspace(0, 4*math.pi, 800)
    r_s = 0.012 * np.exp(0.3063*t_s)
    xs_s = 0.38 + r_s*np.cos(t_s)
    ys_s = 0.45 + r_s*np.sin(t_s)
    mask = (xs_s>0.02)&(xs_s<0.98)&(ys_s>0.02)&(ys_s<0.98)
    ax_phi.plot(xs_s[mask], ys_s[mask], color="#FFD700", lw=1.6, alpha=0.9)

    ax_phi.text(0.5, 0.92, PHI_LABEL, ha="center", color="#FFD700", fontsize=7.5, fontfamily="monospace", style="italic", transform=ax_phi.transAxes)
    ax_phi.text(0.5, 0.83, f"Img φ ≈ {avg_r:.6f}", ha="center", color=closeness_color(avg_c), fontsize=7.5, fontfamily="monospace", transform=ax_phi.transAxes)
    ax_phi.text(0.5, 0.74, f"Closeness: {avg_c:.1f}%", ha="center", color=closeness_color(avg_c), fontsize=7.5, fontfamily="monospace", transform=ax_phi.transAxes)

    fig.suptitle(f"  φ Golden Ratio Analysis  ·  Mode: {result.mode.upper()}  ·  {len(result.measurements)} Measurements  ·  Avg Closeness: {avg_c:.2f}%", color="#FFD700", fontsize=11.5, y=0.985, fontfamily="monospace", fontweight="bold")
    fig.savefig(out_path, dpi=155, bbox_inches="tight", facecolor="#060606", edgecolor="none")
    plt.close(fig)
    return out_path

# ─────────────────────────────────────────────────────────────────────────────
# CONSOLE REPORT
# ─────────────────────────────────────────────────────────────────────────────
def print_report(result: PhiResult):
    SEP = "═" * 65
    sep = "─" * 65
    print(f"\n{SEP}")
    print(f"   GOLDEN RATIO (φ) ADVANCED ANATOMICAL REPORT")
    print(f"   True Mathematical φ = {PHI:.10f}")
    print(SEP)
    print(f"   Mode: {result.mode.upper()}   |   Measurements: {len(result.measurements)}")
    print(sep)
    print(f"   {'Measurement':<28} {'Ratio':>10}  {'Closeness':>10}  Bar")
    print(f"   {'─'*28} {'─'*10}  {'─'*10}  ───")
    for m in result.measurements:
        bar = "█" * max(1,int(m["closeness"]/10)) + "░"*(10-max(1,int(m["closeness"]/10)))
        print(f"   {m['name']:<28} {m['ratio']:>10.4f}  {m['closeness']:>9.2f}%  {bar}")
    print(sep)
    
    # Utilizing our new higher-order statistical ensemble score
    final_score = result.advanced_score
    
    # Simple fallback average for display depth, calculated cleanly inline
    ratios = [m["ratio"] for m in result.measurements]
    display_avg_r = float(np.mean(ratios)) if ratios else 0.0
    
    print(f"   Baseline Average Ratio : {display_avg_r:.6f}")
    print(f"   COMPOSITE HARMONY SCORE: {final_score:.2f}% (Variance Penalized)")
    
    best = result.best
    if best:
        print(f"\n   Best Individual Match  : {best['name']}")
        print(f"   Best Ratio             : {best['ratio']:.6f}")
        print(f"   Best Closeness         : {best['closeness']:.2f}%")
    print(sep)
    
    # Enhanced verdicts tuned to our strict, high-penalty algorithm
    if final_score >= 88:    v = "🌟  Exceptional, true geometric facial harmony!"
    elif final_score >= 75:  v = "✨  Strong structural alignment with the Golden Ratio."
    elif final_score >= 50:  v = "🔶  Moderate or localized facial symmetry."
    else:                    v = "🥔  Low overall structural harmony (Potato Threshold)."
    print(f"\n   Verdict: {v}")
    print(SEP + "\n")
# ─────────────────────────────────────────────────────────────────────────────
# MAIN EXECUTION INTERFACE
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n╔═══════════════════════════════════════════════════╗")
    print("║   Golden Ratio (φ) Image Analyzer                 ║")
    print("║       Made by Ashutosh Gupta                      ║")
    print("╚═══════════════════════════════════════════════════╝\n")

    while True:
        path = input("Enter image path (or 'q' to quit): ").strip().strip('"\'')
        if path.lower() in ("q","quit","exit"):
            print("Goodbye!"); sys.exit(0)
        if not os.path.isfile(path):
            print(f"  ⚠  Not found: {path}\n"); continue
        break

    print(f"\n  Loading: {path}")
    img = cv2.imread(path)
    if img is None:
        print("  ✖  Could not load image."); sys.exit(1)

    print("  Detecting true facial anatomy using MediaPipe…")
    result = compute_face_phi(img)

    if result is None:
        print("  ℹ  No face detected — falling back to structural φ analysis…")
        edges = detect_edges(img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        struct_pts = get_structural_keypoints(gray)
        result = compute_structural_phi(img, edges, struct_pts)
    else:
        print("  ✔  Facial anatomy mapped successfully!")
        edges = detect_edges(img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        struct_pts = get_structural_keypoints(gray)

    print_report(result)

    base, _ = os.path.splitext(path)
    out_path = base + "_golden_anatomy_analysis.png"

    print("  Rendering high-precision dashboard…")
    generate_output(img, result, edges, struct_pts, out_path)
    print(f"  ✅  Saved → {out_path}\n")

if __name__ == "__main__":
    main()
