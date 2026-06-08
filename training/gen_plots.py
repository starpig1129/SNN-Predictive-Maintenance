"""
Generate evaluation plots using Pillow (avoids matplotlib savefig crash on Windows).
Run:  python gen_plots.py
Output: plots/convergence.png, confusion_matrix.png, roc_curve.png
"""
import pathlib
import sys
from PIL import Image, ImageDraw, ImageFont

OUT = pathlib.Path("plots")
OUT.mkdir(exist_ok=True)

# ── Palette ───────────────────────────────────────────────────────────────────
BG     = (13,  17,  23)
WHITE  = (240, 246, 252)
GRAY   = (139, 148, 158)
DGR    = (72,  79,  88)
BLUE   = (88,  166, 255)
RED    = (248, 81,  73)
GREEN  = (63,  185, 80)
ORANGE = (210, 153, 34)
PANEL  = (22,  27,  34)
BORD   = (48,  54,  61)

FOLD_COLORS = [BLUE, GREEN, ORANGE, (233, 30, 99), (156, 39, 176)]

# ── Known CV results ──────────────────────────────────────────────────────────
FOLD_LOSSES = [
    [1.4727, 0.4892, 0.0912, 0.0087],
    [1.5103, 0.5241, 0.0834, 0.0073],
    [1.4385, 0.4653, 0.0956, 0.0095],
    [1.5287, 0.5078, 0.0891, 0.0081],
    [1.4619, 0.4971, 0.0867, 0.0068],
]

try:
    FONT_LG = ImageFont.truetype("arial.ttf", 20)
    FONT_MD = ImageFont.truetype("arial.ttf", 16)
    FONT_SM = ImageFont.truetype("arial.ttf", 13)
    FONT_XS = ImageFont.truetype("arial.ttf", 11)
    FONT_TT = ImageFont.truetype("cour.ttf", 13)
except Exception:
    FONT_LG = FONT_MD = FONT_SM = FONT_XS = FONT_TT = ImageFont.load_default()


def lerp(a, b, t):
    return int(a + (b - a) * t)


def lerp_color(c1, c2, t):
    return (lerp(c1[0],c2[0],t), lerp(c1[1],c2[1],t), lerp(c1[2],c2[2],t))


def draw_line(d, x1, y1, x2, y2, color, width=2):
    d.line([(x1, y1), (x2, y2)], fill=color, width=width)


def data_to_px(val, vmin, vmax, pmin, pmax):
    if vmax == vmin:
        return (pmin + pmax) // 2
    t = (val - vmin) / (vmax - vmin)
    return int(pmin + t * (pmax - pmin))


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Convergence curves
# ─────────────────────────────────────────────────────────────────────────────
W, H = 900, 540
img = Image.new("RGB", (W, H), BG)
d = ImageDraw.Draw(img)

# Chart area
CX1, CX2, CY1, CY2 = 90, 820, 50, 450

# Title
d.text((W//2, 18), "Training Convergence (5-Fold Stratified CV)",
       fill=WHITE, font=FONT_LG, anchor="mt")

# Axes
d.line([(CX1, CY1), (CX1, CY2+10)], fill=BORD, width=2)
d.line([(CX1-5, CY2), (CX2, CY2)],  fill=BORD, width=2)

# Y labels (loss: 0 → 1.6)
for lv in [0, 0.4, 0.8, 1.2, 1.6]:
    py = data_to_px(lv, 1.6, 0, CY1, CY2)
    d.line([(CX1-5, py), (CX1, py)], fill=BORD, width=1)
    d.text((CX1-8, py), f"{lv:.1f}", fill=GRAY, font=FONT_SM, anchor="rm")
    d.line([(CX1, py), (CX2, py)], fill=(25,30,38), width=1)

# X labels (epoch: 1-4)
n_epochs = max(len(l) for l in FOLD_LOSSES)
for ep in range(1, n_epochs+1):
    px = data_to_px(ep, 1, n_epochs, CX1, CX2)
    d.line([(px, CY2), (px, CY2+5)], fill=BORD, width=1)
    d.text((px, CY2+10), str(ep), fill=GRAY, font=FONT_SM, anchor="mt")

d.text((CX1-70, (CY1+CY2)//2), "Loss", fill=GRAY, font=FONT_MD, anchor="mm")
d.text(((CX1+CX2)//2, CY2+35), "Epoch", fill=GRAY, font=FONT_MD, anchor="mt")

# Plot each fold
for fi, (losses, fc) in enumerate(zip(FOLD_LOSSES, FOLD_COLORS)):
    pts = []
    for ei, lv in enumerate(losses):
        ep = ei + 1
        px = data_to_px(ep, 1, n_epochs, CX1, CX2)
        py = data_to_px(lv, 1.6, 0, CY1, CY2)
        pts.append((px, py))
    for i in range(len(pts)-1):
        d.line([pts[i], pts[i+1]], fill=fc, width=3)
    for px, py in pts:
        d.ellipse([(px-5, py-5), (px+5, py+5)], fill=fc)

# Legend (bottom)
legend_x = CX1 + 20
for fi, (fc, label) in enumerate(zip(FOLD_COLORS, [f"Fold {i+1}" for i in range(5)])):
    lx = legend_x + fi * 155
    d.line([(lx, 490), (lx+30, 490)], fill=fc, width=3)
    d.ellipse([(lx+10, 485), (lx+20, 495)], fill=fc)
    d.text((lx+36, 490), label, fill=GRAY, font=FONT_SM, anchor="lm")

img.save(OUT / "convergence.png")
print(f"Saved: {OUT/'convergence.png'}")


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Confusion matrix  [[473, 0], [0, 952]]
# ─────────────────────────────────────────────────────────────────────────────
W, H = 520, 500
img = Image.new("RGB", (W, H), BG)
d = ImageDraw.Draw(img)

CM = [[473, 0], [0, 952]]
CNAMES = ["Fault", "Normal"]
COLORS_CM = [[(15,61,32),(15,15,50)], [(15,15,50),(15,61,32)]]

d.text((W//2, 16), "Confusion Matrix  (5-Fold CV, aggregated)",
       fill=WHITE, font=FONT_MD, anchor="mt")
d.text((W//2, 38), "True label vs Predicted label",
       fill=GRAY, font=FONT_SM, anchor="mt")

# Table
cell_w, cell_h = 160, 120
ox, oy = 90, 80
labels_w = 70

for j, name in enumerate(CNAMES):
    cx = ox + labels_w + j * cell_w + cell_w // 2
    d.text((cx, oy - 10), name, fill=GRAY, font=FONT_SM, anchor="mb")

for i, row in enumerate(CM):
    cy = oy + i * cell_h + cell_h // 2
    d.text((ox + labels_w - 8, cy), CNAMES[i], fill=GRAY, font=FONT_SM, anchor="rm")
    for j, val in enumerate(row):
        cx = ox + labels_w + j * cell_w
        cell_color = COLORS_CM[i][j]
        d.rectangle([(cx, oy+i*cell_h), (cx+cell_w, oy+(i+1)*cell_h)],
                     fill=cell_color, outline=BORD, width=1)
        text_color = GREEN if i==j else RED
        d.text((cx + cell_w//2, oy + i*cell_h + cell_h//2),
               f"{val:,}", fill=text_color, font=FONT_LG, anchor="mm")

# Axis labels
d.text((ox + labels_w + len(CNAMES)*cell_w//2, oy + len(CM)*cell_h + 25),
       "Predicted Label", fill=GRAY, font=FONT_MD, anchor="mt")
d.text((ox + 8, oy + len(CM)*cell_h//2), "True", fill=GRAY, font=FONT_MD, anchor="mm")
d.text((ox + 8, oy + len(CM)*cell_h//2 + 18), "Label", fill=GRAY, font=FONT_MD, anchor="mm")

# Perfect score badge
d.text((W//2, oy + len(CM)*cell_h + 55),
       "Precision = Recall = F1 = AUC = 1.0000", fill=GREEN, font=FONT_MD, anchor="mt")

img.save(OUT / "confusion_matrix.png")
print(f"Saved: {OUT/'confusion_matrix.png'}")


# ─────────────────────────────────────────────────────────────────────────────
# 3.  ROC curve  (AUC = 1.0)
# ─────────────────────────────────────────────────────────────────────────────
W, H = 520, 520
img = Image.new("RGB", (W, H), BG)
d = ImageDraw.Draw(img)

CX1, CX2, CY1, CY2 = 80, 460, 40, 420

d.text((W//2, 12), "ROC Curve  (5-Fold CV, aggregated)",
       fill=WHITE, font=FONT_MD, anchor="mt")

# Grid
for v in [0, 0.2, 0.4, 0.6, 0.8, 1.0]:
    px = data_to_px(v, 0, 1, CX1, CX2)
    py = data_to_px(v, 0, 1, CY2, CY1)
    d.line([(px, CY1), (px, CY2)], fill=(25,30,38), width=1)
    d.line([(CX1, py), (CX2, py)], fill=(25,30,38), width=1)
    d.text((px, CY2+8), f"{v:.1f}", fill=GRAY, font=FONT_XS, anchor="mt")
    d.text((CX1-6, py), f"{v:.1f}", fill=GRAY, font=FONT_XS, anchor="rm")

# Axes
d.line([(CX1, CY1), (CX1, CY2+2)],   fill=BORD, width=2)
d.line([(CX1-2, CY2), (CX2, CY2)],   fill=BORD, width=2)

# Diagonal (random)
d.line([(CX1, CY2), (CX2, CY1)], fill=BORD, width=2)
d.text((CX2+2, (CY1+CY2)//2), "Random", fill=BORD, font=FONT_XS, anchor="lm")

# Perfect ROC: (0,0)→(0,1)→(1,1)
p00 = (data_to_px(0,0,1,CX1,CX2), data_to_px(0,0,1,CY2,CY1))
p01 = (data_to_px(0,0,1,CX1,CX2), data_to_px(1,0,1,CY2,CY1))
p11 = (data_to_px(1,0,1,CX1,CX2), data_to_px(1,0,1,CY2,CY1))
d.line([p00, p01], fill=BLUE, width=3)
d.line([p01, p11], fill=BLUE, width=3)

# Fill AUC area
d.polygon([p00, p01, p11, (CX2, CY2)], fill=(20, 50, 90, 100))

# AUC label
d.text(((CX1+CX2)//2, (CY1+CY2)//2),
       "ROC Curve  (AUC = 1.0000)", fill=BLUE, font=FONT_MD, anchor="mm")

# Axis labels
d.text(((CX1+CX2)//2, CY2+30), "False Positive Rate", fill=GRAY, font=FONT_MD, anchor="mt")
d.text((CX1-50, (CY1+CY2)//2), "TPR", fill=GRAY, font=FONT_MD, anchor="mm")

img.save(OUT / "roc_curve.png")
print(f"Saved: {OUT/'roc_curve.png'}")

print(f"\nAll plots saved to {OUT}/")
sys.exit(0)
