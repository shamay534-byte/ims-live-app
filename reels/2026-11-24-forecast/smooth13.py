import numpy as np, cv2
def smooth_face(img, fmask, strength=0.6):
    """img float32 HxWx3 (full res), fmask float 0..1 HxW face-skin confidence. Edge-preserving skin smoothing, natural."""
    m = (fmask > 0.35).astype(np.uint8)
    ys, xs = np.where(m)
    if len(ys) < 50: return img
    y0, y1 = max(0, ys.min() - 40), min(img.shape[0], ys.max() + 40)
    x0, x1 = max(0, xs.min() - 40), min(img.shape[1], xs.max() + 40)
    crop = img[y0:y1, x0:x1]
    c8 = np.clip(crop, 0, 255).astype(np.uint8)
    sm = cv2.bilateralFilter(c8, 0, 22, 9).astype(np.float32)
    sm = cv2.bilateralFilter(np.clip(sm, 0, 255).astype(np.uint8), 0, 16, 5).astype(np.float32)
    # keep a little fine grain so it never looks plastic
    grain = crop - cv2.GaussianBlur(crop, (0, 0), 0.8)
    out = sm + grain * 0.35
    mm = cv2.GaussianBlur(cv2.erode(m[y0:y1, x0:x1], np.ones((7, 7), np.uint8)).astype(np.float32), (0, 0), 6)[..., None]
    img = img.copy()
    img[y0:y1, x0:x1] = crop + (out - crop) * mm * strength
    return img
