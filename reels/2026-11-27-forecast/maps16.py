# build 3D-shaded map textures from the 4 ECMWF 500hPa maps (colour -> geopotential via the legend)
from PIL import Image
import numpy as np, cv2, json
D = "r15/maps/"
a = np.array(Image.open(D + "m1.png").convert("RGB")).astype(int)
row = a[720]
segs = []; prev = None
for x in range(330, 958):
    c = tuple(row[x])
    if c != prev:
        segs.append([x, x, c]); prev = c
    else: segs[-1][1] = x
legend = [s for s in segs if s[1] - s[0] >= 10 and s[2] not in [(255,255,255),(0,0,0)]]
print(len(legend), legend[0], legend[-1])
cols = np.array([s[2] for s in legend], np.float32)
vals = 476 + 4 * np.arange(len(cols))  # 476..600 gpdm
X0, X1, Y0, Y1 = 7, 951, 21, 709
for i in range(1, 5):
    im = np.array(Image.open(D + f"m{i}.png").convert("RGB"))[Y0:Y1, X0:X1].astype(np.float32)
    d = ((im[:, :, None, :] - cols[None, None]) ** 2).sum(-1)
    k = d.argmin(-1); ok = d.min(-1) < 60  # exact palette pixels only (not lines/text)
    h = np.where(ok, vals[k], 0).astype(np.float32)
    # fill lines/text: inpaint the non-palette pixels
    hm = cv2.inpaint(((h - 470) * 1.8).clip(0, 255).astype(np.uint8), (~ok).astype(np.uint8) * 255, 7, cv2.INPAINT_TELEA)
    hf = cv2.GaussianBlur(hm.astype(np.float32) / 1.8 + 470, (0, 0), 6)
    np.save(D + f"h{i}.npy", hf)
    # palette-clean colour texture (lines removed) + contours kept as a thin white overlay
    lines = (~ok) & (im.mean(-1) > 200)   # white isobars
    coast = (~ok) & (im.mean(-1) < 60)    # black coastlines
    col = cv2.inpaint(im.astype(np.uint8), ((~ok) * 255).astype(np.uint8), 5, cv2.INPAINT_TELEA)
    col = cv2.bilateralFilter(col, 0, 18, 5)
    Image.fromarray(col).save(D + f"c{i}.png")
    Image.fromarray(lines.astype(np.uint8) * 255).save(D + f"iso{i}.png")
    Image.fromarray(coast.astype(np.uint8) * 255).save(D + f"coast{i}.png")
    print(i, h[ok].min(), h[ok].max(), ok.mean())
