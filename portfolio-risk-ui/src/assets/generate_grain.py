import base64
import io
import random

from PIL import Image

random.seed(7)

SIZE = 64

# Uniform 0-255 per-pixel noise already spans the full 8-bit range (measured
# mean ~125, stdev ~74 — a textbook uniform distribution), so "widen the
# range" isn't available as a lever, 0-255 is the ceiling. What actually
# increases perceived grain contrast is reshaping the distribution to push
# more mass toward the extremes instead of spreading it evenly: a linear
# contrast stretch around the midpoint, clipped to 0-255. CONTRAST > 1 means
# values within the inner band get stretched to the full range and anything
# outside it clips to pure black/white, giving more, sharper-looking specks
# at the same element opacity.
#
# 2.5 was too aggressive once the visibility bug got fixed (z-index -1 -> 1)
# and opacity got tuned up to compensate for it — ~62% of pixels clipping to
# pure 0/255 read as blocky at any opacity high enough to actually see it.
# 1.3 is a light touch: still a bit more separation than pure uniform noise
# (helps individual specks read at very low opacity), without clipping most
# of the tile to flat black/white.
CONTRAST = 1.3

def stretch(v):
    x = (v - 127.5) * CONTRAST + 127.5
    return max(0, min(255, round(x)))

img = Image.new("L", (SIZE, SIZE))
pixels = [stretch(random.randint(0, 255)) for _ in range(SIZE * SIZE)]
img.putdata(pixels)

buf = io.BytesIO()
img.save(buf, format="PNG", optimize=True)
png_bytes = buf.getvalue()

b64 = base64.b64encode(png_bytes).decode("ascii")

print(f"PNG size: {len(png_bytes)} bytes")
print(f"Base64 size: {len(b64)} chars")
print(f"Data URI size: {len(b64) + len('data:image/png;base64,')} chars")

with open("grain_b64.txt", "w") as f:
    f.write(b64)

with open("grain.png", "wb") as f:
    f.write(png_bytes)
