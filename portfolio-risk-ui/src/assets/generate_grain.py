import base64
import io
import random

from PIL import Image

random.seed(7)

SIZE = 64

img = Image.new("L", (SIZE, SIZE))
pixels = [random.randint(0, 255) for _ in range(SIZE * SIZE)]
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
