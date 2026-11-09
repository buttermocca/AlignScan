from __future__ import annotations
from pathlib import Path
import math
from PIL import Image, ImageDraw


def ensure_assets(folder: str | Path) -> None:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "splash_mascot.gif").exists() and (folder / "app_icon.ico").exists():
        return
    frames = []
    for frame in range(16):
        image = Image.new("RGBA", (512, 512), (255, 250, 247, 255))
        draw = ImageDraw.Draw(image)
        bob = round(5 * math.sin(frame * math.pi / 8))
        draw.ellipse((125, 420, 387, 465), fill=(230, 211, 204, 120))
        draw.line((256, 95 + bob, 256, 66 + bob), fill=(93, 173, 145, 255), width=9)
        draw.ellipse((241, 45 + bob, 271, 75 + bob), fill=(255, 207, 102, 255), outline=(235, 160, 95, 255), width=4)
        draw.rounded_rectangle((130, 105 + bob, 382, 420 + bob), radius=48,
                               fill=(255, 255, 250, 255), outline=(255, 140, 153, 255), width=13)
        draw.arc((180, 88 + bob, 330, 140 + bob), 180, 360, fill=(255, 190, 196, 255), width=8)
        for y, width in ((345, 118), (370, 87), (395, 104)):
            draw.rounded_rectangle((256 - width // 2, y + bob, 256 + width // 2, y + 7 + bob),
                                   radius=3, fill=(207, 225, 216, 255))
        if frame in (7, 15):
            draw.arc((185, 220 + bob, 225, 242 + bob), 190, 350, fill=(56, 75, 66, 255), width=8)
            draw.arc((287, 220 + bob, 327, 242 + bob), 190, 350, fill=(56, 75, 66, 255), width=8)
        else:
            draw.ellipse((195, 217 + bob, 216, 247 + bob), fill=(56, 75, 66, 255))
            draw.ellipse((296, 217 + bob, 317, 247 + bob), fill=(56, 75, 66, 255))
            draw.ellipse((200, 220 + bob, 207, 228 + bob), fill=(255, 255, 255, 255))
            draw.ellipse((301, 220 + bob, 308, 228 + bob), fill=(255, 255, 255, 255))
        draw.ellipse((158, 252 + bob, 195, 277 + bob), fill=(255, 190, 194, 150))
        draw.ellipse((317, 252 + bob, 354, 277 + bob), fill=(255, 190, 194, 150))
        draw.arc((220, 244 + bob, 292, 306 + bob), 10, 170, fill=(235, 115, 132, 255), width=8)
        draw.arc((93, 260 + bob, 155, 325 + bob), 190, 310, fill=(255, 140, 153, 255), width=13)
        draw.arc((356, 260 + bob, 419, 325 + bob), 230, 350, fill=(255, 140, 153, 255), width=13)
        for cx, cy, radius in ((95, 160, 10), (414, 180, 8), (407, 385, 7)):
            draw.line((cx - radius, cy, cx + radius, cy), fill=(255, 194, 89, 255), width=5)
            draw.line((cx, cy - radius, cx, cy + radius), fill=(255, 194, 89, 255), width=5)
        frames.append(image)
    frames[0].save(folder / "splash_mascot.gif", save_all=True, append_images=frames[1:],
                   duration=95, loop=0, disposal=2)
    icon = frames[0].resize((256, 256), Image.Resampling.LANCZOS).convert("RGB")
    icon.save(folder / "app_icon.png")
    icon.save(folder / "app_icon.ico", sizes=[(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
