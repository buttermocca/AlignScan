from __future__ import annotations
from pathlib import Path
import math
from PIL import Image, ImageDraw


def _make_banner(path: Path) -> None:
    """Create a small pastel mascot header image with a receipt, a chibi girl and a cat."""
    image = Image.new("RGB", (420, 139), (240, 243, 255))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((3, 3, 416, 135), radius=25,
                           fill=(251, 249, 255), outline=(190, 189, 244), width=4)
    # Soft sparkles
    for x, y, radius in ((25, 24, 5), (192, 25, 6), (238, 102, 4), (393, 32, 5)):
        draw.line((x-radius, y, x+radius, y), fill=(255, 186, 203), width=3)
        draw.line((x, y-radius, x, y+radius), fill=(255, 186, 203), width=3)
    # Receipt mascot
    draw.ellipse((36, 105, 123, 128), fill=(224, 219, 237))
    draw.rounded_rectangle((43, 28, 116, 117), radius=17,
                           fill=(255, 255, 250), outline=(255, 139, 164), width=4)
    draw.ellipse((60, 54, 68, 66), fill=(57, 61, 91))
    draw.ellipse((91, 54, 99, 66), fill=(57, 61, 91))
    draw.arc((68, 61, 92, 87), 5, 175, fill=(232, 109, 146), width=3)
    for y, width in ((91, 45), (101, 32)):
        draw.rounded_rectangle((79-width//2, y, 79+width//2, y+4),
                               radius=2, fill=(178, 198, 222))
    draw.ellipse((48, 67, 58, 75), fill=(255, 194, 204))
    draw.ellipse((101, 67, 111, 75), fill=(255, 194, 204))
    # Chibi girl's hair, face, bow and shoulders
    draw.ellipse((152, 11, 245, 116), fill=(109, 75, 99))
    draw.ellipse((167, 23, 231, 99), fill=(255, 222, 207), outline=(225, 151, 156), width=2)
    draw.pieslice((156, 4, 241, 81), 180, 355, fill=(111, 74, 104))
    draw.polygon([(166, 43), (154, 89), (174, 78), (188, 48)], fill=(111, 74, 104))
    draw.polygon([(216, 40), (242, 85), (221, 82), (205, 47)], fill=(111, 74, 104))
    draw.ellipse((184, 53, 192, 65), fill=(58, 57, 83))
    draw.ellipse((211, 53, 219, 65), fill=(58, 57, 83))
    draw.ellipse((186, 55, 189, 59), fill=(255, 255, 255))
    draw.ellipse((213, 55, 216, 59), fill=(255, 255, 255))
    draw.arc((193, 62, 211, 82), 8, 170, fill=(211, 102, 132), width=3)
    draw.ellipse((175, 67, 185, 74), fill=(255, 163, 174))
    draw.ellipse((219, 67, 229, 74), fill=(255, 163, 174))
    draw.ellipse((220, 15, 242, 37), fill=(255, 133, 174))
    draw.ellipse((231, 7, 250, 28), fill=(255, 187, 204))
    draw.polygon([(225, 24), (244, 35), (227, 44)], fill=(255, 117, 164))
    # Cat friend
    draw.ellipse((287, 45, 399, 127), fill=(255, 255, 246), outline=(153, 148, 215), width=4)
    draw.polygon([(296, 65), (301, 27), (331, 55)], fill=(255, 255, 246), outline=(153, 148, 215))
    draw.polygon([(359, 56), (386, 24), (390, 69)], fill=(255, 255, 246), outline=(153, 148, 215))
    draw.ellipse((316, 76, 326, 90), fill=(57, 61, 91))
    draw.ellipse((359, 76, 369, 90), fill=(57, 61, 91))
    draw.polygon([(337, 91), (348, 91), (343, 98)], fill=(240, 144, 166))
    draw.arc((324, 91, 361, 116), 5, 175, fill=(158, 111, 144), width=3)
    draw.ellipse((304, 93, 317, 101), fill=(255, 190, 204))
    draw.ellipse((369, 93, 382, 101), fill=(255, 190, 204))
    image.save(path, optimize=True)


def ensure_assets(folder: str | Path) -> None:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    required = ("splash_mascot.gif", "app_icon.ico", "mascot_banner.png")
    if all((folder / name).exists() for name in required):
        return

    frames = []
    for frame in range(16):
        image = Image.new("RGBA", (512, 512), (255, 250, 247, 255))
        draw = ImageDraw.Draw(image)
        bob = round(5 * math.sin(frame * math.pi / 8))
        draw.ellipse((125, 420, 387, 465), fill=(230, 211, 204, 120))
        draw.line((256, 95 + bob, 256, 66 + bob), fill=(93, 173, 145, 255), width=9)
        draw.ellipse((241, 45 + bob, 271, 75 + bob), fill=(255, 207, 102, 255),
                     outline=(235, 160, 95, 255), width=4)
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
    icon.save(folder / "app_icon.ico",
              sizes=[(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
    banner_path = folder / "mascot_banner.png"
    if not banner_path.exists():
        _make_banner(banner_path)
