from io import BytesIO

from PIL import Image, ImageDraw

from app.schemas import DemoCaseInfo


CASES = {
    "synthetic-fire": DemoCaseInfo(
        case_id="synthetic-fire",
        name="合成明火候选案例",
        description="用于演示火焰候选区域、检测框和证据链，不代表真实工业现场。",
        expected_signal="fire",
    ),
    "synthetic-smoke": DemoCaseInfo(
        case_id="synthetic-smoke",
        name="合成烟雾候选案例",
        description="用于演示烟雾候选区域和不确定性提示，不代表真实工业现场。",
        expected_signal="smoke",
    ),
    "synthetic-clear": DemoCaseInfo(
        case_id="synthetic-clear",
        name="合成无明确证据案例",
        description="用于演示未检出不等于安全以及人工复核边界。",
        expected_signal="no_detection",
    ),
}


def render_demo_case(case_id: str) -> bytes:
    if case_id not in CASES:
        raise KeyError(case_id)
    image = Image.new("RGB", (720, 460), color=(22, 34, 49))
    draw = ImageDraw.Draw(image)
    draw.rectangle((70, 75, 650, 400), fill=(35, 53, 70), outline=(70, 105, 130), width=3)
    draw.rectangle((105, 280, 615, 370), fill=(50, 67, 82))
    if case_id == "synthetic-fire":
        draw.rectangle((300, 200, 420, 355), fill=(228, 82, 24))
        draw.ellipse((322, 120, 398, 270), fill=(244, 137, 35))
    elif case_id == "synthetic-smoke":
        for box, gray in [((285, 190, 440, 315), 135), ((315, 125, 465, 245), 150), ((360, 80, 500, 190), 165)]:
            draw.ellipse(box, fill=(gray, gray, gray))
    else:
        draw.rectangle((280, 210, 440, 350), fill=(28, 100, 145))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()

