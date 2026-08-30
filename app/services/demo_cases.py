import base64
from io import BytesIO

from PIL import Image, ImageDraw

from app.schemas import DemoCaseInfo
from app.services.demo_assets import VERIFIED_FIRE_JPEG_B64, VERIFIED_SMOKE_JPEG_B64


CASES = {
    "verified-fire": DemoCaseInfo(
        case_id="verified-fire",
        name="开放授权明火案例",
        description="用于演示明火定位、多模型复核与证据链，不作为模型精度统计。",
        expected_signal="fire",
        synthetic=False,
        source_note="IFireSmoke固定验证样本 fire_002",
        license="CC-BY-4.0",
    ),
    "verified-smoke": DemoCaseInfo(
        case_id="verified-smoke",
        name="开放授权烟雾案例",
        description="用于演示烟雾定位、不确定性提示和人工复核边界，不作为模型精度统计。",
        expected_signal="smoke",
        synthetic=False,
        source_note="IFireSmoke固定验证样本 smoke_002",
        license="CC-BY-4.0",
    ),
    "synthetic-clear": DemoCaseInfo(
        case_id="synthetic-clear",
        name="合成无明确证据案例",
        description="用于演示未检出不等于安全以及人工复核边界。",
        expected_signal="no_detection",
    ),
}


def render_demo_case(case_id: str) -> bytes:
    legacy_case_ids = {"synthetic-fire", "synthetic-smoke"}
    if case_id not in CASES and case_id not in legacy_case_ids:
        raise KeyError(case_id)
    if case_id == "verified-fire":
        return base64.b64decode(VERIFIED_FIRE_JPEG_B64)
    if case_id == "verified-smoke":
        return base64.b64decode(VERIFIED_SMOKE_JPEG_B64)
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
