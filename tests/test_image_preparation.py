import asyncio
from hashlib import sha256
from io import BytesIO

from PIL import Image, ImageDraw
import httpx
import pytest

from app.main import app, agent
from app.services.images import prepare_image
from app.services.vision.demo import DemoColorDetector, InvalidImageError
from app.services.vision.firebench import UnavailableFireClassifier
from app.services.vision.fusion import MultimodelVisionDetector


def photo(orientation=1):
    image = Image.new("RGB", (120, 80), (10, 25, 40))
    ImageDraw.Draw(image).rectangle((10, 20, 49, 59), fill=(240, 90, 20))
    exif = Image.Exif()
    exif[274] = orientation
    exif[270] = "metadata must not reach model workers"
    output = BytesIO()
    image.save(output, format="JPEG", exif=exif, quality=95)
    return output.getvalue()


@pytest.mark.parametrize("orientation", range(1, 9))
def test_all_exif_orientations_and_metadata_removal(orientation):
    original = photo(orientation)
    prepared = prepare_image(original)
    with Image.open(BytesIO(prepared.content)) as image:
        assert image.size == ((80, 120) if orientation >= 5 else (120, 80))
        assert image.mode == "RGB"
        assert image.format == "PNG"
        assert not image.getexif()
        assert "exif" not in image.info
    assert prepared.metadata.sha256 == sha256(original).hexdigest()
    assert prepared.metadata.orientation_corrected is (orientation != 1)
    assert prepared.metadata.original_width == 120


def test_rotated_detection_uses_display_coordinates():
    result = asyncio.run(agent.analyze(photo(6), "phone.jpg", "核查火焰"))
    assert (result.vision.image_width, result.vision.image_height) == (80, 120)
    box = result.vision.detections[0].bbox
    assert 18 <= box.x1 <= 22 and 58 <= box.x2 <= 62
    assert 8 <= box.y1 <= 12 and 48 <= box.y2 <= 52
    assert result.vision.input_image.orientation_corrected


def test_visual_branches_receive_identical_normalized_pixels():
    class Detector(DemoColorDetector):
        async def detect(self, image_bytes, file_name):
            self.received = (image_bytes, file_name)
            return await super().detect(image_bytes, file_name)

    class Classifier(UnavailableFireClassifier):
        async def classify(self, image_bytes, file_name):
            self.received = (image_bytes, file_name)
            return await super().classify(image_bytes, file_name)

    detector = Detector()
    classifier = Classifier("test")
    prepared = prepare_image(photo(6))
    result = asyncio.run(MultimodelVisionDetector(detector, classifier).detect(prepared.content, "normalized.png"))
    assert detector.received == classifier.received == (prepared.content, "normalized.png")
    assert (result.image_width, result.image_height) == (80, 120)


def test_transparent_pixels_use_white_background_not_hidden_fire_color():
    output = BytesIO()
    Image.new("RGBA", (60, 40), (240, 90, 20, 0)).save(output, format="PNG")
    prepared = prepare_image(output.getvalue())
    with Image.open(BytesIO(prepared.content)) as image:
        assert image.getpixel((0, 0)) == (255, 255, 255)
    assert prepared.metadata.transparency_composited
    assert not asyncio.run(DemoColorDetector().detect(prepared.content, "image.png")).detections


@pytest.mark.parametrize("kind", ["broken", "wrong-format", "tiny", "animated", "oversized"])
def test_invalid_images_are_rejected_before_model_invocation(kind, monkeypatch):
    output = BytesIO()
    if kind == "broken":
        payload = b"not an image"
    elif kind == "animated":
        Image.new("RGB", (10, 10), "red").save(
            output, format="PNG", save_all=True,
            append_images=[Image.new("RGB", (10, 10), "blue")], duration=100, loop=0,
        )
        payload = output.getvalue()
    else:
        size = (1, 4) if kind == "tiny" else (20, 20)
        Image.new("RGB", size).save(output, format="BMP" if kind == "wrong-format" else "PNG")
        payload = output.getvalue()
        if kind == "oversized":
            monkeypatch.setattr("app.services.images.MAX_IMAGE_PIXELS", 100)

    async def no_model(*args, **kwargs):
        pytest.fail("Invalid image must not reach any model")

    monkeypatch.setattr(agent.detector, "detect", no_model)

    async def request():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            return await client.post("/api/v1/analyze", files={"image": ("image.png", payload, "image/png")})

    response = asyncio.run(request())
    assert response.status_code == 400
    assert response.json()["detail"]


def test_truncated_png_is_rejected():
    prepared = prepare_image(photo())
    with pytest.raises(InvalidImageError):
        prepare_image(prepared.content[:len(prepared.content) // 2])
