"""One bounded, orientation-corrected image shared by every visual branch."""
from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
import warnings

from PIL import Image, ImageOps, UnidentifiedImageError

from app.schemas import ImageInputMetadata
from app.services.vision.demo import InvalidImageError

MAX_IMAGE_PIXELS = 24_000_000
MAX_IMAGE_SIDE = 12_000


@dataclass(frozen=True)
class PreparedImage:
    content: bytes
    metadata: ImageInputMetadata


def prepare_image(payload: bytes) -> PreparedImage:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(payload)) as source:
                image_format = source.format
                if image_format not in {"JPEG", "PNG", "WEBP"}:
                    raise InvalidImageError("图片实际格式不受支持，请使用 JPEG、PNG 或 WebP")
                width, height = source.size
                if min(width, height) < 2:
                    raise InvalidImageError("图片尺寸过小，宽高均须至少为 2 像素")
                if width * height > MAX_IMAGE_PIXELS or max(width, height) > MAX_IMAGE_SIDE:
                    raise InvalidImageError("图片像素过多，请缩小至 2400 万像素以内，且单边不超过 12000 像素")
                source.verify()
            with Image.open(BytesIO(payload)) as source:
                if getattr(source, "n_frames", 1) != 1:
                    raise InvalidImageError("暂不支持动图，请选择一张静态图片或导出单帧后上传")
                orientation = source.getexif().get(274, 1)
                oriented = ImageOps.exif_transpose(source)
                alpha = "A" in oriented.getbands() or "transparency" in oriented.info
                if alpha:
                    rgba = oriented.convert("RGBA")
                    background = Image.new("RGBA", rgba.size, "white")
                    image = Image.alpha_composite(background, rgba).convert("RGB")
                else:
                    image = oriented.convert("RGB")
                image.info.clear()
                output = BytesIO()
                image.save(output, format="PNG")
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise InvalidImageError("图片像素过多，请缩小图片后重试") from exc
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
        if isinstance(exc, InvalidImageError):
            raise
        raise InvalidImageError("上传图片不完整或无法解析，请重新导出图片后重试") from exc
    return PreparedImage(
        content=output.getvalue(),
        metadata=ImageInputMetadata(
            sha256=sha256(payload).hexdigest(),
            original_format=image_format,
            original_width=width,
            original_height=height,
            orientation_corrected=orientation in {2, 3, 4, 5, 6, 7, 8},
            transparency_composited=alpha,
        ),
    )
