"""Offline Chinese-CLIP inference in a shared normalized 512-dimensional space."""
from functools import lru_cache
import json
import logging
from pathlib import Path
import re
from threading import RLock
from time import perf_counter
from typing import Protocol

import numpy as np
from PIL import Image

from app.business import DomainError
from app.config import settings
from app.observability import emit


REPO_ID = 'OFA-Sys/chinese-clip-vit-base-patch16'
DIMENSION = 512
PREPROCESSING = 'rgb-exif-jpeg-clip-v1'


class MultimodalEncoder(Protocol):
    @property
    def revision(self) -> str: ...
    def encode_images(self, images: list[Image.Image]) -> np.ndarray: ...
    def encode_texts(self, texts: list[str]) -> np.ndarray: ...


def normalized(values, count):
    array = np.asarray(values, dtype=np.float32)
    if array.shape != (count, DIMENSION) or not np.isfinite(array).all():
        raise DomainError('MULTIMODAL_ENCODER_ERROR', '向量必须是有限的 512 维数值')
    norm = np.linalg.norm(array.astype(np.float64), axis=1, keepdims=True)
    if (norm <= 1e-12).any():
        raise DomainError('MULTIMODAL_ENCODER_ERROR', '模型返回零向量')
    return (array / norm).astype(np.float32)


class ChineseCLIPEncoder:
    def __init__(self, path=None, device=None, batch_size=None):
        self.path = Path(path or settings.chinese_clip_model_path).resolve()
        self.device = device or settings.chinese_clip_device
        self.batch_size = batch_size or settings.chinese_clip_batch_size
        self.lock = RLock()
        self.model = self.processor = None
        self.loaded_revision = None

    @property
    def revision(self):
        try:
            manifest = json.loads((self.path/'model_manifest.json').read_text(encoding='utf-8'))
            revision = manifest['revision']
            if (manifest['repo_id'] != REPO_ID or not re.fullmatch(r'[a-f0-9]{40}', revision)
                    or not (self.path/'config.json').is_file()
                    or not any((self.path/name).is_file() for name in
                               ('model.safetensors', 'pytorch_model.bin'))):
                raise ValueError
            return revision
        except (OSError, ValueError, KeyError, TypeError):
            raise DomainError('MULTIMODAL_MODEL_NOT_READY',
                              '缺少有效的本地 Chinese-CLIP，请运行 python scripts/download_chinese_clip.py') from None

    def load(self):
        revision = self.revision
        if self.model is not None:
            if self.loaded_revision != revision:
                raise DomainError('MULTIMODAL_MODEL_NOT_READY', '模型版本已更换，请重建两个多模态索引并重启服务')
            return
        try:
            import torch
            from transformers import ChineseCLIPModel, ChineseCLIPProcessor
            device = ('cuda' if torch.cuda.is_available() else 'cpu') if self.device == 'auto' else self.device
            model = ChineseCLIPModel.from_pretrained(str(self.path), local_files_only=True)
            if model.config.projection_dim != DIMENSION:
                raise ValueError('Unexpected Chinese-CLIP dimension')
            processor = ChineseCLIPProcessor.from_pretrained(str(self.path), local_files_only=True, backend='pil')
            self.model, self.processor = model.to(device).eval(), processor
        except Exception as error:
            logging.getLogger(__name__).warning('Chinese-CLIP load failed: %s', type(error).__name__)
            raise DomainError('MULTIMODAL_MODEL_NOT_READY',
                              '本地 Chinese-CLIP 加载失败，请检查完整权重和设备配置') from None
        self.loaded_revision = revision
        logging.getLogger(__name__).info('Chinese-CLIP path=%s revision=%s dimensions=%s device=%s',
                                        self.path, revision, DIMENSION, device)

    def encode_images(self, images):
        if not images or any(not isinstance(image, Image.Image) for image in images):
            raise DomainError('INVALID_ARGUMENT', '需要至少一张已解码图片')
        return self.encode(images, 'image')

    def encode_texts(self, texts):
        if not texts or any(not isinstance(text, str) or not text.strip() for text in texts):
            raise DomainError('INVALID_ARGUMENT', '需要至少一条非空文本')
        return self.encode(texts, 'text')

    def encode(self, values, modality):
        started = perf_counter()
        with self.lock:
            self.load()
            import torch
            batches = []
            try:
                with torch.inference_mode():
                    for offset in range(0, len(values), self.batch_size):
                        batch = values[offset:offset+self.batch_size]
                        if modality == 'image':
                            inputs = self.processor(images=[image.convert('RGB') for image in batch],
                                                    return_tensors='pt').to(self.model.device)
                            features = self.model.get_image_features(**inputs)
                        else:
                            inputs = self.processor(text=batch, return_tensors='pt', padding=True, truncation=True,
                                                    max_length=min(64, self.model.config.text_config.max_position_embeddings)
                                                    ).to(self.model.device)
                            features = self.model.get_text_features(**inputs)
                        # Transformers 5 returns the projected vector in pooler_output.
                        tensor = features if isinstance(features, torch.Tensor) else features.pooler_output
                        batches.append(tensor.float().cpu().numpy())
                output = normalized(np.concatenate(batches), len(values))
            except Exception as error:
                logging.getLogger(__name__).warning('Chinese-CLIP inference failed: %s', type(error).__name__)
                raise DomainError('MULTIMODAL_ENCODER_ERROR',
                                  'Chinese-CLIP 编码失败，请检查输入或使用 CPU 设备') from None
        emit(f'multimodal.encode.{modality}', count=len(values), model_revision=self.loaded_revision,
             duration_ms=(perf_counter()-started)*1000)
        return output


@lru_cache(maxsize=4)
def local_encoder(path, device, batch_size):
    return ChineseCLIPEncoder(path, device, batch_size)
