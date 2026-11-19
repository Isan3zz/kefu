import os

import pytest

from app.config import settings
from app.business import DomainError
from scripts.multimodal_smoke import external_smoke, local_smoke


@pytest.mark.models
def test_MM28_real_local_model():
    if os.getenv('RUN_MM_LOCAL_MODEL') != '1':
        pytest.skip('Set RUN_MM_LOCAL_MODEL=1 and install the local Chinese-CLIP snapshot')
    from app.multimodal.encoder import ChineseCLIPEncoder
    try:
        assert ChineseCLIPEncoder(settings.chinese_clip_model_path).revision
    except DomainError as error:
        pytest.skip(error.message)
    result = local_smoke()
    assert result['indexed_images'] == result['indexed_texts'] == 3
    assert result['missing_files'] == result['orphan_vectors'] == 0


@pytest.mark.real
async def test_MM29_real_external_vlm():
    if os.getenv('RUN_MM_EXTERNAL_VLM') != '1':
        pytest.skip('Set RUN_MM_EXTERNAL_VLM=1 to send generated fixtures to the configured provider')
    if not all((settings.multimodal_external_vlm_images_enabled, settings.vlm_model, settings.llm_base_url,
                settings.llm_api_key, settings.vlm_supports_vision, settings.vlm_supports_tool_calling)):
        pytest.skip('Explicit VLM endpoint, credentials, capabilities and consent settings required')
    result = await external_smoke()
    assert result['image_count'] == 2 and result['native_tool_call']
