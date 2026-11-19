"""Stable descriptions for local, read-only tools exposed by the Agent Loop."""
from app.multimodal.retrieval import TOOL_ARGS


DESCRIPTIONS = {
    'search_products_by_image': 'Find catalog candidates similar to a current attachment image_id. Similarity is not identification. Use search_knowledge with a returned product_id for documentation.',
    'search_text_by_image': 'Find short catalog descriptions related to a current attachment image_id. Use search_knowledge for authoritative policies and manuals.',
    'search_images_by_text': 'Find catalog images matching a short visual description. Put explicit budgets in filters.max_price_cents, category or in_stock in filters as applicable.',
    'search_products_by_image_and_text': 'Find products similar to a current attachment image_id and visual description. Pass an explicit price ceiling in filters.max_price_cents (yuan × 100), stock/category in filters.',
}


def tool_schemas(names):
    return [{'type': 'function', 'function': {
        'name': name, 'description': DESCRIPTIONS[name], 'parameters': TOOL_ARGS[name].model_json_schema(),
    }} for name in names]
