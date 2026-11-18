import re
from decimal import Decimal

from app.memory.validator import normalized

AMOUNT = r'([0-9]{1,6}(?:\.[0-9]{1,2})?|[零一二两三四五六七八九十百千]{1,9})(?![0-9.,零一二两三四五六七八九十百千])'
BUDGET = re.compile(r'预算\s*(?:(?:一般|通常|最高|可以|控制在|不超过|不高于|上限|最多|到|为|约|就|在|是)\s*){0,6}' + AMOUNT + r'\s*(?:元|块)?')
CURRENT_LIMIT = re.compile(AMOUNT + r'\s*(?:元|块)\s*(?:以内|以下|封顶)')
DIGITS = {char: value for value, char in enumerate('零一二三四五六七八九')} | {'两': 2}
UNITS = {'十': 10, '百': 100, '千': 1000}


def ceiling(text):
    text = normalized(text)
    match = BUDGET.search(text) or CURRENT_LIMIT.search(text)
    if not match or re.match(r'[\s]*(?:到|至|-|~|左右|以上|起|万|亿)', text[match.end():]):
        return None
    amount = match[1]
    if amount[0].isascii():
        return int(Decimal(amount) * 100)
    value, digit, last_unit = 0, 0, 10000
    for char in amount:
        if char in DIGITS:
            if digit:
                return None
            digit = DIGITS[char]
        else:
            unit = UNITS[char]
            if unit >= last_unit or (not digit and char != '十'):
                return None
            value += (digit or 1) * unit
            digit, last_unit = 0, unit
    if digit and last_unit > 10 and last_unit < 10000 and '零' not in amount:
        return None
    return (value + digit) * 100


def budget_for(query, selected):
    current = ceiling(query)
    if current is not None:
        return current
    if '耳机' in query:
        for item in selected:
            if item.subject_key == 'headphones.budget' and item.memory_type == 'stable_constraint':
                return ceiling(item.content)
    return None
