import pytest
from app.rag import chunks

def test_long_chinese_paragraph_has_exact_overlap_without_duplicated_tail():
    text=''.join(chr(0x4e00+i) for i in range(1000))
    result=chunks(text)
    assert len(result)==3 and all(len(chunk)<=400 for chunk in result)
    assert all(previous[-60:]==following[:60] for previous,following in zip(result,result[1:]))
    assert result[0]+''.join(chunk[60:] for chunk in result[1:])==text

def test_chunker_prefers_paragraph_boundary_and_keeps_unicode():
    text='甲'*300+'\n\n'+'乙'*300
    result=chunks(text)
    assert result[0].endswith('\n\n')
    assert result[0]+''.join(chunk[60:] for chunk in result[1:])==text
    with pytest.raises(ValueError): chunks(text,size=60,overlap=60)
    near_overlap=chunks('甲\n\n乙\n\n'*20,size=10,overlap=9)
    assert near_overlap and all(len(part)<=10 for part in near_overlap)
