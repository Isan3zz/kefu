import hashlib
from types import SimpleNamespace
import httpx
import pytest

def test_range_download_verifies_hash_and_reuses_completed_file(tmp_path):
    from scripts.download_models import download_ranges
    data=b'local-model-test-bytes'; calls=[]
    metadata=SimpleNamespace(size=len(data),etag=hashlib.sha256(data).hexdigest(),location='http://download.local/weights')
    def server(request):
        bounds=request.headers['Range'].removeprefix('bytes='); start,end=map(int,bounds.split('-')); calls.append((start,end))
        return httpx.Response(206,headers={'Content-Range':f'bytes {start}-{end}/{len(data)}'},content=data[start:end+1])
    path=download_ranges('BAAI/test','pinned-revision','weights.bin',tmp_path,metadata,httpx.MockTransport(server),chunk_size=5,workers=2)
    assert path.read_bytes()==data and len(calls)==5
    download_ranges('BAAI/test','pinned-revision','weights.bin',tmp_path,metadata,httpx.MockTransport(server),chunk_size=5)
    assert len(calls)==5

def test_range_download_never_promotes_wrong_bytes(tmp_path):
    from scripts.download_models import download_ranges
    metadata=SimpleNamespace(size=4,etag=hashlib.sha256(b'good').hexdigest(),location='http://download.local/weights')
    def server(request): return httpx.Response(206,headers={'Content-Range':'bytes 0-3/4'},content=b'evil')
    with pytest.raises(ValueError,match='hash verification'):
        download_ranges('BAAI/test','revision','weights.bin',tmp_path,metadata,httpx.MockTransport(server),chunk_size=4)
    assert not (tmp_path/'weights.bin').exists()
