import shutil
from pathlib import Path
from app.skills import SkillManager

def test_selection_does_not_load_body_and_reloads_version_consistently(tmp_path,monkeypatch):
    root=tmp_path/'skills'; shutil.copytree('app/catalog',root)
    original=Path.read_text
    def deny_body(self,*args,**kwargs): raise AssertionError('Body loaded before activation')
    monkeypatch.setattr(Path,'read_text',deny_body)
    manager=SkillManager(root)
    selected=manager.select('退订单 O1001',{'intent':'refund'})
    assert selected.skill_name=='refund-processing' and selected.confidence==1
    assert manager.select('退款',{'intent':'other'}).skill_name is None
    monkeypatch.setattr(Path,'read_text',original)
    first=manager.load(selected.skill_name)
    path=root/'refund-processing/SKILL.md'
    next_version = first['version'] + '-test'
    path.write_text(path.read_text(encoding='utf-8').replace(f'"{first["version"]}"', f'"{next_version}"'),encoding='utf-8')
    second=manager.load(selected.skill_name)
    assert second['version']==next_version and first['hash']!=second['hash']
