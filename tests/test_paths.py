from reproducibility.paths import ROOT, WORK, DATA, resolve_path

def test_historical_imports_data_and_cache_paths_resolve():
    assert resolve_path('/home/guillaume/github/needlefinder/NeedleFinder/NeedleFinder') == str(ROOT/'NeedleFinder')
    assert resolve_path('/home/guillaume/github/needlefinder/vault/sources/a.json') == str(ROOT/'vault/sources/a.json')
    assert resolve_path('/tmp/cases.pt') == str(WORK/'cases.pt')
    assert resolve_path('/home/guillaume/Dropbox/GYN Cases') == str(DATA/'GYN Cases')
    assert resolve_path('/tmp-other/file') == '/tmp-other/file'
