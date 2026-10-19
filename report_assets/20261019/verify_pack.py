"""Offline integrity, link, privacy-scope and headline-metric checks (stdlib only)."""
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlsplit, unquote
import csv
import hashlib
import json

ROOT = Path(__file__).resolve().parent


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        self.links.extend(v for k, v in attrs if k in ('href', 'src'))


def main():
    manifest = json.loads((ROOT / 'manifest.json').read_text(encoding='utf-8'))
    for item in manifest['files'] + manifest['derived_files']:
        path = ROOT / item['file']
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == item.get('packed_sha256', item.get('sha256')), item['file']
    checked_links = 0
    for path in ROOT.rglob('*.html'):
        parser = Links()
        parser.feed(path.read_text(encoding='utf-8'))
        for link in parser.links:
            parsed = urlsplit(link)
            if parsed.scheme or not parsed.path:
                continue
            assert (path.parent / unquote(parsed.path)).is_file(), (path, link)
            checked_links += 1
    for path in ROOT.rglob('*'):
        if not path.is_file():
            continue
        assert path.suffix.lower() not in ('.jpg', '.jpeg', '.pt', '.pth', '.npz', '.npy', '.joblib', '.zip')
        if path.suffix == '.csv':
            with path.open(encoding='utf-8-sig', newline='') as handle:
                header = next(csv.reader(handle))
            assert 'imageid' not in [h.lower() for h in header], path
    photo = json.loads((ROOT / 'photometric/results.json').read_text(encoding='utf-8'))['rows']
    ablation = json.loads((ROOT / 'data/final_ablation.json').read_text(encoding='utf-8'))['results']
    error = json.loads((ROOT / 'data/final_error_analysis.json').read_text(encoding='utf-8'))
    tables = json.loads((ROOT / 'tables/report_tables.json').read_text(encoding='utf-8'))['tables']
    for source, row in zip(photo, tables['PHOTO'][1:]):
        for key, col in [('mse', 1), ('mae', 2), ('tail_mse', 3)]:
            assert abs(float(row[col].replace(',', '')) - source[key]) <= 0.0051
    for source, row in zip(ablation, tables['ABLATION'][1:]):
        assert abs(float(row[1].replace(',', '')) - source['validation']['mse']) <= 0.0051
        assert abs(float(row[3].replace(',', '')) - source['meta_crossfit']['mse']) <= 0.0051
    assert error['overall']['n'] == 1601
    assert sum(row['n'] for row in error['price_bands']) == 1601
    assert 'case_selection' not in error
    assert abs(error['overall']['mse'] - ablation[-1]['validation']['mse']) < 1e-8
    print(json.dumps({'hashes_checked': len(manifest['files']) + len(manifest['derived_files']),
                      'local_links_checked': checked_links, 'metric_checks': 'passed',
                      'scope': 'No raw images, model arrays, or per-image CSV labels.'}, indent=2))


if __name__ == '__main__':
    main()
