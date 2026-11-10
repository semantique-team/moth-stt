"""Small checks for manifest adaptation and timing aggregation (no release assets)."""
import importlib.util
import json
from copy import deepcopy
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location('runtime_benchmark',
                                             Path(__file__).with_name('benchmark.py'))
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


def test_official_normalization_and_no_eta():
    row = benchmark.scoring_row({'id': 'en', 'group': 'en_us', 'reference': 'wrong',
                                 'raw_reference': 'Two dogs / 3 cats.'})
    assert row['norm'] == 'two dogs three cats'
    assert row['en_spans'] == []
    assert benchmark.scoring_row({'id': 'noise', 'group': 'nonspeech',
                                  'reference': ''})['norm'] == ''


def test_total_rtf_is_duration_weighted():
    rows = [{'seconds': 2, 'vad': {'total_seconds': 1, 'stages': {'vad': 0.1}}},
            {'seconds': 8, 'vad': {'total_seconds': 1, 'stages': {'vad': 0.2}}}]
    result = benchmark.summarize_timings(rows, 'vad')
    assert result['rtf'] == 0.2
    assert result['realtime_factor_x'] == 5
    assert result['stage_seconds']['vad'] == pytest.approx(0.3)


def test_portable_metadata_does_not_rewrite_transcripts(tmp_path):
    root = benchmark.ROOT
    assert benchmark.portable_path(root / 'runs' / 'file.json') == 'runs/file.json'
    assert benchmark.portable_path(root.parent / 'moth-stt') == '../moth-stt'
    with pytest.raises(ValueError, match='repository or sibling'):
        benchmark.portable_path(root.anchor)
    transcript = str(root / 'this is literal transcript text')
    output = tmp_path / 'hyp.jsonl'
    benchmark.write_jsonl(output, [{'id': 'clip', 'hyp': transcript}])
    assert json.loads(output.read_text())['hyp'] == transcript
    assert benchmark.portable_log(f'{root}/tests: error') == './tests: error'
    assert benchmark.portable_log(f'{root.parent}/moth-stt: error') == '../moth-stt: error'
    assert benchmark.portable_log('unrelated /other/project') == 'unrelated /other/project'


def test_original_release_pin_rejects_self_consistent_replacement():
    record = json.loads((benchmark.ROOT / benchmark.RELEASE_RECORD).read_text())
    config = {'version': record['release'], 'decode': record['decode'], 'vad': record['vad']}
    report = {'files': deepcopy(record['files'])}
    benchmark.check_original_release(config, report, record)
    for field in ('sha256', 'bytes'):
        changed = deepcopy(report)
        changed['files']['model.int8.onnx'][field] = 'replacement'
        with pytest.raises(ValueError, match='committed E-024'):
            benchmark.check_original_release(config, changed, record)
    with pytest.raises(ValueError, match='version'):
        benchmark.check_original_release({**config, 'version': 'other'}, report, record)
    with pytest.raises(ValueError, match='settings'):
        benchmark.check_original_release({**config, 'decode': {}}, report, record)


def test_metric_sanitization_only_changes_known_path_fields():
    literal = str(benchmark.ROOT / 'literal transcript')
    metrics = {'manifest': str(benchmark.HERE / 'manifest.official.jsonl'),
               'hyp': str(benchmark.HERE / 'hyp.vad.jsonl'), 'config_path': None,
               'vs': {'other': str(benchmark.HERE / 'hyp.no_vad.jsonl')},
               'overall': {'eta_errors': [{'got': literal}]}}
    result = benchmark.portable_metrics(metrics)
    assert result['manifest'] == 'runs/runtime-2026-11-10/manifest.official.jsonl'
    assert result['vs']['other'] == 'runs/runtime-2026-11-10/hyp.no_vad.jsonl'
    assert result['config_path'] is None
    assert result['overall']['eta_errors'][0]['got'] == literal
