"""Acceptance record, not a quality experiment. No audio/features/logits are persisted.

Run from the repository root with the locked uv environment; see RESULT.md.
The input is the existing JSON array manifest with paths relative to its directory.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from collections import Counter, defaultdict
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf

from stt import eval as official_eval
from stt import onnx_runner, text
from stt.decode import greedy
from stt.onnx_runner import ReleaseMoth
from stt.release import validate_release
from stt.text import normalize

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RELEASE_RECORD = 'runs/E-024/release-0.7.0.json'


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True) + '\n', encoding='utf-8')


def write_jsonl(path, rows):
    path.write_text(''.join(json.dumps(row, ensure_ascii=True) + '\n' for row in rows),
                    encoding='utf-8')


def portable_path(path):
    """Only explicit path metadata is converted, never transcript/JSON string contents."""
    resolved = Path(path).resolve()
    if not resolved.is_relative_to(ROOT.parent):
        raise ValueError('benchmark inputs must be in the repository or sibling directories')
    return Path(os.path.relpath(resolved, ROOT)).as_posix()


def portable_log(log):
    """Sanitize tool diagnostics only, replacing known project roots with relative roots."""
    for root, replacement in ((ROOT, '.'), (ROOT.parent, '..')):
        for spelling in (str(root), root.as_posix()):
            log = log.replace(spelling, replacement)
    return log


def portable_metrics(metrics):
    # The official scorer's Path strings use native separators; touch path fields only.
    for key in ('manifest', 'hyp', 'config_path'):
        if metrics.get(key) is not None:
            metrics[key] = portable_path(metrics[key])
    if metrics.get('vs') and metrics['vs'].get('other') is not None:
        metrics['vs']['other'] = portable_path(metrics['vs']['other'])
    return metrics


def check_original_release(config, report, record):
    if config['version'] != record['release']:
        raise ValueError('release version differs from committed E-024 record')
    if config['decode'] != record['decode'] or config['vad'] != record['vad']:
        raise ValueError('release settings differ from committed E-024 record')
    actual = {name: {key: entry[key] for key in ('bytes', 'sha256')}
              for name, entry in report['files'].items()}
    if actual != record['files']:
        raise ValueError('release files differ from committed E-024 hashes/sizes')


def scoring_row(row):
    lang = 'en' if row['group'] == 'en_us' else 'it'
    reference = row.get('raw_reference', row['reference'])
    # Mixed text uses the official Italian fallback; no unvalidated English-span offsets/ETA.
    return {'id': row['id'], 'norm': normalize(reference, lang), 'lang': lang,
            'category': row['group'], 'en_spans': [], 'source': 'reused_fleurs_acceptance'}


def summarize_timings(rows, mode):
    seconds = sum(row['seconds'] for row in rows)
    total = sum(row[mode]['total_seconds'] for row in rows)
    stages = defaultdict(float)
    for row in rows:
        for key, value in row[mode]['stages'].items():
            stages[key] += value
    return {'clips': len(rows), 'audio_seconds': seconds, 'total_seconds': total,
            'rtf': total / seconds, 'realtime_factor_x': seconds / total,
            'stage_seconds': dict(stages),
            'other_seconds': total - sum(stages.values())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--out', type=Path, default=HERE)
    parser.add_argument('--parity-clips', type=int, default=10,
                        help='round-robin IT/EN/spliced real clips; default 10')
    args = parser.parse_args()
    if Path.cwd().resolve() != ROOT:
        parser.error('run from the repository root so official metric paths remain relative')
    started = time.perf_counter()
    out = args.out.resolve()
    if not out.is_relative_to(HERE):
        parser.error('output must stay within the owned run directory')
    out.mkdir(parents=True, exist_ok=True)
    source = args.manifest.resolve()
    release_path = portable_path(args.release)
    source_path = portable_path(source)
    out_path = portable_path(out)
    record_bytes = subprocess.check_output(['git', 'show', f'HEAD:{RELEASE_RECORD}'], cwd=ROOT)
    record = json.loads(record_bytes)
    record_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT,
                                           text=True).strip()
    rows = json.loads(source.read_text(encoding='utf-8'))
    expected = {'it_it': 40, 'en_us': 40, 'spliced_it_en': 20, 'nonspeech': 4}
    if Counter(row['group'] for row in rows) != expected:
        raise ValueError(f'expected reused 40 IT + 40 EN + 20 spliced + 4 controls: {expected}')
    if len({row['id'] for row in rows}) != len(rows):
        raise ValueError('duplicate input IDs')
    if not 0 <= args.parity_clips <= 100:
        parser.error('--parity-clips must be between 0 and 100')
    groups = [[row for row in rows if row['group'] == group]
              for group in ('it_it', 'en_us', 'spliced_it_en')]
    interleaved = [row for i in range(40) for group in groups for row in group[i:i + 1]]
    paired_ids = {row['id'] for row in interleaved[:args.parity_clips]}
    modules = ['stt/onnx_runner.py', 'stt/release.py', 'stt/decode.py', 'stt/lm.py',
               'stt/features.py', 'stt/vad.py', 'stt/text.py', 'stt/eval.py',
               'web/decode.js', 'web/lm.js', 'web/package.json', 'tests/test_decoder_parity.py']
    code_before = {name: sha256(ROOT / name) for name in modules}
    load_start = time.perf_counter()
    runtime = ReleaseMoth(args.release.resolve())
    load_seconds = time.perf_counter() - load_start
    release_before = runtime.report
    config = runtime.config
    check_original_release(config, release_before, record)
    write_jsonl(out / 'manifest.official.jsonl', [scoring_row(row) for row in rows])
    node = subprocess.Popen(['node', str(HERE / 'decode.mjs'), str(args.release.resolve())],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            text=True, encoding='utf-8', bufsize=1)
    try:
        ready = json.loads(node.stdout.readline())
        assert ready['ready']
        timings, parity, vad_hyps, baseline_hyps, audio_hashes = [], [], [], [], {}
        # Wrap calls only to measure the actual ReleaseMoth path; no alternate VAD implementation.
        stages = defaultdict(float)
        observations = {}
        captured = []

        def timed(name, function):
            def call(*values, **kwargs):
                begin = time.perf_counter()
                result = function(*values, **kwargs)
                stages[name] += time.perf_counter() - begin
                if name == 'vad':
                    observations['speech_fraction'] = float((result > 0.5).mean())
                elif name == 'segmentation':
                    observations['detected_bounds_samples'] = result
                elif name == 'acoustic' and observations.get('capture'):
                    captured.append(result)
                return result
            return call

        with ExitStack() as stack:
            for name, attribute in [('vad', 'speech_probabilities'),
                                    ('segmentation', 'segment_bounds'), ('mel', 'log_mel'),
                                    ('decoder', 'beam_search'), ('decoder', 'greedy')]:
                stack.enter_context(patch.object(onnx_runner, attribute,
                                    timed(name, getattr(onnx_runner, attribute))))
            stack.enter_context(patch.object(runtime.model, 'log_probs',
                                timed('acoustic', runtime.model.log_probs)))
            for index, row in enumerate(rows):
                audio_path = source.parent / Path(row['path'].replace('\\', '/'))
                read_start = time.perf_counter()
                digest = sha256(audio_path)
                if row.get('sha256') and row['sha256'] != digest:
                    raise ValueError(f'audio hash mismatch: {row["id"]}')
                audio, rate = sf.read(audio_path, dtype='float32')
                if rate != 16000 or audio.ndim != 1 or not 560 <= audio.size <= 480000:
                    raise ValueError(f'expected mono 16 kHz, <=30 seconds: {audio_path}')
                if not np.isfinite(audio).all() or np.max(np.abs(audio)) > 1:
                    raise ValueError(f'invalid PCM: {audio_path}')
                audio_hashes[row['id']] = digest
                item = {'id': row['id'], 'group': row['group'], 'seconds': audio.size / rate,
                        'read_hash_seconds': time.perf_counter() - read_start}
                stages.clear()
                observations.clear()
                begin = time.perf_counter()
                hyp, reported = runtime.transcribe(audio, sample_rate=rate, segment=True)
                item['vad'] = {'total_seconds': time.perf_counter() - begin,
                               'runtime_reported_seconds': reported, 'stages': dict(stages),
                               **observations}
                item['vad']['gated'] = observations['speech_fraction'] <= config['vad']['threshold']
                bounds = observations.get('detected_bounds_samples', [])
                item['vad']['decoded_spans'] = 0 if item['vad']['gated'] else max(1, len(bounds))
                vad_hyps.append({'id': row['id'], 'hyp': hyp})
                stages.clear()
                observations.clear()
                observations['capture'] = row['id'] in paired_ids
                captured.clear()
                begin = time.perf_counter()
                mel = onnx_runner.log_mel(audio, filters=runtime.filters)
                settings = config['decode']
                baseline, _ = runtime.model.transcribe(
                    mel, decoder=settings['default'], beam_width=settings['beam'],
                    top_k=settings['top_k'], lm=runtime.lm, lm_weight=settings['lm_weight'],
                    token_bonus=settings['token_bonus'])
                item['no_vad'] = {'total_seconds': time.perf_counter() - begin,
                                  'stages': dict(stages)}
                baseline_hyps.append({'id': row['id'], 'hyp': baseline})
                if row['id'] in paired_ids:
                    assert len(captured) == 1
                    logits = np.asarray(captured[0], dtype='<f4', order='C')
                    raw = logits.tobytes()
                    payload = {'id': row['id'], 'frames': logits.shape[0],
                               'float32_base64': base64.b64encode(raw).decode('ascii')}
                    ipc_start = time.perf_counter()
                    node.stdin.write(json.dumps(payload) + '\n')
                    node.stdin.flush()
                    actual = json.loads(node.stdout.readline())
                    py_start = time.perf_counter()
                    py_greedy = greedy(logits, logits.shape[0], runtime.model.pieces)
                    py_greedy_seconds = time.perf_counter() - py_start
                    parity.append({**actual, 'python_hyp': baseline, 'python_greedy': py_greedy,
                                   'beam_exact': baseline == actual['hyp'],
                                   'greedy_exact': py_greedy == actual['greedy'],
                                   'shape': list(logits.shape),
                                   'logits_sha256': hashlib.sha256(raw).hexdigest(),
                                   'ipc_and_python_greedy_seconds': time.perf_counter() - ipc_start,
                                   'python_greedy_seconds': py_greedy_seconds,
                                   'python_beam_seconds': item['no_vad']['stages']['decoder']})
                timings.append(item)
                write_jsonl(out / 'hyp.vad.jsonl', vad_hyps)
                write_jsonl(out / 'hyp.no_vad.jsonl', baseline_hyps)
                print(f'{index + 1}/104 {row["id"]} '
                      f'vad={item["vad"]["total_seconds"]:.3f}s '
                      f'baseline={item["no_vad"]["total_seconds"]:.3f}s', flush=True)
    finally:
        node.stdin.close()
        node.wait(timeout=30)
    if node.returncode:
        raise RuntimeError(f'Node failed: {node.returncode}')
    manifest = portable_path(out / 'manifest.official.jsonl')
    for mode in ('no_vad', 'vad'):
        command = ['--manifest', manifest, '--hyp', portable_path(out / f'hyp.{mode}.jsonl'),
                   '--out', portable_path(out / f'metrics.{mode}.json'), '--breakdown', 'category']
        if mode == 'vad':
            command += ['--vs', portable_path(out / 'metrics.no_vad.json')]
        assert official_eval.main(command) == 0
        metric_path = out / f'metrics.{mode}.json'
        write_json(metric_path, portable_metrics(json.loads(metric_path.read_text(encoding='utf-8'))))
    tests_start = time.perf_counter()
    tests = subprocess.run([sys.executable, '-m', 'pytest', '-q',
                            'tests/test_decoder_parity.py', portable_path(HERE / 'test_benchmark.py')],
                           cwd=ROOT, capture_output=True, text=True, check=False)
    (out / 'validation-tests.txt').write_text(portable_log(tests.stdout + tests.stderr),
                                            encoding='utf-8')
    _, release_after = validate_release(args.release)
    check_original_release(config, release_after, record)
    code_after = {name: sha256(ROOT / name) for name in modules}
    metrics = {mode: json.loads((out / f'metrics.{mode}.json').read_text(encoding='utf-8'))
               for mode in ('vad', 'no_vad')}
    report = {
        'purpose': 'runtime acceptance only; no training or quality optimization',
        'command': [portable_path(sys.executable),
                    portable_path(HERE / 'benchmark.py'), '--release', release_path,
                    '--manifest', source_path, '--out', out_path,
                    '--parity-clips', str(args.parity_clips)],
        'release': {**release_before, 'release': release_path},
        'original_release_pin': {'record': RELEASE_RECORD, 'git_commit': record_commit,
                                 'record_sha256': hashlib.sha256(record_bytes).hexdigest(),
                                 'version': record['release'], 'all_seven_files_match': True},
        'release_files_unchanged': release_before['files'] == release_after['files'],
        'config': config, 'input_manifest_sha256': sha256(source),
        'audio_sha256': audio_hashes, 'code_sha256': code_before,
        'code_unchanged_during_run': code_before == code_after,
        'normalizer': 'stt.text.normalize', 'normalizer_sha256': sha256(Path(text.__file__)),
        'scorer': 'stt.eval.main', 'scorer_sha256': sha256(Path(official_eval.__file__)),
        'host': {'platform': platform.platform(), 'processor': platform.processor(),
                 'logical_cpu_count': os.cpu_count(), 'python': sys.version,
                 'node': subprocess.check_output(['node', '--version'], text=True).strip(),
                 'versions': {name: importlib.metadata.version(name)
                              for name in ('numpy', 'onnxruntime', 'soundfile', 'num2words')},
                 'provider': runtime.model.provider,
                 'ort_intra_op_threads': runtime.model.session.get_session_options().intra_op_num_threads,
                 'thread_environment': {key: os.environ.get(key) for key in
                                        ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')}},
        'timing': {'python_release_load_seconds': load_seconds, 'node': ready,
                   'vad': summarize_timings(timings, 'vad'),
                   'no_vad': summarize_timings(timings, 'no_vad'),
                   'per_clip': timings, 'validation_tests_seconds': time.perf_counter() - tests_start,
                   'whole_benchmark_seconds': time.perf_counter() - started},
        'parity': {'clips': parity, 'n': len(parity),
                   'beam_exact': sum(row['beam_exact'] for row in parity),
                   'greedy_exact': sum(row['greedy_exact'] for row in parity)},
        'tests': {'returncode': tests.returncode, 'synthetic_paired_fixtures': 459,
                  'log': 'validation-tests.txt'},
        'limitations': [
            'Reused known public FLEURS sample; not an independent quality test.',
            'Spliced clips reuse constituent IT/EN audio; samples are correlated.',
            'No natural code-switching set or validated en_spans; ETA is null, no natural ETA claim.',
            'Mixed numeric text uses official Italian fallback, not language-aware span normalization.',
            'Four existing synthetic silence/white-noise controls only; no real-world nonspeech claim.',
            'JS decoder only on native ONNX float32 outputs; no browser/WASM acoustic or JS mel parity.',
            'Serial VAD then baseline, no warmup, default ORT threads, shared host; timings not isolated.',
            'RTF excludes asset loading, file IO/hash, scoring, Node IPC and validation tests.',
            'Per-utterance bootstrap is descriptive, not independent-speaker inference.',
        ],
    }
    report['acceptance_pass'] = (report['original_release_pin']['all_seven_files_match']
                                 and report['release_files_unchanged']
                                 and report['code_unchanged_during_run']
                                 and tests.returncode == 0 and bool(parity)
                                 and all(row['beam_exact'] and row['greedy_exact'] for row in parity)
                                 and metrics['vad']['noise']['total_words'] == 0)
    write_json(out / 'report.json', report)
    lines = ['# Runtime Acceptance: 2026-11-10', '',
             (f'Acceptance checks: **{"PASS" if report["acceptance_pass"] else "FAIL"}**. '
              'Original release 0.7.0, unchanged assets; no training or tuning.'), '',
             ('Version, settings and all seven file hashes/sizes match the committed '
              '`runs/E-024/release-0.7.0.json` record; pin commit and record hash in `report.json`.'), '',
             '| Path | Clips | WER | Total runtime s | RTF | Realtime x |',
             '|---|---:|---:|---:|---:|---:|']
    for mode in ('vad', 'no_vad'):
        timing = report['timing'][mode]
        lines.append(f'| {mode} | 104 | {metrics[mode]["overall"]["wer"]:.4%} | '
                     f'{timing["total_seconds"]:.3f} | {timing["rtf"]:.4f} | '
                     f'{timing["realtime_factor_x"]:.2f} |')
    lines += ['', (f'Audio: {report["timing"]["vad"]["audio_seconds"]:.2f}s. '
                   f'Python asset load: {load_seconds:.3f}s. No warmup.'), '',
              (f'Identical-logit JS/Python: beam {report["parity"]["beam_exact"]}/{len(parity)}, '
              f'greedy {report["parity"]["greedy_exact"]}/{len(parity)} exact raw transcripts. '
              'Round-robin first IT/EN/spliced clips; real binary LM, canonical Python mel, no VAD.'), '',
              ('Existing 459 paired synthetic decoder fixtures and benchmark self-checks: '
              f'{"PASS" if tests.returncode == 0 else "FAIL"} (`validation-tests.txt`).'), '',
              (f'VAD synthetic controls: {metrics["vad"]["noise"]["total_words"]} words / 4 clips. '
              f'No-VAD controls: {metrics["no_vad"]["noise"]["total_words"]} words.'), '',
              f'Model SHA256: `{release_before["files"][config["model"]["file"]]["sha256"]}`.',
              f'Normalizer SHA256: `{report["normalizer_sha256"]}`.', '',
              '## Stage Timing', '', '| Path | Stage | Seconds |', '|---|---|---:|']
    for mode in ('vad', 'no_vad'):
        for stage, seconds in report['timing'][mode]['stage_seconds'].items():
            lines.append(f'| {mode} | {stage} | {seconds:.3f} |')
    segmented = sum(row['vad']['decoded_spans'] > 1 for row in timings)
    gated = sum(row['vad']['gated'] for row in timings)
    delta = metrics['vad']['vs']
    lines += ['', f'Segmented speech clips: {segmented}; gated clips: {gated}.', '',
              (f'VAD minus no-VAD WER: {100 * delta["wer_diff"]:+.3f} percentage points, '
               f'paired 95% CI [{100 * delta["wer_diff_ci"][0]:+.3f}, '
               f'{100 * delta["wer_diff_ci"][1]:+.3f}]. No quality improvement demonstrated.'), '',
              '## Reproduce', '', 'From the repository root in PowerShell:', '',
              '```powershell',
              'uv sync --locked --all-extras',
              ('uv run --locked --all-extras python runs/runtime-2026-11-10/benchmark.py --release ../moth-stt '
              '--manifest ../moth-stt-experiments/evaluation/manifest.json'), '```', '',
              ('Requires Python 3.12+, Node, NumPy, ONNX Runtime, soundfile, num2words and pytest. '
              'uv sync installs locked dependencies. The sibling manifest/audio and original release '
              'are prerequisites; the benchmark does not download assets. '
              'Hypotheses and official normalized reference JSONL are retained; no audio or logits.'), '',
              ('Official `stt.eval.main` scores normalized raw references for mono-language clips and '
              'the existing reference for spliced clips. All `en_spans` are empty: ETA is null. '
              'See `metrics.vad.json` for paired bootstrap versus the fresh baseline, '
              '`report.json` for per-stage/per-clip timing, source/asset/audio hashes and environment.'), '',
              '## Limitations', '']
    lines += ['- ' + limitation for limitation in report['limitations']]
    (out / 'RESULT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({'acceptance_pass': report['acceptance_pass'],
                      'vad_wer': metrics['vad']['overall']['wer'],
                      'no_vad_wer': metrics['no_vad']['overall']['wer']}), flush=True)
    return 0 if report['acceptance_pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
