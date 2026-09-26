#!/usr/bin/env python3
"""Check the project page against the saved results.

    python3 site/tools/check_evidence.py            # before committing
    python3 site/tools/check_evidence.py --release  # before sending the link to anyone

Every number on the page comes from a file in results/. This script recomputes
the numbers from those files and fails if the page shows anything different,
redraws the charts to make sure they have not drifted, and checks the rules the
page follows: no corpus audio, the demo voice named as text-to-speech, nothing
presented as a live model. Standard library only.
"""
import html
import json
import pathlib
import re
import sys

SITE = pathlib.Path(__file__).resolve().parent.parent
RESULTS = SITE.parent / 'results'
HTML = (SITE / 'index.html').read_text(encoding='utf-8')
TEXT = re.sub(r'\s+', ' ', html.unescape(re.sub(r'<[^>]+>', ' ', re.sub(r'<svg.*?</svg>', ' ', HTML, flags=re.S))))


def load(name):
    return json.loads((RESULTS / name).read_text(encoding='utf-8'))


fails, count = [], 0


def check(name, ok, detail=''):
    global count
    count += 1
    if not ok:
        fails.append(f'{name}: {detail}')
    print(f'  {"PASS" if ok else "FAIL"}  {name}' + (f'  ({detail})' if detail and not ok else ''))


def shown(label, text):
    check(f'page shows {label}: {text}', text in TEXT, 'not found on the page')


def pct(x):
    return f'{100 * x:.2f}%'


print('-- numbers, recomputed from results/ --')
retrain = load('retraining.json')
runs = retrain['runs']
best = retrain['chosen_on_validation']
for name in ['zero_shot', 'checkpoint_2025', 'fixed', 'synthetic', best]:
    shown(f'{name} word error rate on the unseen clips', pct(runs[name]['headline_test']['wer']))
change = runs[best]['headline_test']['change_from_2025_points']
shown('the best model\'s change from 2025', f'{abs(change):.2f} points')
lo, hi = runs[best]['headline_test']['change_interval_95_points']
check('a gain whose interval crosses zero is not called certain',
      lo < 0 < hi and "can&rsquo;t yet call certain" in HTML or not lo < 0 < hi)

old = load('rescoring_2025/speech_recognition.json')['metrics']['beam']
shown('the 2025 headline', pct(old['wer']))
shown('its word edits', f"{old['word_errors']} word edits across {old['reference_words']:,} reference words")

splits = load('rescoring_2025/data_splits.json')
unseen = splits['asr_test_rows'] - splits['asr_test_clips_in_shared_sessions']
shown('training clips', f"{splits['asr_train_rows']} real training clips")
shown('overlapping test clips', f"{splits['asr_test_clips_in_shared_sessions']} of my {splits['asr_test_rows']} test clips")
shown('unseen test clips', f'{unseen} unseen clips')
check('the clean test set in results matches the split',
      all(runs[n]['test']['beam']['reference_words'] == runs['checkpoint_2025']['test']['beam']['reference_words']
          for n in runs))

agree = load('extraction_agreement.json')['models']
b, o, z = agree[best], agree['checkpoint_2025'], agree['zero_shot']
shown('commands right, best model', f"command right in {b['command']['clips_where_it_matches']} of the "
      f"{b['command']['clips_with_field']} clips")
shown('commands right, 2025', f"(the 2025 model: {o['command']['clips_where_it_matches']})")
shown('numbers right', f"numbers right in {b['value']['clips_where_it_matches']} of {b['value']['clips_with_field']}")
shown('callsigns right', f"{b['callsign']['clips_where_it_matches']} of {b['callsign']['clips_with_field']}")
core = 'callsign_command_value_identical'
shown('callsign, command and numbers all right',
      f"Callsign, command and numbers all matched in {b[core]} of the {b['clips']} clips (the 2025 model: "
      f"{o[core]}; unmodified Whisper-small: {z[core]})")
shown('every field right, with waypoints', f"with them, every field matched in only {b['all_fields_identical']}")
shown('Whisper-small soup on whole instructions', f"the Whisper-small soup got {agree['soup_fixed_real'][core]}")
check('the best model is not the top on whole instructions, as the page says',
      max(m[core] for m in agree.values()) > b[core])
shown('missed callsigns counted', f"every one of the {b['missed_callsigns']['count']} missed callsigns")
if (RESULTS / 'callsign_errors.json').exists():
    causes = load('callsign_errors.json')['models'][best]
    check('callsign causes cover every missed callsign', causes['missed_callsigns'] == b['missed_callsigns']['count'])
    shown('missed callsigns that were letters or digits',
          f"{causes['by_cause']['letters or digits']} were spelled-out letters and numbers heard wrong")
    shown('missed callsigns that were the tagger', f"Another {causes['by_cause']['tagger']} were heard right")
check('"every missed callsign started with a known word" matches the saved analysis',
      b['missed_callsigns']['starting_with_a_word_never_in_training'] == 0)

tag = load('rescoring_2025/entity_extraction.json')
shown('tagger weighted F1', f"{tag['report']['weighted avg']['f1-score']:.4f}")
cm = tag['confusion_matrix']
check('confusion matrix on the page matches the saved one',
      all(f'>{v}</td>' in HTML for row in cm for v in row), 'a cell differs')
shown('ordinary words tagged as waypoints', f'Of the {sum(cm[0])} words those rules called ordinary, it labelled {cm[0][-1]}')

if 'noise_sweep' in retrain:
    sweep = retrain['noise_sweep']['results']
    at = lambda key, snr: next(p['wer'] for p in sweep[key] if p['added_noise_snr_db'] == snr)
    for key, snr in [('retrained', None), ('retrained', 10), ('retrained', 0), ('zero_shot', None), ('zero_shot', 0)]:
        shown(f'noise sweep {key} at {snr} dB', pct(at(key, snr)))
    lost = lambda key, snr: f"{100 * (at(key, snr) - at(key, None)):.2f}"
    shown('points lost to noise', f"{lost('retrained', 10)} at 10 dB where the 2025 model loses "
          f"{lost('checkpoint_2025', 10)}, and {lost('retrained', 0)} at 0 dB against {lost('checkpoint_2025', 0)}")
    check('the noise sweep is of the model chosen on validation',
          f'/{best}/' in retrain['noise_sweep']['retrained_model'], retrain['noise_sweep']['retrained_model'])

cv_path = RESULTS / 'cross_validation.json'
if cv_path.exists():
    cv = load('cross_validation.json')
    lo_cv, hi_cv = cv['retrained']['interval_95']
    shown('cross-validated word error rate', f"{pct(cv['retrained']['wer'])} (95% range {pct(lo_cv)} to {pct(hi_cv)})")
    shown('cross-validated baseline', pct(cv['zero_shot']['wer']))
    folds = [f['wer'] for f in cv['folds']]
    shown('fold range', f'{pct(min(folds))} to {pct(max(folds))}')
    check('cross-validation covers every clip', cv['clips'] == splits['asr_train_rows'] + splits['asr_test_rows'])
    paper = (SITE.parent / 'paper' / 'atco_paper.md').read_text(encoding='utf-8')
    z_lo, z_hi = cv['zero_shot']['interval_95']
    check('the paper reports the cross-validation',
          f"| {pct(cv['zero_shot']['wer'])} | {100 * z_lo:.2f} to {100 * z_hi:.2f} |" in paper
          and f"| {pct(cv['retrained']['wer'])} | {100 * lo_cv:.2f} to {100 * hi_cv:.2f} |" in paper)

for name, run in retrain.get('ablations', {}).items():
    shown(f'ablation {name}', pct(run['headline_test']['wer']))

if (RESULTS / 'letters_digits.json').exists():
    ld = load('letters_digits.json')['test']
    shown('letters and digits, best model', pct(ld[best]['accuracy']))
    shown('letters and digits, 2025 model', f"(the 2025 model: {pct(ld['checkpoint_2025']['accuracy'])})")
    lw = load('letters_digits.json')['rejected_trials']['letter_weighted_loss']['epoch1']
    base = load('letters_digits.json')['validation'][best]
    gained = {1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five'}.get(lw['right'] - base['right'])
    shown('letter-weighted trial gain', f"{gained} more of {lw['spelled_words']}")
    shown('letter-weighted trial word error rate',
          f"from {pct(retrain['runs'][best]['validation']['beam']['wer'])} to {pct(lw['wer'])}")

shown('synthetic clips', '4,808 clips')
check('historical values keep their unknown-metric label', 'metric not established' in TEXT)
check('historical values are not turned into a word error rate', not re.search(r'23\.(82|77|33)', TEXT))

print('\n-- README, evaluation notes and report match results/ --')
README = (SITE.parent / 'README.md').read_text(encoding='utf-8')
for name, run in runs.items():
    h = run['headline_test']
    lo_, hi_ = (f'{100 * x:.2f}' for x in h['interval_95'])
    check(f'README shows {name}', f"{pct(h['wer'])}" in README and f'{lo_} to {hi_}' in README,
          f"expected {pct(h['wer'])} and {lo_} to {hi_}")

large = (RESULTS / 'original_2025' / 'asr_large_lora_robustness.txt').read_text(encoding='utf-8')
large_wer = re.findall(r'\|\s*(\d+\.\d\d)%', large)[0::2]
check('README quotes the 2025 large-v3 scores exactly',
      len(large_wer) == 3 and all(f'{v}%' in README for v in large_wer), f'expected {large_wer}')

for doc in ['docs/evaluation.md', 'paper/atco_paper.md']:
    body = (SITE.parent / doc).read_text(encoding='utf-8')
    for name, run in runs.items():
        h = run['headline_test']
        row = f"| {pct(run['validation'][h['decoding']]['wer'])} | {pct(h['wer'])} |"
        check(f'{doc} shows {name}', row in body, f'expected "{row}"')

print('\n-- charts match results/ --')
sys.path.insert(0, str(SITE / 'tools'))
import build_charts  # noqa: E402

for name, draw in [('bars', build_charts.bars), ('noise', build_charts.noise)]:
    m = re.search(rf'<!-- chart:{name} -->\n(.*?)\n<!-- /chart:{name} -->', HTML, re.S)
    if name == 'noise' and 'noise_sweep' not in retrain:
        continue
    check(f'chart "{name}" is on the page and up to date', bool(m) and m.group(1) == draw(),
          'run python3 site/tools/build_charts.py')

print('\n-- rules --')
audio = [p.name for p in (SITE / 'audio').iterdir() if p.suffix in {'.wav', '.m4a', '.mp3', '.flac', '.ogg'}]
check('the only audio file is the text-to-speech demo call', audio == ['phrase.m4a'], str(audio))
check('demo audio is disclosed as text-to-speech',
      'text-to-speech voice' in TEXT and 'not a recording of a real controller' in TEXT)
check('demo audio does not autoplay', 'preload="none"' in HTML and 'autoplay' not in HTML)
check('the traced call is labelled as a saved run', 'Archived' in TEXT and 'not live' in TEXT)
check('no live model implied', 'No model runs in this page' in TEXT)
check('word error rate is not described as words wrong', not re.search(r'% of (the )?words (wrong|misheard)', TEXT))
check('no categorical Haneda claim', not re.search(r'every word was heard correctly', TEXT))
check('research prototype warning present', 'Research prototype. Not for operational use.' in TEXT)

print('\n-- structure --')
check('exactly one h1', HTML.count('<h1') == 1)
ids = set(re.findall(r'id="([^"]+)"', HTML))
dead = [a for a in re.findall(r'href="#([^"]+)"', HTML) if a not in ids]
check('no dead in-page links', not dead, str(dead))
for group in re.findall(r'data-covers="([^"]+)"', HTML):
    for sid in group.split():
        check(f'menu covers a real section: {sid}', sid in ids)
check('no third-party scripts', not re.search(r'<script[^>]+src="https?://', HTML))
check('results readable without JavaScript', '<div class="werbars"' in HTML.split('<script')[0])
em = HTML.count('&mdash;') + HTML.count('—')
check('no more than three em-dashes', em <= 3, f'{em} found')

if '--release' in sys.argv:
    # Every GitHub link must work for someone who is not logged in: the file is on
    # main AND the repository is public.
    import subprocess
    import urllib.request

    print('\n-- release --')
    repo = 'https://github.com/Sineceptor/ATCO'
    try:
        code = urllib.request.urlopen(urllib.request.Request(repo, headers={'User-Agent': 'Mozilla/5.0'}),
                                      timeout=10).status
    except Exception as e:
        code = getattr(e, 'code', 'unreachable')
    check('repository is public', code == 200, f'HTTP {code}: readers would see a 404 on every code link')
    for path in sorted(set(re.findall(r'https://github\.com/Sineceptor/ATCO/blob/main/([^"#]+)', HTML))):
        ok = subprocess.run(['git', 'cat-file', '-e', f'origin/main:{path}'], cwd=SITE.parent,
                            capture_output=True).returncode == 0
        check(f'linked file is on main: {path}', ok, 'not pushed yet')

print(f'\n{count - len(fails)}/{count} checks passed')
if fails:
    print('\nFAILURES:')
    for f in fails:
        print('  -', f)
    sys.exit(1)
