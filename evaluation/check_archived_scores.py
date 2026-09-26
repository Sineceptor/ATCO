"""Recalculate archived ASR scores from their mismatch logs and matching manifests.

The old evaluators log mismatches only. Unlisted rows are therefore treated as
exact matches. This checks the saved arithmetic; it is not another model run.
"""

import argparse
import json
from pathlib import Path
import re

import jiwer

from .evaluate_models import ROOT, digest, error_rates
from .historical_normalization import final_processing_pipeline


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--data-dir", type=Path, default=ROOT / "datasets")
    cli.add_argument("--output", type=Path, default=ROOT / "outputs/evaluation")
    args = cli.parse_args()
    raw = jiwer.Compose(
        [
            jiwer.ToUpperCase(),
            jiwer.RemovePunctuation(),
            jiwer.RemoveMultipleSpaces(),
            jiwer.Strip(),
        ]
    )
    cleaned = final_processing_pipeline
    summaries = []
    for report_name, manifest_name, normalize in [
        ("archived/quick_test_predictions.txt", "archived/quick_test_manifest.jsonl", raw),
        ("archived/beam_predictions.txt", "speech/test.jsonl", cleaned),
    ]:
        report_path = args.data_dir / report_name
        manifest_path = args.data_dir / manifest_name
        records = re.findall(
            r"(?:File: |WAV: )([^\n]+)\n(?:Ref|REF): ([^\n]*)\n(?:Hyp|HYP): ([^\n]*)",
            report_path.read_text(),
        )
        mismatches = {name: (ref, hyp) for name, ref, hyp in records}
        assert len(mismatches) == len(records), "Duplicate logged clips"
        data = [
            json.loads(s) for s in manifest_path.read_text().splitlines() if s.strip()
        ]
        filenames = {row["audio_file"] for row in data}
        assert len(filenames) == len(data), "Duplicate manifest clips"
        assert not (mismatches.keys() - filenames), "Logged clips missing from manifest"
        references, predictions = [], []
        for row in data:
            reference = normalize(row["ASR_transcript_clean"])
            old_ref, prediction = mismatches.get(
                row["audio_file"], (reference, reference)
            )
            assert old_ref == reference, f"Reference changed: {row['audio_file']}"
            references.append(reference)
            predictions.append(prediction)
        summaries.append(
            dict(
                report=report_name,
                manifest=manifest_name,
                clips=len(data),
                mismatch_records=len(records),
                unlogged_assumed_matches=len(data) - len(records),
                report_sha256=digest(report_path),
                manifest_sha256=digest(manifest_path),
                **error_rates(references, predictions),
            )
        )
    result = {
        "method": "Archived mismatches plus matching manifest; unlogged rows assumed perfect, "
        "following the old evaluator's log format. No new model inference.",
        "reports": summaries,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "archived_speech_scores.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
