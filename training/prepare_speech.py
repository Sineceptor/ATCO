import json
import re
import os


def deep_clean_atc(text):
    """Remove annotation tags and normalise the transcript to lowercase."""
    if not text:
        return ""


    text = re.sub(r"([A-Za-z]+)\s*\(\s*.*?\-?\s*\)", r"\1", text)
    text = re.sub(r"\(\s*.*?\-?\s*\)\s*([A-Za-z]+)", r"\1", text)


    text = re.sub(
        r"\[\s*(HES|NOISE|UNINTELLIGIBLE|COUGH|SILENCE)\s*\]", " ", text, flags=re.IGNORECASE
    )


    text = re.sub(r"\[NE-[A-Z]+\]", " ", text, flags=re.IGNORECASE)


    text = text.upper().replace("DECIMAL", "POINT").replace("-", " ")


    text = re.sub(r"[^A-Z0-9\s]", " ", text)


    return " ".join(text.lower().split()).strip()


def run_cleaning_task(input_file, output_file, show_samples=15):
    if not os.path.exists(input_file):
        print(f"错误：找不到文件 '{input_file}'")
        return

    cleaned_results = []
    comparison_log = []

    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            original_text = item.get("ASR_transcript_clean", "")


            new_text = deep_clean_atc(original_text)

            if not new_text:
                continue

            item["ASR_transcript_clean"] = new_text
            cleaned_results.append(item)


            if original_text.upper() != new_text.upper() and len(comparison_log) < show_samples:
                comparison_log.append((original_text, new_text))
            elif "HES" in original_text.upper() and len(comparison_log) < show_samples:
                comparison_log.append((original_text, new_text))

    print(f"\n>>> 正在处理: {input_file}")
    print("-" * 60)
    for idx, (old, new) in enumerate(comparison_log):
        print(f"示例 {idx + 1}:")
        print(f"  [原句]: {old}")
        print(f"  [洗后]: {new}")
        print("-" * 20)

    with open(output_file, "w", encoding="utf-8") as f:
        for item in cleaned_results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f'处理完成！原始样本: {len(cleaned_results)} 条')
    print(f"结果已保存至: {output_file}\n")


if __name__ == "__main__":
    import argparse
    from pathlib import Path
    cli = argparse.ArgumentParser(description="Clean a speech manifest without overwriting its source.")
    cli.add_argument("input", type=Path)
    cli.add_argument("output", type=Path)
    args = cli.parse_args()
    if args.input.resolve() == args.output.resolve():
        cli.error("Choose a separate output file.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    run_cleaning_task(args.input, args.output)
