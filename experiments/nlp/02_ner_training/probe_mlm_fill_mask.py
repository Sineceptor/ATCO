# ---------------------------------------------------------------
# Archived experiment (late 2025). Original file: NLP/test_bert_mlm.py
# What it does: Prints top-5 fill-mask predictions for seven ATC sentences to see whether MLM pretraining learned phraseology. No metric.
# Not maintained: paths and dependencies are as they were at the time.
# ---------------------------------------------------------------
import os
import tensorflow as tf
from transformers import BertTokenizerFast, TFBertForMaskedLM
import logging

# Configuration
# Saved base model.
MODEL_PATH = "./bert_atc_pretrain_base"

# 想要测试的句子 (其中 [MASK] 是让模型猜的词)
TEST_SENTENCES = [
    "CLEARED TO [MASK] RUNWAY TWO FOUR",  # 期待: LAND, TAKEOFF
    "DESCEND TO FLIGHT [MASK] NINE ZERO",  # 期待: LEVEL
    "CONTACT TOWER ONE ONE EIGHT [MASK] ONE",  # 期待: DECIMAL, POINT
    "REPORT [MASK] ESTABLISHED RUNWAY TWO FIVE",  # 期待: FINAL, WHEN, FULLY
    "TURN [MASK] HEADING TWO THREE ZERO",  # 期待: LEFT, RIGHT
    "WIND TWO THREE ZERO DEGREES EIGHT [MASK]",  # 期待: KNOTS
    "ROGER [MASK]",  # 期待: WILCO, BYE, GOOD, THANKS
]


# 屏蔽干扰日志
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("transformers").setLevel(logging.ERROR)


def main():
    if not os.path.exists(MODEL_PATH):
        print(f"找不到模型路径: {MODEL_PATH}")
        print("请先运行 train_mlm_depth.py 训练底座！")
        return

    print(f"正在加载模型: {MODEL_PATH}...")

    try:
        # 加载分词器和模型
        tokenizer = BertTokenizerFast.from_pretrained(MODEL_PATH)
        model = TFBertForMaskedLM.from_pretrained(MODEL_PATH)
    except Exception as e:
        print(f"加载失败: {e}")
        return

    print("模型加载成功！开始测试...\n")
    print("=" * 60)

    for sentence in TEST_SENTENCES:
        # 编码输入
        inputs = tokenizer(sentence, return_tensors="tf")

        # 找到 [MASK] 的位置
        # Find the mask token in the first input sequence.
        mask_token_index = tf.where(inputs["input_ids"][0] == tokenizer.mask_token_id)

        if len(mask_token_index) == 0:
            print(f"句子中没有 [MASK]: {sentence}")
            continue

        # 取第一个 mask 的位置
        mask_index = mask_token_index[0][0]

        # 模型预测
        outputs = model(inputs)
        predictions = outputs.logits

        # 获取该位置概率最高的 Top 5 候选词
        mask_prediction_logits = predictions[0, mask_index, :]
        top_5_ids = tf.math.top_k(mask_prediction_logits, k=5).indices.numpy()

        print(f"原句: {sentence}")
        print(f"预测 (Top 5):")

        for i, token_id in enumerate(top_5_ids):
            token_str = tokenizer.decode([token_id])
            print(f"   {i + 1}. {token_str}")
        print("-" * 60)


if __name__ == "__main__":
    main()
