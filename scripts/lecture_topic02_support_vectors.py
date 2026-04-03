from __future__ import annotations

"""课题二教学脚本：支持向量、对偶直觉和 KKT。

这个脚本要在主课题模型已经训练完成之后再运行。
它会把训练好的线性 SVM 中的支持向量提取出来，
再映射回原始训练样本表，方便你在课堂上直接指出：

“到底是哪些样本在决定最终的分类面。”
"""

import argparse
from pathlib import Path

import joblib
import pandas as pd

from lecture_experiments.common import LectureConfig, ensure_dir, prepare_split_data, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Topic 2: inspect support vectors in the AD vs NORMAL linear model.")
    parser.add_argument("--model-path", default="results/main_topic_ad_vs_normal_m00/best_model_linear.joblib")
    parser.add_argument("--output-dir", default="results/lecture_topic02_support_vectors")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    # 这里先重建与讲课脚本一致的数据划分，
    # 然后再读取已经训练好的线性模型。
    config = LectureConfig()
    prepared = prepare_split_data(labels=["AD", "NORMAL"], config=config)
    model = joblib.load(args.model_path)
    svm = model.named_steps["svm"]

    # 由于 SVM 位于流水线最后一步，所以支持向量是定义在 PCA 变换后的空间里的。
    # 为了讲课更直观，我们把它们重新映射回训练样本行，看看具体是哪些受试者。
    X_train_scaled = model.named_steps["scaler"].transform(prepared["X_train"])
    X_train_pca = model.named_steps["pca"].transform(X_train_scaled)
    support_indices = svm.support_

    split_df = prepared["subset"].copy()
    split_df["split"] = prepared["split_column"]
    train_df = split_df[split_df["split"] == "train"].reset_index(drop=True)
    support_df = train_df.iloc[support_indices].copy()
    support_df["support_rank"] = range(1, len(support_df) + 1)
    support_df["distance_to_hyperplane"] = abs(svm.decision_function(X_train_pca[support_indices]))
    support_df["pc1"] = X_train_pca[support_indices, 0]
    support_df["pc2"] = X_train_pca[support_indices, 1]
    support_df.to_csv(output_dir / "support_vectors.csv", index=False)

    summary = {
        "support_vectors_total": int(len(support_indices)),
        "support_vectors_per_class": svm.n_support_.tolist(),
        "explanation": [
            "Only support vectors directly determine the separating hyperplane.",
            "Samples farther from the margin do not change the final solution once the margin is fixed.",
            "This is the most intuitive project connection to the dual problem and KKT conditions.",
        ],
    }
    save_json(output_dir / "summary.json", summary)
    print(f"Saved support-vector table to: {output_dir / 'support_vectors.csv'}")
    print(f"Saved summary to: {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
