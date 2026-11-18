import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
from sklearn.model_selection import StratifiedKFold

from model_3dcnn_brnn import CNN3DBRNN


def load_subject_sequence(rows):
    volumes = []

    for path in rows["image_path"]:
        volume = np.load(path).astype(np.float32)

        if volume.shape != (96, 96, 96):
            raise ValueError(f"Invalid MRI shape: {path}")

        volumes.append(volume)

    array = np.stack(volumes)
    return torch.from_numpy(array[None, :, None])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default="Data_small/tables_assessment/dl_manifest_adni.csv"
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expected-subjects", type=int, default=24)
    parser.add_argument("--output-dir", default="Data_small/models/3dcnn_brnn")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(2)

    device = torch.device("cpu")
    manifest = pd.read_csv(args.manifest)

    subjects = manifest.groupby("subject_id").agg(
        label=("label", "first"),
        label_count=("label", "nunique")
    ).reset_index()

    if (subjects["label_count"] != 1).any():
        raise ValueError("Inconsistent labels within a subject.")

    if subjects["subject_id"].duplicated().any():
        raise ValueError("Duplicate subject IDs.")

    X = subjects["subject_id"].to_numpy()
    y = subjects["label"].astype(int).to_numpy()

    if len(subjects) != args.expected_subjects:
        raise ValueError(
            f"Expected {args.expected_subjects} subjects, found {len(subjects)}."
        )

    splitter = StratifiedKFold(
        n_splits=2,
        shuffle=True,
        random_state=args.seed
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_predictions = []
    fold_results = []

    for fold, (train_idx, test_idx) in enumerate(
        splitter.split(X, y), start=1
    ):
        train_ids = X[train_idx]
        test_ids = X[test_idx]

        train_labels = y[train_idx]
        positive_count = int(train_labels.sum())
        negative_count = len(train_labels) - positive_count

        pos_weight = torch.tensor(
            [negative_count / positive_count],
            dtype=torch.float32
        )

        model = CNN3DBRNN().to(device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=0.001
        )

        criterion = torch.nn.BCEWithLogitsLoss(
            pos_weight=pos_weight
        )

        print(f"\n========== FOLD {fold} ==========")
        print(
            f"Train: {len(train_ids)} subjects "
            f"({positive_count} positive, {negative_count} negative)"
        )
        print(f"Test: {len(test_ids)} subjects")

        for epoch in range(args.epochs):
            model.train()
            order = list(train_ids)
            random.shuffle(order)
            epoch_losses = []

            for subject_id in order:
                rows = manifest[
                    manifest["subject_id"] == subject_id
                ].sort_values("sequence_index")

                x = load_subject_sequence(rows).to(device)
                label = torch.tensor(
                    [float(rows["label"].iloc[0])],
                    dtype=torch.float32,
                    device=device
                )

                optimizer.zero_grad(set_to_none=True)

                logits = model(x)
                loss = criterion(logits, label)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=1.0
                )
                optimizer.step()

                epoch_losses.append(loss.item())

            print(
                f"Epoch {epoch + 1}/{args.epochs} "
                f"- Loss: {np.mean(epoch_losses):.4f}"
            )

        model.eval()
        fold_predictions = []

        with torch.no_grad():
            for subject_id in test_ids:
                rows = manifest[
                    manifest["subject_id"] == subject_id
                ].sort_values("sequence_index")

                x = load_subject_sequence(rows).to(device)
                probability = torch.sigmoid(model(x)).item()

                fold_predictions.append({
                    "subject_id": subject_id,
                    "fold": fold,
                    "actual": int(rows["label"].iloc[0]),
                    "probability": probability,
                    "predicted": int(probability >= 0.5)
                })

        predictions = pd.DataFrame(fold_predictions)

        actual = predictions["actual"]
        predicted = predictions["predicted"]
        probabilities = predictions["probability"]

        auc = roc_auc_score(actual, probabilities)

        results = {
            "fold": fold,
            "accuracy": accuracy_score(actual, predicted),
            "balanced_accuracy": balanced_accuracy_score(
                actual, predicted
            ),
            "f1": f1_score(actual, predicted, zero_division=0),
            "roc_auc": auc
        }

        fold_results.append(results)
        all_predictions.extend(fold_predictions)

        print("\nFold metrics:")
        for metric, value in results.items():
            if metric != "fold":
                print(f"{metric}: {value:.4f}")

        print("Confusion matrix:")
        print(confusion_matrix(actual, predicted))

        torch.save(
            model.state_dict(),
            output_dir / f"model_fold_{fold}.pt"
        )

    predictions_df = pd.DataFrame(all_predictions)
    results_df = pd.DataFrame(fold_results)

    predictions_df.to_csv(
        output_dir / "predictions.csv", index=False
    )
    results_df.to_csv(
        output_dir / "fold_metrics.csv", index=False
    )

    print("\n========== CROSS-VALIDATION SUMMARY ==========")
    print(results_df.to_string(index=False))

    print("\nMean metrics:")
    print(results_df.drop(columns="fold").mean().round(4).to_string())

    print("\nSaved results to:", output_dir)


if __name__ == "__main__":
    main()
