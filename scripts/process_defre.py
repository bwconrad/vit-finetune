import argparse
import os
import shutil

import pandas as pd


def main(root):
    for split in ["train", "val", "test"]:
        # Create directories
        split_dir = os.path.join(root, split)
        for i in range(3263):
            os.makedirs(os.path.join(split_dir, str(i)), exist_ok=True)

        # Load image labels
        anns = pd.read_csv(
            os.path.join(
                root,
                "labels",
                f"{split}.csv",
            ),
            names=["label", "path"],
        )

        # Move images to correct directories
        image_dir = os.path.join(root, "faces")
        for label, path in zip(anns.label.to_list(), anns.path.to_list()):
            shutil.copy(
                os.path.join(image_dir, path), os.path.join(split_dir, str(label))
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", "-i", type=str, required=True, help="Path to DAF:RE dataset"
    )
    args = parser.parse_args()
    main(args.input)
