import pandas as pd
from pathlib import Path
import json
import argparse
import subprocess
from pathlib import Path

from neural_style_transfer import main

WEIGHTS = [3e4]


def cartesian_product(d):
    index = pd.MultiIndex.from_product(d.values(), names=d.keys())
    return pd.DataFrame(index=index).reset_index()


def prepare_configs(content, style, output, weight, run_configs):
    configs = {
        "content_img_name": content,
        "style_img_name": style,
        "output_img_name": output,
        "output_path": run_configs.output_path,
        "height": 500,
        "content_weight": 1e5,
        "style_weight": weight,
        "tv_weight": 1e0,
        "optimizer": "lbfgs",
        "model": "vgg19",
        "init_method": "content",
        "saving_freq": -1,
        "upscale": 2,
        "algorithm": run_configs.algorithm,
        "gif": False,
    }

    return configs


def run(run_configs):
    root_path = Path("./data/")
    metadata_path = root_path / run_configs.output_path / "metadata"
    metadata_path.mkdir(exist_ok=True, parents=True)
    output_folder = root_path / run_configs.output_path

    style_df = pd.read_csv(root_path / "metadata" / "style.csv")
    style_df = style_df[~style_df["File_name"].isna()]
    style_df = style_df.sort_values("File_name").reset_index(drop=True)

    content_df = pd.read_csv(root_path / "metadata" / "content.csv")
    content_df = content_df[~content_df["File_name"].isna()]
    content_df = content_df.sort_values("File_name").reset_index(drop=True)

    print(content_df.shape, style_df.shape)

    start = 1
    print(f"Starting from name {start}")

    prod = cartesian_product(
        {
            "content": content_df["File_name"],
            "style": style_df["File_name"],
            "weight": WEIGHTS,
        }
    )
    prod["index"] = list(prod.index + start)
    prod["result"] = "todo"
    prod["to_review"] = False
    prod.to_csv(f"status.csv", index=False)

    for i, row in prod.iloc.iterrows():
        weight = row["weight"]
        index = str(row["index"])

        content = content_df.loc[content_df["File_name"] == row["content"]].iloc[0]
        style = style_df.loc[style_df["File_name"] == row["style"]].iloc[0]

        print(f"Processing content image: {row['content']}")
        print(f"Processing style: {row['style']}")
        print(f"Processing output name: {index}")

        optimization_configs = prepare_configs(
            content["File_name"], style["File_name"], index, weight, run_configs
        )

        try:
            result = main(optimization_configs)
        except Exception as e:
            print(e)
            result = False

        if result is True:
            print("Successful execution")

            metadata = {
                "description": "The most iconic pieces of art, reimagined by AI.",
                "image": "TBD",
                "name": f"{content['Title']} X {style['Title']}",
                "animation_url": "TBD",
                "attributes": [
                    {"trait_type": "Content", "value": content["Title"]},
                    {"trait_type": "Content Author", "value": content["Author"]},
                    {"trait_type": "Style", "value": style["Title"]},
                    {"trait_type": "Style Author", "value": style["Author"]},
                    {"trait_type": "File Name", "value": index},
                    {
                        "trait_type": "AI Version",
                        "value": "v1" if run_configs.algorithm == "original" else "v2",
                    },
                ],
            }

            file_metadata_path = metadata_path / (index + ".json")
            with open(file_metadata_path, "w") as f:
                json.dump(metadata, f)

            prod.loc[i, "done"] = "success"

        else:
            print("Failed execution")
            prod.loc[i, "done"] = "failed"

        if i % 1000 == 0:
            print("Saving snapshot to file and syncing output")
            prod.to_csv(f"status.csv", index=False)

            subprocess.run(
                f"aws s3 sync {output_folder} s3://{run_configs.bucket_name}/{run_configs.output_path}/",
                shell=True,
            )
            [f.unlink() for f in (output_folder / "metadata").glob("*") if f.is_file()]
            [f.unlink() for f in (output_folder / "images").glob("*") if f.is_file()]
            [f.unlink() for f in (output_folder / "gifs").glob("*") if f.is_file()]

        print("\n\n")

    subprocess.run(
        f"aws s3 sync {output_folder} s3://{run_configs.bucket_name}/{run_configs.output_path}/",
        shell=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, help="output path", default="v2")
    parser.add_argument("--bucket_name", type=str, default="neuralism-assets")
    parser.add_argument(
        "--algorithm",
        type=str,
        choices=["original", "pama"],
        help="Neural Style Transfer model",
        default="pama",
    )
    parser.add_argument(
        "--upscale", type=int, choices=[2, 4, 8], help="upscaling factor", nargs="?"
    )
    args = parser.parse_args()

    run_configs = dict()
    for arg in vars(args):
        run_configs[arg] = getattr(args, arg)

    run(run_configs)
