import pandas as pd
from pathlib import Path
import json
import argparse
import numpy as np

from neural_style_transfer import main


def prepare_configs(content, style, output, weight, output_path, height):
    configs = {
        "content_img_name": content,
        "style_img_name": style,
        "output_img_name": output,
        "output_path": output_path,
        "height": height,
        "content_weight": 1e5,
        "style_weight": weight,
        "tv_weight": 1e0,
        "optimizer": "lbfgs",
        "model": "vgg19",
        "init_method": "content",
        "saving_freq": -1,
        "upscale": 2,
        "algorithm": "original",
        "gif": False,
    }

    return configs


def parse_matrix(root_path):
    df = pd.read_csv(root_path / "metadata" / "final_matrix.csv")

    # Take only the relevant part of the dataframe
    df = df.iloc[4:, 3:]
    df = df.set_index("Unnamed: 3")
    df.columns = df.iloc[0]
    df = df.iloc[1:]
    df.columns.name = ""
    df.index.name = ""

    # Remove NAs
    df.dropna(how="all", axis=0, inplace=True)
    df.dropna(how="all", axis=1, inplace=True)

    # Convert all numbers to float
    df = df.applymap(lambda x: float(x.replace(",", ".")))

    return df


def run(n_random, output_path, height, content_name=None, style_name=None, prefix=None):
    root_path = Path("./data/")
    metadata_path = root_path / "output" / "metadata"
    metadata_path.mkdir(exist_ok=True, parents=True)

    # df = parse_matrix(root_path)
    df = None

    style_df = pd.read_csv(root_path / "metadata" / "style.csv")
    style_df = style_df[~style_df["File_name"].isna()]
    style_df = style_df.sort_values("File_name").reset_index(drop=True)

    content_df = pd.read_csv(root_path / "metadata" / "content.csv")
    content_df = content_df[~content_df["File_name"].isna()]
    content_df = content_df.sort_values("File_name").reset_index(drop=True)

    print(content_df.shape, style_df.shape)

    content_indices = np.random.randint(0, content_df.shape[0], n_random)
    style_indices = np.random.randint(0, style_df.shape[0], n_random)

    metadata_path = root_path / "output" / "metadata"
    metadata_path.mkdir(exist_ok=True, parents=True)

    for i, (content_id, style_id) in enumerate(zip(content_indices, style_indices)):
        if prefix is None:
            index = f"random_3_{i}"
        else:
            index = f"{prefix}_{i}"

        if content_name is None:
            print("Using random content")
            content = content_df.iloc[content_id]
        else:
            content = content_df[content_df["File_name"] == content_name].iloc[0]

        if style_name is None:
            print("Using random style")
            style = style_df.iloc[style_id]
        else:
            style = style_df[style_df["File_name"] == style_name].iloc[0]

        try:
            weight = df.loc[content["File_name"], style["File_name"]]
        except:
            weight = 3e4

        print(f"Processing content image: {content['File_name']}")
        print(f"Processing style: {style['File_name']}")
        print(f"Processing weight: {weight}")
        print(f"Processing output name: {index}")

        configs = prepare_configs(
            content["File_name"], style["File_name"], index, weight, output_path, height
        )

        try:
            result = main(configs)
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
                    {"trait_type": "Orientation", "value": content["Orientation"]},
                    {"trait_type": "File Name", "value": index},
                    {"trait_type": "Style weight", "value": weight},
                ],
            }

            file_metadata_path = metadata_path / (index + ".json")
            with open(file_metadata_path, "w") as f:
                json.dump(metadata, f)

        else:
            print("Failed execution")

        #         if i%100==0:
        #             print('Saving snapshot to file')
        #             prod.to_csv(f'status_{pc_number}.csv', index=False)

        print("\n\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, help="output path", default="output")
    parser.add_argument(
        "--height",
        type=int,
        nargs="+",
        help="height of content and style images",
        default=500,
    )
    parser.add_argument(
        "--random", type=int, nargs="+", help="number of random images", default=200
    )
    parser.add_argument(
        "--content", type=str, nargs="?", help="content image to use", default=None
    )
    parser.add_argument(
        "--style", type=str, nargs="?", help="style image to use", default=None
    )
    parser.add_argument(
        "--prefix", type=str, nargs="?", help="prefix for the name", default=None
    )
    args = parser.parse_args()

    run(
        args.random,
        args.output_path,
        args.height,
        args.content,
        args.style,
        args.prefix,
    )
