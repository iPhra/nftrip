import pandas as pd
from pathlib import Path
import json
import argparse
from subprocess import run
import shutil

from neural_style_transfer import main

WEIGHTS = [3e4]
BUCKET_NAME = 'neuralism-assets'
PREFIX = 'v2'


def cartesian_product(d):
    index = pd.MultiIndex.from_product(d.values(), names=d.keys())
    return pd.DataFrame(index=index).reset_index()


def prepare_configs(content, style, output, weight, output_path):
    configs = {
        "content_img_name": content,
        "style_img_name": style,
        "output_img_name": output,
        "output_path": output_path,
        "height": 500,
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


def run(output_path):
    root_path = Path('./data/')
    metadata_path = root_path / output_path / 'metadata'
    metadata_path.mkdir(exist_ok=True, parents=True)
    output_folder = root_path / output_path
    
    style_df = pd.read_csv(root_path / 'metadata' / 'style.csv')
    style_df = style_df[~style_df['File_name'].isna()]
    style_df = style_df.sort_values('File_name').reset_index(drop=True)

    content_df = pd.read_csv(root_path / 'metadata' / 'content.csv')
    content_df = content_df[~content_df['File_name'].isna()]
    content_df = content_df.sort_values('File_name').reset_index(drop=True)

    print(content_df.shape, style_df.shape)
    
    start = 1
    print(f'Starting from name {start}')
    
    prod = cartesian_product({
        'content': content_df['File_name'],
        'style': style_df['File_name'],
        'weight': WEIGHTS,
    })
    prod['index'] = list(prod.index+start)
    prod['result'] = 'todo'
    prod['to_review'] = False
    prod.to_csv(f'status.csv', index=False)

    for i, row in prod.iterrows():
        weight = row['weight']
        index = str(row['index'])

        content = content_df.loc[content_df['File_name'] == row['content']].iloc[0]
        style = style_df.loc[style_df['File_name'] == row['style']].iloc[0]

        print(f"Processing content image: {row['content']}")
        print(f"Processing style: {row['style']}")
        print(f"Processing weight: {row['weight']}")
        print(f"Processing output name: {index}")
        
        configs = prepare_configs(
            content["File_name"], style["File_name"], index, weight, output_path
        )

        try:
            result = main(configs)
        except Exception as e:
            print(e)
            result = False
        
        if result is True:
            print('Successful execution')
            
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
                ],
            }

            file_metadata_path = metadata_path / (index + '.json')
            with open(file_metadata_path, 'w') as f:
                json.dump(metadata, f)

            prod.loc[i, 'done'] = 'success'
        
        else:
            print('Failed execution')
            prod.loc[i, 'done'] = 'failed'
            
        if i%1000==0:
            print('Saving snapshot to file and syncing output')
            prod.to_csv(f'status.csv', index=False)

            run(f"aws s3 sync {output_folder} s3://{BUCKET_NAME}/{PREFIX}/output/")
            shutil.rmtree(output_folder, ignore_errors=True)

        print('\n\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, help='output path', default='output')
    parser.add_argument("--height", type=int, nargs='+', help="height of content and style images", default=500)
    args = parser.parse_args()
    
    run(args.output_path, args.height)