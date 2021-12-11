import pandas as pd
from pathlib import Path
import json
import argparse

from neural_style_transfer import main

WEIGHTS = [5e2, 5e4]


def cartesian_product(d):
    index = pd.MultiIndex.from_product(d.values(), names=d.keys())
    return pd.DataFrame(index=index).reset_index()


def prepare_configs(content, style, output, weight, output_path, height):
    configs = {
        'content_img_name': content,
        'style_img_name': style,
        'output_img_name': output,
        'output_path': output_path,
        'height': height,
        'content_weight': 1e5,
        'style_weight': weight,
        'tv_weight': 1e0,
        'optimizer': 'lbfgs',
        'model': 'vgg19',
        'init_method': 'content',
        'saving_freq': 50
    }
    
    return configs


def run(pc_number, weights, output_path, height):
    root_path = Path('./data/')
    metadata_path = root_path / 'output' / 'metadata'
    metadata_path.mkdir(exist_ok=True, parents=True)
    
    style_df = pd.read_csv(root_path / 'metadata' / 'style.csv')
    style_df = style_df[~style_df['File_name'].isna()]
    style_df = style_df.sort_values('File_name').reset_index(drop=True)

    content_df = pd.read_csv(root_path / 'metadata' / 'content.csv')
    content_df = content_df[~content_df['File_name'].isna()]
    content_df = content_df.sort_values('File_name').reset_index(drop=True)

    print(content_df.shape, style_df.shape)
    
    elements = content_df.shape[0]//5
    print(elements)
    content_df = content_df.loc[pc_number*elements:pc_number*elements + elements -1]
    
    start = 1 + pc_number*elements*style_df.shape[0]*len(weights)
    print(f'Starting from name {start}')
    
    prod = cartesian_product({
        'content': content_df['File_name'],
        'style': style_df['File_name'],
        'weight': weights,
    })
    prod['index'] = list(prod.index+start)
    prod['result'] = 'todo'
    prod['to_review'] = False
    prod.to_csv(f'status_{pc_number}.csv', index=False)

    for i, row in prod.iloc[6:].iterrows():
        weight = row['weight']
        index = str(row['index'])

        content = content_df.loc[content_df['File_name'] == row['content']]
        style = style_df.loc[style_df['File_name'] == row['style']]
        if (content.shape[0]>1) or (style.shape[0]>1):
            print('Found more than one file with the given file_name')
            prod.loc[i, 'to_review'] = True

        print(f"Processing content image: {row['content']}")
        print(f"Processing style: {row['style']}")
        print(f"Processing weight: {row['weight']}")
        print(f"Processing output name: {index}")
        
        configs = prepare_configs(row['content'], row['style'], index, weight, output_path, height)
        
        try:
            result = main(configs)
        except:
            print("Execution failed for current image")
            result = False
        
        if result is True:
            print('Successful execution')
            
            metadata = {
                "description": "The most iconic pieces of art, reimagined by AI.",
                "image": "TBD",
                "name" : f"{content.iloc[0]['Title']} X {style.iloc[0]['Title']}",
                "animation_url": "TBD",
                "attributes": [
                    {
                        "trait_type": "Content",
                        "value": content.iloc[0]['Title']
                    },
                    {
                        "trait_type": "Content Author",
                        "value": content.iloc[0]['Author']
                    },
                    {
                        "trait_type": "Style",
                        "value": style.iloc[0]['Title']
                    },
                    {
                        "trait_type": "Style Author",
                        "value": style.iloc[0]['Author']
                    },
                    {
                        "trait_type": "Orientation",
                        "value": content.iloc[0]['Orientation']
                    },
                    {
                        "trait_type": "File Name",
                        "value": index
                    },
                    {
                        "trait_type": "Style weight",
                        "value": weight
                    }
                ]
            }

            file_metadata_path = metadata_path / (index + '.json')
            with open(file_metadata_path, 'w') as f:
                json.dump(metadata, f)

            prod.loc[i, 'done'] = 'success'
        
        else:
            print('Failed execution')
            prod.loc[i, 'done'] = 'failed'
            
        if i%100==0:
            print('Saving snapshot to file')
            prod.to_csv(f'status_{pc_number}.csv', index=False)

        print('\n\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pc_number", type=int, help="height of content and style images", default=0)
    parser.add_argument("--output_path", type=str, help='output path', default='output')
    parser.add_argument("--height", type=int, nargs='+', help="height of content and style images", default=500)
    args = parser.parse_args()
    
    run(args.pc_number, WEIGHTS, args.output_path, args.height)