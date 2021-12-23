import pandas as pd
from pathlib import Path
import json
import argparse
import numpy as np

from neural_style_transfer import main


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
        'optimizer': 'adam',
        'model': 'vgg19',
        'init_method': 'content',
        'saving_freq': -1
    }
    
    return configs


def parse_matrix(root_path):
    df = pd.read_csv(root_path / 'metadata' / 'final_matrix.csv')

    # Take only the relevant part of the dataframe
    df = df.iloc[4:,3:]
    df = df.set_index('Unnamed: 3')
    df.columns = df.iloc[0]
    df = df.iloc[1:]
    df.columns.name = ''
    df.index.name = ''

    # Remove NAs
    df.dropna(how='all', axis=0, inplace=True)
    df.dropna(how='all', axis=1, inplace=True)

    # Convert all numbers to float
    df = df.applymap(lambda x: float(x.replace(',','.')))
    
    return df


def run(n_random, output_path, height):
    root_path = Path('./data/')
    metadata_path = root_path / 'output' / 'metadata'
    metadata_path.mkdir(exist_ok=True, parents=True)
    
    df = parse_matrix(root_path)
    
    style_df = pd.read_csv(root_path / 'metadata' / 'style.csv')
    style_df = style_df[~style_df['File_name'].isna()]
    style_df = style_df.sort_values('File_name').reset_index(drop=True)

    content_df = pd.read_csv(root_path / 'metadata' / 'content.csv')
    content_df = content_df[~content_df['File_name'].isna()]
    content_df = content_df.sort_values('File_name').reset_index(drop=True)

    print(content_df.shape, style_df.shape)
    
    content_indices = np.random.randint(0, content_df.shape[0], n_random)
    style_indices = np.random.randint(0, style_df.shape[0], n_random)

    metadata_path = root_path / 'output' / 'metadata'
    metadata_path.mkdir(exist_ok=True, parents=True)
    
    for i, (content_id, style_id) in enumerate(zip(content_indices, style_indices)):
        index = f'random_{i}'
        
        content = content_df.iloc[content_id]
        style = style_df.iloc[style_id]

        try:
            weight = df.loc[content['File_name'],style['File_name']]
        except:
            weight = 5e4
        
        print(f"Processing content image: {content['File_name']}")
        print(f"Processing style: {style['File_name']}")
        print(f"Processing weight: {weight}")
        print(f"Processing output name: {index}")
            
        configs = prepare_configs(content['File_name'], style['File_name'], index, weight, output_path, height)
        
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
        
        else:
            print('Failed execution')
            
#         if i%100==0:
#             print('Saving snapshot to file')
#             prod.to_csv(f'status_{pc_number}.csv', index=False)

        print('\n\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, help='output path', default='output')
    parser.add_argument("--height", type=int, nargs='+', help="height of content and style images", default=500)
    parser.add_argument("--random", type=int, nargs='+', help="number of random images", default=200)
    args = parser.parse_args()
    
    run(args.random, args.output_path, args.height)