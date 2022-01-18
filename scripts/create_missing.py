import pandas as pd
from pathlib import Path
import json
import argparse
import boto3

from ..neural_style_transfer import main

WEIGHTS = [5e2, 5e4]
client = boto3.client('s3')


def cartesian_product(d):
    index = pd.MultiIndex.from_product(d.values(), names=d.keys())
    return pd.DataFrame(index=index).reset_index()


def iterate_bucket_items(bucket, prefix):
    """
    Generator that iterates over all objects in a given s3 bucket

    See http://boto3.readthedocs.io/en/latest/reference/services/s3.html#S3.Client.list_objects_v2 
    for return data format
    :param bucket: name of s3 bucket
    :return: dict of metadata for an object
    """

    paginator = client.get_paginator('list_objects_v2')
    page_iterator = paginator.paginate(Bucket=bucket, Prefix=prefix)

    for page in page_iterator:
        if page['KeyCount'] > 0:
            for item in page['Contents']:
                yield item


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


def cartesian_product(d):
    index = pd.MultiIndex.from_product(d.values(), names=d.keys())
    return pd.DataFrame(index=index).reset_index()


def run(output_path, height):
    def check_component(component):
        prefix = f'output/{component}/'
        
        for item in iterate_bucket_items(bucket='neuralism-assets', prefix=prefix):
            if '.' not in item['Key']:
                continue
                
            key = item['Key'].split('/')[-1].split('.')[0]
            if component=='gifs':
                key = key[1:]
            
            df.loc[prod['index']==int(key), component] = True

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

    prod = cartesian_product({
        'content': content_df['File_name'],
        'style': style_df['File_name'],
        'weight': WEIGHTS,
    })

    prod['index'] = list(prod.index+1)
    prod['gifs'] = False
    prod['images'] = False
    prod['metadata'] = False
    prod['to_review'] = False
    
    content_merged = prod.merge(content_df, left_on='content', right_on='File_name')
    content_merged = content_merged.drop(['Orientation', 'Resolution', 'File_name', 'Looks_good', 'Copyright link'], axis=1)
    content_merged.columns = ['content', 'style', 'weight', 'index', 'gifs', 'images', 'metadata', 'to_review', 'content_title', 'content_author', 'content_copyight']

    df = content_merged.merge(style_df, left_on='style', right_on='File_name')
    df = df.drop(['Resolution', 'File_name', 'Strength'], axis=1)
    df.columns = ['content', 'style', 'weight', 'index', 'gifs', 'images', 'metadata', 'to_review', 'content_title', 'content_author', 'content_copyright', 'style_title', 'style_author', 'style_copyright']

    df = df[['index', 'to_review', 'content', 'content_title', 'content_author', 'content_copyright', 'style', 'style_title', 'style_author', 'style_copyright', 'weight', 'gifs', 'images', 'metadata']]
    df[['content_copyright', 'style_copyright']] = False

    df = df.sort_values('index').reset_index(drop=True)

    check_component('gifs')
    check_component('images')
    check_component('metadata')

    missing = df[(~df['images']) | (~df['gifs']) | (~df['metadata'])]

    for i, row in missing.iterrows():
        weight = row['weight']
        index = str(row['index'])

        content = content_df.loc[content_df['File_name'] == row['content']]
        style = style_df.loc[style_df['File_name'] == row['style']]

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
        
        else:
            print('Failed execution')
            
        if i%100==0:
            print('Saving snapshot to file')
            missing.to_csv(f'missing.csv', index=False)

        print('\n\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, help='output path', default='output')
    parser.add_argument("--height", type=int, nargs='+', help="height of content and style images", default=500)
    args = parser.parse_args()
    
    run(args.output_path, args.height)