import torch
import numpy as np
from pathlib import Path
from PIL import Image
from huggingface_hub import hf_hub_url, cached_download
from rudalle.realesrgan.model import RealESRGAN
import logging

logger = logging.getLogger(__name__)


class ESRGANUpscale:
    MODELS = {
        'x2': dict(
            scale=2,
            repo_id='shonenkov/rudalle-utils',
            filename='RealESRGAN_x2.pth',
        ),
        'x4': dict(
            scale=4,
            repo_id='shonenkov/rudalle-utils',
            filename='RealESRGAN_x4.pth',
        ),
        'x8': dict(
            scale=8,
            repo_id='shonenkov/rudalle-utils',
            filename='RealESRGAN_x8.pth',
        ),
    }

    def __init__(self, upscale:int):
        self.upscale = upscale
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        logger.debug(f"Using {device} to upscale")
        self.model = RealESRGAN(device, upscale)
        self._load_weights()
        logger.debug("Model loaded!")
    
    def _load_weights(self):
        weights_path = Path('models') / f"RealESRGAN_x{self.upscale}.pth"
        if not weights_path.exists():
            self.download_weights()

        self.model.load_weights(weights_path)
    
    def _download_weights(self):
        logger.info('Downloading pre-trained weights for the model..')
        for model, config in self.MODELS.items():
            config_file_url = hf_hub_url(
                repo_id=config['repo_id'], filename=config['filename'])
            cached_download(config_file_url, cache_dir='models',
                            force_filename=config['filename'])

    def __call__(self, images_path, return_image=False):
        logger.info(f'Upscaling result with a factor of {self.upscale}')
        result_img_path = list(sorted(images_path.glob('*.png')))[-1]
        input_image = Image.open(result_img_path)     
        input_image = input_image.convert('RGB')

        with torch.no_grad():
            sr_image = self.model.predict(np.array(input_image))
        
        outpath = str(result_img_path)[:-4]+"_esr.png"
        sr_image.save(outpath)

        if return_image:
            return sr_image
