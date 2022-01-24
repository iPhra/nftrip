import os
import logging
import torch
from torchvision.utils import save_image
from PIL import Image, ImageFile
from .net import Net
from .utils import DEVICE, test_transform
Image.MAX_IMAGE_PIXELS = None  
ImageFile.LOAD_TRUNCATED_IMAGES = True


logger = logging.getLogger(__name__)


class PAMA:
    def __init__(self, config):
        self.config = config
        self.content_img_path = (
            config["content_images_dir"] / config["content_img_name"]
        )
        self.style_img_path = config["style_images_dir"] / config["style_img_name"]
        out_dir_name = (
            "combined_"
            + os.path.split(self.content_img_path)[1].split(".")[0]
            + "_"
            + os.path.split(self.style_img_path)[1].split(".")[0]
        )
        self.dump_path = self.config["output_img_dir"] / out_dir_name
        self.dump_path.mkdir(exist_ok=True, parents=True)

        # remove all files in the folder
        [f.unlink() for f in self.dump_path.glob("*") if f.is_file()]

    def predict(self):
        logger.info("Starting prediction")

        model = Net()
        model.eval()
        model = model.to(DEVICE)

        tf = test_transform()
        Ic = tf(Image.open(self.content_img_path)).to(DEVICE)
        Is = tf(Image.open(self.style_img_path)).to(DEVICE)

        Ic = Ic.unsqueeze(dim=0)
        Is = Is.unsqueeze(dim=0)

        with torch.no_grad():
            Ics = model(Ic, Is)

        out_img_name = 'result.png'
        save_image(Ics[0], self.dump_path / out_img_name)

        return self.dump_path