import argparse
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path
from time import time

import imageio
from PIL import Image

from models import Original, PAMA 
from utils.upscale import ESRGANUpscale as Upscaler

logs_folder = Path("logs")
logs_folder.mkdir(exist_ok=True, parents=True)
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(f'logs/log-{datetime.now().strftime("%d_%m_%Y")}.log'),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def copy_output(optimization_config, results_path):
    optimization_config["images_path"].mkdir(exist_ok=True, parents=True)
    optimization_config["gifs_path"].mkdir(exist_ok=True, parents=True)
    destination_filename = optimization_config["output_img_name"].split(".")[0]

    source_img_filename = sorted(list(results_path.glob("*.png")))[-1]
    logger.info(f"Copying {source_img_filename}")
    destination_img_file = optimization_config["images_path"] / (
        destination_filename + ".png"
    )
    shutil.copy(source_img_filename, destination_img_file)

    if optimization_config["gif"]:
        source_gif_filename = list(results_path.glob("*.gif"))[0]
        logger.info(f"Copying {source_gif_filename}")
        destination_video_file = optimization_config["gifs_path"] / (
            "g" + destination_filename + ".gif"
        )
        logger.info(f"Copying to {destination_video_file}")
        shutil.copy(source_gif_filename, destination_video_file)

    shutil.rmtree(results_path, ignore_errors=True)


def make_gif(config, results_path):
    logger.info("Creating gif..")

    results = list(sorted(results_path.glob("*.png")))[-1]
    transf = Image.open(results)

    content_img_path = config["content_images_dir"] / config["content_img_name"]
    orig = Image.open(content_img_path)
    orig = orig.resize((transf.width, transf.height), Image.ANTIALIAS)

    images = []
    for i in range(0, 255, 15):
        orig_new = orig.copy()
        orig_new.putalpha(255 - i)

        transf_new = transf.copy()
        transf_new.putalpha(i)

        new = Image.alpha_composite(orig_new, transf_new)
        images.append(new)

    images = images + images[::-1]
    imageio.mimsave(f"{results_path}/out.gif", images, duration=0.01)


def main(optimization_config):
    start = time()

    default_resource_dir = Path(os.path.dirname(__file__)) / "data"
    content_images_dir = default_resource_dir / "content-images"
    style_images_dir = default_resource_dir / "style-images"
    output_img_dir = default_resource_dir / "output-images"
    img_format = (4, ".png")  # saves images in the format: %04d.png

    output_path = default_resource_dir / optimization_config["output_path"]
    images_path = output_path / "images"
    gif_path = output_path / "gifs"

    # just wrapping settings into a dictionary
    optimization_config["content_images_dir"] = content_images_dir
    optimization_config["style_images_dir"] = style_images_dir
    optimization_config["output_img_dir"] = output_img_dir
    optimization_config["images_path"] = images_path
    optimization_config["gifs_path"] = gif_path
    optimization_config["img_format"] = img_format

    logger.debug(optimization_config)

    if optimization_config["algorithm"] == "original":
        model = Original(optimization_config)
    elif optimization_config["algorithm"] == "pama":
        model = PAMA(optimization_config)
    else:
        raise ValueError("Invalid algorithm provided")
    images_path = model.predict()

    if optimization_config["gif"]:
        make_gif(optimization_config, images_path)

    if (upscale_factor := optimization_config["upscale"]) is not None:
        upscaler = Upscaler(upscale_factor)
        upscaler(images_path)

    # copy results to the respective folder
    copy_output(optimization_config, images_path)

    logger.info(f"Time elapsed: {time()-start}")

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--content_img_name", type=str, help="content image name", default="adamo.jpg"
    )
    parser.add_argument(
        "--style_img_name", type=str, help="style image name", default="yellow_accompaniment.jpg"
    )
    parser.add_argument(
        "--output_img_name", type=str, help="output image name", default="test.png"
    )
    parser.add_argument("--output_path", type=str, help="output path", default="output")
    parser.add_argument(
        "--height",
        type=int,
        nargs="+",
        help="height of content and style images",
        default=500,
    )

    parser.add_argument(
        "--content_weight",
        type=float,
        help="weight factor for content loss",
        default=1e5,
    )
    parser.add_argument(
        "--style_weight", type=float, help="weight factor for style loss", default=3e4
    )
    parser.add_argument(
        "--tv_weight",
        type=float,
        help="weight factor for total variation loss",
        default=1e0,
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        choices=["original", "pama"],
        help="Neural Style Transfer model",
        default="original",
    )

    parser.add_argument(
        "--optimizer", type=str, choices=["lbfgs", "adam"], default="lbfgs"
    )
    parser.add_argument(
        "--model", type=str, choices=["vgg16", "vgg19"], default="vgg19"
    )
    parser.add_argument(
        "--init_method",
        type=str,
        choices=["random", "content", "style"],
        default="content",
    )
    parser.add_argument("--gif", dest="gif", action="store_true")
    parser.add_argument("--no-gif", dest="gif", action="store_false")
    parser.add_argument(
        "--saving_freq",
        type=int,
        help="saving frequency for intermediate images (-1 means only final)",
        default=-1,
    )
    parser.add_argument(
        "--upscale", type=int, choices=[2, 4, 8], help="upscaling factor", nargs="?"
    )
    parser.set_defaults(gif=False)
    args = parser.parse_args()

    optimization_config = dict()
    for arg in vars(args):
        optimization_config[arg] = getattr(args, arg)

    main(optimization_config)
