import logging
from PIL import Image
import imageio

logger = logging.getLogger(__name__)


def make_gif(config, results_path):
    logger.info('Creating gif..')

    results = list(sorted(results_path.glob('*.png')))[-1]
    transf = Image.open(results)

    content_img_path = config['content_images_dir'] / \
        config['content_img_name']
    orig = Image.open(content_img_path)
    orig = orig.resize((transf.width, transf.height), Image.ANTIALIAS)

    images = []
    for i in range(0, 255, 15):
        orig_new = orig.copy()
        orig_new.putalpha(255-i)

        transf_new = transf.copy()
        transf_new.putalpha(i)

        new = Image.alpha_composite(orig_new, transf_new)
        images.append(new)

    images = images + images[::-1]
    imageio.mimsave(f'{results_path}/out.gif', images, duration=0.01)
