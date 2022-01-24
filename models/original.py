import logging
import os

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.autograd import Variable
from torch.optim import LBFGS, Adam
from torchvision import transforms

from models.definitions.vgg_nets import Vgg16, Vgg16Experimental, Vgg19

logger = logging.getLogger(__name__)

IMAGENET_MEAN_255 = [123.675, 116.28, 103.53]
IMAGENET_STD_NEUTRAL = [1, 1, 1]
ITERATIONS_DICT = {
    "lbfgs": 1000,
    "adam": 1000,
}


class Original:
    def __init__(self, config):
        self.config = config
        self.content_img_path = (
            config["content_images_dir"] / config["content_img_name"]
        )
        self.style_img_path = config["style_images_dir"] / config["style_img_name"]
        self.optimizer = config["optimizer"]
        self.height = config["height"]
        self.iterations = ITERATIONS_DICT[self.optimizer]
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

    def load_image(self, img_path, target_shape=None):
        if not os.path.exists(img_path):
            raise Exception(f"Path does not exist: {img_path}")
        img = cv.imread(img_path)[
            :, :, ::-1
        ]  # [:, :, ::-1] converts BGR (opencv format...) into RGB

        if target_shape is not None:  # resize section
            if (
                isinstance(target_shape, int) and target_shape != -1
            ):  # scalar -> implicitly setting the height
                current_height, current_width = img.shape[:2]
                new_height = target_shape
                new_width = int(current_width * (new_height / current_height))
                img = cv.resize(
                    img, (new_width, new_height), interpolation=cv.INTER_CUBIC
                )
            else:  # set both dimensions to target shape
                img = cv.resize(
                    img,
                    (target_shape[1], target_shape[0]),
                    interpolation=cv.INTER_CUBIC,
                )

        # this need to go after resizing - otherwise cv.resize will push values outside of [0,1] range
        img = img.astype(np.float32)  # convert from uint8 to float32
        img /= 255.0  # get to [0, 1] range
        return img

    def prepare_img(self, img_path, target_shape, device):
        img = self.load_image(img_path, target_shape=target_shape)

        # normalize using ImageNet's mean
        # [0, 255] range worked much better for me than [0, 1] range (even though PyTorch models were trained on latter)
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.mul(255)),
                transforms.Normalize(mean=IMAGENET_MEAN_255, std=IMAGENET_STD_NEUTRAL),
            ]
        )

        img = transform(img).to(device).unsqueeze(0)

        return img
        
    def generate_out_img_name(self):
        prefix = (
            os.path.basename(self.config["content_img_name"]).split(".")[0]
            + "_"
            + os.path.basename(self.config["style_img_name"]).split(".")[0]
        )
        # called from the reconstruction script
        if "reconstruct_script" in self.config:
            suffix = f'_o_{self.optimizer}_h_{str(self.height)}_m_{self.config["model"]}{self.onfig["img_format"][1]}'
        else:
            suffix = f'_o_{self.optimizer}_i_{self.config["init_method"]}_h_{str(self.height)}_m_{self.config["model"]}_cw_{self.config["content_weight"]}_sw_{self.config["style_weight"]}_tv_{self.config["tv_weight"]}{self.config["img_format"][1]}'
        return prefix + suffix

    def save_and_maybe_display(
        self, optimizing_img, dump_path, img_id, should_display=False
    ):
        saving_freq = self.config["saving_freq"]
        out_img = optimizing_img.squeeze(axis=0).to("cpu").detach().numpy()
        out_img = np.moveaxis(
            out_img, 0, 2
        )  # swap channel from 1st to 3rd position: ch, _, _ -> _, _, chr

        # for saving_freq == -1 save only the final result (otherwise save with frequency saving_freq and save the last pic)
        if (
            img_id == self.iterations - 1
            or (saving_freq > 0 and img_id % saving_freq == 0)
            or ((saving_freq > 0) and (img_id < 20))
        ):
            img_format = self.config["img_format"]
            out_img_name = (
                str(img_id).zfill(img_format[0]) + img_format[1]
                if saving_freq != -1
                else self.generate_out_img_name()
            )
            dump_img = np.copy(out_img)
            dump_img += np.array(IMAGENET_MEAN_255).reshape((1, 1, 3))
            dump_img = np.clip(dump_img, 0, 255).astype("uint8")
            cv.imwrite(os.path.join(dump_path, out_img_name), dump_img[:, :, ::-1])

        if should_display:
            plt.imshow(np.uint8(self.get_uint8_range(out_img)))
            plt.show()

    @staticmethod
    def get_uint8_range(x):
        if isinstance(x, np.ndarray):
            x -= np.min(x)
            x /= np.max(x)
            x *= 255
            return x
        else:
            raise ValueError(f"Expected numpy array got {type(x)}")

    @staticmethod
    def prepare_model(model, device):
        # we are not tuning model weights -> we are only tuning optimizing_img's pixels! (that's why requires_grad=False)
        experimental = False
        if model == "vgg16":
            if experimental:
                # much more flexible for experimenting with different style representations
                model = Vgg16Experimental(requires_grad=False, show_progress=True)
            else:
                model = Vgg16(requires_grad=False, show_progress=True)
        elif model == "vgg19":
            model = Vgg19(requires_grad=False, show_progress=True)
        else:
            raise ValueError(f"{model} not supported.")

        content_feature_maps_index = model.content_feature_maps_index
        style_feature_maps_indices = model.style_feature_maps_indices
        layer_names = model.layer_names

        content_fms_index_name = (
            content_feature_maps_index,
            layer_names[content_feature_maps_index],
        )
        style_fms_indices_names = (style_feature_maps_indices, layer_names)
        return model.to(device).eval(), content_fms_index_name, style_fms_indices_names

    @staticmethod
    def gram_matrix(x, should_normalize=True):
        (b, ch, h, w) = x.size()
        features = x.view(b, ch, w * h)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t)
        if should_normalize:
            gram /= ch * h * w
        return gram

    @staticmethod
    def total_variation(y):
        return torch.sum(torch.abs(y[:, :, :, :-1] - y[:, :, :, 1:])) + torch.sum(
            torch.abs(y[:, :, :-1, :] - y[:, :, 1:, :])
        )

    def build_loss(
        self,
        neural_net,
        optimizing_img,
        target_representations,
        content_feature_maps_index,
        style_feature_maps_indices,
    ):
        target_content_representation = target_representations[0]
        target_style_representation = target_representations[1]

        current_set_of_feature_maps = neural_net(optimizing_img)

        current_content_representation = current_set_of_feature_maps[
            content_feature_maps_index
        ].squeeze(axis=0)
        content_loss = torch.nn.MSELoss(reduction="mean")(
            target_content_representation, current_content_representation
        )

        style_loss = 0.0
        current_style_representation = [
            self.gram_matrix(x)
            for cnt, x in enumerate(current_set_of_feature_maps)
            if cnt in style_feature_maps_indices
        ]
        for gram_gt, gram_hat in zip(
            target_style_representation, current_style_representation
        ):
            style_loss += torch.nn.MSELoss(reduction="sum")(gram_gt[0], gram_hat[0])
        style_loss /= len(target_style_representation)

        tv_loss = self.total_variation(optimizing_img)

        total_loss = (
            self.config["content_weight"] * content_loss
            + self.config["style_weight"] * style_loss
            + self.config["tv_weight"] * tv_loss
        )

        return total_loss, content_loss, style_loss, tv_loss

    def make_tuning_step(
        self,
        neural_net,
        optimizer,
        target_representations,
        content_feature_maps_index,
        style_feature_maps_indices,
    ):
        # Builds function that performs a step in the tuning loop
        def tuning_step(optimizing_img):
            total_loss, content_loss, style_loss, tv_loss = self.build_loss(
                neural_net,
                optimizing_img,
                target_representations,
                content_feature_maps_index,
                style_feature_maps_indices,
            )
            # Computes gradients
            total_loss.backward()
            # Updates parameters and zeroes gradients
            optimizer.step()
            optimizer.zero_grad()
            return total_loss, content_loss, style_loss, tv_loss

        # Returns the function that will be called inside the tuning loop
        return tuning_step

    def predict(self):
        logger.info("Starting prediction")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.debug(f"Using device {device} for training")

        content_img = self.prepare_img(str(self.content_img_path), self.height, device)
        style_img = self.prepare_img(str(self.style_img_path), self.height, device)

        if self.config["init_method"] == "random":
            # white_noise_img = np.random.uniform(-90., 90., content_img.shape).astype(np.float32)
            gaussian_noise_img = np.random.normal(
                loc=0, scale=90.0, size=content_img.shape
            ).astype(np.float32)
            init_img = torch.from_numpy(gaussian_noise_img).float().to(device)
        elif self.config["init_method"] == "content":
            init_img = content_img
        else:
            # init image has same dimension as content image - this is a hard constraint
            # feature maps need to be of same size for content image and init image
            style_img_resized = self.prepare_img(
                self.style_img_path, np.asarray(content_img.shape[2:]), device
            )
            init_img = style_img_resized

        # we are tuning optimizing_img's pixels! (that's why requires_grad=True)
        optimizing_img = Variable(init_img, requires_grad=True)

        (
            neural_net,
            content_feature_maps_index_name,
            style_feature_maps_indices_names,
        ) = self.prepare_model(self.config["model"], device)
        logger.debug(f'Using {self.config["model"]} in the optimization procedure')

        content_img_set_of_feature_maps = neural_net(content_img)
        style_img_set_of_feature_maps = neural_net(style_img)

        target_content_representation = content_img_set_of_feature_maps[
            content_feature_maps_index_name[0]
        ].squeeze(axis=0)
        target_style_representation = [
            self.gram_matrix(x)
            for cnt, x in enumerate(style_img_set_of_feature_maps)
            if cnt in style_feature_maps_indices_names[0]
        ]
        target_representations = [
            target_content_representation,
            target_style_representation,
        ]

        # Start of optimization procedure
        unsuccessful = False
        if self.config["optimizer"] == "adam":
            logger.info("Using ADAM")
            try:
                optimizer = Adam((optimizing_img,), lr=1e1)
                tuning_step = self.make_tuning_step(
                    neural_net,
                    optimizer,
                    target_representations,
                    content_feature_maps_index_name[0],
                    style_feature_maps_indices_names[0],
                )

                for cnt in range(self.ITERATIONS[self.config["optimizer"]]):
                    _ = tuning_step(optimizing_img)
                    with torch.no_grad():
                        self.save_and_maybe_display(
                            optimizing_img,
                            self.dump_path,
                            cnt,
                            should_display=False,
                        )
            except Exception as e:
                logger.info(e)
                unsuccessful = True

        elif self.optimizer == "lbfgs":
            logger.info("Using LBFGS")
            try:
                # line_search_fn does not seem to have significant impact on result
                optimizer = LBFGS(
                    (optimizing_img,),
                    max_iter=self.iterations,
                    line_search_fn="strong_wolfe",
                )
                cnt = 0

                def closure():
                    nonlocal cnt
                    if torch.is_grad_enabled():
                        optimizer.zero_grad()
                    total_loss, _, _, _ = self.build_loss(
                        neural_net,
                        optimizing_img,
                        target_representations,
                        content_feature_maps_index_name[0],
                        style_feature_maps_indices_names[0],
                    )
                    if total_loss.requires_grad:
                        total_loss.backward()
                    with torch.no_grad():
                        self.save_and_maybe_display(
                            optimizing_img,
                            self.dump_path,
                            cnt,
                            should_display=False,
                        )

                    cnt += 1
                    return total_loss

                optimizer.step(closure)
            except Exception as e:
                logger.info(e)
                unsuccessful = True

        logger.debug(f"Executed {cnt} iterations")
        if ((cnt < self.iterations / 10) or unsuccessful == True) & (
            self.optimizer == "lbfgs"
        ):
            self.optimizer = "adam"
            return self.predict()

        logger.info("Training finished")
        return self.dump_path
