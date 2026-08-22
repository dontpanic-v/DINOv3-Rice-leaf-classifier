import timm
from torchvision import transforms

TIER_CONFIG = {
    "light": {
        "rotation_degrees": 15,
        "color_jitter": None,
        "random_grayscale_p": 0.0,
        "random_affine": None,
        "random_erasing_p": 0.0,
    },
    "moderate": {
        "rotation_degrees": 20,
        "color_jitter": dict(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        "random_grayscale_p": 0.0,
        "random_affine": dict(degrees=0, translate=(0.1, 0.1)),
        "random_erasing_p": 0.2,
    },
    "heavy": {
        "rotation_degrees": 30,
        "color_jitter": dict(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15),
        "random_grayscale_p": 0.2,
        "random_affine": dict(degrees=0, translate=(0.15, 0.15), scale=(0.85, 1.15)),
        "random_erasing_p": 0.25,
    },
}


def resolve_data_config(backbone_name, input_size):
    model = timm.create_model(backbone_name, pretrained=False, num_classes=0)
    model.set_input_size(img_size=input_size)
    return timm.data.resolve_model_data_config(model)


def build_transform(tier, image_size, mean, std):
    config = TIER_CONFIG[tier]
    steps = [
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=config["rotation_degrees"]),
    ]
    if config["color_jitter"]:
        steps.append(transforms.ColorJitter(**config["color_jitter"]))
    if config["random_grayscale_p"] > 0:
        steps.append(transforms.RandomGrayscale(p=config["random_grayscale_p"]))
    if config["random_affine"]:
        steps.append(transforms.RandomAffine(**config["random_affine"]))
    steps.append(transforms.ToTensor())
    steps.append(transforms.Normalize(mean=mean, std=std))
    if config["random_erasing_p"] > 0:
        steps.append(transforms.RandomErasing(p=config["random_erasing_p"]))
    return transforms.Compose(steps)
