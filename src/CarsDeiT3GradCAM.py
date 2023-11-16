import argparse
import cv2
import numpy as np
import torch
from torchvision.io.image import read_image, ImageReadMode
from torchvision.transforms.functional import normalize, resize, to_pil_image

from pytorch_grad_cam import GradCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM, \
    FullGrad

from pytorch_grad_cam.utils.image import show_cam_on_image, \
    preprocess_image
from pytorch_grad_cam.ablation_layer import AblationLayerVit
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--image-path',
        type=str,
        default='/storage/Cars/test_images/Hyundai Elantra Sedan 2007/00014.jpg',
        help='Input image path')
    parser.add_argument('--aug_smooth', action='store_true',
                        help='Apply test time augmentation to smooth the CAM')
    parser.add_argument(
        '--eigen_smooth',
        action='store_true',
        help='Reduce noise by taking the first principle componenet'
        'of cam_weights*activations')

    parser.add_argument(
        '--method',
        type=str,
        default='gradcam',
        help='Can be gradcam/gradcam++/scorecam/xgradcam/ablationcam')

    args = parser.parse_args()
    args.use_cuda = True
    if args.use_cuda:
        print('Using GPU for acceleration')
    else:
        print('Using CPU for computation')

    return args

import os

root='/storage/Cars/test_images'
classes = [d.name for d in os.scandir(root) if d.is_dir()]

images = []
for i, c in enumerate(classes):
    for f in os.listdir(os.path.join(root, c)):
        images += [os.path.join(root, c, f)]

import random
random.shuffle(images)
        
def make_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)
        
def reshape_transform(tensor, height=14, width=14):
    result = tensor[:, 1:, :].reshape(tensor.size(0),
                                      height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result


if __name__ == '__main__':
    """ python vit_gradcam.py --image-path <path_to_image>
    Example usage of using cam-methods on a VIT network.

    """

    args = get_args()
    methods = \
        {"gradcam": GradCAM,
         "scorecam": ScoreCAM,
         "gradcam++": GradCAMPlusPlus,
         "ablationcam": AblationCAM,
         "xgradcam": XGradCAM,
         "eigencam": EigenCAM,
         "eigengradcam": EigenGradCAM,
         "layercam": LayerCAM,
         "fullgrad": FullGrad}

    if args.method not in list(methods.keys()):
        raise Exception(f"method should be one of {list(methods.keys())}")

    print("1========================================================================")

    NUM_FEATURES = 72
    NUM_CLASSES = 196
    
    import timm
    model = timm.create_model("deit3_medium_patch16_224.fb_in22k_ft_in1k", pretrained=True).cuda()
    model.head = torch.nn.Linear(512, NUM_FEATURES).cuda()
    model.load_state_dict(torch.load("CarsDeiT3Auto.pt"))
    model.eval()
    
    from SubsetLoss import BSSLoss
    loss_fn = BSSLoss(NUM_FEATURES, add_predicate_matrix=True, n_classes=NUM_CLASSES).cuda()
    loss_fn.load_state_dict(torch.load("CarsDeiT3AutoLossFN.pt"))
    loss_fn.eval()

    print("2========================================================================")

    target_layers = [model.blocks[-1].norm1]

    cam = methods[args.method](model=model,
                                   target_layers=target_layers,
                                   use_cuda=args.use_cuda,
                                   reshape_transform=reshape_transform)
    cam.batch_size = 32
    
    predicate_matrix = loss_fn.get_predicate_matrix()
    print(predicate_matrix.shape)
    
    from tqdm import tqdm
    for i, image in enumerate(tqdm(images[:100])):
        c = classes.index(image.split("/")[-2])
        rgb_img = cv2.imread(image, 1)[:, :, ::-1]
        rgb_img = cv2.resize(rgb_img, (224, 224))
        rgb_img = np.float32(rgb_img) / 255
        input_tensor = preprocess_image(rgb_img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        voutput = model(input_tensor.cuda())
        
        make_dir(f"results/Cars-CAM")
        make_dir(f"results/Cars-CAM/img{i}")
        for j in range(NUM_FEATURES):
            if voutput[0][j] > 0.5:
                grayscale_cam = cam(input_tensor=input_tensor,
                            targets=[ClassifierOutputTarget(j)],
                            eigen_smooth=args.eigen_smooth,
                            aug_smooth=args.aug_smooth)

                grayscale_cam = grayscale_cam[0, :]

                cam_image = show_cam_on_image(rgb_img, grayscale_cam)
                
                if predicate_matrix[c][j] == 1:
                    cv2.imwrite(f"results/Cars-CAM/img{i}/feature{j}.jpg", cam_image)
                else:
                    cv2.imwrite(f"results/Cars-CAM/img{i}/NOTfeature{j}.jpg", cam_image)
