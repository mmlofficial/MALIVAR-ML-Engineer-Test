import os
import sys
import argparse
import cv2
import numpy as np
from typing import List

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from confignet import ConfigNet, LatentGAN, FaceImageNormalizer

from model import resolve_single
from model.srgan import generator

dataset_directory = "../dataset"

def parse_args(args):
    model_base_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    confignet_model_paths = {
        256: os.path.join(model_base_dir, "confignet_256", "model.json"),
        512: os.path.join(model_base_dir, "confignet_512", "model.json")
    }
    latentgan_model_paths = {
        256: os.path.join(model_base_dir, "latentgan_256", "model.json"),
        512: os.path.join(model_base_dir, "latentgan_512", "model.json")
    }

    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", help="Path to either a directory of images or an individual image", default=None)
    parser.add_argument("--resolution", type=int, help="Path to ConfigNetModel", default=512)
    parser.add_argument("--max_angle", type=int, help="Max angle to turn head in degrees", default=60)
    parser.add_argument("--enable_sr", type=int, help="Enable applying super-resolution to images", default=1)

    args = parser.parse_args(args)
    args.confignet_model_path = confignet_model_paths[args.resolution]
    args.latent_gan_model_path = latentgan_model_paths[args.resolution]

    return args

def process_image(image_path: str, resolution: int) -> List[np.ndarray]:
    '''Load the input images and normalize them'''
    if os.path.isfile(image_path):
        img = cv2.imread(image_path)
        img = FaceImageNormalizer.normalize_individual_image(img, (resolution, resolution))
        return [img]
    else:
        raise ValueError("Image path is neither directory nor file")

def get_new_embeddings(input_images, latentgan_model: LatentGAN, confignet_model: ConfigNet):
    '''Samples new embeddings from either:
        - the LatentGAN if no input images were provided
        - by embedding the input images into the latent space using the real encoder
    '''
    if input_images is None:
        
        embeddings = latentgan_model.generate_latents(1, truncation=0.7)
        rotations = np.zeros((1, 3), dtype=np.float32)
        orig_images = confignet_model.generate_images(embeddings, rotations)
    else:
        sample_indices = np.random.randint(0, len(input_images), 1)
        orig_images = np.array([input_images[x] for x in sample_indices])
        embeddings, rotations = confignet_model.encode_images(orig_images)

    return embeddings, rotations, orig_images

def set_gaze_direction_in_embedding(latents: np.ndarray, eye_pose: np.ndarray, confignet_model: ConfigNet) -> np.ndarray:
    '''Sets the selected eye pose in the specified latent variables
       This is accomplished by passing the eye pose through the synthetic data encoder
       and setting the corresponding part of the latent vector.
    '''
    latents = confignet_model.set_facemodel_param_in_latents(latents, "bone_rotations:left_eye", eye_pose)
    return latents

def get_embedding_with_new_attribute_value(parameter_name:str, latents: np.ndarray, confignet_model: ConfigNet) -> np.ndarray:
    '''Samples a new value of the currently controlled face attribute and sets in the latent embedding'''

    new_param_value = confignet_model.facemodel_param_distributions[parameter_name].sample(1)[0]
    modified_latents = confignet_model.set_facemodel_param_in_latents(latents, parameter_name, new_param_value)

    return modified_latents

def to_grad(rad):
    return rad * 180 / np.pi

def to_rad(grad):
    return grad * np.pi / 180


def run(args):
    args = parse_args(args)
    if args.image_path is not None:
        input_images = process_image(args.image_path, args.resolution)
        latentgan_model = None
    else:
        input_images = None
        print("WARNING: no input image directory specified, embeddings will be sampled using Laten GAN")
        latentgan_model = LatentGAN.load(args.latent_gan_model_path)
    confignet_model = ConfigNet.load(args.confignet_model_path)

    #basic_ui = BasicUI(confignet_model)

    # Sample latent embeddings from input images if available and if not sample from Latent GAN
    current_embedding_unmodified, current_rotation, orig_images = get_new_embeddings(input_images, latentgan_model, confignet_model)
    # Set next embedding value for rendering
    if args.enable_sr == 1:
        modelSR = generator()
        modelSR.load_weights('evaluation/weights/srgan/gan_generator.h5')
   
    yaw_min_angle = -args.max_angle
    pitch_min_angle = -args.max_angle
    yaw_max_angle = args.max_angle
    pitch_max_angle = args.max_angle
    delta_angle = 5

    rotation_offset = np.zeros((1, 3))
    
    eye_rotation_offset = np.zeros((1, 3))

    facemodel_param_names = list(confignet_model.config["facemodel_inputs"].keys())
    # remove eye rotation as in the demo it is controlled separately
    eye_rotation_param_idx = facemodel_param_names.index("bone_rotations:left_eye")
    facemodel_param_names.pop(eye_rotation_param_idx)

    render_input_interp_0 = current_embedding_unmodified
    render_input_interp_1 = current_embedding_unmodified


    interpolation_coef = 0
    if not os.path.exists(dataset_directory):
        os.makedirs(dataset_directory)
    # This interpolates between the previous and next set embeddings
    current_renderer_input = render_input_interp_0 * (1 - interpolation_coef) + render_input_interp_1 * interpolation_coef
    # Set eye gaze direction as controlled by the user
    current_renderer_input = set_gaze_direction_in_embedding(current_renderer_input, eye_rotation_offset, confignet_model)
    
    # all angles
    #image = Image.open(args.image_path)
    #print(np.array(image))
    #return
    i = 1
    print('All angles')
    for yaw in range(yaw_min_angle, yaw_max_angle+1, delta_angle):
        for pitch in range(pitch_min_angle, pitch_max_angle+1, delta_angle):
            rotation_offset[0, 0] = to_rad(yaw)
            rotation_offset[0, 1] = to_rad(pitch)
            generated_imgs = confignet_model.generate_images(current_renderer_input, current_rotation + rotation_offset)
            if args.enable_sr == 1:
                img = cv2.resize(generated_imgs[0], (256,256))
                sr_img = resolve_single(modelSR, img)
                cv2.imwrite(dataset_directory + '/%d_%d.png'%(yaw, pitch), np.array(sr_img))
            else:
                img = cv2.resize(generated_imgs[0], (1024,1024))
                cv2.imwrite(dataset_directory + '/%d_%d.png'%(yaw, pitch), img)
            print(i)
            i+=1
            
    #all random
    # 100 картинок со случайными поворотами от -20 до 20, поворотами глаз, выражений лица
    print('All random')
    current_attribute_name = facemodel_param_names[1] #blendshape_values
    frame_embedding = render_input_interp_0 * (1 - interpolation_coef) + render_input_interp_1 * interpolation_coef
    for i in range(100):
        eye_rotation_offset[0, 2] = to_rad(np.random.randint(-40, 40))
        eye_rotation_offset[0, 0] = to_rad(np.random.randint(-40, 40))
        rotation_offset[0, 0] = to_rad(np.random.randint(-20, 20))
        rotation_offset[0, 1] = to_rad(np.random.randint(-20, 20))
        frame_embedding = set_gaze_direction_in_embedding(frame_embedding, eye_rotation_offset, confignet_model)
        new_embedding_value = get_embedding_with_new_attribute_value(current_attribute_name, frame_embedding, confignet_model)

        generated_imgs = confignet_model.generate_images(new_embedding_value, current_rotation + rotation_offset)

        if args.enable_sr == 1:
            img = cv2.resize(generated_imgs[0], (256,256))
            sr_img = resolve_single(modelSR, img)
            cv2.imwrite(dataset_directory + '/random_%d.png'%(i), np.array(sr_img))
        else:
            img = cv2.resize(generated_imgs[0], (1024,1024))
            cv2.imwrite(dataset_directory + '/random_%d.png'%(i), img)
        print(i)


if __name__ == "__main__":
    run(sys.argv[1:])