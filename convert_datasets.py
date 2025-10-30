import json
import numpy as np
import os
import argparse


def merge_json(input_dir, opengl_system=False):

    input_image_dir = os.path.join(input_dir, "cameras")
    all_images = os.listdir(input_image_dir)
    num_train = int(len(all_images) * 0.9)
    train_split = all_images[:num_train]
    test_split = all_images[num_train:]
    
    output_dict_train = {
        "camera_model": "PINHOLE",
        "frames": [],
    }
    print(f"Number of training images: {len(train_split)}")
    print(f"Number of test images: {len(test_split)}")
    for name in train_split:
        if name.endswith('.json'):
            with open(os.path.join(input_image_dir, name), 'r') as f:
                data = json.load(f)
                K = np.array(data['K'], dtype=np.float64).reshape((4, 4))
                if "h" not in output_dict_train:
                    output_dict_train["w"], output_dict_train["h"] = data["img_size"]
                frame_dict = {
                    "file_path": "./images/" + name.replace(".json", ".png"),
                    "fl_x": K[0, 0],
                    "fl_y": K[1, 1],
                    "cx": K[0, 2],
                    "cy": K[1, 2],
                }
                w2c = np.array(data['W2C'], dtype=np.float64).reshape((4, 4))
                c2w = np.linalg.inv(w2c)
                # from opencv to opengl
                if opengl_system:
                    c2w[0:3,2] *= -1 # flip the y and z axis
                    c2w[0:3,1] *= -1
                    c2w = c2w[[1,0,2,3],:]
                    c2w[2,:] *= -1 # flip whole world upside down

                frame_dict["transform_matrix"] = c2w.tolist()
                output_dict_train["frames"].append(frame_dict)
    
    with open(os.path.join(input_dir, "transforms_train.json"), 'w') as f:
        json.dump(output_dict_train, f, indent=2, sort_keys=True)

    output_dict_test = {
        "camera_model": "PINHOLE",
        "frames": [],
    }
    for name in test_split:
        if name.endswith('.json'):
            with open(os.path.join(input_image_dir, name), 'r') as f:
                data = json.load(f)
                K = np.array(data['K'], dtype=np.float64).reshape((4, 4))
                if "h" not in output_dict_test:
                    output_dict_test["w"], output_dict_test["h"] = data["img_size"]
                frame_dict = {
                    "file_path": "./images/" + name.replace(".json", ".png"),
                    "fl_x": K[0, 0],
                    "fl_y": K[1, 1],
                    "cx": K[0, 2],
                    "cy": K[1, 2],
                }
                w2c = np.array(data['W2C'], dtype=np.float64).reshape((4, 4))
                c2w = np.linalg.inv(w2c)
                # from opencv to opengl
                if opengl_system:
                    c2w[0:3,2] *= -1 # flip the y and z axis
                    c2w[0:3,1] *= -1
                    c2w = c2w[[1,0,2,3],:]
                    c2w[2,:] *= -1 # flip whole world upside down

                frame_dict["transform_matrix"] = c2w.tolist()
                output_dict_test["frames"].append(frame_dict)
    
    with open(os.path.join(input_dir, "transforms_test.json"), 'w') as f:
        json.dump(output_dict_test, f, indent=2, sort_keys=True)

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', type=str, default=None, help='Folder containing input data.')

    args = parser.parse_args()
    merge_json(args.input_folder)
