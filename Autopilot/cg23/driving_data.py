import cv2
import random
import numpy as np
import os

xs = []
ys = []

# points to the end of the last batch
train_batch_pointer = 0
val_batch_pointer = 0

# 在CARLA仿真数据集上训练
# with open(r"F:\PyCharm 2023.1\work\SDGeneralization\Autopilot\driving_dataset\carla_train.txt") as f:
#     for index, line in enumerate(f):
#         # 如果设置只读取前36799条数据(前XX条数据为carla收集的原始数据集,后面的数据为变异增强后的carla数据集)
#         if index > 36799:
#             break
#         line_values = line.split(",")[0].split()
#         image_path = line_values[0]
#         angle = float(line_values[1])
#         xs.append(os.path.join(r'F:\PyCharm 2023.1\work\SDGeneralization\Autopilot\driving_dataset\carla_train', image_path))
#         ys.append(angle * np.pi / 180)

# 在域迁移的udacity数据集上训练
# with open(r"F:\PyCharm 2023.1\work\SDGeneralization\Autopilot\driving_dataset\udacity_train.txt") as f:
#     for index, line in enumerate(f):
#         line_values = line.split(",")[0].split()
#         image_path = line_values[0]
#         angle = float(line_values[1])
#         xs.append(os.path.join(r'F:\PyCharm 2023.1\work\SDGeneralization\Autopilot\driving_dataset\udacity_train', image_path))
#         ys.append(angle * np.pi / 180)

# 在域迁移的california数据集上训练
with open(r"F:\PyCharm 2023.1\work\SDGeneralization\Autopilot\driving_dataset\california_train.txt") as f:
    for index, line in enumerate(f):
        line_values = line.split(",")[0].split()
        image_path = line_values[0]
        angle = float(line_values[1])
        xs.append(os.path.join(r'F:\PyCharm 2023.1\work\SDGeneralization\Autopilot\driving_dataset\california_train', image_path))
        ys.append(angle * np.pi / 180)

# shuffle list of images
c = list(zip(xs, ys))
random.shuffle(c)
xs, ys = zip(*c)

train_xs = xs[:int(len(xs) * 0.8)]
train_ys = ys[:int(len(xs) * 0.8)]

val_xs = xs[-int(len(xs) * 0.2):]
val_ys = ys[-int(len(xs) * 0.2):]

num_train_images = len(train_xs)
num_val_images = len(val_xs)


def LoadTrainBatch(batch_size):
    global train_batch_pointer
    x_out = []
    y_out = []
    for i in range(0, batch_size):
        x_out.append(
            cv2.resize(cv2.imread(train_xs[(train_batch_pointer + i) % num_train_images])[-200:], (128, 128)) / 255.0)
        y_out.append([train_ys[(train_batch_pointer + i) % num_train_images]])
    train_batch_pointer += batch_size
    return x_out, y_out


def LoadValBatch(batch_size):
    global val_batch_pointer
    x_out = []
    y_out = []
    for i in range(0, batch_size):
        x_out.append(cv2.resize(cv2.imread(val_xs[(val_batch_pointer + i) % num_val_images])[-200:], (128, 128)) / 255.0)
        y_out.append([val_ys[(val_batch_pointer + i) % num_val_images]])
    val_batch_pointer += batch_size
    return x_out, y_out
