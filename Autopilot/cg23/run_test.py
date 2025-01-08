import tensorflow._api.v2.compat.v1 as tf

tf.disable_v2_behavior()
from model import build_cnn
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import openpyxl


def run_dataset(testing_data_path):
    # 给定测试集的路径,读取图片并测试模型
    smoothed_angle = 0

    images = [img for img in os.listdir(testing_data_path) if img.endswith(('.png', '.jpg', '.jpeg'))]
    images.sort(key=lambda x: int(os.path.splitext(x)[0]))  # 按文件名数字顺序排序

    # 先删除predicted_steers.txt文件,再创建一个新的
    if os.path.exists('../predicted_steers.txt'):
        os.remove('../predicted_steers.txt')

    # 将当前的预测结果写入predicted_steers.txt
    with open('../predicted_steers.txt', 'a') as f:
        for _, filename in enumerate(images):
            # 不可视化测试过程
            full_image = cv2.imread(os.path.join(testing_data_path, filename))
            image = cv2.resize(full_image[-200:], (128, 128)) / 255.0
            degrees = model.output.eval(feed_dict={model.input: [image]})[0][0] * 180.0 / np.pi
            smoothed_angle += 0.2 * pow(abs((degrees - smoothed_angle)), 2.0 / 3.0) * (degrees - smoothed_angle) / abs(
                degrees - smoothed_angle)
            f.write(f'{filename} {smoothed_angle}\n')

            # 可视化测试过程
            # img = cv2.imread('../steering_wheel_image.jpg', 0)
            # rows, cols = img.shape
            # while cv2.waitKey(50) != ord('q'):
            #     full_image = cv2.imread(os.path.join(testing_data_path, filename))
            #     image = cv2.resize(full_image[-200:], (128, 128)) / 255.0
            #     degrees = model.output.eval(feed_dict={model.input: [image]})[0][0] * 180.0 / np.pi
            #     smoothed_angle += 0.2 * pow(abs((degrees - smoothed_angle)), 2.0 / 3.0) * (
            #                 degrees - smoothed_angle) / abs(degrees - smoothed_angle)
            #     f.write(f'{filename} {smoothed_angle}\n')
            #     cv2.imshow("frame", full_image)
            #     M = cv2.getRotationMatrix2D((cols / 2, rows / 2), -smoothed_angle, 1)
            #     dst = cv2.warpAffine(img, M, (cols, rows))
            #     cv2.imshow("steering wheel", dst)
            #     break

    cv2.destroyAllWindows()


def plot_comparative_curves(testing_data_path, truth_angles):
    predicted_steers = []
    actual_steers = []
    image_paths = []

    # 抽取真实的方向盘转角
    with open(truth_angles) as f:
        for line in f:
            line_values = line.split(",")[0].split()
            image_paths.append(line_values[0])
            actual_steers.append(float(line_values[1]) * np.pi / 180)

    # 预测模型的输出
    smoothed_angle = 0
    for _, filename in enumerate(tqdm(image_paths)):
        full_image = cv2.imread(os.path.join(testing_data_path, filename))
        image = cv2.resize(full_image[-200:], (128, 128)) / 255.0
        degrees = model.output.eval(feed_dict={model.input: [image]})[0][0]
        # 平滑转角输出
        smoothed_angle += 0.2 * pow(abs((degrees - smoothed_angle)), 2.0 / 3.0) * (degrees - smoothed_angle) / abs(
            degrees - smoothed_angle)
        predicted_steers.append(smoothed_angle)

    # 在同一副图中绘制预测的和实际的路径
    plt.figure(figsize=(10, 6))
    plt.plot(predicted_steers, 'r.-', label='predict')
    plt.plot(actual_steers, 'b.-', label='truth')
    plt.legend(loc='best')
    plt.title("Predicted vs Truth")
    plt.show()

    # 打印预测值和真实值的RMSE
    rmse = np.sqrt(np.mean((np.array(predicted_steers) - np.array(actual_steers)) ** 2))
    print(f'预测值和真实值之间的RMSE: {rmse}')


if __name__ == '__main__':
    # 恢复模型
    model = build_cnn()
    sess = tf.InteractiveSession()
    saver = tf.train.Saver()
    saver.restore(sess, "save/california.ckpt")

    image_path = '../driving_dataset/california_test'  # 需要测试的图片路径

    # 运行测试集
    # run_dataset(testing_data_path=image_path)

    # optional: 将predicted_steers写入transfer_results.xlsx文件
    # excel_path = os.path.join('../transfer_results.xlsx')
    # workbook = openpyxl.load_workbook(excel_path)
    # worksheet = workbook.worksheets[7]
    # with open('../predicted_steers.txt') as f:
    #     predicted_steers = [line.split()[1] for line in f]
    # for idx, steer in enumerate(predicted_steers):
    #     worksheet.cell(row=idx + 2, column=2, value=float(steer) * np.pi / 180)
    # workbook.save(excel_path)

    # 绘制预测模型和真实模型的对比图
    ground_truth_file = '../driving_dataset/california_test.txt'
    plot_comparative_curves(testing_data_path=image_path, truth_angles=ground_truth_file)
