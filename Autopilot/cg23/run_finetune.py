from model import build_cnn, build_finetune_model
from tensorflow.core.protobuf import saver_pb2
import driving_data
import tensorflow._api.v2.compat.v1 as tf
import os

tf.disable_v2_behavior()
LOGDIR = './save'

# 构建微调模型，指定微调的层数
pretrained_model = build_cnn(weights_path="./save/carla_original.ckpt")
fine_tune_model = build_finetune_model(pretrained_model)
if not os.path.exists(LOGDIR):
    os.makedirs(LOGDIR)
sess = tf.InteractiveSession()
L2NormConst = 1e-4

# 定义损失函数
y_true = tf.placeholder(tf.float32, shape=[None, 1])
loss = tf.reduce_mean(tf.square(tf.subtract(y_true, fine_tune_model.output))) + tf.add_n(
    [tf.nn.l2_loss(v) for v in tf.trainable_variables()]) * L2NormConst
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
sess.run(tf.global_variables_initializer())

# 创建一个监控cost tensor的summary
train_loss_summary = tf.summary.scalar("transfer_learning_loss", loss)
avg_val_loss_placeholder = tf.placeholder(tf.float32, shape=(), name='transfer_val_loss')
avg_val_loss_summary = tf.summary.scalar("transfer_val_loss", avg_val_loss_placeholder)
saver = tf.train.Saver(write_version=saver_pb2.SaverDef.V2)
logs_path = './finetune_logs/original2california'
summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

# 定义常见的训练参数
epochs = 100  # 微调只需少量 epoch
batch_size = 64
min_val_loss = float('inf')  # 只有当验证集的loss小于min_val_loss时,才保存模型

# 微调训练
for epoch in range(epochs):
    for step in range(int(driving_data.num_train_images / batch_size)):
        xs, ys = driving_data.LoadTrainBatch(batch_size)
        train_step.run(feed_dict={fine_tune_model.input: xs, y_true: ys})
        # write logs at every iteration
        summary = train_loss_summary.eval(feed_dict={fine_tune_model.input: xs, y_true: ys})
        summary_writer.add_summary(summary, epoch * driving_data.num_train_images / batch_size + step)

    # 每个epoch结束后,计算验证集的loss
    val_loss_values = []
    for step in range(int(driving_data.num_val_images / batch_size)):
        xs, ys = driving_data.LoadValBatch(batch_size)
        val_loss_value = loss.eval(feed_dict={fine_tune_model.input: xs, y_true: ys})
        val_loss_values.append(val_loss_value)
    avg_val_loss = sum(val_loss_values) / len(val_loss_values)
    print("Epoch: %d, Validation Loss: %g" % (epoch, avg_val_loss))

    # write validation loss to Tensorboard
    avg_val_loss_summary_str = avg_val_loss_summary.eval(feed_dict={avg_val_loss_placeholder: avg_val_loss})
    summary_writer.add_summary(avg_val_loss_summary_str, epoch)

    # save the model after each epoch
    if avg_val_loss < min_val_loss:
        min_val_loss = avg_val_loss
        checkpoint_path = os.path.join(LOGDIR, "carla_original2california.ckpt")
        filename = saver.save(sess, checkpoint_path)
        print("Fine-tuned model saved at epoch %d" % epoch)

print("Fine-tuning completed.")
