import tensorflow as tf


from core.efficientdet import EfficientDet, PostProcessing
from data.dataloader import DetectionDataset, DataLoader
from configuration import Config
from utils.visualize import visualize_training_results

import numpy as np
import matplotlib.pyplot as plt


def print_model_summary(network):
    sample_inputs = tf.random.normal(shape=(Config.batch_size, Config.get_image_size()[0], Config.get_image_size()[1], Config.image_channels))
    sample_outputs = network(sample_inputs, training=True)
    network.summary()


if __name__ == '__main__':
    # GPU settings
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    # dataset
    # train에 사용할 데이터셋을 불러오기
    train_dataset = DetectionDataset("train")
    train_data, train_size = train_dataset.generate_datatset()
    data_loader = DataLoader()
    train_steps_per_epoch = tf.math.ceil(train_size / Config.batch_size)

    # validation loss 계산에 사용할 데이터셋 불러오기
    valid_dataset = DetectionDataset("valid")
    valid_data, valid_size = valid_dataset.generate_datatset()
    valid_steps_per_epoch = tf.math.ceil(train_size / Config.batch_size)

    # model
    efficientdet = EfficientDet()
    print_model_summary(efficientdet)

    load_weights_from_epoch = Config.load_weights_from_epoch
    if Config.load_weights_before_training:
        efficientdet.load_weights(filepath=Config.save_model_dir+"epoch-{}".format(load_weights_from_epoch))
        print("Successfully load weights!")
    else:
        load_weights_from_epoch = -1

    post_process = PostProcessing()

    # optimizer
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=1e-4,
                                                                 decay_steps=train_steps_per_epoch * Config.learning_rate_decay_epochs,
                                                                 decay_rate=0.96)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)


    # metrics
    loss_metric_train = tf.metrics.Mean()
    loss_metric_valid = tf.metrics.Mean()

    temp_loss_1 = []
    temp_loss_2 = []

    def train_step(batch_images, batch_labels):
        with tf.GradientTape() as tape:
            pred = efficientdet(batch_images, training=True)
            loss_value = post_process.training_procedure(pred, batch_labels)
        gradients = tape.gradient(target=loss_value, sources=efficientdet.trainable_variables)
        optimizer.apply_gradients(grads_and_vars=zip(gradients, efficientdet.trainable_variables))
        loss_metric_train.update_state(values=loss_value)

    # validation set에 대한 loss 계산해주는 함수
    def valid_step(batch_images, batch_labels):
        with tf.GradientTape() as tape:
            pred = efficientdet(batch_images, training=False)
            loss_value = post_process.training_procedure(pred, batch_labels)
        loss_metric_valid.update_state(values=loss_value)

    # # early stop - loss 가 떨어지지 않는 경우 조정해주는 함수
    # def early_stop(val_loss, epoch):
    #     if



    for epoch in range(load_weights_from_epoch + 1, Config.epochs):
        print("Epoch: {}/{} 시작 ".format(epoch, Config.epochs))

        for step, batch_data  in enumerate(train_data):
            images_train, labels_train = data_loader.read_batch_data(batch_data)
            train_step(images_train, labels_train)

            if step%100==0:
                print("step: {}/{}, loss: {}".format(      step,
                                                       train_steps_per_epoch,
                                                       loss_metric_train.result()))

        temp_loss_1.append(loss_metric_train.result())
        loss_metric_train.reset_states()

        for step, batch_data in enumerate(valid_data):
            images, labels = data_loader.read_batch_data(batch_data)
            valid_step(images, labels)
            if step % 100 == 0:
                print("step: {}/{}, val_loss: {}".format(step,
                                                 valid_steps_per_epoch,
                                                 loss_metric_valid.result()))
        temp_loss_2.append(loss_metric_valid.result())

        loss_metric_valid.reset_states()

        if epoch % Config.save_frequency == 0:
            efficientdet.save_weights(filepath=Config.save_model_dir+"epoch-{}".format(epoch), save_format="tf")

        if Config.test_images_during_training:
            visualize_training_results(pictures=Config.test_images_dir_list, model=efficientdet, epoch=epoch)

        if epoch >= 2:
            x_len = np.arange(epoch+1)
            plt.plot(x_len, temp_loss_1, marker='-', c='red', label="Train-set Loss")
            plt.plot(x_len, temp_loss_2, marker='-', c='blue', label="Valid-set Loss")
            plt.legend(loc='upper right')
            plt.grid()
            plt.xlabel('epoch')
            plt.ylabel('step_loss')
            plt.show()


    efficientdet.save_weights(filepath=Config.save_model_dir + "saved_model", save_format="tf")

    x_len = np.arange(Config.epochs)

    plt.plot(x_len, temp_loss_1, marker='-', c='red', label="Train-set Loss")
    plt.plot(x_len, temp_loss_2, marker='-', c='blue', label="Valid-set Loss")
    plt.legend(loc='upper right')
    plt.grid()
    plt.xlabel('epoch')
    plt.ylabel('step_loss')
    plt.show()
