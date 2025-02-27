import tensorflow as tf
import numpy as np

from configuration import Config


class DetectionDataset:
    def __init__(self, type):
        if type == "train":
            self.txt_file = Config.train_txt_file_dir
        elif type == "valid":
            self.txt_file = Config.val_txt_file_dir
        self.batch_size = Config.batch_size

    @staticmethod
    def __get_length_of_dataset(dataset):
        length = 0
        for _ in dataset:
            length += 1
        return length



    def generate_datatset(self):
        dataset = tf.data.TextLineDataset(filenames=self.txt_file)
        length_of_dataset = DetectionDataset.__get_length_of_dataset(dataset)
        train_dataset = dataset.batch(batch_size=self.batch_size)
        return train_dataset, length_of_dataset


class DataLoader:

    input_image_height = Config.get_image_size()[0]
    input_image_width = Config.get_image_size()[1]
    resize_mode = Config.resize_mode

    def __init__(self):
        self.max_boxes_per_image = Config.max_boxes_per_image

    def read_batch_data(self, batch_data):
        batch_size = batch_data.shape[0]
        image_file_list = []
        boxes_list = []
        for n in range(batch_size):
            image_file, boxes = self.__get_image_information(single_line=batch_data[n])
            image_file_list.append(image_file)
            boxes_list.append(boxes)
        boxes = np.stack(boxes_list, axis=0)
        image_tensor_list = []
        for image in image_file_list:
            image_tensor = DataLoader.image_preprocess(is_training=True, image_dir=image)
            image_tensor_list.append(image_tensor)
        images = tf.stack(values=image_tensor_list, axis=0)
        return images, boxes

    def __get_image_information(self, single_line):
        """
        :param single_line: tensor
        :return:
        image_file: string, image file dir
        boxes_array: numpy array, shape = (max_boxes_per_image, 5(xmin, ymin, xmax, ymax, class_id))
        """
        line_string = bytes.decode(single_line.numpy(), encoding="utf-8")
        line_list = line_string.strip().split(" ")
        image_file, image_height, image_width = line_list[:3]
        image_height, image_width = int(float(image_height)), int(float(image_width))
        boxes = []
        num_of_boxes = (len(line_list) - 3) / 5
        if int(num_of_boxes) == num_of_boxes:
            num_of_boxes = int(num_of_boxes)
        else:
            raise ValueError("num_of_boxes must be type 'int'.")
        for index in range(num_of_boxes):
            if index < self.max_boxes_per_image:
                xmin = int(float(line_list[3 + index * 5]))
                ymin = int(float(line_list[3 + index * 5 + 1]))
                xmax = int(float(line_list[3 + index * 5 + 2]))
                ymax = int(float(line_list[3 + index * 5 + 3]))
                class_id = int(line_list[3 + index * 5 + 4])
                xmin, ymin, xmax, ymax = DataLoader.box_preprocess(image_height, image_width, xmin, ymin, xmax, ymax)
                boxes.append([xmin, ymin, xmax, ymax, class_id])
        num_padding_boxes = self.max_boxes_per_image - num_of_boxes
        if num_padding_boxes > 0:
            for i in range(num_padding_boxes):
                boxes.append([0, 0, 0, 0, -1])
        boxes_array = np.array(boxes, dtype=np.float32)
        return image_file, boxes_array

    @classmethod
    def box_preprocess(cls, h, w, xmin, ymin, xmax, ymax):
        if DataLoader.resize_mode == "RESIZE":
            resize_ratio = [DataLoader.input_image_height / h, DataLoader.input_image_width / w]
            xmin = int(resize_ratio[1] * xmin)
            xmax = int(resize_ratio[1] * xmax)
            ymin = int(resize_ratio[0] * ymin)
            ymax = int(resize_ratio[0] * ymax)
            return xmin, ymin, xmax, ymax

    @classmethod
    def image_preprocess(cls, is_training, image_dir):
        image_raw = tf.io.read_file(filename=image_dir)
        decoded_image = tf.io.decode_image(contents=image_raw, channels=3, dtype=tf.dtypes.float32)
        if DataLoader.resize_mode == "RESIZE":
            decoded_image = tf.image.resize(images=decoded_image, size=(DataLoader.input_image_height, DataLoader.input_image_width))
        return decoded_image
