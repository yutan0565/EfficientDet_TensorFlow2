import tensorflow as tf
from configuration import Config
from core.efficientnet import get_efficient_net
from core.bifpn import BiFPN
from core.prediction_net import BoxClassPredict


class EfficientDet(tf.keras.Model):
    def __init__(self):
        super(EfficientDet, self).__init__()
        self.backbone = get_efficient_net(width_coefficient=Config.get_width_coefficient(),
                                          depth_coefficient=Config.get_depth_coefficient(),
                                          dropout_rate=Config.get_dropout_rate())
        self.bifpn = BiFPN(output_channels=Config.get_w_bifpn(), layers=Config.get_d_bifpn())
        self.prediction_net = BoxClassPredict(filters=Config.get_w_bifpn(),
                                              depth=Config.get_d_class(),
                                              num_classes=Config.num_classes,
                                              num_anchors=Config.num_anchor_per_pixel)

    def call(self, inputs, training=None, mask=None):
        """
        :param inputs: 4-D tensor, shape: (N, H, W, C)
        :param training:
        :param mask:
        :return: x: tuple, (box_preds, class_preds)
        """
        x = self.backbone(inputs, training=training)
        x = self.bifpn(x, training=training)
        x = self.prediction_net(x, training=training)
        return x


class PostProcessing:
    def __init__(self):
        super(PostProcessing, self).__init__()
        pass