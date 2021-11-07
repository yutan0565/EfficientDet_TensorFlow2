

class Config:
    epochs = 50 #601
    batch_size = 1 # 32
    learning_rate_decay_epochs = 10

    # save model
    save_model_dir = "saved_model/"
    best_model_dir = "best_model/"

    load_weights_before_training = False
    load_weights_from_epoch = 10   # last로 자동으로 만들어주는 코드 작성
    
    load_weights_from_epoch_quan = 10
    
    save_frequency = 1

    test_images_during_training = False
    training_results_save_dir = "./test_pictures/"
    test_images_dir_list = ["", ""]

    network_type = "D0"

    # image size: (height, width)
    image_size = {"D0": (512, 512), "D1": (640, 640), "D2": (768, 768), "D3": (896, 896), "D4": (1024, 1024),
                  "D5": (1280, 1280), "D6": (1408, 1408), "D7": (1536, 1536)}
    image_channels = 3

    # efficientnet
    width_coefficient = {"D0": 1.0, "D1": 1.0, "D2": 1.1, "D3": 1.2, "D4": 1.4, "D5": 1.6, "D6": 1.8, "D7": 1.8}
    depth_coefficient = {"D0": 1.0, "D1": 1.1, "D2": 1.2, "D3": 1.4, "D4": 1.8, "D5": 2.2, "D6": 2.6, "D7": 2.6}
    dropout_rate = {"D0": 0.2, "D1": 0.2, "D2": 0.3, "D3": 0.3, "D4": 0.4, "D5": 0.4, "D6": 0.5, "D7": 0.5}

    # bifpn channels
    w_bifpn = {"D0": 64, "D1": 88, "D2": 112, "D3": 160, "D4": 224, "D5": 288, "D6": 384, "D7": 384}
    # bifpn layers
    d_bifpn = {"D0": 2, "D1": 3, "D2": 4, "D3": 5, "D4": 6, "D5": 7, "D6": 8, "D7": 8}
    # box/class layers
    d_class = {"D0": 3, "D1": 3, "D2": 3, "D3": 4, "D4": 4, "D5": 4, "D6": 5, "D7": 5}

    # nms
    score_threshold = 0.000001  # 0.01
    iou_threshold = 0.5
    max_box_num = 1

    # dataset
    num_classes = 1

    #여기서 사용하게 될 데이터 먼저  고르기
    pascal_voc_root = "./data/pick_smoke/" # "./data/fire_smoke_aug/"
    pascal_voc_classes = {"smoke": 0}
    max_boxes_per_image = 20
    resize_mode = "RESIZE"

    # txt file
    # 해당 데이터셋에 맞는 annotation 파일 생성
    type = "train"
    load_flag = type
    work_type = type  ## 데이터 분리할때 사용
    train_txt_file_dir = pascal_voc_root + "train_annotations.txt"
    val_txt_file_dir = pascal_voc_root +  "val_annotations.txt"
    test_txt_file_dir = pascal_voc_root + "test_annotations.txt"

    # test image    --
    #연기 샘플
    test_image_dir = "./test_pictures/ck0k9etuqjxhh0848k6i1mw2f_jpeg.rf.65e16bd6d3135e530e8e7677b83b6481.jpg"

    # 흡연 샘플
    #test_image_dir = "./test_pictures/smoking_women.jpg"

    # anchors
    num_anchor_per_pixel = 9
    ratios = [0.5, 1, 2]
    scales = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]
    downsampling_strides = [8, 16, 32, 64, 128]
    sizes = [32, 64, 128, 256, 512]

    # focal loss
    alpha = 0.25
    gamma = 2.0


    @classmethod
    def get_image_size(cls):
        return cls.image_size[cls.network_type]

    @classmethod
    def get_width_coefficient(cls):
        return cls.width_coefficient[cls.network_type]

    @classmethod
    def get_depth_coefficient(cls):
        return cls.depth_coefficient[cls.network_type]

    @classmethod
    def get_dropout_rate(cls):
        return cls.dropout_rate[cls.network_type]

    @classmethod
    def get_w_bifpn(cls):
        return cls.w_bifpn[cls.network_type]

    @classmethod
    def get_d_bifpn(cls):
        return cls.d_bifpn[cls.network_type]

    @classmethod
    def get_d_class(cls):
        return cls.d_class[cls.network_type]
