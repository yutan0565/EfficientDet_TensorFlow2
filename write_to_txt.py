from data.voc import ParseVOC
from configuration import Config

if __name__ == '__main__':

    if Config.load_flag  == "train":
        parse_voc = ParseVOC("train")
        parse_voc.write_data_to_txt(txt_dir=Config.train_txt_file_dir)
    elif Config.load_flag  == "valid":
        parse_voc = ParseVOC("valid")
        parse_voc.write_data_to_txt(txt_dir=Config.val_txt_file_dir)
    elif Config.load_flag  == "test":
        parse_voc = ParseVOC("test")
        parse_voc.write_data_to_txt(txt_dir=Config.test_txt_file_dir)
    else:
        print("train, valid, test 중에 입력")