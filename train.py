from datetime import datetime
from pathlib import Path

import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms

from data_aug import MyRightShift, MyReshape
from data_set import Ub_Leaking_Dataset
from model_select import Trainer
from utils import normalize_data
from utils import show_results


def process_fold(fold_k, dataset_df, train_transforms, test_transforms,
                 name="shallow_alexnet", epochs=100, batch_size=32, num_of_workers=0):
    """
    按10fold交叉验证
    :param name: 调用的模型
    :param fold_k: 设置验证的fold
    :param dataset_df: 要训练的数据总集的dataframe，此工程中是ub_leaking
    :param train_transforms: 训练集的data_aug
    :param test_transforms: 测试集的data_aug
    :param epochs: 要训练的轮数
    :param batch_size: 批量
    :param num_of_workers: 多线程读取
    :return:
    """
    # get number of classes
    num_of_classes = len(dataset_df["classID"].unique())

    # split the data
    train_df = dataset_df[dataset_df['fold'] != fold_k]
    test_df = dataset_df[dataset_df['fold'] == fold_k]

    # normalize the data
    train_df, test_df = normalize_data(train_df, test_df)

    # init train data loader
    train_ds = Ub_Leaking_Dataset(train_df, transform=train_transforms)
    train_loader = DataLoader(train_ds,
                              batch_size=batch_size,
                              shuffle=True,
                              pin_memory=True,
                              num_workers=num_of_workers)

    # init test data loader
    test_ds = Ub_Leaking_Dataset(test_df, transform=test_transforms)
    test_loader = DataLoader(test_ds,
                             batch_size=batch_size,
                             shuffle=False,
                             pin_memory=True,
                             num_workers=num_of_workers)

    # init model
    trainer = Trainer(name, num_of_classes)
    trainer.init()

    # pre-training accuracy
    score = trainer.evaluate(test_loader)
    print("Pre-training accuracy: %.4f%%" % (100 * score[1]))

    # train the model
    start_time = datetime.now()
    history = trainer.fit(train_loader, epochs=epochs, val_loader=test_loader)
    end_time = datetime.now() - start_time
    print("\nTraining completed in time: {}".format(end_time))

    return history


def train_main(model_name, epoch=10, batch_size=16):
    save_dir = Path("saved/")
    exper_name = model_name    # 任务名称
    run_id = datetime.now().strftime(r'%m%d_%H%M%S')    # 任务ID：月日_时分秒
    # save_dir = save_dir / 'models' / exper_name / run_id    # 模型保存路径
    trend_dir = save_dir / 'log' / exper_name / run_id    # 结果保存路径
    exist_ok = run_id == ''
    # save_dir.mkdir(parents=True, exist_ok=exist_ok)
    trend_dir.mkdir(parents=True, exist_ok=exist_ok)
    ub_leaking = pd.read_pickle("ub_leaking.pkl")
    train_transforms = transforms.Compose([MyRightShift(input_size=128,
                                                        width_shift_range=13,
                                                        shift_probability=0.9),
                                           MyReshape(output_size=(1, 128, 128))])

    test_transforms = transforms.Compose([MyReshape(output_size=(1, 128, 128))])
    foldk = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    REPEAT = 1    # 一个fold验证重复几遍
    trend = []
    for i in range(REPEAT):
        print("-" * 80)
        print("\n({})\n".format(i + 1))
        history = process_fold(foldk[9], ub_leaking,
                               name=model_name,
                               train_transforms=train_transforms,
                               test_transforms=test_transforms,
                               epochs=epoch,
                               num_of_workers=4,
                               batch_size=batch_size)
        trend.append(history)
    pd.DataFrame(trend).to_csv(path_or_buf=trend_dir / "trend.csv")    # 将运算结果保存为CSV文件
    return trend


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    trend = train_main("vgg11", epoch=50, batch_size=16)
    show_results(trend)
    # print(trend.head())
