# Use this code to pretrain an autoencoder using unlabeled gene mutation or gene expression data of tumors 
# Output of this code is used by TrainModel.py to initialize the weights of IC50 predictor on cell lines
from tensorflow import keras
import pickle
import numpy as np
from keras import Dense  # 从 keras.layers 导入 Dense

# load tabular data
def load_data(filename, log_trans=False, label=False):
    
    data = []
    gene_names = []
    data_labels = []
    lines = open(filename).readlines()
    sample_names = lines[0].replace('\n', '').split('\t')[1:]  # 获取样本名称
    if label:
        data_labels = lines[1].replace('\n', '').split('\t')[1:]  # 如果有标签，获取标签名称
        dx = 2  # 如果有标签，从第二行开始读取数据
    else:
        dx = 1  # 没有标签，从第一行开始读取数据

    for line in lines[dx:]:  # 从指定行开始读取数据
        values = line.replace('\n', '').split('\t')
        gene = str.upper(values[0])  # 将基因名称转为大写
        gene_names.append(gene)
        data.append(values[1:])  # 只保留数值部分
    data = np.array(data, dtype=float)  # 确保数据是浮点类型
    if log_trans:
        data = np.log2(data + 1)  # 对数据进行对数转换
    data = np.transpose(data)  # 转置数据
    return data, data_labels, sample_names, gene_names

# save model parameters to pickle
def save_weight_to_pickle(model, file_name):
    print('saving weights')
    weight_list = []
    for layer in model.layers:
        weight_list.append(layer.get_weights())
    with open(file_name, 'wb') as handle:
        pickle.dump(weight_list, handle)

if __name__ == '__main__':
    # load tabular mutation or expression data
    data, data_labels, sample_names, gene_names = load_data("../pretrain_data/tcga_exp.txt")
    print(f"Data shape: {data.shape}")  # 打印數據形狀
    if data.size == 0:
        raise ValueError("Loaded data is empty. Check the input file or data loading logic.")

    # 检查数据是否成功加载并是有效的数值类型
    if not np.issubdtype(data.dtype, np.number):
        raise ValueError("Loaded data contains non-numeric values.")

    input_dim = data.shape[1]  # 输入维度

    # set hyperparameters
    first_layer_dim = 4096
    second_layer_dim = 128
    third_layer_dim = 32
    batch_size = 64
    epoch_size = 100
    activation_func = 'relu'
    init = 'he_uniform'

    # model construction and training
    model = keras.Sequential()
    model.add(Dense(units=first_layer_dim, input_shape=(input_dim,), activation=activation_func, kernel_initializer=init))
    model.add(Dense(units=second_layer_dim, activation=activation_func, kernel_initializer=init))
    model.add(Dense(units=third_layer_dim, activation=activation_func, kernel_initializer=init))
    model.add(Dense(units=second_layer_dim, activation=activation_func, kernel_initializer=init))
    model.add(Dense(units=first_layer_dim, activation=activation_func, kernel_initializer=init))
    model.add(Dense(units=input_dim, activation='sigmoid', kernel_initializer=init))  # GPT建議使用 sigmoid 激活函數

    model.compile(loss='mse', optimizer='adam')  # 编译模型
    model.fit(data, data, epochs=epoch_size, batch_size=batch_size, shuffle=True)  # 训练模型

    cost = model.evaluate(data, data, verbose=0)  # 评估模型
    print('Training completed.\nCost=%.4f' % cost)

    # save model parameters to pickle file, which will be used in TrainModel.py
    save_weight_to_pickle(model, 'tcga_pretrained_autoencoder_exp.pickle')  # 保存模型权重
