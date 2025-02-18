from keras import models
from keras.layers import Dense, Input
from keras.callbacks import EarlyStopping
import numpy as np

# 載入數據
def load_data(filename, log_trans=False, label=False):
    data = []
    gene_names = []
    data_labels = []
    lines = open(filename).readlines()
    sample_names = lines[0].replace('\n', '').split('\t')[1:]
    if label:
        print("有標記", filename)
        data_labels = lines[1].replace('\n', '').split('\t')[1:]
        dx = 2
    else:
        dx = 1

    for line in lines[dx:]:
        values = line.replace('\n', '').split('\t')
        gene = str.upper(values[0])
        gene_names.append(gene)
        data.append(values[1:])
    # print("data: ", data[0:2])
    data = np.array(data, dtype='float32')
    if log_trans:
        data = np.log2(data + 1)
    data = np.transpose(data)

    return data, data_labels, sample_names, gene_names


if __name__ == '__main__':
    # 載入基因表達和藥物敏感性資料
    data_exp, data_labels_exp, sample_names_exp, gene_names_exp = load_data("data/training/gdsc_ccle_overlap_tpm.txt")
    data_drug, data_labels_drug, sample_names_drug, drug_names_drug = load_data("data/training/gdsc_ccle_overlap_ic50_reverse.txt", label=True)

    # 設定模型超參數
    activation_func = 'relu'
    activation_func2 = 'linear'
    init = 'he_uniform'  # 隨機初始化方式
    dense_layer_dim = 128
    batch_size = 16
    num_epoch = 50

    # 將樣本分為90%訓練/驗證和10%測試
    id_rand = np.random.permutation(data_drug.shape[0])
    id_train = id_rand[:int(data_drug.shape[0] * 0.9)]
    id_test = id_rand[int(data_drug.shape[0] * 0.9):]

    # 基因表達模型（完全隨機初始化）
    input_dim = data_exp.shape[1]  # 輸入的基因數量
    input_exp = Input(shape=(input_dim,))
    model_exp = Dense(units=1024, activation=activation_func, kernel_initializer=init)(input_exp)
    model_exp = Dense(units=256, activation=activation_func, kernel_initializer=init)(model_exp)
    model_exp = Dense(units=64, activation=activation_func, kernel_initializer=init)(model_exp)

    # 添加最後的預測層
    model_final = Dense(units=dense_layer_dim, activation=activation_func, kernel_initializer=init)(model_exp)
    model_final = Dense(units=dense_layer_dim, activation=activation_func, kernel_initializer=init)(model_final)
    model_final = Dense(units=dense_layer_dim, activation=activation_func, kernel_initializer=init)(model_final)
    model_final = Dense(units=dense_layer_dim, activation=activation_func, kernel_initializer=init)(model_final)
    output = Dense(units=data_drug.shape[1], activation=activation_func2, kernel_initializer=init)(model_final)

    # 定義完整模型
    model = models.Model(inputs=input_exp, outputs=output)

    # 編譯模型
    model.compile(loss='mse', optimizer='adam')

    # 設定提早停止
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=1, mode='min')

    # 訓練模型
    model.fit(
        data_exp[id_train], data_drug[id_train],
        epochs=num_epoch,
        validation_split=1 / 9,
        batch_size=batch_size,
        shuffle=True,
        callbacks=[early_stopping]
    )

    # 評估模型表現
    cost_testing = model.evaluate(data_exp[id_test], data_drug[id_test], verbose=0, batch_size=batch_size)
    print('randomInitial 訓練完成。測試成本 = %.4f' % cost_testing)

    # 儲存訓練好的模型
    model.save("data/output/model_random_initialization.h5")

    # 使用訓練好的模型進行預測並儲存結果
    data_pred = model.predict(data_exp, batch_size=batch_size, verbose=0)
    np.savetxt('data/output/predicted_IC50_random.txt', np.transpose(data_pred), delimiter='\t', fmt='%.4f')
