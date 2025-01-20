from keras import models
from keras.layers import Dense, Input
from keras.callbacks import EarlyStopping
import numpy as np
import pickle
import os
import tensorflow as tf
from tensorflow.keras.backend import clear_session

# 設置GPU內存增長
physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

# 載入數據的函數
def load_data(filename, log_trans=False, label=False):
    data = []
    gene_names = []
    data_labels = []
    lines = open(filename).readlines()
    sample_names = lines[0].replace('\n', '').split('\t')[1:]
    if label:
        data_labels = lines[1].replace('\n', '').split('\t')[1:]
        dx = 2
    else:
        dx = 1

    for line in lines[dx:]:
        values = line.replace('\n', '').split('\t')
        gene = str.upper(values[0])
        gene_names.append(gene)
        data.append(values[1:])

    data = np.array(data, dtype='float32')
    if log_trans:
        data = np.log2(data + 1)
    data = np.transpose(data)

    return data, data_labels, sample_names, gene_names

if __name__ == '__main__':
    # 載入基因表達和藥物敏感性資料
    data_exp, data_labels_exp, sample_names_exp, gene_names_exp = load_data("data/training/gdsc_ccle_overlap_tpm.txt")
    data_drug, data_labels_drug, sample_names_drug, drug_names_drug = load_data("data/training/gdsc_ccle_overlap_ic50_reverse.txt", label=True)

    # 載入預先訓練的自動編碼器模型參數
    premodel_exp = pickle.load(open('data/output/tcga_pretrained_autoencoder_exp_3.pickle', 'rb'))

    # 訓練10次，記錄early stop的epoch和loss
    results = []
    num_runs = 10
    batch_size = 16
    num_epoch = 50
    dense_layer_dim = 128
    activation_func = 'relu'
    activation_func2 = 'linear'
    init = 'he_uniform'

    for run in range(num_runs):
        print(f"開始第 {run + 1} 次訓練")

        # 將樣本分為90%訓練/驗證和10%測試
        id_rand = np.random.permutation(data_drug.shape[0])
        id_train = id_rand[:int(data_drug.shape[0] * 0.9)]
        id_test = id_rand[int(data_drug.shape[0] * 0.9):]

        # 定義模型
        input_exp = Input(shape=(premodel_exp[0][0].shape[0],))
        model_exp = Dense(units=1024, activation=activation_func)(input_exp)
        model_exp = Dense(units=256, activation=activation_func)(model_exp)
        model_exp = Dense(units=128, activation=activation_func)(model_exp)

        model_final = Dense(units=dense_layer_dim, activation=activation_func, kernel_initializer=init)(model_exp)
        model_final = Dense(units=dense_layer_dim, activation=activation_func, kernel_initializer=init)(model_final)
        model_final = Dense(units=dense_layer_dim, activation=activation_func, kernel_initializer=init)(model_final)
        output = Dense(units=data_drug.shape[1], activation=activation_func2, kernel_initializer=init)(model_final)

        model = models.Model(inputs=input_exp, outputs=output)
        model.compile(loss='mse', optimizer='adam')

        # 設定early stopping
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=1, mode='min', restore_best_weights=True)

        # 訓練模型
        history = model.fit(
            data_exp[id_train], data_drug[id_train],
            epochs=num_epoch,
            validation_split=1/9,
            batch_size=batch_size,
            shuffle=True,
            callbacks=[early_stopping],
            verbose=0  # No need to display loss for each epoch
        )

        # 評估模型
        cost_testing = model.evaluate(data_exp[id_test], data_drug[id_test], verbose=0, batch_size=batch_size)
        early_stopped_epoch = len(history.history['val_loss'])

        print(f"第 {run + 1} 次訓練完成: Early Stop 發生在 Epoch {early_stopped_epoch}，測試 Loss = {cost_testing:.4f}")

        # 清理模型以釋放內存
        clear_session()

        # 存儲結果
        results.append((run + 1, early_stopped_epoch, cost_testing))

    # 將結果輸出到文件
    output_file = "data/output/early_stop_statistics.txt"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        f.write("Run\tEarlyStopEpoch\tTestLoss\n")
        for run, epoch, loss in results:
            f.write(f"{run}\t{epoch}\t{loss:.4f}\n")

    print(f"所有訓練完成，結果已保存到 {output_file}")
