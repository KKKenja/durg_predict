from keras import models
from keras.layers import Dense, Input
from keras.callbacks import EarlyStopping
import numpy as np
import pickle
from scipy.stats import pearsonr

# 載入數據
def load_data(filename, log_trans=False, label=False):
    data = []
    gene_names = []
    data_labels = []
    lines = open(filename).readlines()
    sample_names = lines[0].replace('\n', '').split('\t')[1:]
    if label:
        print("有標記",filename)
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

# 基因選擇（基於相關性篩選）
def correlation_based_feature_selection(data_exp, data_drug, sample_names_exp, sample_names_drug, threshold=0.5):
    # 確保細胞系名稱對齊
    common_samples = list(set(sample_names_exp) & set(sample_names_drug))

    # 根據共同細胞系篩選出對應的數據
    exp_selected = []
    drug_selected = []
    
    for sample in common_samples:
        exp_idx = sample_names_exp.index(sample)
        drug_idx = sample_names_drug.index(sample)
        
        exp_selected.append(data_exp[exp_idx])  # 根據細胞系篩選基因表達
        drug_selected.append(data_drug[drug_idx])  # 根據細胞系篩選IC50數據

    exp_selected = np.array(exp_selected)
    drug_selected = np.array(drug_selected)

    # 計算每個基因與IC50之間的相關性
    corr_scores = []
    for i in range(exp_selected.shape[0]):  # 每個基因是行
        corr, _ = pearsonr(exp_selected[i], drug_selected)  # 計算基因與IC50之間的相關性
        corr_scores.append(abs(corr))  # 使用絕對相關性

    corr_scores = np.array(corr_scores)
    
    # 篩選出相關性高於閾值的基因索引
    selected_genes_idx = np.where(corr_scores >= threshold)[0]
    return selected_genes_idx


if __name__ == '__main__':
    # 載入基因表達和藥物敏感性資料
    data_exp, data_labels_exp, sample_names_exp, gene_names_exp = load_data("data/training/gdsc_ccle_overlap_tpm.txt")
    data_drug, data_labels_drug, sample_names_drug, drug_names_drug = load_data("data/training/gdsc_ccle_overlap_ic50_reverse.txt", label=True)

    # 計算並篩選與IC50相關性高的基因
    selected_genes_idx = correlation_based_feature_selection(data_exp, data_drug, sample_names_exp, sample_names_drug, threshold=0.5)
    
    # 根據篩選結果，選擇相應的基因表達數據
    data_exp_selected = data_exp[selected_genes_idx]
    gene_names_selected = [gene_names_exp[i] for i in selected_genes_idx]
    print("data_exp_selected shape:", data_exp_selected.shape)
    print("data_drug shape:", data_drug.shape)
    # 載入預先訓練的自動編碼器模型參數
    premodel_exp = pickle.load(open('data/output/tcga_pretrained_autoencoder_exp_3.pickle', 'rb'))
    
    # 設定模型超參數
    activation_func = 'relu'
    activation_func2 = 'linear'
    init = 'he_uniform'
    dense_layer_dim = 128
    batch_size = 16
    num_epoch = 50

    # 將樣本分為90%訓練/驗證和10%測試
    id_rand = np.random.permutation(data_drug.shape[0])
    id_train = id_rand[:int(data_drug.shape[0] * 0.9)]
    id_test = id_rand[int(data_drug.shape[0] * 0.9):]

    # 基因表達模型
    input_exp = Input(shape=(data_exp_selected.shape[0],))  # 輸入選擇後的基因數量
    model_exp = Dense(units=4096, activation=activation_func)(input_exp)
    model_exp = Dense(units=2048, activation=activation_func)(model_exp)
    model_exp = Dense(units=512, activation=activation_func)(model_exp)
    model_exp = Dense(units=128, activation=activation_func)(model_exp)

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
        data_exp_selected[id_train], data_drug[id_train],
        epochs=num_epoch,
        validation_split=1/9,
        batch_size=batch_size,
        shuffle=True,
        callbacks=[early_stopping]
    )

    # 評估模型表現
    cost_testing = model.evaluate(data_exp_selected[id_test], data_drug[id_test], verbose=0, batch_size=batch_size)
    print('訓練完成。測試成本 = %.4f' % cost_testing)

    # 儲存訓練好的模型
    model.save("data/output/model_final_with_cfs.h5")

    # 使用訓練好的模型進行預測並儲存結果
    data_pred = model.predict(data_exp_selected, batch_size=batch_size, verbose=0)
    np.savetxt('data/output/predicted_IC50_with_cfs.txt', np.transpose(data_pred), delimiter='\t', fmt='%.4f')
