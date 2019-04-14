def getError(val_predict_file, train_corpus, error_eva_file):
    predict_file = open(val_predict_file, 'r', encoding='utf8')
    train_corpus = open(train_corpus, 'r', encoding='utf8')
    error_eva = open(error_eva_file, 'w', encoding='utf8')

    predict_file.readline()
    train_corpus.readline()
    predict_lines = predict_file.readlines()
    train_lines = train_corpus.readlines()
    Tp = 0
    train_total = 0
    predict_total = 0

    contentid_train = {}
    id_train = 0
    id2text = {}
    for i in range(len(train_lines)):
        train_total += 1
        id_train_last = id_train
        id_train = train_lines[i].strip().split(',')[0]
        text_train = train_lines[i].strip().split(',')[1]

        sub_train = train_lines[i].strip().split(',')[2]
        # sen_train = train_lines[i].strip().split(',')[3]
        sen_train = str(0)
        if id_train_last == id_train:
            contentid_train[id_train].append([sub_train, sen_train])
            id2text[id_train].append([text_train])
        else:
            contentid_train[id_train] = [[sub_train, sen_train]]
            id2text[id_train] = [[text_train]]

    for i in range(len(predict_lines)):
        predict_total += 1
        id_predict = predict_lines[i].strip().split(',')[0]
        sub_predict = predict_lines[i].strip().split(',')[1]
        # sen_predict = predict_lines[i].strip().split(',')[2]
        sen_predict = str(0)
        result_predict = [sub_predict, sen_predict]
        if result_predict in contentid_train[id_predict]:
            Tp += 1
        else:
            error_eva.write(id_predict+'\t'+str(id2text[id_predict])+'\t'+str(contentid_train[id_predict])+'\t'+'error:'+sub_predict+'\n')
    error_eva.close()
    predict_file.close()
    train_corpus.close()
    Fp = predict_total - Tp
    Fn = train_total - Tp
    P = Tp / (Tp + Fp)
    R = Tp / (Tp + Fn)
    F1 = 2 * P * R / (P + R)
    return F1


if __name__ == "__main__":
    print("开始...")
    predict_file = '../result/val_public_predict_0929.csv'
    train_corpus = '../valData.csv'

    error_eva_file = '../test/error_eva.txt'
    F1 = getError(predict_file, train_corpus, error_eva_file)
    print(F1)
