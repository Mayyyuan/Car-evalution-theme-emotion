import tensorflow as tf
from model import *
from utils import *
from evaluate import *
import numpy as np
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

tf.flags.DEFINE_string('train_corpus', '../textPreprocess/lesstrain_convert10cates.txt', 'train corpus path')
tf.flags.DEFINE_string('train_raw', '../train.csv', 'train corpus path')
tf.flags.DEFINE_string('val_corpus', '../val.csv', 'val corpus path')
tf.flags.DEFINE_string('val_data', '../valData.csv', 'val corpus path')
tf.flags.DEFINE_string('test_corpus', '../test_public.csv', 'test data')

tf.flags.DEFINE_string('val_id', '../textPreprocess/val_id_no.txt', 'val id')
tf.flags.DEFINE_string('test_id', '../textPreprocess/test_id.txt', 'test id')
tf.flags.DEFINE_string('val_predict_file', '../result/val_public_predict_no.csv', 'val result data')
tf.flags.DEFINE_string('test_predict_file', '../result/test_public_predict_0929(oridata)_10cates_30epoch.csv',
                       'predict result data')
tf.flags.DEFINE_string('train_loss_file', '../result/trainloss_0919_10cates_90epoch.txt', 'train loss')
tf.flags.DEFINE_string('acc_validation_file', '../result/acc_0919_10cates_90epoch.txt', 'validation acc')
tf.flags.DEFINE_string('cate_model', '../model/cateModel_0928.h5', 'the 10 cate model.')
tf.flags.DEFINE_string('sentiment_model', '../model/sentimentModel.h5', 'the 3 cate sentiment model.')
tf.flags.DEFINE_string('w2v_model', '../model/w2vmodel/wiki_400.model', 'the w2v model.')

tf.flags.DEFINE_string('dict_file', '../textPreprocess/dict.txt', 'predict result data')
tf.flags.DEFINE_string('stop_words_file', '../case/stopwords.txt', 'stop words')
tf.flags.DEFINE_string('word_list', '../textPreprocess/trainTestIntersectionWord.txt', 'word list')
tf.flags.DEFINE_string('car_list', '../case/car.txt', 'word list')
tf.flags.DEFINE_string('digit_list', '../case/5_digit.txt', 'digit list')
tf.flags.DEFINE_string('fraction_list', '../case/6_fraction.txt', 'fraction list')
tf.flags.DEFINE_string('distance_list', '../case/7_distance.txt', 'distance list')
tf.flags.DEFINE_string('money_list', '../case/8_money.txt', 'money list')
tf.flags.DEFINE_string('city_list', '../case/9_city.txt', 'city list')

tf.flags.DEFINE_integer('max_sequence_length', 50, 'the sentence max length')
tf.flags.DEFINE_integer('embedding_size', 400, 'the word vector dimension')
tf.flags.DEFINE_integer('vocab_length', 18721, 'the train vocab size')
tf.flags.DEFINE_integer('epoch', 40, 'epoch')
tf.flags.DEFINE_integer('embedding_dim', 400, 'test data')
tf.flags.DEFINE_integer('batch_size', 64, 'batch size')
tf.flags.DEFINE_integer('cate_num', 10, 'category number')
tf.flags.DEFINE_integer('evaluate_step', 100, 'batch size')
tf.flags.DEFINE_float('lr', 0.001, 'learning rate')
tf.flags.DEFINE_float('dropout', 0.3, 'drop out')
tf.flags.DEFINE_float('VALIDATION_SPLIT', 0.001, 'portion of validation set')

tf.flags.DEFINE_boolean('train_sentiment_model', True, 'if train sentiment model or not')
FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()

######################################################## data preprocess ###############################################
train_X = []
train_Y = []
random = np.random.RandomState(0)  # for random drop
train = open(FLAGS.train_corpus, 'r', encoding='utf8')

word2id = {}
wordListFile = open(FLAGS.word_list, 'r', encoding='utf8')
wordListFile.readline()
for line in wordListFile.readlines():
    line = line.strip().split(',')
    word2id[line[0]] = int(line[2])  # word+id
wordListFile.close()

car_list, digit_list, fraction_list, distance_list, \
money_list, city_list = getList(FLAGS.car_list, FLAGS.digit_list, FLAGS.fraction_list,
                                FLAGS.distance_list, FLAGS.money_list, FLAGS.city_list)

jieba.load_userdict(FLAGS.dict_file)
for line in train.readlines():
    line = line.strip().split('\t')
    curSequence = getSequenceId(line[1], word2id, car_list, digit_list, fraction_list, distance_list, money_list,
                                city_list, shuffle=False, drop=False)
    curLabel = getMultihotLabel(line[2], FLAGS.cate_num)
    train_X.append(curSequence)
    train_Y.append(curLabel)
    # curSequence = getSequenceId(line[1], word2id, car_list, digit_list, fraction_list, distance_list, money_list,
    #                             city_list, shuffle=True, drop=False)
    # train_X.append(curSequence)
    # train_Y.append(curLabel)
train.close()

train_X = pad_sequences(train_X, maxlen=FLAGS.max_sequence_length, padding='post', truncating='post',
                        value=0.)
train_Y = np.array(train_Y)

train_sample_num = train_X.shape[0]
word_vector = get_word_vector(word2id, FLAGS.w2v_model, FLAGS.embedding_size)

#############################################shuffle and split train/validation#####################################
nb_validation_samples = int(FLAGS.VALIDATION_SPLIT * train_sample_num)

x_train = train_X[:-nb_validation_samples]  # train
y_train = train_Y[:-nb_validation_samples]

x_val = train_X[-nb_validation_samples:]  # val
y_val = train_Y[-nb_validation_samples:]

print('train data size is '+str(x_train.shape[0]))
del train_X
del train_Y
print('split data is ok!')
ret = batch_iter(x_train.shape[0], x_train, y_train, FLAGS.batch_size, FLAGS.epoch, shuffle=True)

batchPerEpoch = train_sample_num/FLAGS.batch_size+1
i = 0
minTrainLoss = 200
maxF1 = 0.0
# bestEpoch = -1
blue = BlueModel(FLAGS)
model = blue.buildCNN(embedding_matrix=word_vector, UsePretrain=True)

train_loss = open(FLAGS.train_loss_file, 'w', encoding='utf8')
val_acc = open(FLAGS.acc_validation_file, 'w', encoding='utf8')
for batch in ret:
    curEpoch = int((i*FLAGS.batch_size)/train_sample_num)
    curBatch = int(i % batchPerEpoch)
    print(str(curEpoch)+'th epoch, '+str(curBatch)+'th batch.')
    i += 1  # batch加1
    x_batch = np.array(batch[0])  # xtrain
    y_batch = np.array(batch[1])  # ytrain
    x_batch_cate = []

    print('batch train X dimension is ['+str(x_batch.shape[0])+','+str(x_batch.shape[1])+']')
    print('batch train Y dimension is [' + str(y_batch.shape[0]) + ',' + str(y_batch.shape[1]) + ']')
    trainloss = model.train_on_batch(x_batch, y_batch)
    train_loss.write(str(trainloss[0])+'\n')
    if i % FLAGS.evaluate_step == 0:
        train_loss.flush()
    print(trainloss)

    F1 = 0.0
    if curEpoch > 15 and i % FLAGS.evaluate_step == 0:
        print('evaluate.....')
        getPredict(model, FLAGS.val_corpus, FLAGS.val_predict_file, FLAGS.max_sequence_length, word2id, car_list,
                   FLAGS.dict_file, digit_list, fraction_list, distance_list, money_list, city_list)
        F1 = evaluate(FLAGS.val_predict_file, FLAGS.val_data)
        print('the validation acc is '+str(F1))
        if F1 > maxF1:
            maxF1 = F1
            val_acc.write(str(curEpoch)+'th epoch, '+str(curBatch)+'th batch.\n'+str(maxF1)+'\n')
            val_acc.flush()
            model.save_weights(FLAGS.cate_model, overwrite=True)
            print('update the model')
            print(F1)
            # bestEpoch = curEpoch
            print('####################################getting result#####################################3#####')
            getPredict(model, FLAGS.test_corpus, FLAGS.test_predict_file, FLAGS.max_sequence_length, word2id, car_list,
                       FLAGS.dict_file, digit_list, fraction_list, distance_list, money_list, city_list)
            print('predict ok!')

    del x_batch
    del y_batch
    # if curEpoch > bestEpoch+20:
    #     break
train_loss.close()
val_acc.close()
print("maxF1:", maxF1)
