import tensorflow as tf
from model import *
from utils import *
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
tf.flags.DEFINE_string('test_predict_file', '../result/test_public_predict_0929(w2v_128)_10cates_30epoch.csv',
                       'predict result data')
tf.flags.DEFINE_string('train_loss_file', '../result/trainloss_0919_10cates_90epoch.txt', 'train loss')
tf.flags.DEFINE_string('acc_validation_file', '../result/acc_0919_10cates_90epoch.txt', 'validation acc')
tf.flags.DEFINE_string('cate_model', '../model/cateModel_0928.h5', 'the 10 cate model.')
tf.flags.DEFINE_string('sentiment_model', '../model/sentimentModel.h5', 'the 3 cate sentiment model.')
tf.flags.DEFINE_string('w2v_model', '../model/w2vmodel/w2vModel_100.model', 'the w2v model.')

tf.flags.DEFINE_string('dict_file', '../textPreprocess/dict.txt', 'predict result data')
tf.flags.DEFINE_string('stop_words_file', '../case/stopwords.txt', 'stop words')
tf.flags.DEFINE_string('word_list', '../textPreprocess/trainTestIntersectionWord.txt', 'word list')
tf.flags.DEFINE_string('car_list', '../case/car.txt', 'word list')
tf.flags.DEFINE_string('digit_list', '../case/5_digit.txt', 'digit list')
tf.flags.DEFINE_string('fraction_list', '../case/6_fraction.txt', 'fraction list')
tf.flags.DEFINE_string('distance_list', '../case/7_distance.txt', 'distance list')
tf.flags.DEFINE_string('city_list', '../case/8_city.txt', 'city list')
tf.flags.DEFINE_string('time_list', '../case/9_time.txt', 'time list')
tf.flags.DEFINE_string('timelength_list', '../case/10_timelength.txt', 'timelength list')

tf.flags.DEFINE_integer('max_sequence_length', 30, 'the sentence max length')
tf.flags.DEFINE_integer('embedding_size', 100, 'the word vector dimension')
tf.flags.DEFINE_integer('vocab_length', 4932, 'the train vocab size')
tf.flags.DEFINE_integer('epoch', 90, 'epoch')
tf.flags.DEFINE_integer('embedding_dim', 100, 'test data')
tf.flags.DEFINE_integer('batch_size', 32, 'batch size')
tf.flags.DEFINE_integer('cate_num', 10, 'category number')
tf.flags.DEFINE_integer('evaluate_step', 100, 'batch size')
tf.flags.DEFINE_float('lr', 0.001, 'learning rate')
tf.flags.DEFINE_float('dropout', 0.3, 'drop out')
tf.flags.DEFINE_float('VALIDATION_SPLIT', 0.001, 'portion of validation set')

tf.flags.DEFINE_boolean('train_sentiment_model', True, 'if train sentiment model or not')

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
blue = BlueModel(FLAGS)
model = blue.buildCNN(embedding_matrix=[], UsePretrain=False)

model.load_weights('../model/cateModel_0928.h5', by_name=False)

word2id = {}
wordListFile = open(FLAGS.word_list, 'r', encoding='utf8')  # 训练集词典
wordListFile.readline()
for line in wordListFile.readlines():
    line = line.strip().split(',')
    word2id[line[0]] = int(line[2])  # word+id
wordListFile.close()
car_list = []
carListFile = open(FLAGS.car_list, 'r', encoding='utf8')
for line in carListFile.readlines():
    car_list.append(line.encode('utf-8').decode('utf-8-sig').strip())
carListFile.close()

digit_list = []
digitListFile = open(FLAGS.digit_list, 'r', encoding='utf8')
for line in digitListFile.readlines():
    digit_list.append(line.encode('utf-8').decode('utf-8-sig').strip().upper())
digitListFile.close()

fraction_list = []
fractionListFile = open(FLAGS.fraction_list, 'r', encoding='utf8')
for line in fractionListFile.readlines():
    fraction_list.append(line.encode('utf-8').decode('utf-8-sig').strip().upper())
fractionListFile.close()

distance_list = []
distanceListFile = open(FLAGS.distance_list, 'r', encoding='utf8')
for line in distanceListFile.readlines():
    distance_list.append(line.encode('utf-8').decode('utf-8-sig').strip().upper())
distanceListFile.close()

city_list = []
cityListFile = open(FLAGS.city_list, 'r', encoding='utf8')
for line in cityListFile.readlines():
    city_list.append(line.encode('utf-8').decode('utf-8-sig').strip().upper())
cityListFile.close()

time_list = []
timeListFile = open(FLAGS.time_list, 'r', encoding='utf8')
for line in timeListFile.readlines():
    time_list.append(line.encode('utf-8').decode('utf-8-sig').strip().upper())
timeListFile.close()

timelength_list = []
timelengthListFile = open(FLAGS.timelength_list, 'r', encoding='utf8')
for line in timelengthListFile.readlines():
    timelength_list.append(line.encode('utf-8').decode('utf-8-sig').strip().upper())
timelengthListFile.close()
print('load word list ok!')

test_file = '../val.csv'
test_predict_file = '../result/val_public_predict_0929.csv'

# test_file = '../test_public.csv'
# test_predict_file = '../result/test_public_predict_0925.csv'


getPredict(model, test_file, test_predict_file, FLAGS.max_sequence_length, word2id, car_list, FLAGS.dict_file,
           digit_list, fraction_list, distance_list, city_list, time_list, timelength_list)
