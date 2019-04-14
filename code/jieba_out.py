import jieba

dict_file = '../textPreprocess/dict.txt'

# doc_file = '../textPreprocess/lesstrain_convert10cates.txt'
o_file = '../test/lesstrain_jieba.txt'

doc_file = '../test/error_eva.txt'
o_file = '../test/error_jieba.txt'

doc = open(doc_file, 'r', encoding='utf8')
o = open(o_file, 'w', encoding='utf8')
# doc.readline()

jieba.load_userdict(dict_file)
word2id = {}
wordListFile = open('../textPreprocess/trainTestIntersectionWord.txt', 'r', encoding='utf8')  # 训练集词典
wordListFile.readline()
for line in wordListFile.readlines():
    line = line.strip().split(',')
    word2id[line[0]] = int(line[2])  # word+id
car_list = []
carListFile = open('../textPreprocess/car.txt', 'r', encoding='utf8')
for line in carListFile.readlines():
    car_list.append(line.encode('utf-8').decode('utf-8-sig').strip())
print('load word list ok!')
wordListFile.close()
carListFile.close()

for line in doc.readlines():
    line = line.strip().split('\t')
    text = line[1]
    for word in jieba.cut(text, cut_all=False, HMM=True):
        if word in car_list:
            o.write(word + 'carcar'+'\\')
        elif all(char.isdigit() for char in word):
            o.write(word +'isdigit'+'\\')
        elif word in word2id:
            o.write(word + '\\')
        else:
            o.write('('+word+')'+'\\')
    o.write('\n')
o.close()
