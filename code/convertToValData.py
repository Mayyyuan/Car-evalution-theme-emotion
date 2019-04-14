train = open('../train.csv', 'r', encoding='utf8')
# test = open('../test_public.csv', 'r', encoding='utf8')
o_val = open('../val_500.csv', 'r', encoding='utf8')
o_valData = open('../valData_500.csv', 'w', encoding='utf8')
# o_valData = open('../trainData.txt', 'w', encoding='utf8')
# o_w2vData = open('../data/w2v_data.txt', 'w', encoding='utf8')
o_val.readline()
train.readline()
o_valData.write("content_id,content,subject,sentiment_value,sentiment_word\n")

# train.readline()
# test.readline()
# cur_id = 0
# for line in train.readlines():
#     last_id = cur_id
#     line = line.strip().split(',')
#     cur_id = line[0]
#     text = line[1]
#     if cur_id != last_id:
#         o_w2vData.write(text+'\n')
# for line in test.readlines():
#     line = line.strip().split(',')
#     text = line[1]
#     o_w2vData.write(text+'\n')
# train.close()
# test.close()
# o_w2vData.close()


idval=[]
for line in o_val.readlines():
    line = line.strip().split(',')
    id=line[0]
    idval.append(id)
print(idval)
print(len(idval))

count = 0
for line in train.readlines():
    line = line.strip().split(',')
    # o_valData.write(line[1] + '\t' + line[2] + '\n')
    id = line[0]
    if id in idval:
        o_valData.write(id + ',' + line[1] + ',' + line[2] + ',' + line[3] + ',' +line[4] + '\n')
        # o_valData.write(line[1] + '\t' + line[2] + '\n')
        count += 1
print(count)
o_val.close()
train.close()
o_valData.close()
