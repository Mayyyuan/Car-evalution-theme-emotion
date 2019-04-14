def replace(predictFile, allZeroPredictFile):
    raw = open(predictFile,'r',encoding='utf8')
    o = open(allZeroPredictFile,'w',encoding='utf8')
    first_line = raw.readline()
    o.write(first_line)
    for line in raw.readlines():
        line = line.strip().split(',')
        o.write(line[0]+','+line[1]+','+'0'+',\n')
    raw.close()
    o.close()


if __name__ =='__main__':
    predictFile = '../result/test_public_predict_0915_1046_tfidf.csv'
    allZeroPredictFile = '../result/test_public_predict_0915_1046_tfidf_all_zero.csv'
    replace(predictFile, allZeroPredictFile)


