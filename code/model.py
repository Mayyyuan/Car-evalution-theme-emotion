from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, Dropout, Concatenate
from keras import optimizers
from keras.models import Model


class BlueModel:
    def __init__(self, flags):
        self.MAX_SEQUENCE_LENGTH = flags.max_sequence_length
        self.EMBEDDING_DIM = flags.embedding_dim
        self.VOCAB_LENGTH = flags.vocab_length
        self.CATE_NUM = flags.cate_num
        self.LR = flags.lr
        self.DROP_OUT = flags.dropout

    def buildCNN(self, embedding_matrix, UsePretrain):
        if UsePretrain:
            embedding_layer = Embedding(
                self.VOCAB_LENGTH+1,
                self.EMBEDDING_DIM,
                weights=[embedding_matrix],
                input_length=self.MAX_SEQUENCE_LENGTH,
                trainable=True)
        else:
            embedding_layer = Embedding(self.VOCAB_LENGTH+1,
                                        self.EMBEDDING_DIM,
                                        input_length=self.MAX_SEQUENCE_LENGTH,
                                        trainable=True)
        convs = []
        filter_sizes = [3, 4, 5]

        sequence_input = Input(shape=(self.MAX_SEQUENCE_LENGTH,), dtype='int32')
        embedded_sequences = embedding_layer(sequence_input)
        embedded_sequences = Dropout(self.DROP_OUT)(embedded_sequences)

        for fsz in filter_sizes:
            l_conv = Conv1D(nb_filter=128, filter_length=fsz,
                            activation='relu')(embedded_sequences)
            l_pool = MaxPooling1D(5)(l_conv)
            convs.append(l_pool)

        # l_merge = Concatenate(axis=-1)(convs)
        # l_cov1 = Conv1D(1024, 5, activation='relu')(l_merge)
        # l_flat = Flatten()(l_cov1)
        # l_flat = Dropout(self.DROP_OUT)(l_flat)
        # l_dense = Dense(1024, activation='relu')(l_flat)
        # l_dense = Dropout(self.DROP_OUT)(l_dense)
        # preds = Dense(self.CATE_NUM, activation='sigmoid')(l_dense)

        l_merge = Concatenate(axis=-1)(convs)
        l_flat = Flatten()(l_merge)  # 1920 shape
        l_flat = Dropout(self.DROP_OUT)(l_flat)
        l_dense = Dense(512, activation='relu')(l_flat)
        l_dense = Dropout(self.DROP_OUT)(l_dense)
        l_dense2 = Dense(256, activation='relu')(l_dense)
        l_dense2 = Dropout(self.DROP_OUT)(l_dense2)
        preds = Dense(self.CATE_NUM, activation='sigmoid')(l_dense2)

        # l_merge = Concatenate(axis=1)(convs)
        # l_cov1 = Conv1D(128, 5, activation='relu')(l_merge)
        # l_pool1 = MaxPooling1D(5)(l_cov1)
        ## l_cov2 = Conv1D(128, 5, activation='relu')(l_pool1)
        ## l_pool2 = MaxPooling1D(5)(l_cov2)
        # l_flat = Flatten()(l_pool1)
        # l_dense = Dense(128, activation='relu')(l_flat)
        # preds = Dense(30, activation='sigmoid')(l_dense)

        model = Model(sequence_input, preds)
        myOptimizer = optimizers.Adam(lr=self.LR, beta_1=0.9, beta_2=0.999, epsilon=None,
                                      decay=0., amsgrad=False)
        model.compile(loss='binary_crossentropy', optimizer=myOptimizer,
                      metrics=['acc'])
        return model

    def buildSentimentCNN(self, embedding_matrix, UsePretrain):
        if UsePretrain:
            embedding_layer = Embedding(
                self.VOCAB_LENGTH + 1,
                self.EMBEDDING_DIM,
                weights=[embedding_matrix],
                input_length=self.MAX_SEQUENCE_LENGTH,
                trainable=True)
        else:
            embedding_layer = Embedding(self.VOCAB_LENGTH + 1,
                                        self.EMBEDDING_DIM,
                                        input_length=self.MAX_SEQUENCE_LENGTH,
                                        trainable=True)
        convs = []
        filter_sizes = [3, 4, 5]

        sequence_input = Input(shape=(self.MAX_SEQUENCE_LENGTH,), dtype='int32')
        cate_input = Input(shape=(self.CATE_NUM, ), dtype='float32')   ## the category signal
        embedded_sequences = embedding_layer(sequence_input)
        embedded_sequences = Dropout(self.DROP_OUT)(embedded_sequences)

        for fsz in filter_sizes:
            l_conv = Conv1D(nb_filter=128, filter_length=fsz,
                            activation='relu')(embedded_sequences)
            l_pool = MaxPooling1D(5)(l_conv)
            convs.append(l_pool)

        l_merge = Concatenate(axis=-1)(convs)
        l_flat = Flatten()(l_merge)

        l_flat = Dropout(self.DROP_OUT)(l_flat)
        l_dense = Dense(512, activation='relu')(l_flat)
        # my_concat = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=-1))
        # l_dense = my_concat([l_dense, cate_input])
        l_dense = Dropout(self.DROP_OUT)(l_dense)
        l_dense2 = Dense(256, activation='relu')(l_dense)
        l_dense2 = Dropout(self.DROP_OUT)(l_dense2)

        l_dense3 = Dense(64, activation='relu')(l_dense2)

        preds = Dense(3, activation='sigmoid')(l_dense3)

        model = Model(inputs=[sequence_input, cate_input], outputs=preds)  # 模型
        myOptimizer = optimizers.Adam(lr=self.LR, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0., amsgrad=False)  # 优化器
        model.compile(loss='binary_crossentropy', optimizer=myOptimizer,metrics=['acc'])  # 自定义损失函数(对数损失函数/最小化目标函数)
        return model
