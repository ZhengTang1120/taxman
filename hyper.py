import numpy as np
import dynet_config
dynet_config.set(
    mem=10240,
    random_seed=1,
    # autobatch=True
)
import dynet as dy

import pickle

class Hyper:
    def __init__(self, words, w2i, embeds, embeds_dim, hidden_size, k):
        self.words = words
        self.vocab_size = len(words)
        self.w2i = w2i
        self.embeds_dim = embeds_dim
        self.hidden_size = hidden_size
        self.embeds = embeds
        self.k = k

        self.model = dy.Model()
        self.trainer = dy.AdamTrainer(self.model)

        self.word_embeddings = self.model.add_lookup_parameters((len(self.w2i), self.embeds_dim))
        for i, w in enumerate(self.words):
            if w in self.embeds:
                self.word_embeddings.init_row(i, self.embeds[w])
        self.Phis = list()
        for i in range(self.k):
            self.Phis.append(self.model.add_parameters((self.embeds_dim, self.embeds_dim)))
        self.W = self.model.add_parameters((1, 24))
        self.b = self.model.add_parameters((1))

    def save(self, name):
        params = (
            self.words, self.w2i, self.embeds_dim, self.embeds, self.hidden_size, self.k
        )
        # save model
        self.model.save(f'{name}.model')
        # save pickle
        with open(f'{name}.pickle', 'wb') as f:
            pickle.dump(params, f)

    @staticmethod
    def load(name):
        with open(f'{name}.pickle', 'rb') as f:
            params = pickle.load(f)
            parser = Hyper(*params)
            parser.model.populate(f'{name}.model')
            return parser

    def train(self, trainning_set):
        loss_chunk = 0
        loss_all = 0
        total_chunk = 0
        total_all = 0
        losses = []
        for datapoint in trainning_set:
            query = datapoint[0]
            eq = dy.average([self.word_embeddings[self.w2i[w]] if w in self.w2i else self.word_embeddings[0] for w in query])
            hyper = datapoint[1]
            eh = dy.average([self.word_embeddings[self.w2i[w]] if w in self.w2i else self.word_embeddings[0] for w in hyper])
            t = dy.scalarInput(datapoint[2])
            Ps = []
            for i in range(self.k):
                Ps.append(self.Phis[i].expr() * eq)
            P = dy.transpose(dy.concatenate_cols(Ps))
            s = P * eh
            y = dy.reshape(dy.logistic(self.W.expr() * s + self.b.expr()), (1,))

            losses.append(dy.binary_log_loss(y, t))

            # process losses in chunks
            if len(losses) > 50:
                loss = dy.esum(losses)
                l = loss.scalar_value()
                loss.backward()
                self.trainer.update()
                dy.renew_cg()
                losses = []
                loss_chunk += l
                loss_all += l
                total_chunk += 1
                total_all += 1

        # consider any remaining losses
        if len(losses) > 0:
            loss = dy.esum(losses)
            loss.scalar_value()
            loss.backward()
            self.trainer.update()
            dy.renew_cg()
        print(f'loss: {loss_all/total_all:.4f}')

    def get_hypers(self, query, vocab, ehs, w2i):
        print (query)
        eq = dy.average([self.word_embeddings[w2i[w]] if w in self.w2i else self.word_embeddings[0] for w in query])
        Ps = []
        for i in range(self.k):
            Ps.append(self.Phis[i].expr() * eq)
        P = dy.transpose(dy.concatenate_cols(Ps))
        s = P * ehs
        y = dy.logistic(self.W.expr() * s + self.b.expr())
        ans = [vocab[i] for i in y.npvalue().reshape(-1).argsort()[-15:]]
        return ans
