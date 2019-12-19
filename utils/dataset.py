import pickle


class Bunch(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self


def bunch2json(bunch, path):
    # bunch 序列化为 JSON
    with open(path, 'wb') as fp:
        pickle.dump(bunch, fp)


def json2bunch(path):
    # JSON 反序列化为 bunch
    with open(path, 'rb') as fp:
        X = pickle.load(fp)
    return X