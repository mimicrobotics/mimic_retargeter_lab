class DataLogger:
    def __init__(self):
        self.data = {}

    def update(self, data_dict):
        for k, v in data_dict.items():
            if k not in self.data:
                self.data[k] = []
            self.data[k].append(v)
