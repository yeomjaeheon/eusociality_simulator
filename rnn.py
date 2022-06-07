import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return 2 * sigmoid(2 * x) - 1

def softmax(x):
    return np.exp(x - np.max(x)) / np.sum(np.exp(x - np.max(x)))

def get_weight(input_size, output_size):
    return np.random.normal(loc = 0.0, scale = 1 / (input_size ** 0.5), size = (input_size, output_size))

class nn:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size, self.hidden_size, self.output_size = input_size, hidden_size, output_size
        #가중치 초기화
        self.wx = get_weight(input_size, hidden_size)
        self.wh = get_weight(hidden_size, hidden_size)
        self.wy = get_weight(hidden_size, output_size)
        #내부 메모리 초기화
        self.h = np.zeros((1, hidden_size))
        #돌연변이 관련
        self.mutate_prob = 0.1
        self.mutate_std = 0.01 #돌연변이 행렬의 표준편차

    def forward(self, input_data):
        self.h = tanh(np.dot(input_data, self.wx) + np.dot(self.h, self.wh))
        #시행 불가능한 액션의 선호도를 -np.inf로 만들어 배제할 것이고 오차역전파는 사용되지 않으므로 출력층에는 활성화 함수 생략
        output = np.dot(self.h, self.wy)[0]
        return output

    def reset(self, wx, wh, wy):
        self.wx = wx
        self.wh = wh
        self.wy = wy

    def mutate(self):
        prob = []
        for i in range(0, 3):
            prob.append(np.random.rand())
        if prob[0] <= self.mutate_prob:
            self.wx += np.random.normal(loc = 0.0, scale = self.mutate_std, size = self.wx.shape)
        if prob[1] <= self.mutate_prob:
            self.wh += np.random.normal(loc = 0.0, scale = self.mutate_std, size = self.wh.shape)
        if prob[2] <= self.mutate_prob:
            self.wy += np.random.normal(loc = 0.0, scale = self.mutate_std, size = self.wy.shape)

    def reset_memory(self):
        #내부 메모리 초기화
        self.h = np.zeros((1, self.hidden_size))