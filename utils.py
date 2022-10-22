import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
class dataset():
    def __init__(self, n_points=10000):
        self.n_points = n_points
        for i in range(n_points):
            label = np.random.choice([0, 1])
            sign = np.random.choice([-1, 1])
            x = np.random.uniform(-1, 1)
            if label == 0:
                y = ((1 - x**2)**0.5) * sign
            else:
                y = ((1 - x**2)**0.5) * sign + 3
            if i == 0:
                self.data = pd.DataFrame({'X': [x], 'y': [y], 'label': [label]})
            else:
                self.data = pd.concat([self.data, pd.DataFrame({'X': [x], 'y': [y], 'label': [label]})], ignore_index=True)

        self.noisy_data = self.data.copy(deep=True)

    def get(self, add_noise=False):
        if add_noise:
            self.noisy_data['X'] = self.data['X'] + np.random.normal(0, 0.1, self.n_points)
            self.noisy_data['y'] = self.data['y'] + np.random.normal(0, 0.1, self.n_points)
        return (self.data if not add_noise else self.noisy_data)

    def plot(self, add_noise=False):
        data = self.data if not add_noise else self.noisy_data
        plt.figure(figsize=(7, 7))
        sns.scatterplot(data=data, x='X', y='y', hue='label', style='label', palette='rainbow').set_title('Dataset')
        plt.show()

class xao_dataset():
    def And():
        data = pd.DataFrame({'X': [0, 0, 1, 1], 'y': [0, 1, 0, 1], 'label': [0, 0, 0, 1]})
        return data

    def Or():
        data = pd.DataFrame({'X': [0, 0, 1, 1], 'y': [0, 1, 0, 1], 'label': [0, 1, 1, 1]})
        return data
    
    def Xor():
        data = pd.DataFrame({'X': [0, 0, 1, 1], 'y': [0, 1, 0, 1], 'label': [0, 1, 1, 0]})
        return data

class perceptron():
    def __init__(self, data, alpha=0.01, epochs=5):
        self.data = data
        self.alpha = alpha
        self.epochs = epochs
        self.w = np.zeros((2, 1))
        self.b = 0

    def train(self, bias_0=False):
        X = self.data[['X', 'y']].values
        y = self.data['label'].values 
        miss = []
        def activation(x):
            return 1 if x >= 0 else 0
        while self.epochs > 0:
            missclassified = 0
            for i in range(self.data.shape[0]):
                error = y[i] - activation(np.dot(X[i, :], self.w) + self.b)
                if error != 0:
                    self.w = self.w + error * X[i, :].reshape(2, 1)
                    if not bias_0:
                        self.b = self.b + error
                    # self.b = self.b + error
                    missclassified += 1
            self.epochs -= 1
            miss.append(missclassified)
            if missclassified == 0:
                break
        
        return self.w, self.b, miss 
    
    def plot(self, w):
        plt.figure(figsize=(7, 7))
        sns.scatterplot(data=self.data, x=self.data['X'], y=self.data['y'], hue='label', style='label', palette='rainbow').set_title('Decision Boundary') #.cat.codes
        x = np.linspace(-1, 1, 200)
        y = -(w[0] * x + self.b) / w[1] if w[1] != 0 else 0*x

        plt.plot(x, y, 'r')
        plt.show()
    
if __name__ == '__main__':
    data = dataset()
    p = perceptron() 
