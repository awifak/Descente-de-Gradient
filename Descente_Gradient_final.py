import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('data.csv', sep=';')
df2 = np.array(df['USA'])
# df2 = df['USA'] -> ne pas faire ça car sinon que des NaN car ça crée des "series" càd liste avec un index
X = df2[:-1] #prend tout sauf la dernière valeur
Y = df2[1:] #prend tout sauf la première valeur


def descente_gradient(t: float, n: int, X, Y, epsilon=1e-6):
    N = len(X)

    a = np.random.uniform(-1, 1)
    b = np.random.uniform(-1, 1)
    #choix de a et b aléatoire

    J_0 = np.inf

    for i in range(n):
        P = a * X + b

        J = (1 / (2 * N)) * np.sum((P - Y) ** 2)

        gradient_a = (1 / N) * np.sum(X * (P - Y))
        gradient_b = (1 / N) * np.sum(P - Y)

        if abs(J_0 - J) < epsilon:
            print(f"Convergence à l'itération {i}")
            break

        J_0 = J #car la variation sera à ce stade très très faible

        a = a - (t * gradient_a)
        b = b - (t * gradient_b)

    return a, b

a, b = descente_gradient(0.01, 10000, X, Y)
print(a, b)

plt.scatter(X, Y, color='blue')
plt.plot(X, a * X + b, color='red')
plt.show()
