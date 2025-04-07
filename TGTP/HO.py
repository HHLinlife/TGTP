import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma


def levy(n, m, beta):
    num = gamma(1 + beta) * np.sin(np.pi * beta / 2)
    den = gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2)
    sigma_u = (num / den) ** (1 / beta)
    u = np.random.normal(0, sigma_u, (n, m))
    v = np.random.normal(0, 1, (n, m))
    z = u / (np.abs(v) ** (1 / beta))
    return z


def Ufun(x, a, k, m):
    return k * ((x - a) ** m) * (x > a) + k * ((-x - a) ** m) * (x < -a)


def F1(x):
    return np.sum(x ** 2)


def F2(x):
    return np.sum(np.abs(x)) + np.prod(np.abs(x))


def F3(x):
    dimension = len(x)
    R = 0
    for i in range(dimension):
        R += np.sum(x[:i + 1]) ** 2
    return R


def F4(x):
    return np.max(np.abs(x))


def F5(x):
    dimension = len(x)
    return np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (x[:-1] - 1) ** 2)


def F6(x):
    return np.sum(np.floor(x + 0.5) ** 2)


def F7(x):
    dimension = len(x)
    return np.sum(np.arange(1, dimension + 1) * (x ** 4)) + np.random.rand()


def F8(x):
    return np.sum(-x * np.sin(np.sqrt(np.abs(x))))


def F9(x):
    dimension = len(x)
    return np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x)) + 10 * dimension


def F10(x):
    dimension = len(x)
    return -20 * np.exp(-0.2 * np.sqrt(np.sum(x ** 2) / dimension)) - np.exp(
        np.sum(np.cos(2 * np.pi * x)) / dimension) + 20 + np.exp(1)


def F11(x):
    dimension = len(x)
    return np.sum(x ** 2) / 4000 - np.prod(np.cos(x / np.sqrt(np.arange(1, dimension + 1)))) + 1


def F12(x):
    dimension = len(x)
    term1 = (np.pi / dimension) * (
            10 * (np.sin(np.pi * (1 + (x[0] + 1) / 4)) ** 2) +
            np.sum(((x[:-1] + 1) / 4) ** 2 * (1 + 10 * (np.sin(np.pi * (1 + (x[1:] + 1) / 4)) ** 2))) +
            ((x[-1] + 1) / 4) ** 2)
    term2 = np.sum(Ufun(x, 10, 100, 4))
    return term1 + term2


def F13(x):
    dimension = len(x)
    term1 = 0.1 * ((np.sin(3 * np.pi * x[0]) ** 2) +
                   np.sum((x[:-1] - 1) ** 2 * (1 + (np.sin(3 * np.pi * x[1:]) ** 2))) +
                   (x[-1] - 1) ** 2 * (1 + (np.sin(2 * np.pi * x[-1]) ** 2)))
    term2 = np.sum(Ufun(x, 5, 100, 4))
    return term1 + term2


def F14(x):
    aS = np.array([[-32, -16, 0, 16, 32] * 5, [-32] * 5 + [-16] * 5 + [0] * 5 + [16] * 5 + [32] * 5])
    bS = np.zeros(25)
    for j in range(25):
        bS[j] = np.sum((x - aS[:, j]) ** 6)
    return (1 / 500 + np.sum(1 / (np.arange(1, 26) + bS))) ** (-1)


def F15(x):
    aK = np.array([0.1957, 0.1947, 0.1735, 0.16, 0.0844, 0.0627, 0.0456, 0.0342, 0.0323, 0.0235, 0.0246])
    bK = 1 / np.array([0.25, 0.5, 1, 2, 4, 6, 8, 10, 12, 14, 16])
    return np.sum((aK - (x[0] * (bK ** 2 + x[1] * bK) / (bK ** 2 + x[2] * bK + x[3]))) ** 2)


def F16(x):
    return 4 * (x[0] ** 2) - 2.1 * (x[0] ** 4) + (x[0] ** 6) / 3 + x[0] * x[1] - 4 * (x[1] ** 2) + 4 * (x[1] ** 4)


def F17(x):
    return (x[1] - (x[0] ** 2) * 5.1 / (4 * (np.pi ** 2)) + 5 / np.pi * x[0] - 6) ** 2 + 10 * (
            1 - 1 / (8 * np.pi)) * np.cos(x[0]) + 10


def F18(x):
    term1 = 1 + (x[0] + x[1] + 1) ** 2 * (19 - 14 * x[0] + 3 * (x[0] ** 2) - 14 * x[1] + 6 * x[0] * x[1] + 3 * (
            x[1] ** 2))
    term2 = 30 + (2 * x[0] - 3 * x[1]) ** 2 * (18 - 32 * x[0] + 12 * (x[0] ** 2) + 48 * x[1] - 36 * x[0] * x[1] + 27 * (
            x[1] ** 2))
    return term1 * term2


def F19(x):
    aH = np.array([[3, 10, 30], [0.1, 10, 35], [3, 10, 30], [0.1, 10, 35]])
    cH = np.array([1, 1.2, 3, 3.2])
    pH = np.array([[0.3689, 0.117, 0.2673], [0.4699, 0.4387, 0.747], [0.1091, 0.8732, 0.5547],
                   [0.03815, 0.5743, 0.8828]])
    R = 0
    for i in range(4):
        R += -cH[i] * np.exp(-(np.sum(aH[i, :] * ((x - pH[i, :]) ** 2))))
    return R


def F20(x):
    aH = np.array([[10, 3, 17, 3.5, 1.7, 8], [0.05, 10, 17, 0.1, 8, 14], [3, 3.5, 1.7, 10, 17, 8],
                   [17, 8, 0.05, 10, 0.1, 14]])
    cH = np.array([1, 1.2, 3, 3.2])
    pH = np.array([[0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886], [0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991],
                   [0.2348, 0.1415, 0.3522, 0.2883, 0.3047, 0.6650],
                   [0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381]])
    R = 0
    for i in range(4):
        R += -cH[i] * np.exp(-(np.sum(aH[i, :] * ((x - pH[i, :]) ** 2))))
    return R


def F21(x):
    aSH = np.array([[4, 4, 4, 4], [1, 1, 1, 1], [8, 8, 8, 8], [6, 6, 6, 6], [3, 7, 3, 7], [2, 9, 2, 9], [5, 5, 3, 3],
                    [8, 1, 8, 1], [6, 2, 6, 2], [7, 3.6, 7, 3.6]])
    cSH = np.array([0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5])
    R = 0
    for i in range(5):
        R += -((x - aSH[i, :]) @ (x - aSH[i, :]).T + cSH[i]) ** (-1)
    return R


def F22(x):
    aSH = np.array([[4, 4, 4, 4], [1, 1, 1, 1], [8, 8, 8, 8], [6, 6, 6, 6], [3, 7, 3, 7], [2, 9, 2, 9], [5, 5, 3, 3],
                    [8, 1, 8, 1], [6, 2, 6, 2], [7, 3.6, 7, 3.6]])
    cSH = np.array([0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5])
    R = 0
    for i in range(7):
        R += -((x - aSH[i, :]) @ (x - aSH[i, :]).T + cSH[i]) ** (-1)
    return R


def F23(x):
    aSH = np.array([[4, 4, 4, 4], [1, 1, 1, 1], [8, 8, 8, 8], [6, 6, 6, 6], [3, 7, 3, 7], [2, 9, 2, 9], [5, 5, 3, 3],
                    [8, 1, 8, 1], [6, 2, 6, 2], [7, 3.6, 7, 3.6]])
    cSH = np.array([0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5])
    R = 0
    for i in range(10):
        R += -((x - aSH[i, :]) @ (x - aSH[i, :]).T + cSH[i]) ** (-1)
    return R


def fun_info(F):
    if F == 'F1':
        fitness = F1
        lowerbound = -100
        upperbound = 100
        dimension = 30
    elif F == 'F2':
        fitness = F2
        lowerbound = -10
        upperbound = 10
        dimension = 30
    elif F == 'F3':
        fitness = F3
        lowerbound = -100
        upperbound = 100
        dimension = 30
    elif F == 'F4':
        fitness = F4
        lowerbound = -100
        upperbound = 100
        dimension = 30
    elif F == 'F5':
        fitness = F5
        lowerbound = -30
        upperbound = 30
        dimension = 30
    elif F == 'F6':
        fitness = F6
        lowerbound = -100
        upperbound = 100
        dimension = 30
    elif F == 'F7':
        fitness = F7
        lowerbound = -1.28
        upperbound = 1.28
        dimension = 30
    elif F == 'F8':
        fitness = F8
        lowerbound = -500
        upperbound = 500
        dimension = 30
    elif F == 'F9':
        fitness = F9
        lowerbound = -5.12
        upperbound = 5.12
        dimension = 30
    elif F == 'F10':
        fitness = F10
        lowerbound = -32
        upperbound = 32
        dimension = 30
    elif F == 'F11':
        fitness = F11
        lowerbound = -600
        upperbound = 600
        dimension = 30
    elif F == 'F12':
        fitness = F12
        lowerbound = -50
        upperbound = 50
        dimension = 30
    elif F == 'F13':
        fitness = F13
        lowerbound = -50
        upperbound = 50
        dimension = 30
    elif F == 'F14':
        fitness = F14
        lowerbound = -65.536
        upperbound = 65.536
        dimension = 2
    elif F == 'F15':
        fitness = F15
        lowerbound = -5
        upperbound = 5
        dimension = 4
    elif F == 'F16':
        fitness = F16
        lowerbound = -5
        upperbound = 5
        dimension = 2
    elif F == 'F17':
        fitness = F17
        lowerbound = np.array([-5, 0])
        upperbound = np.array([10, 15])
        dimension = 2
    elif F == 'F18':
        fitness = F18
        lowerbound = -2
        upperbound = 2
        dimension = 2
    elif F == 'F19':
        fitness = F19
        lowerbound = 0
        upperbound = 1
        dimension = 3
    elif F == 'F20':
        fitness = F20
        lowerbound = 0
        upperbound = 1
        dimension = 6
    elif F == 'F21':
        fitness = F21
        lowerbound = 0
        upperbound = 10
        dimension = 4
    elif F == 'F22':
        fitness = F22
        lowerbound = 0
        upperbound = 10
        dimension = 4
    elif F == 'F23':
        fitness = F23
        lowerbound = 0
        upperbound = 10
        dimension = 4
    return lowerbound, upperbound, dimension, fitness


def HO(SearchAgents, Max_iterations, lowerbound, upperbound, dimension, fitness):
    lowerbound = np.ones(dimension) * lowerbound
    upperbound = np.ones(dimension) * upperbound

    # Initialization
    X = np.zeros((SearchAgents, dimension))
    for i in range(dimension):
        X[:, i] = lowerbound[i] + np.random.rand(SearchAgents) * (upperbound[i] - lowerbound[i])

    fit = np.zeros(SearchAgents)
    for i in range(SearchAgents):
        L = X[i, :]
        fit[i] = fitness(L)

    best_so_far = np.zeros(Max_iterations)

    for t in range(Max_iterations):
        best = np.min(fit)
        location = np.argmin(fit)
        if t == 0:
            Xbest = X[location, :]
            fbest = best
        elif best < fbest:
            fbest = best
            Xbest = X[location, :]

        # Phase 1: The hippopotamuses position update in the river or pond (Exploration)
        for i in range(int(SearchAgents / 2)):
            Dominant_hippopotamus = Xbest
            I1 = np.random.randint(1, 3)
            I2 = np.random.randint(1, 3)
            Ip1 = np.random.randint(0, 2, 2)
            RandGroupNumber = np.random.choice(SearchAgents)
            RandGroup = np.random.choice(SearchAgents, RandGroupNumber, replace=False)

            if len(RandGroup) != 1:
                MeanGroup = np.mean(X[RandGroup, :], axis=0)
            else:
                MeanGroup = X[RandGroup[0], :]

            Alfa = [
                I2 * np.random.rand(dimension) + (1 - Ip1[0]),
                2 * np.random.rand(dimension) - 1,
                np.random.rand(dimension),
                I1 * np.random.rand(dimension) + (1 - Ip1[1]),
                np.random.rand()
            ]
            A = Alfa[np.random.randint(0, 5)]
            B = Alfa[np.random.randint(0, 5)]

            X_P1 = X[i, :] + np.random.rand() * (Dominant_hippopotamus - I1 * X[i, :])
            T = np.exp(-t / Max_iterations)
            if T > 0.6:
                X_P2 = X[i, :] + A * (Dominant_hippopotamus - I2 * MeanGroup)
            else:
                if np.random.rand() > 0.5:
                    X_P2 = X[i, :] + B * (MeanGroup - Dominant_hippopotamus)
                else:
                    X_P2 = (upperbound - lowerbound) * np.random.rand(dimension) + lowerbound

            X_P2 = np.clip(X_P2, lowerbound, upperbound)

            F_P1 = fitness(X_P1)
            if F_P1 < fit[i]:
                X[i, :] = X_P1
                fit[i] = F_P1

            F_P2 = fitness(X_P2)
            if F_P2 < fit[i]:
                X[i, :] = X_P2
                fit[i] = F_P2

        # Phase 2: Hippopotamus defense against predators (Exploration)
        for i in range(int(SearchAgents / 2), SearchAgents):
            predator = lowerbound + np.random.rand(dimension) * (upperbound - lowerbound)
            F_HL = fitness(predator)
            distance2Leader = np.abs(predator - X[i, :])
            b = np.random.uniform(2, 4)
            c = np.random.uniform(1, 1.5)
            d = np.random.uniform(2, 3)
            l = np.random.uniform(-2 * np.pi, 2 * np.pi)
            RL = 0.05 * levy(SearchAgents, dimension, 1.5)

            if fit[i] > F_HL:
                X_P3 = RL[i, :] * predator + (b / (c - d * np.cos(l))) * (1 / distance2Leader)
            else:
                X_P3 = RL[i, :] * predator + (b / (c - d * np.cos(l))) * (1 / (2 * distance2Leader + np.random.rand(dimension)))

            X_P3 = np.clip(X_P3, lowerbound, upperbound)

            F_P3 = fitness(X_P3)
            if F_P3 < fit[i]:
                X[i, :] = X_P3
                fit[i] = F_P3

        # Phase 3: Hippopotamus Escaping from the Predator (Exploitation)
        for i in range(SearchAgents):
            LO_LOCAL = lowerbound / (t + 1)
            HI_LOCAL = upperbound / (t + 1)
            Alfa = [
                2 * np.random.rand(dimension) - 1,
                np.random.rand(),
                np.random.randn()
            ]
            D = Alfa[np.random.randint(0, 3)]
            X_P4 = X[i, :] + np.random.rand() * (LO_LOCAL + D * (HI_LOCAL - LO_LOCAL))
            X_P4 = np.clip(X_P4, lowerbound, upperbound)

            F_P4 = fitness(X_P4)
            if F_P4 < fit[i]:
                X[i, :] = X_P4
                fit[i] = F_P4

        best_so_far[t] = fbest
        print(f'Iteration {t + 1}: Best Cost = {best_so_far[t]}')

    Best_score = fbest
    Best_pos = Xbest
    HO_curve = best_so_far

    return Best_score, Best_pos, HO_curve



def main():
    # Select the test function, can choose from 'F1' to 'F23'
    Fun_name = 'F2'
    # Number of search agents (number of hippopotamuses, i.e., population members)
    SearchAgents = 16
    # Maximum number of iterations
    Max_iterations = 500

    # Get relevant information of the objective function, including lower bound, upper bound, dimension, and fitness function handle
    lowerbound, upperbound, dimension, fitness = fun_info(Fun_name)

    # Call the HO function for optimization
    Best_score, Best_pos, HO_curve = HO(SearchAgents, Max_iterations, lowerbound, upperbound, dimension, fitness)

    # Display the best solution and the best optimal value
    print(f'The best solution obtained by HO for {Fun_name} is : {Best_pos}')
    print(f'The best optimal value of the objective function found by HO for {Fun_name} is : {Best_score}')

    # Plot the iteration curve
    plt.figure()
    plt.semilogy(HO_curve, color='#b28d90', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Best score obtained so far')
    plt.box(True)
    plt.legend(['HO'])
    plt.show()


if __name__ == "__main__":
    main()
