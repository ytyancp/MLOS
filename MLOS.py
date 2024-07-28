import math

import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial.distance import pdist
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import random

#Set the majority label to 1 and the minority label to 0

k = 5  ###Number of nearest neighbors
aer = -0.5  ###Making the parameter relatively small and ensuring that it gets smaller for increasingly sparse data


def imDataSet(path):
    dataset = np.loadtxt(open(path, "rb"), delimiter=",",
                         skiprows=0)  # read-in data

    return dataset


def Pre_Swim(dataset):
    majority = []
    minority = []
    for x in dataset:
        if x[len(x) - 1] == 1:
            majority.append(x)
        elif x[len(x) - 1] == 0:
            minority.append(x)
    # print(minority)
    # print(majority)
    return minority, majority


class SingularMatrixException(Exception):
    def __init__(self):
        Exception.__init__(self, "Singular data matrix... Using subspaces")


def _msqrt(X):
    #Compute the square root matrix of a symmetric square matrix X
    (L, V) = np.linalg.eig(X)  # Calculate the eigenvalues and eigenvectors
    return V.dot(np.diag(np.sqrt(L))).dot(V.T)


class MLOS:

    def __init__(self, minClass=None, subSpaceSampling=False, len=1):
        self.minClass = minClass
        self.subSpaceSampling = subSpaceSampling
        self.len = len

    # The data passed is transposed, so rows are features and columns are instances
    def fit_Sample(self, data, labels, numSamples):

        if self.minClass == None:
            self.minClass = np.argmin(np.bincount(labels.astype(int)))
        syntheticInstances = []
        data_maj_orig = data[np.where(labels != self.minClass)[0], :]
        data_min_orig = data[np.where(labels == self.minClass)[0], :]
        # print(data_min_orig,data_maj_orig)
        if (np.sum(labels == self.minClass) == 1):
            data_min_orig = data_min_orig.reshape(1, len(data_min_orig))
            # trnMinData    = trnMinData.reshape(1,len(trnMinData))
        ## STEP 1: CENTRE
        ## Centered on the majority class and centered on the minority class
        scaler = StandardScaler(with_std=False)  # The mean is subtracted from the standardized majority class, and the variance is not standardized
        T_maj = np.transpose(scaler.fit_transform(data_maj_orig))  # Finding the rank of rotation

        T_min = np.transpose(scaler.transform(data_min_orig))  # We use transform, which normalizes the minority class by the majority class mean
        if numSamples == 0:
            numSamples = len(data_maj_orig) - len(data_min_orig)
        ## STEP 2: WHITEN
        C_inv = None
        C = np.cov(T_maj)  # Covariance matrix of the majority class
        # Compute the rank of the majority class data matrix and invert it if possible
        data_rank = np.linalg.matrix_rank(data_maj_orig)
        if data_rank < T_maj.shape[0]:  # There exist linearly dependent columns, so the inverse will be singular
            if self.subSpaceSampling == False:
                print(
                    "The majority class has linearly dependent columns ")
                return data, labels
            else:

                QR = np.linalg.qr(data_maj_orig)
                indep = QR[1].diagonal() > 0
                data = data[:, indep]
                print("The majority class has linearly dependent columns. " + str(
                    sum(indep == True)) + " The original independent columns " + str(
                    data_maj_orig.shape[1]) )

        else:
            try:
                C_inv = np.linalg.inv(C)  # The inverse of the covariance matrix
                # print("协方差逆", C_inv)
            except np.linalg.LinAlgError as e:
                if 'Singular matrix' in str(e):
                    print("The majority class data is singular.")
                    X_new = data_min_orig[np.random.choice(data_min_orig.shape[0], numSamples, replace=True), :]
                    X_new = X_new + (0.1 * np.random.normal(0, data_maj_orig.std(0), X_new.shape))
                    y_new = np.repeat(self.minClass, numSamples)
                    data = np.concatenate([X_new, data])
                    labels = np.append(y_new, labels)
                    return data, labels

        M = _msqrt(
            C_inv)  # C_inv Is the inverse of the covariance matrix, and M is the matrix used for the Mahalanobis transform
        # print("M",M)
        M_inv = np.linalg.inv(M)  # Mahalanobis Transform - Mahalanobis minority class
        # 相乘
        # print(M) nn = get_NNlen(np.transpose(W_min), np.transpose(W_maj), self.len, paramer)
        W_min = M.dot(T_min)  # Mahalanobis Transform - Mahalanobis minority class
        W_maj = M.dot(T_maj)  # Mahalanobis Transform - Mahalanobis majority class
        a = np.concatenate((np.transpose(W_maj), np.transpose(W_min)), axis=0)
        a_label = np.concatenate((np.ones(len(np.transpose(W_maj))), np.zeros(len(np.transpose(W_min)))), axis=0)

        ###############################################改动部分###################################################################

        # Increasing the weight of the minority class, the weight of the overlapping part is higher,
        # the weight of the safe region is lower, and the noise samples or the synthetic samples of the sub-concept part are less.

        nn = get_NNlen(np.transpose(W_min), np.transpose(W_maj), self.len)

       #bodnum minority class boundary points are 0 and bodnum all point boundaries are shown


        ## STEP 3: Find the mean and feature boundaries to be used in the generation process
        min_means = W_min.mean(1)
        min_stds = W_min.std(1)
        ## STEP 4: Generate the synthetic instance
        #  Randomly copy the WHITENED MINORITY class INSTNACES<numSamples> times to generate synthetic INSTANCES from them

        choice = np.random.choice(data_min_orig.shape[0], numSamples, replace=True)
        smpInitPts = W_min[:, choice]  # The minority class points are randomly selected to form an array
        # smpInitPts.shape[1]=248
        # print(nn)
        ays = []
        ays1 = []
        for smpInd in range(smpInitPts.shape[1]):  # Repeat "times" so that we get a balanced dataset
            new_w_raw = []
            new = None
            new_w = None
            smp = smpInitPts[:, smpInd]
            random_r = random.uniform(0, nn[choice[smpInd]])
            random_vector = generate_random_vector(len(data[0]))
            normalized_vector = normalize_vector(random_vector)
            new_w_raw=smp+random_r*normalized_vector

            new_w = np.array(new_w_raw) / ((np.linalg.norm(new_w_raw) / np.linalg.norm(smp)))

            # Generate  point
            ## Step 5: Retract to the original empty
            ays.append(new_w)
            ays1.append(new_w_raw)
            new = M_inv.dot(np.array(new_w))
            syntheticInstances.append(new)

        x = np.concatenate((a, a_label.reshape(-1, 1)), axis=1)
        mins = list(np.transpose(W_min))
        mins = np.concatenate((mins, np.zeros(len(mins)).reshape(-1, 1)), axis=1)  # Minority class samples
        majs = list(np.transpose(W_maj))
        majs = np.concatenate((majs, np.ones(len(majs)).reshape(-1, 1)), axis=1)  # Minority class samples
        ays1 = np.concatenate((ays1, (np.ones(len(ays1)) * 2).reshape(-1, 1)), axis=1)  # Generate samples before transfer
        ays = np.concatenate((ays, (np.ones(len(ays1)) * 2).reshape(-1, 1)), axis=1)  # After generating the sample transfer
        ays1 = np.concatenate((mins, ays1), axis=0)  # Generate samples before transfer and minority class synthesis
        ays = np.concatenate((mins, ays), axis=0)  # Generate samples after transfer and minority class synthesis
        a = np.concatenate((majs, ays1), axis=0)  # Generate samples before transfer and minority class synthesis
        b = np.concatenate((majs, ays), axis=0)  # Generate samples after transfer and minority class synthesis


        MAJ = []
        for mj in np.transpose(W_maj):
            MAJ.append(M_inv.dot(np.array(mj)))
        MAJ = scaler.inverse_transform(np.array(MAJ))
        data_min_orig = np.array(data_min_orig)
        MAJ = np.array(MAJ)
        data = np.concatenate((data_min_orig, MAJ), axis=0)


        labels = np.concatenate((np.zeros(len(data_min_orig)), np.ones(len(MAJ))), axis=0)
        sampled_data = np.concatenate([scaler.inverse_transform(np.array(syntheticInstances)), data])  # 平移函数
        sampled_labels = np.append([self.minClass] * len(syntheticInstances), labels)
        return sampled_data, sampled_labels
def normalize_vector(q):
    q_length = np.linalg.norm(q)
    normalized_q = q / q_length
    return normalized_q
def generate_random_vector(n):
    return [random.uniform(-1, 1) for _ in range(n)]
def deleteall(min, maj):

    d = pdist(maj)
    mean_distance = np.mean(d)

    mean_standard = np.std(d)
    paramer = aer * mean_standard * mean_distance
    num, lens, dif, num_indices, dif_all = get_borderline_num(maj, min, k)  # Find the coordinates of the boundary points in the majority class
    nozero = np.where(num != 0)[0]
    e = [0 for n in range(len(nozero))]  ###Arrays are created dynamically based on the data dimension

    for i in range(len(nozero)):
        e[i] = Density(paramer, maj[nozero[i]], min)
    emean = np.mean(e)

    fmaj = []

    for i in range(len(nozero)):
        if emean <= e[i]:
            fmaj.append(maj[nozero[i]])

    num, lens, dif, num_indices, dif_all = get_borderline_num(fmaj, min, k)  # Find the coordinates of the boundary points in the majority class
    nbr = (num_indices * dif)[:, 1:]
    nnrbf = np.zeros((len(fmaj), k))
    for i in range(len(fmaj)):
        for j in range(k):
            if int(nbr[i][j]) != 0:
                nnrbf[i][j] = Density(paramer, min[int(nbr[i][j]) - len(fmaj)], min)
    fmin = []
    for i in range(len(fmaj)):
        max_index = np.argmax(nbr[i])
        fmin.append(min[int(nbr[i][max_index]) - len(fmaj)])


    main_set = {tuple(row) for row in maj}
    elements_set = {tuple(row) for row in fmaj}
    main_set1 = {tuple(row) for row in min}
    elements_set1 = {tuple(row) for row in fmin}
    # Using sets for difference set operations
    result_set = main_set.difference(elements_set)
    result_set1 = main_set1.difference(elements_set1)
    # Convert the result back to a NumPy array
    result_array = np.array([list(row) for row in result_set])
    result_array1 = np.array([list(row) for row in result_set1])
    # Test1(result_array)
    return result_array, result_array1


def get_NNlen(min, maj, len1):
    num, lens, dif, num_indices, dif_all = get_borderline_num(min, maj, k)
    num_indices = num_indices[:, 1:]
    dif = dif[:, 1:]
    bodnum = np.round((num_indices) * dif).astype(int)
    bodnum2 = np.round((num_indices)).astype(int)
    da = np.concatenate((min, maj))
    d = pdist(maj)
    mean_distance = np.mean(d)
    # print(mean_distance,"22",np.mean(maj_x))
    mean_standard = np.std(d)
    paramer = aer * mean_standard * mean_distance
    # print(type(min))
    e = [0 for n in range(len(min))]  ###Arrays are created dynamically based on the data dimension
    # Calculate the density of the minority class samples
    for i in range(len(min)):
        e[i] = Density(paramer, min[i], maj)
    bod_de = np.zeros((len(min), k), dtype=float)
    bod_de2 = np.zeros((len(min), k), dtype=float)
    # v
    for i in range(len(min)):
        for j in range(k):
            bod_de2[i][j] = Density(paramer, da[bodnum2[i][j]], maj)
        bod_de[i] = bod_de2[i] * dif[i]
    # Find the point with the closest density
    nn = [0 for n in range(len(min))]  ###根据数据维数动态创建数组
    for i in range(len(min)):
        result = find_closest_elements(bod_de[i], e[i])
        # print(e[i], bod_de[i],result)
        if result != -1:
            nn[i] = np.linalg.norm(da[bodnum[i][result]] - min[i])
        else:  # So if you don't have an outlier,
            result = find_closest_elements(bod_de2[i], e[i])

            nn[i] = len1 * np.linalg.norm(da[bodnum2[i][result]] - min[i])

    # print(nn)
    return nn
    # #####New data processing


def find_closest_elements(arr, n):
    min_difference = float('inf')
    closest_index = -1

    for i, num in enumerate(arr):
        if num != 0:
            diff = abs(num - n)
            if diff < min_difference:
                min_difference = diff
                closest_index = i

    return closest_index


def Density(paramer, min, maj):
    rbf = 0.0
    for j in range(len(maj)):
        rbf = np.exp(-1 * ((paramer * distance(maj[j], min)) ** 2)) + rbf
    return rbf


# Clean up the overlapping majority classes
def deletemaj(min, maj, bord_maj_num, lens, dif_all):
    bord_maj_num = bord_maj_num[:, 1:]
    if bord_maj_num.size == 0:  #######Edge majority class empty directly returns without deletion
        return maj
    duplicates = np.unique(np.array(bord_maj_num).reshape(-1))

    del_num = np.array(duplicates)
    maj1 = maj[~np.isin(np.arange(maj.shape[0]), del_num - len(min))]

    return maj1


# Find the probability density of the minority class against the majority class
def Density(paramer, min, maj):
    rbf = 0.0
    for j in range(len(maj)):
        rbf = np.exp(-1 * ((paramer * distance(maj[j], min)) ** 2)) + rbf
    # print(rbf)
    return rbf


def distance(x, y):
    return np.sum(np.abs(x - y))


# Find the number of samples of the majority class nearest to the minority class
def get_borderline_num(X_minority, X_majority, k_neighbor=5):
    # print(X_minority)
    X = np.concatenate((X_minority, X_majority), axis=0)
    y = np.concatenate((np.zeros(len(X_minority)), np.ones(len(X_majority))))
    # 计算距离矩阵
    nbrs = NearestNeighbors(n_neighbors=k_neighbor + 1).fit(X)
    distances, indices = nbrs.kneighbors(X)
    # 将每个样本的近邻中少数类样本的数量计算出来
    minority_indices = y[indices] == 1
    # print(y[indices])
    minority_counts = np.sum(minority_indices, axis=1)
    # 返回每个多数类样本的近邻中少数类样本的数量
    counts = minority_counts[:len(X_minority)]

    dif = y[indices][:len(X_minority), :]
    num_indices = indices[:len(X_minority), :]
    # num_indices = num_indices[:, 1:]
    # The return value, counts, is the number of heterogeneous samples
    # lens returns all nearest neighbor distances
    # dif returns the distribution of all neighbors
    #
    # num_indices returns the indices of the nearest neighbors
    # y[indices] returns the types of the nearest points
    return counts, distances, dif, num_indices, y[indices]


def Show_picture(dataset, mean):
    plt.gca().set(xlim=(0, 5), ylim=(0, 5))

    for x in dataset:
        if x[len(x) - 1] == 1:
            plt.plot(x[0], x[1], '.', c="y")
        elif x[len(x) - 1] == 0:
            plt.plot(x[0], x[1], '.', c="r")
        else:
            plt.plot(x[0], x[1], '.', c="g")
    plt.plot(mean[0], mean[1], '*', c="b", linewidth=2)
    plt.show()


def Test1(dataset):
    plt.gca().set(xlim=(-3, 5), ylim=(-3, 4))
    i = 0;
    j = 0;
    k = 0;
    for x in dataset:
        if x[2] - 0 == 0:
            if j == 0:
                plt.plot(x[0], x[1], 'o', c="b", markersize=4, markerfacecolor='white', label="Majority")
            else:
                plt.plot(x[0], x[1], 'o', c="b", markersize=4, markerfacecolor='white')
            j = j + 1
        elif x[2] - 1 == 0:
            if i == 0:
                plt.plot(x[0], x[1], 'o', c="r", markersize=4, markerfacecolor='white', label="Minority")
            else:
                plt.plot(x[0], x[1], 'o', c="r", markersize=4, markerfacecolor='white')
            i = i + 1
        else:
            if k == 0:
                plt.plot(x[0], x[1], 'o', c="y", markersize=4, markerfacecolor='white', label="Synthetic")
            else:
                plt.plot(x[0], x[1], 'o', c="y", markersize=4, markerfacecolor='white')
            k = k + 1
    plt.legend()
    plt.show()


if __name__ == '__main__':

    #test
    mean = [0, 0]
    path = r"D:\python_project\cover\testdataset\1\case_study_05.csv"
    data = imDataSet(path)
    min, maj = Pre_Swim(data)
    x_train = data[:, :-1]
    y_train = data[:, -1]
    sw = MLOS(len=1)
    X_res, y_res = sw.fit_Sample(x_train, y_train, len(maj) - len(min))
    X_res = np.array(X_res)
    c = np.append(X_res, y_res.reshape((len(y_res), 1)), axis=1)
    d = np.concatenate((min, maj))
    Show_picture(d, mean)
    Show_picture(c, mean)
