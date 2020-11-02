import pandas as pd
import numpy as np
import time
from sklearn import preprocessing
from sklearn import metrics
from sklearn.metrics import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.random_projection import SparseRandomProjection
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn import cluster
from sklearn.neural_network import MLPClassifier
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
matplotlib.rc('figure', max_open_warning = 0)


RANDOM_STATE = 21

# load the dataset
def load_dataset(full_path):
    wine = pd.read_csv(full_path, delimiter=";")

    wine['quality'] = wine['quality'].replace([3,4,5,6,7,8,9],['0','0','0','1','1','1','1'])
    X = wine.iloc[:, :-1].values
    y = wine.iloc[:, -1].values
    # print("x before", X.shape)
    # print("y before", y.shape)

    X = preprocessing.scale(X)

    y = LabelEncoder().fit_transform(y)
    # print("x After", X.shape)
    # print("y After", y.shape)
    # u, count = np.unique(y, return_counts=True)
    # print(u)
    # print(count)

    return X, y

def classification_metrics(Y_pred, Y_true):
    accuracy_lr = accuracy_score(Y_true, Y_pred)
    return accuracy_lr

def display_metrics_NN(classifierName,Y_pred,Y_true):
    print("______________________________________________")
    print(("Classifier: "+classifierName))
    acc = classification_metrics(Y_pred,Y_true)
    print(("Accuracy: "+str(acc)))
    print("______________________________________________")
    print("")
    return str(acc)

def display_metrics(title, ylabel, algo_label):
    print("--------------------------------------")
    print(title)
    print("--------------------------------------")
    print("Homogeneity: %0.3f" % metrics.homogeneity_score(ylabel, algo_label))
    print("Completeness: %0.3f" % metrics.completeness_score(ylabel, algo_label))
    print("V-measure: %0.3f" % metrics.v_measure_score(ylabel, algo_label))
    print("Rand Score: %0.3f" % metrics.adjusted_rand_score(ylabel, algo_label))
    print("Mutual Info: %0.3f" % metrics.adjusted_mutual_info_score(ylabel, algo_label))
    print(" ")

def plot_learning_curves_NN(title, train_losses, train_accuracies, valid_accuracies):
    # Reference : https://matplotlib.org/tutorials/introductory/pyplot.html
    # Reference: https://stackoverflow.com/questions/4805048/how-to-get-different-colored-lines-for-different-plots-in-a-single-figure
    # Reference: https://github.com/ast0414/CSE6250BDH-LAB-DL/blob/master/3_RNN.ipynb
    plt.figure()
    plt.grid()
    plt.plot(np.arange(len(train_losses)), train_losses, label='Training Loss')
    plt.ylabel('Loss')
    plt.xlabel('epoch')
    plt.title('Loss Curve_'  + title)
    plt.legend(loc="best")
    plt.savefig("UL1_MLP_Loss_Curve_" + title + ".png")

    plt.figure()
    plt.grid()
    plt.plot(np.arange(len(train_accuracies)), train_accuracies, label='Training Accuracy')
    plt.plot(np.arange(len(valid_accuracies)), valid_accuracies, label='Validation Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('epoch')
    plt.title('Accuracy Curve_'  + title)
    plt.legend(loc="best")
    plt.savefig("UL1_MLP_Accuracy_Curve_" + title + ".png")

def neural_network(title, xtrain, ytrain, xtest, ytest):
    mlpClass_lc = MLPClassifier(hidden_layer_sizes=(5, 2), random_state=RANDOM_STATE, max_iter=5000, warm_start=True, alpha=0.1)
    num_epochs = 1000
    train_losses, train_accuracies, valid_accuracies = np.empty(num_epochs), np.empty(num_epochs), np.empty(num_epochs)
    # Split training set into training and validation
    X_train_NN, X_val, Y_train_NN, Y_val = train_test_split(xtrain, ytrain, test_size=0.2, random_state=RANDOM_STATE)
    for i in range(num_epochs):
        mlpClass_lc.fit(X_train_NN, Y_train_NN)
        train_losses[i] = mlpClass_lc.loss_
        train_accuracies[i] = accuracy_score(Y_train_NN, mlpClass_lc.predict(X_train_NN))
        valid_accuracies[i] = accuracy_score(Y_val, mlpClass_lc.predict(X_val))

    t_bef = time.time()
    Y_pred = mlpClass_lc.predict(xtest)
    t_aft = time.time()
    mlp_accuracy = display_metrics_NN(title, Y_pred, ytest)
    query_time = t_aft - t_bef

    plot_learning_curves_NN(title, train_losses, train_accuracies, valid_accuracies)

    return query_time

def main():

    full_path = 'data/winequality-white.csv'
    X, y = load_dataset(full_path)
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

    kmeans_train_time = np.zeros(5)
    kmeans_predict_time = np.zeros(5)
    em_train_time = np.zeros(5)
    em_predict_time = np.zeros(5)
    nn_predict_time = np.zeros(9)

    # Ref: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
    # https://www.edupristine.com/blog/beyond-k-means
    # https://www.linkedin.com/pulse/finding-optimal-number-clusters-k-means-through-elbow-asanka-perera
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=RANDOM_STATE)
        kmeans.fit(X_train)
        wcss.append(kmeans.inertia_)
    plt.figure()
    plt.plot(range(1, 11), wcss)
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.savefig("UL1_WCSS.png")

    t_bef = time.time()
    kmeans = KMeans(n_clusters=2, random_state=RANDOM_STATE).fit(X_train)
    t_aft = time.time()
    kmeans_train_time[0] = t_aft - t_bef
    kmeans_label_train = kmeans.labels_
    centroids = kmeans.cluster_centers_

    t_bef = time.time()
    kmeans_label_test = kmeans.predict(X_test)
    t_aft = time.time()
    kmeans_predict_time[0] = t_aft - t_bef

    plt.figure()
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True,figsize=(10,6))
    ax1.set_title("Original")
    ax1.scatter(X_train[:,0],X_train[:,1],c=Y_train, alpha=0.5)
    ax2.set_title('K Means')
    ax2.scatter(X_train[:,0],X_train[:,1],c=kmeans_label_train, alpha=0.5)
    ax2.scatter(centroids[:, 0], centroids[:, 1], c='red')
    plt.savefig("UL1_kmeans.png")

    display_metrics("Original Kmeans Train", Y_train, kmeans_label_train)
    display_metrics("Original Kmeans Test", Y_test, kmeans_label_test)

    # Reference: https://jakevdp.github.io/PythonDataScienceHandbook/05.09-principal-component-analysis.html#:~:text=Choosing%20the%20number%20of%20components,pca%20%3D%20PCA().
    # PCA -------------------------------------------------------------
    plt.figure()
    pca = PCA().fit(X_train)
    eigenvalues = pca.explained_variance_
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance')
    plt.savefig("UL1_PCA_variance.png")

    pca = PCA(n_components = 9, random_state=RANDOM_STATE)
    pca.fit(X_train)
    pca_trans_train = pca.transform(X_train)
    pca_trans_test = pca.transform(X_test)

    # Run on tranformed PCA dataset
    wcss_pca = []
    for i in range(1, 11):
        kmeans_pca = KMeans(n_clusters=i, init='k-means++', random_state=RANDOM_STATE)
        kmeans_pca.fit(pca_trans_train)
        wcss_pca.append(kmeans_pca.inertia_)
    plt.figure()
    plt.plot(range(1, 11), wcss_pca)
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.savefig("UL1_WCSS_After_PCA.png")

    t_bef = time.time()
    kmeans_pca = KMeans(n_clusters=2, random_state=RANDOM_STATE).fit(pca_trans_train)
    t_aft = time.time()
    kmeans_train_time[1] = t_aft - t_bef

    kmeans_pca.predict(pca_trans_train)
    kmeans_pca_label = kmeans_pca.labels_
    centroids_pca = kmeans_pca.cluster_centers_

    t_bef = time.time()
    kmeans_pca_label_test = kmeans_pca.predict(pca_trans_test)
    t_aft = time.time()
    kmeans_predict_time[1] = t_aft - t_bef

    display_metrics("Kmeans Train after PCA", Y_train, kmeans_pca_label)
    display_metrics("Kmeans Test after PCA", Y_test, kmeans_pca_label_test)

    plt.figure()
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True,figsize=(10,6))
    ax1.set_title("K means before PCA")
    ax1.scatter(X_train[:,0],X_train[:,1],c=kmeans_label_train, alpha=0.5)
    ax1.scatter(centroids[:, 0], centroids[:, 10], c='red')
    ax2.set_title("K Means after PCA")
    ax2.scatter(pca_trans_train[:,0],pca_trans_train[:,1],c=kmeans_pca_label, alpha=0.5)
    ax2.scatter(centroids_pca[:, 0], centroids_pca[:, 8], c='red')
    plt.savefig("UL1_kmeans_aft_PCA.png")

    # ICA -------------------------------------------------------------
    dims = range(1,11)
    kurt = []
    for dim in dims:
        ica = FastICA(n_components=dim, random_state=RANDOM_STATE)
        tmp = ica.fit_transform(X_train)
        tmp = pd.DataFrame(tmp)
        tmp = tmp.kurt(axis=0)
        kurt.append(tmp.abs().mean())
    plt.figure()
    plt.title("ICA Kurtosis")
    plt.xlabel("Independent Components")
    plt.ylabel("Avg Kurtosis Across IC")
    plt.plot(dims, kurt)
    plt.savefig("UL1_ICA_kurtosis.png")

    ica = FastICA(n_components = 9, algorithm = 'parallel',whiten = True, random_state=RANDOM_STATE)
    ica.fit(X_train)
    ica_trans_train = ica.transform(X_train)
    ica_trans_test = ica.transform(X_test)

    # Run on tranformed ICA dataset
    wcss_ica = []
    for i in range(1, 11):
        kmeans_ica = KMeans(n_clusters=i, init='k-means++', random_state=RANDOM_STATE)
        kmeans_ica.fit(ica_trans_train)
        wcss_ica.append(kmeans_ica.inertia_)
    plt.figure()
    plt.plot(range(1, 11), wcss_ica)
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.savefig("UL1_WCSS_After_ICA.png")

    t_bef = time.time()
    kmeans_ica = KMeans(n_clusters=2, random_state=RANDOM_STATE).fit(ica_trans_train)
    t_aft = time.time()
    kmeans_train_time[2] = t_aft - t_bef

    kmeans_ica.predict(ica_trans_train)
    kmeans_ica_label = kmeans_ica.labels_
    centroids_ica = kmeans_ica.cluster_centers_

    t_bef = time.time()
    kmeans_ica_label_test = kmeans_ica.predict(ica_trans_test)
    t_aft = time.time()
    kmeans_predict_time[2] = t_aft - t_bef

    display_metrics("Kmeans Train after ICA", Y_train, kmeans_ica_label)
    display_metrics("Kmeans Test after ICA", Y_test, kmeans_ica_label_test)

    plt.figure()
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True,figsize=(10,6))
    ax1.set_title("K means before ICA")
    ax1.scatter(X_train[:,0],X_train[:,1],c=kmeans_label_train, alpha=0.5)
    ax1.scatter(centroids[:, 0], centroids[:, 1], c='red')
    ax2.set_title('K Means after ICA')
    ax2.scatter(ica_trans_train[:,0],ica_trans_train[:,1],c=kmeans_ica_label, alpha=0.5)
    ax2.scatter(centroids_ica[:, 0], centroids_ica[:, 1], c='red')
    plt.savefig("UL1_kmeans_aft_ICA.png")

    # RP -------------------------------------------------------------
    rp = SparseRandomProjection(n_components=9, random_state=RANDOM_STATE)
    rp.fit(X_train)
    rp_trans_train = rp.transform(X_train)
    rp_trans_test = rp.transform(X_test)

    # Run on tranformed RP dataset
    wcss_rp = []
    for i in range(1, 11):
        kmeans_rp = KMeans(n_clusters=i, init='k-means++', random_state=RANDOM_STATE)
        kmeans_rp.fit(rp_trans_train)
        wcss_rp.append(kmeans_rp.inertia_)
    plt.figure()
    plt.plot(range(1, 11), wcss_rp)
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.savefig("UL1_WCSS_After_RP.png")

    t_bef = time.time()
    kmeans_rp = KMeans(n_clusters=2, random_state=RANDOM_STATE).fit(rp_trans_train)
    t_aft = time.time()
    kmeans_train_time[3] = t_aft - t_bef

    kmeans_rp_label = kmeans_rp.labels_
    centroids_rp = kmeans_rp.cluster_centers_

    t_bef = time.time()
    kmeans_rp_label_test = kmeans_rp.predict(rp_trans_test)
    t_aft = time.time()
    kmeans_predict_time[3] = t_aft - t_bef

    display_metrics("Kmeans Train after RP", Y_train, kmeans_rp_label)
    display_metrics("Kmeans Test after RP", Y_test, kmeans_rp_label_test)

    plt.figure()
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True,figsize=(10,6))
    ax1.set_title("K means before RP")
    ax1.scatter(X_train[:,0],X_train[:,1],c=kmeans_label_train, alpha=0.5)
    ax1.scatter(centroids[:, 0], centroids[:, 1], c='red')
    ax2.set_title("K Means after RP")
    ax2.scatter(rp_trans_train[:,0],rp_trans_train[:,1],c=kmeans_rp_label, alpha=0.5)
    ax2.scatter(centroids_rp[:, 0], centroids_rp[:, 1], c='red')
    plt.savefig("UL1_kmeans_aft_RP.png")


    # LDA -------------------------------------------------------------
    lda = LDA(n_components=1)
    lda_trans_train = lda.fit_transform(X_train, Y_train)
    lda_trans_test = lda.transform(X_test)

    # Run on tranformed LDA dataset
    wcss_lda = []
    for i in range(1, 11):
        kmeans_lda = KMeans(n_clusters=i, init='k-means++', random_state=RANDOM_STATE)
        kmeans_lda.fit(lda_trans_train)
        wcss_lda.append(kmeans_lda.inertia_)
    plt.figure()
    plt.plot(range(1, 11), wcss_lda)
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.savefig("UL1_WCSS_After_LDA.png")

    t_bef = time.time()
    kmeans_lda = KMeans(n_clusters=2, random_state=RANDOM_STATE).fit(lda_trans_train)
    t_aft = time.time()
    kmeans_train_time[4] = t_aft - t_bef

    kmeans_lda_label = kmeans_lda.labels_
    centroids_lda = kmeans_lda.cluster_centers_

    t_bef = time.time()
    kmeans_lda_label_test = kmeans_lda.predict(lda_trans_test)
    t_aft = time.time()
    kmeans_predict_time[4] = t_aft - t_bef

    display_metrics("Kmeans Train after LDA", Y_train, kmeans_lda_label)
    display_metrics("Kmeans Test after LDA", Y_test, kmeans_lda_label_test)

    plt.figure()
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True,figsize=(10,6))
    ax1.set_title("K means before LDA")
    ax1.scatter(X_train[:,0],X_train[:,1],c=kmeans_label_train, alpha=0.5)
    ax1.scatter(centroids[:, 0], centroids[:, 1], c='red')
    ax2.set_title("K Means after LDA")
    ax2.scatter(lda_trans_train[:,0] , lda_trans_train[:,0], c=kmeans_lda_label, alpha=0.5)
    ax2.scatter(centroids_lda[:, 0], centroids_lda[:, 0], c='red')
    plt.savefig("UL1_kmeans_aft_LDA.png")

    # Train time Different Dimensionality reduction algorithms kmeans
    classifier = ['Kmeans', 'Kmeans with PCA', 'Kmeans with ICA', 'Kmeans with RP', 'Kmeans with LDA']
    np_classifier = np.array(classifier)
    plt.figure()
    plt.barh(np_classifier, kmeans_train_time, align = 'center')
    plt.title('Kmeans Train Time')
    plt.ylabel('Name')
    plt.xlabel('Time (seconds)')
    plt.savefig('UL1_Kmeans_Traintime.png', bbox_inches = "tight")

    # Predict time Different Dimensionality reduction algorithms kmeans
    plt.figure()
    plt.barh(np_classifier, em_predict_time, align = 'center')
    plt.title('Kmeans Query Time')
    plt.ylabel('Name')
    plt.xlabel('Time (seconds)')
    plt.savefig('UL1_Kmeans_Querytime.png', bbox_inches = "tight")

# Expectation Maximization ---------------------------------------------------
    silhouette_score = []
    for i in range(2, 12):
        gmm = GaussianMixture(n_components=i, n_init=2, random_state=RANDOM_STATE).fit(X_train)
        gmm_predict = gmm.predict(X_train)
        silhouette_score.append(metrics.silhouette_score(X_train, gmm_predict, metric='euclidean'))
    plt.figure()
    plt.plot(range(1, 11), silhouette_score)
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('silhouette_score')
    plt.savefig("UL1_SS.png")

    t_bef = time.time()
    gmm = GaussianMixture(n_components=2).fit(X_train)
    t_aft = time.time()
    em_train_time[0] = t_aft - t_bef

    gmm_label_train = gmm.predict(X_train)

    t_bef = time.time()
    gmm_label_test = gmm.predict(X_test)
    t_aft = time.time()
    em_predict_time[0] = t_aft - t_bef

    display_metrics("Original EM Train", Y_train, gmm_label_train)
    display_metrics("Original EM Test", Y_test, gmm_label_test)

    plt.figure()
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True,figsize=(10,6))
    ax1.set_title("Original")
    ax1.scatter(X_train[:,0],X_train[:,1],c=Y_train, alpha=0.5)
    ax2.set_title("Expectation Maximization")
    ax2.scatter(X_train[:,0],X_train[:,1],c=gmm_label_train, alpha=0.5)
    plt.savefig("UL1_EM.png")

    # PCA --------------------
    silhouette_score_pca = []
    for i in range(2, 12):
        gmm_pca = GaussianMixture(n_components=i, n_init=2, random_state=RANDOM_STATE).fit(pca_trans_train)
        gmm_predict_pca = gmm_pca.predict(pca_trans_train)
        silhouette_score_pca.append(metrics.silhouette_score(pca_trans_train, gmm_predict_pca, metric='euclidean'))
    plt.figure()
    plt.plot(range(1, 11), silhouette_score_pca)
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('silhouette_score')
    plt.savefig("UL1_SS_After_PCA.png")

    t_bef = time.time()
    gmm_pca = GaussianMixture(n_components=2).fit(pca_trans_train)
    t_aft = time.time()
    em_train_time[1] = t_aft - t_bef

    gmm_label_pca = gmm_pca.predict(pca_trans_train)

    t_bef = time.time()
    gmm_label_pca_test = gmm_pca.predict(pca_trans_test)
    t_aft = time.time()
    em_predict_time[1] = t_aft - t_bef

    display_metrics("EM Train after PCA", Y_train, gmm_label_pca)
    display_metrics("EM Test after PCA", Y_test, gmm_label_pca_test)

    plt.figure()
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True,figsize=(10,6))
    ax1.set_title("EM before PCA")
    ax1.scatter(X_train[:,0],X_train[:,1],c=gmm_label_train, alpha=0.5)
    ax2.set_title("EM after PCA")
    ax2.scatter(pca_trans_train[:,0],pca_trans_train[:,1],c=gmm_label_pca, alpha=0.5)
    plt.savefig("UL1_EM_aft_PCA.png")

    # ICA --------------------
    silhouette_score_ica = []
    for i in range(2, 12):
        gmm_ica = GaussianMixture(n_components=i, n_init=2, random_state=RANDOM_STATE).fit(ica_trans_train)
        gmm_predict_ica = gmm_ica.predict(ica_trans_train)
        silhouette_score_ica.append(metrics.silhouette_score(ica_trans_train, gmm_predict_ica, metric='euclidean'))
    plt.figure()
    plt.plot(range(1, 11), silhouette_score_ica)
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('silhouette_score')
    plt.savefig("UL1_SS_After_ICA.png")

    t_bef = time.time()
    gmm_ica = GaussianMixture(n_components=2).fit(ica_trans_train)
    t_aft = time.time()
    em_train_time[2] = t_aft - t_bef

    gmm_label_ica = gmm_ica.predict(ica_trans_train)

    t_bef = time.time()
    gmm_label_ica_test = gmm_ica.predict(ica_trans_test)
    t_aft = time.time()
    em_predict_time[2] = t_aft - t_bef

    display_metrics("EM Train after ICA", Y_train, gmm_label_ica)
    display_metrics("EM Test after ICA", Y_test, gmm_label_ica_test)

    plt.figure()
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True,figsize=(10,6))
    ax1.set_title("EM before ICA")
    ax1.scatter(X_train[:,0],X_train[:,1],c=gmm_label_train, alpha=0.5)
    ax2.set_title("EM after ICA")
    ax2.scatter(ica_trans_train[:,0],ica_trans_train[:,1],c=gmm_label_ica, alpha=0.5)
    plt.savefig("UL1_EM_aft_ICA.png")

    # RP --------------------
    silhouette_score_rp = []
    for i in range(2, 12):
        gmm_rp = GaussianMixture(n_components=i, n_init=2, random_state=RANDOM_STATE).fit(rp_trans_train)
        gmm_predict_rp = gmm_rp.predict(rp_trans_train)
        silhouette_score_rp.append(metrics.silhouette_score(rp_trans_train, gmm_predict_rp, metric='euclidean'))
    plt.figure()
    plt.plot(range(1, 11), silhouette_score_rp)
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('silhouette_score')
    plt.savefig("UL1_SS_After_RP.png")

    t_bef = time.time()
    gmm_rp = GaussianMixture(n_components=2).fit(rp_trans_train)
    t_aft = time.time()
    em_train_time[3] = t_aft - t_bef

    gmm_label_rp = gmm_rp.predict(rp_trans_train)

    t_bef = time.time()
    gmm_label_rp_test = gmm_rp.predict(rp_trans_test)
    t_aft = time.time()
    em_predict_time[3] = t_aft - t_bef

    display_metrics("EM Train after RP", Y_train, gmm_label_rp)
    display_metrics("EM Test after RP", Y_test, gmm_label_rp_test)

    plt.figure()
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True,figsize=(10,6))
    ax1.set_title("EM before RP")
    ax1.scatter(X_train[:,0],X_train[:,1],c=gmm_label_train, alpha=0.5)
    ax2.set_title("EM after RP")
    ax2.scatter(rp_trans_train[:,0],rp_trans_train[:,1],c=gmm_label_rp, alpha=0.5)
    plt.savefig("UL1_EM_aft_RP.png")

    # LDA --------------------
    silhouette_score_lda = []
    for i in range(2, 12):
        gmm_lda = GaussianMixture(n_components=i, n_init=2, random_state=RANDOM_STATE).fit(lda_trans_train)
        gmm_predict_lda = gmm_lda.predict(lda_trans_train)
        silhouette_score_lda.append(metrics.silhouette_score(lda_trans_train, gmm_predict_lda, metric='euclidean'))
    plt.figure()
    plt.plot(range(1, 11), silhouette_score_lda)
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('silhouette_score')
    plt.savefig("UL1_SS_After_LDA.png")

    t_bef = time.time()
    gmm_lda = GaussianMixture(n_components=2).fit(lda_trans_train)
    t_aft = time.time()
    em_train_time[4] = t_aft - t_bef

    gmm_label_lda = gmm_lda.predict(lda_trans_train)

    t_bef = time.time()
    gmm_label_lda_test = gmm_lda.predict(lda_trans_test)
    t_aft = time.time()
    em_predict_time[4] = t_aft - t_bef

    display_metrics("EM Train after LDA", Y_train, gmm_label_lda)
    display_metrics("EM Test after LDA", Y_test, gmm_label_lda_test)

    plt.figure()
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True,figsize=(10,6))
    ax1.set_title("EM before LDA")
    ax1.scatter(X_train[:,0],X_train[:,1],c=gmm_label_train, alpha=0.5)
    ax2.set_title("EM after LDA")
    ax2.scatter(lda_trans_train[:,0],lda_trans_train[:,0],c=gmm_label_lda, alpha=0.5)
    plt.savefig("UL1_EM_aft_LDA.png")

    # Train time Different Dimensionality reduction algorithms kmeans
    classifier = ['EM', 'EM with PCA', 'EM with ICA', 'EM with RP', 'EM with LDA']
    np_classifier = np.array(classifier)
    plt.figure()
    plt.barh(np_classifier, em_train_time, align = 'center')
    plt.title('EM Train Time')
    plt.ylabel('Name')
    plt.xlabel('Time (seconds)')
    plt.savefig('UL1_EM_Traintime.png', bbox_inches = "tight")

    # Predict time Different Dimensionality reduction algorithms kmeans
    plt.figure()
    plt.barh(np_classifier, em_predict_time, align = 'center')
    plt.title('EM Query Time')
    plt.ylabel('Name')
    plt.xlabel('Time (seconds)')
    plt.savefig('UL1_EM_Querytime.png', bbox_inches = "tight")

# #4.  Neural Network with projected data ---------------------------------------------------
#     # Original run NN
#     querytime = neural_network("Original NN", X_train, Y_train, X_test, Y_test)
#     print("Original NN" , querytime)
#     # nn_predict_time = np.append(nn_predict_time, [querytime])
#     # NN with PCA
#     querytime = neural_network("NN with PCA", pca_trans_train, Y_train, pca_trans_test, Y_test)
#     print("NN with PCA" , querytime)
#     # nn_predict_time = np.append(nn_predict_time, [querytime])
#     # NN with ICA
#     querytime = neural_network("NN with ICA", ica_trans_train, Y_train, ica_trans_test, Y_test)
#     print("NN with ICA" , querytime)
#     # nn_predict_time = np.append(nn_predict_time, [querytime])
#     # NN with RP
#     querytime = neural_network("NN with RP", rp_trans_train, Y_train, rp_trans_test, Y_test)
#     print("NN with RP" , querytime)
#     # nn_predict_time = np.append(nn_predict_time, [querytime])
#     # NN with LDA
#     querytime = neural_network("NN with LDA", lda_trans_train, Y_train, lda_trans_test, Y_test)
#     print("NN with LDA" , querytime)
#     # nn_predict_time = np.append(nn_predict_time, [querytime])
#
# #5.  Neural Network with projected data and clustering -------------------------
#     pca_trans_train_NN = np.column_stack((pca_trans_train, kmeans_pca_label))
#     pca_trans_test_NN = np.column_stack((pca_trans_test, kmeans_pca_label_test))
#     ica_trans_train_NN = np.column_stack((ica_trans_train, kmeans_ica_label))
#     ica_trans_test_NN = np.column_stack((ica_trans_test, kmeans_ica_label_test))
#     rp_trans_train_NN = np.column_stack((rp_trans_train, kmeans_rp_label))
#     rp_trans_test_NN = np.column_stack((rp_trans_test, kmeans_rp_label_test))
#     lda_trans_train_NN = np.column_stack((lda_trans_train, kmeans_lda_label))
#     lda_trans_test_NN = np.column_stack((lda_trans_test, kmeans_lda_label_test))
#
#     # NN with PCA
#     querytime = neural_network("NN with PCA clustering", pca_trans_train_NN, Y_train, pca_trans_test_NN, Y_test)
#     print("NN with PCA clustering" , querytime)
#     # nn_predict_time = np.append(nn_predict_time, [querytime])
#     # NN with ICA
#     querytime = neural_network("NN with ICA clustering", ica_trans_train_NN, Y_train, ica_trans_test_NN, Y_test)
#     print("NN with ICA clustering" , querytime)
#     # nn_predict_time = np.append(nn_predict_time, [querytime])
#     # NN with RP
#     querytime = neural_network("NN with RP clustering", rp_trans_train_NN, Y_train, rp_trans_test_NN, Y_test)
#     print("NN with RP clustering" , querytime)
#     # nn_predict_time = np.append(nn_predict_time, [querytime])
#     # NN with LDA
#     querytime = neural_network("NN with LDA clustering", lda_trans_train_NN, Y_train, lda_trans_test_NN, Y_test)
#     print("NN with LDA clustering" , querytime)


if __name__ == "__main__":
    main()


    # # # FeatureAgglomeration -------------------------------------------------------------
    # # # Reference: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.FeatureAgglomeration.html#sklearn.cluster.FeatureAgglomeration
    # agglo = cluster.FeatureAgglomeration(n_clusters=2)
    # agglo.fit(X_train)
    # agglo_trans = agglo.transform(X_train)
    #
    # # Run on tranformed Agglo dataset
    # wcss_agglo = []
    # for i in range(1, 11):
    #     kmeans_agglo = KMeans(n_clusters=i, init='k-means++', random_state=RANDOM_STATE)
    #     kmeans_agglo.fit(agglo_trans)
    #     wcss_agglo.append(kmeans_agglo.inertia_)
    # plt.figure()
    # plt.plot(range(1, 11), wcss_agglo)
    # plt.title('Elbow Method')
    # plt.xlabel('Number of clusters')
    # plt.ylabel('WCSS')
    # plt.savefig("UL1_WCSS_After_Agglo.png")
    #
    # kmeans_agglo = KMeans(n_clusters=2, random_state=RANDOM_STATE).fit(agglo_trans)
    # kmeans_agglo_label = kmeans_agglo.labels_
    # centroids_agglo = kmeans_agglo.cluster_centers_
    #
    # plt.figure()
    # f, (ax1, ax2) = plt.subplots(1, 2, sharey=True,figsize=(10,6))
    # ax1.set_title("K means before Agglo")
    # ax1.scatter(X_train[:,0],X_train[:,10],c=kmeans_label_train, alpha=0.5)
    # ax1.scatter(centroids[:, 0], centroids[:, 10], c='red')
    # ax2.set_title("K Means after Agglo")
    # ax2.scatter(agglo_trans[:,0],agglo_trans[:,1],c=kmeans_agglo_label, alpha=0.5)
    # ax2.scatter(centroids_agglo[:, 0], centroids_agglo[:, 1], c='red')
    # plt.savefig("UL1_kmeans_aft_Agglo.png")
