import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
from os.path import join, exists
from os import makedirs
from sklearn.metrics import roc_curve, auc
from numpy import interp
from scipy.optimize import brentq
from itertools import cycle
from scipy.interpolate import interp1d



def plot_values(x, y):
    """
        Simple plot of points given their coordinates.
        :param x: x coordinates
        :param y: y coordinates
    """

    plt.scatter(x, y)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Simple plot x/y")
    plt.show()


def save_plot_values(x, y, save_dir, range=False):
    """
    Simple plot of points given their coordinates.
    :param x: x coordinates
    :param y: y coordinates
    :param save_dir: directory in which save the plot
    :param range: if True, plot x and y in range [-1.5, 1.5, -1.5, 1]; Default = False
    """
    plt.scatter(x, y)
    if range:
        plt.axis([-1.5, 1.5, -1.5, 1])
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Simple plot x/y")
    plt.savefig(save_dir)
    plt.close()


def plot_values_dictionary(coordinates):
    """
    Simple plot of points given their coordinates.
    :param coordinates: dictionary with two keys:
                        'x' (lower case) -> x coordinates
                        'y' (lower case) -> y coordinates
    """
    plt.scatter(coordinates['x'], coordinates['y'])
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Simple plot x/y")
    plt.show()


def plot_dens_dur_matrix(values, bins_endpoints, step, threshold, save_image, save_image_dir=""):
    """
    Plotter for matrices over a grid in which a circle with a radius proportional to the value inside a cell is drawn
    on the correspondent grid cell's center.
    :param values: matrices of values to be plot in the center of a grid's cells
    :param bins_endpoints: array with the x-y bins endpoints; [x_max, x_min, y_max, y_min]
    :param step: dimension of a grid's cell (cells are squares)
    :param threshold: threshold over which the circle is drawn with a black border
    :param save_image: True if the plot has to be saved; False otherwise
    :param save_image_dir: directory in which to save the plot
    """
    circle_radius = 0.2

    bin_x = np.arange(bins_endpoints[1] + (step / 2), bins_endpoints[0] + step, step)
    bin_y = np.arange(bins_endpoints[3] + (step / 2), bins_endpoints[2] + step, step)

    rows = len(bin_y) - 1
    columns = len(bin_x) - 1

    fig, ax = plt.subplots()

    plt.axis([bins_endpoints[1], bins_endpoints[0], bins_endpoints[3], bins_endpoints[2]])
    plt.grid(True)

    for i in range(0, rows):
        for j in range(0, columns):
            if values[i, j] > threshold:
                circle = Circle((bin_x[j], bin_y[rows - 1 - i]), circle_radius, edgecolor='Black')
            else:
                circle = Circle((bin_x[j], bin_y[rows - 1 - i]), (circle_radius * (values[i, j]/threshold)))

            ax.add_patch(circle)

    if save_image:
        plt.savefig(save_image_dir)
        plt.close()
    else:
        plt.show()


def plot_arc_matrix(bins_endpoints, step, arc_matrix, n_bin_x, threshold, save_image, save_image_dir=""):
    """
    Plotter for a matrix over a grid in which a line with width proportional to the value inside a cell is drawn
    between two grid's cells indicated by the row and column indices of the matrix cell.
    :param bins_endpoints: array with the x-y bins endpoints; [x_max, x_min, y_max, y_min]
    :param step: dimension of a grid's cell (cells are squares)
    :param arc_matrix: matrices of values to be plot in the center of a grid's cells
    :param n_bin_x: number of columns of the matrix from which the matrix to be plot has been created. Used to convert
                    the row and column indices of the given matrix into the indices of the originating matrix
    :param threshold: threshold over which the circle is drawn with a black border
    :param save_image: True if the plot has to be saved; False otherwise
    :param save_image_dir: directory in which to save the plot
    """
    line_width = 1

    bin_x = np.arange(bins_endpoints[1] + (step / 2), bins_endpoints[0] + step, step)
    bin_y = np.arange(bins_endpoints[3] + (step / 2), bins_endpoints[2] + step, step)

    rows = len(arc_matrix[0])

    plt.axis([bins_endpoints[1], bins_endpoints[0], bins_endpoints[3], bins_endpoints[2]])
    plt.grid(True)

    for i in range(0, rows):
        for j in range(0, i):
            if i != j and arc_matrix[i, j] + arc_matrix[j, i] > 0:
                arc_weight = arc_matrix[i, j] + arc_matrix[j, i]

                # (i-1)*n + (j-1)
                cell_1_raw = ((i // (n_bin_x - 1)) + 1) * 10 + (i % (n_bin_x - 1)) + 1
                cell_2_raw = ((j // (n_bin_x - 1)) + 1) * 10 + (j % (n_bin_x - 1)) + 1

                cell_1_x = (cell_1_raw // 10) - 1
                cell_1_y = (cell_1_raw % 10) - 1

                cell_2_x = (cell_2_raw // 10) - 1
                cell_2_y = (cell_2_raw % 10) - 1

                plt.plot([bin_x[cell_1_x], bin_x[cell_2_x]], [bin_y[cell_1_y], bin_y[cell_2_y]],
                         color='Blue',
                         linestyle='-',
                         linewidth=(line_width * (arc_weight/threshold)))

    if save_image:
        plt.savefig(save_image_dir)
        plt.close()
    else:
        plt.show()


def plot_cms_curve(ranks, cms_values, save_path, filename, show=False, save=False):
    """
    Plot the given cms curve and save it in the save_path directory with the given filename.
    :param ranks: array of increasing ranks (from 1 to n_observers_in_gallery)
    :param cms_values: array with all the cumulative match scores
    :param save_path: directory where save the cms plot
    :param filename: string specifying the name of the file to save
    """
    plt.figure()

    plt.plot(ranks, cms_values,
             label=('CMS(1) = {0:0.4f}'.format(cms_values[0][0])),
             color='navy', linestyle='-', linewidth=4)
    plt.scatter(ranks, cms_values, color='darkblue')

    plt.xlim([-0.1, ranks[len(ranks)-1]+0.1])
    plt.ylim([0.75, 1.03])
    plt.xlabel('Rank')
    plt.ylabel('Cumulative Match Score')
    plt.title('CMS Curve ' + filename)
    plt.legend(loc="lower right")

    if save:
        filename = filename + "_cms"
        if not exists(save_path):
            makedirs(save_path)
        plt.savefig(join(save_path, filename))
        plt.savefig(join(save_path, filename))

    if show:
        plt.show()

    plt.close()

def build_cms_curve(y_test, y_score, n_classes, save_path, filename, show=False):

    # order the scores and labels vectors
    cms_y_test = np.transpose(y_test)
    # print("CMS y test shape ", np.shape(cms_y_test))
    cms_y_scores = np.transpose(y_score)
    # print("CMS y scores shape ", np.shape(cms_y_scores))
    cms_y_scores_ordered = np.zeros((n_classes, 1))

    for i in range(np.shape(cms_y_test)[1]):
        s_y_test = np.reshape(cms_y_test[:, i], (n_classes, 1))
        s_y_scores = np.reshape(cms_y_scores[:, i], (n_classes, 1))

        to_order = np.concatenate((s_y_test, s_y_scores), 1)
        # print("To order \n", to_order)

        a = to_order[to_order[:, 0].argsort()]  # First sort doesn't need to be stable.

        dist_vector = a[a[:, 1].argsort(kind='mergesort')]
        dist_vector = np.flipud(dist_vector)
        # print("Ordered \n", dist_vector)

        cms_y_scores_ordered = np.concatenate((cms_y_scores_ordered, np.reshape(dist_vector[:, 0], (n_classes, 1))), 1)

    cms_y_scores_ordered = cms_y_scores_ordered[:, 1:]
    # print("cms y score ", cms_y_scores_ordered)
    # print("cms y score shape ", np.shape(cms_y_scores_ordered))

    ranks = np.arange(0, np.shape(cms_y_scores_ordered)[0], 1)
    cms_values = np.zeros((np.shape(cms_y_scores_ordered)[0], 1))
    cumulative_value = 0

    for r in ranks:
        cumulative_value += np.count_nonzero(cms_y_scores_ordered[r, :])
        cms_values[r] = cumulative_value / np.shape(cms_y_scores_ordered)[1]

    plot_cms_curve(ranks, cms_values, save_path, filename, show)

def plot_roc_values(fpr, tpr, roc_auc, eer, save_path, filename, save=False):
    """
    Plot and save the ROC curve for a multiclass problem given the FalsePositiveRates and the TruePositiveRates.
    :param fpr: matrix (n_observers_in_gallery, n_observers_in_probe) which columns are the fpr of single classes,
                aka single probe observers
    :param tpr: matrix (n_observers_in_gallery, n_observers_in_probe) which columns are the tpr of single classes,
                aka single probe observers
    :param roc_auc: AUC value calculated from the ROC curve
    :param eer: EER calculated from the ROC curve
    :param save_path: directory where save the roc plot
    :param filename: string specifying the name of the file to save
    """
    lw = 2

    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label=('micro-avg | AUC = {0:0.4f}'.format(roc_auc["micro"]) + ' | EER = {0:0.4f})'.format(eer)),
             color='deeppink', linestyle='-', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label=('macro-avg | AUC = {0:0.4f}'.format(roc_auc["macro"]) + ' | EER = {0:0.4f})'.format(eer)),
             color='navy', linestyle='-', linewidth=4)

    # colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    # for i, color in zip(range(n_classes), colors):
    #     plt.plot(fpr[i], tpr[i], color=color, lw=lw,
    #              label='ROC curve of class {0} (area = {1:0.2f})'
    #                    ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve ' + filename)
    plt.legend(loc="lower right")

    if save:
        filename = filename + "_roc"
        if not exists(save_path):
            makedirs(save_path)
        plt.savefig(join(save_path, filename))
        plt.savefig(join(save_path, filename))

    plt.show()

def build_roc_curve(y_test, y_score, n_classes, save_path, filename, show=False):
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Compute macro-average ROC curve and ROC area
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    eer = brentq(lambda x: 1. - x - interp1d(fpr["micro"], tpr["micro"])(x), 0., 1.)

    lw = 2

    if save_path is not None:
        plt.figure()
        plt.plot(fpr["micro"], tpr["micro"],
                 label='micro-average ROC curve (AUC = {0:0.2f} | '
                       ''.format(roc_auc["micro"]) + ' EER = {0:0.4f})'.format(eer),
                 color='navy', linestyle=':', linewidth=4)

        '''plt.plot(fpr["macro"], tpr["macro"],
                 label='macro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc["macro"]),
                 color='navy', linestyle=':', linewidth=4)'''

        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlim([-0.05, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve ' + filename)
        plt.legend(loc="lower right")

        filename = filename + "_roc"
        if not exists(save_path):
            makedirs(save_path)
        plt.savefig(join(save_path, filename))
        if show==True:
            plt.show()
        plt.close()

    return roc_auc['micro'], eer, fpr, tpr

    def plot_roc_curve(ax, fpr, tpr, color="navy"):
        ax.plot(fpr["micro"], tpr["micro"],
                 label='micro-average ROC curve (AUC = {0:0.2f} | '
                       ''.format(roc_auc["micro"]) + ' EER = {0:0.4f})'.format(eer),
                 color=color, linestyle=':', linewidth=4)

        ax.plot([0, 1], [0, 1], 'k--', lw=lw)
        ax.xlim([-0.05, 1.0])
        ax.ylim([0.0, 1.05])
        ax.xlabel('False Positive Rate')
        ax.ylabel('True Positive Rate')
        ax.title('ROC Curve ' + filename)
        ax.legend(loc="lower right")

        '''filename = filename + "_roc"
        if not exists(save_path):
            makedirs(save_path)
        plt.savefig(join(save_path, filename))'''
        
        if show==True:
            ax.show()
        plt.close()