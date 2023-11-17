import numpy as np
import cv2
from cv2.gapi import BGR2RGB
from matplotlib import pyplot as plt
import math
from a2_utils import *
from a3_utils import *
from UZ_utils import *
import os


def subplot_template(images, titles, height, width):
    plt.figure(figsize=(12, 8))
    for i in range(width * height):
        plt.subplot(height, width, i + 1)
        plt.imshow(images[i], cmap='gray')
        plt.title(titles[i])

    plt.tight_layout()
    plt.show()


def gauss(sigma):
    kernel_size = 2 * math.ceil(3 * sigma) + 1
    kernel = np.zeros(kernel_size)
    for i in range(kernel_size):
        x = i - kernel_size // 2
        kernel[i] = (1 / (sigma * math.sqrt(2 * math.pi))) * math.exp(((x / sigma) ** 2) / (-2))
    kernel = kernel / np.sum(kernel)
    kernel = kernel.reshape(1, -1)  # just so it has 1xn dimensions
    return kernel


def gaussdx(sigma):
    kernel_size = 2 * math.ceil(3 * sigma) + 1
    kernel = np.zeros(kernel_size)
    for i in range(kernel_size):
        x = i - kernel_size // 2
        kernel[i] = (1 / ((sigma ** 3) * math.sqrt(2 * math.pi))) * (-x) * math.exp(((x / sigma) ** 2) / (-2))
    kernel = kernel.reshape(1, -1)
    return kernel / np.abs(kernel).sum()


"""def gaussfilter(image, sigma):
    g_x = gauss(sigma)
    g_y = g_x.T
    image_copy = cv2.filter2D(image, -1, g_x)
    image_copy = cv2.filter2D(image_copy, -1, g_y)

    return image_copy


def gaussfilter_derivative(image, sigma):
    g_x = gaussdx(sigma)
    g_y = g_x.T
    image_copy = cv2.filter2D(image, -1, g_x)
    image_copy = cv2.filter2D(image_copy, -1, g_y)

    return image_copy


def derivative_wrt_x(image, sigma):
    g_x = gaussdx(sigma)
    g_y = gauss(sigma).T
    image_copy = cv2.filter2D(image, -1, g_x)
    image_copy = cv2.filter2D(image_copy, -1, g_y)

    return image_copy


def derivative_wrt_y(image, sigma):
    g_x = gauss(sigma)
    g_y = gaussdx(sigma).T
    image_copy = cv2.filter2D(image, -1, g_x)
    image_copy = cv2.filter2D(image_copy, -1, g_y)

    return image_copy"""


def partial_derivatives(original_image, sigma):
    g_x = gauss(sigma)
    g_y = g_x.T
    g_xx = gaussdx(sigma)
    g_yy = g_xx.T
    # I_x = gxx * (g_y * I)
    I_x = cv2.filter2D(cv2.filter2D(original_image, -1, -g_y), -1, g_xx)
    # I_y = gyy * (g_x * I)
    I_y = cv2.filter2D(cv2.filter2D(original_image, -1, g_x), -1, -g_yy)
    # I_xx = gxx * (g_y * I_x)
    I_xx = cv2.filter2D(cv2.filter2D(I_x, -1, -g_y), -1, g_xx)
    # I_xy = gyy * (g_x * I_x)
    I_xy = cv2.filter2D(cv2.filter2D(I_x, -1, g_x), -1, -g_yy)
    # I_yy = gyy * (g_x * I_y)
    I_yy = cv2.filter2D(cv2.filter2D(I_y, -1, g_x), -1, -g_yy)

    return I_x, I_y, I_xx, I_xy, I_yy


def magnitude_and_angles(original_image, sigma):
    I_x, I_y, _, _, _ = partial_derivatives(original_image, sigma)
    I_mag = np.sqrt(I_x ** 2 + I_y ** 2)
    I_dir = np.arctan2(I_y, I_x)

    return I_mag, I_dir


"""def divide_image(original_image, m, n):
    height, width = original_image.shape

    number_of_rows = height // m
    number_of_columns = width // n

    image_grids = [original_image[i * 8:(i + 1) * 8, j * 8:(j + 1) * 8] for i in range(number_of_rows) for j in
                   range(number_of_columns)]

    return image_grids


def hist_of_gradient_magnitudes(I_mag, I_dir, number_of_bins):
    H = np.zeros(number_of_bins)
    magnitude_vector = I_mag.reshape(-1, order='C')
    length = len(magnitude_vector)
    angle_vector = I_dir.reshape(-1, order='C')
    min_value = min(angle_vector)
    max_value = max(angle_vector)
    unit = (max_value - min_value) / number_of_bins
    for i in range(length):
        x = (angle_vector[i] - min_value) / unit
        if x == number_of_bins:
            x = x - 1
        H[int(x)] += magnitude_vector[i]

    return H


def myhist3(image, number_of_bins):
    H = np.zeros((number_of_bins, number_of_bins, number_of_bins))
    im_red = image[:, :, 0]
    im_green = image[:, :, 1]
    im_blue = image[:, :, 2]
    red_vector = im_red.reshape(-1, order='C')
    green_vector = im_green.reshape(-1, order='C')
    blue_vector = im_blue.reshape(-1, order='C')
    length = len(red_vector)
    unit = 1 / number_of_bins

    for i in range(length):
        x = red_vector[i] / unit
        y = green_vector[i] / unit
        z = blue_vector[i] / unit

        # edge cases
        if x == number_of_bins:
            x = x - 1
        if y == number_of_bins:
            y = y - 1
        if z == number_of_bins:
            z = z - 1

        H[int(x), int(y), int(z)] += 1

    return H / H.sum()


def directory_histograms(path, number_of_bins):
    # slika, 3d histogram, 1d histogram
    image_list = os.listdir(path)
    n = len(image_list)
    image_data = []
    histograms = np.empty(n, dtype=object)
    histograms_1d = np.empty(n, dtype=object)

    for i in range(n):
        full_path = os.path.join(path, image_list[i])
        img = imread(full_path)
        histograms[i] = myhist3(img, number_of_bins)
        histograms_1d[i] = histograms[i].reshape(-1)
        image_data.append((img, histograms[i], histograms_1d[i]))

    return image_data


def compare_histograms(h1, h2, measure):
    distance = np.inf
    h1_1d = h1.reshape(-1)
    h2_1d = h2.reshape(-1)
    if measure == 'L2':
        i = (h1_1d - h2_1d) ** 2
        distance = i.sum()
        distance = math.sqrt(distance)

    elif measure == 'chi':
        epsilon = 1e-10
        i = ((h1_1d - h2_1d) ** 2) / (h1_1d + h2_1d + epsilon)
        distance = i.sum()
        distance = distance / 2

    elif measure == 'I':
        i = np.minimum(h1_1d, h2_1d)
        distance = i.sum()
        distance = 1 - distance

    elif measure == 'H':
        i = (np.sqrt(h1_1d) - np.sqrt(h2_1d)) ** 2
        distance = i.sum()
        distance = distance / 2
        distance = math.sqrt(distance)

    else:
        print("invalid distance measure")

    return distance


def similar_images(im_data, h1, mesure):
    distances = []
    n = len(im_data)
    for i in range(n):
        h2 = im_data[i][1]
        distance = compare_histograms(h1, h2, mesure)
        distances.append((im_data[i][0], im_data[i][2], distance))

    # sorted_distances = sorted(distances, key=lambda x: x[2])

    return distances


def similar_images_new(im_data, g1, mesure):
    distances = []
    n = len(im_data)
    for i in range(n):
        g2 = im_data[i][1]
        distance = compare_histograms(g1, g2, mesure)
        distances.append((im_data[i][0], im_data[i][1], distance))

    # sorted_distances = sorted(distances, key=lambda x: x[2])

    return distances


def gradient_feature(original_image, number_of_bins):
    I_mag, I_dir = magnitude_and_angles(original_image)
    I_mag_grids = divide_image(I_mag, 8, 8)
    I_dir_grids = divide_image(I_dir, 8, 8)
    H_total = []
    for i in range(8):
        for j in range(8):
            H_ij = hist_of_gradient_magnitudes(I_mag_grids[i][j], I_dir_grids[i][j], number_of_bins)
            H_total.append(H_ij)

    return H_total


def show_sorted_distances(sorted_distances, mesure):
    plt.figure(figsize=(20, 5))
    plt.subplot(2, 6, 1)
    plt.imshow(sorted_distances[0][0])
    plt.title("original image")
    plt.subplot(2, 6, 2)
    plt.imshow(sorted_distances[1][0])
    plt.title("1)")
    plt.subplot(2, 6, 3)
    plt.imshow(sorted_distances[2][0])
    plt.title("2)")
    plt.subplot(2, 6, 4)
    plt.imshow(sorted_distances[3][0])
    plt.title("3)")
    plt.subplot(2, 6, 5)
    plt.imshow(sorted_distances[4][0])
    plt.title("4)")
    plt.subplot(2, 6, 6)
    plt.imshow(sorted_distances[5][0])
    plt.title("5)")
    plt.subplot(2, 6, 7)
    x = np.arange(len(sorted_distances[0][1]))
    plt.bar(x, sorted_distances[0][1])
    plt.title("%s = %.2f" % (mesure, sorted_distances[0][2]))
    plt.subplot(2, 6, 8)
    x = np.arange(len(sorted_distances[1][1]))
    plt.bar(x, sorted_distances[1][1])
    plt.title("%s = %.2f" % (mesure, sorted_distances[1][2]))
    plt.subplot(2, 6, 9)
    x = np.arange(len(sorted_distances[2][1]))
    plt.bar(x, sorted_distances[2][1])
    plt.title("%s = %.2f" % (mesure, sorted_distances[2][2]))
    plt.subplot(2, 6, 10)
    x = np.arange(len(sorted_distances[3][1]))
    plt.bar(x, sorted_distances[3][1])
    plt.title("%s = %.2f" % (mesure, sorted_distances[3][2]))
    plt.subplot(2, 6, 11)
    x = np.arange(len(sorted_distances[4][1]))
    plt.bar(x, sorted_distances[4][1])
    plt.title("%s = %.2f" % (mesure, sorted_distances[4][2]))
    plt.subplot(2, 6, 12)
    x = np.arange(len(sorted_distances[5][1]))
    plt.bar(x, sorted_distances[5][1])
    plt.title("%s = %.2f" % (mesure, sorted_distances[5][2]))
    plt.tight_layout()
    plt.show()


def prev_assignment():
    im_data = directory_histograms('dataset', 8)
    slike = [data[0] for data in im_data]
    n = len(im_data)
    image = im_data[19][0]
    h1 = im_data[19][1]
    distances_l2 = similar_images(im_data, h1, 'L2')
    sorted_distances_l2 = sorted(distances_l2, key=lambda x: x[2])
    show_sorted_distances(sorted_distances_l2, 'L2')


def directory_histograms_new(path, number_of_bins):
    # siva slika, gradient feature
    image_list = os.listdir(path)
    n = len(image_list)
    image_data = []
    gradient_features = np.empty(n, dtype=object)

    for i in range(n):
        full_path = os.path.join(path, image_list[i])
        img = imread_gray(full_path)
        gradient_features[i] = gradient_feature(img, 8)
        image_data.append((img, gradient_features[i]))

    return image_data


def new_compare():
    im_data = directory_histograms_new('dataset', 8)
    slike = [data[0] for data in im_data]
    n = len(im_data)
    image = im_data[19][0]
    g1 = im_data[19][1]
    distances_l2 = similar_images_new(im_data, g1, 'L2')
    sorted_distances_l2 = sorted(distances_l2, key=lambda x: x[2])
    show_sorted_distances(sorted_distances_l2, 'L2')

"""


def findedges(image, sigma, theta):
    I_mag, _ = magnitude_and_angles(image, sigma)
    I_e = np.where(I_mag > theta, 1, 0)
    return I_e


def non_maxima(image, sigma):
    I_mag, I_dir = magnitude_and_angles(image, sigma)
    res = I_mag.copy()
    height, width = I_mag.shape
    for pixel_x in range(height):
        for pixel_y in range(width):
            magnitude = I_mag[pixel_x, pixel_y]
            # n1 and n2 are neighboring pixels parallel to the gradient direction
            n1_x = pixel_x + round(np.cos(I_dir[pixel_x, pixel_y]))
            n1_y = pixel_y + round(np.sin(I_dir[pixel_x, pixel_y]))
            n2_x = pixel_x - round(np.cos(I_dir[pixel_x, pixel_y]))
            n2_y = pixel_y - round(np.sin(I_dir[pixel_x, pixel_y]))
            # 8 neighbors
            # eight_neighbors_x = [pixel_x - 1, pixel_x - 1, pixel_x - 1, pixel_x, pixel_x, pixel_x + 1, pixel_x + 1, pixel_x + 1]
            # eight_neighbors_y = [pixel_y - 1, pixel_y, pixel_y + 1, pixel_y - 1, pixel_y + 1, pixel_y - 1, pixel_y, pixel_y + 1]
            if 0 <= n1_x < height and 0 <= n1_y < width and 0 <= n2_x < height and 0 <= n2_y < width:
                if I_mag[pixel_x, pixel_y] < I_mag[n1_x, n1_y] or I_mag[pixel_x, pixel_y] < I_mag[n2_x, n2_y]:
                    res[pixel_x, pixel_y] = 0
            else:
                pass
    return res


def exercise1():
    print("Exercise 1: Image derivatives\na) Follow the equations above and derive the equations used to compute first "
          "and second derivatives with respect to y: I_y(x, y), I_yy(x, y), as well as the mixed derivative I_xy(x, y)")
    print("\nb) Implementing a function that computes the derivative of a 1-D Gaussian kernel: ")
    print("\nc) Analyzing the filter by using an impulse response function: ")

    impulse = np.zeros((50, 50))
    impulse[25, 25] = 1

    sigma = 3
    G = gauss(sigma)  # Gaussian kernel
    G_T = G.T
    D = gaussdx(sigma)  # Gaussian derivative kernel
    D_T = D.T

    order = [(G, -D_T), (-D, G_T), (G, G_T), (G_T, -D), (-D_T, G)]
    titles = ["Impulse", "G, Dt", "D, Gt", "G, Gt", "Gt, D", "Dt, G"]
    images = [impulse]

    for i in range(len(order)):
        img = cv2.filter2D(cv2.filter2D(impulse, -1, order[i][0]), -1, order[i][1])
        images.append(img)

    subplot_template(images, titles, 2, 3)

    print("Is the order of operations important?\nNo, we can see that the convolutions are commutative and the order "
          "is not important, the results are the same.")

    print("\nd) Implementing a function that uses functions gauss and gaussdx to compute both partial derivatives of "
          "a given image with respect to x and with respect to y.")
    original_image = imread_gray('images/museum.jpg')
    sigma = 1
    I_x, I_y, I_xx, I_xy, I_yy = partial_derivatives(original_image, sigma)
    I_mag, I_dir = magnitude_and_angles(original_image, sigma)

    images = [original_image, I_x, I_y, I_mag, I_xx, I_xy, I_yy, I_dir]
    titles = ["original_image", "I_x", "I_y", "I_mag", "I_xx", "I_xy", "I_yy", "I_dir"]
    subplot_template(images, titles, 2, 4)

    """for i in range(len(images)):
        imshow(images[i], i)"""

    """print("\ne) ")

    prev_assignment()
    new_compare()

    # original_image = imread_gray('images/museum.jpg')
    # print(len(gradient_feature(original_image, 8)))
"""


def exercise2():
    print("\nExercise 2: Edges in images\na) Creating a function 'findedges': ")
    image = imread_gray('images/museum.jpg')
    sigma = 0.75
    thetas = [0.1, 0.15, 0.3, 0.45, 0.65, 0.8]
    I_e_s = []
    titles = []
    for i in range(len(thetas)):
        I_e_s.append(findedges(image, sigma, thetas[i]))
        titles.append("theta = %.2f" % thetas[i])
    subplot_template(I_e_s, titles, 2, 3)
    I_e = findedges(image, 0.75, 0.098)     # imo the best results, gotten with some experimenting

    '''print("\nb) Implementing non-maxima suppression: ")

    image_nm = non_maxima(image, 0.1)
    imshow(image_nm, "non maxima")

    print("\nc) Adding the final step after performing non-maxima suppression along edges: ")
    imshow(image, "original")
    imshow(np.where(I_e > 0.16, 1, 0), "threshold 0.16")
    imshow(image_nm, "non maxima")'''


def exercise3():
    print("\nExercise 3: ")


if __name__ == '__main__':
    # exercise1()
    exercise2()
    # exercise3()
