import cv2
import numpy as np
import matplotlib.pyplot as plt


triangle = np.array([[2, 4], [1, 1], [3, 1], [2, 4]])
rectangle = np.array([[1, 3], [1, 1], [4, 1], [4, 3], [1, 3]])


def rotate_cv2(figure, angle_degree):
    center = np.mean(figure, axis=0)
    rotation_matrix = cv2.getRotationMatrix2D(tuple(center), angle_degree, 1)
    figure_homogeneous = np.hstack([figure, np.ones((figure.shape[0], 1))])
    rotated_figure_standard = np.dot(figure_homogeneous, rotation_matrix.T)
    rotated_figure = rotated_figure_standard[:, :2]
    return rotated_figure


def scale_cv2(figure, coeficcient):
    scale_matrix = np.array([[coeficcient, 0], [0, coeficcient]], dtype=np.float32)
    scaled_figure = cv2.transform(figure.reshape(1, -1, 2), scale_matrix).reshape(-1, 2)
    return scaled_figure


def reflect_cv2(figure, axis):
    if axis == 'x':
        reflect_matrix = np.array([[1, 0], [0, -1]], dtype=np.float32)
    elif axis == 'y':
        reflect_matrix = np.array([[-1, 0], [0, 1]], dtype=np.float32)
    reflected_figure = cv2.transform(figure.reshape(1, -1, 2), reflect_matrix).reshape(-1, 2)
    return reflected_figure


def shear_cv2(figure, axis, factor):
    if axis == 'x':
        shear_matrix = np.array([[1, factor, 0], [0, 1, 0]], dtype=np.float32)
    elif axis == 'y':
        shear_matrix = np.array([[1, 0, 0], [factor, 1, 0]], dtype=np.float32)
    figure_reshaped = figure.reshape(1, -1, 2).astype(np.float32)
    sheared_figure = cv2.transform(figure_reshaped, shear_matrix).reshape(-1, 2)
    return sheared_figure


def custom_transform_cv2(figure, transform_matrix):
    transformed_figure = cv2.gemm(figure.astype(np.float32), transform_matrix.astype(np.float32), 1, None, 0)
    return transformed_figure


image = cv2.imread('image.jpg')

def rotate_image(image, angle):
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
    return rotated_image

def scale_image(image, scale_factor):
    scaled_image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
    return scaled_image

def reflect_image(image, axis): # in cv2 # axis = 1 for x, 0 for y, -1 for xy
    reflected_image = cv2.flip(image, axis)
    return reflected_image


def shear_image(image, axis, factor):
    rows, cols = image.shape[:2]
    if axis == 'x':
        shear_matrix = np.array([[1, factor, 0], [0, 1, 0]], dtype=np.float32)
    elif axis == 'y':
        shear_matrix = np.array([[1, 0, 0], [factor, 1, 0]], dtype=np.float32)
    sheared_image = cv2.warpAffine(image, shear_matrix, (cols, rows))
    return sheared_image

def custom_tranform_image(image, transformation_matrix):
    rows, cols = image.shape[:2]
    transformed_image = cv2.warpAffine(image, transformation_matrix, (cols, rows))
    return transformed_image

rotated_image = rotate_image(image, 45)
mirrored_image = reflect_image(image, 1)
scaled_image = scale_image(image, 2)

plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.title('Original Image')
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

plt.subplot(2, 2, 2)
plt.title('Rotated Image')
plt.imshow(cv2.cvtColor(rotated_image, cv2.COLOR_BGR2RGB))

plt.subplot(2, 2, 3)
plt.title('Mirrored Image')
plt.imshow(cv2.cvtColor(mirrored_image, cv2.COLOR_BGR2RGB))

plt.subplot(2, 2, 4)
plt.title('Scaled Image')
plt.imshow(cv2.cvtColor(scaled_image, cv2.COLOR_BGR2RGB))

plt.show()