import numpy as np
import matplotlib.pyplot as plt

def rotate(figure, angle_degree):
    angle_rad = np.radians(angle_degree)

    rotation_matrix = np.array([[np.cos(angle_rad), -np.sin(angle_rad)], [np.sin(angle_rad), np.cos(angle_rad)]])
    rotated_points = np.dot(figure, rotation_matrix)

    return rotated_points, rotation_matrix

def scale(figure, coefficient):
    scaled_points = figure*coefficient
    return scaled_points

def reflect(figure, axis):
    if axis == "x":
       reflect_points = np.array([[1, 0], [0, -1]])
    elif axis == "y":
        reflect_points = np.array([[-1, 0], [0, 1]])
    elif axis == "origin":
       reflect_points = np.array([[0, 0]]) - figure
    else:
        print("This type of reflextion is not implemented")

    reflected_points = np.dot(figure, reflect_points)
    return reflected_points

def nahyl_axis(figure, axis, factor):
    if axis == 'x':
        new_matrix = np.array([[1, factor], [0, 1]])
    elif axis == 'y':
        new_matrix = np.array([[1, 0], [factor, 1]])
    else:
        print("Axis must be 'x' or 'y'")

    new_points = np.dot(figure, new_matrix)
    return new_points

def custom_transform(figure, transformation_matrix):
    transformed_points = np.dot(figure, transformation_matrix)
    return transformed_points

batman = np.array([[0, 0], [1, 0.2], [0.4, 1], [0.5, 0.4], [0, 0.8], [-0.5, 0.4], [-0.4, 1], [-1, 0.2], [0, 0]])
triangle = np.array([[2, 4], [1, 1], [3, 1], [2, 4]])
rectangle = np.array([[1, 3], [1, 1], [4, 1], [4, 3], [1, 3]])

random_transformation_matrix = np.array([[0.65, -0.6], [0, 1]])

rotated_batman, rotate_matrix = rotate(batman, 45)
scaled_triangle = scale(triangle, 12)
nahyl_batman = nahyl_axis(batman, "x", 3)
print("Rotation matrix:\n", rotate_matrix)
transformed_batman = custom_transform(batman, random_transformation_matrix)
x = transformed_batman[:, 0]
y = transformed_batman[:, 1]

plt.figure(figsize=(8, 6))
plt.plot(x, y, marker='o', linestyle='-')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)
plt.axis('equal')
plt.show()




