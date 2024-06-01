import numpy as np
import matplotlib.pyplot as plt


def rotate(figure, angle_degree):
    angle_rad = np.radians(angle_degree)

    rotation_matrix = np.array([[np.cos(angle_rad), -np.sin(angle_rad)], [np.sin(angle_rad), np.cos(angle_rad)]])
    rotated_points = np.dot(figure, rotation_matrix)

    return rotated_points, rotation_matrix

def scale(figure, coefficient):
    scale_matrix = np.array([[coefficient, 0], [0, coefficient]])
    scaled_points = np.dot(figure, scale_matrix)
    return scaled_points, scale_matrix

def reflect(figure, axis):
    if axis == "x":
        reflect_matrix = np.array([[1, 0], [0, -1]])
    elif axis == "y":
        reflect_matrix = np.array([[-1, 0], [0, 1]])
    elif axis == "origin":
        reflect_matrix = -np.eye(2)
    else:
        print("This type of reflection is not implemented")
        return figure, None
    reflected_points = np.dot(figure, reflect_matrix)
    return reflected_points, reflect_matrix

def nahyl_axis(figure, axis, factor):
    if axis == 'x':
        shear_matrix = np.array([[1, factor], [0, 1]])
    elif axis == 'y':
        shear_matrix = np.array([[1, 0], [factor, 1]])
    else:
        print("Axis must be 'x' or 'y'")
        return figure, None
    sheared_points = np.dot(figure, shear_matrix)
    return sheared_points, shear_matrix

def custom_transform(figure, transformation_matrix):
    transformed_points = np.dot(figure, transformation_matrix)
    return transformed_points, transformation_matrix

triangle = np.array([[2, 4], [1, 1], [3, 1], [2, 4]])
rectangle = np.array([[1, 3], [1, 1], [4, 1], [4, 3], [1, 3]])

random_transformation_matrix = np.array([[0.65, -0.6], [0, 1]])
scaled_triangle = scale(triangle, 12)
nahyl_rectangle = nahyl_axis(rectangle, "x", 3)
x = scaled_triangle[:, 0]
y = nahyl_rectangle[:, 1]

plt.figure(figsize=(8, 6))
plt.plot(x, y, marker='o', linestyle='-')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)
plt.axis('equal')
plt.show()



# 3D figures
pyramid = np.array([
    [0, 0, 0],
    [1, 0, 0],
    [1, 1, 0],
    [0, 1, 0],
    [0.5, 0.5, 1]
])

random_transformation_matrix_3d = np.random.rand(3, 3)
transformed_pyramid = custom_transform(pyramid, random_transformation_matrix_3d)
fig = plt.figure(figsize=(12, 6))
pyramid_edges = [
    [0, 1], [1, 2], [2, 3], [3, 0],
    [0, 4], [1, 4], [2, 4], [3, 4]
]
ax1 = fig.add_subplot(121, projection='3d')
for edge in pyramid_edges:
    ax1.plot([pyramid[edge[0], 0], pyramid[edge[1], 0]],
             [pyramid[edge[0], 1], pyramid[edge[1], 1]],
             [pyramid[edge[0], 2], pyramid[edge[1], 2]],)
ax1.set_title("Original Pyramid")
ax1.set_xlabel("X")
ax1.set_ylabel("Y")
ax1.set_zlabel("Z")


ax2 = fig.add_subplot(122, projection='3d')
for edge in pyramid_edges:
    ax2.plot([transformed_pyramid[edge[0], 0], transformed_pyramid[edge[1], 0]],
             [transformed_pyramid[edge[0], 1], transformed_pyramid[edge[1], 1]],
             [transformed_pyramid[edge[0], 2], transformed_pyramid[edge[1], 2]],)
ax2.set_title("Transformed Pyramid")
ax2.set_xlabel("X")
ax2.set_ylabel("Y")
ax2.set_zlabel("Z")

plt.show()

