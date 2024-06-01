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

def reflect_3d(figure, axis):
    if axis == 'x':
        reflection_matrix = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])
    elif axis == 'y':
        reflection_matrix = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]])
    elif axis == 'z':
        reflection_matrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]])
    else:
        print("Invalid axis for reflection")
        return figure, None
    reflected_points = np.dot(figure, reflection_matrix)
    return reflected_points, reflection_matrix


figures = {
    "triangle": np.array([[2, 4], [1, 1], [3, 1], [2, 4]]),
    "rectangle": np.array([[1, 3], [1, 1], [4, 1], [4, 3], [1, 3]])
}


def plot_transformations(figure, figure_name):
    rotated_figure, rotate_matrix = rotate(figure, 45)
    scaled_figure, scale_matrix = scale(figure, 2)
    reflected_figure_x, reflect_matrix_x = reflect(figure, "x")
    sheared_figure, shear_matrix = nahyl_axis(figure, "x", 1)
    custom_matrix = np.array([[0.6, -1], [0, 7]])
    custom_transformed_figure, custom_matrix = custom_transform(figure, custom_matrix)

    print(f"Rotation matrix:\n{rotate_matrix}")
    print(f"Scale matrix:\n{scale_matrix}")
    print(f"Reflection matrix (x-axis):\n{reflect_matrix_x}")
    print(f"Shear matrix (x-axis):\n{shear_matrix}")
    print(f"Custom transformation matrix:\n{custom_matrix}")

    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    axs = axs.flatten()

    axs[0].plot(figure[:, 0], figure[:, 1], marker='o', linestyle='-', label="Original " + figure_name)
    axs[0].set_title("Original " + figure_name)
    axs[0].grid(True)
    axs[0].axis('equal')


    axs[1].plot(figure[:, 0], figure[:, 1], marker='o', linestyle='-', label="Original " + figure_name)
    axs[1].plot(rotated_figure[:, 0], rotated_figure[:, 1], marker='o', linestyle='-', label="Rotated " + figure_name)
    axs[1].set_title("Rotated " + figure_name)
    axs[1].grid(True)
    axs[1].axis('equal')


    axs[2].plot(figure[:, 0], figure[:, 1], marker='o', linestyle='-', label="Original " + figure_name)
    axs[2].plot(scaled_figure[:, 0], scaled_figure[:, 1], marker='o', linestyle='-', label="Scaled " + figure_name)
    axs[2].set_title("Scaled " + figure_name)
    axs[2].grid(True)
    axs[2].axis('equal')

    axs[3].plot(figure[:, 0], figure[:, 1], marker='o', linestyle='-', label="Original " + figure_name)
    axs[3].plot(reflected_figure_x[:, 0], reflected_figure_x[:, 1], marker='o', linestyle='-',
                label="Reflected x-axis " + figure_name)
    axs[3].set_title("Reflected x-axis " + figure_name)
    axs[3].grid(True)
    axs[3].axis('equal')

    axs[4].plot(figure[:, 0], figure[:, 1], marker='o', linestyle='-', label="Original " + figure_name)
    axs[4].plot(sheared_figure[:, 0], sheared_figure[:, 1], marker='o', linestyle='-', label="Sheared " + figure_name)
    axs[4].set_title("Sheared " + figure_name)
    axs[4].grid(True)
    axs[4].axis('equal')

    axs[5].plot(figure[:, 0], figure[:, 1], marker='o', linestyle='-', label="Original " + figure_name)
    axs[5].plot(custom_transformed_figure[:, 0], custom_transformed_figure[:, 1], marker='o', linestyle='-',
                label="Custom Transformed " + figure_name)
    axs[5].set_title("Custom Transformed " + figure_name)
    axs[5].grid(True)
    axs[5].axis('equal')

    for ax in axs:
        ax.legend()

    plt.tight_layout()
    plt.show()

figure_name = "triangle"
figure = figures[figure_name]
plot_transformations(figure, figure_name)



# 3D figures
pyramid = np.array([
    [0, 0, 0],
    [1, 0, 0],
    [1, 1, 0],
    [0, 1, 0],
    [0.5, 0.5, 1]
])

random_transformation_matrix_3d = np.random.rand(3, 3)
transformed_pyramid, transform_matrix_pyramid = custom_transform(pyramid, random_transformation_matrix_3d)
reflected_pyramid, reflect_matrix_pyramid = reflect_3d(pyramid, "z")

print("\n\nWorking with 3D figure")
print(f"Custom transform: \n{transform_matrix_pyramid}")
print(f"Reflected z-axis: \n{reflect_matrix_pyramid}")

fig, axs = plt.subplots(1, 3, figsize=(15, 10))
axs = axs.flatten()
pyramid_edges = [
    [0, 1], [1, 2], [2, 3], [3, 0],
    [0, 4], [1, 4], [2, 4], [3, 4]
]

ax1 = fig.add_subplot(131, projection='3d')
for edge in pyramid_edges:
    ax1.plot([pyramid[edge[0], 0], pyramid[edge[1], 0]],
             [pyramid[edge[0], 1], pyramid[edge[1], 1]],
             [pyramid[edge[0], 2], pyramid[edge[1], 2]], color='blue')
ax1.set_title("Original Pyramid")
ax1.view_init(azim=45, elev=30)

ax2 = fig.add_subplot(132, projection='3d')
for edge in pyramid_edges:
    ax2.plot([transformed_pyramid[edge[0], 0], transformed_pyramid[edge[1], 0]],
             [transformed_pyramid[edge[0], 1], transformed_pyramid[edge[1], 1]],
             [transformed_pyramid[edge[0], 2], transformed_pyramid[edge[1], 2]], color='green')
ax2.set_title("Transformed Pyramid")
ax2.view_init(azim=45, elev=30)

ax3 = fig.add_subplot(133, projection='3d')
for edge in pyramid_edges:
    ax3.plot([reflected_pyramid[edge[0], 0], reflected_pyramid[edge[1], 0]],
             [reflected_pyramid[edge[0], 1], reflected_pyramid[edge[1], 1]],
             [reflected_pyramid[edge[0], 2], reflected_pyramid[edge[1], 2]], color='red')
ax3.set_title("Reflected Pyramid")
ax3.view_init(azim=45, elev=30)

plt.tight_layout(pad=2)
plt.show()

