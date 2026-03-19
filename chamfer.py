import trimesh
import numpy as np
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
import os

# Create directory if it doesn't exist
os.makedirs("output", exist_ok=True)


def normalize_scale(mesh):
    # Normalize the scale of the mesh to fit within a unit cube
    vertices = mesh.vertices - mesh.centroid
    max_extent = np.max(np.linalg.norm(vertices, axis=1))
    mesh.vertices = vertices / max_extent
    return mesh

def sample_points(mesh, num_points=10000):
    points, _ = trimesh.sample.sample_surface(mesh, num_points)
    return points

def chamfer_distance(A, B):
    """
    Computes the chamfer distance between two sets of points A and B.
    """
    tree = KDTree(B)
    dist_A = tree.query(A)[0]
    tree = KDTree(A)
    dist_B = tree.query(B)[0]
    return np.mean(dist_A) + np.mean(dist_B)

def plot_sampled_points(points, title, filename):
    """
    Plot the sampled points in a 3D scatter plot and save it as an image using OpenCV.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1)
    ax.set_title(title)

    # Remove grid and axis scales
    ax.grid(False)
    ax.set_axis_off()

    # Hide tick marks and labels
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    
    # Save plot to an image file
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename, img)

    plt.close(fig)

# Load meshes — update these paths to your mesh files
mesh3 = trimesh.load('<path/to/predicted_mesh.ply>', force='mesh')  # Predicted mesh
mesh1 = trimesh.load('<path/to/gt_mesh.obj>', force='mesh')  # Ground truth mesh

# Normalize scales
mesh1 = normalize_scale(mesh1)
mesh3 = normalize_scale(mesh3)

# Sample points from the meshes
points1 = sample_points(mesh1, num_points=10000)
points3 = sample_points(mesh3, num_points=10000)

# Calculate Chamfer Distance
print("Chamfer Distance EventNeuS:", chamfer_distance(points1, points3))

plot_sampled_points(points1, "GT Mesh Sampled Points", "./output/mesh_gt.png")
plot_sampled_points(points3, "Predicted Mesh Sampled Points", "./output/mesh_pred.png")