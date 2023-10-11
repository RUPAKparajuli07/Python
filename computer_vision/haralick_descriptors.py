import imageio
import numpy as np

def root_mean_square_error(original: np.ndarray, reference: np.ndarray) -> float:
    """Calculate the Root Mean Squared Error between two numpy arrays.

    Args:
        original (np.ndarray): The original image.
        reference (np.ndarray): The reference image.

    Returns:
        float: The Root Mean Squared Error (RMSE) value.
    """
    return np.sqrt(((original - reference) ** 2).mean())

def normalize_image(image: np.ndarray, cap: float = 255.0, data_type: np.dtype = np.uint8) -> np.ndarray:
    """Normalize a 2D numpy array to a specific range and data type.

    Args:
        image (np.ndarray): The input image.
        cap (float): The maximum cap amount for normalization.
        data_type (np.dtype): The numpy data type to set the output variable to.

    Returns:
        np.ndarray: The normalized 2D numpy array.
    """
    normalized = (image - np.min(image)) / (np.max(image) - np.min(image)) * cap
    return normalized.astype(data_type)

def normalize_array(array: np.ndarray, cap: float = 1) -> np.ndarray:
    """Normalize a 1D array to a specific range.

    Args:
        array (np.ndarray): The 1D array containing values to be normalized.
        cap (float): The maximum cap amount for normalization.

    Returns:
        np.ndarray: The normalized 1D numpy array.
    """
    diff = np.max(array) - np.min(array)
    return (array - np.min(array)) / (1 if diff == 0 else diff) * cap

def grayscale(image: np.ndarray) -> np.ndarray:
    """Convert an RGB image to grayscale using luminance weights.

    Args:
        image (np.ndarray): The RGB image.

    Returns:
        np.ndarray: The grayscale image.
    """
    return np.dot(image[:, :, 0:3], [0.299, 0.587, 0.114]).astype(np.uint8)

def binarize(image: np.ndarray, threshold: float = 127.0) -> np.ndarray:
    """Binarize a grayscale image using a specified threshold.

    Args:
        image (np.ndarray): The grayscale image to be binarized.
        threshold (float): The threshold value for binarization.

    Returns:
        np.ndarray: The binarized image.
    """
    return np.where(image > threshold, 1, 0)

def transform(image: np.ndarray, kind: str, kernel: np.ndarray | None = None) -> np.ndarray:
    """Apply an image transformation using erosion or dilation with a kernel.

    Args:
        image (np.ndarray): The binary image to be transformed.
        kind (str): The type of transformation: 'erosion' or 'dilation'.
        kernel (np.ndarray | None): The kernel to be used in convolution.

    Returns:
        np.ndarray: The transformed binary image.
    """
    if kernel is None:
        kernel = np.ones((3, 3))

    constant = 1 if kind == 'erosion' else 0
    apply = np.max if kind == 'erosion' else np.min

    center_x, center_y = (x // 2 for x in kernel.shape)
    transformed = np.zeros(image.shape, dtype=np.uint8)
    padded = np.pad(image, 1, "constant", constant_values=constant)

    for x in range(center_x, padded.shape[0] - center_x):
        for y in range(center_y, padded.shape[1] - center_y):
            center = padded[
                x - center_x : x + center_x + 1, y - center_y : y + center_y + 1
            ]
            transformed[x - center_x, y - center_y] = apply(center[kernel == 1])

    return transformed

def opening_filter(image: np.ndarray, kernel: np.ndarray | None = None) -> np.ndarray:
    """Apply an opening filter to a binary image.

    Args:
        image (np.ndarray): The binary image.
        kernel (np.ndarray | None): The kernel to be used in convolution.

    Returns:
        np.ndarray: The result of the opening filter.
    """
    if kernel is None:
        np.ones((3, 3))

    return transform(transform(image, "dilation", kernel), "erosion", kernel)

def closing_filter(image: np.ndarray, kernel: np.ndarray | None = None) -> np.ndarray:
    """Apply a closing filter to a binary image.

    Args:
        image (np.ndarray): The binary image.
        kernel (np.ndarray | None): The kernel to be used in convolution.

    Returns:
        np.ndarray: The result of the closing filter.
    """
    if kernel is None:
        kernel = np.ones((3, 3))
    return transform(transform(image, "erosion", kernel), "dilation", kernel)

def binary_mask(image_gray: np.ndarray, image_map: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Apply a binary mask to an image based on a mapping mask.

    Args:
        image_gray (np.ndarray): The grayscale image.
        image_map (np.ndarray): The binary mapping mask.

    Returns:
        tuple[np.ndarray, np.ndarray]: The mapped true value mask and its complementary false value mask.
    """
    true_mask, false_mask = image_gray.copy(), image_gray.copy()
    true_mask[image_map == 1] = 1
    false_mask[image_map == 0] = 0
    return true_mask, false_mask

def matrix_concurrency(image: np.ndarray, coordinate: tuple[int, int]) -> np.ndarray:
    """Calculate sample co-occurrence matrix based on input image and coordinates.

    Args:
        image (np.ndarray): The binary image.
        coordinate (tuple[int, int]): The coordinate to calculate the matrix.

    Returns:
        np.ndarray: The co-occurrence matrix.
    """
    matrix = np.zeros([np.max(image) + 1, np.max(image) + 1])

    offset_x, offset_y = coordinate

    for x in range(1, image.shape[0] - 1):
        for y in range(1, image.shape[1] - 1):
            base_pixel = image[x, y]
            offset_pixel = image[x + offset_x, y + offset_y]

            matrix[base_pixel, offset_pixel] += 1
    matrix_sum = np.sum(matrix)
    return matrix / (1 if matrix_sum == 0 else matrix_sum)

def haralick_descriptors(matrix: np.ndarray) -> list[float]:
    """Calculate Haralick descriptors based on a co-occurrence matrix.

    Args:
        matrix (np.ndarray): The co-occurrence matrix.

    Returns:
        list[float]: A list of Haralick descriptors.
    """
    i, j = np.ogrid[0: matrix.shape[0], 0: matrix.shape[1]
    prod = np.multiply(i, j)
    sub = np.subtract(i, j)

    maximum_prob = np.max(matrix)
    correlation = prod * matrix
    energy = np.power(matrix, 2)
    contrast = matrix * np.power(sub, 2)

    dissimilarity = matrix * np.abs(sub)
    inverse_difference = matrix / (1 + np.abs(sub))
    homogeneity = matrix / (1 + np.power(sub, 2))
    entropy = -(matrix[matrix > 0] * np.log(matrix[matrix > 0]))

    return [
        maximum_prob,
        correlation.sum(),
        energy.sum(),
        contrast.sum(),
        dissimilarity.sum(),
        inverse_difference.sum(),
        homogeneity.sum(),
        entropy.sum(),
    ]

def get_descriptors(masks: tuple[np.ndarray, np.ndarray], coordinate: tuple[int, int]) -> np.ndarray:
    """Calculate all Haralick descriptors for a sequence of co-occurrence matrices.

    Args:
        masks (tuple[np.ndarray, np.ndarray]): Tuple of two binary masks.
        coordinate (tuple[int, int]): The coordinate to calculate co-occurrence matrices.

    Returns:
        np.ndarray: The Haralick descriptors for the given masks.
    """
    descriptors = np.array(
        [haralick_descriptors(matrix_concurrency(mask, coordinate)) for mask in masks]
    )
    return np.concatenate(descriptors, axis=None)

def euclidean(point_1: np.ndarray, point_2: np.ndarray) -> np.float32:
    """Calculate the Euclidean distance between two points represented as numpy arrays.

    Args:
        point_1 (np.ndarray): The first point.
        point_2 (np.ndarray): The second point.

    Returns:
        np.float32: The Euclidean distance between the two points.
    """
    return np.sqrt(np.sum(np.square(point_1 - point_2))

def get_distances(descriptors: np.ndarray, base: int) -> list[tuple[int, float]]:
    """Calculate Euclidean distances between a selected base descriptor and all other descriptors.

    Args:
        descriptors (np.ndarray): The array of descriptors.
        base (int): The index of the base descriptor.

    Returns:
        list[tuple[int, float]]: A list of tuples containing index and distance.

    Example:
        >>> get_distances(np.array([2.0, 1.0, 4.0, 3.0, 6.0]), 2)
        [(2, 0.0), (3, 1.0), (1, 3.0), (0, 6.0), (4, 7.0)]
    """
    distances = np.array(
        [euclidean(descriptor, descriptors[base]) for descriptor in descriptors]
    )
    normalized_distances: list[float] = normalize_array(distances, 1).tolist()
    enum_distances = list(enumerate(normalized_distances))
    enum_distances.sort(key=lambda tup: tup[1], reverse=True)
    return enum_distances

# Main execution
if __name__ == "__main__":
    # Index to compare Haralick descriptors to
    index = int(input("Enter the index to compare Haralick descriptors to: "))

    # Coordinate for Haralick descriptor calculations
    q_value_list = [int(value) for value in input("Enter the coordinate (x, y) for Haralick descriptor calculations: ").split()]
    q_value = (q_value_list[0], q_value_list[1])

    # Format for the morphology operation (1 for opening, else for closing)
    parameters = {"format": int(input("Enter the format (1 for opening, else for closing): ")), "threshold": int(input("Enter the threshold: "))}

    # Number of images to perform methods on
    b_number = int(input("Enter the number of images: "))

    files, descriptors = [], []

    for i in range(b_number):
        file = input(f"Enter the path of image {i + 1}: ").rstrip()
        files.append(file)

        # Open the given image and calculate the morphological filter, respective masks, and correspondent Harralick Descriptors
        image = imageio.imread(file).astype(np.float32)
        gray = grayscale(image)
        threshold = binarize(gray, parameters["threshold"])

        morphological = (
            opening_filter(threshold)
            if parameters["format"] == 1
            else closing_filter(threshold)
        )
        masks = binary_mask(gray, morphological)
        descriptors.append(get_descriptors(masks, q_value))

    # Transform ordered distances array into a sequence of indexes corresponding to original file position
    distances = get_distances(np.array(descriptors), index)
    indexed_distances = np.array(distances).astype(np.uint8)[:, 0]

    # Finally, print distances considering the Haralick descriptions from the base file to all other images using the morphology method of choice
    print(f"Query: {files[index]}")
    print("Ranking:")
    for idx, file_idx in enumerate(indexed_distances):
        print(f"({idx}) {files[file_idx]}")
