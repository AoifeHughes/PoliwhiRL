# -*- coding: utf-8 -*-
import hashlib
from skimage.metrics import structural_similarity as ssim
import cv2
from sklearn.neighbors import NearestNeighbors
import numpy as np
import os


class ImageMemory:
    def __init__(self, max_images=10000):
        self.max_images = max_images  # Maximum number of images to store
        self.images = {}  # Stores images in memory, mapped by hash
        self.image_order = []  # Track order of images for removing oldest
        self.feature_index = NearestNeighbors(
            n_neighbors=1, algorithm="auto"
        )  # For nearest neighbors searches
        self.features = np.array([]).reshape(
            0, 30 * 30 * 3
        )  # Stored features for all images, initialized for reshaping
        self.ids = []  # Image identifiers corresponding to features
        self._reset_state()

    def _hash_image(self, image):
        """Generate a hash for an image."""
        return hashlib.sha256(image.tobytes()).hexdigest()

    def _compute_features(self, image):
        """Compute a simplified feature vector for the image."""
        resized = cv2.resize(
            image, (30, 30), interpolation=cv2.INTER_AREA
        )  # Resize considering ndarray input
        return resized.flatten()

    def check_if_image_exists(self, target_image, threshold=100):
        """Check if a similar image exists."""
        target_features = self._compute_features(target_image)
        if self.features.shape[0] > 0:
            distances, indices = self.feature_index.kneighbors(
                [target_features], n_neighbors=1
            )
            if distances[0][0] < threshold:
                # Found a similar image, return its identifier
                target_hash = self.ids[indices[0][0]]

                return True, target_hash
        return False, None

    def pop_image(self, image_hash):
        """Remove an image from memory based on its hash."""
        if image_hash in self.images:
            del self.images[image_hash]  # Remove image from memory
            self.image_order.remove(image_hash)  # Remove image from order
            index = self.ids.index(
                image_hash
            )  # Find index of image in features and ids
            self.features = np.delete(
                self.features, index, axis=0
            )  # Remove image from features
            self.ids.pop(index)  # Remove image from ids
            self.feature_index.fit(self.features)  # Update nearest neighbors index

    def check_and_store_image(self, target_image, threshold=100):
        """Check if a similar image exists; store the new image in memory if
        not."""
        if isinstance(target_image, str):
            target_image = cv2.imread(target_image)
            target_image = cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB)
            target_image = target_image.astype(np.uint8)
        exists, target_hash = self.check_if_image_exists(target_image, threshold)
        if exists:
            return False, target_hash

        # Check if storage limit is reached and pop the oldest image if necessary
        if len(self.images) >= self.max_images:
            oldest_hash = self.image_order.pop(0)  # Remove the oldest image reference
            self.pop_image(oldest_hash)  # Remove the oldest image from memory
        # No similar image found, add new image to storage
        target_hash = self._hash_image(target_image)
        self.images[target_hash] = target_image  # Store image in memory
        self.image_order.append(target_hash)  # Track image order
        if len(self.features) == 0:
            self.features = np.array([self._compute_features(target_image)])
        else:
            self.features = np.vstack(
                [self.features, self._compute_features(target_image)]
            )
        self.ids.append(target_hash)
        # Update the nearest neighbors index with the new features
        self.feature_index.fit(self.features)
        return True, target_hash

    def compare_images(self, img1, img2):
        """Compute SSIM between two images."""
        # Ensure img1 and img2 are in the correct format if not already
        img1 = img1.astype(np.float32) / 255.0
        img2 = img2.astype(np.float32) / 255.0
        return ssim(img1, img2, multichannel=True)

    def num_images(self):
        """Return the number of images stored in memory."""
        return len(self.images)

    def _reset_state(self):
        """Reset the storage state to its initial configuration."""
        self.images = {}
        self.image_order = []
        self.features = np.array([]).reshape(
            0, 30 * 30 * 3
        )  # Reset features for reshaping
        self.ids = []
        # Reinitialize the nearest neighbors index with no data
        self.feature_index = NearestNeighbors(n_neighbors=1, algorithm="auto")

    def reset(self):
        """Public method to reset the object state for reuse."""
        self._reset_state()

    def save_image(self, image_hash, file_path):
        """Save an image from memory to a file based on its hash."""
        if image_hash in self.images:
            image_data = self.images[image_hash]
            success = cv2.imwrite(file_path, image_data)
            return success
        else:
            return False

    def save_all_images(self, folder_path):
        """Save all images stored in memory to the specified folder, maintaining the order they were added."""
        # Check if the folder exists, if not, create it
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        for index, image_hash in enumerate(self.image_order):
            image_data = self.images[image_hash]
            # Format the file name to include the index for ordering
            file_name = f"image_{index+1:04d}.jpg"  # Pad the index with zeros
            file_path = os.path.join(folder_path, file_name)
            success = cv2.imwrite(file_path, image_data)
            if not success:
                print(f"Failed to save image: {file_name}")
