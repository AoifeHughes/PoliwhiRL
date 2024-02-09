# -*- coding: utf-8 -*-
import hashlib
from skimage.metrics import structural_similarity as ssim
import cv2
from sklearn.neighbors import NearestNeighbors
import numpy as np


class ImageMemory:
    def __init__(self, max_images=1000):
        self.max_images = max_images  # Maximum number of images to store
        self.images = {}  # Stores images in memory, mapped by hash
        self.image_order = []  # Track order of images for removing oldest
        self.feature_index = NearestNeighbors(
            n_neighbors=1, algorithm="auto"
        )  # For nearest neighbors searches
        self.features = []  # Stored features for all images
        self.ids = []  # Image identifiers corresponding to features
        self._reset_state()

    def _hash_image(self, image):
        """Generate a hash for an image."""
        return hashlib.sha256(image.tobytes()).hexdigest()

    def _compute_features(self, image):
        """Compute a simplified feature vector for the image."""
        resized = cv2.resize(np.array(image), (30, 30), interpolation=cv2.INTER_AREA)
        return resized.flatten()

    def check_and_store_image(self, target_image, threshold=0.99):
        """Check if a similar image exists; store the new image in memory if not."""
        target_features = self._compute_features(target_image)
        if len(self.features) > 0:
            distances, indices = self.feature_index.kneighbors(
                [target_features], n_neighbors=1
            )
            if distances[0][0] < threshold:
                # Found a similar image, return its identifier
                target_hash = self.ids[indices[0][0]]
                return False, target_hash

        # Check if storage limit is reached and pop the oldest image if necessary
        if len(self.images) >= self.max_images:
            oldest_hash = self.image_order.pop(0)  # Remove the oldest image reference
            self.images.pop(oldest_hash)  # Remove the oldest image from storage

        # No similar image found, add new image to storage
        target_hash = self._hash_image(target_image)
        self.images[target_hash] = target_image  # Store image in memory
        self.image_order.append(target_hash)  # Track image order
        self.features.append(target_features)
        self.ids.append(target_hash)
        # Update the nearest neighbors index with the new features
        self.feature_index.fit(self.features)
        return True, target_hash

    def compare_images(self, img1, img2):
        """Compute SSIM between two images."""
        return ssim(img1, img2)

    def num_images(self):
        """Return the number of images stored in memory."""
        return len(self.images)

    def _reset_state(self):
        """Reset the storage state to its initial configuration."""
        self.images = {}
        self.image_order = []
        self.features = []
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
