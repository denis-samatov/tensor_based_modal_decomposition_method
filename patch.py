import re

file_path = "algorithm/TBMD/utils/geometry.py"
with open(file_path, "r") as f:
    content = f.read()

replacement = '''    def compute_proximity_penalty(self, sensor_positions: np.ndarray,
                                   min_distance: float) -> np.ndarray:
        """Computes a penalty for placing sensors too close to existing ones.

        Args:
            sensor_positions (np.ndarray): The indices of currently placed
                sensors, with shape (N_sensors,).
            min_distance (float): The minimum allowed distance between sensors,
                in coordinate units.

        Returns:
            np.ndarray: The penalty values for each cell, with shape
            (N_cells,). Higher values are less desirable.
        """
        N = len(self.mesh.coordinates)

        if len(sensor_positions) == 0:
            return np.zeros(N)

        # Create cache key to quickly return previously computed penalty
        pos_bytes = sensor_positions.tobytes()

        if hasattr(self, '_prox_last_pos_bytes') and self._prox_last_pos_bytes == pos_bytes and getattr(self, '_prox_last_min_dist', None) == min_distance:
            return self._prox_penalty_cache

        # Check if we can perform a fast incremental O(N) update instead of rebuilding KDTree
        if hasattr(self, '_prox_last_positions') and len(sensor_positions) == len(self._prox_last_positions) + 1:
            if np.array_equal(sensor_positions[:-1], self._prox_last_positions) and getattr(self, '_prox_last_min_dist', None) == min_distance:
                new_sensor_idx = int(sensor_positions[-1])
                penalty, new_min_dists = self.update_proximity_penalty(
                    new_sensor_idx, self._prox_min_dists_cache, min_distance
                )

                # Update cache
                self._prox_last_pos_bytes = pos_bytes
                self._prox_penalty_cache = penalty
                self._prox_min_dists_cache = new_min_dists
                self._prox_last_positions = sensor_positions.copy()
                self._prox_last_min_dist = min_distance
                return penalty

        # Get coordinates of existing sensors
        sensor_coords = self.mesh.coordinates[sensor_positions]

        # Compute distance from each cell to nearest sensor (Full Recomputation)
        tree = KDTree(sensor_coords)
        distances, _ = tree.query(self.mesh.coordinates)

        # Apply penalty: exponential decay with distance
        penalty = np.exp(-distances / (min_distance + 1e-10))

        # Update cache
        self._prox_last_pos_bytes = pos_bytes
        self._prox_penalty_cache = penalty
        self._prox_min_dists_cache = distances
        self._prox_last_positions = sensor_positions.copy()
        self._prox_last_min_dist = min_distance

        return penalty'''

old_method_pattern = r'    def compute_proximity_penalty\(self, sensor_positions: np\.ndarray,\n\s*min_distance: float\) -> np\.ndarray:\n[\s\S]*?return penalty'
new_content = re.sub(old_method_pattern, replacement, content)

with open(file_path, "w") as f:
    f.write(new_content)
