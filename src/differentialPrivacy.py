import numpy as np
import torch


class DifferentialPrivacy:
    """
    Differential privacy class.
    """
    def __init__(
        self,
        granularity=1.0,
        clipping_threshold=1.0,
        noise_scale=1.0,
        modulus=2**16,
        rotation_type="hd",
        zeroing=False,
    ):
        self.granularity = granularity
        self.clipping_threshold = clipping_threshold
        self.noise_scale = noise_scale
        self.modulus = modulus
        self.rotation_type = rotation_type
        self.zeroing = zeroing

    def scale_and_clip(self, vector):
        """
        Scale and clip the vector.
        """
        vector_norm = np.linalg.norm(vector, 2)
        return (
            (1 / self.granularity)
            * min(1, self.clipping_threshold / vector_norm)
            * vector
        )

    def pad_to_nearest_power_of_two(self, vector):
        """
        Pad the vector to the nearest power of two.
        """
        next_power_of_two = int(2 ** np.ceil(np.log2(len(vector))))
        return np.pad(vector, (0, next_power_of_two - len(vector)))

    def rotate(self, vector):
        """
        Rotate the vector.
        """
        if self.rotation_type == "hd":
            # Hadamard transform
            return self.fwht(vector)
        elif self.rotation_type == "dft":
            # Discrete Fourier transform
            return np.fft.fft(vector).real
        else:
            raise ValueError("Invalid rotation type. Choose 'hd' or 'dft'.")

    def add_noise(self, vector):
        """
        Add noise to the vector.
        """
        noise_vector = np.round(
            np.random.normal(
                loc=0,
                scale=self.noise_scale / self.granularity,
                size=len(vector),
            )
        ).astype(int)
        return (vector + noise_vector) % self.modulus

    def process_parameters(self, parameters):
        """
        Apply differential privacy to the parameters before aggregation.

        NOTE: This method could be called before or after the model was trained.
            - Before training: Apply differential privacy to the data.
            - After training: Apply differential privacy to the model's parameters.
        """
        noisy_vectors = []
        for param in parameters:
            vector = param.cpu().detach().numpy().flatten()
            scaled_vector = self.scale_and_clip(vector)
            padded_vector = self.pad_to_nearest_power_of_two(scaled_vector)
            rotated_vector = self.rotate(padded_vector)
            rounded_vector = np.round(rotated_vector).astype(int)
            noisy_vector = self.add_noise(rounded_vector)
            if self.zeroing:
                # TODO: Implement zeroing
                pass
            noisy_vectors.append(noisy_vector)
        return noisy_vectors

    def normalize_vector(self, vector):
        """
        Normalize the input vector to the range [-self.modulus // 2, self.modulus // 2).
        """
        return (vector - self.modulus // 2) % self.modulus - self.modulus // 2

    def fwht(self, data):
        """
        Fast Walsh-Hadamard Transform.
        TODO: Implement a more efficient version of the FWHT.
        """
        h = 1
        while h < len(data):
            for i in range(0, len(data), h * 2):
                for j in range(i, i + h):
                    data[j], data[j + h] = data[j] + data[j + h], data[j + h] - data[j]
            h *= 2
        return data

    def inverse_rotate(self, vector):
        """
        Apply the inverse rotation to the input vector, depending on the rotation type.
        """
        # Inverse Hadamard transform
        if self.rotation_type == "hd":
            return self.granularity * self.fwht(vector)
        # Inverse Discrete Fourier transform
        elif self.rotation_type == "dft":
            return self.granularity * np.fft.ifft(vector).real
        else:
            raise ValueError("Invalid rotation type. Choose 'hd' or 'dft'.")

    def secure_aggregate(self, aggregated_vectors):
        """
        Securely aggregate the vectors.
        """
        result_vectors = []
        for aggregated_vector in aggregated_vectors:
            adjusted_vector = self.normalize_vector(aggregated_vector)
            result_vector = self.inverse_rotate(adjusted_vector)
            result_vectors.append(torch.Tensor(result_vector))
        return result_vectors
