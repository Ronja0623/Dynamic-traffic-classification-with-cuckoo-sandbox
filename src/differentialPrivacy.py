import numpy as np
import torch

class DifferentialPrivacy:
    def __init__(self, granularity=1.0, clipping_threshold=1.0, noise_scale=1.0, modulus=2**16, rotation_type="hd", zeroing=False):
        self.granularity = granularity
        self.clipping_threshold = clipping_threshold
        self.noise_scale = noise_scale
        self.modulus = modulus
        self.rotation_type = rotation_type
        self.zeroing = zeroing
    
    def scale_and_clip(self, vector):
        vector_norm = np.linalg.norm(vector, 2)
        return (1 / self.granularity) * min(1, self.clipping_threshold / vector_norm) * vector

    def pad_to_nearest_power_of_two(self, vector):
        next_power_of_two = int(2 ** np.ceil(np.log2(len(vector))))
        return np.pad(vector, (0, next_power_of_two - len(vector)))

    def rotate(self, vector):
        if self.rotation_type == "hd":
            return self.fwht(vector)
        elif self.rotation_type == "dft":
            return np.fft.fft(vector).real
        else:
            raise ValueError("Invalid rotation type. Choose 'hd' or 'dft'.")

    def add_noise(self, vector):
        noise_vector = np.round(
            np.random.normal(
                loc=0,
                scale=self.noise_scale / self.granularity,
                size=len(vector),
            )
        ).astype(int)
        return (vector + noise_vector) % self.modulus

    def process_parameters(self, parameters):
        aggregated_vectors = []
        for param in parameters:
            vector = param.cpu().detach().numpy().flatten()
            scaled_vector = self.scale_and_clip(vector)
            padded_vector = self.pad_to_nearest_power_of_two(scaled_vector)
            rotated_vector = self.rotate(padded_vector)
            rounded_vector = np.round(rotated_vector).astype(int)
            noisy_vector = self.add_noise(rounded_vector)
            if self.zeroing:
                pass
            aggregated_vectors.append(noisy_vector)
        return aggregated_vectors
    
    def adjust_vector(self, aggregated_vector):
        return (aggregated_vector - self.modulus // 2) % self.modulus - (self.modulus // 2)
    
    def fwht(self, data):
        h = 1
        while h < len(data):
            for i in range(0, len(data), h * 2):
                for j in range(i, i + h):
                    data[j], data[j + h] = data[j] + data[j + h], data[j + h] - data[j]
            h *= 2
        return data

    def inverse_rotate(self, adjusted_vector):
        if self.rotation_type == "hd":
            return self.granularity * self.fwht(adjusted_vector)
        elif self.rotation_type == "dft":
            return self.granularity * np.fft.ifft(adjusted_vector).real
        else:
            raise ValueError("Invalid rotation type. Choose 'hd' or 'dft'.")

    def secure_aggregate(self, aggregated_vectors):
        result_vectors = []
        for aggregated_vector in aggregated_vectors:
            adjusted_vector = self.adjust_vector(aggregated_vector)
            result_vector = self.inverse_rotate(adjusted_vector)
            result_vectors.append(torch.Tensor(result_vector))
        return result_vectors
