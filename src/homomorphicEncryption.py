import tenseal as ts
import torch


class HomomorphicEncryption:
    """
    Homomorphic encryption class.
    """
    def __init__(self):
        self.context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=8192,
            coeff_mod_bit_sizes=[60, 40, 40, 60],
        )
        self.context.global_scale = 2**40
        self.context.generate_galois_keys()

    def encrypt(self, parameters):
        """
        Encrypt the parameters.
        """
        encrypted_vectors = []
        for param in parameters:
            np_param = param.cpu().detach().numpy().flatten()
            encrypted_vector = ts.ckks_vector(self.context, np_param)
            encrypted_vectors.append(encrypted_vector)
        return encrypted_vectors

    def decrypt(self, encrypted_vectors):
        """
        Decrypt the encrypted vectors.
        """
        return [
            torch.tensor(vec.decrypt(), dtype=torch.float32)
            for vec in encrypted_vectors
        ]
