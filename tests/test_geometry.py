import numpy as np
import pytest
from NeedleFinderLib import GeometryUtils as geo

class Volume:
    def __init__(self, matrix): self.matrix = matrix
    def GetIJKToRASMatrix(self, target): target._m = self.matrix.copy()

@pytest.mark.parametrize('perm', [(0, 1, 2), (0, 2, 1), (2, 1, 0)])
@pytest.mark.parametrize('sign', [1, -1])
def test_ras_ijk_with_anisotropic_spacing_translation_and_axis_flip(perm, sign):
    matrix = np.eye(4)
    matrix[:3, :3] = np.diag([0.7 * sign, 1.2, 3.0])[:, perm]
    matrix[:3, 3] = [13, -25, 71]
    ijk = np.array([4.5, 8.25, 2.0])
    expected = matrix[:3, :3] @ ijk + matrix[:3, 3]
    volume = Volume(matrix)
    np.testing.assert_allclose(geo.ijk_to_ras(ijk, volume), expected)
    np.testing.assert_allclose(geo.ras_to_ijk(expected, volume), ijk)


def test_interpolation_and_out_of_volume_zero_padding():
    def sample(x, y, z, c=0):
        return x + 2*y + 3*z if min(x, y, z) >= 0 and max(x, y, z) <= 3 else 0
    assert geo.trilinear_interp(sample, 0.5, 1.5, 2.5) == 11
    assert geo.trilinear_interp(sample, -2, 1, 1) == 0
    assert geo.trilinear_interp(sample, 0.2, 1.2, 2.2, False) == 8


def test_zero_vector_normalization_and_physical_distance():
    vector, length = geo.normalized([0, 0, 0])
    np.testing.assert_array_equal(vector, [0, 0, 0])
    assert length == 0
    assert geo.distance_3d([0, 0, 0], [3, 4, 12]) == 13
