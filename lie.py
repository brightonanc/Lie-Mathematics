
import numpy as np
from scipy.linalg import expm, logm

I2 = np.array([
    [1, 0],
    [0, 1]
], dtype=np.float)
I3 = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
], dtype=np.float)
I4 = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
], dtype=np.float)
J = np.array([
    [ 0,  1],
    [-1,  0]
], dtype=np.float)
L1 = np.array([
    [ 0,  0,  0],
    [ 0,  0, -1],
    [ 0,  1,  0]
], dtype=np.float)
L2 = np.array([
    [ 0,  0,  1],
    [ 0,  0,  0],
    [-1,  0,  0]
], dtype=np.float)
L3 = np.array([
    [ 0, -1,  0],
    [ 1,  0,  0],
    [ 0,  0,  0]
], dtype=np.float)

class LieAlgebra:
    pass
class LieGroup:
    pass

"""
The so(2) Lie Algebra (2D rotations)
"""
class so2(LieAlgebra):
    def __init__(self, omega):
        self.omega = omega
    @staticmethod
    def from_vec(vec):
        """
            [w]
        """
        return so2(vec[0])
    @staticmethod
    def from_mat(mat):
        """
            [ 0, -w],
            [ w,  0]
        """
        return so2(mat[1, 0])
    @staticmethod
    def from_group(lie_group):
        """
            [ cos(w), -sin(w)],
            [ sin(w),  cos(w)]
        """
        if not isinstance(lie_group, SO2):
            raise ValueError('Must be SO(2) Lie group')
        return lie_log(lie_group, 1)
    def to_vec(self):
        return np.array([self.omega])
    def to_mat(self):
        return -self.omega * J
    def to_group(self):
        return lie_exp(self, 1)

"""
The SO(2) Lie Group (2D rotations)
"""
class SO2(LieGroup):
    def __init__(self, mat):
        self.mat = mat

"""
The so(3) Lie Algebra (3D rotations)
"""
class so3(LieAlgebra):
    def __init__(self, omega1, omega2, omega3):
        self.omega1 = omega1
        self.omega2 = omega2
        self.omega3 = omega3
    @staticmethod
    def from_vec(vec):
        """
            [w1, w2, w3]
        """
        return so3(vec[0], vec[1], vec[2])
    @staticmethod
    def from_mat(mat):
        """
            [   0, -w3,  w2],
            [  w3,   0, -w1],
            [ -w2,  w1,   0]
        """
        return so3(-mat[1, 2], mat[0, 2], -mat[0, 1])
    @staticmethod
    def from_group(lie_group):
        if not isinstance(lie_group, SO3):
            raise ValueError('Must be SO(3) Lie group')
        return lie_log(lie_group, 1)
    def to_vec(self):
        return np.array([self.omega1, self.omega2, self.omega3])
    def to_mat(self):
        return (self.omega1 * L1) + (self.omega2 * L2) + (self.omega3 * L3)
    def to_group(self):
        return lie_exp(self, 1)

"""
The SO(3) Lie Group (3D rotations)
"""
class SO3(LieGroup):
    def __init__(self, mat):
        self.mat = mat

class e2:
    def __init__(self, v1, v2):
        self.v1 = v1
        self.v2 = v2
    @staticmethod
    def from_vec(vec):
        """
            [v1, v2]
        """
        return e2(vec[0], vec[1])
    @staticmethod
    def from_mat(mat):
        """
            [v1],
            [v2]
        """
        return e2(mat[0, 0], mat[1, 0])
    def to_vec(self):
        return np.array([self.v1, self.v2])
    def to_mat(self):
        return np.array([[self.v1], [self.v2]])

class e3:
    def __init__(self, v1, v2, v3):
        self.v1 = v1
        self.v2 = v2
        self.v3 = v3
    @staticmethod
    def from_vec(vec):
        """
            [v1, v2, v3]
        """
        return e3(vec[0], vec[1], vec[2])
    @staticmethod
    def from_mat(mat):
        """
            [v1],
            [v2],
            [v3]
        """
        return e3(mat[0, 0], mat[1, 0], mat[2, 0])
    def to_vec(self):
        return np.array([self.v1, self.v2, self.v3])
    def to_mat(self):
        return np.array([[self.v1], [self.v2], [self.v3]])

"""
The se(2) Lie Algebra (2D rotations/translations)
"""
class se2(LieAlgebra):
    def __init__(self, so2_elem, e2_elem):
        self.so2_elem = so2_elem
        self.e2_elem = e2_elem
    @staticmethod
    def from_vec(vec):
        """
            [v1, v2, w]
        """
        return se2(so2.from_vec(vec[[2]]), e2.from_vec(vec[:2]))
    @staticmethod
    def from_mat(mat):
        """
            [ 0, -w, v1],
            [ w,  0, v2],
            [ 0,  0,  0]
        """
        return se2(so2.from_mat(mat[:2, :2]), e2.from_mat(mat[:2, [2]]))
    @staticmethod
    def from_group(lie_group):
        if not isinstance(lie_group, SE2):
            raise ValueError('Must be SE(2) Lie group')
        return lie_log(lie_group, 1)
    def to_vec(self):
        return np.concatenate((self.e2_elem.to_vec(), self.so2_elem.to_vec()))
    def to_mat(self):
        mat = np.zeros((3, 3))
        mat[:2, :2] = self.so2_elem.to_mat()
        mat[:2, [2]] = self.e2_elem.to_mat()
        return mat
    def to_group(self):
        return lie_exp(self, 1)

"""
The SE(2) Lie Group (2D rotations/translations)
"""
class SE2(LieGroup):
    def __init__(self, mat):
        self.mat = mat


"""
The se(3) Lie Algebra (3D rotations/translations)
"""
class se3(LieAlgebra):
    def __init__(self, so3_elem, e3_elem):
        self.so3_elem = so3_elem
        self.e3_elem = e3_elem
    @staticmethod
    def from_vec(vec):
        """
            [v1, v2, v3, w1, w2, w3]
        """
        return se3(so3.from_vec(vec[3:]), e3.from_vec(vec[:3]))
    @staticmethod
    def from_mat(mat):
        """
            [   0, -w3,  w2,  v1],
            [  w3,   0, -w1,  v2],
            [ -w2,  w1,   0,  v3],
            [   0,   0,   0,   0]
        """
        return se3(so3.from_mat(mat[:3, :3]), e3.from_mat(mat[:3, [3]]))
    @staticmethod
    def from_group(lie_group):
        if not isinstance(lie_group, SE2):
            raise ValueError('Must be SE(2) Lie group')
        return lie_log(lie_group, 1)
    def to_vec(self):
        return np.concatenate((self.e3_elem.to_vec(), self.so3_elem.to_vec()))
    def to_mat(self):
        mat = np.zeros((4, 4))
        mat[:3, :3] = self.so3_elem.to_mat()
        mat[:3, [3]] = self.e3_elem.to_mat()
        return mat
    def to_group(self):
        return lie_exp(self, 1)

"""
The SE(3) Lie Group (3D rotations/translations)
"""
class SE3(LieGroup):
    def __init__(self, mat):
        self.mat = mat


def lie_exp(lie_algebra, t):
    if not isinstance(lie_algebra, LieAlgebra):
        raise ValueError('Exponents only exist for lie algebras')
    if isinstance(lie_algebra, so2):
        omega = lie_algebra.omega
        return SO2(
            (np.cos(omega * t) * I2) - (np.sin(omega * t) * J)
        )
    elif isinstance(lie_algebra, so3):
        mag_omega = np.linalg.norm(lie_algebra.to_vec())
        if 0 == mag_omega:
            return SO3(I3)
        omega_hat = lie_algebra.to_mat()
        omega_hat_sq = omega_hat @ omega_hat
        return SO3(
            I3 + ((np.sin(mag_omega * t) / mag_omega) * omega_hat) + \
            (((1 - np.cos(mag_omega * t)) / (mag_omega ** 2)) * omega_hat_sq)
        )
    elif isinstance(lie_algebra, se2):
        mat = np.zeros((3, 3))
        exp_omega_hat_t = lie_exp(lie_algebra.so2_elem, t).mat
        omega = lie_algebra.so2_elem.omega
        v = lie_algebra.e2_elem.to_mat()
        mat[:2, :2] = exp_omega_hat_t
        if 0 == omega:
            mat[:2, [2]] = t * v
        else:
            mat[:2, [2]] = (1/omega) * J @ (exp_omega_hat_t - I2) @ v
        mat[2, 2] = 1
        return SE2(mat)
    elif isinstance(lie_algebra, se3):
        mat = np.zeros((4, 4))
        exp_omega_hat_t = lie_exp(lie_algebra.so3_elem, t).mat
        mag_omega = np.linalg.norm(lie_algebra.so3_elem.to_vec())
        v = lie_algebra.e3_elem.to_mat()
        mat[:3, :3] = exp_omega_hat_t
        if 0 == mag_omega:
            mat[:3, [3]] = t * v
        else:
            mag_omega_t = mag_omega * t
            mag_omega_sq = mag_omega ** 2
            mag_omega_cu = mag_omega ** 3
            omega_hat = lie_algebra.so3_elem.to_mat()
            omega_hat_sq = omega_hat @ omega_hat
            mat[:3, [3]] = (
                (t * I3) +
                (((1 - np.cos(mag_omega_t)) / mag_omega_sq) * omega_hat) +
                (((mag_omega_t - np.sin(mag_omega_t)) / mag_omega_cu) *
                 omega_hat_sq)
            ) @ v
        mat[3, 3] = 1
        return SE3(mat)


def lie_log(lie_group, t):
    if not isinstance(lie_group, LieGroup):
        raise ValueError('Logs only exist for lie groups')
    if 0 == t:
        raise ValueError('Logs only exist for nonzero t')
    if isinstance(lie_group, SO2):
        R = lie_group.mat
        # Have to have signum provide 1 for x >= 0 and -1 for x < 0 (range
        # is (-pi, pi] ). np.sign provides 0 for x = 0, so I can't use it here
        if R[1, 0] >= 0:
            sign = 1
        else:
            sign = -1
        omega_t = sign * np.arccos(R[0, 0])
        return so2(
            (1/t) * omega_t
        )
    elif isinstance(lie_group, SO3):
        R = lie_group.mat
        if np.all(R == I3):
            return so3(0, 0, 0)
        if np.all(R == R.T):
            mag_omega_sq = (np.pi / t) ** 2
            outer_prod = 0.5 * mag_omega_sq * (R + I3)
            omega1 = np.sqrt(outer_prod[0, 0])
            if 0 != omega1:
                omega2 = np.sign(outer_prod[1, 0]) * np.sqrt(outer_prod[1, 1])
                omega3 = np.sign(outer_prod[2, 0]) * np.sqrt(outer_prod[2, 2])
            else:
                omega2 = np.sqrt(outer_prod[1, 1])
                if 0 != omega2:
                    omega3 = np.sign(outer_prod[2, 1]) * \
                             np.sqrt(outer_prod[2, 2])
                else:
                    omega3 = np.sqrt(outer_prod[2, 2])
            return so3(omega1, omega2, omega3)
        A = 0.5 * (R - R.T)
        B = 0.5 * (R + R.T)
        _A_ = np.sqrt(-0.5 * np.trace(A @ A))
        _B_ = 0.5 * (np.trace(B) - 1)
        omega_hat_t = (np.arccos(_B_) / _A_) * A
        return so3.from_mat(
            (1/t) * omega_hat_t
        )
    elif isinstance(lie_group, SE2):
        exp_xi_hat_t = lie_group.mat
        exp_omega_hat_t = exp_xi_hat_t[:2, :2]
        SO2_elem = SO2(exp_omega_hat_t)
        so2_elem = lie_log(SO2_elem, t)
        omega = so2_elem.omega
        if 0 == omega:
            v = (1/t) * exp_xi_hat_t[:2, [2]]
        else:
            v = -omega * np.linalg.inv(exp_omega_hat_t - I2) @ J @ \
                    exp_xi_hat_t[:2, [2]]
        e2_elem = e2.from_mat(v)
        return se2(so2_elem, e2_elem)
    elif isinstance(lie_group, SE3):
        exp_xi_hat_t = lie_group.mat
        exp_omega_hat_t = exp_xi_hat_t[:3, :3]
        SO3_elem = SO3(exp_omega_hat_t)
        so3_elem = lie_log(SO3_elem, t)
        mag_omega = np.linalg.norm(so3_elem.to_vec())
        if 0 == mag_omega:
            v = (1/t) * exp_xi_hat_t[:3, [3]]
        else:
            mag_omega_t = mag_omega * t
            mag_omega_sq = mag_omega ** 2
            mag_omega_cu = mag_omega ** 3
            omega_hat = so3_elem.to_mat()
            omega_hat_sq = omega_hat @ omega_hat
            v = np.linalg.inv(
                (t * I3) +
                (((1 - np.cos(mag_omega_t)) / mag_omega_sq) * omega_hat) +
                (((mag_omega_t - np.sin(mag_omega_t)) / mag_omega_cu) *
                 omega_hat_sq)
            ) @ exp_xi_hat_t[:3, [3]]
        e3_elem = e3.from_mat(v)
        return se3(so3_elem, e3_elem)


if __name__ == '__main__':
    np.random.seed(0)
    w_lower = -10
    w_upper = 10
    v_lower = -10
    v_upper = 10
    t_upper = 10
    def rand_omega(n=None):
        vec = ((w_upper - w_lower) * np.random.random_sample(n)) + w_lower
        if n is None:
            return vec
        return vec.tolist()
    def rand_v(n=None):
        vec = ((v_upper - v_lower) * np.random.random_sample(n)) + v_lower
        if n is None:
            return vec
        return vec.tolist()
    def rand_t(n=None):
        vec = t_upper * np.random.random_sample(n)
        if n is None:
            return vec
        return vec.tolist()
    # Test SO2
    N = 100
    # (omega, t)
    test_vec = [
        (0, 0),
        (0, 5),
        (-np.pi, 0),
        (-np.pi, 5),
        (np.pi, 0),
        (np.pi, 5),
        (rand_omega(), 0),
        (rand_omega(), 5)
    ]
    test_vec += zip(rand_omega(N), rand_t(N))
    for omega, t in test_vec:
        vec = np.array([omega])
        mat = np.array([[0, -omega], [omega, 0]])
        group = expm(mat)
        lie_algebra = so2(omega)
        lie_vec = lie_algebra.to_vec()
        lie_mat = lie_algebra.to_mat()
        lie_group = lie_algebra.to_group()
        assert np.all(lie_vec == vec)
        assert np.all(lie_mat == mat)
        assert np.all(np.isclose(lie_group.mat, group))
        exp_ = expm(mat * t)
        lie_exp_ = lie_exp(lie_algebra, t)
        assert np.all(np.isclose(lie_exp_.mat, exp_))
        if 0 != t:
            lie_log_ = lie_log(lie_exp_, t)
            lie_exp_recon = lie_exp(lie_log_, t)
            assert np.all(np.isclose(lie_exp_recon.mat, exp_))
    # Test SO3
    N = 100
    # (omega1, omega2, omega3, t)
    test_vec = [
        (0, 0, 0, 0),
        (0, 0, 0, 5),
        (-np.pi, 0, 0, 0),
        (-np.pi, 0, 0, 5),
        (np.pi, 0, 0, 0),
        (np.pi, 0, 0, 5),
        (0, -np.pi, 0, 0),
        (0, -np.pi, 0, 5),
        (0, np.pi, 0, 0),
        (0, np.pi, 0, 5),
        (0, 0, -np.pi, 0),
        (0, 0, -np.pi, 5),
        (0, 0, np.pi, 0),
        (0, 0, np.pi, 5),
        (np.pi - 1, np.sqrt((2 * np.pi) - 1), 0, 0),
        (np.pi - 1, np.sqrt((2 * np.pi) - 1), 0, 5),
        (np.pi - 1, 0, np.sqrt((2 * np.pi) - 1), 0),
        (np.pi - 1, 0, np.sqrt((2 * np.pi) - 1), 5),
        (0, np.pi - 1, np.sqrt((2 * np.pi) - 1), 0),
        (0, np.pi - 1, np.sqrt((2 * np.pi) - 1), 5),
        (rand_omega(), rand_omega(), rand_omega(), 0),
        (rand_omega(), rand_omega(), rand_omega(), 5),
    ]
    test_vec += zip(rand_omega(N), rand_omega(N), rand_omega(N), rand_t(N))
    for omega1, omega2, omega3, t in test_vec:
        vec = np.array([omega1, omega2, omega3])
        mat = np.array([[0, -omega3, omega2],
                        [omega3, 0, -omega1],
                        [-omega2, omega1, 0]])
        group = expm(mat)
        lie_algebra = so3(omega1, omega2, omega3)
        lie_vec = lie_algebra.to_vec()
        lie_mat = lie_algebra.to_mat()
        lie_group = lie_algebra.to_group()
        assert np.all(lie_vec == vec)
        assert np.all(lie_mat == mat)
        assert np.all(np.isclose(lie_group.mat, group))
        exp_ = expm(mat * t)
        lie_exp_ = lie_exp(lie_algebra, t)
        assert np.all(np.isclose(lie_exp_.mat, exp_))
        if 0 != t:
            lie_log_ = lie_log(lie_exp_, t)
            lie_exp_recon = lie_exp(lie_log_, t)
            assert np.all(np.isclose(lie_exp_recon.mat, exp_))
    # Test SE2
    N = 100
    # (omega, v1, v2, t)
    test_vec = [
        (0, 0, 0, 0),
        (0, 0, 0, 5),
        (0, 2, 0, 0),
        (0, 2, 0, 5),
        (0, 0, 2, 0),
        (0, 0, 2, 5),
        (-np.pi, 0, 0, 0),
        (-np.pi, 0, 0, 5),
        (-np.pi, 2, 0, 0),
        (-np.pi, 2, 0, 5),
        (-np.pi, 0, 2, 0),
        (-np.pi, 0, 2, 5),
        (np.pi, 0, 0, 0),
        (np.pi, 0, 0, 5),
        (np.pi, 2, 0, 0),
        (np.pi, 2, 0, 5),
        (np.pi, 0, 2, 0),
        (np.pi, 0, 2, 5),
        (rand_omega(), 0, 0, 0),
        (rand_omega(), 0, 0, 5),
        (rand_omega(), 2, 0, 0),
        (rand_omega(), 2, 0, 5),
        (rand_omega(), 0, 2, 0),
        (rand_omega(), 0, 2, 5)
    ]
    test_vec += zip(rand_omega(N), rand_v(N), rand_v(N), rand_t(N))
    for omega, v1, v2, t in test_vec:
        vec = np.array([v1, v2, omega])
        mat = np.array([[0, -omega, v1],
                        [omega, 0, v2],
                        [0, 0, 0]])
        group = expm(mat)
        lie_algebra = se2(so2(omega), e2(v1, v2))
        lie_vec = lie_algebra.to_vec()
        lie_mat = lie_algebra.to_mat()
        lie_group = lie_algebra.to_group()
        assert np.all(lie_vec == vec)
        assert np.all(lie_mat == mat)
        assert np.all(np.isclose(lie_group.mat, group))
        exp_ = expm(mat * t)
        lie_exp_ = lie_exp(lie_algebra, t)
        assert np.all(np.isclose(lie_exp_.mat, exp_))
        if 0 != t:
            lie_log_ = lie_log(lie_exp_, t)
            lie_exp_recon = lie_exp(lie_log_, t)
            assert np.all(np.isclose(lie_exp_recon.mat, exp_))
    # Test SE3
    N = 100
    # (omega1, omega2, omega3, v1, v2, v3, t)
    test_vec = [
        (0, 0, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 0, 5),
        (0, 0, 0, 2, 0, 0, 0),
        (0, 0, 0, 2, 0, 0, 5),
        (0, 0, 0, 0, 2, 0, 0),
        (0, 0, 0, 0, 2, 0, 5),
        (0, 0, 0, 0, 0, 2, 0),
        (0, 0, 0, 0, 0, 2, 5),
        (-np.pi, 0, 0, 0, 0, 0, 0),
        (-np.pi, 0, 0, 0, 0, 0, 5),
        (-np.pi, 0, 0, 2, 0, 0, 0),
        (-np.pi, 0, 0, 2, 0, 0, 5),
        (-np.pi, 0, 0, 0, 2, 0, 0),
        (-np.pi, 0, 0, 0, 2, 0, 5),
        (-np.pi, 0, 0, 0, 0, 2, 0),
        (-np.pi, 0, 0, 0, 0, 2, 5),
        (np.pi, 0, 0, 0, 0, 0, 0),
        (np.pi, 0, 0, 0, 0, 0, 5),
        (np.pi, 0, 0, 2, 0, 0, 0),
        (np.pi, 0, 0, 2, 0, 0, 5),
        (np.pi, 0, 0, 0, 2, 0, 0),
        (np.pi, 0, 0, 0, 2, 0, 5),
        (np.pi, 0, 0, 0, 0, 2, 0),
        (np.pi, 0, 0, 0, 0, 2, 5),
        (0, -np.pi, 0, 0, 0, 0, 0),
        (0, -np.pi, 0, 0, 0, 0, 5),
        (0, -np.pi, 0, 2, 0, 0, 0),
        (0, -np.pi, 0, 2, 0, 0, 5),
        (0, -np.pi, 0, 0, 2, 0, 0),
        (0, -np.pi, 0, 0, 2, 0, 5),
        (0, -np.pi, 0, 0, 0, 2, 0),
        (0, -np.pi, 0, 0, 0, 2, 5),
        (0, np.pi, 0, 0, 0, 0, 0),
        (0, np.pi, 0, 0, 0, 0, 5),
        (0, np.pi, 0, 2, 0, 0, 0),
        (0, np.pi, 0, 2, 0, 0, 5),
        (0, np.pi, 0, 0, 2, 0, 0),
        (0, np.pi, 0, 0, 2, 0, 5),
        (0, np.pi, 0, 0, 0, 2, 0),
        (0, np.pi, 0, 0, 0, 2, 5),
        (0, 0, -np.pi, 0, 0, 0, 0),
        (0, 0, -np.pi, 0, 0, 0, 5),
        (0, 0, -np.pi, 2, 0, 0, 0),
        (0, 0, -np.pi, 2, 0, 0, 5),
        (0, 0, -np.pi, 0, 2, 0, 0),
        (0, 0, -np.pi, 0, 2, 0, 5),
        (0, 0, -np.pi, 0, 0, 2, 0),
        (0, 0, -np.pi, 0, 0, 2, 5),
        (0, 0, np.pi, 0, 0, 0, 0),
        (0, 0, np.pi, 0, 0, 0, 5),
        (0, 0, np.pi, 2, 0, 0, 0),
        (0, 0, np.pi, 2, 0, 0, 5),
        (0, 0, np.pi, 0, 2, 0, 0),
        (0, 0, np.pi, 0, 2, 0, 5),
        (0, 0, np.pi, 0, 0, 2, 0),
        (0, 0, np.pi, 0, 0, 2, 5),
        (np.pi - 1, np.sqrt((2 * np.pi) - 1), 0, 0, 0, 0, 0),
        (np.pi - 1, np.sqrt((2 * np.pi) - 1), 0, 0, 0, 0, 5),
        (np.pi - 1, np.sqrt((2 * np.pi) - 1), 0, 2, 0, 0, 0),
        (np.pi - 1, np.sqrt((2 * np.pi) - 1), 0, 2, 0, 0, 5),
        (np.pi - 1, np.sqrt((2 * np.pi) - 1), 0, 0, 2, 0, 0),
        (np.pi - 1, np.sqrt((2 * np.pi) - 1), 0, 0, 2, 0, 5),
        (np.pi - 1, np.sqrt((2 * np.pi) - 1), 0, 0, 0, 2, 0),
        (np.pi - 1, np.sqrt((2 * np.pi) - 1), 0, 0, 0, 2, 5),
        (np.pi - 1, 0, np.sqrt((2 * np.pi) - 1), 0, 0, 0, 0),
        (np.pi - 1, 0, np.sqrt((2 * np.pi) - 1), 0, 0, 0, 5),
        (np.pi - 1, 0, np.sqrt((2 * np.pi) - 1), 2, 0, 0, 0),
        (np.pi - 1, 0, np.sqrt((2 * np.pi) - 1), 2, 0, 0, 5),
        (np.pi - 1, 0, np.sqrt((2 * np.pi) - 1), 0, 2, 0, 0),
        (np.pi - 1, 0, np.sqrt((2 * np.pi) - 1), 0, 2, 0, 5),
        (np.pi - 1, 0, np.sqrt((2 * np.pi) - 1), 0, 0, 2, 0),
        (np.pi - 1, 0, np.sqrt((2 * np.pi) - 1), 0, 0, 2, 5),
        (0, np.pi - 1, np.sqrt((2 * np.pi) - 1), 0, 0, 0, 0),
        (0, np.pi - 1, np.sqrt((2 * np.pi) - 1), 0, 0, 0, 5),
        (0, np.pi - 1, np.sqrt((2 * np.pi) - 1), 2, 0, 0, 0),
        (0, np.pi - 1, np.sqrt((2 * np.pi) - 1), 2, 0, 0, 5),
        (0, np.pi - 1, np.sqrt((2 * np.pi) - 1), 0, 2, 0, 0),
        (0, np.pi - 1, np.sqrt((2 * np.pi) - 1), 0, 2, 0, 5),
        (0, np.pi - 1, np.sqrt((2 * np.pi) - 1), 0, 0, 2, 0),
        (0, np.pi - 1, np.sqrt((2 * np.pi) - 1), 0, 0, 2, 5),
        (rand_omega(), rand_omega(), rand_omega(), 0, 0, 0, 0),
        (rand_omega(), rand_omega(), rand_omega(), 0, 0, 0, 5),
        (rand_omega(), rand_omega(), rand_omega(), 2, 0, 0, 0),
        (rand_omega(), rand_omega(), rand_omega(), 2, 0, 0, 5),
        (rand_omega(), rand_omega(), rand_omega(), 0, 2, 0, 0),
        (rand_omega(), rand_omega(), rand_omega(), 0, 2, 0, 5),
        (rand_omega(), rand_omega(), rand_omega(), 0, 0, 2, 0),
        (rand_omega(), rand_omega(), rand_omega(), 0, 0, 2, 5),
    ]
    test_vec += zip(rand_omega(N), rand_omega(N), rand_omega(N),
                    rand_v(N), rand_v(N), rand_v(N), rand_t(N))
    for omega1, omega2, omega3, v1, v2, v3, t in test_vec:
        vec = np.array([v1, v2, v3, omega1, omega2, omega3])
        mat = np.array([[0, -omega3, omega2, v1],
                        [omega3, 0, -omega1, v2],
                        [-omega2, omega1, 0, v3],
                        [0, 0, 0, 0]])
        group = expm(mat)
        lie_algebra = se3(so3(omega1, omega2, omega3), e3(v1, v2, v3))
        lie_vec = lie_algebra.to_vec()
        lie_mat = lie_algebra.to_mat()
        lie_group = lie_algebra.to_group()
        assert np.all(lie_vec == vec)
        assert np.all(lie_mat == mat)
        assert np.all(np.isclose(lie_group.mat, group))
        exp_ = expm(mat * t)
        lie_exp_ = lie_exp(lie_algebra, t)
        assert np.all(np.isclose(lie_exp_.mat, exp_))
        if 0 != t:
            lie_log_ = lie_log(lie_exp_, t)
            lie_exp_recon = lie_exp(lie_log_, t)
            assert np.all(np.isclose(lie_exp_recon.mat, exp_))
    print('All tests suceeded')

