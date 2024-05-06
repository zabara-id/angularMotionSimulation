import numpy as np
import matplotlib.pyplot as plt
from pyquaternion import Quaternion as Quat


# Класс кватернионов
class Quaternion:
    def __init__(self, w, x, y, z):
        self.w = w
        self.x = x
        self.y = y
        self.z = z

    def __mul__(self, other):
        w1, x1, y1, z1 = self.w, self.x, self.y, self.z
        w2, x2, y2, z2 = other.w, other.x, other.y, other.z
        return Quaternion(
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2,
            w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
        )
    
    def __truediv__(self, scalar):
        return Quaternion(
            self.w / scalar,
            self.x / scalar,
            self.y / scalar,
            self.z / scalar
        )

    def __add__(self, other):
        return Quaternion(
            self.w + other.w,
            self.x + other.x,
            self.y + other.y,
            self.z + other.z
        )

    def __rmul__(self, scalar):
        return Quaternion(
            scalar * self.w,
            scalar * self.x,
            scalar * self.y,
            scalar * self.z
        )

    def to_euler_angles(self):
        # Roll (вращение вокруг оси x)
        sinr_cosp = 2 * (self.w * self.x + self.y * self.z)
        cosr_cosp = 1 - 2 * (self.x * self.x + self.y * self.y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        # Pitch (вращение вокруг оси y)
        sinp = 2 * (self.w * self.y - self.z * self.x)
        pitch = np.arcsin(sinp)

        # Yaw (вращение вокруг оси z)
        siny_cosp = 2 * (self.w * self.z + self.x * self.y)
        cosy_cosp = 1 - 2 * (self.y * self.y + self.z * self.z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw
    

    def conjugate(self):
        return Quaternion(self.w, -self.x, -self.y, -self.z)
    
    def noise(self):
        return Quaternion()

    @property
    def norm(self):
        return np.sqrt(self.w**2 + self.x**2 + self.y**2 + self.z**2)
    
    @property
    def to_vec(self):
        return np.array([self.x, self.y, self.z])



def elem2rv(orbit: np.ndarray, nu: float) -> np.ndarray:
    mu = 398600.4415e9  # m^3/s^2
    p = orbit[0] * (1 - orbit[1]**2)
    E = 2 * np.arctan(np.sqrt((1 - orbit[1]) / (1 + orbit[1])) * np.tan(nu / 2))  # эксцентрическая аномалия
    b = orbit[0] * np.sqrt(1-orbit[1]**2)
    plane_coords = np.array([orbit[0] * (np.cos(E)-orbit[1]), b * np.sin(E), 0.])  # координаты в орбитальной плоскости
    plane_velocity = np.sqrt(mu / p) * np.array([-np.sin(nu), orbit[1]+np.cos(nu), 0.])

    A1 = np.array([[np.cos(orbit[4]), np.sin(orbit[4]), 0.],
                  [-np.sin(orbit[4]), np.cos(orbit[4]), 0.],
                  [0., 0., 1]])

    A2 = np.array([[1., 0., 0.],
                  [0., np.cos(orbit[2]), np.sin(orbit[2])],
                  [0., -np.sin(orbit[2]), np.cos(orbit[2])]])

    A3 = np.array([[np.cos(orbit[3]), np.sin(orbit[3]), 0.],
                  [-np.sin(orbit[3]), np.cos(orbit[3]), 0.],
                  [0., 0., 1]])

    B = A1.T @ A2.T @ A3.T  # результирующая матрица поворота
    coords = B @ plane_coords
    velocity = B @ plane_velocity

    return np.concatenate([coords, velocity])


def from_euler_angles(roll, pitch, yaw):
    cr = np.cos(roll / 2)
    sr = np.sin(roll / 2)
    cp = np.cos(pitch / 2)
    sp = np.sin(pitch / 2)
    cy = np.cos(yaw / 2)
    sy = np.sin(yaw / 2)

    # Расчет кватерниона
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return Quaternion(w, x, y, z)


def quaternion_derivative(q, omega):
    omega_quat = Quaternion(0, omega[0], omega[1], omega[2])
    dq_dt = 0.5 * q * omega_quat
    return dq_dt


def quattrans(q: Quaternion, vect:np.array):
    # w_q, x_q, y_q, z_q = q.w, q.x, q.y, q.z
    x_vect, y_vect, z_vect = vect

    q_ = q.conjugate()
    v_q = Quaternion(0, x_vect, y_vect, z_vect)

    temp1 = q_ * v_q
    temp2 = temp1 * q

    return np.array([temp2.x, temp2.y, temp2.z])


def matrix_to_quaternion(R: np.ndarray) -> Quaternion:
    w = 0.5 * np.sqrt(1 + R[0, 0] + R[1, 1] + R[2, 2])
    x = np.sign(R[2, 1] - R[1, 2]) * 0.5 * np.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2])
    y = np.sign(R[0, 2] - R[2, 0]) * 0.5 * np.sqrt(1 - R[0, 0] + R[1, 1] - R[2, 2])
    z = np.sign(R[1, 0] - R[0, 1]) * 0.5 * np.sqrt(1 - R[0, 0] - R[1, 1] + R[2, 2])
    return Quaternion(w, x, y, z)


def quaternion_to_matrix(q: Quaternion) -> np.ndarray:
    w, x, y, z = q.w, q.x, q.y, q.z

    q_vec = np.array([q.x, q.y, q.z])
    q_x_matrix = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])

    return ((2*w**2 - 1) * np.eye(3) - 2*w*q_x_matrix + 2*np.outer(q_vec, q_vec)).T


def aero(v, rho=1e-13, eps=0.1, u_div_v=0.1, rc=np.array([0.1, 0.2, 0.3]), n = np.array([0., 0., 1.]), S=1):
    ev = v / np.linalg.norm(v)
    evn = np.dot(ev, n)
    if evn < 0:
        n *= -1

    evn = np.abs(evn)
    I1, I2, I3 = evn * S * ev, evn**2 * S * n, evn * S * n
    J1, J2, J3 = evn * S * np.cross(ev, rc), evn**2 * S * np.cross(n, rc), evn * S * np.cross(n, rc)

    F_aero = -rho * np.linalg.norm(v)**2 * ((1-eps)*I1 + 2*eps*I2 + (1-eps)*u_div_v*I3)
    M_aero = -rho * np.linalg.norm(v)**2 * ((1-eps)*J1 + 2*eps*J2 + (1-eps)*u_div_v*J3)
    
    return F_aero, M_aero


def control(state, q, omega, W_ref_IF_prev, dt, mu, I, Kw=1, Kq=1):
    r, v = state[:3], state[3:]
    r_BF = quattrans(q, r)

    M_grav = 3 * mu / (np.linalg.norm(r_BF)**5) * np.cross(r_BF, I @ r_BF)
    # M_aero = quattrans(q, aero(v)[1])
    M_aero = aero(quattrans(q,v))[1]
    
    M_ext = M_grav

    e3 = r / np.linalg.norm(r)
    e2 = np.cross(r, v) / np.linalg.norm(np.cross(r, v))
    e1 = np.cross(e2, e3)
    K_matrix = np.array([e1, e2, e3]).T

    B = matrix_to_quaternion(K_matrix)
    
    W_ref_OSK = np.array([0., np.linalg.norm(np.cross(quattrans(B, r), quattrans(B, v))) / np.linalg.norm(quattrans(B, r))**2, 0.])
    A = B.conjugate() * q

    if A.w < 0:
        A = Quaternion(-A.w, -A.x, -A.y, -A.z)

    W_rel_BF = omega - quattrans(A, W_ref_OSK)

    W_ref_IF_now = np.cross(r, v) / (np.linalg.norm(r)**2)
    dW_ref = (W_ref_IF_now - W_ref_IF_prev) / dt

    dW_ref = quattrans(q, dW_ref)

    u = -M_ext + np.cross(omega, I @ omega) - I @ np.cross(W_rel_BF, quattrans(A,  W_ref_OSK)) + I @ dW_ref  - Kw * W_rel_BF - Kq* A.to_vec
    # u = - Kw * W_rel_BF - Kq* A.to_vec
    # print('M_ext = ', M_ext)

    return u


def RS(state, q, omega, W_ref_IF_prev, dt, mu, I):
    r = state[:3]
    nr = np.linalg.norm(r)
    v = state[3:]

    F_aero_IF, M_aero = aero(quattrans(q,v))  # сила и момент в CСК

    # M_aero = quattrans(q, M_aero)  # перевели момент в ССК

    F_aero_IF = quattrans(q.conjugate(), F_aero_IF)
    
    dstate_dt = np.concatenate((v, -mu/nr**3 * r + F_aero_IF))  # без аеро

    dq_dt = quaternion_derivative(q, omega)

    r_BF = quattrans(q, r)  # переход R_c в связную СК

    grav_moment = 3*mu/np.linalg.norm(r_BF)**5 * np.cross(r_BF, I @ r_BF)

    u = control(state, q, omega, W_ref_IF_prev, dt, mu, I)

    domega_dt = np.linalg.inv(I) @ (grav_moment + M_aero - np.cross(omega, I @ omega) + u)

    return dstate_dt, dq_dt, domega_dt


def RK4step(state, q, omega, dt, W_ref_IF_prev, mu=398600.4415e9, I=np.diag([2, 3, 4])):

    kstate1, kq1, komega1 = RS(state, q, omega, W_ref_IF_prev, dt, mu, I)
    kstate2, kq2, komega2 = RS(state+kstate1*dt/2, q+dt/2 * kq1, omega + komega1*dt/2, W_ref_IF_prev, dt, mu, I)
    kstate3, kq3, komega3 = RS(state+kstate2*dt/2, q+dt/2 * kq2, omega + komega2*dt/2, W_ref_IF_prev, dt, mu, I)
    kstate4, kq4, komega4 = RS(state+kstate3*dt, q+dt * kq3, omega+komega3*dt, W_ref_IF_prev, dt, mu, I)

    state_next = state + (kstate1 + 2*kstate2 + 2*kstate3 + kstate4) * dt / 6
    
    q_next = q + dt / 6 * (kq1 + 2*kq2 + 2*kq3 + kq4)
    qnn =  q_next.norm
    q_next = q_next / qnn

    omega_next = omega + (komega1 + 2*komega2 + 2*komega3 + komega4) * dt / 6

    W_ref_IF_next = np.cross(state_next[:3], state_next[3:]) / np.linalg.norm(state_next[:3])**2

    return  state_next, q_next, omega_next, W_ref_IF_next


# Шаг времени
dt = 0.1

# Кол-во шагов
num_steps = 1000


rv = elem2rv(np.array([6871.e3, 0., 0.9, 1.37, 0.7]), nu=np.pi/2)
initial_r = rv[:3]
initial_v = rv[3:]

state, q  = np.concatenate((initial_r, initial_v)), Quaternion(1, 5, 2, 0.01)
q = 1 / q.norm * q
omega = np.array([0.46, 0.67, 0.045]) + np.cross(quattrans(q, initial_r), quattrans(q, initial_v))/np.linalg.norm(quattrans(q, initial_r))**2
W_ref_IF = np.cross(initial_r, initial_v) / (np.linalg.norm(initial_r)**2)

omega_trajectory = []
quaternion_trajectory = [q]  # Траектория кватернионов ИСК -> ССК
r_trajectory = []

# Тракектория каждой компоненты кватерниона ОСК -> ССК
Aw_trajectory = []
Ax_trajectory = []
Ay_trajectory = []
Az_trajectory = []

# Интегрироваие 3-осное
for t in np.arange(0, num_steps * dt, dt):
    state, q, omega, W_ref_IF_next = RK4step(state, q, omega, dt, W_ref_IF)
    r, v = state[:3], state[3:]
    # print('r norm = ', np.linalg.norm(r))
    r_trajectory.append(np.linalg.norm(r))
    
    e3 = r / np.linalg.norm(r)
    e2 = np.cross(r, v) / np.linalg.norm(np.cross(r, v))
    e1 = np.cross(e2, e3)
    K_matrix = np.array([e1, e2, e3]).T
    B = matrix_to_quaternion(K_matrix)
    if B.w < 0:
        B = Quaternion(-B.w, -B.x, -B.y, -B.z)

    A = B.conjugate() * q
    Aw_trajectory.append(A.w)
    Ax_trajectory.append(A.x)
    Ay_trajectory.append(A.y)
    Az_trajectory.append(A.z)

    omega_trajectory.append(omega)
    # quaternion_trajectory.append(q)
    W_ref_IF = W_ref_IF_next


t = np.arange(0, num_steps * dt, dt)

# print(t.shape)
# print(np.array(Aw_trajectory).shape)

# print(t[:50])
# print(Aw_trajectory[:50])


# plt.figure(figsize=(12, 6))
# plt.title("Кватернион ОСК -> ССК во времени")
# plt.subplot(4, 1, 1)
# plt.plot(t, Aw_trajectory, 'b-')
# plt.ylabel("A.w")
# plt.subplot(4, 1, 2)
# plt.plot(t, Ax_trajectory, 'g-')
# plt.ylabel("A.x")
# plt.subplot(4, 1, 3)
# plt.plot(t, Ay_trajectory, 'r-')
# plt.ylabel("A.y")
# plt.subplot(4, 1, 4)
# plt.plot(t, Az_trajectory, 'k-')
# plt.ylabel("A.z")
# plt.xlabel("Время (с)")

plt.plot(t, Aw_trajectory, label='w')
plt.plot(t, Ax_trajectory, label='ix')
plt.plot(t, Ay_trajectory, label='iy')
plt.plot(t, Az_trajectory, label='iz')
plt.xlabel("Время, сек")
plt.grid(True)
plt.legend()
plt.show()