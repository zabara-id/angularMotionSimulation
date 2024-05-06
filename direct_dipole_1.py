import numpy as np
import matplotlib.pyplot as plt


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

    @property
    def norm(self):
        return np.sqrt(self.w**2 + self.x**2 + self.y**2 + self.z**2)


def elem2rv(orbit: np.ndarray, nu: float) -> np.ndarray:
    mu = 398600.0  # km^3/s^2
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


# Производная кватерниона
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


def RS(state, q, omega, mu, I):
    r = state[:3]
    nr = np.linalg.norm(r)
    v = state[3:]
    dstate_dt = np.concatenate((v, -mu/nr**3 * r))

    dq_dt = quaternion_derivative(q, omega)

    r_BF = quattrans(q, r)  # переход R_c в связную СК
    I_omega = I * omega
    omega_cross = np.cross(omega, I_omega)
    grav_moment = 3*mu/np.linalg.norm(r_BF)**5 * np.cross(r_BF, I*r_BF)
    domega_dt = np.linalg.inv(I) @ (grav_moment - omega_cross) 

    return dstate_dt, dq_dt, domega_dt


def RK4step(state, q, omega, dt, mu=398600.4415, I=np.diag([3, 3, 1])):
    B = magnitometr_direct()
    kstate1, kq1, komega1 = RS(state, q, omega, mu, I)
    kstate2, kq2, komega2 = RS(state+kstate1*dt/2, q+dt/2 * kq1, omega + komega1*dt/2, mu, I)
    kstate3, kq3, komega3  = RS(state+kstate2*dt/2, q+dt/2 * kq2, omega + komega2*dt/2, mu, I)
    kstate4, kq4, komega4 = RS(state+kstate3*dt, q+dt * kq3, omega+komega3*dt, mu, I)

    state_next = state + (kstate1 + 2*kstate2 + 2*kstate3 + kstate4) * dt / 6
    
    q_next = q + dt / 6 * (kq1 + 2*kq2 + 2*kq3 + kq4)
    qnn =  q_next.norm
    q_next = q_next / qnn

    omega_next = omega + (komega1 + 2*komega2 + 2*komega3 + komega4) * dt / 6

    return  state_next, q_next, omega_next


def magnitometr_direct(state, q, mu_e=7.812e6):
    r = state[:3]
    nr = np.linalg.norm(r)
    
    k_vec = np.array([0, 0, -1])

    B_IF = -mu_e / nr**5 * (nr**2 * k_vec - 3*np.dot(k_vec, r)*r)

    B_BF = quattrans(q, B_IF)

    return B_BF


def RS_bdot(state, q, omega, B_prev, dt, mu, I, k):
    r = state[:3]
    nr = np.linalg.norm(r)
    v = state[3:]
    B_now = magnitometr_direct(state, q)
    moment = -k *  (B_now - B_prev) / dt
    dstate_dt = np.concatenate((v, -mu/nr**3 * r))

    dq_dt = quaternion_derivative(q, omega)

    r_BF = quattrans(q, r)  # переход R_c в связную СК
    I_omega = I @ omega
    omega_cross = np.cross(omega, I_omega)
    grav_moment = 3*mu/np.linalg.norm(r_BF)**5 * np.cross(r_BF, I @ r_BF)
    mag_moment = np.cross(moment, B_now)
    # print(mag_moment.shape)
    # print(mag_moment)
    # print((grav_moment - omega_cross).shape)
    # print(grav_moment - omega_cross)
    domega_dt = np.linalg.inv(I) @ (grav_moment + mag_moment - omega_cross) 

    return dstate_dt, dq_dt, domega_dt


def RK4step_bdot(state, q, omega, dt, B_prev, mu=398600.4415, I=np.diag([2., 2., 3.5]), k=1e8):
    
    kstate1, kq1, komega1 = RS_bdot(state, q, omega, B_prev, dt,  mu, I, k)
    kstate2, kq2, komega2 = RS_bdot(state+kstate1*dt/2, q+dt/2 * kq1, omega + komega1*dt/2, B_prev, dt,  mu, I, k)
    kstate3, kq3, komega3 = RS_bdot(state+kstate2*dt/2, q+dt/2 * kq2, omega + komega2*dt/2, B_prev, dt,  mu, I, k)
    kstate4, kq4, komega4 = RS_bdot(state+kstate3*dt, q+dt * kq3, omega+komega3*dt, B_prev, dt,  mu, I, k)

    state_next = state + (kstate1 + 2*kstate2 + 2*kstate3 + kstate4) * dt / 6
    
    q_next = q + dt / 6 * (kq1 + 2*kq2 + 2*kq3 + kq4)
    qnn =  q_next.norm
    q_next = q_next / qnn

    omega_next = omega + (komega1 + 2*komega2 + 2*komega3 + komega4) * dt / 6

    B_next = magnitometr_direct(state_next, q_next)

    return  state_next, q_next, omega_next, B_next


rv = elem2rv(np.array([7000., 1e-5, np.deg2rad(45), 0., np.deg2rad(7)]), nu=0.6)
initial_r = rv[:3]
initial_v = rv[3:]

print(initial_v)

# initial_r = np.array([0, 0, 6.8e3])
# initial_v = np.array([0, 8., 0])

state, q  = np.concatenate((initial_r, initial_v)), Quaternion(1, 1, 2, 5)
q = 1 / q.norm * q
omega = np.array([0.5, 0.5, 0.0005]) + np.cross(quattrans(q, initial_r), quattrans(q, initial_v))/np.linalg.norm(initial_r)**2
B = magnitometr_direct(state, q)

# Шаг времени
dt = 0.1

# Интегрирование с использованием динамического уравнения Эйлера
num_steps = 10000
omega_trajectory = [omega]
quaternion_trajectory = [q]  # Траектория кватернионов


# Интегрироваие Bdot
for t in np.arange(0, num_steps * dt, dt):
    state, q, omega, B_next = RK4step_bdot(state, q, omega, dt, B)
    # print(omega[2])
    # print(f'B = {np.linalg.norm(B)} Tl')
    omega_trajectory.append(omega)
    quaternion_trajectory.append(q)
    B = B_next


omega_array = np.array(omega_trajectory)
t = np.concatenate((np.array([0]), np.arange(0, num_steps * dt, dt)), axis=0)

# print(omega_array[:, 2])

# print(omega_array.shape)
# print(len(omega_array[0][0]))

plt.figure(figsize=(12, 6))
plt.subplot(3, 1, 1)
plt.plot(t, omega_array[:,0], 'b-')
plt.ylabel(r"$\omega_x$")
plt.subplot(3, 1, 2)
plt.plot(t, omega_array[:,1], 'g-')
plt.ylabel(r"$\omega_y$")
plt.subplot(3, 1, 3)
plt.plot(t, omega_array[:,2], 'k-')
plt.ylabel(r"$\omega_z$")

plt.xlabel("Время (с)")
plt.show()