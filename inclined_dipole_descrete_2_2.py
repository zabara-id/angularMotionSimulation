import numpy as np
import matplotlib.pyplot as plt


# Класс кватернионов
class Quaternion:
    """класс кватернионов
    """

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
    """функция перевода из кеплеровых элементов в (r, v)

    Args:
        orbit (np.ndarray): [a, e, i, omega, Omega]; полуось в метрах 
        nu (float): истинная аномалия

    Returns:
        np.ndarray: (r, v); радиус-вектор в километрах
    """
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


def quaternion_derivative(q, omega):
    """уравнение Пуассона для вращения твёрдого тела
    """
    omega_quat = Quaternion(0, omega[0], omega[1], omega[2])
    dq_dt = 0.5 * q * omega_quat
    return dq_dt


def quattrans(q: Quaternion, vect:np.ndarray) -> np.ndarray:
    """функция перехода в новую СК при помощи кватерниона

    Args:
        q (Quaternion): кватернион ориентации
        vect (np.array): преобразуемый вектор

    Returns:
        _type_: вектор в новой СК
    """
    x_vect, y_vect, z_vect = vect

    q_ = q.conjugate()
    v_q = Quaternion(0, x_vect, y_vect, z_vect)

    temp1 = q_ * v_q
    temp2 = temp1 * q

    return np.array([temp2.x, temp2.y, temp2.z])


def magnitometr_inclined(state: np.ndarray, q: Quaternion, dt, num_step, mu_e=7.812e6) -> np.ndarray:
    """функцция получения магнитного поля в модели наклонного диполя

    Args:
        state (np.ndarray): положение (r, v)
        q (Quaternion): кватернион ИСК->ССК
        dt (_type_): шаг по времени
        num_step (_type_): номер шага
        mu_e (_type_, optional): магнитный паарметр планеты. Defaults to 7.812e6.

    Returns:
        np.ndarray: магнитное поле в модели наклонного диполя
    """

    r = state[:3]
    nr = np.linalg.norm(r)

    OMEGA_E = 7.292115e-5  # угловая ск-ть вращения Земли
    delta = np.deg2rad(168.)
    lmbd0 = np.deg2rad(70.)
    lmbd = OMEGA_E * num_step * dt + lmbd0

    k_vec_IF = np.array([np.sin(delta)*np.cos(lmbd), np.sin(delta)*np.sin(lmbd), np.cos(delta)])

    B_IF = -mu_e / nr**5 * (nr**2 * k_vec_IF - 3*np.dot(k_vec_IF, r)*r)

    B_BF = quattrans(q, B_IF)

    return B_BF


def RS_bdot(state: np.ndarray, q, omega: np.ndarray, B_prev: np.ndarray, moment_prev: np.ndarray,
            dt, num_step, mu, I, k) -> tuple:
    """функция правых частей с алгоритмом Bdot

    Args:
        state (np.ndarray): (r, v)
        q (_type_): кватернион ИСК -> ССК
        omega (np.ndarray): абсолютная угловая скорость в ССК
        B_prev (np.ndarray): поле с предыдущего шага
        moment_prev (np.ndarray): момент с предыдущего шага
        dt (_type_): шаг по времени
        num_step (_type_): номер шага
        mu (_type_): гравитационный параметр планеты
        I (_type_): тензор инерции КА
        k (_type_): множитель магнитного момента

    Returns:
        tuple: (dstate_dt, dq_dt, domega_dt, moment)
    """
    r = state[:3]
    nr = np.linalg.norm(r)
    v = state[3:]
    B_now = magnitometr_inclined(state, q, dt, num_step)
    if num_step % 2 == 0:
        moment = -k *  (B_now - B_prev) / dt
    else:
        moment = moment_prev


    # # Обрезка компонент, ограничения управления (если нужно)
    # if not (np.abs(moment)<1).all():
    #     moment_limited = np.where(np.abs(moment)<1, moment, np.ones(np.shape(moment)[0])*np.sign(moment))
    #     moment = moment_limited

    # # Ограничение нормы
    # if np.linalg.norm(moment)>1:
    #     moment_limited = moment / np.linalg.norm(moment) * 1
    #     moment = moment_limited

    dstate_dt = np.concatenate((v, -mu/nr**3 * r))

    dq_dt = quaternion_derivative(q, omega)

    r_BF = quattrans(q, r)  # переход R_c в связную СК
    I_omega = I @ omega
    omega_cross = np.cross(omega, I_omega)
    grav_moment = 3*mu/np.linalg.norm(r_BF)**5 * np.cross(r_BF, I @ r_BF)
    mag_moment = np.cross(moment, B_now)
    domega_dt = np.linalg.inv(I) @ (grav_moment + mag_moment - omega_cross) 

    return dstate_dt, dq_dt, domega_dt, moment


def RK4step_bdot(state, q, omega, dt, B_prev, moment_prev, num_step, mu=398600.4415, I=np.diag([2, 3, 4]), k=1e8):

    kstate1, kq1, komega1, _ = RS_bdot(state, q, omega, B_prev, moment_prev, dt, num_step, mu, I, k)
    kstate2, kq2, komega2, _ = RS_bdot(state+kstate1*dt/2, q+dt/2 * kq1, omega + komega1*dt/2, B_prev, moment_prev, dt, num_step, mu, I, k)
    kstate3, kq3, komega3, _ = RS_bdot(state+kstate2*dt/2, q+dt/2 * kq2, omega + komega2*dt/2, B_prev, moment_prev, dt, num_step, mu, I, k)
    kstate4, kq4, komega4, _ = RS_bdot(state+kstate3*dt, q+dt * kq3, omega+komega3*dt, B_prev, moment_prev, dt, num_step, mu, I, k)

    state_next = state + (kstate1 + 2*kstate2 + 2*kstate3 + kstate4) * dt / 6
    
    q_next = q + dt / 6 * (kq1 + 2*kq2 + 2*kq3 + kq4)
    qnn =  q_next.norm
    q_next = q_next / qnn

    omega_next = omega + (komega1 + 2*komega2 + 2*komega3 + komega4) * dt / 6

    # B и m на шаге сохраняем для учета дискретности
    B_next = magnitometr_inclined(state_next, q_next, dt, num_step)
    m = RS_bdot(state, q, omega, B_prev, moment_prev, dt, num_step, mu, I, k)[3]

    return  state_next, q_next, omega_next, B_next, m


# Шаг времени
dt = 0.1

# Кол-во шагов
num_steps = 10000


rv = elem2rv(np.array([7000., 1e-5, np.deg2rad(45), 0., np.deg2rad(7)]), nu=0.6)
initial_r = rv[:3]
initial_v = rv[3:]

state, q  = np.concatenate((initial_r, initial_v)), Quaternion(1, 1, 2, 5)
q = 1 / q.norm * q
omega = np.array([0.1, 0.2, 0.03]) + np.cross(quattrans(q, initial_r), quattrans(q, initial_v))/np.linalg.norm(initial_r)**2
B = magnitometr_inclined(state, q, dt, 0)
m = np.array([0., 0., 0.])


# Интегрирование с использованием динамического уравнения Эйлера
omega_trajectory = [omega]
quaternion_trajectory = [q]  # Траектория кватернионов
moment_trajectory = [m]
temp = 0

# Интегрироваие Bdot
for t in np.arange(0, num_steps * dt, dt):
    state, q, omega, B_next, m_next = RK4step_bdot(state, q, omega, dt, B, m, temp, k=5e8)
    temp += 1
    # print(omega[2])
    # print(f'B = {np.linalg.norm(B)} Tl')
    omega_trajectory.append(omega)
    quaternion_trajectory.append(q)
    moment_trajectory.append(m_next)
    # print(m)
    B = B_next
    m = m_next


t = np.concatenate((np.array([0]), np.arange(0, num_steps * dt, dt)), axis=0)

# Графики компонент угловой скорости
omega_array = np.array(omega_trajectory)
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


# Графики компонент момента <=> тока на катушках
moments_array = np.array(moment_trajectory)
moments_array_normed = moments_array / np.max(moments_array)
plt.figure(figsize=(12, 6))
plt.subplot(3, 1, 1)
plt.plot(t, moments_array[:,0], 'b-')
plt.ylabel(r"$I_1$")
plt.subplot(3, 1, 2)
plt.plot(t, moments_array[:,1], 'g-')
plt.ylabel(r"$I_2$")
plt.subplot(3, 1, 3)
plt.plot(t, moments_array[:,2], 'k-')
plt.ylabel(r"$I_3$")

plt.xlabel("Время (с)")
plt.show()