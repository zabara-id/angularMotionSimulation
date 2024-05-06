import numpy as np
import matplotlib.pyplot as plt


class Quaternion:
    """класс кватернионов
    """

    def __init__(self, w, x, y, z):
        self.w = w
        self.x = x
        self.y = y
        self.z = z

    def __str__(self):
        return f"{self.w} + {self.x}i + {self.y}j + {self.z}k"

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
    """функция перевода из кеплеровых элементов в (r, v)

    Args:
        orbit (np.ndarray): [a, e, i, omega, Omega]; полуось в метрах 
        nu (float): истинная аномалия

    Returns:
        np.ndarray: (r, v); радиус-вектор в километрах
    """
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


def matrix_to_quaternion(R: np.ndarray) -> Quaternion:
    """функция сопоставляет матрице один из 2-х возможных кватернионов
    """
    w = 0.5 * np.sqrt(1 + R[0, 0] + R[1, 1] + R[2, 2])
    x = np.sign(R[2, 1] - R[1, 2]) * 0.5 * np.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2])
    y = np.sign(R[0, 2] - R[2, 0]) * 0.5 * np.sqrt(1 - R[0, 0] + R[1, 1] - R[2, 2])
    z = np.sign(R[1, 0] - R[0, 1]) * 0.5 * np.sqrt(1 - R[0, 0] - R[1, 1] + R[2, 2])
    return Quaternion(w, x, y, z)


def quaternion_to_matrix(q: Quaternion) -> np.ndarray:
    """функция сопоставляет кватерниону матрицу
    """
    w, x, y, z = q.w, q.x, q.y, q.z

    q_vec = np.array([q.x, q.y, q.z])
    q_x_matrix = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])

    return ((2*w**2 - 1) * np.eye(3) - 2*w*q_x_matrix + 2*np.outer(q_vec, q_vec)).T


def aero(v: np.ndarray, rho=1e-13, eps=0.1, u_div_v=0.1,
         rc=np.array([0.1, 0.2, 0.3]), n = np.array([0., 0., 1.]), S=1.) -> tuple:
    """функция вычисления аэродинамического момента и силы для пластины

    Args:
        v (np.ndarray): скорость аппарата в ССК
        rho (_type_, optional): плотность атмосферы. Defaults to 1e-13.
        eps (float, optional): коэффициент отражения. Defaults to 0.1.
        u_div_v (float, optional): коэффициент проникновения. Defaults to 0.1.
        rc (_type_, optional): положение пластины. Defaults to np.array([0.1, 0.2, 0.3]).
        n (_type_, optional): вектор нормали пластины. Defaults to np.array([0., 0., 1.]).
        S (int, optional): площадь пластины в м^2. Defaults to 1.

    Returns:
        tupple: (F_aero, M_aero)
    """
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


def omega_sensor(omega: np.ndarray, num_step, a1:float=1, a2:float=1, a3:float=1) -> np.ndarray:
    """функция для имитации измерений угловой скорости с добавлением шума

    Args:
        omega (np.ndarray): вектор угловой скорости [w1, w2, w3] в рад/сек
        a1 (float, optional): коэффициент усиления для первой компоненты угловой скорости. Defaults to 0.99.
        a2 (float, optional): коэффициент усиления для второй компоненты угловой скорости. Defaults to 1.01.
        a3 (float, optional): коэффициент усиления для третьей компоненты угловой скорости. Defaults to 0.98.

    Returns:
        np.ndarray: измерение ДУС с шумом
    """
    np.random.seed(num_step)
    w1, w2, w3 = omega
    adjusted_omega = np.array([a1 * w1, a2 * w2, a3 * w3])
    noise = np.random.normal(0, 5e-3, 3)
    return adjusted_omega + noise


def star_senqor(q: Quaternion, S_real: Quaternion, num_step) -> Quaternion:
    """функция для имитации измерений звёздного датчика

    Args:
        q (Quaternion): реальное значение кватерниона ориентации
        S_real (Quaternion): значение кватерниона S_real (задаётся однократно вместе с НУ)

    Returns:
        Quaternion: измерение звёздного датчика
    """
    np.random.seed(num_step)
    S_nominal = Quaternion(1, 0, 0, 0)

    dlmbd_vec = np.array([
        np.random.normal(0, 3600*4.85e-6),  # ошибка 10" для первой оси
        np.random.normal(0, 3600*4.85e-6),  # ошибка 10" для второй оси
        np.random.normal(0, 3600*4.85e-6)   # ошибка 50" для третьей оси
    ])

    dlmbd_scalar = np.sqrt(1-np.linalg.norm(ds_real_vec)**2)

    dLMBD = Quaternion(dlmbd_scalar, dlmbd_vec[0], dlmbd_vec[1], dlmbd_vec[2])

    return q * S_real * dLMBD * S_nominal.conjugate()


def control(state: np.ndarray, q: Quaternion, omega: np.ndarray, W_ref_IF_prev: np.ndarray, dt, mu, I, Kw=1, Kq=1) -> np.ndarray:
    """функция идеального 3-х осного Ляпуновского управления (орбитальная стабилизация)

    Args:
        state (np.ndarray): (r,v); заданно в СИ
        q (Quaternion): кватернион ИСК -> ССК
        omega (np.ndarray): абсолютная угловая скорость в ССК
        W_ref_IF_prev (np.ndarray): заданная угловая скорость с предыдущего шага для подсчёта производной
        dt (_type_): шаг интегрирования
        mu (_type_): гравитационный параметр планеты
        I (_type_): тензор инерции КА
        Kw (int, optional): коэффициент усиления по w. Defaults to 1.
        Kq (int, optional): коэффициент усиления по q. Defaults to 1.

    Returns:
        np.ndarray: управляющее воздействие
    """
    r, v = state[:3], state[3:]
    r_BF = quattrans(q, r)

    M_grav = 3 * mu / (np.linalg.norm(r_BF)**5) * np.cross(r_BF, I @ r_BF)
    M_aero = aero(quattrans(q,v))[1]
    
    M_ext = M_grav # + M_aero

    # Получаем кватернион перехода ИСК -> ОСК
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

    return u, W_rel_BF


def RS(state: np.ndarray, q: np.ndarray, omega: np.ndarray, H: np.ndarray, W_ref_IF_prev: np.ndarray, 
       dt, mu, I, S_real, num_step, u_prev) -> tuple:
    """функция правых частей с учётом маховиков

    Args:
        state (np.ndarray): (r,v); заданно в СИ
        q (Quaternion): кватернион ИСК -> ССК
        omega (np.ndarray): абсолютная угловая скорость в ССК
        H (np.ndarray): угловой момент маховиков
        W_ref_IF_prev (np.ndarray): заданная угловая скорость с предыдущего шага для подсчёта производной
        dt (_type_): шаг интегрирования
        mu (_type_): гравитационный параметр планеты
        I (_type_): тензор инерции КА
        S_real (_type_): заданный кватернион ОСК(ИСК) -> ДСК
        num_step (_type_): номер шага (для учёта дискретности управления)
        u_prev (_type_): управляющее воздействие с предыдущего шага (для учёта дискретности управления)

    Returns:
        tuple: (dstate_dt, dq_dt, domega_dt, dH, u, W_rel)
    """
    r = state[:3]
    nr = np.linalg.norm(r)
    v = state[3:]

    F_aero_IF, M_aero = aero(quattrans(q,v))  # сила и момент в CСК
    F_aero_IF, M_aero = np.zeros(3), np.zeros(3)  # сила и момент в CСК


    F_aero_IF = quattrans(q.conjugate(), F_aero_IF)  # сила в ИСК
    
    dstate_dt = np.concatenate((v, -mu/nr**3 * r + F_aero_IF))

    dq_dt = quaternion_derivative(q, omega)

    r_BF = quattrans(q, r)  # переход R_c в связную СК

    grav_moment = 3*mu/np.linalg.norm(r_BF)**5 * np.cross(r_BF, I @ r_BF)

    # В управление передаём уже данные с ДУС и звёздника, учитываем дискретность в 0.2 секунды
    if num_step % 2 == 0:
        u, W_rel = control(state, star_senqor(q, S_real, num_step), omega_sensor(omega, num_step), W_ref_IF_prev, dt, mu, I)
    else:
        u, W_rel = u_prev, control(state, star_senqor(q, S_real, num_step), omega_sensor(omega, num_step), W_ref_IF_prev, dt, mu, I)[1]

    domega_dt = np.linalg.inv(I) @ (grav_moment + M_aero - np.cross(omega, I @ omega) + u)

    dH = -u - np.cross(omega_sensor(omega, num_step), H)

    return dstate_dt, dq_dt, domega_dt, dH, u, W_rel


def RK4step(state: np.ndarray, q: Quaternion, omega: np.ndarray, H: np.ndarray, dt, W_ref_IF_prev: np.ndarray, S_real: Quaternion,
             num_step, u_prev, mu=398600.4415e9, I=np.diag([2, 3, 4])) -> tuple:
    """функция шага РК4

    Returns:
        tuple: (state_next, q_next, omega_next, H_next, W_ref_IF_next, u_next, W_rel_next)
    """

    kstate1, kq1, komega1, kH1, _, _  = RS(state, q, omega, H, W_ref_IF_prev, dt, mu, I, S_real, num_step, u_prev)
    kstate2, kq2, komega2, kH2, _, _ = RS(state+kstate1*dt/2, q+dt/2 * kq1, omega + komega1*dt/2, H+kH1*dt/2, W_ref_IF_prev, dt, mu, I, S_real, num_step, u_prev)
    kstate3, kq3, komega3, kH3, _, _ = RS(state+kstate2*dt/2, q+dt/2 * kq2, omega + komega2*dt/2, H+kH2*dt/2, W_ref_IF_prev, dt, mu, I, S_real, num_step, u_prev)
    kstate4, kq4, komega4, kH4, _, _ = RS(state+kstate3*dt, q+dt * kq3, omega+komega3*dt, H+kH3*dt, W_ref_IF_prev, dt, mu, I, S_real, num_step, u_prev)

    state_next = state + (kstate1 + 2*kstate2 + 2*kstate3 + kstate4) * dt / 6
    
    q_next = q + dt / 6 * (kq1 + 2*kq2 + 2*kq3 + kq4)
    qnn =  q_next.norm
    q_next = q_next / qnn

    omega_next = omega + (komega1 + 2*komega2 + 2*komega3 + komega4) * dt / 6

    H_next = H + (kH1 + 2*kH2 + 2*kH3 + kH4) * dt / 6

    W_ref_IF_next = np.cross(state_next[:3], state_next[3:]) / np.linalg.norm(state_next[:3])**2

    u_next, W_rel_next = RS(state_next, q_next, omega_next, H_next, W_ref_IF_prev, dt, mu, I, S_real, num_step, u_prev)[4:]

    return  state_next, q_next, omega_next, H_next, W_ref_IF_next, u_next, W_rel_next


# Шаг по времени, сек
dt = 0.1

# Кол-во шагов
num_steps = 5000

# Задание (r, v) через орбитальные элементы
rv = elem2rv(np.array([6871.e3, 0., 0.9, 1.37, 0.7]), nu=np.pi/2)
initial_r = rv[:3]
initial_v = rv[3:]

# Начальные условия для вектора состояния ((r0,v0); q0; omega0; H0)
state, q  = np.concatenate((initial_r, initial_v)), Quaternion(1, 5, 2, 0.01)
q = 1 / q.norm * q
omega = np.array([0.16, 0.27, 0.045]) + np.cross(quattrans(q, initial_r), quattrans(q, initial_v))/np.linalg.norm(initial_r)**2
H = np.zeros(3)
W_ref_IF = np.cross(initial_r, initial_v) / (np.linalg.norm(initial_r)**2)

# Однократное задание кватерниона S_real
ds_real_vec = np.array([
        np.random.normal(0, 100*4.85e-6),  # ошибка 10" для первой оси
        np.random.normal(0, 100*4.85e-6),  # ошибка 10" для второй оси
        np.random.normal(0, 500*4.85e-6)   # ошибка 50" для третьей оси
    ])

ds_real_vec = np.zeros(3)
ds_real_scalar = np.sqrt(1-np.linalg.norm(ds_real_vec)**2)
S_real = Quaternion(ds_real_scalar, ds_real_vec[0], ds_real_vec[1], ds_real_vec[2])
print(S_real)

omega_trajectory = []
quaternion_trajectory = [q]  # Траектория кватернионов ИСК -> ССК
r_trajectory = []

# Тракектория каждой компоненты кватерниона ОСК -> ССК
Aw_trajectory = []
Ax_trajectory = []
Ay_trajectory = []
Az_trajectory = []

# Траектория углового момента маховиков H
H_trajectory = []

# Траектория управления
u_trajectory = []

# Траектория относительной угловой скорсти W_rel
W_rel_trajectory = []

# Задание переменных для учёта дискретности управления
temp = 0
u_prev = np.array([0., 0., 0.,])

# Интегрироваие
for t in np.arange(0, num_steps * dt, dt):
    state, q, omega, H, W_ref_IF_next, u, W_rel = RK4step(state, q, omega, H, dt, W_ref_IF, S_real, temp, u_prev)
    
    # Обновление парметров для учёта дискретности управления
    u_prev = u
    temp += 1

    H_trajectory.append(H)
    u_trajectory.append(u)
    W_rel_trajectory.append(W_rel)

    r, v = state[:3], state[3:]
    r_trajectory.append(np.linalg.norm(r))
    
    # Получение кватерниона A
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


# Массив врмени моделирования
t = np.arange(0, num_steps * dt, dt)





# График относительного кватерниона ориентации A
plt.plot(t, Aw_trajectory, label='w')
plt.plot(t, Ax_trajectory, label='ix')
plt.plot(t, Ay_trajectory, label='jy')
plt.plot(t, Az_trajectory, label='kz')
plt.title("Относительный кватернион А")
plt.xlabel("Время, сек")
plt.grid(True)
plt.legend()
plt.show()

# График относительной угловой скорости
W_rel_trajectory_array = np.array(W_rel_trajectory)
plt.subplot(3, 1, 1)
plt.plot(t, W_rel_trajectory_array[:,0], 'b-')
plt.ylabel(r"$\omega_{relx}$")
plt.title(r"Относительная угловая скорость $\omega_{rel}$")
plt.grid(True)
plt.subplot(3, 1, 2)
plt.plot(t, W_rel_trajectory_array[:,1], 'g-')
plt.ylabel(r"$\omega_{rely}$")
plt.grid(True)
plt.subplot(3, 1, 3)
plt.plot(t, W_rel_trajectory_array[:,2], 'k-')
plt.ylabel(r"$\omega_{relz}$")
plt.xlabel("Время (с)")
plt.grid(True)
plt.show()

# График управляющего воздействия
u_trajectory_array = np.array(u_trajectory)
plt.subplot(3, 1, 1)
plt.plot(t, u_trajectory_array[:,0], 'b-')
plt.ylabel(r"$u_1$")
plt.title(r"Управляющее воздействие $u$")
plt.grid(True)
plt.subplot(3, 1, 2)
plt.plot(t, u_trajectory_array[:,1], 'g-')
plt.ylabel(r"$u_2$")
plt.grid(True)
plt.subplot(3, 1, 3)
plt.plot(t, u_trajectory_array[:,2], 'k-')
plt.ylabel(r"$u_3$")
plt.xlabel("Время (с)")
plt.grid(True)
plt.show()

# График угловой скорости маховиков
H_trajectory_array = np.array(H_trajectory)
I1 = 2e-3
I2 = 2.5e-3
I3 = 3e-3
plt.subplot(3, 1, 1)
plt.plot(t, H_trajectory_array[:,0] / I1, 'b-')
plt.ylabel(r"$\omega_{fw1}$")
plt.title(r"Угловая скорость маховиков $\omega_{fw}$")
plt.grid(True)
plt.subplot(3, 1, 2)
plt.plot(t, H_trajectory_array[:,1] / I2, 'g-')
plt.ylabel(r"$\omega_{fw2}$")
plt.grid(True)
plt.subplot(3, 1, 3)
plt.plot(t, H_trajectory_array[:,2] / I3, 'k-')
plt.ylabel(r"$\omega_{fw3}$")
plt.xlabel("Время (с)")
plt.grid(True)
plt.show()