import numpy as np
import matplotlib.pyplot as plt
# resultados reproducibles
SEMILLA = 2026
rng_global = np.random.default_rng(SEMILLA)

# ---------------------------------------------------------------------------
# Ejercicio a) Regla 184 de Wolfram
def paso_regla184(estado, regla=184):
    """
    Aplica un paso de una regla de Wolfram a un anillo binario.
    El vecindario (izquierda, centro, derecha) se codifica como
    4*izq + 2*centro + der. El nuevo estado es el bit de la regla
    en esa posición
    """
    L = len(estado)
    nuevo = np.zeros(L, dtype=int)

    for i in range(L):
        izq = estado[(i - 1) % L]
        cen = estado[i]
        der = estado[(i + 1) % L]
        indice = 4 * izq + 2 * cen + der
        nuevo[i] = (regla >> indice) & 1

    return nuevo

def simular_regla184(estado_inicial, T):
    """
    Simula T pasos de la Regla 184 y devuelve la matriz espacio-temporal
    La fila 0 corresponde al estado inicial
    """
    L = len(estado_inicial)
    M = np.zeros((T + 1, L), dtype=int)
    M[0] = estado_inicial

    estado = estado_inicial.copy()
    for t in range(T):
        estado = paso_regla184(estado)
        M[t + 1] = estado

    return M

def flujo_regla184(M, L, t_trans=10):
    """
    Calcula el flujo medio de la Regla 184
    En cada paso, el número de partículas que avanzan es el número de
    pares 10. El flujo se divide entre L para obtener partículas por
    celda y por paso
    """
    movimientos = []
    for t in range(t_trans, M.shape[0]):
        estado = M[t]
        avanza = np.sum((estado == 1) & (np.roll(estado, -1) == 0))
        movimientos.append(avanza / L)
    return np.mean(movimientos)

def ejercicio_a():
    print("\n--- Ejercicio a) Regla 184")

    # Cálculo manual de tres generaciones para L = 10
    L = 10
    S0 = np.array([1, 1, 0, 1, 0, 0, 1, 0, 1, 0], dtype=int)
    M = simular_regla184(S0, T=3)

    print("S0 =", S0, " N =", S0.sum())
    for t in range(1, 4):
        print(f"S{t} =", M[t], " N =", M[t].sum())

    # Diagramas espacio-temporales para dos densidades
    T = 50
    densidades = [0.3, 0.7]
    fig, ejes = plt.subplots(1, 2, figsize=(11, 4))

    for ax, rho in zip(ejes, densidades):
        N = int(round(rho * L))
        estado = np.zeros(L, dtype=int)
        posiciones = rng_global.choice(L, N, replace=False)
        estado[posiciones] = 1

        M = simular_regla184(estado, T)
        J_num = flujo_regla184(M, L)
        J_teo = min(rho, 1.0 - rho)

        print(f"rho = {rho:.1f}: J_num = {J_num:.4f}, J_det = {J_teo:.4f}")

        ax.imshow(M, aspect="auto", cmap="binary", interpolation="nearest")
        ax.set_title(f"Regla 184, rho = {rho:.1f}")
        ax.set_xlabel("Celda")
        ax.set_ylabel("Paso temporal")

    fig.tight_layout()
    fig.savefig("fig_ejercicio_a_regla184.png", dpi=150)
    plt.close(fig)

# ---------------------------------------------------------------------------
# Ejercicio b) Modelo de Nagel-Schreckenberg
def paso_nasch(pos, vel, L, vmax, p, rng):
    """
    Un paso síncrono del modelo NaSch.
    pos y vel deben estar ordenados por posición en el anillo
    """
    N = len(pos)
    pos = np.asarray(pos, dtype=int)
    vel = np.asarray(vel, dtype=int)

    # Distancia libre al vehículo de adelante
    distancia = (np.roll(pos, -1) - pos - 1) % L

    # 1) Aceleración
    v1 = np.minimum(vel + 1, vmax)

    # 2) Frenado de seguridad
    v2 = np.minimum(v1, distancia)

    # 3) Desaceleración estocástica
    ruido = rng.random(N) < p
    v_nueva = np.maximum(v2 - ruido.astype(int), 0)

    # 4) Movimiento en el anillo
    pos_nueva = (pos + v_nueva) % L

    # Se reordena para mantener el orden espacial
    orden = np.argsort(pos_nueva)
    return pos_nueva[orden], v_nueva[orden]

def simular_nasch(L, N, vmax, p, T, rng):
    """
    Simula T pasos con N vehículos partiendo del reposo
    Devuelve la matriz espacio-temporal y las posiciones finales
    """
    pos = np.sort(rng.choice(L, N, replace=False))
    vel = np.zeros(N, dtype=int)

    M = np.zeros((T + 1, L), dtype=int)
    M[0, pos] = 1

    for t in range(T):
        pos, vel = paso_nasch(pos, vel, L, vmax, p, rng)
        M[t + 1, pos] = 1

    return M, pos, vel

def simular_nasch_velocidad(L, N, vmax, p, T, rng):
    """
    Igual que simular_nasch, pero devuelve la velocidad media en cada paso
    """
    pos = np.sort(rng.choice(L, N, replace=False))
    vel = np.zeros(N, dtype=int)
    vel_prom = np.zeros(T + 1)
    vel_prom[0] = vel.mean()

    for t in range(T):
        pos, vel = paso_nasch(pos, vel, L, vmax, p, rng)
        vel_prom[t + 1] = vel.mean()

    return vel_prom

def paso_manual_nasch(x, v, L, vmax, p):
    """
    Versión manual de un paso de NaSch para revisar la tabla del informe
    Se supone que ningún vehículo se desacelera por ruido
    """
    x = np.array(x, dtype=int)
    v = np.array(v, dtype=int)
    N = len(x)

    distancia = (np.roll(x, -1) - x - 1) % L
    v1 = np.minimum(v + 1, vmax)
    v2 = np.minimum(v1, distancia)
    v_nueva = v2.copy()
    x_nueva = (x + v_nueva) % L

    return x_nueva, v_nueva, distancia, v1, v2

def ejercicio_b():
    print("\n--- Ejercicio b) Nagel-Schreckenberg")

    L = 100
    vmax = 5
    p = 0.3

    # Ejemplo manual con tres vehículos
    x0 = [10, 15, 18]
    v0 = [2, 4, 1]
    x1, v1, d, v1p, v2p = paso_manual_nasch(x0, v0, L, vmax, p)

    print("Ejemplo manual:")
    print("distancias =", d)
    print("v tras acelerar =", v1p)
    print("v tras frenar =", v2p)
    print("x(1) =", x1)
    print("v(1) =", v1)

    # Simulación principal
    T = 200
    rng = np.random.default_rng(42)
    M, _, _ = simular_nasch(L, 25, vmax, p, T, rng)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(M, aspect="auto", cmap="binary", interpolation="nearest")
    ax.set_xlabel("Celda de la pista")
    ax.set_ylabel("Paso temporal")
    ax.set_title(f"NaSch: L={L}, N=25, vmax={vmax}, p={p}")
    fig.tight_layout()
    fig.savefig("fig_ejercicio_b_nasch_espaciotemporal.png", dpi=150)
    plt.close(fig)

    # Diagrama fundamental.
    densidades = np.linspace(0.02, 0.95, 30)
    flujos = []
    T_flujo = 500
    T_trans = 100

    for i, rho in enumerate(densidades):
        N_rho = max(1, int(round(rho * L)))
        rng_rho = np.random.default_rng(1000 + i)
        vel_prom = simular_nasch_velocidad(L, N_rho, vmax, p, T_flujo, rng_rho)
        v_media = vel_prom[T_trans:].mean()
        J = (N_rho / L) * v_media
        flujos.append(J)

    flujos = np.array(flujos)
    J_det = np.minimum(densidades, 1.0 - densidades)

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.plot(densidades, flujos, "o-", label="NaSch numérico")
    ax2.plot(densidades, J_det, "r--", label=r"$J_{\det}=\min(\rho,1-\rho)$")
    ax2.set_xlabel(r"Densidad $\rho$")
    ax2.set_ylabel(r"Flujo $J(\rho)$")
    ax2.grid(True)
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig("fig_ejercicio_b_diagrama_fundamental.png", dpi=150)
    plt.close(fig2)

    i_max = np.argmax(flujos)
    print("Densidad crítica aproximada:", densidades[i_max])
    print("Flujo máximo aproximado:", flujos[i_max])

# ---------------------------------------------------------------------------
# Ejercicio c) Juego de la Vida de Conway
def contar_vecinos_vida(S):
    """
    Cuenta los ocho vecinos de Moore para cada celda
    Se usan desplazamientos periódicos con np.roll
    """
    V = np.zeros_like(S, dtype=int)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            V += np.roll(np.roll(S, -dx, axis=0), -dy, axis=1)
    return V

def paso_vida(S):
    """
    Aplica un paso del Juego de la Vida
    Nace con 3 vecinos; sobrevive con 2 o 3 vecinos
    """
    V = contar_vecinos_vida(S)
    nace = (S == 0) & (V == 3)
    sobrevive = (S == 1) & ((V == 2) | (V == 3))
    return (nace | sobrevive).astype(int)

def simular_vida(S0, T):
    """
    Simula T generaciones y guarda la densidad de celdas vivas
    """
    S = S0.copy()
    densidades = [S.mean()]
    snapshots = {0: S.copy()}

    for t in range(1, T + 1):
        S = paso_vida(S)
        densidades.append(S.mean())
        if t in (50, 100, 200, 300):
            snapshots[t] = S.copy()

    return np.array(densidades), snapshots

def ejercicio_c():
    print("\n--- Ejercicio c) Juego de la Vida")

    N = 100
    T = 300
    rho0 = 0.2
    rng = np.random.default_rng(123)

    S0 = (rng.random((N, N)) < rho0).astype(int)
    densidades, snapshots = simular_vida(S0, T)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(densidades)
    ax.set_xlabel("Generación")
    ax.set_ylabel("Densidad de celdas vivas")
    ax.set_title("Juego de la Vida: evolución de la densidad")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig("fig_ejercicio_c_vida_densidad.png", dpi=150)
    plt.close(fig)

    # Figura con algunos estados
    tiempos = [0, 50, 100, 200, 300]
    fig2, ejes = plt.subplots(1, len(tiempos), figsize=(15, 3.5))
    for ax, t in zip(ejes, tiempos):
        ax.imshow(snapshots[t], cmap="binary", interpolation="nearest")
        ax.set_title(f"t = {t}")
        ax.axis("off")
    fig2.tight_layout()
    fig2.savefig("fig_ejercicio_c_vida_estados.png", dpi=150)
    plt.close(fig2)

# ---------------------------------------------------------------------------
# Ejercicio d) Percolación por sitios en 2D
def hay_percolacion(mask):
    """
    Verifica si existe un camino de celdas ocupadas desde la fila superior
    hasta la fila inferior. Se usa búsqueda en profundidad desde la fila 0
    """
    L = mask.shape[0]
    visitado = np.zeros_like(mask, dtype=bool)
    pila = []

    for j in range(L):
        if mask[0, j]:
            pila.append((0, j))
            visitado[0, j] = True

    while pila:
        i, j = pila.pop()
        if i == L - 1:
            return True

        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ni, nj = i + di, j + dj
            if 0 <= ni < L and 0 <= nj < L:
                if mask[ni, nj] and not visitado[ni, nj]:
                    visitado[ni, nj] = True
                    pila.append((ni, nj))

    return False

def simular_percolacion(L, p, N, rng):
    """
    Estima la probabilidad de percolación con N realizaciones
    """
    exitos = 0
    for _ in range(N):
        mask = rng.random((L, L)) < p
        if hay_percolacion(mask):
            exitos += 1
    return exitos / N

def estimar_pc(p_vals, P):
    """
    Estima el punto donde P(p) cruza 0.5 por interpolación lineal
    """
    for i in range(1, len(P)):
        if P[i - 1] < 0.5 <= P[i]:
            return p_vals[i - 1] + (0.5 - P[i - 1]) * (
                p_vals[i] - p_vals[i - 1]
            ) / (P[i] - P[i - 1])
    return None

def simular_difusion_2d(mask, alpha, dx, dt, pasos):
    """
    Resuelve la ecuación de calor 2D con el Laplaciano de cinco puntos
    Solo se actualizan las celdas ocupadas del dominio percolante
    """
    L = mask.shape[0]
    T = np.zeros((L, L))
    T[L // 2, L // 2] = 100.0

    r = alpha * dt / (dx ** 2)
    if r > 0.25:
        raise ValueError("El paso temporal no cumple el criterio CFL r <= 1/4.")

    for _ in range(pasos):
        T_new = T.copy()
        laplaciano = (
            np.roll(T, 1, axis=0)
            + np.roll(T, -1, axis=0)
            + np.roll(T, 1, axis=1)
            + np.roll(T, -1, axis=1)
            - 4.0 * T
        ) / (dx ** 2)

        T_new[mask] = T[mask] + alpha * dt * laplaciano[mask]
        T_new[~mask] = 0.0
        T = T_new

    return T

def ejercicio_d():
    print("\n--- Ejercicio d) Percolación 2D")

    N = 100
    p_vals = np.linspace(0.3, 0.8, 11)
    tamanos = [20, 50, 100]
    resultados = {}

    fig, ax = plt.subplots(figsize=(7, 5))

    for L in tamanos:
        rng = np.random.default_rng(500 + L)
        P = np.array([simular_percolacion(L, p, N, rng) for p in p_vals])
        resultados[L] = P
        pc = estimar_pc(p_vals, P)
        print(f"L = {L}: pc estimado = {pc}")

        ax.plot(p_vals, P, "o-", label=f"L = {L}")

    ax.axvline(0.59274, color="k", linestyle="--", label=r"$p_c \approx 0.59274$")
    ax.set_xlabel("Probabilidad de ocupación p")
    ax.set_ylabel("Probabilidad de percolación P(p)")
    ax.set_title("Transición de percolación por sitios en 2D")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig("fig_ejercicio_d_percolacion.png", dpi=150)
    plt.close(fig)

    # Difusión térmica sobre un cluster percolante
    L = 50
    p = 0.65
    rng = np.random.default_rng(999)
    mask = None
    for _ in range(200):
        candidata = rng.random((L, L)) < p
        if hay_percolacion(candidata):
            mask = candidata
            break

    alpha = 1.0
    dx = 1.0
    dt = 0.2
    pasos = 200
    T_final = simular_difusion_2d(mask, alpha, dx, dt, pasos)

    fig2, ejes = plt.subplots(1, 2, figsize=(10, 4))
    ejes[0].imshow(mask, cmap="gray", interpolation="nearest")
    ejes[0].set_title("Dominio percolante")
    ejes[0].axis("off")

    im = ejes[1].imshow(T_final, cmap="hot", interpolation="nearest")
    ejes[1].set_title("Temperatura tras difusión")
    ejes[1].axis("off")
    fig2.colorbar(im, ax=ejes[1], fraction=0.046, pad=0.04)

    fig2.tight_layout()
    fig2.savefig("fig_ejercicio_d_difusion.png", dpi=150)
    plt.close(fig2)

if __name__ == "__main__":
    ejercicio_a()
    ejercicio_b()
    ejercicio_c()
    ejercicio_d()