import simpy
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ==========================================
# 1. CONFIG — глобальні параметри симуляції
# ==========================================

RANDOM_SEED = 42

NUM_PRINTERS = 10            # Кількість 3D-принтерів у системі
NUM_JOBS = 10               # Скільки завдань буде згенеровано

# Середній інтервал між надходженням завдань (експоненційний розподіл)
INTERARRIVAL_TIME = 10.0
SETUP_TIME = 15.0            # Час, потрібний для зміни матеріалу на принтері
SWITCHING_PENALTY_COEF = 5.0  # Штраф MAS при прогнозуванні вставки jobs
NETWORK_DELAY = 0.0          # Мережеві затримки при передачі job диспетчеру

# Матеріали та кольори для графіків
MATERIALS = ["PLA", "ABS", "PETG"]
COLORS = {
    "PLA": "#5DADE2",
    "ABS": "#F4D03F",
    "PETG": "#58D68D",
    "SETUP": "#212F3C",
}

# ==========================================
# 2. DATA STRUCTURES — структура з даними задач
# ==========================================


@dataclass
class PrintJob:
    job_id: int                   # Унікальний ID завдання
    print_time: float             # Тривалість друку
    material: str                 # Матеріал, потрібний для друку
    created_at: float             # Час створення робочого завдання
    start_time: Optional[float] = None
    finish_time: Optional[float] = None
    status: str = "NEW"           # NEW → IN_PROGRESS → COMPLETED
    executed_on: Optional[int] = None  # На якому принтері виконано

# ==========================================
# 3. PRINTER AGENT — поведінка одного принтера
# ==========================================


class PrinterAgent:
    def __init__(self, env: simpy.Environment, printer_id: int, start_material: str, strategy: str):
        self.env = env
        self.id = printer_id
        self.strategy = strategy

        # Робоча черга, поточний матеріал і активна робота
        self.queue: List[PrintJob] = []
        self.current_material = start_material
        self.current_job: Optional[PrintJob] = None

        self.new_job_event = env.event()   # Сигнал про появу нової роботи

        # Статистика
        self.setup_count = 0
        # (start, duration, type, material)
        self.history: List[Tuple[float, float, str, str]] = []

        self.estimated_finish_time: float = 0.0  # Прогноз завершення робіт

        # Запускаємо головний цикл агента
        self.process = env.process(self.run())

    def notify_new_job(self):
        """Активує подію, коли приходить нова робота в чергу."""
        if not self.new_job_event.triggered:
            self.new_job_event.succeed()

    def update_estimated_finish_time(self) -> float:
        """
        Оцінює, коли принтер закінчить ВСІ роботи в черзі (поточна + черга).
        Це використовується централізованим диспетчером.
        """
        time_left = 0.0

        # Додаємо залишковий час поточного job, якщо він активний
        if self.current_job and self.current_job.start_time is not None:
            elapsed = self.env.now - self.current_job.start_time
            remaining = max(0.0, self.current_job.print_time - elapsed)
            time_left += remaining
            sim_mat = self.current_job.material
        else:
            sim_mat = self.current_material

        # Прогнозуємо всю чергу
        for job in self.queue:
            if job.material != sim_mat:
                time_left += SETUP_TIME
                sim_mat = job.material
            time_left += job.print_time

        self.estimated_finish_time = self.env.now + time_left
        return self.estimated_finish_time

    def perform_setup_if_needed(self, job: PrintJob):
        """Виконується зміна матеріалу (setup), якщо job потребує іншого матеріалу."""
        if self.current_material != job.material:
            self.history.append((self.env.now, SETUP_TIME, "SETUP", "SETUP"))
            yield self.env.timeout(SETUP_TIME)
            self.current_material = job.material
            self.setup_count += 1

    def run(self):
        """
        Основний цикл принтера:
        - чекає на job
        - виконує setup при потребі
        - друкує job
        """
        while True:
            if not self.queue:
                # Немає робіт — чекаємо сигналу від диспетчера
                yield self.new_job_event
                self.new_job_event = self.env.event()
                continue

            # Вибираємо наступну роботу (FIFO або оптимізовану)
            job = self.queue.pop(0)
            self.current_job = job
            job.executed_on = self.id
            job.start_time = self.env.now
            job.status = "IN_PROGRESS"

            # Якщо матеріал не співпадає — робимо переналаштування
            if self.current_material != job.material:
                yield from self.perform_setup_if_needed(job)

            # Записуємо в історію
            self.history.append(
                (self.env.now, job.print_time, "JOB", job.material))

            # Виконуємо друк
            yield self.env.timeout(job.print_time)

            job.finish_time = self.env.now
            job.status = "COMPLETED"
            self.current_job = None

    # ----------------------------
    # MAS bidding: прогноз вставки job у оптимальне місце черги
    # ----------------------------

    def calculate_smart_bid(self, new_job: PrintJob) -> Tuple[float, int]:
        """
        MAS: перевіряє всі можливі позиції вставки new_job у чергу
        та оцінює загальну вартість (час + штрафи).
        """
        best_cost = float('inf')
        best_index = 0

        for insert_idx in range(len(self.queue) + 1):
            cost = self._simulate_cost(new_job, insert_idx)
            if cost < best_cost:
                best_cost = cost
                best_index = insert_idx

        return best_cost, best_index

    def _simulate_cost(self, new_job: PrintJob, index: int) -> float:
        """
        Проганяє симуляцію на основі поточної черги:
        - враховує залишковий час поточного job
        - додає setup при зміні матеріалу
        - додає штраф (setup * SWITCH_PENALTY)
        """
        temp_queue = self.queue[:]
        temp_queue.insert(index, new_job)

        sim_time = self.env.now

        # Додаємо час активного job
        if self.current_job and self.current_job.start_time is not None:
            elapsed = self.env.now - self.current_job.start_time
            sim_time += max(0.0, self.current_job.print_time - elapsed)
            sim_mat = self.current_job.material
        else:
            sim_mat = self.current_material

        total_setup_penalty = 0.0

        # Проганяємо всі роботи у тимчасовій черзі
        for job in temp_queue:
            if job.material != sim_mat:
                sim_time += SETUP_TIME
                total_setup_penalty += SETUP_TIME * SWITCHING_PENALTY_COEF
                sim_mat = job.material
            sim_time += job.print_time

        return sim_time + total_setup_penalty


# ==========================================
# 4. DISPATCHERS — алгоритми розподілу
# ==========================================

def centralized_dispatcher(env: simpy.Environment, job_store: simpy.Store, printers: List[PrinterAgent]):
    """
    Централізований диспетчер:
    - отримує роботу
    - прогнозує час завершення кожного принтера
    - додає job у чергу того принтера, де job найшвидше завершиться
    (враховує setup у кінці черги)
    """
    while True:
        job: PrintJob = yield job_store.get()
        yield env.timeout(NETWORK_DELAY)

        # Оновити прогнозовані часи
        for p in printers:
            p.update_estimated_finish_time()

        best_printer = None
        best_score = float('inf')

        # Пошук найкращого принтера
        for p in printers:
            if p.queue:
                last_material = p.queue[-1].material
            elif p.current_job:
                last_material = p.current_job.material
            else:
                last_material = p.current_material

            added_setup = SETUP_TIME if last_material != job.material else 0.0

            finish_prediction = p.estimated_finish_time + added_setup + job.print_time

            if finish_prediction < best_score:
                best_score = finish_prediction
                best_printer = p

        best_printer.queue.append(job)
        best_printer.notify_new_job()


def mas_dispatcher(env: simpy.Environment, job_store: simpy.Store, printers: List[PrinterAgent]):
    """
    MAS (Multi-Agent System):
    - кожен принтер робить "ставку" (bid) — прогнозує оптимальне місце вставки job
    - найменша вартість (час + штрафи) перемагає
    """
    while True:
        job: PrintJob = yield job_store.get()
        yield env.timeout(NETWORK_DELAY)

        bids = []
        for p in printers:
            cost, idx = p.calculate_smart_bid(job)
            bids.append((cost, p, idx))

        # Обираємо принтер з мінімальним cost
        _, winner, idx = min(bids, key=lambda x: x[0])

        yield env.timeout(NETWORK_DELAY)

        winner.queue.insert(idx, job)
        winner.notify_new_job()


# ==========================================
# 5. SIMULATION / GENERATOR / PLOTTING
# ==========================================

def sample_print_time_triangular() -> float:
    """Трикутний розподіл тривалості друку (min, max, mode)."""
    return random.triangular(60.0, 240.0, 150.0)


def run_simulation(strategy: str, random_seed: int = RANDOM_SEED) -> Tuple[List[PrinterAgent], List[PrintJob]]:
    """
    Запускає симуляцію:
    - генерує принтери
    - генерує jobs
    - запускає диспетчер (CENTRALIZED або MAS)
    - чекає завершення всіх jobs
    """
    random.seed(random_seed)
    env = simpy.Environment()
    store = simpy.Store(env)

    # Створюємо принтери
    printers: List[PrinterAgent] = []
    for i in range(NUM_PRINTERS):
        mat = random.choice(MATERIALS)
        printers.append(PrinterAgent(env, i, mat, strategy))

    jobs: List[PrintJob] = []

    # Генератор завдань
    def generator():
        for i in range(NUM_JOBS):
            yield env.timeout(random.expovariate(1.0 / INTERARRIVAL_TIME))
            pt = sample_print_time_triangular()
            job = PrintJob(i, pt, random.choice(MATERIALS), env.now)
            jobs.append(job)
            store.put(job)

    env.process(generator())

    # Обираємо відповідний диспетчер
    if strategy == "CENTRALIZED":
        env.process(centralized_dispatcher(env, store, printers))
    else:
        env.process(mas_dispatcher(env, store, printers))

    # Чекаємо завершення всіх jobs
    all_jobs_done = env.event()

    def completion_monitor():
        while True:
            finished_count = sum(1 for j in jobs if j.status == "COMPLETED")
            if finished_count == NUM_JOBS:
                all_jobs_done.succeed()
                break
            yield env.timeout(10.0)

    env.process(completion_monitor())
    env.run(until=all_jobs_done)

    return printers, jobs


def plot_gantt(printers: List[PrinterAgent], title: str):
    """
    Будує Gantt chart історії роботи кожного принтера
    (jobs + setup).
    """
    fig, ax = plt.subplots(figsize=(15, 8))
    sorted_printers = sorted(printers, key=lambda p: p.id)
    y_ticks, y_labels = [], []

    for i, p in enumerate(sorted_printers):
        y_val = i * 10
        y_ticks.append(y_val + 5)
        y_labels.append(f"P-{p.id}")

        # Малюємо всі операції принтера (JOB + SETUP)
        for start, duration, type_op, mat in p.history:
            color = COLORS.get(mat, "gray")
            edge = 'None'
            height = 8
            y_offset = 1

            if type_op == "SETUP":
                height, y_offset, edge = 6, 2, 'black'
                color = COLORS["SETUP"]

            ax.broken_barh([(start, duration)], (y_val + y_offset, height),
                           facecolors=color, edgecolor=edge, linewidth=0.5)

            # Підписуємо матеріал всередині довгих барів
            if duration > 20 and type_op == "JOB":
                ax.text(start + duration / 2, y_val + 5, mat, ha='center',
                        va='center', color='white', fontsize=7, fontweight='bold')

    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)
    ax.set_xlabel("Час (хвилини)")
    ax.set_title(title)
    ax.grid(True, axis='x', linestyle='--', alpha=0.3)

    patches = [mpatches.Patch(color=c, label=m) for m, c in COLORS.items()]
    ax.legend(handles=patches, loc='upper right')
    plt.tight_layout()


def get_metrics(printers: List[PrinterAgent], jobs: List[PrintJob]):
    """
    Обчислює:
    - кількість setup
    - загальний час роботи (makespan)
    - середній час перебування job у системі (flow time)
    """
    setup_count = sum(p.setup_count for p in printers)

    finished_times = [j.finish_time for j in jobs if j.finish_time is not None]
    if not finished_times:
        return 0, 0.0

    makespan = max(finished_times)

    wait_times = [(j.finish_time - j.created_at)
                  for j in jobs if j.finish_time]
    avg_flow = sum(wait_times) / len(wait_times) if wait_times else 0

    return setup_count, makespan, avg_flow


# ==========================================
# 6. RUN — основний запуск двох стратегій
# ==========================================

if __name__ == "__main__":
    # Виводимо параметри симуляції
    print("\n>>> Параметри симуляції:")
    print("-" * 50)
    print(f"RANDOM_SEED         = {RANDOM_SEED}")
    print(" ")
    print(f"Принтерів           = {NUM_PRINTERS}")
    print(f"Завдань             = {NUM_JOBS}")
    print(" ")
    print(f"Частота завдань     = {INTERARRIVAL_TIME}")
    print(f"Заміна матеріалу    = {SETUP_TIME}")
    print(f"Затримка мережі     = {NETWORK_DELAY}")
    print(f"MAS коефіцієнт      = {SWITCHING_PENALTY_COEF}")
    print(" ")
    print(f"Матеріали           = {MATERIALS}")
    print("-" * 50)
    print(" ")

    # Запускаємо обидві стратегії
    print(">>> Запуск симуляції CENTRALIZED...")
    printers_c, jobs_c = run_simulation("CENTRALIZED")

    print(">>> Запуск симуляції MAS...")
    printers_m, jobs_m = run_simulation("MAS")

    # Метрики
    sc_c, mk_c, flow_c = get_metrics(printers_c, jobs_c)
    sc_m, mk_m, flow_m = get_metrics(printers_m, jobs_m)

    def safe_diff(m, c):
        if c == 0:
            return 0
        return (m - c) / c * 100

    # Процентні різниці (перевернуті знаки: позитивне = краще MAS)
    diff_sc = safe_diff(sc_m, sc_c)
    diff_mk = safe_diff(mk_m, mk_c)
    diff_flow = safe_diff(flow_m, flow_c)

    # Таблиця результатів
    print("\n" + "=" * 100)
    print(f"{'Метрики':<25} | {'CENTRALIZED':<20} | {'MAS':<20} | {'Різниця (%)':<10}")
    print("-" * 100)
    print(f"{'Переналаштувань':<25} | {sc_c:<20} | {sc_m:<20} | {diff_sc:+.1f}%")
    print(f"{'Загальний час роботи':<25} | {mk_c:<20.1f} | {mk_m:<20.1f} | {diff_mk:+.1f}%")
    print(f"{'Час очікування':<25} | {flow_c:<20.1f} | {flow_m:<20.1f} | {diff_flow:+.1f}%")
    print("=" * 100)
    print(" ")

    # Побудова gantt-діаграм
    plot_gantt(printers_c, "Centralized Queue (FIFO + Tail Check)")
    plot_gantt(printers_m, "MAS Queue (Optimization via Insertion)")
    plt.show()
