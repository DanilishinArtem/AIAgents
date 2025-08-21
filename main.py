import numpy as np
from typing import List, Tuple

from utils import HMMTwoAgents
# from utils import HMMTwoAgentsLogLinear as HMMTwoAgents

# ------------------ Пример использования ------------------
if __name__ == "__main__":
    # Определим словарь действий
    actions = {0: "SQL", 1: "Text", 2: "Image", 3: "Analyze", 4: "Compile", 5: "Fix"}
    A = len(actions)
    K = 3  # число скрытых режимов

    model = HMMTwoAgents(n_states=K, n_actions=A, rand_state=42)

    # Синтетические данные: сгенерируем несколько траекторий
    sequences = []
    for _ in range(1):
        s, y = model.sample(T=30)
        sequences.append(y)
    print("[DEBUG] s: {}, y: {}".format(s, y))
    # Сбросим параметры (чтобы EM не тривиально "запомнил" генератор)
    model = HMMTwoAgents(n_states=K, n_actions=A, rand_state=123)

    print("Начинаем EM...")
    ll = model.fit_em(sequences, n_iter=100)
    print("Готово. Финальное loglik:", ll)

    # Покажем пример предсказания состояний
    s_hat = model.viterbi(sequences[0])
    print("Viterbi states (пример):", s_hat)
    # result = [actions[s_hat[item]] for item in range(len(s_hat))]
    # print("Viterbi states (пример):", result)

