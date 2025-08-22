import numpy as np
from typing import List, Tuple

# from utils import HMMTwoAgents
from utils import HMMTwoAgentsLogLinear as HMMTwoAgents

# ------------------ Пример использования ------------------
if __name__ == "__main__":
    # Определим словарь действий
    actions = {0: "SQL", 1: "TextGen", 2: "ImageGen", 3: "CodeAnalyze", 4: "CodeCompile", 5: "FixCode", 6: "Stay"}
    features = {0: "SQLReqDuration", 1: "PromptLen", 2: "CodeLen", 3: "CodeComplexity", 4: "CodeCoverage", 5: "StayDuration"}
    A = len(actions)
    K = 3  # число скрытых режимов

    model = HMMTwoAgents(n_states=K, n_actions=A, rand_state=42)

    # Синтетические данные: сгенерируем несколько траекторий
    sequences = []
    T_init = 30
    for index in range(30):
        s, y = model.sample(T=T_init)
        sequences.append(y)
    print("======================= Initial parameters =======================")
    obs_init_readable = [[actions[z1], actions[z2]] for z1, z2 in y]
    print("Hidden states: {}".format(s))
    print("Actions: {}".format(obs_init_readable))
        
    # Сбросим параметры (чтобы EM не тривиально "запомнил" генератор)
    model = HMMTwoAgents(n_states=K, n_actions=A, rand_state=123)

    print("Начинаем EM...")
    ll = model.fit_em(sequences, n_iter=100)
    print("Готово. Финальное loglik:", ll)
    # Покажем пример предсказания состояний
    s_hat = model.viterbi(sequences[0])
    
    T_new = 30
    states_sim, obs_sim = model.sample(T_new)
    
    obs_sim_readable = [[actions[z1], actions[z2]] for z1, z2 in obs_sim]
    print("======================= Simulated parameters =======================")
    print("Hidden states: {}".format(states_sim))
    print("Simulated actions: {}".format(obs_sim_readable))