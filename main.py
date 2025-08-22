import numpy as np
from typing import List, Tuple

from utils import HMMTwoAgentsFeatures

# ------------------ Пример использования ------------------
if __name__ == "__main__":
    actions = {0:"SQL",1:"TextGen",2:"ImageGen",3:"CodeAnalyze",4:"CodeCompile",5:"FixCode",6:"Stay"}
    features = {0:"SQLReqDuration",1:"PromptLen",2:"CodeLen",3:"CodeComplexity",4:"CodeCoverage",5:"StayDuration"}

    K, A, F = 3, len(actions), len(features)
    model = HMMTwoAgentsFeatures(n_states=K, n_actions=A, n_features=F, rand_state=42)

    # сгенерируем синтетику и обучим
    s_true, z_seq, x_seq = model.sample(T=50)
    print("======================= Initial parameters =======================")
    initActions = [[actions[z1], actions[z2]] for z1, z2 in z_seq]
    print("Hidden states: {}".format(s_true))
    print("Actions: {}".format(initActions))

    # для тренировки сделаем новые рандом-параметры (как и раньше)
    model = HMMTwoAgentsFeatures(n_states=K, n_actions=A, n_features=F, rand_state=0)

    ll = model.fit_em(sequences=[(z_seq, x_seq)], n_iter=50, tol=1e-4, verbose=True)
    T_new = 30
    s_mod, z_mod, x_mod = model.sample(T_new)
    print("======================= Simulated parameters =======================")
    simActions = [[actions[z1], actions[z2]] for z1, z2 in z_mod]
    print("Simulated hidden states: {}".format(s_mod))
    print("Simulated actions: {}".format(simActions))