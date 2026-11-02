import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import csv
from dataclasses import dataclass
from air_features import air_features
import pandas as pd
from paretoset import paretoset
import math
from Total_Transfer_Matrix import Impedance
from root_finder import calcolo_zero

###########################################################
## Parameters
# Main waveguide
r_mw = 0.05;

# Probability of random initial point
perc = 1;

# Neural network
dim1 = 512
dim2 = dim1;

# Air parameters
air = air_features()
c0 = air.c0
Z0 = air.Z0

# Secant method
tol = 1e-8
maxit = 1000;
x0 = 1
x1 = 2

# Steps
paso = 1e-4 
n_experiments = 750;
n_iterations = 2000;

# Reward
pesi = [5,4,6]

# N. cifre decimali
cifre = 1e4


#############################################################
# Cartella principale
n_exp = n_experiments
main_dir = f"Optimization_3D"
os.makedirs(main_dir, exist_ok=True)

# Sottocartella dei grafici 
grafici_dir = os.path.join(main_dir, f"Grafici_3D")
os.makedirs(grafici_dir, exist_ok=True)


# File CSV
csv_completo = os.path.join(main_dir, f"Valori_completi_3D.csv")
with open(csv_completo, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Experiment", "Iteration", "l_c", "r_c", "l_n", "r_n", "V", "w", "R"])

csv_finale = os.path.join(main_dir, f"Valori_finali_3D.csv")
with open(csv_finale, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Experiment", "l_c", "r_c", "l_n", "r_n", "V", "w", "R", "N.iter."])



##############################################################################
# Environment
class Environment:
    def __init__(self):
        self.lim_min = np.array([0.01, 0.0036, 0.01, 0.002])
        self.lim_max = np.array([0.08, 0.04, 0.04, 0.022])
        self.reset()

    def reset(self, rand = False):
        while True:
            self.state = np.array([
            np.random.uniform(self.lim_min[0], self.lim_max[0]),
            np.random.uniform(self.lim_min[1], self.lim_max[1]),
            np.random.uniform(self.lim_min[2], self.lim_max[2]),
            np.random.uniform(self.lim_min[3], self.lim_max[3])
            ])
            self.state = np.round(self.state, 4)
            if self.state[3] <= 0.55 * self.state[1]:
                break
        self.V, self.w, self.R = self.compute_V_w_R(self.state)
        return self.state.copy(), self.V, self.w, self.R


    def normalized_state(self):
        return (self.state - self.lim_min)/(self.lim_max - self.lim_min)
    
    def compute_V_w_R(self, s):
        l_c, r_c, l_n, r_n = s
        def correction_serie(r_big, r_small):
                rr = r_small/r_big
                dl = 0.82*(1 - 1.35*rr + 0.31*rr**3)*r_small
                return dl
        dl1 = correction_serie(r_c, r_n)
        dl2 = correction_serie(r_mw, r_n)

        @dataclass
        class Parameters:
            l: float
            r: float
        lista_parametri = [Parameters(l=dl2, r=r_n), Parameters(l=l_n, r=r_n),
            Parameters(l=dl1, r=r_n), Parameters(l=l_c, r=r_c)]

        # Impedance
        blocco_Z = Impedance(lista_parametri)
        def Z_imag(w):
            return blocco_Z(w)[3]
        def Z_r(w):
            return blocco_Z(w)[1]

        # Angular resonance frequency
        w = calcolo_zero(Z_imag, x0, x1, tol, maxit)

        # Reflection coefficient
        Z = Z_r(w)
        Z0_norm = Z0 / (np.pi * r_mw**2)
        R = np.abs((Z - Z0_norm)/(Z + Z0_norm))

        # Volume
        V = np.pi*(r_c**2 * l_c + r_n**2 * l_n)
        return V, w, R
    
    

    def step(self, action_idx):
        actions = [
            (0, +paso), (0, -paso),
            (1, +paso), (1, -paso),
            (2, +paso), (2, -paso),
            (3, +paso), (3, -paso),
            (0, 0)]

        prev_V = self.V
        prev_w = self.w
        prev_R = self.R

        var_idx, delta = actions[action_idx]
        old_state = self.state.copy()
        new_state = old_state.copy()
        new_state[var_idx] += delta
        new_state[var_idx] = np.clip(new_state[var_idx], self.lim_min[var_idx], self.lim_max[var_idx])
        new_state = np.round(new_state, 4)

        # Control: r_n < 0.55 * r_c
        coeff = new_state[3] / new_state[1]
        if coeff > 0.55:
            if var_idx == 3:
                new_r_n = new_state[1] * 0.55
                new_state[3] = math.floor(new_r_n * cifre) / cifre
            elif var_idx == 1:
                new_r_c = new_state[3] / 0.55
                new_state[1] = math.ceil(new_r_c * cifre) / cifre
        

        self.state = new_state
        self.V, self.w, self.R = self.compute_V_w_R(self.state)

        if np.array_equal(new_state, old_state):
            reward = -1
            nullo = 1
        else:
            reward = self.get_reward(prev_V, self.V, prev_w, self.w, prev_R, self.R)
            nullo = 0

        return self.state.copy(), self.V, self.w, self.R, reward, nullo

    def get_reward(self, prev_V, curr_V, prev_w, curr_w, prev_R, curr_R): 
        def delta_reward(old, new, toll):
            K = (old - new) / abs(old+1e-15)
            if abs(K) < toll:
                return 0
            else:
                return np.sign(K)

        r_V = delta_reward(prev_V, curr_V, 1e-4)
        r_w = delta_reward(prev_w, curr_w, 1e-4)
        r_R = delta_reward(prev_R, curr_R, 1e-4)
        
        reward = (pesi[0] * r_V + pesi[1] * r_w + pesi[2] * r_R)
        return reward

####################################################################################
# Agent
class Agent:
    def __init__(self, lr=2e-3, gamma=0.9, epsilon=1.0):
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = 0.2 
        self.epsilon_decay = 0.99999

        self.batch_size = 64
        self.memory = deque(maxlen=100000)
        self.model = nn.Sequential(
            nn.Linear(4, dim1),
            nn.ReLU(),
            nn.Linear(dim1, dim2),
            nn.ReLU(),
            nn.Linear(dim2, 9)
        )
            
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def select_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        with torch.no_grad():
            q_values = self.model(state_tensor).squeeze(0).numpy()

        chosen_action = int(np.argmax(q_values))
        

        if random.random() < self.epsilon:
            random_action = random.randint(0, 8)
            return random_action

        return chosen_action

    def store(self, transition):
        self.memory.append(transition)
        self.epsilon = max(self.epsilon_min, self.epsilon*self.epsilon_decay)

    def train(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states = zip(*batch)

        states = torch.as_tensor(np.array(states), dtype=torch.float32)
        next_states = torch.as_tensor(np.array(next_states), dtype=torch.float32)
        rewards = torch.as_tensor(rewards, dtype=torch.float32).unsqueeze(1)
        actions = torch.as_tensor(actions, dtype=torch.int64).unsqueeze(1)

        next_q_values = self.model(next_states).max(1)[0].detach().unsqueeze(1)
        target = rewards + self.gamma * next_q_values
        q_values = self.model(states).gather(1, actions)

        loss = nn.MSELoss()(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

##########################################################################
# Main 

env = Environment()
agent = Agent()

resultados_finales = []
V_history = []
w_history = []
R_history = []


for j in tqdm(range(n_experiments)):
    state, V, w, R = env.reset()
    
    cont_azione_nulla = 0

    for i in range(n_iterations):
    
        input_state = env.normalized_state()
        action = agent.select_action(input_state)

        l_c, r_c, l_n, r_n = state

        next_state, next_V, next_w, next_R, reward, nullo = env.step(action)
        if nullo == 1:
            cont_azione_nulla += 1
        else:
            cont_azione_nulla = 0
        
        next_input = env.normalized_state()

        # Network Training
        agent.store((input_state, action, reward, next_input))
        if j < 350 or i % 5 == 0:
            agent.train()
        
        with open(csv_completo, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([j+1, i+1, l_c, r_c, l_n, r_n, V, w, R])

        V_history.append(V)
        w_history.append(w)
        R_history.append(R)

        state, V, w, R = next_state, next_V, next_w, next_R

        # Interrompo l'episodio dopo più di 10 azioni nulle consecutive
        if cont_azione_nulla >= 10:
            break

    resultados_finales.append({"Experiment": j + 1, "V": V, "w": w, "R": R, "State": state,})

    # Salvo dati finali dell’esperimento
    l_c, r_c, l_n, r_n = state
    with open(csv_finale, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([j+1, l_c, r_c, l_n, r_n, V, w, R, i+1])


####################################################################################
# Pareto front
V_val = np.array([r["V"] for r in resultados_finales])
w_val = np.array([r["w"] for r in resultados_finales])
R_val = np.array([r["R"] for r in resultados_finales])

## CSV + GRAFICI PARETO
pareto_data=np.column_stack((V_val,w_val,R_val))
is_pareto=paretoset(pareto_data,sense=["min","min","min"])
pareto_indices=np.where(is_pareto)[0]

pareto_df=pd.DataFrame({
    "Experiment":[resultados_finales[i]["Experiment"] for i in pareto_indices],
    "l_c":[resultados_finales[i]["State"][0] for i in pareto_indices],
    "r_c":[resultados_finales[i]["State"][1] for i in pareto_indices],
    "l_n":[resultados_finales[i]["State"][2] for i in pareto_indices],
    "r_n":[resultados_finales[i]["State"][3] for i in pareto_indices],
    "V":V_val[is_pareto],
    "w":w_val[is_pareto],
    "R":R_val[is_pareto]
})

pareto_df=pareto_df.sort_values("R").reset_index(drop=True)

csv_pareto=os.path.join(main_dir,f"Fronte_Pareto_3D.csv")
pareto_df.to_csv(csv_pareto,index=False)


