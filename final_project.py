import taichi as ti
import argparse
import os
import math
import numpy as np
import matplotlib.pyplot as plt
import random
import time
import csv
from starting_robots import *


real = ti.f32
ti.init(default_fp=real, arch=ti.gpu, flatten_if=True, debug=True)

dim = 2
n_particles = 8192
n_solid_particles = 0
n_actuators = 0
n_grid = 128
dx = 1 / n_grid
inv_dx = 1 / dx
dt = 1e-3
p_vol = 1
E = 10
mu = E
la = E
max_steps = 2048
steps = 1024
gravity = 3.8
target = [0.8, 0.2]


scalar = lambda: ti.field(dtype=real)
vec = lambda: ti.Vector.field(dim, dtype=real)
mat = lambda: ti.Matrix.field(dim, dim, dtype=real)


actuator_id = ti.field(ti.i32)
particle_type = ti.field(ti.i32)
x, v = vec(), vec()
grid_v_in, grid_m_in = vec(), scalar()
grid_v_out = vec()
C, F = mat(), mat()

loss = scalar()

n_sin_waves = 4
weights = scalar()
bias = scalar()
x_avg = vec()

actuation = scalar()
actuation_omega = 20
act_strength = 4


def allocate_fields():

    ti.root.dense(ti.ij, (n_actuators, n_sin_waves)).place(weights)
    ti.root.dense(ti.i, n_actuators).place(bias)

    ti.root.dense(ti.ij, (max_steps, n_actuators)).place(actuation)
    ti.root.dense(ti.i, n_particles).place(actuator_id, particle_type)
    ti.root.dense(ti.k, max_steps).dense(ti.l, n_particles).place(x, v, C, F)
    ti.root.dense(ti.ij, n_grid).place(grid_v_in, grid_m_in, grid_v_out)
    
    ti.root.place(loss, x_avg)

    ti.root.lazy_grad()

def reset_fields():
    ti.root.deactivate_all()


@ti.kernel
def reset_all_fields():
    # Reset particle-related fields
    for f in range(max_steps):
        for i in range(n_particles):
            x[f, i] = [0.0, 0.0]   
            v[f, i] = [0.0, 0.0]   
            C[f, i] = [[0.0, 0.0], [0.0, 0.0]]   
            F[f, i] = [[0.0, 0.0], [0.0, 0.0]]   
            actuator_id[i] = -1  
            particle_type[i] = 0  

    for i, j in grid_v_in:
        grid_v_in[i, j] = [0.0, 0.0]
        grid_m_in[i, j] = 0.0

    loss[None] = 0.0
    x_avg[None] = [0.0, 0.0]

    for j in range(n_sin_waves):
        weights[0, j] = 0.0   

    bias[0] = 0.0
  
    for t, i in actuation:
        actuation[t, i] = 0.0


#clear gradients 
@ti.kernel
def reset_gradients():
    for f, i in x:
        x.grad[f, i] = [0, 0]
        v.grad[f, i] = [0, 0]
        C.grad[f, i] = [[0, 0], [0, 0]]
        F.grad[f, i] = [[0, 0], [0, 0]]

    for i, j in grid_v_in:
        grid_v_in.grad[i, j] = [0.0, 0.0]
        grid_m_in.grad[i, j] = 0.0

    
    for t in range(max_steps):
        for i in range(n_actuators):
            actuation.grad[t, i] = 0.0

@ti.kernel
def clear_grid():
    for i, j in grid_m_in:
        grid_v_in[i, j] = [0, 0]
        grid_m_in[i, j] = 0
        grid_v_in.grad[i, j] = [0, 0]
        grid_m_in.grad[i, j] = 0
        grid_v_out.grad[i, j] = [0, 0]

@ti.kernel
def clear_particle_grad():
    for f, i in x:
        x.grad[f, i] = [0, 0]
        v.grad[f, i] = [0, 0]
        C.grad[f, i] = [[0, 0], [0, 0]]
        F.grad[f, i] = [[0, 0], [0, 0]]


@ti.kernel
def clear_actuation_grad():
    for t, i in actuation:
        actuation[t, i] = 0.0


@ti.kernel
def p2g(f: ti.i32):
    for p in range(n_particles):
        base = ti.cast(x[f, p] * inv_dx - 0.5, ti.i32)
        fx = x[f, p] * inv_dx - ti.cast(base, ti.i32)
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
        new_F = (ti.Matrix.diag(dim=2, val=1) + dt * C[f, p]) @ F[f, p]
        J = (new_F).determinant()
        if particle_type[p] == 0:  # fluid
            sqrtJ = ti.sqrt(J)
            new_F = ti.Matrix([[sqrtJ, 0], [0, sqrtJ]])

        F[f + 1, p] = new_F
        r, s = ti.polar_decompose(new_F)

        act_id = actuator_id[p]

        act = actuation[f, ti.max(0, act_id)] * act_strength
        if act_id == -1:
            act = 0.0
        # ti.print(act)

        A = ti.Matrix([[0.0, 0.0], [0.0, 1.0]]) * act
        cauchy = ti.Matrix([[0.0, 0.0], [0.0, 0.0]])
        mass = 0.0
        if particle_type[p] == 0:
            mass = 4
            cauchy = ti.Matrix([[1.0, 0.0], [0.0, 0.1]]) * (J - 1) * E
        else:
            mass = 1
            cauchy = 2 * mu * (new_F - r) @ new_F.transpose() + \
                     ti.Matrix.diag(2, la * (J - 1) * J)
        cauchy += new_F @ A @ new_F.transpose()
        stress = -(dt * p_vol * 4 * inv_dx * inv_dx) * cauchy
        affine = stress + mass * C[f, p]
        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                offset = ti.Vector([i, j])
                dpos = (ti.cast(ti.Vector([i, j]), real) - fx) * dx
                weight = w[i][0] * w[j][1]
                grid_v_in[base +
                          offset] += weight * (mass * v[f, p] + affine @ dpos)
                grid_m_in[base + offset] += weight * mass


bound = 3
coeff = 0.5


@ti.kernel
def grid_op():
    for i, j in grid_m_in:
        inv_m = 1 / (grid_m_in[i, j] + 1e-10)
        v_out = ti.Vector([0.0, 0.0])
        v_out = inv_m * grid_v_in[i, j]
        v_out[1] -= dt * gravity
        if i < bound and v_out[0] < 0:
            v_out[0] = 0
            v_out[1] = 0
        if i > n_grid - bound and v_out[0] > 0:
            v_out[0] = 0
            v_out[1] = 0
        if j < bound and v_out[1] < 0:
            v_out[0] = 0
            v_out[1] = 0
            normal = ti.Vector([0.0, 1.0])
            lsq = (normal**2).sum()
            if lsq > 0.5:
                if ti.static(coeff < 0):
                    v_out[0] = 0
                    v_out[1] = 0
                else:
                    lin = v_out.dot(normal)
                    if lin < 0:
                        vit = v_out - lin * normal
                        lit = vit.norm() + 1e-10
                        if lit + coeff * lin <= 0:
                            v_out[0] = 0
                            v_out[1] = 0
                        else:
                            v_out = (1 + coeff * lin / lit) * vit
        if j > n_grid - bound and v_out[1] > 0:
            v_out[0] = 0
            v_out[1] = 0

        grid_v_out[i, j] = v_out


@ti.kernel
def g2p(f: ti.i32):
    for p in range(n_particles):
        base = ti.cast(x[f, p] * inv_dx - 0.5, ti.i32)
        fx = x[f, p] * inv_dx - ti.cast(base, real)
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1.0)**2, 0.5 * (fx - 0.5)**2]
        new_v = ti.Vector([0.0, 0.0])
        new_C = ti.Matrix([[0.0, 0.0], [0.0, 0.0]])

        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                dpos = ti.cast(ti.Vector([i, j]), real) - fx
                g_v = grid_v_out[base[0] + i, base[1] + j]
                weight = w[i][0] * w[j][1]
                new_v += weight * g_v
                new_C += 4 * weight * g_v.outer_product(dpos) * inv_dx

        v[f + 1, p] = new_v
        x[f + 1, p] = x[f, p] + dt * v[f + 1, p]
        C[f + 1, p] = new_C


@ti.kernel
def compute_actuation(t: ti.i32):
    for i in range(n_actuators):
        act = 0.0
        for j in ti.static(range(n_sin_waves)):
            act += weights[i, j] * ti.sin(actuation_omega * t * dt +
                                          2 * math.pi / n_sin_waves * j)
        act += bias[i]
        actuation[t, i] = ti.tanh(act)


@ti.kernel
def compute_x_avg():
    for i in range(n_particles):
        contrib = 0.0
        if particle_type[i] == 1:
            contrib = 1.0 / n_solid_particles
        ti.atomic_add(x_avg[None], contrib * x[steps - 1, i])


@ti.kernel
def compute_loss():
    dist = x_avg[None][0] + n_particles/1000000
    loss[None] = -dist


@ti.ad.grad_replaced
def advance(s):
    clear_grid()
    compute_actuation(s)
    p2g(s)
    grid_op()
    g2p(s)


@ti.ad.grad_for(advance)
def advance_grad(s):
    clear_grid()
    p2g(s)
    grid_op()
    g2p.grad(s)
    grid_op.grad()
    p2g.grad(s)
    compute_actuation.grad(s)


def forward(total_steps=steps):
    # simulation
    for s in range(total_steps - 1):
        advance(s)
    x_avg[None] = [0, 0]
    compute_x_avg()
    compute_loss()

class Scene:
    def __init__(self):
        self.n_particles = 0
        self.n_solid_particles = 0
        self.x = []
        self.actuator_id = []
        self.particle_type = []
        self.offset_x = 0
        self.offset_y = 0
 

    def add_rect(self, x, y, w, h, actuation, ptype=1):
        if ptype == 0:
            assert actuation == -1
        global n_particles
        w_count = int(w / dx) * 2
        h_count = int(h / dx) * 2
        real_dx = w / w_count
        real_dy = h / h_count
        for i in range(w_count):
            for j in range(h_count):
                self.x.append([
                    x + (i + 0.5) * real_dx + self.offset_x,
                    y + (j + 0.5) * real_dy + self.offset_y
                ])
                self.actuator_id.append(actuation)
                self.particle_type.append(ptype)
                self.n_particles += 1
                self.n_solid_particles += int(ptype == 1)
    
    def add_rect_hollow(self, x, y, w, h, inner_offset, actuation, ptype=1):
        
        #add reactangle with a cutout as specified by inner_offset field
        if ptype == 0:
            assert actuation == -1
        global n_particles
        w_count = int(w / dx) * 2
        h_count = int(h / dx) * 2
        real_dx = w / w_count
        real_dy = h / h_count

        #inside boundaries
        inner_i_min = int(inner_offset / real_dx)
        inner_i_max = w_count - inner_i_min
        inner_j_min = int(inner_offset / real_dy)
        inner_j_max = h_count - inner_j_min

        for i in range(w_count):
            for j in range(h_count):

                # if inside, skip
                if inner_i_min <= i < inner_i_max and inner_j_min <= j < inner_j_max:
                    continue

                #otherwise add particles
                else:
                    pos_x = x + (i + 0.5) * real_dx + self.offset_x
                    pos_y = y + (j + 0.5) * real_dy + self.offset_y

                    self.x.append([pos_x, pos_y])
                    self.actuator_id.append(actuation)
                    self.particle_type.append(ptype)
                    self.n_particles += 1
                    self.n_solid_particles += int(ptype == 1)
                     

    def set_offset(self, x, y):
        self.offset_x = x
        self.offset_y = y

    def finalize(self):
        global n_particles, n_solid_particles
        n_particles = self.n_particles
        n_solid_particles = self.n_solid_particles
        print('n_particles', n_particles)
        print('n_solid', n_solid_particles)

    def set_n_actuators(self, n_act):
        global n_actuators
        n_actuators = n_act
    
#used to create random robot initial states
def robot(scene):
    scene.set_offset(0.1, 0.03)

    num_rectangles = random.randint(3, 5)  
    print(num_rectangles)
    size_range_min = 0.02 #limb size min
    size_range_max = 0.10 #limb size max
    hollow_pct = 0.3 #probability of rectangles being  hollow
    solid_acutation_pct = 0.3 #prob of rectangles being solid actuators

    #keep track of the total number of actators
    num_actuators = 0 
    actuation = []
    robot_data = [] #to reconsturct it later
    
    #determine if the body rectangle is solid or flexible
    if random.random() < solid_acutation_pct:
        actuation_type = -1
    else:
        actuation_type = num_actuators
        num_actuators += 1

    #set the dimensions of the body rectangle
    current_x, current_y = 0, 0.05
    current_width = 0.3
    current_height = 0.1
 
   
    # Place the first (body) rectangle (either hollow or solid)
    if random.random() < hollow_pct: 
        inner_offset = random.uniform(0.01, min(current_width, current_height) / 2) #inner offset has to be less than half the width/height
        scene.add_rect_hollow(x=current_x, y=current_y, w=current_width, h=current_height, inner_offset=inner_offset, actuation=actuation_type)
    else:
        scene.add_rect(x=current_x, y=current_y, w=current_width, h=current_height, actuation=actuation_type)
        inner_offset = -1 #no inner offset


    robot_data.append((current_x, current_y, current_width, current_height, actuation_type, inner_offset))


    #main loop for adding limbs
    for i in range(num_rectangles):
        x_offset_len = current_width/(num_rectangles+1)

        #determine actuator type
        if random.random() < solid_acutation_pct:
            actuation.append(-1)
        else:
            actuation.append(num_actuators)
            num_actuators += 1
        
        #determine rectangle width and height 
        width = random.uniform(size_range_min,x_offset_len*0.8)
        height = random.uniform(size_range_min,size_range_max)
    
        next_x = current_x+i*x_offset_len
        next_y = current_y - height  

        #add the rectangle to the scene & choose if its hollow or solid
        if random.random() < hollow_pct:
            inner_offset = random.uniform(0.01, min(width, height) / 2)
            scene.add_rect_hollow(x=next_x, y=next_y, w=width, h=height, inner_offset=inner_offset, actuation=actuation[i])
        else:
            scene.add_rect(x=next_x, y=next_y, w=width, h=height, actuation=actuation[i])
        
        robot_data.append((next_x, next_y, width, height, actuation[i], inner_offset))


    if num_actuators == 0: #this will cause it to crash if we dont add an actuator 
        scene.add_rect(x=0.00, y=0.00, w=0.01, h=0.01, actuation=0)
        num_actuators = 1
    
    scene.set_n_actuators(num_actuators+1) #bc 0 counts as an actuator  
    return robot_data, num_actuators+1

gui = ti.GUI("Differentiable MPM", (640, 640), background_color=0xFFFFFF)

def visualize(s, folder):
    aid = actuator_id.to_numpy()
    colors = np.empty(shape=n_particles, dtype=np.uint32)
    particles = x.to_numpy()[s]
    if particles.shape[0] < n_particles:
        diff = n_particles - particles.shape[0]
        particles = np.pad(particles, ((0, diff), (0, 0)), mode='edge')  # Repeat the last value
    elif particles.shape[0] > n_particles:
        particles = particles[:n_particles]  #
    
    actuation_ = actuation.to_numpy()

    for i in range(n_particles):
        color = 0x111111
        try:
            if 0 <= aid[i] < actuation_.shape[1]:  # Ensure index is within bounds
                act = actuation_[s - 1, int(aid[i])]
                color = ti.rgb_to_hex((0.5 - act, 0.5 - abs(act), 0.5 + act))
        except IndexError as e:
            pass
          
        colors[i] = color

    n = min(colors.shape[0], len(particles))
    gui.circles(pos=particles[:n], color=colors[:n], radius=1.5)
    gui.line((0.05, 0.02), (0.95, 0.02), radius=3, color=0x0)

    os.makedirs(folder, exist_ok=True)
    gui.show(f'{folder}/{s:04d}.png')

def visualize_winner(s, folder):
    aid = actuator_id.to_numpy()
    n = min(len(aid), n_particles)
    colors = np.empty(shape=n, dtype=np.uint32)
    particles = x.to_numpy()[s]
    actuation_ = actuation.to_numpy()
    for i in range(n):
        color = 0x111111
        if aid[i] != -1:
            act = actuation_[s - 1, int(aid[i])]
            color = ti.rgb_to_hex((0.5 - act, 0.5 - abs(act), 0.5 + act))
        colors[i] = color
   
    gui.circles(pos=particles, color=colors, radius=1.5)
    gui.line((0.05, 0.02), (0.95, 0.02), radius=3, color=0x0)

    os.makedirs(folder, exist_ok=True)
    gui.show(f'{folder}/{s:04d}.png')


def apply_mutation(row):
    #three possible mutations: make a leg larger, make a leg hollow, move a leg left or right. can also do nothing

    current_x, current_y, current_width, current_height, actuation_type, inner_offset = row

    if random.random() < 0.25: #make it not hollow or hollow of its solid
       if inner_offset == -1:
        new_row =  current_x, current_y, current_width, current_height, actuation_type, (0.7*min(current_width/2, current_height/2))
       else:
        new_row =  current_x, current_y, current_width, current_height, actuation_type, -1
    elif random.random() < 0.5: #make the leg taller or shorter
        new_height = max(0.01,current_height*np.random.normal(1,0.3))
        delta_h = current_height - new_height
        new_row =  current_x, current_y+delta_h, current_width, new_height, actuation_type, inner_offset
    elif random.random() < 0.75: #change the x and y position
        new_row = current_x+np.random.normal(0,current_width/2), current_y, current_width, current_height, actuation_type, inner_offset
    else: #do nothing
        new_row = row 

    return new_row
    
def save_robot_to_csv(robot_data, num_act, filename="robot_optimization_results.csv"):
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(robot_data + [num_act])

def sim_one_robot_mutate(robo_data, num_act, mutate_num):
    new_robo_data = []
    iterations = 50
    scene = Scene()

    new_robo_data.append(robo_data[0])
    for row in robo_data[1:]:
        row = apply_mutation(row)
        new_robo_data.append(row)

    winning_robot(scene,new_robo_data, num_act)
    scene.finalize()

    visualize(0, 'diffmpm/test/design_{}/'.format(mutate_num))

    
    for i in range(n_actuators):
        for j in range(n_sin_waves):
            weights[i, j] = np.random.randn() * 0.01

    for i in range(scene.n_particles):
        x[0, i] = scene.x[i]
        F[0, i] = [[1, 0], [0, 1]]
        actuator_id[i] = scene.actuator_id[i]
        particle_type[i] = scene.particle_type[i]

    losses = []
    for iter in range(iterations):
        with ti.ad.Tape(loss):
            forward()
        l = loss[None] 
        losses.append(l)
        print('i=', iter, 'loss=', l)
        if np.isnan(l):
            print("breaking")
            losses.append(0)
            reset_all_fields()
            reset_gradients()
            break
        learning_rate = 0.1

        for i in range(n_actuators):
            for j in range(n_sin_waves):
                # print(weights.grad[i, j])
                weights[i, j] -= learning_rate * weights.grad[i, j]
            bias[i] -= learning_rate * bias.grad[i]

        if iter % 10 == 0:
            # visualize
            forward(1500)
            for s in range(15, 1500, 16):
                visualize(s, 'diffmpm/test/design_{}/iter{:03d}/'.format(mutate_num, iter))

    min_loss = losses[-1]  #weigh how good it was based on how much "matter" it took
    print('min_loss=', min_loss)
    return min_loss, new_robo_data, num_act


def vis_win(robo_data, num_act, iterations=30):
    for i in range(1):
        scene = Scene()  # Create a new scene for each robot
        winning_robot(scene, robo_data, num_act)  # Generate a robot
        scene.finalize()
        # allocate_fields()

        for i in range(n_actuators):
            for j in range(n_sin_waves):
                weights[i, j] = np.random.randn() * 0.01

        for i in range(scene.n_particles):
            x[0, i] = scene.x[i]
            F[0, i] = [[1, 0], [0, 1]]
            actuator_id[i] = scene.actuator_id[i]
            particle_type[i] = scene.particle_type[i]


        losses = []
        for iter in range(iterations):
            with ti.ad.Tape(loss):
                forward()
            l = loss[None] 
            losses.append(l)
            print('i=', iter, 'loss=', l)
            if np.isnan(l):
                print("breaking")
                losses.append(0)
                break
            learning_rate = 0.1

            for i in range(n_actuators):
                for j in range(n_sin_waves):
                    # print(weights.grad[i, j])
                    weights[i, j] -= learning_rate * weights.grad[i, j]
                bias[i] -= learning_rate * bias.grad[i]

            if iter % 10 == 0:
                # visualize
                forward(1500)
                for s in range(15, 1500, 16):
                    visualize(s, 'diffmpm/iter{:03d}/'.format(iter))

def robo_data_sim(robo_data, num_act, first=False, iterations=30):
    if first:
        scene = Scene()  # Create a new scene for each robot
        winning_robot(scene, robo_data, num_act)  # Generate a robot
        scene.finalize()
        allocate_fields() #only can allocate once
        print("FIRST")
    else:
        scene = Scene()  # Create a new scene for each robot
        winning_robot(scene, robo_data, num_act)  # Generate a robot
        scene.finalize()

    for i in range(n_actuators):
        for j in range(n_sin_waves):
            weights[i, j] = np.random.randn() * 0.01

    for i in range(scene.n_particles):
        x[0, i] = scene.x[i]
        F[0, i] = [[1, 0], [0, 1]]
        actuator_id[i] = scene.actuator_id[i]
        particle_type[i] = scene.particle_type[i]

    losses = []
    for iter in range(iterations):
        with ti.ad.Tape(loss):
            forward()
        l = loss[None] 
        losses.append(l)
        print('i=', iter, 'loss=', l)
        if np.isnan(l):
            print("breaking")
            losses.append(0)
            break
        learning_rate = 0.1

        for i in range(n_actuators):
            for j in range(n_sin_waves):
                # print(weights.grad[i, j])
                weights[i, j] -= learning_rate * weights.grad[i, j]
            bias[i] -= learning_rate * bias.grad[i]

        if iter  % 20 == 0:
            # visualize
            forward(1500)
            for s in range(15, 1500, 16):
                visualize(s, 'diffmpm/iter{:03d}/'.format(iter))
    return losses[-1], robo_data, num_act


#creates a robot from robot_data
def winning_robot(scene, robot_data, num_actuators):

    scene.set_offset(0.1, 0.03)
    #add each limb from the robot_data array
    for row in robot_data: 
        # print("adding", row)
        current_x, current_y, current_width, current_height, actuation_type, inner_offset = row
        if inner_offset == -1:
            scene.add_rect(x=current_x, y=current_y, w=current_width, h=current_height, actuation=actuation_type)
        else: 
            scene.add_rect_hollow(x=current_x, y=current_y, w=current_width, h=current_height, inner_offset=inner_offset, actuation=actuation_type)
    scene.set_n_actuators(num_actuators)
    return True   

def main_sim():
    all_losses = []

    #run code once to get a starting robot
    robo_data, num_act  = robot_eight_uniform()

    min_loss, robo_data, num_act = robo_data_sim(robo_data, num_act, True, 50)
    print("INITIAL ROBOT LOSS:", min_loss)
    leading_design = robo_data
    leading_design_num_act = num_act
    leading_loss = min_loss
    all_losses.append(leading_loss)
    vis_win(leading_design, leading_design_num_act, 1)
    
    # Entering mutation loop given the initial leading design
    print("ENTERING MUTATION LOOP")
    for i in range(50): 
        #mutate the current leading design 
        min_loss, new_robo_data, num_act = sim_one_robot_mutate(leading_design, leading_design_num_act, i)

        #if the mutation is positive (lower loss), set it as the new best design 
        if min_loss < leading_loss: 
            leading_design = new_robo_data
            leading_design_num_act = num_act
            leading_loss = min_loss
            save_robot_to_csv(leading_design, leading_design_num_act, "robot_optimization_results.csv")
        all_losses.append(leading_loss)

    print("OVERALL BEST LOSS:", leading_loss)
    print("FINAL WINNING DESIGN")
    print(leading_design)
    print(leading_design_num_act)
    vis_win(leading_design, leading_design_num_act, 100)
    
    #graph the loss vs iteration  
    x_val = range(1, len(all_losses) + 1)
    plt.plot(x_val, all_losses, marker='o', linestyle='-')
    plt.xticks(x_val)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Loss per Iteration")
    plt.show()


if __name__ == '__main__':
    main_sim()
