import torch
import numpy as np
import timeit
import matplotlib.pyplot as plt
import math
from scipy import optimize
from sympy import lambdify
from sympy import symbols
import sympy as sym
from numpy import random
import torch.nn.functional as F
from pysr import PySRRegressor
from scipy.special import logsumexp
from stable_V import ReHU, ICNN
from torch import nn, optim
import pandas as pd


# Definning Symbolic Regression Model
model_1 = PySRRegressor(
    niterations=40,  # < Increase me for better results
    binary_operators=["+", "*", "-", "/"],
    unary_operators=[
        "square",  "cos", "sin"],
    equation_file='lyapunov1.csv',
    optimize_probability=1.0,  # ensure constant optimization after each iteration
    loss="loss(prediction, target) = (prediction - target)^2 ",
    turbo = True,
)

def f_value(x):
    # Dynamics representation for Neural Network

    # parameter setting
    c = 2
    velocity = 6

    # derivative calculation
    y = []
    for r in range(0,len(x)):
        f = [ velocity * torch.sin(x[r][1]),
              -x[r][1] - c * velocity * (torch.sin(x[r][1]) * x[r][0]) / x[r][1]]
        y.append(f)
    y = torch.tensor(y)
    return y

def derivative_calculate(v, u1, u2):
    # Dynamics representation for root finding verification
    
    #  parameter setting
    c = 2
    velocity = 6

    u1_deri = velocity * sym.sin(u2)
    u2_deri = - u2 - c * velocity * (sym.sin(u2) * u1) / u2

    return v.diff(u1) * u1_deri + v.diff(u2) * u2_deri


def find_root(func, guess):
    # Given an analytical function, find the root of that function

    root = optimize.fsolve(func, guess)

    # check if root is in the domain
    outer_root = False
    if (root[0]< - 2):
        outer_root = True
    elif (root[0]> 2):
        outer_root = True

    if (root[1] < -math.pi):
        outer_root = True
    elif (root[1] > math.pi):
        outer_root = True
    
    # check if root is at origin
    if (root[1] == 0):
        outer_root = True
        
    res = func(root)

    # check if root is valid
    res_success = (np.abs(res[0])<=0.001 and ((np.linalg.norm(root)>0.001) and (not outer_root)))
    if np.linalg.norm(guess)==0:
        res_success = ((np.abs(res[0])<=0.001 and (np.linalg.norm(root)<0.001)) or (np.abs(res[0])>0.0001))

    return root, res_success


def counter_exp_finder_deri(root1, func1, root2, func2, num=200):
    # Find counter example from analytical function
    counter_example = []
    pd_counter_example = []
    
    # random distance generation, model
    distance = np.linspace(np.array([0,0]),np.array([-1.5, 1.5]),num)
    for i in range(len(distance)):
        distance[i][1] = distance[i][1] + np.random.randn()
    distance = np.concatenate((distance,random.randn(distance.shape[0],distance.shape[1])*0.25),axis=0)

    # check lyapunov conditions
    for j in range(len(distance)):
        # random checking points generation from random distance
        root1_minus = root1 - distance[j]
        root1_minus[0] = np.clip(root1_minus[0], -2, 2)
        root1_minus[1] = np.clip(root1_minus[1], -2, 2)

        root1_plus = root1 + distance[j]
        root1_plus[0] = np.clip(root1_plus[0], -2, 2)
        root1_plus[1] = np.clip(root1_plus[1], -2, 2)

        root2_minus = root2 - distance[j]
        root2_minus[0] = np.clip(root2_minus[0], -math.pi, math.pi)
        root2_minus[1] = np.clip(root2_minus[1],-math.pi, math.pi)

        root2_plus = root2 + distance[j]
        root2_plus[0] = np.clip(root2_plus[0],-math.pi, math.pi)
        root2_plus[1] = np.clip(root2_plus[1],-math.pi, math.pi)

        # Lyapunov valud and Lie derivative value of checking points
        value1 = func1(root1_minus)
        value2 = func1(root1_plus)

        value3 = func2(root2_minus)
        value4 = func2(root2_plus)

        # check for positive semi-definite condition
        if value1[0] <= 0:
            if ((root1_minus[0] == 0) and (root1_minus[1] == 0)):
                if value1[0] < 0:
                    pd_counter_example.append((root1_minus).copy())
                else:
                    pd_counter_example.append((root1_minus).copy())
        
        if value2[0] <= 0:
            if ((root1_plus[0] == 0) and (root1_plus[1] == 0)):
                if value1[0] < 0:
                    pd_counter_example.append((root1_plus).copy())
                else:
                    pd_counter_example.append((root1_plus).copy())
        
        # check for non-negative condition on Lie derivative
        if value3[0] >= 0:
            if ((root2_minus)[0] == 0) and ((root2_minus)[1] == 0):
                if value3[0] > 0.0001:
                    counter_example.append((root2_minus).copy())
            else:
                if value3[0] >= 0.0001:
                    counter_example.append((root2_minus).copy())
        if value4[0] >= 0:
            if ((root2_plus)[0] == 0) and ((root2_plus)[1] == 0):
                if value4[0] > 0.0001:
                    counter_example.append((root2_plus).copy())
            else:
                if value4[0] >= 0.0001:
                    counter_example.append((root2_plus).copy())
        
        # check if function value at root root equals 0
        if func1([0, 0])[0] != 0:
            pd_counter_example.append([0, 0])

    return counter_example, pd_counter_example

def check_options(sympy_expr, model, guess_pos, guess_neg, guess_zero):

    # Perform root finding verification for an analytical function

    valid = False
    x0 = symbols('x0')
    x1 = symbols('x1')
    numpy_expr = lambdify((x0, x1), sympy_expr, "numpy")
    v_dot = derivative_calculate(sympy_expr, x0, x1)
    numpy_v_dot = lambdify((x0, x1), v_dot, "numpy")

    def function1(x):
        return [numpy_expr(np.array(x[0]), np.array(x[1])), 0]

    def function2(x):
        if x[1] == 0:
            return [-0.0001, 0]
        return [logsumexp(numpy_v_dot(np.array(x[0]), np.array(x[1]))), 0]

    # find root(s) for function itself and its Lie derivative
    root1_pos, root_1_pos_sucs = find_root(function1,guess_pos)
    root1_neg, root_1_neg_sucs = find_root(function1,guess_neg)
    root1_zero, root_1_zero_sucs = find_root(function1,guess_zero)
    
    root2_pos, root_2_pos_sucs = find_root(function2,guess_pos)
    root2_neg, root_2_neg_sucs = find_root(function2,guess_neg)
    root2_zero, root_2_zero_sucs = find_root(function2,guess_zero)
    
    # ramdom sample checks around root for Lyapunov conditions
    counter_example_pos, pd_pos = counter_exp_finder_deri(root1_pos, function1, root2_pos, function2)
    counter_example_neg, pd_neg = counter_exp_finder_deri(root1_neg, function1, root2_neg, function2)
    counter_example_zero, pd_zero = counter_exp_finder_deri(root1_zero, function1, root2_zero, function2)

    # concat counterexamples from different initial guesses of root location
    counter_exp = []
    if len(counter_example_pos)!=0:
        counter_exp.append(counter_example_pos)
    if len(counter_example_neg)!=0:
        counter_exp.append(counter_example_neg)
    if len(counter_example_zero)!=0:
        counter_exp.append(counter_example_zero)
    if len(counter_exp) != 0:
        counter_exp = np.concatenate((counter_exp))
    
    # condition checking and counter example gathering
    counter_exp = np.array(counter_exp)
    root2_valid = (((not root_2_neg_sucs) and (not root_2_pos_sucs))) #zero root or no root
    if ((root2_valid) and (len(counter_exp) <= 0 ) and ((len(pd_pos) + len(pd_neg) + len(pd_zero)) == 0)): 
        valid = True
        print('we nailed it!')
        risk = 0
        return valid, counter_exp, risk
    else:
        if len(counter_exp)==0:
            valid = False
            print('we nearly nailed it!')
            risk = 0
            return valid, counter_exp, risk

        lyap = -numpy_expr(counter_exp[:,0],counter_exp[:,1])
        lyap = lyap * (lyap>0)

        lie = numpy_v_dot(counter_exp[:,0],counter_exp[:,1])
        lie = lie * (lie>0)
        risk = 0.1*(lyap).sum() + (lie).sum()

    return valid, counter_exp, risk


if __name__ == '__main__':
    
    # Neural Netwok Learning Setting

    N = 500             # sample size
    D_in = 2            # input dimension
    D_out = 1           # output dimension
    hp = 128            # hidden layer size
    h=100
    rehu=7
    torch.manual_seed(13)
    
    # input data 
    x = torch.Tensor(N, D_in).uniform_(-2, 2).double().to('cuda:0')
    x[:,1] = torch.Tensor(N, 1).uniform_(-math.pi, math.pi).squeeze().double()
    x_0 = torch.zeros([1, 2]).double().to('cuda:0')
    out_iteration = 0
    
    valid = False
    pysr_res = False

    # Neural Lyapunov Function training
    while out_iteration < 2 and not valid:

        start = timeit.default_timer()
        model = ICNN([D_in, hp, hp, D_out], activation=ReHU(float(7.0))).to('cuda:0').double()

        L = []
        a = 0
        max_iteration = 5000
        learning_rate = 0.001
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

        while a < max_iteration and not valid:
            V_candidate= model(x)
            counter_num = torch.count_nonzero(F.relu(-V_candidate.clone()))
            X0 = model(x_0)
            f = f_value(x.detach().cpu())

            # Compute Lie-Derivative
            compute_batch_jacobian = torch.vmap(torch.func.jacrev(model))
            grad_V = compute_batch_jacobian(x).squeeze()
            
            L_V = torch.matmul(grad_V.unsqueeze(1), f.unsqueeze(-1).to('cuda:0')).squeeze()
            
            counter_num_lie = torch.count_nonzero(F.relu(L_V.clone()))
            
            # Calculate Lyapunov risk
            Lyapunov_risk = F.relu(L_V+0.1).sum()
            dis = F.mse_loss(V_candidate.squeeze(), ((torch.square(x[:,0]) + 0.5*torch.square(x[:,1])).to('cuda:0').squeeze())) 

            # measure approximation error from regression formula
            if pysr_res:
                x_numpy = x.cpu().clone().detach().numpy()
                prsr_V = numpy_expr(x_numpy[:,0],x_numpy[:,1])
                dis = F.mse_loss(V_candidate.squeeze(), torch.tensor(prsr_V).to('cuda:0').squeeze())
            
            
            L.append(Lyapunov_risk.item())
            optimizer.zero_grad()
            Lyapunov_risk.backward()
            optimizer.step()
            # if a % 200 ==0:
               # scheduler.step()

            # Verificaiton process
            if a % 50 == 0:
                print(a, "Lyapunov Risk=",Lyapunov_risk.item(), dis.item())
                print('numbr of real counter_exp: ', counter_num.detach().cpu().numpy(), counter_num_lie.detach().cpu().numpy())

                # Generate data for regression
                input = torch.Tensor(N, D_in).uniform_(-2, 2).double()
                input[:,1] = torch.Tensor(N, 1).uniform_(-math.pi, math.pi).squeeze().double()
                input[0,:] = torch.zeros((2)).double()

                predict = np.zeros(len(input))

                for i in range(len(input)):
                    predict[i] = model(input[i].to('cuda:0'))[0].detach().cpu().numpy() 
                
                # perform symbolic regression
                model_1.fit(input, predict)
                num_eq = len(model_1.equations_)
                
                print('===========Verifying==========')
                guess_pos = random.rand(1, 2) * 2
                guess_neg = -random.rand(1, 2) * 2
                guess_zero = np.zeros((1,2))
                counter_ex_list = []
                risk_list = []
                id_list = []
                
                # Root Finding Verification for each analytical formula
                for i in range(num_eq):
                    sympy_expr = model_1.equations_.iloc[-i].sympy_format
                    if sympy_expr.is_constant():
                        continue

                    valid, counter_exp, risk = check_options(sympy_expr, model, guess_pos, guess_neg, guess_zero)
                    if valid:
                        stop = timeit.default_timer()
                        print(sympy_expr)
                        print("Total time: ", stop - start)
                        print('Total iteration: ', a)
                        break

                    else:
                        risk_list.append(risk)
                        counter_ex_list.append(counter_exp)
                        id_list.append(i)

                # store the neural network model once succeed
                if valid:
                    torch.save(model.state_dict(), 'nn_lyapunov.pt')
                    break

                # find the formula with closest approximation to Neural Lyapunov Function
                express_id = np.argmin(np.array(risk_list))
                sympy_expr = model_1.equations_.iloc[-id_list[express_id]].sympy_format
                x0 = symbols('x0')
                x1 = symbols('x1')
                numpy_expr = lambdify((x0, x1), sympy_expr, "numpy")
                pysr_res = True
                v_dot = derivative_calculate(sympy_expr, x0, x1)
                numpy_v_dot = lambdify((x0, x1), v_dot, "numpy")
                
                # counter example feedback
                if not valid:
                    x = torch.cat((x, torch.tensor(np.array(counter_exp)).to('cuda:0')), 0)
                    if x.shape[0]>2000:
                        x = x[-2000:,:]
                    print(f"Not a Lyapunov function. Found {len(counter_exp)} counter examples!!!!!")
                    print('now we use: ', sympy_expr)
            a += 1
        out_iteration += 1