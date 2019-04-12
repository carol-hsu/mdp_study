import mdptoolbox
import argparse
import numpy as np

def machine_state_example(num, crash_prob, reward):
    #
    # action[3] = ["no move", "add one machine", "remove one machine"]
    #
    S = num + 1
    p = crash_prob
    P = np.zeros((3, S, S))
    P[0, :, :] = (1 - p) * np.diag(np.ones(S))
    P[0, :, 0] = p
    P[0, 0, 0] = 1

    P[1, :, :] = (1 - p) * np.diag(np.ones(S - 1), 1)
    P[1, :, 0] = p
    P[1, S-1, S-1] = 1-p

    P[2, :, :] = (1 - p) * np.diag(np.ones(S - 1), -1)
    P[2, :, 0] = p
    P[2, 0, 0] = 1
    P[2, 1, 0] = 1

    R = np.zeros((S, 3))
    R[:, 0] = [ a*reward for a in range(round(S/2))] + [ round(S/2)*reward - a+1 for a in range(round(S/2), S)]
    R[:, 1][:S-1] = R[:, 0][1:]
    R[:, 2][1:] = R[:, 0][:S-1]

    return P, R

def customer_state_example(add_prob, lose_prob):
    #
    # states = [ machine off/no customer, machine on/ no customer,
    #            machine off/has customer, machine on/has customer ]
    # action = ["no move", "add one machine", "remove one machine"]
    #
    P = np.zeros((3, 4, 4))
    P[0, :, :] = [[1, 0, 0, 0], \
                  [0, 1-add_prob, 0, add_prob], \
                  [lose_prob, 0, 1-lose_prob, 0], \
                  [0, 0, 0, 1]]

    P[1, :, :] = [ P[0, 1, :], P[0, 1, :], \
                   P[0, 3, :], P[0, 3, :] ]

    P[2, :, :] = [ P[0, 0, :], P[0, 0, :], \
                   P[0, 2, :], P[0, 2, :] ]

    R = [ [ 0, -1,  0], [-1,  0,  0], [ 0,  2,  0], [ 2,  2,  0] ]
    np_R = np.array([np.array(r) for r in R]) 
    return P, np_R


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--case", default=0, type=int,
            help="Which case to run [0] machine-only-state case [1] machine-with-customer-state case")
    ap.add_argument("-d", "--discount", default=0.96, type=float,
            help="Discount for value iteration and policy iteration")
    ap.add_argument("-o", "--operation", default=0, type=int,
            help="Applying which method to find policy [0] value iteration [1] policy iteration [2] RL algo")
    params = vars(ap.parse_args())

    P, R = None, None
    if  params["case"] == 0:
        print("machine-only-state: \n======================")
        P, R = machine_state_example(50, 0.2, 2)
    else:
        print("machine-with-customer-state: \n======================")
        P, R = customer_state_example(0.6, 0.9)

    if params["operation"] > 1:
        print("do RL stuff")
    else:
        op = None

        if params["operation"] == 0:
            print("Using value iteration:")
            op = mdptoolbox.mdp.ValueIteration(P, R, params["discount"])
        else:
            print("Using policy iteration:")
            op = mdptoolbox.mdp.PolicyIteration(P, R, params["discount"])

        op.run()

        print("iters: "+str(op.iter))
        print("time: "+str(op.time*1000)+" ms")
        print("policy: "+str(op.policy))
