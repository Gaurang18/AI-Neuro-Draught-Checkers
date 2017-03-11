import numpy as np

# params has dimensions of layers, lambda, alpha etc.

##OWN
def initialize_network(params):
    inp_dim = params['input_dimension']
    h1_dim = params['h1_dimension']
    V_InHide = np.random.rand(h1_dim, inp_dim + 1)/10
    V_HideOut = np.random.rand(1, h1_dim + 1)/10
    e_InHidden = np.zeros((h1_dim, inp_dim + 1))
    e_HiddenOut = np.zeros((1, h1_dim + 1))
    return [V_InHide, V_HideOut, e_InHidden, e_HiddenOut]

##OWN
def evaluateNN(network, input):
    V_IH = network[0]
    V_HO = network[1]

    hideSum = np.dot(V_IH, np.append(input, 1))
    hide = 1 / (1 + np.exp(-hideSum))
    opsum = np.dot(V_HO, np.append(hide, 1))
    out = 1 / (1 + np.exp(-opsum))
    return out[0]

##OWN
def backpropagate(params, network, opnext, opresent, inpnext):
    V_IH = network[0]
    V_HO = network[1]
    e_IH = network[2]
    e_HO = network[3]

    alpha = params['alpha']
    lmbda = params['lmbda']

    V_HO = V_HO + alpha * (opnext - opresent) * e_HO
    V_IH = V_IH + alpha * (opnext - opresent) * e_IH

    hideSum = np.dot(V_IH, np.append(inpnext, 1))
    hide = 1 / (1 + np.exp(-hideSum))
    hidebias = np.append(hide, 1)
    hide = np.reshape(hide, (1, hide.shape[0]))

    nxtop = evaluateNN(network, inpnext)
    inpnext = np.append(inpnext, 1)
    newinp = inpnext.reshape((1, inpnext.shape[0]))

    e_HO = lmbda * e_HO + (1 - nxtop) * nxtop * hidebias.T
    e_IH = lmbda*e_IH+(1-nxtop)*nxtop*np.dot((((1-hide)*hide)*V_HO[0,0:-1].T).T, newinp)
    return [V_IH, V_HO, e_IH, e_HO]
