import numpy as np
import operator


def gaussian_prob(x, para_tuple):
    """Compute the probability of a given x value

    Args:
        x (float): observation value
        para_tuple (tuple): contains two elements, (mean, standard deviation)

    Return:
        Probability of seeing a value "x" in a Gaussian distribution.

    Note:
        We simplify the problem so you don't have to take care of integrals.
        Theoretically speaking, the returned value is not a probability of x,
        since the probability of any single value x from a continuous 
        distribution should be zero, instead of the number outputed here.
        By definition, the Gaussian percentile of a given value "x"
        is computed based on the "area" under the curve, from left-most to x. 
        The proability of getting value "x" is zero bcause a single value "x"
        has zero width, however, the probability of a range of value can be
        computed, for say, from "x - 0.1" to "x + 0.1".

    """
    if para_tuple == (None, None):
        return 0.0

    mean, std = para_tuple
    gaussian_percentile = (2 * np.pi * std**2)**-0.5 * \
                          np.exp(-(x - mean)**2 / (2 * std**2))
    return gaussian_percentile


def part_1_a():
    """Provide probabilities for the word HMMs outlined below.

    Word BUY, CAR, and HOUSE.

    Review Udacity Lesson 8 - Video #29. HMM Training

    Returns:
        tuple() of
        (prior probabilities for all states for word BUY,
         transition probabilities between states for word BUY,
         emission parameters tuple(mean, std) for all states for word BUY,
         prior probabilities for all states for word CAR,
         transition probabilities between states for word CAR,
         emission parameters tuple(mean, std) for all states for word CAR,
         prior probabilities for all states for word HOUSE,
         transition probabilities between states for word HOUSE,
         emission parameters tuple(mean, std) for all states for word HOUSE,)


        Sample Format (not complete):
        (
            {'B1': prob_of_starting_in_B1, 'B2': prob_of_starting_in_B2, ...},
            {'B1': {'B1': prob_of_transition_from_B1_to_B1,
                    'B2': prob_of_transition_from_B1_to_B2,
                    'B3': prob_of_transition_from_B1_to_B3,
                    'Bend': prob_of_transition_from_B1_to_Bend},
             'B2': {...}, ...},
            {'B1': tuple(mean_of_B1, standard_deviation_of_B1),
             'B2': tuple(mean_of_B2, standard_deviation_of_B2), ...},
            {'C1': prob_of_starting_in_C1, 'C2': prob_of_starting_in_C2, ...},
            {'C1': {'C1': prob_of_transition_from_C1_to_C1,
                    'C2': prob_of_transition_from_C1_to_C2,
                    'C3': prob_of_transition_from_C1_to_C3,
                    'Cend': prob_of_transition_from_C1_to_Cend},
             'C2': {...}, ...}
            {'C1': tuple(mean_of_C1, standard_deviation_of_C1),
             'C2': tuple(mean_of_C2, standard_deviation_of_C2), ...}
            {'H1': prob_of_starting_in_H1, 'H2': prob_of_starting_in_H2, ...},
            {'H1': {'H1': prob_of_transition_from_H1_to_H1,
                    'H2': prob_of_transition_from_H1_to_H2,
                    'H3': prob_of_transition_from_H1_to_H3,
                    'Hend': prob_of_transition_from_H1_to_Hend},
             'H2': {...}, ...}
            {'H1': tuple(mean_of_H1, standard_deviation_of_H1),
             'H2': tuple(mean_of_H2, standard_deviation_of_H2), ...}
        )
    """

    # accuracy to 3 decimal digits


    # TODO: complete this function.
    #raise NotImplementedError()

    """Word BUY"""
    b_prior_probs = {
        'B1': 0.333,
        'B2': 0.000,
        'B3': 0.000,
        'Bend': 0.000,
    }
    b_transition_probs = {
        'B1': {'B1': 0.625, 'B2': 0.375, 'B3': 0.000, 'Bend': 0.000},
        'B2': {'B1': 0.000, 'B2': 0.625, 'B3': 0.375, 'Bend': 0.000},
        'B3': {'B1': 0.000, 'B2': 0.000, 'B3': 0.625, 'Bend': 0.375},
        'Bend': {'B1': 0.000, 'B2': 0.000, 'B3': 0.000, 'Bend': 1.0},
    }

    # Parameters for end state is not required
    b_emission_paras = {
        'B1': (41.75, 2.773),
        'B2': (58.625, 5.678),
        'B3': (53.125, 5.418),
        'Bend': (None, None)
    }

    """Word CAR"""
    c_prior_probs = {
        'C1': 0.333,
        'C2': 0.000,
        'C3': 0.000,
        'Cend': 0.000,
    }
    c_transition_probs = {
        'C1': {'C1': 0.667, 'C2': 0.333, 'C3': 0.000, 'Cend': 0.000},
        'C2': {'C1': 0.000, 'C2': 0.0, 'C3': 1.0, 'Cend': 0.000},
        'C3': {'C1': 0.000, 'C2': 0.000, 'C3': 0.800, 'Cend': 0.200},
        'Cend': {'C1': 0.000, 'C2': 0.000, 'C3': 0.000, 'Cend': 1.0},
    }
    # Parameters for end state is not required
    c_emission_paras = {
        'C1': (35.667, 4.899),
        'C2': (43.667, 1.700),
        'C3': (44.200, 7.341),
        'Cend': (None, None)
    }

    """Word HOUSE"""
    h_prior_probs = {
        'H1': 0.333,
        'H2': 0.000,
        'H3': 0.000,
        'Hend': 0.000,
    }
    # Probability of a state changing to another state.
    h_transition_probs = {
        'H1': {'H1': 0.667, 'H2': 0.333, 'H3': 0.0, 'Hend': 0.0},
        'H2': {'H1': 0.0, 'H2': 0.857, 'H3': 0.143, 'Hend': 0.0},
        'H3': {'H1': 0.0, 'H2': 0.0, 'H3': 0.812, 'Hend': 0.188},
        'Hend': {'H1': 0.0, 'H2': 0.0, 'H3': 0.0, 'Hend': 1.0},
    }
    # Parameters for end state is not required
    h_emission_paras = {
        'H1': (45.333, 3.972),
        'H2': (34.952, 8.127),
        'H3': (67.438, 5.733),
        'Hend': (None, None)
    }

    return (b_prior_probs, b_transition_probs, b_emission_paras,
            c_prior_probs, c_transition_probs, c_emission_paras,
            h_prior_probs, h_transition_probs, h_emission_paras,)


def viterbi(evidence_vector, states, prior_probs,
            transition_probs, emission_paras):
    """Viterbi Algorithm to calculate the most likely states give the evidence.

    Args:
        evidence_vector (list): List of right hand Y-axis positions (interger).

        states (list): List of all states in a word. No transition between words.
                       example: ['B1', 'B2', 'B3', 'Bend']

        prior_probs (dict): prior distribution for each state.
                            example: {'X1': 0.25,
                                      'X2': 0.25,
                                      'X3': 0.25,
                                      'Xend': 0.25}

        transition_probs (dict): dictionary representing transitions from each
                                 state to every other state.

        emission_paras (tuple): parameters of Gaussian distribution 
                                from each state.

    Return:
        tuple of
        ( A list of states the most likely explains the evidence,
          probability this state sequence fits the evidence as a float )

    Note:
        You are required to use the function gaussian_prob to compute the
        emission probabilities.

    """
    
    # TODO: complete this function.
    #raise NotImplementedError()
    # print transition_probs

    #print len(transition_probs)
    # Init
    if len(evidence_vector) ==0:
        return (None, 0)

    V = [{}]
    # pay attention mid way zero
    for i in states:
        # print len(states)
        prob = prior_probs[i]*gaussian_prob(evidence_vector[0], emission_paras[i])
        V[0][i] = {"prob": prob, "prev": None}

    for j in range(1, len(evidence_vector)):
        V.append({})
        for state in states:
            # record the prob of next level
            p = np.zeros(len(states))
            # count every instance
            for st, prev_st in enumerate(states):
                if V[j - 1][prev_st]['prob'] == 0.0:
                    p[st] = 0.0
                elif state in transition_probs[prev_st]:
                    prob1 = transition_probs[prev_st][state] * gaussian_prob(evidence_vector[j], emission_paras[state])
                    p[st] = V[j - 1][prev_st]['prob'] * prob1
                else:
                    p[st] = 0.0

            index = np.argmax(p)
            # print index
            prev_state = states[index]

            V[j][state] = {"prob": p[np.argmax(p)], "prev": prev_state}


    sequence = []
    # probability = 0.0
    max_prob = max(value["prob"] for value in V[-1].values())
    if max_prob != 0.0:
        previous = None
        for st, pr in V[-1].items():
            if pr["prob"] ==max_prob:
                sequence.append(st)
                previous = st
                break
        #bfrom bac to construct the path based on the prev
        for i in range(len(V)-2,-1,-1):
            sequence.insert(0, V[i+1][previous]['prev'])
            # Update to tracback upper level
            previous = V[i+1][previous]['prev']

    probability = max_prob
    # print probability
    return sequence, probability


def part_2_a():
    """Provide probabilities for the word HMMs outlined below.

    Now, at each time frame you are given with 2 observations (right hand Y
    position & left hand Y position). Use the result you derived in
    part_1_a, accompany with the provided probability for left hand, create
    a tuple of (right-y, left-y) to represent high-dimention transition & 
    emission probabilities.
    """

    # TODO: complete this function.
    # raise NotImplementedError()
    #
    """Word BUY"""
    b_prior_probs = {
        'B1': 0.333,
        'B2': 0.000,
        'B3': 0.000,
        'Bend': 0.000,
    }
    # example: {'B1': {'B1' : (right-hand Y, left-hand Y), ... }
    b_transition_probs = {
        'B1': {'B1': (0.625, 0.700), 'B2': (0.375, 0.300), 'B3': (0., 0.), 'Bend': (0., 0.)},
        'B2': {'B1': (0., 0.), 'B2': (0.625, 0.050), 'B3': (0.375, 0.950), 'Bend': (0., 0.)},
        'B3': {'B1': (0., 0.), 'B2': (0., 0.), 'B3': (0.625, 0.727), 'Bend': (0.125, 0.091), 'C1': (0.125, 0.091),
               'H1': (0.125, 0.091)},
        'Bend': {'B1': (0., 0.), 'B2': (0., 0.), 'B3': (0., 0.), 'Bend': (1.000, 1.000)},
    }
    # example: {'B1': [(right-mean, right-std), (left-mean, left-std)] ...}
    b_emission_paras = {
        'B1': [(41.750, 2.773), (108.2, 17.314)],
        'B2': [(58.625, 5.678), (78.67, 1.886)],
        'B3': [(53.125, 5.418), (64.182, 5.573)],
        'Bend': [(None, None), (None, None)]
    }

    """Word Car"""
    c_prior_probs = {
        'C1': 0.333,
        'C2': 0.000,
        'C3': 0.000,
        'Cend': 0.000,
    }
    c_transition_probs = {
        'C1': {'C1': (0.667, 0.700), 'C2': (0.333, 0.300), 'C3': (0., 0.), 'Cend': (0., 0.)},
        'C2': {'C1': (0., 0.), 'C2': (0.000, 0.625), 'C3': (1.000, 0.375), 'Cend': (0., 0.)},
        'C3': {'C1': (0., 0.), 'C2': (0., 0.), 'C3': (0.800, 0.625), 'Cend': (0.067, 0.125), 'B1': (0.067, 0.125),
               'H1': (0.067, 0.125)},
        'Cend': {'C1': (0., 0.), 'C2': (0., 0.), 'C3': (0., 0.), 'Cend': (1.000, 1.000)},
    }
    c_emission_paras = {
        'C1': [(35.667, 4.899), (56.3, 10.659)],
        'C2': [(43.667, 1.700), (37.11, 4.306)],
        'C3': [(44.200, 7.341), (50.000, 7.826)],
        'Cend': [(None, None), (None, None)]
    }

    """Word HOUSE"""
    h_prior_probs = {
        'H1': 0.333,
        'H2': 0.000,
        'H3': 0.000,
        'Hend': 0.000,
    }
    h_transition_probs = {
        'H1': {'H1': (0.667, 0.700), 'H2': (0.333, 0.300), 'H3': (0., 0.), 'Hend': (0., 0.)},
        'H2': {'H1': (0., 0.), 'H2': (0.857, 0.842), 'H3': (0.143, 0.158), 'Hend': (0., 0.)},
        'H3': {'H1': (0., 0.), 'H2': (0., 0.), 'H3': (0.812, 0.824), 'Hend': (0.063, 0.059), 'B1': (0.063, 0.059),
               'C1': (0.063, 0.059)},
        'Hend': {'H1': (0., 0.), 'H2': (0., 0.), 'H3': (0., 0.), 'Hend': (1.000, 1.000)},
    }
    h_emission_paras = {
        'H1': [(45.333, 3.972), (53.600, 7.392)],
        'H2': [(34.952, 8.127), (37.168, 8.875)],
        'H3': [(67.438, 5.733), (74.176, 8.347)],
        'Hend': [(None, None), (None, None)]
    }
    return (b_prior_probs, b_transition_probs, b_emission_paras,
            c_prior_probs, c_transition_probs, c_emission_paras,
            h_prior_probs, h_transition_probs, h_emission_paras,)


def multidimensional_viterbi(evidence_vector, states, prior_probs,
                             transition_probs, emission_paras):
    """Decode the most likely word phrases generated by the evidence vector.

    States, prior_probs, transition_probs, and emission_probs will now contain
    all the words from part_2_a.
    """
    # TODO: complete this function.
    # raise NotImplementedError()


    if len(evidence_vector) ==0:
        return (None, 0)

    V = [{}]
    evidence_vector_r, evidence_vector_l = zip(*evidence_vector)

    # pay attention mid way zero
    for i in states:
        # print len(states)
        prob = prior_probs[i]*gaussian_prob(evidence_vector_r[0], emission_paras[i][0])\
               *gaussian_prob(evidence_vector_l[0], emission_paras[i][1])
        V[0][i] = {"prob": prob, "prev": None}


    for j in range(1, len(evidence_vector)):

        V.append({})
        for state in states:
            # record the prob of next level
            p = np.zeros(len(states))
            # count every instance
            for st, prev_st in enumerate(states):
                if V[j - 1][prev_st]["prob"] == 0.0:
                    p[st] = 0.0

                elif state in transition_probs[prev_st]:
                    prob1 = transition_probs[prev_st][state][0] * transition_probs[prev_st][state][1]\
                            *gaussian_prob(evidence_vector_r[j], emission_paras[state][0])*gaussian_prob(evidence_vector_l[j], emission_paras[state][1])

                    p[st] = V[j - 1][prev_st]["prob"] * prob1
                else:
                    p[st] = 0.0

            index = np.argmax(p)
            # print index
            prev_state = states[index]

            V[j][state] = {"prob": p[np.argmax(p)], "prev": prev_state}


    sequence = []
    # probability = 0.0
    max_prob = max(value["prob"] for value in V[-1].values())
    if max_prob != 0.0:
        previous = None
        for st, pr in V[-1].items():
            if pr["prob"] ==max_prob:
                sequence.append(st)
                previous = st
                break

        # from back to construct the path based on the prev stored
        for i in range(len(V)-2,-1,-1):
            sequence.insert(0, V[i+1][previous]["prev"])
            # Update to tracback upper level
            previous = V[i+1][previous]["prev"]

    probability = max_prob

    print sequence
    print probability
    # print "*****************************"
    return sequence, probability


def return_your_name():
    """Return your name
    """
    # TODO: finish this

    return "yyang773"
    # raise NotImplementedError()
