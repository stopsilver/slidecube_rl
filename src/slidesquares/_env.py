"""
Generic square env representation and registry
"""
import logging
import random

log = logging.getLogger("square.env")
_registry = {}


class SquareEnv:
    def __init__(self, name, state_type, action_type, initial_state, is_goal_pred,
                 action_enum, transform_func, inverse_action_func,
                 same_layer_action,
                 render_func, encoded_shape, encode_func,
                 state_cost_func):
        self.name = name
        self._state_type = state_type
        self._action_type = action_type
        self.initial_state = initial_state
        self._is_goal_pred = is_goal_pred
        self.action_enum = action_enum
        self._transform_func = transform_func
        self._inverse_action_func = inverse_action_func
        self._same_layer_action = same_layer_action
        self._render_func = render_func
        self.encoded_shape = encoded_shape
        self._encode_func = encode_func
        self._state_cost_func=state_cost_func

    def __repr__(self):
        return "SquareEnv(%r)" % self.name

    # wrapper functions
    def is_goal(self, state):
        assert isinstance(state, self._state_type)
        return self._is_goal_pred(state)

    def transform(self, state, action):
        assert isinstance(state, self._state_type)
        assert isinstance(action, self._action_type)
        return self._transform_func(state, action)

    def inverse_action(self, action):
        return self._inverse_action_func(action)

    def render(self, state):
        assert isinstance(state, self._state_type)
        return self._render_func(state)

    def encode_inplace(self, state):
        assert isinstance(state, self._state_type)
        return self._encode_func(state)

    # Utility functions
    def sample_action(self, prev_action=None):
        while True:
            res = self.action_enum[random.randrange(len(self.action_enum))]
            # if prev_action is None or self.inverse_action(res) != prev_action:
            #     return res
            if prev_action is None :
                return res  
            if self._same_layer_action(res,prev_action) == False :
                return res            

    def scramble(self, actions):
        s = self.initial_state
        for action in actions:
            s = self.transform(s, action)
        return s

    def is_state(self, state):
        return isinstance(state, self._state_type)

    def scramble_square(self, scrambles_count, return_inverse=False, include_initial=False):
        """
        Generate sequence of random square scrambles
        :param scrambles_count: count of scrambles to perform
        :param return_inverse: if True, inverse action is returned
        :return: list of tuples (depth, state[, inverse_action])
        """
        assert isinstance(scrambles_count, int)
        assert scrambles_count > 0

        state = self.initial_state
        result = []
        if include_initial:
            assert not return_inverse
            result.append((1, state))
        prev_action = None
        for depth in range(scrambles_count):
            action = self.sample_action(prev_action=prev_action)
            state = self.transform(state, action)
            prev_action = action
            if return_inverse:
                inv_action = self.inverse_action(action)
                res = (depth+1, state, inv_action)
            else:
                res = (depth+1, state)
            result.append(res)
        return result

    def explore_state(self, state):
        """
        Expand square state by applying every action to it
        :param state: state to explore
        :return: tuple of two lists: [states reachable], [flag that state is initial]
        """
        res_states, res_flags = [], []
        for action in self.action_enum:
            new_state = self.transform(state, action)
            is_init = self.is_goal(new_state)
            res_states.append(new_state)
            res_flags.append(is_init)
        return res_states, res_flags

    def state_cost(self, state):
        return self._state_cost_func(state)


def register(square_env):
    assert isinstance(square_env, SquareEnv)
    global _registry

    if square_env.name in _registry:
        log.warning("Square environment %s is already registered, ignored", square_env)
    else:
        _registry[square_env.name] = square_env


def get(name):
    assert isinstance(name, str)
    return _registry.get(name)


def names():
    return list(sorted(_registry.keys()))
