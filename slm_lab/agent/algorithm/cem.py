'''
The cross entropy method algorithm
From Schulman's lecture on DRL https://www.youtube.com/watch?v=aUrX-rP_ss4 and http://rl-gym-doc.s3-website-us-west-2.amazonaws.com/mlss/lab1.html
'''
from slm_lab.agent.algorithm.base import Algorithm
from slm_lab.lib.decorator import lab_api
import numpy as np


class CEM(Algorithm):
    '''
    Cross entropy method agent that works in both discrete and continuous envs
    '''

    @lab_api
    def post_body_init(self):
        '''Initializes the part of algorithm needing a body to exist first.'''
        body = self.agent.nanflat_body_a[0]  # singleton algo
        algorithm_spec = self.agent.spec['algorithm']
        util.set_attr(self, _.pick(algorithm_spec, [
            'gamma',  # the discount factor
            'batch_size',
        ]))
        self.theta_dim = (body.state_dim + 1) * body.action_dim
        self.theta_mean = np.zeros(self.theta_dim)
        self.theta_std = np.ones(self.theta_dim)

        self.thetas = [np.random.normal(self.theta_mean, self.theta_std, self.theta_dim)
                       for _ in range(self.batch_size)]
        self.reset()

    @lab_api
    def body_act_discrete(self, body, state):
        '''Random discrete action'''
        y = state.dot(self.W) + self.b
        action = y.argmax()
        return action

    @lab_api
    def body_act_continuous(self, body, state):
        '''Random continuous action'''
        action = np.clip(state.dot(self.W) + self.b,
                         body.env.u_env.action_space.low, body.env.u_env.action_space.high)
        return action

    @lab_api
    def train(self):
        loss = np.nan
        return loss

    @lab_api
    def update(self):
        explore_var = np.nan
        return explore_var

    def reset(self):
        thetas = [np.random.normal(theta_mean, theta_std, theta_dim)
                  for _ in range(batch_size)]
        # TODO uh need to control to run one game per theta, then reset above
        # TODO update theta from return

        assert len(self.theta) == (
            body.state_dim + 1) * body.action_dim  # +1 for bias
        self.W = self.theta[0: body.state_dim * body.action_dim
                            ].reshape(body.state_dim, body.action_dim)
        self.b = self.theta[body.state_dim * body.action_dim: None]
        if body.is_discrete:
            self.b = self.b.reshape(1, body.action_dim)
