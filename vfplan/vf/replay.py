import torch


class ReplayBuffer:
    """Circular buffer of VF transitions (s, p, r, s', P', mask).

    `state_shapes` lists the shape of each state component (e.g. heightfield and
    character state for the platform VF); `P'` holds all plans sampled at s'.
    """

    def __init__(self, capacity, state_shapes, plan_shape, next_plans_shape):
        self.states = [torch.zeros(capacity, *shape) for shape in state_shapes]
        self.next_states = [torch.zeros(capacity, *shape) for shape in state_shapes]
        self.plans = torch.zeros(capacity, *plan_shape)
        self.next_plans = torch.zeros(capacity, *next_plans_shape)
        self.rewards = torch.zeros(capacity, 1)
        self.masks = torch.ones(capacity, 1)
        self.index = 0
        self.capacity = capacity
        self.reached_capacity = False

    def __len__(self):
        return self.capacity if self.reached_capacity else self.index

    def insert(self, state, plan, reward, next_state, next_plans, mask):
        for buf, value in zip(self.states, state):
            buf[self.index].copy_(torch.as_tensor(value, dtype=torch.float32).reshape(buf.shape[1:]))
        for buf, value in zip(self.next_states, next_state):
            buf[self.index].copy_(torch.as_tensor(value, dtype=torch.float32).reshape(buf.shape[1:]))
        self.plans[self.index].copy_(torch.as_tensor(plan, dtype=torch.float32))
        self.next_plans[self.index].copy_(torch.as_tensor(next_plans, dtype=torch.float32))
        self.rewards[self.index] = float(reward)
        self.masks[self.index] = float(mask)
        if self.index + 1 == self.capacity:
            self.reached_capacity = True
        self.index = (self.index + 1) % self.capacity

    def sample(self, batch_size):
        assert batch_size <= len(self)
        idx = torch.randperm(len(self))[:batch_size]
        return (
            [s[idx] for s in self.states],
            self.plans[idx],
            self.rewards[idx],
            [s[idx] for s in self.next_states],
            self.next_plans[idx],
            self.masks[idx],
        )
