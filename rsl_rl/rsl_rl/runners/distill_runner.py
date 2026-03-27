# -*- coding: utf-8 -*-
import torch
import torch.optim as optim

from rsl_rl.rsl_rl.algorithms.distill_ppo import DistillPPO

class DistillRunner:
    """
    Teacher frozen + Student train
    """
    def __init__(self, env, model, train_cfg, device="cpu"):
        self.env = env
        self.model = model
        self.cfg = train_cfg
        self.device = device

        self.optimizer = optim.Adam(
            self.model.student_actor.parameters(),
            lr=getattr(train_cfg, "learning_rate", 1e-4)
        )
        self.algo = DistillPPO(
            student_model=self.model,
            optimizer=self.optimizer,
            distill_coef=getattr(train_cfg, "distill_coef", 1.0)
        )

    def learn(self, num_learning_iterations=1000):
        obs = self.env.reset()
        for it in range(num_learning_iterations):
            obs_teacher = obs["teacher"]
            obs_student = obs["student"]

            with torch.no_grad():
                action_teacher = self.model.act_teacher(obs_teacher)

            # student action用于驱动环境（也可混合teacher action，先简）
            action_student = self.model.act_student(obs_student)
            obs, rewards, dones, infos = self.env.step(action_student)

            stats = self.algo.update(obs_student, action_teacher)

            if it % 50 == 0:
                print(f"[Distill] iter={it} loss={stats['loss']:.6f}")