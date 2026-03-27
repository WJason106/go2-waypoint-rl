# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F

class DistillPPO:
    """
    最小蒸馏算法骨架（先做action regression）
    """
    def __init__(self, student_model, optimizer, distill_coef=1.0):
        self.student_model = student_model
        self.optimizer = optimizer
        self.distill_coef = distill_coef

    def update(self, obs_student, action_teacher):
        action_student = self.student_model.act_student(obs_student)
        loss_distill = F.mse_loss(action_student, action_teacher)
        loss = self.distill_coef * loss_distill

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

        return {
            "loss": float(loss.item()),
            "loss_distill": float(loss_distill.item()),
        }