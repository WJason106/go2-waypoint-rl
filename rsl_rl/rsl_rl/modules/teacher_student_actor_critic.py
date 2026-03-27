# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

class MLPActor(nn.Module):
    def __init__(self, in_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, action_dim),
        )

    def forward(self, x):
        return self.net(x)

class TeacherStudentActorCritic(nn.Module):
    """
    teacher: frozen
    student: trainable
    """
    def __init__(self, teacher_model: nn.Module, student_in_dim: int, action_dim: int):
        super().__init__()
        self.teacher = teacher_model
        self.student_actor = MLPActor(student_in_dim, action_dim)

        for p in self.teacher.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def act_teacher(self, obs_teacher):
        # 要求teacher_model提供act()；没有则退化forward
        if hasattr(self.teacher, "act"):
            return self.teacher.act(obs_teacher)
        return self.teacher(obs_teacher)

    def act_student(self, obs_student):
        return self.student_actor(obs_student)