from torch import autograd
from torch.autograd.gradcheck import zero_gradients
import torch.nn.functional as F
from .helpers import *


class AttackBSS:

    def __init__(
            self,
            targeted=True, max_epsilon=16, norm=float('inf'),
            step_alpha=None, num_steps=None, cuda=True, debug=False):

        self.targeted = targeted
        self.eps = 5.0 * max_epsilon / 255.0
        self.num_steps = num_steps or 10
        self.norm = norm
        if not step_alpha:
            if norm == float('inf'):
                self.step_alpha = self.eps / self.num_steps
            else:
                # Different scaling required for L2 and L1 norms to get anywhere
                if norm == 1:
                    self.step_alpha = 500.0  # L1 needs a lot of (arbitrary) love
                else:
                    self.step_alpha = 1.0
        else:
            self.step_alpha = step_alpha
        self.loss_fn = torch.nn.CrossEntropyLoss(size_average=False)
        if cuda:
            self.loss_fn = self.loss_fn.cuda()
        self.debug = debug

    def run(self, model, input, target, batch_idx=0):
        input_var = autograd.Variable(input, requires_grad=True)
        target_var = autograd.Variable(target)
        GT_var = autograd.Variable(target)
        eps = self.eps

        step = 0
        while step < self.num_steps:
            zero_gradients(input_var)
            output = model(input_var)

            if not step:
                GT_var.data = output.data.max(1)[1]

            score = output

            score_GT = score.gather(1, GT_var.unsqueeze(1))
            score_target = score.gather(1, target_var.unsqueeze(1))

            loss = (score_target - score_GT).sum()
            loss.backward()

            step_alpha = self.step_alpha * (GT_var.data == output.data.max(1)[1]).float()
            step_alpha = step_alpha.unsqueeze(1).unsqueeze(1).unsqueeze(1)

            if step_alpha.sum() == 0:
                break

            pert = ((score_GT.data - score_target.data).unsqueeze(1).unsqueeze(1))
            normed_grad = step_alpha * (pert+1e-4) * input_var.grad.data / (l2_norm(input_var.grad.data))

            # perturb current input image by normalized and scaled gradient
            overshoot = 0.0
            step_adv = input_var.data + (1+overshoot) * normed_grad

            total_adv = step_adv - input

            # apply total adversarial perturbation to original image and clip to valid pixel range
            input_adv = input + total_adv
            input_adv = torch.clamp(input_adv, -2.5, 2.5)
            input_var.data = input_adv
            step += 1

        return input_adv

