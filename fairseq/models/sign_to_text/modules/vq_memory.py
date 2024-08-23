import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq.modules import MultiheadAttention

class VQMemory(nn.Module):
    def __init__(
        self,
        memory_num,
        attention_heads,
        dim,
        num_vars,
        temp,
        groups,
        vq_dim,
        activation=nn.GELU(),
        weight_proj_depth=1,
        weight_proj_factor=1,
    ):
        super().__init__()

        self.memory_num = memory_num
        self.use_memory = (memory_num > 0)
        self.attention_heads = attention_heads
        self.dim = dim
        self.num_vars = num_vars
        self.num_groups = groups
        # self.share_codebook = (memory_num <= 0)
        self.share_codebook = True


        assert (
                vq_dim % groups == 0
        ), f"dim {vq_dim} must be divisible by groups {groups} for concatenation"

        var_dim = vq_dim // groups

        if self.use_memory:
            self.memory_module = nn.Parameter(torch.FloatTensor(self.memory_num, 1, self.dim))
            nn.init.uniform_(self.memory_module)

            self.memory_attention = MultiheadAttention(self.dim, self.attention_heads)

        if self.num_vars > 0:
            self.vars = nn.Parameter(torch.FloatTensor(1, self.num_groups * self.num_vars, var_dim))
            nn.init.uniform_(self.vars)

            if weight_proj_depth > 1:

                def block(input_dim, output_dim):
                    return nn.Sequential(nn.Linear(input_dim, output_dim), activation)

                inner_dim = self.dim * weight_proj_factor
                self.weight_proj = nn.Sequential(
                    *[
                        block(self.dim if i == 0 else inner_dim, inner_dim)
                        for i in range(weight_proj_depth - 1)
                    ],
                    nn.Linear(inner_dim, groups * num_vars),
                )
            else:
                self.weight_proj = nn.Linear(self.dim, groups * num_vars)
                nn.init.normal_(self.weight_proj.weight, mean=0, std=1)
                nn.init.zeros_(self.weight_proj.bias)

        self.max_temp, self.min_temp, self.temp_decay = temp
        self.curr_temp = self.max_temp

    def set_num_updates(self, num_updates):
        self.curr_temp = max(
            self.max_temp * self.temp_decay ** num_updates, self.min_temp
        )

    def forward(self, x, padding_mask):
        """
        :param x: T x B x C
        :param padding_mask: B x T
        :return: x, encoder_padding_mask, memory_info
        """
        vq_info = {}

        T, B, C = x.size()
        if self.use_memory:
            x = self.memory_attention(
                query=self.memory_module.repeat(1, B, 1),
                key=x,
                value=x,
                key_padding_mask=padding_mask
            )[0]
            encoder_padding_mask = torch.zeros(B, x.size(0), dtype=torch.bool).to(x.device)
            T, B, C = x.size()
        else:
            encoder_padding_mask = padding_mask


        if self.num_vars > 0:
            x = x.transpose(0, 1)
            x = x.reshape(-1, C)
            x = self.weight_proj(x)
            x = x.view(B * T * self.num_groups, -1)

            vq_info["vq_logits"] = torch.softmax(
                x.view(B * T, self.num_groups, -1).float(), dim=-1
            )
            vq_info["vq_targets"] = (
                x.argmax(dim=-1)
                .view(B * T, self.num_groups)
                .detach()
            )
            vq_info["temp"] = self.curr_temp

            if self.training:
                x = F.gumbel_softmax(x.float(), tau=self.curr_temp, hard=True).type_as(x)
            else:
                _, k = x.max(-1)
                x = (
                    x.new_zeros(*x.shape)
                    .scatter_(-1, k.view(-1, 1), 1.0)
                    .view(B * T, self.num_groups, -1)
                )

            x = x.view(B * T, -1)
            vars = self.vars
            x = x.unsqueeze(-1) * vars
            x = x.view(B * T, self.num_groups, self.num_vars, -1)
            x = x.sum(-2)
            x = x.view(B, T, -1)
            x = x.transpose(0,1)

        return x, encoder_padding_mask, vq_info



