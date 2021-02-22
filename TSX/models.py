from torch import nn
import torch
torch.autograd.set_detect_anomaly(True)


class IMVTensorLSTM(torch.jit.ScriptModule):
    __constants__ = ["n_units", "input_dim"]

    def __init__(self, input_dim, output_dim, n_units, device, init_std=0.02):
        super().__init__()
        self.device = device
        self.U_j = nn.Parameter(torch.randn(input_dim, 1, n_units) * init_std)
        self.U_i = nn.Parameter(torch.randn(input_dim, 1, n_units) * init_std)
        self.U_f = nn.Parameter(torch.randn(input_dim, 1, n_units) * init_std)
        self.U_o = nn.Parameter(torch.randn(input_dim, 1, n_units) * init_std)
        self.W_j = nn.Parameter(torch.randn(input_dim, n_units, n_units) * init_std)
        self.W_i = nn.Parameter(torch.randn(input_dim, n_units, n_units) * init_std)
        self.W_f = nn.Parameter(torch.randn(input_dim, n_units, n_units) * init_std)
        self.W_o = nn.Parameter(torch.randn(input_dim, n_units, n_units) * init_std)
        self.b_j = nn.Parameter(torch.randn(input_dim, n_units) * init_std)
        self.b_i = nn.Parameter(torch.randn(input_dim, n_units) * init_std)
        self.b_f = nn.Parameter(torch.randn(input_dim, n_units) * init_std)
        self.b_o = nn.Parameter(torch.randn(input_dim, n_units) * init_std)
        self.F_alpha_n = nn.Parameter(torch.randn(input_dim, n_units, 1) * init_std)
        self.F_alpha_n_b = nn.Parameter(torch.randn(input_dim, 1) * init_std)
        self.F_beta = nn.Linear(2 * n_units, 1, bias=False)
        self.Phi = nn.Linear(2 * n_units, output_dim, bias=False)
        self.n_units = n_units
        self.input_dim = input_dim

    @torch.jit.script_method
    def forward(self, x):
        h_tilda_t = torch.zeros(x.shape[0], self.input_dim, self.n_units).to(self.device)
        c_tilda_t = torch.zeros(x.shape[0], self.input_dim, self.n_units).to(self.device)
        outputs = torch.jit.annotate(List[Tensor], [])
        for t in range(x.shape[1]):
            # eq 1
            j_tilda_t = torch.tanh(torch.einsum("bij,ijk->bik", h_tilda_t, self.W_j) + \
                                   torch.einsum("bij,jik->bjk", x[:, t, :].unsqueeze(1), self.U_j) + self.b_j)
            # eq 5
            i_tilda_t = torch.sigmoid(torch.einsum("bij,ijk->bik", h_tilda_t, self.W_i) + \
                                      torch.einsum("bij,jik->bjk", x[:, t, :].unsqueeze(1), self.U_i) + self.b_i)
            f_tilda_t = torch.sigmoid(torch.einsum("bij,ijk->bik", h_tilda_t, self.W_f) + \
                                      torch.einsum("bij,jik->bjk", x[:, t, :].unsqueeze(1), self.U_f) + self.b_f)
            o_tilda_t = torch.sigmoid(torch.einsum("bij,ijk->bik", h_tilda_t, self.W_o) + \
                                      torch.einsum("bij,jik->bjk", x[:, t, :].unsqueeze(1), self.U_o) + self.b_o)
            # eq 6
            c_tilda_t = c_tilda_t * f_tilda_t + i_tilda_t * j_tilda_t
            # eq 7
            h_tilda_t = (o_tilda_t * torch.tanh(c_tilda_t))
            k = torch.sum(h_tilda_t[0, :, :], dim=1)
            outputs += [h_tilda_t]
        outputs = torch.stack(outputs)
        outputs = outputs.permute(1, 0, 2, 3)
        # eq 8
        alphas = torch.tanh(torch.einsum("btij,ijk->btik", outputs, self.F_alpha_n) + self.F_alpha_n_b)
        alphas = torch.exp(alphas)
        alphas = alphas / torch.sum(alphas, dim=1, keepdim=True)
        g_n = torch.sum(alphas * outputs, dim=1)
        hg = torch.cat([g_n, h_tilda_t], dim=2)
        mu = self.Phi(hg)
        betas = torch.tanh(self.F_beta(hg))
        betas = torch.exp(betas)
        betas = betas / torch.sum(betas, dim=1, keepdim=True)
        mean = torch.sum(betas * mu, dim=1)

        return mean, alphas, betas


class IMVTensorLSTMMultiTask(torch.jit.ScriptModule):
    __constants__ = ["n_units", "input_dim", "input_task_feature", "task_num", "input_share_dim"]

    def __init__(self, input_share_dim, input_task_feature, task_num, output_dim, n_units, device, em, drop_prob, init_std=0.02):
        super().__init__()
        input_dim = input_share_dim + input_task_feature*task_num
        self.device = device
        self.EM = em
        self.U_j = nn.Parameter(torch.randn(input_dim, 1, n_units) * init_std)
        self.U_i = nn.Parameter(torch.randn(input_dim, 1, n_units) * init_std)
        self.U_f = nn.Parameter(torch.randn(input_dim, 1, n_units) * init_std)
        self.U_o = nn.Parameter(torch.randn(input_dim, 1, n_units) * init_std)
        self.W_j = nn.Parameter(torch.randn(input_dim, n_units, n_units) * init_std)
        self.W_i = nn.Parameter(torch.randn(input_dim, n_units, n_units) * init_std)
        self.W_f = nn.Parameter(torch.randn(input_dim, n_units, n_units) * init_std)
        self.W_o = nn.Parameter(torch.randn(input_dim, n_units, n_units) * init_std)
        self.b_j = nn.Parameter(torch.randn(input_dim, n_units) * init_std)
        self.b_i = nn.Parameter(torch.randn(input_dim, n_units) * init_std)
        self.b_f = nn.Parameter(torch.randn(input_dim, n_units) * init_std)
        self.b_o = nn.Parameter(torch.randn(input_dim, n_units) * init_std)
        self.F_alpha_n = nn.Parameter(torch.randn(input_dim, n_units, 1) * init_std)
        self.F_alpha_n_b = nn.Parameter(torch.randn(input_dim, 1) * init_std)
        self.log_vars = nn.Parameter(torch.FloatTensor([-0.5]*task_num))

        self.F_beta = nn.Linear(2 * n_units, output_dim, bias=False)
        self.Phi = nn.Linear(2 * n_units, int(n_units/2), bias=False)
        self.Phi_mu = nn.Linear(2 * n_units, output_dim, bias=False)
        self.mv_dense_mean = nn.Linear(int(n_units / 2), output_dim, bias=False)

        #torch.nn.init.xavier_uniform_(self.F_beta.weight)
        #torch.nn.init.xavier_uniform_(self.Phi.weight)
        #torch.nn.init.xavier_uniform_(self.Phi_mu.weight)
        #torch.nn.init.xavier_uniform_(self.mv_dense_mean.weight)

        # self.mv_dense_var_n = nn.Parameter(torch.randn(input_dim, n_units, 1) * init_std)
        # self.mv_dense_var_b = nn.Parameter(torch.randn(input_dim, 1) * init_std)

        self.loss = torch.nn.MSELoss()
        self.dropout = nn.Dropout(drop_prob)

        self.n_units = n_units
        self.input_dim = input_dim
        self.input_task_feature = input_task_feature
        self.task_num = task_num
        self.input_share_dim = input_share_dim
        self.torch_pi = torch.acos(torch.zeros(1)).item() * 2

    @torch.jit.script_method
    def forward(self, x, y, task_idx, activated_share_columns):
        h_tilda_t = torch.zeros(x.shape[0], self.input_dim, self.n_units).to(self.device)
        c_tilda_t = torch.zeros(x.shape[0], self.input_dim, self.n_units).to(self.device)

        task_idx = task_idx[0]
        activated_share_columns = activated_share_columns[0, :]

        outputs = torch.jit.annotate(List[Tensor], [])
        task_vector = torch.repeat_interleave(torch.tensor([int(i == task_idx) for i in range(self.task_num)]),
                                              self.input_task_feature, dim=0)
        task_bias_matrix = torch.repeat_interleave(task_vector[:, None], self.n_units, dim=1)
        share_columns_activation_matrix = torch.repeat_interleave(activated_share_columns[:, None], self.n_units, dim=1)
        # task_bias = torch.cat((task_bias_matrix, torch.ones(self.input_share_dim, self.n_units)), 0).to(self.device)
        # concatenation can't use torch.cat
        t1_shape = task_bias_matrix.shape[0]
        t2_shape = self.input_share_dim

        test_vector = torch.zeros(task_bias_matrix.shape[0] + self.input_share_dim, dtype=torch.long)
        test_vector[:t1_shape] = task_vector
        test_vector[t1_shape:] = activated_share_columns
        indices = (test_vector == 0).nonzero().squeeze()
        complement_indices = (test_vector != 0).nonzero().squeeze()

        second_dim_shape = self.n_units
        task_bias = torch.ones((t1_shape + t2_shape, second_dim_shape))
        task_bias[:t1_shape, :] = task_bias_matrix
        task_bias[t1_shape:, :] = share_columns_activation_matrix
        task_bias = task_bias.to(self.device)

        x_task_matrix = torch.repeat_interleave(task_vector[None, :], x.shape[1], dim=0).to(self.device)
        # x_tilda = torch.cat((x_task_matrix.expand(*(x.shape[0], x_task_matrix.shape[0], x_task_matrix.shape[1])),
        #                     x[:, :, self.input_task_feature:]), dim=2).to(self.device)
        # concatenation can't use torch.cat
        x1_shape = x_task_matrix.shape[1]
        x2_shape = self.input_share_dim
        first_dim_shape = x.shape[0]
        second_dim_shape = x.shape[1]
        x_tilda = torch.ones((first_dim_shape, second_dim_shape, x1_shape + x2_shape))
        x_tilda[:, :, :x1_shape] = x_task_matrix.expand(*(x.shape[0], x_task_matrix.shape[0], x_task_matrix.shape[1]))
        x_tilda[:, :, x1_shape:] = x[:, :, self.input_task_feature:]
        x_tilda = x_tilda.to(self.device)

        for i in range(self.input_task_feature):
            x_tilda[:, :, task_idx * self.input_task_feature + i] = x[:, :, 0 + i]

        for t in range(x.shape[1]):
            # eq 1
            j_tilda_t = torch.tanh(torch.einsum("bij,ijk->bik", h_tilda_t, self.W_j) +
                                   torch.einsum("bij,jik->bjk", x_tilda[:, t, :].unsqueeze(1), self.U_j) +
                                   task_bias * self.b_j)
            # eq 5
            i_tilda_t = torch.sigmoid(torch.einsum("bij,ijk->bik", h_tilda_t, self.W_i) +
                                      torch.einsum("bij,jik->bjk", x_tilda[:, t, :].unsqueeze(1), self.U_i) +
                                      task_bias * self.b_i)
            f_tilda_t = torch.sigmoid(torch.einsum("bij,ijk->bik", h_tilda_t, self.W_f) +
                                      torch.einsum("bij,jik->bjk", x_tilda[:, t, :].unsqueeze(1), self.U_f) +
                                      task_bias * self.b_f)
            o_tilda_t = torch.sigmoid(torch.einsum("bij,ijk->bik", h_tilda_t, self.W_o) +
                                      torch.einsum("bij,jik->bjk", x_tilda[:, t, :].unsqueeze(1), self.U_o) +
                                      task_bias * self.b_o)
            # eq 6
            c_tilda_t = c_tilda_t * f_tilda_t + i_tilda_t * j_tilda_t
            # eq 7
            h_tilda_t = (o_tilda_t * torch.tanh(c_tilda_t))
            outputs += [h_tilda_t]
        outputs = torch.stack(outputs)
        outputs = outputs.permute(1, 0, 2, 3)
        # eq 8
        alphas = torch.tanh(torch.einsum("btij,ijk->btik", outputs, self.F_alpha_n) + self.F_alpha_n_b)
        alphas = torch.exp(alphas)
        alphas = alphas / torch.sum(alphas, dim=1, keepdim=True)
        g_n = torch.sum(alphas * outputs, dim=1)
        hg = torch.cat([g_n, h_tilda_t], dim=2)

        betas = torch.tanh(self.F_beta(hg))
        betas = self.dropout(betas)
        betas_1 = betas[:, complement_indices, :]
        betas_2 = betas[:, indices, :]
        betas_1 = torch.exp(betas_1)
        betas_1 = betas_1 / torch.sum(betas_1, dim=1, keepdim=True)

        if self.EM:
            mu = self.Phi(hg)
            h_mean = self.mv_dense_mean(mu)
            h_mean_real = h_mean[:, complement_indices, :]
            mean = torch.sum(betas_1 * h_mean[:, complement_indices, :], dim=1) + torch.sum(betas_2 * h_mean[:, indices, :], dim=1)

            y_tile = torch.repeat_interleave(y.unsqueeze(1), len(complement_indices), dim=1)
            tmp_llk = torch.exp(-0.5 * torch.square(y_tile - h_mean_real)) / (2.0 * self.torch_pi) ** 0.5
            lk = torch.sum(tmp_llk * betas_1, dim=1)
            neg_llk = torch.sum(-1.0 * torch.log(lk + 1e-5), dim=0)
            # -- posterior variable
            betas_temp = torch.zeros(betas.shape).to(self.device)
            tmp_energy = torch.exp(-0.5 * torch.square(y_tile - h_mean_real)) + 0.001
            joint_llk = tmp_energy * betas_1
            normalizer = torch.sum(joint_llk, 1, keepdim=True)
            joint_llk_normalized = 1.0 * joint_llk / (normalizer + 1e-10)
            betas_temp[:, complement_indices, :] = joint_llk
            loss = torch.squeeze(neg_llk)
        else:
            mu = self.Phi_mu(hg)
            mean = torch.sum(betas_1 * mu[:, complement_indices, :], dim=1) + torch.sum(betas_2 * mu[:, indices, :], dim=1)
            betas_temp = torch.zeros(betas.shape).to(self.device)
            betas_temp[:, complement_indices, :] = betas_1
            loss = self.loss(mean, y)
        theta = self.log_vars[task_idx]
        return mean, alphas, betas_temp, theta, loss


