import torch.nn.functional as F
import torch
from .srnn_model import *
import time


class HHEdgeSelfAttn(nn.Module):
    """
    Class for the human-human attention,
    uses a multi-head self attention proposed by https://arxiv.org/abs/1706.03762
    """

    def __init__(self, args):
        super(HHEdgeSelfAttn, self).__init__()
        self.args = args

        # Store required sizes
        # todo: hard-coded for now
        # with human displacement: + 2
        # pred 4 steps + disp: 12
        # pred 4 steps + no disp: 10
        # pred 5 steps + no disp: 12
        # pred 5 steps + no disp + probR: 17
        # Gaussian pred 5 steps + no disp: 27
        # pred 8 steps + no disp: 18
        if args.env_name in ['CrowdSimPred-v0', 'CrowdSimPredRealGST-v0']:
            self.input_size = 12
        elif args.env_name == 'CrowdSimVarNum-v0':
            self.input_size = 2  # 4
        else:
            raise NotImplementedError
        self.num_attn_heads = 8
        self.attn_size = 512

        # Linear layer to embed input
        self.embedding_layer = nn.Sequential(nn.Linear(self.input_size, 128), nn.ReLU(),
                                             nn.Linear(128, self.attn_size), nn.ReLU()
                                             )

        self.q_linear = nn.Linear(self.attn_size, self.attn_size)
        self.v_linear = nn.Linear(self.attn_size, self.attn_size)
        self.k_linear = nn.Linear(self.attn_size, self.attn_size)

        # multi-head self attention
        self.multihead_attn = torch.nn.MultiheadAttention(self.attn_size, self.num_attn_heads)

    # Given a list of sequence lengths, create a mask to indicate which indices are padded
    # e.x. Input: [3, 1, 4], max_human_num = 5
    # Output: [[1, 1, 1, 0, 0], [1, 0, 0, 0, 0], [1, 1, 1, 1, 0]]
    def create_attn_mask(self, each_seq_len, seq_len, nenv, max_human_num):
        # mask with value of False means padding and should be ignored by attention
        # why +1: use a sentinel in the end to handle the case when each_seq_len = 18
        if self.args.no_cuda:
            mask = torch.zeros(seq_len * nenv, max_human_num + 1).cpu()
        else:
            mask = torch.zeros(seq_len * nenv, max_human_num + 1).cuda()
        mask[torch.arange(seq_len * nenv), each_seq_len.long()] = 1.
        mask = torch.logical_not(mask.cumsum(dim=1))
        # remove the sentinel
        mask = mask[:, :-1].unsqueeze(-2)  # seq_len*nenv, 1, max_human_num
        return mask

    def forward(self, inp, each_seq_len):
        '''
        Forward pass for the model
        params:
        inp : input edge features
        each_seq_len:
        if self.args.sort_humans is True, the true length of the sequence. Should be the number of detected humans
        else, it is the mask itself
        '''
        # inp is padded sequence [seq_len, nenv, max_human_num, 2]
        seq_len, nenv, max_human_num, _ = inp.size()
        if self.args.sort_humans:
            attn_mask = self.create_attn_mask(each_seq_len, seq_len, nenv, max_human_num)  # [seq_len*nenv, 1, max_human_num]
            attn_mask = attn_mask.squeeze(1)  # if we use pytorch builtin function
        else:
            # combine the first two dimensions
            attn_mask = each_seq_len.reshape(seq_len * nenv, max_human_num)

        input_embedded = self.embedding_layer(inp)
        input_emb = input_embedded.view(seq_len * nenv, max_human_num, -1)
        input_emb = torch.transpose(input_emb, dim0=0, dim1=1)  # if we use pytorch builtin function, v1.7.0 has no batch first option
        q = self.q_linear(input_emb)
        k = self.k_linear(input_emb)
        v = self.v_linear(input_emb)

        # z=self.multihead_attn(q, k, v, mask=attn_mask)
        z, _ = self.multihead_attn(q, k, v, key_padding_mask=torch.logical_not(attn_mask))  # if we use pytorch builtin function
        z = torch.transpose(z, dim0=0, dim1=1)  # if we use pytorch builtin function
        # print("s-n-max: ", seq_len, nenv, max_human_num, 'z.size: ', z.size())
        return z


class RHEdgeAttention(nn.Module):
    '''
    Class for the robot-human attention module
    '''

    def __init__(self, args):
        '''
        Initializer function
        params:
        args : Training arguments
        infer : Training or test time (True at test time)
        '''
        super(RHEdgeAttention, self).__init__()

        self.args = args

        # Store required sizes
        self.human_human_edge_rnn_size = args.human_human_edge_rnn_size
        self.human_node_rnn_size = args.human_node_rnn_size
        self.attention_size = args.attention_size

        # Linear layer to embed temporal edgeRNN hidden state
        self.temporal_edge_layer = nn.ModuleList()
        self.spatial_edge_layer = nn.ModuleList()

        self.temporal_edge_layer.append(nn.Linear(self.human_human_edge_rnn_size, self.attention_size))

        # Linear layer to embed spatial edgeRNN hidden states
        self.spatial_edge_layer.append(nn.Linear(self.human_human_edge_rnn_size, self.attention_size))

        # number of agents who have spatial edges (complete graph: all 6 agents; incomplete graph: only the robot)
        self.agent_num = 1
        self.num_attention_head = 1

        # new cross-attention procedure, by zq: begin
        self.selfattn_inputsize = args.human_human_edge_rnn_size
        self.selfattn_hiddensize = 256
        self.selfattn_heads = 8

        self.query_h = nn.Linear(self.selfattn_inputsize, self.selfattn_hiddensize)
        self.key_h = nn.Linear(self.selfattn_inputsize, self.selfattn_hiddensize)
        self.value_h = nn.Linear(self.selfattn_inputsize, self.selfattn_hiddensize)

        self.query_r = nn.Linear(self.selfattn_inputsize, self.selfattn_hiddensize)
        self.key_r = nn.Linear(self.selfattn_inputsize, self.selfattn_hiddensize)
        self.value_r = nn.Linear(self.selfattn_inputsize, self.selfattn_hiddensize)

        self.multihead_attn_h2r = torch.nn.MultiheadAttention(self.selfattn_hiddensize, self.selfattn_heads)
        self.multihead_attn_r2h = torch.nn.MultiheadAttention(self.selfattn_hiddensize, self.selfattn_heads)

        self.conv20to1_h2r = nn.Conv2d(20, 1, kernel_size=1)
        self.conv20to1_r2h = nn.Conv2d(20, 1, kernel_size=1)

        # new cross-attention procedure, by zq: end

        # by zq: MRF weight fusion: begin
        self.weight_1_linear = nn.Linear(40, 1)  # 输入维度为20*2=40，输出维度为1
        self.weight_2_linear = nn.Linear(40, 1)
        # by zq: MRF weight fusion: end

        # by zq: ssa_attn_operation: begin
        self.ssa_linear_12to256 = nn.Linear(12, self.human_human_edge_rnn_size)
        self.hr_feature_linear = nn.Linear(512, 1)  # 输入维度为20*2=40，输出维度为1
        self.ssa_feature_linear = nn.Linear(512, 1)

    def create_FOV_mask(self, each_seq_len, seq_len, nenv, max_human_num):
        # mask with value of False means padding and should be ignored by attention
        # why +1: use a sentinel in the end to handle the case when each_seq_len = 18
        if self.args.no_cuda:
            mask = torch.zeros(seq_len * nenv, max_human_num + 1).cpu()
        else:
            mask = torch.zeros(seq_len * nenv, max_human_num + 1).cuda()
        mask[torch.arange(seq_len * nenv), each_seq_len.long()] = 1.
        mask = torch.logical_not(mask.cumsum(dim=1))
        # remove the sentinel
        mask = mask[:, :-1].unsqueeze(-2)  # seq_len*nenv, 1, max_human_num
        return mask

    def create_sxy_attn_mask(self, human_points, robot_points, robot_speed, mask):
        # sxy defined the following rules

        human_points_tmp = human_points[:, :, :, 0:2]
        human_speed_tmp = human_points[:, :, :, 2:4] - human_points[:, :, :, 0:2]

        robot_points_tmp = robot_points[:, :, :, 0:2]
        robot_speed_tmp = robot_speed

        # mask = self.create_sxy_attn_mask_for_current_robot(points=human_points_tmp, velocities=human_speed_tmp, agent=robot_points_tmp, agent_velocity=robot_speed_tmp)
        points = human_points_tmp
        velocities = human_speed_tmp
        agent = robot_points_tmp
        agent_velocity = robot_speed_tmp

        P, Q, num, _ = points.shape

        # Expand agent and agent_velocity to match the shape of points and velocities
        agent_expanded = agent.expand(P, Q, num, 2)
        vc = agent_velocity.expand(P, Q, num, 2)

        # Calculate vp_n2c and vv_n2c using matrix operations
        vp_n2c = points - agent_expanded
        vv_n2c = velocities - vc

        # Calculate the cosine of the angle between vp_n2c and vv_n2c
        vpn_vv_dot = torch.einsum('...i,...i->...', vp_n2c, vv_n2c)
        vp_agent_dot = torch.einsum('...i,...i->...', vp_n2c, vc)
        vp_n2c_norm = torch.norm(vp_n2c, dim=-1)
        vv_n2c_norm = torch.norm(vv_n2c, dim=-1)
        vc_norm = torch.norm(vc, dim=-1)

        cosine_angle1 = vpn_vv_dot / (vp_n2c_norm * vv_n2c_norm)
        cosine_angle2 = vp_agent_dot / (vp_n2c_norm * vc_norm)

        # Create the mask using logical operators and indexing
        mask_cos1 = torch.zeros_like(cosine_angle1, dtype=torch.bool).cuda()
        mask_cos2 = torch.zeros_like(cosine_angle1, dtype=torch.bool).cuda()
        mask_cos3 = torch.zeros_like(cosine_angle1, dtype=torch.bool).cuda()
        mask_norm = torch.zeros_like(cosine_angle1, dtype=torch.bool).cuda()
        # mask = torch.zeros_like(cosine_angle1, dtype=torch.bool)
        mask_cos1[cosine_angle1 < 0] = True
        mask_cos2[cosine_angle2 > -0.5] = True
        mask_cos3[cosine_angle2 > 0.5] = True

        mask = (mask_cos1 & mask_cos2) | mask_cos3

        # count = torch.sum(~mask)
        # print(mask.device, count.item())

        # mask = torch.zeros(P, Q, num).cuda()
        # mask[((cosine_angle1 < 0) & (cosine_angle2 > -0.5)) | (torch.norm(vp_n2c, dim=-1) < 2)] = torch.abs(
        #     cosine_angle1[((cosine_angle1 < 0) & (cosine_angle2 > -0.5)) | (torch.norm(vp_n2c, dim=-1) < 2)].float()) / torch.norm(
        #     vp_n2c[((cosine_angle1 < 0) & (cosine_angle2 > -0.5)) | (torch.norm(vp_n2c, dim=-1) < 2)], dim=-1).float()

        return mask

    def create_priori_knowlege_attn_weight(self, human_points, robot_points, robot_speed, mask):
        human_points_tmp = human_points[:, :, :, 0:2]
        human_speed_tmp = human_points[:, :, :, 2:4] - human_points[:, :, :, 0:2]

        robot_points_tmp = robot_points[:, :, :, 0:2]
        robot_speed_tmp = robot_speed

        hp = human_points_tmp
        hv = human_speed_tmp
        rp = robot_points_tmp
        rv = robot_speed_tmp

        seq_len, nenv, h_num, _ = hp.shape

        # Expand robot_points and robot_velocities to match the shape of human_points and human_velocities
        rp_expanded = rp.expand(seq_len, nenv, h_num, 2)
        rv_expanded = rv.expand(seq_len, nenv, h_num, 2)

        dp_r2h = hp - rp_expanded
        dv_r2h = hv - rv_expanded

        return human_points

    def create_h2r_attn_weight_liushuijng(self, robot_states, human_states, mask=None):
        """
        创建人对机器人的注意力矩阵，参数如下：
        :param robot_states: nn.linear(robot_states)
        :param human_states: nn.linear(human_spatial_edges (output of HHA))
        :param mask: 需要关注的人的mask
        :return: h2r_attn
        """
        seq_len, nenv, num_edges, _ = human_states.size()  # [1, 12, 30, 256] in testing,  [12, 30, 256] in training
        h2r_attn_weight = robot_states * human_states
        h2r_attn_weight = torch.sum(h2r_attn_weight, dim=3)

        # Variable length
        temperature = num_edges / np.sqrt(self.attention_size)
        h2r_attn_weight = torch.mul(h2r_attn_weight, temperature)

        # if we don't want to mask invalid humans, attn_mask is None and no mask will be applied
        # else apply attn masks
        if mask is not None:
            h2r_attn_weight = h2r_attn_weight.masked_fill(mask == 0, -1e9)

        # Softmax
        h2r_attn_weight = h2r_attn_weight.view(seq_len, nenv, self.agent_num, self.human_num)
        h2r_attn_weight = torch.nn.functional.softmax(h2r_attn_weight, dim=-1)
        h2r_attn_weight = h2r_attn_weight.view(seq_len * nenv * self.agent_num, self.human_num).unsqueeze(-1)  # [seq_len*nenv*6, 5, 1]

        return h2r_attn_weight

    def creat_MultiEnv_SSA_mask(self, input_h_ps, input_h_vs, input_r_p, input_r_v, mask=None):

        h_ps = input_h_ps.clone()
        h_vs = input_h_vs.clone()
        r_p = input_r_p.clone()
        r_v = input_r_v.clone()


        h_ps[input_h_ps == 0] = 0.001
        h_vs[input_h_vs == 0] = 0.001
        r_p[input_r_p == 0] = 0.001
        r_v[input_r_v == 0] = 0.001

        seq_len, nenv, num, _ = h_ps.shape


        r_ps = r_p.repeat(1, 1, num, 1)
        r_vs = r_v.repeat(1, 1, num, 1)

        # compute points vectors: robot to human
        d_r2p = h_ps - r_ps
        d_p2r = r_ps - h_ps

        # compute h_vs norms
        h_vs_norms = torch.norm(h_vs, dim=3)
        # compute r_vs norms
        r_vs_norms = torch.norm(r_vs, dim=3)
        # compute d_r2p and d_p2r norms
        dist_r2h = torch.norm(d_r2p, dim=3)
        l = d_r2p_norms = d_p2r_norms = dist_r2h

        # compute the cosine of the angle between r_vs and d_r2p
        cosine_r2d = torch.sum(r_vs * d_r2p, dim=3) / (r_vs_norms * d_r2p_norms)
        # compute the cosine of the angle between p_vs and d_p2r
        cosine_h2d = torch.sum(h_vs * d_p2r, dim=3) / (h_vs_norms * d_p2r_norms)

        # compute safety
        safety = (r_vs_norms * cosine_r2d + h_vs_norms * cosine_h2d) / (l ** 2)
        safety[safety <= 0] = 0.00000001

        if mask is not None:
            SSA_weight = safety.masked_fill(mask == 0, -1e9)

        SSA_weight = torch.softmax(SSA_weight, dim=2)

        SSA_weight_masked = SSA_weight.view(seq_len * nenv * self.agent_num, self.human_num).unsqueeze(-1)

        return SSA_weight_masked

    def multi_weights_fusion(self, weight_1, weight_2):
        nenv, h_num,_ = weight_1.size()
        weight_1 = torch.squeeze(weight_1, dim=2)  # weight_1.squeeze()
        weight_2 = torch.squeeze(weight_2, dim=2)  # weight_2.squeeze()
        concat = torch.cat((weight_1, weight_2), dim=-1)

        # 计算线性层的输出
        ww1 = self.weight_1_linear(concat)
        ww2 = self.weight_2_linear(concat)

        ww_cat = torch.cat((ww1, ww2), dim=-1)
        ww_cat = F.softmax(ww_cat, dim=-1)

        lambda_1 = ww_cat[:, 0:1]
        lambda_2 = ww_cat[:, 1:2]

        lambda_1_expand = lambda_1.repeat(1, h_num)
        lambda_2_expand = lambda_2.repeat(1, h_num)

        fused_weight = weight_1 * lambda_1_expand + weight_2 * lambda_2_expand

        fused_weight = fused_weight.unsqueeze(-1)
        return fused_weight

    def hr_attn_operation(self, h_spatials, attn_weight):
        seq_len, nenv, num_edges, h_size = h_spatials.size()  # [1, 12, 30, 256] in testing,  [12, 30, 256] in training
        h_spatials = h_spatials.view(seq_len, nenv, self.agent_num, self.human_num, h_size)
        h_spatials = h_spatials.view(seq_len * nenv * self.agent_num, self.human_num, h_size).permute(0, 2, 1)  # [seq_len*nenv*6, 5, 256] -> [seq_len*nenv*6, 256, 5]

        weighted_value = torch.bmm(h_spatials, attn_weight)  # [seq_len*nenv*6, 256, 1]

        # reshape back
        weighted_value = weighted_value.squeeze(-1).view(seq_len, nenv, self.agent_num, h_size)  # [seq_len, 12, 6 or 1, 256]
        return weighted_value, attn_weight

    def SSA_attn_operation(self, h_spatials, attn_weight):
        seq_len, nenv, num_edges, h_size = h_spatials.size()  # [1, 12, 30, 256] in testing,  [12, 30, 256] in training
        h_spatials = h_spatials.view(seq_len, nenv, self.agent_num, self.human_num, h_size)
        h_spatials = h_spatials.view(seq_len * nenv * self.agent_num, self.human_num, h_size).permute(0, 2, 1)  # [seq_len*nenv*6, 5, 256] -> [seq_len*nenv*6, 256, 5]

        weighted_value = torch.bmm(h_spatials, attn_weight)  # [seq_len*nenv*6, 256, 1]

        # reshape back
        weighted_value = weighted_value.squeeze(-1).view(seq_len, nenv, self.agent_num, h_size)  # [seq_len, 12, 6 or 1, 256]
        return weighted_value, attn_weight

    def attn_feature_fusion(self, hr_attn_feature, ssa_attn_feature):
        seq_len, nenv, nr, csize = hr_attn_feature.size()
        concat = torch.cat((hr_attn_feature, ssa_attn_feature), dim=-1)

        # 计算线性层的输出
        w1 = self.hr_feature_linear(concat)
        w2 = self.ssa_feature_linear(concat)

        w_cat = torch.cat((w1, w2), dim=-1)
        w_cat = F.softmax(w_cat, dim=-1)

        lambda_1 = w_cat[:, :, :, 0:1]
        lambda_2 = w_cat[:, :, :, 1:2]

        # print("lambda_1, lambda_2 : ",lambda_1, " ", lambda_2)

        lambda_1_expand = lambda_1.repeat(1, 1, 1, csize)
        lambda_2_expand = lambda_2.repeat(1, 1, 1, csize)

        fused_weight = hr_attn_feature * lambda_1_expand + ssa_attn_feature * lambda_2_expand

        return fused_weight


    def h_QV_r_K_attn(self, robot_states, human_states, h_spatials, mask=None):
        '''
        注意力机制的实现，参数如下：
        :param robot_states: [1, 16, 20, 64]
        :param human_states: [1, 16, 20, 64]
        :param h_spatials:   [1, 16, 20, 256]
        :param mask:         [1, 16, 20]
        :return:
        :       weighted_value  [1,  16, 1, 256]
        :       h2r_attn_weight [16, 20, 1]
        '''
        seq_len, nenv, num_edges, _ = human_states.size()  # [1, 12, 30, 256] in testing,  [12, 30, 256] in training
        h2r_attn_weight = robot_states * human_states
        h2r_attn_weight = torch.sum(h2r_attn_weight, dim=3)

        # Variable length
        temperature = num_edges / np.sqrt(self.attention_size)
        h2r_attn_weight = torch.mul(h2r_attn_weight, temperature)

        # if we don't want to mask invalid humans, attn_mask is None and no mask will be applied
        # else apply attn masks
        if mask is not None:
            h2r_attn_weight = h2r_attn_weight.masked_fill(mask == 0, -1e9)

        # Softmax
        h2r_attn_weight = h2r_attn_weight.view(seq_len, nenv, self.agent_num, self.human_num)
        h2r_attn_weight = torch.nn.functional.softmax(h2r_attn_weight, dim=-1)
        h2r_attn_weight = h2r_attn_weight.view(seq_len * nenv * self.agent_num, self.human_num).unsqueeze(-1)  # [seq_len*nenv*6, 5, 1]

        # soft*V
        seq_len, nenv, num_edges, h_size = h_spatials.size()  # [1, 12, 30, 256] in testing,  [12, 30, 256] in training
        h_spatials = h_spatials.view(seq_len, nenv, self.agent_num, self.human_num, h_size)
        h_spatials = h_spatials.view(seq_len * nenv * self.agent_num, self.human_num, h_size).permute(0, 2, 1)  # [seq_len*nenv*6, 5, 256] -> [seq_len*nenv*6, 256, 5]

        weighted_value = torch.bmm(h_spatials, h2r_attn_weight)  # [seq_len*nenv*6, 256, 1]

        # reshape back
        weighted_value = weighted_value.squeeze(-1).view(seq_len, nenv, self.agent_num, h_size)  # [seq_len, 12, 6 or 1, 256]
        return weighted_value, h2r_attn_weight

    def h_QV_r_K_attn_modified_backup(self, current_attn_head, robot_states, human_states, mask=None):
        """
        注意力机制的实现，参数如下：
        :param current_attn_head: 如果用多头注意力机制，需要指定当前的注意力头
        :param robot_states: [1, 16, 1, 256]
        :param human_states: [1, 16, 20, 256]
        :param mask:         [1, 16, 20]
        :return:
        :       weighted_value  [1,  16, 1, 256]
        :       h2r_attn_weight [16, 20, 1]
        """
        i = current_attn_head
        # Embed the temporal edgeRNN hidden state
        robot_embed = self.temporal_edge_layer[i](robot_states)
        robot_embed = robot_embed.repeat_interleave(self.human_num, dim=2)

        # Embed the spatial edgeRNN hidden states
        human_embed = self.spatial_edge_layer[i](human_states)

        # Dot based attention
        seq_len, nenv, num_edges, _ = human_embed.size()
        h2r_attn_weight = robot_embed * human_embed
        h2r_attn_weight = torch.sum(h2r_attn_weight, dim=3)

        # Variable length
        temperature = num_edges / np.sqrt(self.attention_size)
        h2r_attn_weight = torch.mul(h2r_attn_weight, temperature)

        # if we don't want to mask invalid humans, attn_mask is None and no mask will be applied
        # else apply attn masks
        if mask is not None:
            h2r_attn_weight = h2r_attn_weight.masked_fill(mask == 0, -1e9)

        # Softmax
        h2r_attn_weight = h2r_attn_weight.view(seq_len, nenv, self.agent_num, self.human_num)
        h2r_attn_weight = torch.nn.functional.softmax(h2r_attn_weight, dim=-1)
        h2r_attn_weight = h2r_attn_weight.view(seq_len * nenv * self.agent_num, self.human_num).unsqueeze(-1)  # [seq_len*nenv*6, 5, 1]

        # soft*V
        seq_len, nenv, num_edges, h_size = human_states.size()  # [1, 12, 30, 256] in testing,  [12, 30, 256] in training
        h_spatials = human_states.view(seq_len, nenv, self.agent_num, self.human_num, h_size)
        h_spatials = h_spatials.view(seq_len * nenv * self.agent_num, self.human_num, h_size).permute(0, 2, 1)  # [seq_len*nenv*6, 5, 256] -> [seq_len*nenv*6, 256, 5]

        weighted_value = torch.bmm(h_spatials, h2r_attn_weight)  # [seq_len*nenv*6, 256, 1]

        # reshape back
        weighted_value = weighted_value.squeeze(-1).view(seq_len, nenv, self.agent_num, h_size)  # [seq_len, 12, 6 or 1, 256]
        return weighted_value, h2r_attn_weight

    def h_QV_r_K_attn_modified(self, current_attn_head, robot_states, human_states, mask=None):
        """
        注意力机制的实现，参数如下：
        :param current_attn_head: 如果用多头注意力机制，需要指定当前的注意力头
        :param robot_states: [1, 16, 1, 256]
        :param human_states: [1, 16, 20, 256]
        :param mask:         [1, 16, 20]
        :return:
        :       weighted_value  [1,  16, 1, 256]
        :       h2r_attn_weight [16, 20, 1]
        """
        i = current_attn_head

        seq_len, nenv, num_edges, h_size = human_states.size()
        masks = mask.unsqueeze(-1).expand(-1, -1, -1, h_size)
        if masks is not None:
            human_embed = human_states.masked_fill(masks == 0, -1e9)
        else:
            human_embed = human_states

        robot_embed = robot_states.repeat_interleave(self.human_num, dim=2)  # [1 16 20 256]

        # Dot based attention
        human_reshaped = human_embed.view(seq_len * nenv, num_edges, h_size)
        robot_reshaped = robot_embed.view(seq_len * nenv, num_edges, h_size)

        q_h = self.query_h(human_reshaped)
        k_h = self.key_h(human_reshaped)
        v_h = self.value_h(human_reshaped)
        q_r = self.query_r(robot_reshaped)
        k_r = self.key_r(robot_reshaped)
        v_r = self.value_r(robot_reshaped)

        weighted_value_h2r, _ = self.multihead_attn_h2r(q_h, k_r, v_r)
        weighted_value_r2h, _ = self.multihead_attn_r2h(q_r, k_h, v_h)

        weighted_value_h2r = weighted_value_h2r.view(seq_len, nenv, num_edges, self.selfattn_hiddensize)
        weighted_value_r2h = weighted_value_r2h.view(seq_len, nenv, num_edges, self.selfattn_hiddensize)

        weighted_value_h2r_permute = weighted_value_h2r.permute(0, 2, 1, 3)
        weighted_value_r2h_permute = weighted_value_r2h.permute(0, 2, 1, 3)

        weightedw_h2r_out = self.conv20to1_h2r(weighted_value_h2r_permute)
        weightedw_r2h_out = self.conv20to1_r2h(weighted_value_r2h_permute)

        w_h2r_out = weightedw_h2r_out.permute(0, 2, 1, 3)
        w_r2h_out = weightedw_r2h_out.permute(0, 2, 1, 3)

        return w_h2r_out, w_r2h_out

    # h_temporal: [seq_len, nenv, 1, 256]
    # h_spatials: [seq_len, nenv, 5, 256]
    def forward(self, h_temporal, robot_node, temporal_edges, h_spatials, spatial_edges, each_seq_len):
        '''
        Forward pass for the model
        params:
        h_temporal :    robot_states    :: Robot states after linear layer 256
        robot_node:     robot_node      :: Current location of the robot: added by sxy
        temporal_edges: robot_velocity  ::Current speed of all robots: added by sxy
        h_spatials :    human_spatial_edges (output of HHA)    :: Hidden states of all spatial edgeRNNs connected to the node. -> human states after HHA
        spatial_edges:  human_spstial_edges (input of HHA)     :: Current location of all humans: added by sxy
        each_seq_len:   detected_human_num  ::  if self.args.sort_humans is True, the true length of the sequence. Should be the number of detected humans; else, it is the mask itself
        '''
        seq_len, nenv, max_human_num, _ = h_spatials.size()
        # find the number of humans by the size of spatial edgeRNN hidden state
        self.human_num = max_human_num // self.agent_num

        weighted_value_list, attn_list = [], []
        weighted_value_h2r_list, weighted_value_r2h_list = [], []
        for i in range(self.num_attention_head):

            # Embed the temporal edgeRNN hidden state
            temporal_embed = self.temporal_edge_layer[i](h_temporal)  # robot states 64
            # temporal_embed = temporal_embed.squeeze(0)

            # Embed the spatial edgeRNN hidden states
            spatial_embed = self.spatial_edge_layer[i](h_spatials)  #

            # Dot based attention
            temporal_embed = temporal_embed.repeat_interleave(self.human_num, dim=2)

            # 创建FOV_mask
            if self.args.sort_humans:
                # attn_mask_sxy = self.create_sxy_attn_mask(human_points=spatial_edges, robot_points=robot_node, robot_speed=temporal_edges)

                FOV_mask = self.create_FOV_mask(each_seq_len, seq_len, nenv, max_human_num)  # [seq_len*nenv, 1, max_human_num]
                FOV_mask = FOV_mask.squeeze(-2).view(seq_len, nenv, max_human_num)
            else:
                FOV_mask = each_seq_len

            # Policy 1: 仅仅考虑FOV内人与机器人，创建 h2r_attn_weight
            h2r_attn_weight = self.create_h2r_attn_weight_liushuijng(temporal_embed, spatial_embed, mask=FOV_mask)  # afore: 仅仅考虑FOV内的人；可以替换为其他

            # Policy 2: 基于先验知识的 SSA
            h_ps = spatial_edges[:, :, :, 0:2]
            h_vs = spatial_edges[:, :, :, 2:4] - spatial_edges[:, :, :, 0:2]
            r_p = robot_node[:, :, :, 0:2]
            r_v = temporal_edges

            SSA_weight = self.creat_MultiEnv_SSA_mask(input_h_ps=h_ps, input_h_vs=h_vs, input_r_p=r_p, input_r_v=r_v, mask=FOV_mask)

            # 多策略融合
            # fused_weight = self.multi_weights_fusion(h2r_attn_weight, SSA_weight)

            # Do weighted hr_attention operation，采用h2r_attn_weight对人h_spatials[1,16,20,256](from HH_Attn)做加权
            hr_attn_weighted_value, hr_attn_weight = self.hr_attn_operation(h_spatials, h2r_attn_weight)  # afore, replaced by self.h_QV_r_K_attn()

            # Do weighted ssa_attention operation，采用ssa_attn_weight对人h_states做加权。首先需要对h_states做线性变换，将其维度从12变为256
            h_states_ssa_linear_256ed = self.ssa_linear_12to256(spatial_edges)
            ssa_attn_weighted_value, ssa_attn_weight = self.SSA_attn_operation(h_states_ssa_linear_256ed, SSA_weight)

            # Do feature fusion
            fused_feature = self.attn_feature_fusion(hr_attn_weighted_value, ssa_attn_weighted_value)

            weighted_value_list.append(fused_feature)
            attn_list.append(hr_attn_weight)

        return weighted_value_list[0], attn_list[0]


class EndRNN(RNNBase):
    '''
    Class for the GRU
    '''

    def __init__(self, args):
        '''
        Initializer function
        params:
        args : Training arguments
        infer : Training or test time (True at test time)
        '''
        super(EndRNN, self).__init__(args, edge=False)

        self.args = args

        # Store required sizes
        self.rnn_size = args.human_node_rnn_size
        self.output_size = args.human_node_output_size
        self.embedding_size = args.human_node_embedding_size
        self.input_size = args.human_node_input_size
        self.edge_rnn_size = args.human_human_edge_rnn_size

        # Linear layer to embed input
        self.encoder_linear = nn.Linear(256, self.embedding_size)

        # ReLU and Dropout layers
        self.relu = nn.ReLU()

        # Linear layer to embed attention module output
        self.edge_attention_embed = nn.Linear(self.edge_rnn_size, self.embedding_size)

        # Output linear layer
        self.output_linear = nn.Linear(self.rnn_size, self.output_size)

    def forward(self, robot_s, h_spatial_other, h, masks):
        '''
        Forward pass for the model
        params:
        pos : input position
        h_temporal : hidden state of the temporal edgeRNN corresponding to this node
        h_spatial_other : output of the attention module
        h : hidden state of the current nodeRNN
        c : cell state of the current nodeRNN
        '''
        # Encode the input position
        encoded_input = self.encoder_linear(robot_s)
        encoded_input = self.relu(encoded_input)

        h_edges_embedded = self.edge_attention_embed(h_spatial_other)
        h_edges_embedded = self.relu(h_edges_embedded)

        concat_encoded = torch.cat((encoded_input, h_edges_embedded), -1)

        x, h_new = self._forward_gru(concat_encoded, h, masks)

        outputs = self.output_linear(x)

        return outputs, h_new


class selfAttn_merge_SRNN(nn.Module):
    """
    Class for the proposed network
    """

    def __init__(self, obs_space_dict, args, infer=False):
        """
        Initializer function
        params:
        args : Training arguments
        infer : Training or test time (True at test time)
        """
        super(selfAttn_merge_SRNN, self).__init__()
        self.infer = infer
        self.is_recurrent = True
        self.args = args

        self.human_num = obs_space_dict['spatial_edges'].shape[0]

        self.seq_length = args.seq_length
        self.nenv = args.num_processes
        self.nminibatch = args.num_mini_batch

        # Store required sizes
        self.human_node_rnn_size = args.human_node_rnn_size
        self.human_human_edge_rnn_size = args.human_human_edge_rnn_size
        self.output_size = args.human_node_output_size

        # Initialize the Node and Edge RNNs
        self.humanNodeRNN = EndRNN(args)

        # Initialize attention module
        self.RH_attn = RHEdgeAttention(args)

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2))

        num_inputs = hidden_size = self.output_size

        self.actor = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic_linear = init_(nn.Linear(hidden_size, 1))
        robot_size = 9
        self.robot_linear = nn.Sequential(init_(nn.Linear(robot_size, 256)), nn.ReLU())  # todo: check dim
        self.human_node_final_linear = init_(nn.Linear(self.output_size, 2))

        if self.args.use_self_attn:
            self.HH_attn = HHEdgeSelfAttn(args)
            self.human_linear_512to256 = nn.Sequential(init_(nn.Linear(512, 256)),
                                                nn.ReLU()
                                                )
        else:
            self.human_linear_512to256 = nn.Sequential(init_(nn.Linear(obs_space_dict['spatial_edges'].shape[1], 128)),
                                                nn.ReLU(),
                                                init_(nn.Linear(128, 256)),
                                                nn.ReLU()
                                                )

        self.temporal_edges = [0]
        self.spatial_edges = np.arange(1, self.human_num + 1)

        dummy_human_mask = [0] * self.human_num
        dummy_human_mask[0] = 1
        if self.args.no_cuda:
            self.dummy_human_mask = Variable(torch.Tensor([dummy_human_mask]).cpu())
        else:
            self.dummy_human_mask = Variable(torch.Tensor([dummy_human_mask]).cuda())

    def forward(self, inputs, rnn_hxs, masks, infer=False):
        global detected_human_num, human_masks

        # start_time = time.time()

        if infer:
            # Test/rollout time
            seq_length = 1
            nenv = self.nenv

        else:
            # Training time
            seq_length = self.seq_length
            nenv = self.nenv // self.nminibatch

        robot_node = reshapeT(inputs['robot_node'], seq_length, nenv)
        temporal_edges = reshapeT(inputs['temporal_edges'], seq_length, nenv)
        spatial_edges = reshapeT(inputs['spatial_edges'], seq_length, nenv)
        human_1 = spatial_edges[0, 0, 0, :]

        # to prevent errors in old models that does not have sort_humans argument
        if not hasattr(self.args, 'sort_humans'):
            self.args.sort_humans = True
        if self.args.sort_humans:
            detected_human_num = inputs['detected_human_num'].squeeze(-1).cpu().int()
            # print(detected_human_num)
        else:
            human_masks = reshapeT(inputs['visible_masks'], seq_length, nenv).float()  # [seq_len, nenv, max_human_num]
            # if no human is detected (human_masks are all False, set the first human to True)
            human_masks[human_masks.sum(dim=-1) == 0] = self.dummy_human_mask

        hidden_states_node_RNNs = reshapeT(rnn_hxs['human_node_rnn'], 1, nenv)
        masks = reshapeT(masks, seq_length, nenv)

        if self.args.no_cuda:
            all_hidden_states_edge_RNNs = Variable(
                torch.zeros(1, nenv, 1 + self.human_num, rnn_hxs['human_human_edge_rnn'].size()[-1]).cpu())
        else:
            all_hidden_states_edge_RNNs = Variable(
                torch.zeros(1, nenv, 1 + self.human_num, rnn_hxs['human_human_edge_rnn'].size()[-1]).cuda())

        robot_states = torch.cat((temporal_edges, robot_node), dim=-1)
        robot_states = self.robot_linear(robot_states)

        # attention modules
        if self.args.sort_humans:
            # HH attention
            if self.args.use_self_attn:
                spatial_attn_out = self.HH_attn(spatial_edges, detected_human_num).view(seq_length, nenv, self.human_num, -1)
            else:
                spatial_attn_out = spatial_edges
            output_spatial = self.human_linear_512to256(spatial_attn_out)

            # RH attention
            hidden_attn_weighted, _ = self.RH_attn(h_temporal=robot_states,
                                                robot_node=robot_node,
                                                temporal_edges=temporal_edges,
                                                h_spatials=output_spatial,
                                                spatial_edges=spatial_edges,
                                                each_seq_len=detected_human_num)
        else:
            # HH attention
            if self.args.use_self_attn:
                spatial_attn_out = self.HH_attn(spatial_edges, human_masks).view(seq_length, nenv, self.human_num, -1)
            else:
                spatial_attn_out = spatial_edges
            output_spatial = self.human_linear_512to256(spatial_attn_out)

            # RH attention
            hidden_attn_weighted, _ = self.RH_attn(robot_states, output_spatial, human_masks)

        # Do a forward pass through GRU
        outputs, h_nodes = self.humanNodeRNN(robot_states, hidden_attn_weighted, hidden_states_node_RNNs, masks)

        # Update the hidden and cell states
        all_hidden_states_node_RNNs = h_nodes
        outputs_return = outputs

        rnn_hxs['human_node_rnn'] = all_hidden_states_node_RNNs
        rnn_hxs['human_human_edge_rnn'] = all_hidden_states_edge_RNNs

        # x is the output and will be sent to actor and critic
        x = outputs_return[:, :, 0, :]

        hidden_critic = self.critic(x)
        hidden_actor = self.actor(x)

        for key in rnn_hxs:
            rnn_hxs[key] = rnn_hxs[key].squeeze(0)

        # end_time = time.time()
        # run_time = end_time - start_time
        # print("run time: ", run_time)

        if infer:
            hidden_critic_critic_linear = self.critic_linear(hidden_critic).squeeze(0)
            return hidden_critic_critic_linear, hidden_actor.squeeze(0), rnn_hxs
            # return self.critic_linear(hidden_critic).squeeze(0), hidden_actor.squeeze(0), rnn_hxs
        else:
            return self.critic_linear(hidden_critic).view(-1, 1), hidden_actor.view(-1, self.output_size), rnn_hxs


def reshapeT(T, seq_length, nenv):
    shape = T.size()[1:]
    return T.unsqueeze(0).reshape((seq_length, nenv, *shape))
