import numpy as np
import torch
import os
from datetime import datetime
from crowd_sim.envs.utils.info import *


def write_csv(filename, corrs, human_counts):
    point_counts = len(corrs)

    h_r_g_counts = human_counts + 2

    points = np.zeros((h_r_g_counts, 3, point_counts))

    for i in range(0, point_counts):
        for j in range(0, human_counts):
            points[j, 0, i] = corrs[i][0][j].center[0]
            points[j, 1, i] = corrs[i][0][j].center[1]
            points[j, 2, i] = corrs[i][0][j].radius
    for i in range(0, point_counts):
        points[human_counts, 0, i] = corrs[i][1].center[0]
        points[human_counts, 1, i] = corrs[i][1].center[1]
        points[human_counts, 2, i] = corrs[i][1].radius
    for i in range(0, point_counts):
        points[human_counts+1, 0, i] = corrs[i][2]['RG_x']
        points[human_counts+1, 1, i] = corrs[i][2]['RG_y']

    points_data = points.astype(float)

    # print("")
    if os.path.exists(filename):
        os.remove(filename)

    # 获取数据的维度
    num_rows, num_cols, num_samples = points_data.shape

    # 将数据重新排列为N*3行的形式
    reshaped_data = np.reshape(points_data, (num_rows * num_cols, num_samples))

    # 将数据保存到CSV文件
    np.savetxt(filename, reshaped_data, delimiter=',')
    # print("Finished")




def evaluate(actor_critic, eval_envs, num_processes, device, test_size, logging, config, args, visualize=False):
    """ function to run all testing episodes and log the testing metrics """

    # initializations
    eval_episode_rewards = []
    eval_recurrent_hidden_states = {}

    node_num = 1
    edge_num = actor_critic.base.human_num + 1
    eval_recurrent_hidden_states['human_node_rnn'] = torch.zeros(num_processes, node_num, actor_critic.base.human_node_rnn_size,
                                                                 device=device)

    eval_recurrent_hidden_states['human_human_edge_rnn'] = torch.zeros(num_processes, edge_num,
                                                                       actor_critic.base.human_human_edge_rnn_size,
                                                                       device=device)

    eval_masks = torch.zeros(num_processes, 1, device=device)

    success_times = []
    collision_times = []
    timeout_times = []

    success = 0
    collision = 0
    timeout = 0
    too_close_ratios = []
    min_dist = []

    collision_cases = []
    timeout_cases = []

    all_path_len = []

    # to make it work with the virtualenv in sim2real
    if hasattr(eval_envs.venv, 'envs'):
        baseEnv = eval_envs.venv.envs[0].env
    else:
        baseEnv = eval_envs.venv.unwrapped.envs[0].env
    time_limit = baseEnv.time_limit

    # start the testing episodes
    for k in range(test_size):
        baseEnv.episode_k = k
        done = False
        rewards = []
        stepCounter = 0
        episode_rew = 0
        obs = eval_envs.reset()
        global_time = 0.0
        path_len = 0.
        too_close = 0.
        last_pos = obs['robot_node'][0, 0, :2].cpu().numpy()

        corrs_all = []

        while not done:
            stepCounter = stepCounter + 1
            # run inference on the NN policy
            with torch.no_grad():
                _, action, _, eval_recurrent_hidden_states = actor_critic.act(
                    obs,
                    eval_recurrent_hidden_states,
                    eval_masks,
                    deterministic=True)
            if not done:
                global_time = baseEnv.global_time

            # if the vec_pretext_normalize.py wrapper is used, send the predicted traj to env
            if args.env_name == 'CrowdSimPredRealGST-v0' and config.env.use_wrapper:
                out_pred = obs['spatial_edges'][:, :, 2:].to('cpu').numpy()
                # send manager action to all processes
                ack = eval_envs.talk2Env(out_pred)
                assert all(ack)
            # render
            if visualize:
                corrs = eval_envs.render()
                corrs_all.append(corrs)


            # Obser reward and next obs
            obs, rew, done, infos = eval_envs.step(action)

            # record the info for calculating testing metrics
            rewards.append(rew)

            path_len = path_len + np.linalg.norm(obs['robot_node'][0, 0, :2].cpu().numpy() - last_pos)
            last_pos = obs['robot_node'][0, 0, :2].cpu().numpy()

            if isinstance(infos[0]['info'], Danger):
                too_close = too_close + 1
                min_dist.append(infos[0]['info'].min_dist)

            episode_rew += rew[0]

            eval_masks = torch.tensor(
                [[0.0] if done_ else [1.0] for done_ in done],
                dtype=torch.float32,
                device=device)

            for info in infos:
                if 'episode' in info.keys():
                    eval_episode_rewards.append(info['episode']['r'])

        # 保存corrs_all中的坐标和半径，以绘制运动轨迹曲线
        # 获取当前系统时间
        current_time = datetime.now()

        # 将时间格式化为指定的字符串形式
        # formatted_time = current_time.strftime("%Y%m%d%H%M%S")
        # formatted_k = str(k)
        # human_counts_4_plot = len(corrs_all[0][0])
        # write_csv("MatlabPlot/data/"+formatted_k + ".csv", corrs_all, human_counts_4_plot)

        # an episode ends!
        # print('')
        # print('Reward={}'.format(episode_rew))
        # print('Episode', k, 'ends in', stepCounter)
        all_path_len.append(path_len)
        too_close_ratios.append(too_close / stepCounter * 100)

        if isinstance(infos[0]['info'], ReachGoal):
            success += 1
            success_times.append(global_time)
            print(k, 'Success')
        elif isinstance(infos[0]['info'], Collision):
            collision += 1
            collision_cases.append(k)
            collision_times.append(global_time)
            print(k, 'Collision')
        elif isinstance(infos[0]['info'], Timeout):
            timeout += 1
            timeout_cases.append(k)
            timeout_times.append(time_limit)
            print(k, 'Time out')
        elif isinstance(infos[0]['info'] is None):
            pass
        else:
            raise ValueError('Invalid end signal from environment')

    # all episodes end
    success_rate = success / test_size
    collision_rate = collision / test_size
    timeout_rate = timeout / test_size
    assert success + collision + timeout == test_size
    avg_nav_time = sum(success_times) / len(
        success_times) if success_times else time_limit  # baseEnv.env.time_limit

    # logging
    logging.info(
        'Testing success rate: {:.2f}, collision rate: {:.2f}, timeout rate: {:.2f}, '
        'nav time: {:.2f}, path length: {:.2f}, average intrusion ratio: {:.2f}%, '
        'average minimal distance during intrusions: {:.2f}'.
        format(success_rate, collision_rate, timeout_rate, avg_nav_time, np.mean(all_path_len),
               np.mean(too_close_ratios), np.mean(min_dist)))

    logging.info('Collision cases: ' + ' '.join([str(x) for x in collision_cases]))
    logging.info('Timeout cases: ' + ' '.join([str(x) for x in timeout_cases]))
    print(" Evaluation using {} episodes: mean reward {:.5f}\n".format(
        len(eval_episode_rewards), np.mean(eval_episode_rewards)))

    eval_envs.close()
