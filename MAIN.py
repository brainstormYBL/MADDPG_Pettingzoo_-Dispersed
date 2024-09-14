from pettingzoo.mpe import simple_adversary_v3, simple_spread_v3, simple_tag_v3
from Runner.runner import RUNNER
from Utils.defineParameteres import define_parameters
from maddpgAgent.maddpgAgent import MADDPGAGENT


def get_env(env_name, ep_len=25):
    """create environment and get observation and action dimension of each agent in this environment"""
    new_env = None
    if env_name == 'simple_adversary_v3':
        new_env = simple_adversary_v3.parallel_env(max_cycles=ep_len)
    if env_name == 'simple_spread_v3':
        new_env = simple_spread_v3.parallel_env(max_cycles=ep_len, render_mode="rgb_array")
    if env_name == 'simple_tag_v3':
        new_env = simple_tag_v3.parallel_env(max_cycles=ep_len)

    new_env.reset()
    _dim_info = {}
    for agent_id in new_env.agents:
        _dim_info[agent_id] = []  # [obs_dim, act_dim]
        _dim_info[agent_id].append(new_env.observation_space(agent_id).shape[0])
        _dim_info[agent_id].append(new_env.action_space(agent_id).n)

    return new_env, _dim_info


if __name__ == '__main__':
    # 定义参数
    args = define_parameters()
    # 创建环境
    env, dim_info = get_env(args.env_name, args.episode_length)
    # 创建MA-DDPG智能体 dim_info: 字典，键为智能体名字 内容为二维数组 分别表示观测维度和动作维度 是观测不是状态 需要注意
    agent = MADDPGAGENT(dim_info, args.buffer_capacity, args.batch_size, args.actor_lr, args.critic_lr)
    # 创建运行对象
    runner = RUNNER(agent, env, args)
    # 开始训练
    runner.train()
