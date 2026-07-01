import os
import time
import argparse
import torch


def get_opts(args=None):
    parser = argparse.ArgumentParser(
        description="基于注意力的模型通过深度学习解决最小顶点覆盖问题")

    # 数据参数
    parser.add_argument("--problem", type=str, default="MWVC", help="需要解决的问题")
    parser.add_argument('--dataset', type=str, help='用于验证的数据集文件')
    parser.add_argument('--n_size', type=int, default=500, help='结点的最大数量')
    parser.add_argument('--min_p', type=float, default=0.001, help='边的最小生成概率')
    parser.add_argument('--max_p', type=float, default=0.05, help='边的最大生成概率')
    parser.add_argument('--graph_size', type=int, default=64, help='图的生成数量')
    parser.add_argument('--text_node_size', type=int, default=0, help='测试图的最小结点数')

    # 模型参数att_bsz
    parser.add_argument('--model', default='attention', help="模型类型，'attention'（默认）")
    parser.add_argument('--n_heads', type=int, default=4, help='注意力头数')
    parser.add_argument('--input_dim', type=int, default=4, help='静态特征维度数，需要跟encoder里的get_input()保持一致')
    parser.add_argument('--move_dim', type=int, default=3, help='动态特征维度数，需要跟Model_WVC里的get_move_input()保持一致')
    parser.add_argument('--embedding_dim', type=int, default=128, help='输入嵌入的维度')
    parser.add_argument('--n_encoder_layers', type=int, default=6, help='编码器层数')
    parser.add_argument('--n_decode_layers', type=int, default=3, help='解码器层数')
    parser.add_argument('--normalization', default='instance', help="归一化类型，'batch'（默认）或 'instance'")

    # 训练参数
    parser.add_argument('--lr_model', type=float, default=1e-4, help="设置当前网络的学习率")
    parser.add_argument('--lr_decay', type=float, default=0.995, help='每个epoch的学习率衰减')
    parser.add_argument('--eval_only', action='store_true', help='设置此值仅评估模型')
    parser.add_argument('--n_epochs', type=int, default=200, help='训练的epoch数量')
    parser.add_argument('--seed', type=int, default=1234, help='使用的随机种子')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='梯度裁剪的最大L2范数，默认1.0（0表示禁用裁剪）')
    parser.add_argument('--no_cuda', action='store_true', help='禁用CUDA')
    parser.add_argument('--debug', action='store_true',
                        help='开启调试模式（torch.autograd.detect_anomaly），会显著拖慢训练，默认关闭')
    parser.add_argument('--eval_num', type=int, default=4000, help='评估步数')
    parser.add_argument('--ppo_num', type=int, default=4, help='收集几步数据进行一次PPO')
    parser.add_argument('--taboo_num', type=int, default=1, help='禁忌长度')
    parser.add_argument('--updata_old', type=int, default=20, help='更新旧模型的步数')
    parser.add_argument('--step_num', type=int, default=200, help='每轮训练的步数')
    
    # 杂项
    parser.add_argument('--run_name', default='run', help='用于识别运行的名称')
    parser.add_argument('--log_step', default=100,type=int, help='每隔多少步输出一次信息')
    parser.add_argument('--output_dir', default='outputs', help='输出模型的目录')
    parser.add_argument('--epoch_start', type=int, default=0,help='从第几个epoch开始（与学习率衰减相关）')
    parser.add_argument('--checkpoint_epochs', type=int, default=1,help='每n个epoch保存检查点（默认1），0表示不保存检查点')
    parser.add_argument('--load_path', help='从中加载模型参数和优化器状态的路径')
    parser.add_argument('--resume', help='从之前的检查点文件恢复')
    parser.add_argument('--gamma', type=float, default=0.99, help='奖励折扣因子')
    parser.add_argument('--lam', type=float, default=0.98, help='GAE lambda')
    parser.add_argument('--epsilon', type=float, default=0.2, help='PPO裁剪范围')
    parser.add_argument('--theta', type=float, default=0.15, help='熵系数')
    parser.add_argument('--huber_delta', type=float, default=1.0, help='Huber loss的阈值')

    opts = parser.parse_args(args)

    opts.use_cuda = torch.cuda.is_available() and not opts.no_cuda
    opts.run_name = "{}_{}".format(opts.run_name, time.strftime("%Y%m%dT%H%M%S"))
    opts.save_dir = os.path.join(
        opts.output_dir,
        opts.problem,
        opts.run_name
    )
    return opts
