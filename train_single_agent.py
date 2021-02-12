import os
import argparse
import torch
import numpy as np
from learner import Learner
from torchkit.pytorch_utils import set_gpu_mode
from data_collection_config import args_ant_semicircle_sparse, \
    args_cheetah_vel, args_point_robot_sparse, args_gridworld, \
    args_point_robot_barrier, args_point_robot_rand_params


def main():
    for i in range(10):
        parser = argparse.ArgumentParser()
        # parser.add_argument('--env-type', default='gridworld')
        # parser.add_argument('--env-type', default='point_robot_sparse')
        # parser.add_argument('--env-type', default='cheetah_vel')
        # parser.add_argument('--env-type', default='ant_semicircle_sparse')
        parser.add_argument('--env-type', default='point_robot_wind')
        # parser.add_argument('--env-type', default='escape_room')


        args, rest_args = parser.parse_known_args()
        env = args.env_type

        # --- GridWorld ---
        if env == 'gridworld':
            args = args_gridworld.get_args(rest_args)
        # --- PointRobot ---
        elif env == 'point_robot_sparse':
            args = args_point_robot_sparse.get_args(rest_args)
        elif env == 'point_robot_wind':
            args = args_point_robot_rand_params.get_args(rest_args)
        elif env == 'escape_room':
            args = args_point_robot_barrier.get_args(rest_args)
        # --- Mujoco ---
        elif env == 'cheetah_vel':
            args = args_cheetah_vel.get_args(rest_args)
        elif env == 'ant_semicircle_sparse':
            args = args_ant_semicircle_sparse.get_args(rest_args)

        set_gpu_mode(torch.cuda.is_available())

        print("start new round - seed ", i+30)
        args.seed = i+30

        if hasattr(args, 'save_buffer') and args.save_buffer:
            os.makedirs(args.main_save_dir, exist_ok=True)

        learner = Learner(args)

        learner.train()


if __name__ == '__main__':
    main()

