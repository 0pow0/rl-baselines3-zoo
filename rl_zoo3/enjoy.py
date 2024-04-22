import argparse
import importlib
import os
import sys

import numpy as np
import torch as th
import yaml
from huggingface_sb3 import EnvironmentName
from stable_baselines3.common.callbacks import tqdm
from stable_baselines3.common.utils import set_random_seed
import pandas as pd

import rl_zoo3.import_envs  # noqa: F401 pylint: disable=unused-import
from rl_zoo3 import ALGOS, create_test_env, get_saved_hyperparams
from rl_zoo3.exp_manager import ExperimentManager
from rl_zoo3.load_from_hub import download_from_hub
from rl_zoo3.utils import StoreDict, get_model_path
import rui.rui_id as rui

import os
import sys
sys.path.insert(0, '/home/rzuo02/work/rl-baselines3-zoo')
sys.path.insert(0, '/home/rzuo02/work/explain-mario-ppo/NN-Explainer/src')

from data.dataset import AtariDataset

sys.path.insert(0, os.path.abspath(".."))

from rui_utils import save_img
import torchvision.transforms as T
import torch
from PIL import Image
import cv2

dataset = AtariDataset("/home/rzuo02/work/rl-baselines3-zoo/datasets/Enduro/")
# change coefficient of default torch grayscale for R: 0.2989 -> 0.299 to achieve consistency with cv2
def rgb_to_grayscale(img, num_output_channels: int = 1):
    if img.ndim < 3:
        raise TypeError(f"Input image tensor should have at least 3 dimensions, but found {img.ndim}")
    # _assert_channels(img, [1, 3])

    if num_output_channels not in (1, 3):
        raise ValueError("num_output_channels should be either 1 or 3")

    if img.shape[-3] == 3:
        r, g, b = img.unbind(dim=-3)
        # This implementation closely follows the TF one:
        # https://github.com/tensorflow/tensorflow/blob/v2.3.0/tensorflow/python/ops/image_ops_impl.py#L2105-L2138
        l_img = (0.299 * r + 0.587 * g + 0.114 * b).to(img.dtype)
        l_img = l_img.unsqueeze(dim=-3)
    else:
        l_img = img.clone()

    if num_output_channels == 3:
        return l_img.expand(img.shape)

    return l_img

# batch: (..., 1, H, W)
def squeeze_rgb_gray_channel(batch):
    return torch.squeeze(batch, dim=-3)

def foo(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame = cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA)
    return frame

transformer = T.Compose([T.Lambda(rgb_to_grayscale), T.Lambda(squeeze_rgb_gray_channel), T.Resize(size=(84, 4*84))])

def enjoy() -> None:  # noqa: C901
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", help="environment ID", type=EnvironmentName, default="CartPole-v1")
    parser.add_argument("-f", "--folder", help="Log folder", type=str, default="rl-trained-agents")
    parser.add_argument("--algo", help="RL Algorithm", default="ppo", type=str, required=False, choices=list(ALGOS.keys()))
    parser.add_argument("-n", "--n-timesteps", help="number of timesteps", default=1000, type=int)
    parser.add_argument("--num-threads", help="Number of threads for PyTorch (-1 to use default)", default=-1, type=int)
    parser.add_argument("--n-envs", help="number of environments", default=1, type=int)
    parser.add_argument("--exp-id", help="Experiment ID (default: 0: latest, -1: no exp folder)", default=0, type=int)
    parser.add_argument("--verbose", help="Verbose mode (0: no output, 1: INFO)", default=1, type=int)
    parser.add_argument(
        "--no-render", action="store_true", default=False, help="Do not render the environment (useful for tests)"
    )
    parser.add_argument("--deterministic", action="store_true", default=False, help="Use deterministic actions")
    parser.add_argument("--device", help="PyTorch device to be use (ex: cpu, cuda...)", default="auto", type=str)
    parser.add_argument(
        "--load-best", action="store_true", default=False, help="Load best model instead of last model if available"
    )
    parser.add_argument(
        "--load-checkpoint",
        type=int,
        help="Load checkpoint instead of last model if available, "
        "you must pass the number of timesteps corresponding to it",
    )
    parser.add_argument(
        "--load-last-checkpoint",
        action="store_true",
        default=False,
        help="Load last checkpoint instead of last model if available",
    )
    parser.add_argument("--stochastic", action="store_true", default=False, help="Use stochastic actions")
    parser.add_argument(
        "--norm-reward", action="store_true", default=False, help="Normalize reward if applicable (trained with VecNormalize)"
    )
    parser.add_argument("--seed", help="Random generator seed", type=int, default=0)
    parser.add_argument("--reward-log", help="Where to log reward", default="", type=str)
    parser.add_argument(
        "--gym-packages",
        type=str,
        nargs="+",
        default=[],
        help="Additional external Gym environment package modules to import",
    )
    parser.add_argument(
        "--env-kwargs", type=str, nargs="+", action=StoreDict, help="Optional keyword argument to pass to the env constructor"
    )
    parser.add_argument(
        "--custom-objects", action="store_true", default=False, help="Use custom objects to solve loading issues"
    )
    parser.add_argument(
        "-P",
        "--progress",
        action="store_true",
        default=False,
        help="if toggled, display a progress bar using tqdm and rich",
    )
    parser.add_argument('--atari-env', choices=['BeamRider', 'Breakout', 'Enduro', 'Pong', 'Qbert', 'Seaquest', 'SpaceInvaders', 'MsPacman', 'Asteroids', 'RoadRunner'],
                        default='Pong', type=str, help='which atari env to use')
    parser.add_argument('--dataset_path', default='datasets', type=str, help='where the collected dataset to be stored')

    args = parser.parse_args()

    df = pd.DataFrame()
    df["id"] = ""
    df["action"] = ""

    # Going through custom gym packages to let them register in the global registory
    for env_module in args.gym_packages:
        importlib.import_module(env_module)

    env_name: EnvironmentName = args.env
    algo = args.algo
    folder = args.folder

    try:
        _, model_path, log_path = get_model_path(
            args.exp_id,
            folder,
            algo,
            env_name,
            args.load_best,
            args.load_checkpoint,
            args.load_last_checkpoint,
        )
    except (AssertionError, ValueError) as e:
        # Special case for rl-trained agents
        # auto-download from the hub
        if "rl-trained-agents" not in folder:
            raise e
        else:
            print("Pretrained model not found, trying to download it from sb3 Huggingface hub: https://huggingface.co/sb3")
            # Auto-download
            download_from_hub(
                algo=algo,
                env_name=env_name,
                exp_id=args.exp_id,
                folder=folder,
                organization="sb3",
                repo_name=None,
                force=False,
            )
            # Try again
            _, model_path, log_path = get_model_path(
                args.exp_id,
                folder,
                algo,
                env_name,
                args.load_best,
                args.load_checkpoint,
                args.load_last_checkpoint,
            )

    print(f"Loading {model_path}")

    # Off-policy algorithm only support one env for now
    off_policy_algos = ["qrdqn", "dqn", "ddpg", "sac", "her", "td3", "tqc"]

    set_random_seed(args.seed)

    if args.num_threads > 0:
        if args.verbose > 1:
            print(f"Setting torch.num_threads to {args.num_threads}")
        th.set_num_threads(args.num_threads)

    is_atari = ExperimentManager.is_atari(env_name.gym_id)
    is_minigrid = ExperimentManager.is_minigrid(env_name.gym_id)

    stats_path = os.path.join(log_path, env_name)
    hyperparams, maybe_stats_path = get_saved_hyperparams(stats_path, norm_reward=args.norm_reward, test_mode=True)

    # load env_kwargs if existing
    env_kwargs = {}
    args_path = os.path.join(log_path, env_name, "args.yml")
    if os.path.isfile(args_path):
        with open(args_path) as f:
            loaded_args = yaml.load(f, Loader=yaml.UnsafeLoader)
            if loaded_args["env_kwargs"] is not None:
                env_kwargs = loaded_args["env_kwargs"]
    # overwrite with command line arguments
    if args.env_kwargs is not None:
        env_kwargs.update(args.env_kwargs)

    log_dir = args.reward_log if args.reward_log != "" else None

    env = create_test_env(
        env_name.gym_id,
        n_envs=args.n_envs,
        stats_path=maybe_stats_path,
        seed=args.seed,
        log_dir=log_dir,
        should_render=not args.no_render,
        hyperparams=hyperparams,
        env_kwargs=env_kwargs,
    )

    kwargs = dict(seed=args.seed)
    if algo in off_policy_algos:
        # Dummy buffer size as we don't need memory to enjoy the trained agent
        kwargs.update(dict(buffer_size=1))
        # Hack due to breaking change in v1.6
        # handle_timeout_termination cannot be at the same time
        # with optimize_memory_usage
        if "optimize_memory_usage" in hyperparams:
            kwargs.update(optimize_memory_usage=False)

    # Check if we are running python 3.8+
    # we need to patch saved model under python 3.6/3.7 to load them
    newer_python_version = sys.version_info.major == 3 and sys.version_info.minor >= 8

    custom_objects = {}
    if newer_python_version or args.custom_objects:
        custom_objects = {
            "learning_rate": 0.0,
            "lr_schedule": lambda _: 0.0,
            "clip_range": lambda _: 0.0,
        }

    if "HerReplayBuffer" in hyperparams.get("replay_buffer_class", ""):
        kwargs["env"] = env

    model = ALGOS[algo].load(model_path, custom_objects=custom_objects, device=args.device, **kwargs)
    obs = env.reset()
    print(obs.shape)

    # Deterministic by default except for atari games
    stochastic = args.stochastic or (is_atari or is_minigrid) and not args.deterministic
    deterministic = not stochastic
    print("RUIII")
    print(deterministic)

    episode_reward = 0.0
    episode_rewards, episode_lengths = [], []
    ep_len = 0
    # For HER, monitor success rate
    successes = []
    lstm_states = None
    episode_start = np.ones((env.num_envs,), dtype=bool)

    generator = range(args.n_timesteps)
    if args.progress:
        if tqdm is None:
            raise ImportError("Please install tqdm and rich to use the progress bar")
        generator = tqdm(generator)
    idx = 0
    # last_x = dataset[0][0]
    try:
        for _ in generator:
            # print(type(obs), obs.shape)
            # x, anno = dataset[idx]
            # x = x.unsqueeze(0)
            # print(x.shape)
            # x = transformer(x)
            # print(x.shape)
            # x = x.split(84, dim=2)
            # print(x[0].shape)
            # x = torch.stack(x, dim=3)
            # print(x[0].shape)
            # x = x.numpy()
            # print((x == obs).all())
            # unique, counts = np.unique((x == obs), return_counts=True)
            # print(dict(zip(unique, counts)))
            # save_img(obs[0, :, :, 0], 'obs0.png')
            # save_img(obs[0, :, :, 1], 'obs1.png')
            # save_img(obs[0, :, :, 2], 'obs2.png')
            # save_img(obs[0, :, :, 3], 'obs3.png')
            # save_img(x[0, :, :, 0], 'x0.png')
            # save_img(x[0, :, :, 1], 'x1.png')
            # save_img(x[0, :, :, 2], 'x2.png')
            # save_img(x[0, :, :, 3], 'x3.png')
            # assert(not torch.all(torch.eq(obs, last_x)))
            # last_x = obs
            # x = obs.unsqueeze(0)
            # x = transformer(x)
            # x = torch.stack(x.split(84, 2), 1)
            # x = x.permute(0, 2, 3, 1).numpy()
            # print(type(x), x.shape)
            # action, lstm_states = model.predict(
                # x,  # type: ignore[arg-type]
                # state=lstm_states,
                # episode_start=episode_start,
                # deterministic=deterministic,
            # )
            # print(action, anno)
            # idx += 1
            # continue
            # assert(action)
            # obs = obs.unsqueeze(0).permute(0, 2, 3, 1).numpy()
            # lstm_states_new = lstm_states
            # episode_start_new = episode_start
            # print(obs, obs.min(), obs.max())
            # obs, anno = dataset[idx]
            action, lstm_states = model.predict(
                obs,  # type: ignore[arg-type]
                state=lstm_states,
                episode_start=episode_start,
                deterministic=True,
            )
            # action_new, __ = model.predict(
            #     x,  # type: ignore[arg-type]
            #     state=lstm_states,
            #     episode_start=episode_start,
            #     deterministic=True,
            # )
            # print(action, action_new)
            # assert(action == action_new)
            # idx += 1
            # print(action, anno)
            # assert(action[0] == anno['action'])
            # idx += 1
            # print('obs.shape', obs.shape)
            # foo = torch.split(torch.from_numpy(obs), 1, dim=3)
            # foo = torch.cat(foo, dim=2).squeeze(3)
            # save_pil_img(foo.squeeze(0).numpy(), 'obs.png')
            # print('foo.shape', foo.shape)
            # print(action, anno)
            # exit()
            # print('rui.already_reset ', rui.already_reset)
            if (rui.empty == 0) and (rui.already_reset == 1):
                for i in range(4):
                    # (1, 210, 160, 12)
                    x = rui.stacked_frames
                    # # print(x.shape)
                    # save_img(x[0, :, :, 0:3], 'x_full0.png')
                    # save_img(x[0, :, :, 3:6], 'x_full1.png')
                    # save_img(x[0, :, :, 6:9], 'x_full2.png')
                    # save_img(x[0, :, :, 9:], 'x_full3.png')

                    # xs = []
                    # for j in range(4):
                    #     xs.append(torch.from_numpy(x[0, :, :, 3*j:3*j+3]))
                    # x = torch.cat(xs, dim=1).permute(2, 0, 1).unsqueeze(0)
                    # print(x.shape)
                    # x = transformer(x)
                    # x = torch.split(x, 84, dim=2)
                    # print(x[0].shape)
                    # x = torch.stack(x, dim=3).numpy()
                    # print(x.shape)
                    # print((x == obs).all())
                    # action_new, __ = model.predict(
                    #     x,  # type: ignore[arg-type]
                    #     state=lstm_states,
                    #     episode_start=episode_start,
                    #     deterministic=True,
                    # )
                    # print(action, action_new)
                    # unique, counts = np.unique((x == obs), return_counts=True)
                    # print(dict(zip(unique, counts)))
                    # save_img(x[0, :, :, 0], 'x0.png')
                    # save_img(x[0, :, :, 1], 'x1.png')
                    # save_img(x[0, :, :, 2], 'x2.png')
                    # save_img(x[0, :, :, 3], 'x3.png')
                    # save_img(obs[0, :, :, 0], 'obs0.png')
                    # save_img(obs[0, :, :, 1], 'obs1.png')
                    # save_img(obs[0, :, :, 2], 'obs2.png')
                    # save_img(obs[0, :, :, 3], 'obs3.png')
                    # assert(action == action_new)
                    # break

                    xs = []
                    for j in range(4):
                        xs.append(foo(x[0, :, :, 3*j:3*j+3]))
                    x = np.stack(xs, axis=2)[np.newaxis, ...]
                    print(x.shape, obs.shape)
                    # print(x)
                    # print(torch.from_numpy(x).shape)
                    # print(torch.from_numpy(x))
                    # exit()
                    print((x == obs).all())
                    # save_img(x[0, :, :, 0], 'x0.png')
                    # save_img(x[0, :, :, 1], 'x1.png')
                    # save_img(x[0, :, :, 2], 'x2.png')
                    # save_img(x[0, :, :, 3], 'x3.png')
                    # save_img(obs[0, :, :, 0], 'obs0.png')
                    # save_img(obs[0, :, :, 1], 'obs1.png')
                    # save_img(obs[0, :, :, 2], 'obs2.png')
                    # save_img(obs[0, :, :, 3], 'obs3.png')
                    action_new, lstm_states = model.predict(
                        x,  # type: ignore[arg-type]
                        state=lstm_states,
                        episode_start=episode_start,
                        deterministic=True,
                    )
                    print(action, action_new, action_new == action)
                    assert(action_new == action)
                    # save_img(rui.stacked_frames[0, :, :, 3*i : 3*i +3], f'{args.dataset_path}/{args.atari_env}/{rui.frame_id}_{i}.png')
                    # save_img(obs[0, :, :, i], f'foo_{i}.png')
                    # save_img(obs[0, :, :, i], f'{args.dataset_path}/{args.atari_env}/{rui.frame_id}_{i}.png')
                    pass
                # df.loc[len(df)] = [rui.frame_id, action[0]]
                rui.empty = 1
                # img = Image.open('/home/rzuo02/work/rl-baselines3-zoo/datasets/Enduro/9999_0.png').convert("L") 
                # print(T.PILToTensor()(img))
                # exit()
                # imgs = []
                # for i in range(4):
                    # img = Image.open(f'foo_{i}.png').convert("L") 
                    # assert(torch.all(torch.eq(torch.from_numpy(obs[0, :, :, i]), T.PILToTensor()(img))))
                    # print(img)
                    # print()
                    # img.save(f'{i}.png')
                    # save_pil_img(T.PILToTensor()(img).squeeze(0).numpy(), f'{i}.png')
                    # imgs.append(img)
                # imgs = [T.PILToTensor()(img) for img in imgs] 
                # imgs = torch.cat(imgs, dim=2)
                # print('imgs.shape', imgs.shape)
                # save_pil_img(imgs.squeeze(0).numpy(), 'new_obs.png')
                # print((imgs == foo).all())
                # assert(torch.all(torch.eq(foo, imgs)))

                # new_obs = imgs.split(84, dim=2)
                # new_obs = torch.stack(new_obs, dim=3)
                # print('new_obs.shape', new_obs.shape)
                # new_obs = new_obs.numpy()
                # # print(action, action_new)
                # print('obs.type', type(obs), 'obs.shape', obs.shape)
                # print('new_obs.type', type(new_obs), 'new_obs.shape', new_obs.shape)
                # assert((obs == new_obs).all())
                print("CONSUMER DONE")

                # action_new, lstm_states_new = model.predict(
                #     new_obs,  # type: ignore[arg-type]
                #     state=lstm_states,
                #     episode_start=episode_start,
                #     deterministic=True,
                # )
                # print('action_type', type(action), 'action', action, 'action_new_type', type(action_new), 'action_new', action_new)
                # assert(action[0] == action_new[0])
            else:
                break
            obs, reward, done, infos = env.step(action)

            episode_start = done

            if not args.no_render:
                env.render("human")

            episode_reward += reward[0]
            ep_len += 1

            if args.n_envs == 1:
                # For atari the return reward is not the atari score
                # so we have to get it from the infos dict
                if is_atari and infos is not None and args.verbose >= 1:
                    episode_infos = infos[0].get("episode")
                    if episode_infos is not None:
                        print(f"Atari Episode Score: {episode_infos['r']:.2f}")
                        print("Atari Episode Length", episode_infos["l"])

                if done and not is_atari and args.verbose > 0:
                    # NOTE: for env using VecNormalize, the mean reward
                    # is a normalized reward when `--norm_reward` flag is passed
                    print(f"Episode Reward: {episode_reward:.2f}")
                    print("Episode Length", ep_len)
                    episode_rewards.append(episode_reward)
                    episode_lengths.append(ep_len)
                    episode_reward = 0.0
                    ep_len = 0

                # Reset also when the goal is achieved when using HER
                if done and infos[0].get("is_success") is not None:
                    if args.verbose > 1:
                        print("Success?", infos[0].get("is_success", False))

                    if infos[0].get("is_success") is not None:
                        successes.append(infos[0].get("is_success", False))
                        episode_reward, ep_len = 0.0, 0

    except KeyboardInterrupt:
        pass

    if args.verbose > 0 and len(successes) > 0:
        print(f"Success rate: {100 * np.mean(successes):.2f}%")

    if args.verbose > 0 and len(episode_rewards) > 0:
        print(f"{len(episode_rewards)} Episodes")
        print(f"Mean reward: {np.mean(episode_rewards):.2f} +/- {np.std(episode_rewards):.2f}")

    if args.verbose > 0 and len(episode_lengths) > 0:
        print(f"Mean episode length: {np.mean(episode_lengths):.2f} +/- {np.std(episode_lengths):.2f}")

    env.close()
    df = df.sort_index()
    # df.to_csv(f'{args.dataset_path}/{args.atari_env}/annotations.csv', index=False)


if __name__ == "__main__":
    enjoy()
