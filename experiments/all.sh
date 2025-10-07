#!/bin/bash

# Run all DQN and PPO experiments
# Each command is fully explicit with all parameters

set -e

LOG_DIR="logs"
mkdir -p $LOG_DIR

echo "=========================================="
echo "Starting All Experiments"
echo "=========================================="

# =============================================================================
# CARTPOLE-V1
# =============================================================================

uv run train.py \
    --algorithm dqn \
    --env CartPole-v1 \
    --episodes 600 \
    --batch-size 128 \
    --buffer-size 10000 \
    --lr 1e-4 \
    --gamma 0.99 \
    --eps-start 0.9 \
    --eps-end 0.05 \
    --eps-decay 1000 \
    --tau 0.005 \
    --log-interval 10 \
    --seed 42 \
    --device cpu \
    --save-dir checkpoints/dqn_cartpole \
    --record-video \
    --video-dir videos/dqn_cartpole \
    2>&1 | tee $LOG_DIR/dqn_cartpole.log

uv run train.py \
    --algorithm ppo \
    --env CartPole-v1 \
    --total-timesteps 100000 \
    --n-steps 128 \
    --n-envs 4 \
    --n-epochs 4 \
    --batch-size 64 \
    --lr 2.5e-4 \
    --gamma 0.99 \
    --gae-lambda 0.95 \
    --clip-epsilon 0.2 \
    --value-coef 0.5 \
    --entropy-coef 0.01 \
    --log-interval 10 \
    --seed 42 \
    --device cpu \
    --save-dir checkpoints/ppo_cartpole \
    --record-video \
    --video-dir videos/ppo_cartpole \
    2>&1 | tee $LOG_DIR/ppo_cartpole.log

# =============================================================================
# ACROBOT-V1
# =============================================================================
uv run train.py \
    --algorithm dqn \
    --env Acrobot-v1 \
    --episodes 1000 \
    --batch-size 128 \
    --buffer-size 10000 \
    --lr 1e-4 \
    --gamma 0.99 \
    --eps-start 0.9 \
    --eps-end 0.05 \
    --eps-decay 1000 \
    --tau 0.005 \
    --log-interval 10 \
    --seed 42 \
    --device cpu \
    --save-dir checkpoints/dqn_acrobot \
    --record-video \
    --video-dir videos/dqn_acrobot \
    2>&1 | tee $LOG_DIR/dqn_acrobot.log

uv run train.py \
    --algorithm ppo \
    --env Acrobot-v1 \
    --total-timesteps 500000 \
    --n-steps 512 \
    --n-envs 4 \
    --n-epochs 4 \
    --batch-size 64 \
    --lr 2.5e-4 \
    --gamma 0.99 \
    --gae-lambda 0.95 \
    --clip-epsilon 0.2 \
    --value-coef 0.5 \
    --entropy-coef 0.01 \
    --log-interval 10 \
    --seed 42 \
    --device cpu \
    --save-dir checkpoints/ppo_acrobot \
    --record-video \
    --video-dir videos/ppo_acrobot \
    2>&1 | tee $LOG_DIR/ppo_acrobot.log

# =============================================================================
# MOUNTAINCAR-V0
# =============================================================================
uv run train.py \
    --algorithm dqn \
    --env MountainCar-v0 \
    --episodes 1000 \
    --batch-size 128 \
    --buffer-size 10000 \
    --lr 1e-4 \
    --gamma 0.99 \
    --eps-start 0.9 \
    --eps-end 0.05 \
    --eps-decay 1000 \
    --tau 0.005 \
    --log-interval 10 \
    --seed 42 \
    --device cpu \
    --save-dir checkpoints/dqn_mountaincar \
    --record-video \
    --video-dir videos/dqn_mountaincar \
    2>&1 | tee $LOG_DIR/dqn_mountaincar.log

uv run train.py \
    --algorithm ppo \
    --env MountainCar-v0 \
    --total-timesteps 500000 \
    --n-steps 512 \
    --n-envs 4 \
    --n-epochs 4 \
    --batch-size 64 \
    --lr 2.5e-4 \
    --gamma 0.99 \
    --gae-lambda 0.95 \
    --clip-epsilon 0.2 \
    --value-coef 0.5 \
    --entropy-coef 0.01 \
    --log-interval 10 \
    --seed 42 \
    --device cpu \
    --save-dir checkpoints/ppo_mountaincar \
    --record-video \
    --video-dir videos/ppo_mountaincar \
    2>&1 | tee $LOG_DIR/ppo_mountaincar.log

# =============================================================================
# LUNARLANDER-V2 (DISCRETE)
# =============================================================================
uv run train.py \
    --algorithm dqn \
    --env LunarLander-v3 \
    --episodes 1000 \
    --batch-size 128 \
    --buffer-size 10000 \
    --lr 5e-4 \
    --gamma 0.99 \
    --eps-start 0.9 \
    --eps-end 0.05 \
    --eps-decay 1000 \
    --tau 0.005 \
    --log-interval 10 \
    --seed 42 \
    --device cpu \
    --save-dir checkpoints/dqn_lunarlander \
    --record-video \
    --video-dir videos/dqn_lunarlander \
    2>&1 | tee $LOG_DIR/dqn_lunarlander.log

uv run train.py \
    --algorithm ppo \
    --env LunarLander-v3 \
    --total-timesteps 1000000 \
    --n-steps 2048 \
    --n-envs 1 \
    --n-epochs 10 \
    --batch-size 64 \
    --lr 3e-4 \
    --gamma 0.99 \
    --gae-lambda 0.95 \
    --clip-epsilon 0.2 \
    --value-coef 0.5 \
    --entropy-coef 0.01 \
    --log-interval 10 \
    --seed 42 \
    --device cpu \
    --save-dir checkpoints/ppo_lunarlander \
    --record-video \
    --video-dir videos/ppo_lunarlander \
    2>&1 | tee $LOG_DIR/ppo_lunarlander.log

# =============================================================================
# PENDULUM-V1 (CONTINUOUS)
# =============================================================================
uv run train.py \
    --algorithm ppo \
    --env Pendulum-v1 \
    --total-timesteps 200000 \
    --n-steps 2048 \
    --n-envs 1 \
    --n-epochs 10 \
    --batch-size 64 \
    --lr 3e-4 \
    --gamma 0.99 \
    --gae-lambda 0.95 \
    --clip-epsilon 0.2 \
    --value-coef 0.5 \
    --entropy-coef 0.0 \
    --log-interval 10 \
    --seed 42 \
    --device cpu \
    --save-dir checkpoints/ppo_pendulum \
    --record-video \
    --video-dir videos/ppo_pendulum \
    2>&1 | tee $LOG_DIR/ppo_pendulum.log

# =============================================================================
# BIPEDALWALKER-V3 (CONTINUOUS)
# =============================================================================
uv run train.py \
    --algorithm ppo \
    --env BipedalWalker-v3 \
    --total-timesteps 2000000 \
    --n-steps 2048 \
    --n-envs 1 \
    --n-epochs 10 \
    --batch-size 64 \
    --lr 3e-4 \
    --gamma 0.99 \
    --gae-lambda 0.95 \
    --clip-epsilon 0.2 \
    --value-coef 0.5 \
    --entropy-coef 0.0 \
    --log-interval 10 \
    --seed 42 \
    --device cpu \
    --save-dir checkpoints/ppo_bipedalwalker \
    --record-video \
    --video-dir videos/ppo_bipedalwalker \
    2>&1 | tee $LOG_DIR/ppo_bipedalwalker.log

# =============================================================================
# LUNARLANDERCONTINUOUS-V2 (CONTINUOUS)
# =============================================================================
uv run train.py \
    --algorithm ppo \
    --env LunarLanderContinuous-v2 \
    --total-timesteps 1000000 \
    --n-steps 2048 \
    --n-envs 1 \
    --n-epochs 10 \
    --batch-size 64 \
    --lr 3e-4 \
    --gamma 0.99 \
    --gae-lambda 0.95 \
    --clip-epsilon 0.2 \
    --value-coef 0.5 \
    --entropy-coef 0.0 \
    --log-interval 10 \
    --seed 42 \
    --device cpu \
    --save-dir checkpoints/ppo_lunarlandercontinuous \
    --record-video \
    --video-dir videos/ppo_lunarlandercontinuous \
    2>&1 | tee $LOG_DIR/ppo_lunarlandercontinuous.log

# =============================================================================
# TAXI-V3 (TOY TEXT)
# =============================================================================
uv run train.py \
    --algorithm ppo \
    --env Taxi-v3 \
    --total-timesteps 200000 \
    --n-steps 256 \
    --n-envs 4 \
    --n-epochs 4 \
    --batch-size 64 \
    --lr 1e-3 \
    --gamma 0.99 \
    --gae-lambda 0.95 \
    --clip-epsilon 0.2 \
    --value-coef 0.5 \
    --entropy-coef 0.01 \
    --log-interval 10 \
    --seed 42 \
    --device cpu \
    --save-dir checkpoints/ppo_taxi \
    --record-video \
    --video-dir videos/ppo_taxi \
    2>&1 | tee $LOG_DIR/ppo_taxi.log

# =============================================================================
# FROZENLAKE-V1 (TOY TEXT)
# =============================================================================
uv run train.py \
    --algorithm ppo \
    --env FrozenLake-v1 \
    --total-timesteps 200000 \
    --n-steps 256 \
    --n-envs 4 \
    --n-epochs 4 \
    --batch-size 64 \
    --lr 1e-3 \
    --gamma 0.99 \
    --gae-lambda 0.95 \
    --clip-epsilon 0.2 \
    --value-coef 0.5 \
    --entropy-coef 0.01 \
    --log-interval 10 \
    --seed 42 \
    --device cpu \
    --record-video \
    --video-dir videos/ppo_frozenlake \
    --save-dir checkpoints/ppo_frozenlake \
    2>&1 | tee $LOG_DIR/ppo_frozenlake.log

# =============================================================================
# CLIFFWALKING-V1 (TOY TEXT)
# =============================================================================
uv run train.py \
    --algorithm ppo \
    --env CliffWalking-v1 \
    --total-timesteps 200000 \
    --n-steps 256 \
    --n-envs 4 \
    --n-epochs 4 \
    --batch-size 64 \
    --lr 1e-3 \
    --gamma 0.99 \
    --gae-lambda 0.95 \
    --clip-epsilon 0.2 \
    --value-coef 0.5 \
    --entropy-coef 0.01 \
    --log-interval 10 \
    --seed 42 \
    --device cpu \
    --save-dir checkpoints/ppo_cliffwalking \
    --record-video \
    --video-dir videos/ppo_cliffwalking \
    2>&1 | tee $LOG_DIR/ppo_cliffwalking.log


# # =============================================================================
# # ATARI - Pong
# # =============================================================================
# uv run train.py \
#     --algorithm ppo \
#     --env ALE/Pong-v5 \
#     --total-timesteps 10000000 \
#     --n-steps 128 \
#     --n-envs 8 \
#     --n-epochs 4 \
#     --batch-size 256 \
#     --lr 2.5e-4 \
#     --gamma 0.99 \
#     --gae-lambda 0.95 \
#     --clip-epsilon 0.1 \
#     --value-coef 0.5 \
#     --entropy-coef 0.01 \
#     --log-interval 20 \
#     --seed 42 \
#     --device cpu \
#     --save-dir checkpoints/ppo_pong \
#     --record-video \
#     --video-dir videos \
#     2>&1 | tee $LOG_DIR/ppo_pong.log

# # =============================================================================
# # ATARI - BREAKOUT
# # =============================================================================
# uv run train.py \
#     --algorithm dqn \
#     --env ALE/Breakout-v5 \
#     --episodes 5000 \
#     --batch-size 32 \
#     --buffer-size 100000 \
#     --lr 2.5e-4 \
#     --gamma 0.99 \
#     --eps-start 1.0 \
#     --eps-end 0.01 \
#     --eps-decay 10000 \
#     --tau 0.001 \
#     --log-interval 50 \
#     --seed 42 \
#     --device cpu \
#     --save-dir checkpoints/dqn_breakout \
#     2>&1 | tee $LOG_DIR/dqn_breakout.log

# uv run train.py \
#     --algorithm ppo \
#     --env ALE/Breakout-v5 \
#     --total-timesteps 10000000 \
#     --n-steps 128 \
#     --n-envs 8 \
#     --n-epochs 4 \
#     --batch-size 256 \
#     --lr 2.5e-4 \
#     --gamma 0.99 \
#     --gae-lambda 0.95 \
#     --clip-epsilon 0.1 \
#     --value-coef 0.5 \
#     --entropy-coef 0.01 \
#     --log-interval 20 \
#     --seed 42 \
#     --device cpu \
#     --save-dir checkpoints/ppo_breakout \
#     2>&1 | tee $LOG_DIR/ppo_breakout.log

# # =============================================================================
# # ATARI - SPACE INVADERS
# # =============================================================================
# uv run train.py \
#     --algorithm dqn \
#     --env ALE/SpaceInvaders-v5 \
#     --episodes 5000 \
#     --batch-size 32 \
#     --buffer-size 100000 \
#     --lr 2.5e-4 \
#     --gamma 0.99 \
#     --eps-start 1.0 \
#     --eps-end 0.01 \
#     --eps-decay 10000 \
#     --tau 0.001 \
#     --log-interval 50 \
#     --seed 42 \
#     --device cpu \
#     --save-dir checkpoints/dqn_spaceinvaders \
#     2>&1 | tee $LOG_DIR/dqn_spaceinvaders.log

# uv run train.py \
#     --algorithm ppo \
#     --env ALE/SpaceInvaders-v5 \
#     --total-timesteps 10000000 \
#     --n-steps 128 \
#     --n-envs 8 \
#     --n-epochs 4 \
#     --batch-size 256 \
#     --lr 2.5e-4 \
#     --gamma 0.99 \
#     --gae-lambda 0.95 \
#     --clip-epsilon 0.1 \
#     --value-coef 0.5 \
#     --entropy-coef 0.01 \
#     --log-interval 20 \
#     --seed 42 \
#     --device cpu \
#     --save-dir checkpoints/ppo_spaceinvaders \
#     2>&1 | tee $LOG_DIR/ppo_spaceinvaders.log

echo ""
echo "=========================================="
echo "ALL EXPERIMENTS COMPLETED"
echo "=========================================="
echo "Logs: $LOG_DIR/"
echo "Models: checkpoints/"
