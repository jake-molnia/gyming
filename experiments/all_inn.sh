#!/bin/bash

set -e

PLOT_DIR="plots/ppo_inn_comparison"
LOG_DIR="logs/ppo_inn"
mkdir -p $PLOT_DIR $LOG_DIR

echo "=========================================="
echo "Running INN across all environments"
echo "=========================================="

# CartPole-v1
uv run train.py --algorithm ppo --env CartPole-v1 \
    --total-timesteps 100000 --n-steps 128 --n-envs 4 --n-epochs 4 --batch-size 64 \
    --lr 2.5e-4 --gamma 0.99 --gae-lambda 0.95 --clip-epsilon 0.2 \
    --value-coef 0.5 --entropy-coef 0.01 --log-interval 10 --seed 42 --device cpu \
    --save-dir checkpoints/ppo_inn_cartpole --plot --use-inn \
    --inn-config '{"hidden_dim": 64, "sparsity": 0.3, "iterations": 1, "init_scale": 0.1, "activation": "relu"}' \
    2>&1 | tee $LOG_DIR/cartpole.log && mv checkpoints/ppo_inn_cartpole/*.png $PLOT_DIR/

# Acrobot-v1
uv run train.py --algorithm ppo --env Acrobot-v1 \
    --total-timesteps 500000 --n-steps 512 --n-envs 4 --n-epochs 4 --batch-size 64 \
    --lr 2.5e-4 --gamma 0.99 --gae-lambda 0.95 --clip-epsilon 0.2 \
    --value-coef 0.5 --entropy-coef 0.01 --log-interval 10 --seed 42 --device cpu \
    --save-dir checkpoints/ppo_inn_acrobot --plot --use-inn \
    --inn-config '{"hidden_dim": 128, "sparsity": 0.5, "iterations": 2, "init_scale": 0.1, "activation": "tanh"}' \
    2>&1 | tee $LOG_DIR/acrobot.log && mv checkpoints/ppo_inn_acrobot/*.png $PLOT_DIR/

# MountainCar-v0
uv run train.py --algorithm ppo --env MountainCar-v0 \
    --total-timesteps 500000 --n-steps 512 --n-envs 4 --n-epochs 4 --batch-size 64 \
    --lr 2.5e-4 --gamma 0.99 --gae-lambda 0.95 --clip-epsilon 0.2 \
    --value-coef 0.5 --entropy-coef 0.01 --log-interval 10 --seed 42 --device cpu \
    --save-dir checkpoints/ppo_inn_mountaincar --plot --use-inn \
    --inn-config '{"hidden_dim": 128, "sparsity": 0.7, "iterations": 3, "init_scale": 0.01, "activation": "relu"}' \
    2>&1 | tee $LOG_DIR/mountaincar.log && mv checkpoints/ppo_inn_mountaincar/*.png $PLOT_DIR/

# LunarLander-v3
uv run train.py --algorithm ppo --env LunarLander-v3 \
    --total-timesteps 1000000 --n-steps 2048 --n-envs 1 --n-epochs 10 --batch-size 64 \
    --lr 3e-4 --gamma 0.99 --gae-lambda 0.95 --clip-epsilon 0.2 \
    --value-coef 0.5 --entropy-coef 0.01 --log-interval 10 --seed 42 --device cpu \
    --save-dir checkpoints/ppo_inn_lunarlander --plot --use-inn \
    --inn-config '{"hidden_dim": 256, "sparsity": 0.5, "iterations": 2, "init_scale": 0.1, "activation": "relu"}' \
    2>&1 | tee $LOG_DIR/lunarlander.log && mv checkpoints/ppo_inn_lunarlander/*.png $PLOT_DIR/

# Taxi-v3
uv run train.py --algorithm ppo --env Taxi-v3 \
    --total-timesteps 200000 --n-steps 256 --n-envs 4 --n-epochs 4 --batch-size 64 \
    --lr 1e-3 --gamma 0.99 --gae-lambda 0.95 --clip-epsilon 0.2 \
    --value-coef 0.5 --entropy-coef 0.01 --log-interval 10 --seed 42 --device cpu \
    --save-dir checkpoints/ppo_inn_taxi --plot --use-inn \
    --inn-config '{"hidden_dim": 128, "sparsity": 0.4, "iterations": 2, "init_scale": 0.5, "activation": "relu"}' \
    2>&1 | tee $LOG_DIR/taxi.log && mv checkpoints/ppo_inn_taxi/*.png $PLOT_DIR/

# FrozenLake-v1
uv run train.py --algorithm ppo --env FrozenLake-v1 \
    --total-timesteps 200000 --n-steps 256 --n-envs 4 --n-epochs 4 --batch-size 64 \
    --lr 1e-3 --gamma 0.99 --gae-lambda 0.95 --clip-epsilon 0.2 \
    --value-coef 0.5 --entropy-coef 0.01 --log-interval 10 --seed 42 --device cpu \
    --save-dir checkpoints/ppo_inn_frozenlake --plot --use-inn \
    --inn-config '{"hidden_dim": 64, "sparsity": 0.6, "iterations": 3, "init_scale": 0.1, "activation": "tanh"}' \
    2>&1 | tee $LOG_DIR/frozenlake.log && mv checkpoints/ppo_inn_frozenlake/*.png $PLOT_DIR/

# CliffWalking-v1
uv run train.py --algorithm ppo --env CliffWalking-v1 \
    --total-timesteps 200000 --n-steps 256 --n-envs 4 --n-epochs 4 --batch-size 64 \
    --lr 1e-3 --gamma 0.99 --gae-lambda 0.95 --clip-epsilon 0.2 \
    --value-coef 0.5 --entropy-coef 0.01 --log-interval 10 --seed 42 --device cpu \
    --save-dir checkpoints/ppo_inn_cliffwalking --plot --use-inn \
    --inn-config '{"hidden_dim": 64, "sparsity": 0.3, "iterations": 1, "init_scale": 0.1, "activation": "relu"}' \
    2>&1 | tee $LOG_DIR/cliffwalking.log && mv checkpoints/ppo_inn_cliffwalking/*.png $PLOT_DIR/

echo ""
echo "=========================================="
echo "PPO+INN EXPERIMENTS COMPLETED"
echo "=========================================="
echo "Plots: $PLOT_DIR/"
echo "Logs: $LOG_DIR/"
echo "Models: checkpoints/ppo_inn_*/"
