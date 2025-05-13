import argparse
import wandb
from wandb.integration.keras import WandbCallback
from keras.datasets import fashion_mnist, mnist
from sklearn.model_selection import train_test_split
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Train a feedforward neural network on MNIST or Fashion-MNIST")
    parser.add_argument('-wp', '--wandb_project', type=str, default='Deep_Learning_Assignment_1',
                        help='Weights & Biases project name')
    parser.add_argument('-we', '--wandb_entity', type=str, default='cs24m023-indian-institute-of-technology-madras',
                        help='Weights & Biases entity/team name')
    parser.add_argument('-d', '--dataset', type=str, default='fashion_mnist',
                        choices=["mnist", "fashion_mnist"],
                        help='Dataset to use')
    parser.add_argument('-e', '--epochs', type=int, default=10, choices=[5, 10],
                        help='Number of training epochs')
    parser.add_argument('-b', '--batch_size', type=int, default=64, choices=[16, 32, 64],
                        help='Batch size for training')
    parser.add_argument('-l', '--loss', type=str, default='cross_entropy',
                        choices=["mean_square", "cross_entropy"],
                        help='Loss function')
    parser.add_argument('-o', '--optimizer', type=str, default='nadam',
                        choices=["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"],
                        help='Optimizer to use')
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3,
                        choices=[1e-3, 1e-4],
                        help='Learning rate')
    parser.add_argument('-m', '--momentum', type=float, default=0.5,
                        help='Momentum for applicable optimizers')
    parser.add_argument('-beta', '--beta', type=float, default=0.5,
                        help='Beta for RMSProp')
    parser.add_argument('-beta1', '--beta1', type=float, default=0.5,
                        help='Beta1 for Adam/Nadam')
    parser.add_argument('-beta2', '--beta2', type=float, default=0.5,
                        help='Beta2 for Adam/Nadam')
    parser.add_argument('-eps', '--epsilon', type=float, default=1e-6,
                        help='Epsilon for optimizers')
    parser.add_argument('-w_d', '--weight_decay', type=float, default=0.0,
                        choices=[0, 0.0005, 0.5],
                        help='L2 weight decay')
    parser.add_argument('-w_i', '--weight_init', type=str, default='random',
                        choices=["random", "xavier"],
                        help='Weight initialization')
    parser.add_argument('-nhl', '--num_layers', type=int, default=4, choices=[3, 4, 5],
                        help='Number of hidden layers')
    parser.add_argument('-sz', '--hidden_size', type=int, default=128,
                        choices=[32, 64, 128],
                        help='Hidden layer size')
    parser.add_argument('-a', '--activation', type=str, default='relu',
                        choices=['sigmoid', 'tanh', 'relu'],
                        help='Activation function')
    parser.add_argument('-oa', '--output_activation', type=str, default='softmax',
                        choices=["softmax"],
                        help='Output activation')
    return parser.parse_args()


# --- Model Implementation (SingleLayer & FeedForwardNN) ---
# (Include the classes SingleLayer and FeedForwardNN here, identical to notebook definitions)

# For brevity, assume the definitions of SingleLayer and FeedForwardNN are defined above


def load_data(choice):
    if choice == 'fashion_mnist':
        return fashion_mnist.load_data()
    else:
        return mnist.load_data()


def main():
    args = parse_args()

    # Initialize wandb
    wandb.login(key='YOUR_WANDB_API_KEY')  # replace with env var or prompt
    wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=vars(args))

    # Load data
    (x_train, y_train), (x_test, y_test) = load_data(args.dataset)

    # Instantiate model
    layers = [784] + [args.hidden_size] * args.num_layers + [10]
    model = FeedForwardNN(
        layers_size=layers,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        l2_lambda=args.weight_decay,
        loss=args.loss,
        activation=args.activation,
        optimizer=args.optimizer,
        weight_type=args.weight_init
    )

    # Train
    model.fit(x_train, y_train, x_test, y_test, batch_size=args.batch_size)


if __name__ == '__main__':
    main()