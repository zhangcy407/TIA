import argparse
from train_eval import train_and_evaluate


def main():
    parser = argparse.ArgumentParser(description="Run model training")
    parser.add_argument('--dataset', type=str, default='./TIAaudio/datasets', help="Path to the datasets folder")
    parser.add_argument('--batch_size', type=int, default=256, help="Batch size for training")
    parser.add_argument('--epochs', type=int, default=300, help="Number of epochs for training")
    parser.add_argument("--model", type=str)

    args = parser.parse_args()

    train_and_evaluate(args.model, args.dataset, args.batch_size, args.epochs)


if __name__ == "__main__":
    main()
