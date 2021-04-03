import argparse
from presetml.generation.vae import run_vae
from presetml.generation import preprocessing
from sklearn.model_selection import train_test_split


def main():
    parser = argparse.ArgumentParser()
    help_ = "Load h5 model trained weights"
    parser.add_argument("-w", "--weights", help=help_)
    help_ = "Use mse loss instead of binary cross entropy (default)"
    parser.add_argument("-m", "--mse", help=help_, action="store_true")
    args = parser.parse_args()
    vectors, labels = preprocessing.load_data()

    X_train, X_test, y_train, y_test = train_test_split(vectors, labels, test_size=0.2)
    data = (X_train, y_train), (X_test, y_test)
    run_vae(args, data)


if __name__ == "__main__":
    main()
