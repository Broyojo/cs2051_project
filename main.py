import math
import sys

from datasets import load_dataset
from matplotlib import pyplot as plt
from tqdm import tqdm

import wandb
from milligrad import Layer

sys.setrecursionlimit(10000)

wandb.init(project="mnist_model_training", entity="broyojo")


class Model:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.l1 = Layer(input_dim, hidden_dim, activation="relu")
        self.l2 = Layer(hidden_dim, output_dim, activation="linear")

    def __call__(self, x):
        x = self.l1(x)
        x = self.l2(x)
        return x


def softmax(x):
    exp_x = [2.7182818284590452353602874713526624977572**i for i in x]

    sum_exp_x = 0
    for a in exp_x:
        sum_exp_x += a

    return [j / sum_exp_x for j in exp_x]


def cross_entropy_loss(y_true, y_pred):
    assert len(y_true) == len(y_pred)

    s = 0

    for p, q in zip(y_true, y_pred):
        s += p * q.log()

    return -s


def mse(y_true, y_pred):
    assert len(y_true) == len(y_pred)

    s = 0
    for y_t, y_p in zip(y_true, y_pred):
        s += (y_t - y_p) ** 2

    s *= 1 / len(y_true)
    return s


def image_to_list(im):
    pixels = list(im.getdata())
    for i in range(len(pixels)):
        pixels[i] /= 255
    return pixels


def make_one_hot(index):
    vector = [0 for _ in range(10)]
    vector[index] = 1
    return vector


def argmax(x):
    max_index = 0
    max_val = x[0]
    for i, a in enumerate(x[1:]):
        if a.value > max_val.value:
            max_val = a
            max_index = i
    return max_index


def main():
    dataset = load_dataset("mnist").shuffle()
    model = Model(28 * 28, 16, 10)

    batch_size = 4

    iters = 0

    try:
        for example in tqdm(dataset["train"].iter(batch_size=batch_size), total=60000):
            total_loss = 0

            for i in range(batch_size):
                image = image_to_list(example["image"][i])
                label = make_one_hot(example["label"][i])
                output = softmax(model(image))
                loss = cross_entropy_loss(y_true=label, y_pred=output)
                total_loss += loss

            average_loss = total_loss / batch_size

            wandb.log({"Loss": average_loss.value})

            print(iters, average_loss.value)

            total_loss.backward()
            total_loss.step(learning_rate=0.01)
            total_loss.zero_grad()

            iters += 1
    except KeyboardInterrupt:
        pass

    good = 0
    total = 0

    for example in dataset["test"]:
        image = image_to_list(example["image"])
        output = softmax(model(image))
        index = argmax(output)

        if index == example["label"]:
            good += 1
        total += 1

        print(f"Accuracy: {good / total * 100}%")


if __name__ == "__main__":
    main()
