import sys
import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Avvia l'interfaccia di rilevamento mani con classificatore Keras (.h5 o .keras)."
    )
    parser.add_argument(
        "-c", "--Classifier",
        type=str,
        required=True,
        help="Percorso al modello Keras (.h5 o .keras)"
    )

    args = parser.parse_args((' '.join(sys.argv[1:])).split())

    if args.Classifier:
        process(args.Classifier)


def process(classifier: str):
    from hand_detection_interface import GUI
    g = GUI(classifier)
    g.main_gui()


if __name__ == '__main__':
    main()
