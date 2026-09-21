#!/usr/bin/env python
# coding: utf-8

import argparse
import os
from hand_detection_interface_final import GUI


def parse_arguments():
    parser = argparse.ArgumentParser(description="Riconoscimento LIS da webcam")
    parser.add_argument(
        "-c", "--Classifier",
        type=str,
        required=False,
        help="Percorso al file del classificatore Keras (.keras o .h5)"
    )
    return parser.parse_args()


def main():
    args = parse_arguments()

    if not os.path.exists(args.Classifier):
        print(f"❌ Modello non trovato: {args.Classifier}")
        return

    print(f"📦 Caricamento classificatore da: {args.Classifier}")
    gui = GUI(args.Classifier)
    gui.main_gui()


if __name__ == "__main__":
    main()
