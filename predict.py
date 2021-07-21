# from model import my_model


def main(args):
    # Define your training procedure here

    # script arguments are accessible as follows:
    # img_path = args.img_path
    # ckpt_path = args.checkpoint_path
    pass

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Inference script',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('img_path', type=str, help='path to the image')
    parser.add_argument('checkpoint_path', type=str, help='path to your model checkpoint')
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
