import argparse
import logging

from src.classifier.data_pipeline import run_pipeline

logger = logging.getLogger(__name__)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

def main():
    video_dir = None

    parser = argparse.ArgumentParser(
        prog=__name__,
        description='Description',
        epilog='Text at the bottom of help',
    )

    parser.add_argument(
        '-t',
        '--train',
        dest='train',
        action='store_true',
        help='When true, trains classification model and saves it as model.pkl',
    )
    parser.add_argument(
        '-d',
        '--data-dir',
        dest='dir',
        type=str,
        help='Enter directory for training data (only applicable when used with -t/--train)',
    )
    parser.add_argument(
        '-e',
        '--evaluation',
        dest='eval',
        action='store_true',
        help='Runs the evaluation after training the model (only applicable when used with -t/--train)'
    )

    args = parser.parse_args()

    if args.train:
        if args.dir:
            video_dir = args.dir
        run_pipeline(video_dir=video_dir, skip_eval=not args.eval)
        

if __name__=='__main__':
    main()