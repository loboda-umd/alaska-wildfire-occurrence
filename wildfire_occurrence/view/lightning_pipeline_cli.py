import sys
import time
import logging
import argparse
from wildfire_occurrence.pipelines.lightning_pipeline \
    import LightningPipeline


# -----------------------------------------------------------------------------
# main
#
# python lightning_pipeline_cli.py -c config.yaml -s all
# -----------------------------------------------------------------------------
def main():

    # Process command-line args.
    desc = 'Use this application to model lightning on production.'
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('-c',
                        '--config-file',
                        type=str,
                        required=True,
                        dest='config_file',
                        help='Path to the configuration file')

    # model to use
    # directory to predict from - regex

    parser.add_argument(
                        '-s',
                        '--pipeline-step',
                        type=str,
                        nargs='*',
                        required=True,
                        dest='pipeline_step',
                        help='Pipeline step to perform',
                        default=[
                            'preprocess', 'train', 'predict',
                            'validate', 'all'],
                        choices=[
                            'preprocess', 'train', 'predict',
                            'validate', 'all'])

    args = parser.parse_args()

    # Setup logging
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s; %(levelname)s; %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # Setup timer to monitor script execution time
    timer = time.time()

    # Initialize pipeline object
    pipeline = LightningPipeline(args.config_file)

    # WRF pipeline steps
    if "preprocess" in args.pipeline_step or "all" in args.pipeline_step:
        pipeline.preprocess()
    if "train" in args.pipeline_step or "all" in args.pipeline_step:
        pipeline.train()
    # if "geogrid" in args.pipeline_step or "all" in args.pipeline_step:
    #    pipeline.geogrid()
    # if "ungrib" in args.pipeline_step or "all" in args.pipeline_step:
    #    pipeline.ungrib()
    # if "metgrid" in args.pipeline_step or "all" in args.pipeline_step:
    #    pipeline.metgrid()
    # if "real" in args.pipeline_step or "all" in args.pipeline_step:
    #    pipeline.real()
    # if "wrf" in args.pipeline_step or "all" in args.pipeline_step:
    #    pipeline.wrf()
    # if "postprocess" in args.pipeline_step or "all" in args.pipeline_step:
    #    pipeline.postprocess()

    logging.info(f'Took {(time.time()-timer)/60.0:.2f} min.')


# -----------------------------------------------------------------------------
# Invoke the main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    sys.exit(main())
