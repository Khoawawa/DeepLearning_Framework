import argparse
from pathlib import Path
import sys
from types import TracebackType
from engines.interfaces.ifactory import IFactory
from engines.utils import resolve_factory
from utils.common_utils import load_config, resolve_config_path, resolve_output_path
from utils.torch_utils import set_seed, resolve_device
from loguru import logger

def configure_logging(log_dir: str | Path = "logs") -> None:
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger.remove()  # drop the default stderr sink so we control formatting

    # Terminal: short, no traceback noise
    logger.add(
        sys.stderr,
        level="INFO",
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
        backtrace=False,
        diagnose=False,
    )

    # File: full traceback + variable values, one file per run
    logger.add(
        log_dir / "run_{time:YYYY-MM-DD_HH-mm-ss}.log",
        level="DEBUG",
        backtrace=True,
        diagnose=True,
    )

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Training script")
    # Essential
    ## config file must reference both model and dataset
    parser.add_argument("-c", "--config", type=str, required=True, help="Config file name")
    # Optional
    parser.add_argument("-m", "--mode", type=str, default="train", help="Choose between train, test and inference", choices=["train", "test", "inference"])
    parser.add_argument("-d", "--device", type=str,default=None, help="Device to train on")
    parser.add_argument("-e", "--epochs", type=int, default=0, help="Number of epochs")
    parser.add_argument("-o", "--output_dir", type=str, default="/runs", help="Output directories")
    parser.add_argument("-r", "--resume_path", type=str, default=None, help="Path to check point to resume from")
    parser.add_argument("--seed", default=42)
    # if left empty will be using the model_name
    parser.add_argument("--run_name", type=str, default=None, help="Name of run, use for output subfolder/logging")
    
    return parser.parse_args()

def run(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = resolve_device(args.device)
    
    cfg_path = resolve_config_path(args.config)
    cfg = load_config(cfg_path)
    
    run_name:str = args.run_name if args.run_name is not None else cfg["model"]["name"]
    output_path = resolve_output_path(
        output_dir= args.output_dir,
        run_name=run_name
    )
    
    factory : IFactory = resolve_factory(cfg.get("trainer_name", ""))

    # if args.mode == "train":
    #     num_epochs = args.epochs if args.epochs > 0 else cfg.get("epochs", 0)
    #     if num_epochs <= 0:
    #         logger.error("Please define the number of epochs in the config file or as a command line argument")
    #         raise ValueError("Please define the number of epochs in the config file or as a command line argument")
    #     dataloader = factory.build_dataloader(cfg["data"], is_train= True)
    #     trainer: ITrainer = factory.build_trainer(cfg["model"]).to(device)
    #     trainer.fit(dataloader, num_epochs=num_epochs, output_path=output_path, resume_path=args.resume_path)
    # elif args.mode == "test":
    #     dataloader = factory.build_dataloader(cfg["data"], is_train = False)
    #     tester = factory.build_tester(cfg["model"]).to(device)
    #     tester.test(dataloader)
    # elif args.mode == "inference":
    #     inferencer = factory.build_inferencer(cfg["model"]).to(device)
    #     # inferencer.infer()
    # else:
    #     logger.error(f"Unknown mode {args.mode}")
    #     raise ValueError(f"Unknown mode {args.mode}")

def main() -> int:
    configure_logging()
    args = parse_args()
    try:
        run(args)
    except Exception as e:
                # Terse line to terminal via our INFO-level sink
        tb = e.__traceback__
        last_frame = tb 
        while last_frame is not None and last_frame.tb_next is not None:
            last_frame = last_frame.tb_next
        
        if last_frame is not None:
            filename = last_frame.tb_frame.f_code.co_filename
            lineno = last_frame.tb_lineno
            logger.error(f"{type(e).__name__}: {e}  ({filename}:{lineno})")
        else:
            logger.error(f"{type(e).__name__}: {e}  (No traceback available)")

        logger.opt(exception=True).debug("Full traceback")
        return 1
    return 0 
        
if __name__ == "__main__":
    raise SystemExit(main())
    