import shutil, importlib.util, pathlib, sys

# check that nnU-net is installed in your env 
spec = importlib.util.find_spec("nnunetv2")
if spec is None:
    print("can't find nnU-net in this env")
    print("   →  please install nnU-net and retry")
    sys.exit(1)

nnunet_path = pathlib.Path(spec.origin).parent
dest = nnunet_path / "training" / "nnUNetTrainer" / "interactive_trainer"
dest.mkdir(exist_ok=True)

shutil.copy("trainer/nnUNetTrainer_interactive.py", dest)
shutil.copy("trainer/interactive_trainer_utils.py", dest)
shutil.copy("trainer/Trainer_newloss.py", dest)

print(f"✅ Trainer installed here -> {dest}")