from argparse import ArgumentParser
import pickle
from os import path
from time import time
import torch
from stimulus import getVideoStimulusLoader
from models import LNP
from receptive_fields import randomRetinalGanglionRFs
import utils
import data_utils

def __createDataFileName(args):
    stim = path.basename(args.stimulus_path).split('.')[0]
    return f"{stim}_ps_{args.population_size}"

args = ArgumentParser()
args.add_argument("stimulus_path")
args.add_argument("out_path")
args.add_argument("--stimulus_repeats", type=int, default=10, help="Number of stimulus repeats during simulation.")
args.add_argument("--timesteps", type=int, default=16, help="Stimulus duration in number of frames.")
#args.add_argument("--stimulus_duration", type=int, default=320, help="Stimulus duration in ms.")
args.add_argument("--population_size", type=int, default=128, help="Number of neurons in a population.")
args.add_argument("--batch_size", type=int, default=1)
args.add_argument("--seed", type=int, default=12345)
args.add_argument("--use_gpu", type=int, default=1)
args = args.parse_args()
utils.saveArgs(path.join(args.out_path, __createDataFileName(args) + ".txt"))

torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda:0" if args.use_gpu and torch.cuda.is_available() else "cpu")

vs, vs_props = getVideoStimulusLoader(args.stimulus_path, args.timesteps, batch_size=args.batch_size)
rfs = randomRetinalGanglionRFs(vs_props["spatial_shape"], args.timesteps, args.population_size)
model = LNP(vs_props["spatial_shape"], args.timesteps, args.population_size, rfs)
model.to(device)

start = time()
print(f"Starting simulation for {args.stimulus_path} and population size {args.population_size} on device: {device}.")
res = []

for r in range(args.stimulus_repeats):
    for i,s in enumerate(vs):
        res.append(model(s.to(device)))
        #if i % 1000 == 0:
            #print(f"Finished step {i}.")
    print(f"Finished repeat {r+1}/{args.stimulus_repeats} (population_size: {args.population_size}).")
res = torch.cat(res, 0).T.cpu()

data = {"stimulus": path.basename(args.stimulus_path),\
        "stimulus_duration_ms": args.timesteps * (1000 / vs_props["fps"]),\
        "stimulus_duration_timesteps": args.timesteps,\
        "stimulus_spatial_shape": vs_props["spatial_shape"],\
        "population_size": args.population_size,\
        "seed": args.seed,\
        "stimulus_repeats": args.stimulus_repeats,\
        "data": res.numpy()}

data_utils.saveSimulatedData(path.join(args.out_path, __createDataFileName(args) + ".pck"), data)
end = time()
print(f"Finished in {end-start} seconds. (population_size: {args.population_size}).")
