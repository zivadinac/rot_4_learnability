from argparse import ArgumentParser
import pickle
from os import path
from time import time
import torch
from stimulus import getVideoStimulusLoader
from models import LNP
from receptive_fields import randomRetinalGanglionRFs
import utils

args = ArgumentParser()
args.add_argument("stimulus_path")
args.add_argument("out_path")
args.add_argument("--timesteps", type=int, default=16, help="Stimulus duration in number of frames.")
#args.add_argument("--stimulus_duration", type=int, default=320, help="Stimulus duration in ms.")
args.add_argument("--population_size", type=int, default=128, help="Number of neurons in a population.")
args.add_argument("--batch_size", type=int, default=1)
args = args.parse_args()

utils.saveArgs(path.join(args.out_path, "args.txt"))

vs, vs_props = getVideoStimulusLoader(args.stimulus_path, args.timesteps, batch_size=args.batch_size)
rfs = randomRetinalGanglionRFs(vs_props["spatial_shape"], args.timesteps, args.population_size)
model = LNP(vs_props["spatial_shape"], args.timesteps, args.population_size, rfs)


start = time()
res = [model.forward(s) for s in vs]

data = {"stimulus": path.basename(args.stimulus_path),\
        "stimulus_duration_ms": args.timesteps * (1000 / vs_props["fps"]),\
        "stimulus_duration_timesteps": args.timesteps,\
        "stimulus_spatial_shape": vs_props["spatial_shape"],\
        "population_size": args.population_size,\
        "data": torch.cat(res, 0).numpy()}

with open(path.join(args.out_path, "data.pck"), "wb") as out_file:
    pickle.dump(data, out_file)
end = time()
print(f"Finished in {end-start} seconds. (batch_size = {args.batch_size}).")
