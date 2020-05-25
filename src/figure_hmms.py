from argparse import ArgumentParser
import matplotlib.pyplot as plt
import TreeHMM


args = ArgumentParser()
args.add_argument("data_path", help="Path base with fitted HMM models (everything except {n_modes}.pck).")
args.add_argument("out_path", help="File in which to save resulting figure.")
args.add_argument("--n_modes", nargs='*', type=int, help="Number of modes in trained HMM models.")
args = args.parse_args()


col_num = 4
row_num = int(len(args.n_modes) // col_num) + int(len(args.n_modes) % col_num > 0)
fig, ax = plt.subplots(row_num, col_num, figsize=(25, 13))
fold = 0

for i in range(row_num):
    for j in range(col_num):
        m = i * col_num + j
        if m >= len(args.n_modes):
            break
        m = args.n_modes[m]
        f = args.data_path + f"{m}.pck"

        hmm = TreeHMM.io.loadTrainedHMM(f)
        ax[i,j].set_title(f"n_modes = {m}")
        ax[i,j].plot(hmm[f"train_log_li"][fold], label="train")
        ax[i,j].plot(hmm[f"test_log_li"][fold], label="test")
        ax[i,j].legend()

plt.savefig(args.out_path)

