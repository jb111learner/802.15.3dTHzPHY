import multiprocessing

from thz_sim_ui.app import run
# 使用conda activate thz 环境

if __name__ == '__main__':
    multiprocessing.freeze_support()
    run()
