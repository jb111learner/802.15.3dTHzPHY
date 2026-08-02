import multiprocessing

from thz_sim_ui.app import run


if __name__ == '__main__':
    multiprocessing.freeze_support()
    run()
