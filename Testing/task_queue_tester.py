
import multiprocessing as mp

from task_queue import TaskQueueManager

from tqdm import tqdm
import random
import time

DELAY=0.5

class DummyProcess(mp.Process):
    def __init__(self):
        super(DummyProcess, self).__init__()


    def set_run_index(self, index, job_id):
        self.index=index
        self.job_id=job_id

    def run(self):
    # def run(self, epoch_depth):
        # self.supress_stdout()

        n_cnt=random.randrange(int(6/DELAY),int(10/DELAY))

        tqdm.set_lock(mp.RLock())

        tqdm_text = "job#" + "{}".format(self.job_id).zfill(3)
        with tqdm(total=n_cnt, desc=tqdm_text, position=self.index) as pbar:
            for i in range(n_cnt):

                delay=random.randrange(70,100)
                time.sleep(delay/100.0)
                pbar.update(1)
            pbar.clear()
            pbar.close()

def main():

    n_jobs=10
    n_worker=12


    jobs=[]
    for i in range(n_jobs):
        jobs.append(DummyProcess())


    task_queue_manager = TaskQueueManager(n_workers=n_worker,
                                        jobs=jobs)


    print("task queue started")

    task_queue_manager.start_and_wait()

    print("Program done")




if __name__=='__main__':
    main()
