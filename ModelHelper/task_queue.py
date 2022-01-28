import time

LOOP_TIME=0.2

class TaskQueueManager():
    def __init__(self, n_workers, jobs):
        self.n_workers=n_workers
        self.jobs=jobs
        self.n_job=len(jobs)

    def start_and_wait(self):

        # running = []
        running_list = {i:None for i in range(self.n_workers)}

        completed=0
        worker_active = 0
        job_counter=0

        while completed<self.n_job:
            #launch jobs if there is free worker
            if worker_active<self.n_workers and not job_counter>=self.n_job:
                # find free index
                for ind in running_list:
                    if running_list[ind] is None:
                        break
                #initialization function
                self.jobs[job_counter].set_run_index(index=ind, job_id=job_counter)
                #start job
                self.jobs[job_counter].start()
                # running.append(job_counter)
                running_list[ind]=job_counter
                worker_active+=1
                job_counter+=1

            #check if any job is done:
            for list_ind in running_list:
                job_id=running_list[list_ind]
                if job_id is None:
                    continue
                #check if alive
                self.jobs[job_id].join(timeout=0)
                if not self.jobs[job_id].is_alive():
                    running_list[list_ind]=None
                    worker_active-=1
                    completed+=1
            ## add loop delay
            time.sleep(LOOP_TIME)
