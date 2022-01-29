import scipy.io as scio
import pickle

Trial=1
#model history
POPULATION_HISTORY_PATH="./history/{0:02d}_population_history.csv".format(Trial)
SAVE_NAME="./history/{0:02d}_run_checkpoint.p".format(Trial)


data = pickle.load( open( SAVE_NAME, "rb" ) )


for key in data:
    print(key,":",data[key])
