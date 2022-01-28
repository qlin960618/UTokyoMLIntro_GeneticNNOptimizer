from chromosome_buffer import ChromosomeBuffer




POPULATION_HISTORY_PATH="./history/population_history.csv"


population_buffer = ChromosomeBuffer(POPULATION_HISTORY_PATH, verbose=False)


e1={'1d_s':45, 'opt':'adam', 'layers':[[1,2,4],[45,7,3]]}
e2={'1d_s':3, 'opt':'adam', 'layers':[[1,2,4],[45,7,3]]}
e3={'1d_s':6, 'opt':'3', 'layers':[[1,2,2],[45,7,3]]}
e4={'1d_s':45, 'opt':'4', 'layers':[[1,2,4],[45,7,3]]}

population_buffer.add_entry(e1, 23)
population_buffer.add_entry(e2, 253)
population_buffer.add_entry(e3, 2)
population_buffer.add_entry(e4, 2.3)
population_buffer.add_entry(e1, 2)
