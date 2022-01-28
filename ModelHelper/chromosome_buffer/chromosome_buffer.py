import pandas as pd
import os
import glob
import json, hashlib
import copy



PKG_NAME = "ChromosomeBuffer"


KEY_NAME = "hashkey"
DICT_FIELD_NAME = ['score', 'dict_str',]
CSV_FIELD_NAME = ['score', 'dict_str', ]

class ChromosomeBuffer():
    def __init__(self, csv_path, verbose=False):
        self.verbose = verbose

        self.csv_path = csv_path


        #make csv_file if non existed
        try:
            os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
        except Exception as e:
            raise Exception(e)

        #initialize dictionary if one doesnt exist.
        if os.path.isfile(csv_path):
            self.dict_ = self.load_dict_from_csv_(csv_path)
        else:
            if self.verbose:
                print("["+PKG_NAME+"]::Dictionary file not found, creating new...")
            self.dict_ = {}


    def get_key_from_dict(self, dictionary_in):
        dict_in = copy.deepcopy(dictionary_in)
        hashkey = hashlib.md5(json.dumps(dict_in, sort_keys=True).encode('utf-8')).hexdigest()
        # print("["+PKG_NAME+"]::make hash", hashkey, dict_in)
        return hashkey


    ############## need to be customize depending on DICT_FIELD_NAME list value ##########
    def add_entry(self, dict_in_, score_, override=False):
        hashkey = self.get_key_from_dict(dict_in_)

        #check existience of the entry
        if hashkey in self.dict_:
            if self.verbose:
                print("["+PKG_NAME+"]::Entry Key", hashkey, dict_in_)
                print("["+PKG_NAME+"]::Entry already exist")
            if not override:
                return False
            print("["+PKG_NAME+"]::Entry overriding:", hashkey)

        dict_str = str(dict_in_)


        self.dict_[hashkey] = {
            "score": score_,
            "dict_str": dict_str,
        }
        #write on every add
        self.write_dict_to_csv_(self.csv_path)

        return True

    def match_entries(self, search_dict):
        hashkey = self.get_key_from_dict(search_dict)

        if hashkey in self.dict_:
            if self.verbose:
                print("["+PKG_NAME+"]::Entry Key", hashkey, search_dict)
                print("["+PKG_NAME+"]::Entry already exist")
            return True, self.dict_[hashkey]["score"]
        else:
            return False, 0


    def write_dict_to_csv_(self, csv_path):

        ind_lst = []
        for ind, field in enumerate(CSV_FIELD_NAME):
            ind_lst.append(DICT_FIELD_NAME.index(field))

        df_dict = {key:[] for key in [KEY_NAME, *tuple(CSV_FIELD_NAME)]}
        for key in self.dict_:
            df_dict[KEY_NAME].append(key)
            for entry in CSV_FIELD_NAME:
                if entry in DICT_FIELD_NAME:
                    df_dict[entry].append(self.dict_[key][entry])
        df_data=pd.DataFrame(df_dict)
        df_data.to_csv(csv_path, index=False)


    ############## need to be customize depending on DICT_FIELD_NAME list value ##########
    def load_dict_from_csv_(self, csv_path):
        dict = {}
        df = pd.read_csv(csv_path)
        # print(df)
        for index, row in df.iterrows():
            # print(row)
            hashkey = row['hashkey']
            dict_str = row['dict_str']
            score = row['score']
            # dict_obj = eval(dict_str)
            # json_str = row['json_str']
            # json_str = json.dumps(dict_obj, sort_keys=True).encode('utf-8')
            dict[hashkey] = {
                "score": score,
                "dict_str": dict_str,
            }

        return dict
