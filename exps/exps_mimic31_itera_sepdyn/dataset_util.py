#!/usr/bin/env python
# encoding: utf-8


import pandas as pd
import json
from pathlib import Path
from IPython import embed

def get_cols(dyn_mode):

    d_items = pd.read_csv("/cluster/work/scaimed/users/wguo/datasets/mimiciv3.1/3.1/icu/d_items.csv", usecols=["itemid", "param_type"]).drop_duplicates("itemid")
    dyn = pd.read_csv("~/projs/DT_mimic/data/mimiciv3.1/processed_icu/12476282_35564830/dynamic.csv", header=0)
    cols_dict = {}
    keys = ["inputevents", "procedureevents", "outputevents", "chartevents", "datetimeevents", "ingredientevents"]

    if dyn_mode == 0: # default.  different modalities
        for k in keys:
            cols_dict[k] =  [c for c in dyn.columns if k in c]

    if dyn_mode == 1:         # 6 event keys × several param_types  → one group per combination
        dyn_cols = dyn.columns.drop("timestamp")

        # 1) Split each column name into itemid and event_key
        col_info = (
            pd.Series(dyn_cols, name="col")
            .str.extract(r"^(?P<itemid>\d+)_(?P<event_key>[A-Za-z]+)$")
        )
        col_info["col"] = dyn_cols.values
        col_info["itemid"] = col_info["itemid"].astype(int)

        # 2) Merge with d_items to bring in param_type
        col_info = col_info.merge(d_items, on="itemid", how="left")

        # 3) Normalise param_type strings to compact, lowercase forms
        def norm_pt(pt):
            if pd.isna(pt):
                return "unknown"
            pt = pt.lower()
            if pt.startswith("date"):
                return "datetime"
            return pt.replace(" ", "")      # e.g. "numeric", "text", "ingredient"
        col_info["param_type"] = col_info["param_type"].apply(norm_pt)

        # 4) Create a combined group key and build the dictionary
        col_info["group"] = col_info["event_key"] + "_" + col_info["param_type"]
        cols_dict = (
            col_info.groupby("group")["col"]
                    .apply(list)
                    .to_dict()
        )

    if dyn_mode == 2: # 8 different param types
        dyn_cols = dyn.columns.drop("timestamp")
        col_info = (pd.Series(dyn_cols, name="col").str.extract(r"^(?P<itemid>\d+)_(?P<event_key>.+)$"))
        col_info["col"] = dyn_cols.values
        col_info["itemid"] = col_info["itemid"].astype(int)
        col_info = col_info.merge(d_items, on="itemid", how="left")
        cols_dict = (col_info.groupby("param_type")["col"].apply(list).to_dict()) # by param
        #numeric_cols = cols_by_param["Numeric"]
        #text_cols = cols_by_param["Text"]


    return cols_dict





## use: get_dynamic_data()
# with open("inputevents_cols.json", "w", encoding="utf-8") as f:
#    json.dump(inputevents_cols, f, ensure_ascii=False, indent=2)
#with open("inputevents_cols.json", encoding="utf-8") as f:
#    inputevents_cols = json.load(f)
#    dyn_ie = dyn[inputevents_cols]
if __name__ == "__main__":
    cols_dict0 = get_cols(0)
    cols_dict1 = get_cols(1)
    cols_dict2 = get_cols(2)
    embed()


'''
In [7]: cols_dict0.keys() #6
Out[7]: dict_keys(['inputevents', 'procedureevents', 'outputevents', 'chartevents', 'datetimeevents', 'ingredientevents'])

In [8]: cols_dict1.keys() #12
Out[8]: dict_keys(['chartevents_checkbox', 'chartevents_numeric', 'chartevents_numericwithtag', 'chartevents_text', 'datetimeevents_datetime', 'ingredientevents_ingredient', 'inputevents_solution', 'outputevents_datetime', 'outputevents_ingredient', 'outputevents_numeric', 'outputevents_text', 'procedureevents_processes'])

In [9]: cols_dict2.keys() #8
Out[9]: dict_keys(['Checkbox', 'Date and time', 'Ingredient', 'Numeric', 'Numeric with tag', 'Processes', 'Solution', 'Text'])
'''