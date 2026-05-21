#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.append("/home/azureuser/cloudfiles/code/Users/karan.paresh/gpt4hana-us-internal")
sys.path.append("/home/azureuser/cloudfiles/code/Users/karan.paresh/gpt4hana-us-internal/Azure-ML-pipeline/cross_components/")
sys.path.append("/home/azureuser/cloudfiles/code/Users/karan.paresh/gpt4hana-base/gpt4hana_base/datasets/dataset_definitions/cdh_soac_az/cdh_soac_az.py")

import os
import sys
import math
import numpy as np
import time
import re
from datetime import datetime
from pathlib import Path
from collections import defaultdict
import torch
from sentence_transformers import SentenceTransformer
from torch import Tensor
from torch_frame import stype
import torch_geometric
import joblib
import sklearn.metrics as skm

import inspect

import argparse
import logging
import pathlib
import fsspec
import mlflow
from mlflow import log_metric, log_param, log_artifact
from mlflow.entities import Metric
import csv
import pandas as pd 
import pickle
import seaborn as sns
import random
from typing import List, Optional
from relbench.base import Table, Dataset, Database
from relbench.modeling.utils import get_stype_proposal
from torch_frame.config.text_embedder import TextEmbedderConfig
from relbench.base import EntityTask, TaskType
from relbench.metrics import accuracy, average_precision, roc_auc, mae, mse, r2, rmse, macro_f1
from torch_frame.config.text_embedder import TextEmbedderConfig
from typing import List, Optional
import torch
from sentence_transformers import SentenceTransformer
from torch import Tensor
from torch_frame import stype
import torch
import torch_geometric

# added by jialin
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
import json
import ast
import dateutil
from typing import Mapping



import gpt4hana_base
from datasets import load_dataset
from gpt4hana_base.datasets import catalog
#from gpt4hana_base.datasets.utils.azure_ml import data_area_azml, get_job_io
from gpt4hana_base.constants import PARQUET_SUBFOLDER
from gpt4hana_base.datasets import load_metadata
import gpt4hana_base

# Tenant ids for train, validation, test
from data_pipeline.use_case_tenant_metadata import SOAC
# Meta data for pkey, fkey, time_col for SOAC

# graph construction
from graph_construction.rel_bench_style import make_pkey_fkey_graph_new


#


#
import torch_geometric
import torch_frame
import itertools
from relbench.modeling.nn import HeteroEncoder, HeteroGraphSAGE, HeteroTemporalEncoder

from torch.nn import BCEWithLogitsLoss
import copy
from typing import Any, Dict, List

from torch import Tensor
from torch.nn import Embedding, ModuleDict
from torch_frame.data.stats import StatType
from torch_geometric.data import HeteroData
from torch_geometric.nn import MLP
from torch_geometric.typing import NodeType

from sklearn.metrics import classification_report

from torch.nn import CrossEntropyLoss
from torch.nn import BCEWithLogitsLoss
from torch.nn import L1Loss

import torch.nn.functional as F

from relbench.base import EntityTask, TaskType
from relbench.metrics import accuracy, average_precision, roc_auc, mae, mse, r2, rmse, macro_f1

from collections import OrderedDict


import gpt4hana_base
import gpt4hana
import pandas as pd
import numpy as np


import pickle
import json
from gpt4hana_base.utils.file_utils import resolve_url


from typing import Union

import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, Union

import pandas as pd
from typing_extensions import Self


from copy import deepcopy
from typing import Dict, List



import inspect
from typing import Mapping, Dict
import pandas as pd   # only for type hints – not strictly required


from torch_frame.utils.infer_stype import infer_series_stype


from torch_frame import stype



import gc


#from gpt4hana_base.datasets.dataset_definitions.cdh_soac_az.cdh_soac_az import config


from relbench.base import EntityTask, TaskType
from relbench.metrics import accuracy, average_precision, roc_auc, mae, mse, r2, rmse, macro_f1



import tempfile
from pathlib import Path
from gpt4hana_base.utils.general import RemotePathAndFilesytem
from gpt4hana_base.utils.databricks import DataArea
import pickle



import pickle
import json
from gpt4hana_base.utils.file_utils import resolve_url


import re                                   # → Regular expressions
                        
from collections import defaultdict               # ← enable postponed evaluation of type hints (PEP 563 / 649)

import pandas as pd                              # ← import the Pandas library for DataFrame manipulation
import re, builtins                              # ← re = regular-expressions, builtins gives access to the global namespace
from copy import deepcopy                        # ← deepcopy lets us copy nested dicts without shared references
from typing import Dict, Tuple, Any 

import argparse
import subprocess

from pandas.api import types as pdt


# Try to import a library version if you have one
try:
    from relbench.metrics import macro_precision
except Exception:
    # Fallback: build it from sklearn
    from sklearn.metrics import precision_score
    def macro_precision(true, pred) -> float:
        assert pred.ndim > 1
        label = pred.argmax(axis=1)
        return skm.precision_score(true, label, average="macro", zero_division = 1)



from pathlib import Path
from typing import Callable, Dict, List


import argparse
from omegaconf import OmegaConf

import cross_component_az
from cross_component_az.parquet_utils import load_parquet_chunks


import os                               # filesystem ops: listdir, path join, isdir
import re                               # for sanitizing variable names
from pathlib import Path                # (not strictly needed; kept for parity)
from typing import Dict, List, Tuple, Callable, Union

import pandas as pd                     # to convert pyarrow.Table -> pandas.DataFrame
import pyarrow as pa                    # load_parquet_chunks returns pa.Table
from gpt4hana_base.utils.file_utils import resolve_url

import math
import re
from typing import Iterable, List, Any, Optional


# In[2]:


#!pip freeze > requirements.txt


# In[3]:


#helper functions to get null counts in the dataframe


def get_null_counts(df, sort=True, only_missing=True):
    """
    Returns a DataFrame with columns and their null (NaN) counts.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        sort (bool): Whether to sort the result by null count in descending order.
        only_missing (bool): Whether to return only columns that have null values.

    Returns:
        pd.DataFrame: A DataFrame with columns and their null counts.
    """
    null_counts = df.isna().sum().reset_index()
    null_counts.columns = ['column', 'num_nulls']
    
    if only_missing:
        null_counts = null_counts[null_counts['num_nulls'] > 0]
    
    if sort:
        null_counts = null_counts.sort_values(by='num_nulls', ascending=False).reset_index(drop=True)

    return null_counts


# **Auto read all the data**

# In[4]:


#all helper functions to load the data



def load_pickle_input(path):
    with open(resolve_url(path), 'rb') as f:
        input_var = pickle.load(f)
    return input_var

def load_json_input(path):
    with open(resolve_url(path), 'r') as f:
        input_var = json.load(f)
    return input_var



def load_parquet_input(path: str, **kwargs) -> pd.DataFrame:
    """
    Load a Parquet file into a pandas DataFrame from the given URL or file path.
    
    Parameters
    ----------
    path : str
        URL or filesystem path to the .parquet file.
    **kwargs : dict
        Additional keyword args passed to pandas.read_parquet().
    
    Returns
    -------
    pd.DataFrame
    """
    resolved = resolve_url(path)
    return pd.read_parquet(resolved, **kwargs)

def load_csv_input(path: str, **kwargs) -> pd.DataFrame:
    """
    Load a CSV file into a pandas DataFrame from the given URL or file path.
    
    Parameters
    ----------
    path : str
        URL or filesystem path to the .csv file.
    **kwargs : dict
        Additional keyword args passed to pandas.read_csv().
    
    Returns
    -------
    pd.DataFrame
    """
    resolved = resolve_url(path)
    return pd.read_csv(resolved, **kwargs)


# In[5]:


parser = argparse.ArgumentParser(description="Process")
parser.add_argument("--tenant_id", type=str, required=True, help="Tenant ID value")
parser.add_argument("--use_case", type=str, required=True, help="use-case id")
args = parser.parse_args()





# In[6]:


#GMUL data area paths for data_init

use_case = args.use_case
tenant_id= args.tenant_id
data_area="gssl"


# This path is only valid for the pre-computed "frozen" outputs. We will not access the real data-init outputs
# using a hardcoded path like this in the future
DATA_INIT_OUTPUT_BASEDIR = f'az://{data_area}/lgm_components/data_initialization_NEW/cache_single_target/{use_case}/{tenant_id}'

RAW_DATA_PATH = f'{DATA_INIT_OUTPUT_BASEDIR}/raw_data.json'
TENANT_METADATA_PATH = f'{DATA_INIT_OUTPUT_BASEDIR}/tenant_metadata.json'
USECASE_METADATA_PATH = f'{DATA_INIT_OUTPUT_BASEDIR}/usecase_metadata.json'


#"az://gmul/lgm_components/frozen_data_DO_NOT_DELETE/filtered/SaDP-PDELIVPROCGPTD/740343471/ISDSALESDOCITEM/740343471_ISDSALESDOCITEM_20250415-184814_0_1_10000.parquet"


# In[7]:


#read all the data

#raw_data = load_json_input(RAW_DATA_PATH)
tenant_metadata = load_json_input(TENANT_METADATA_PATH)
usecase_metadata = load_json_input(USECASE_METADATA_PATH)


# In[8]:


tenant_metadata


# In[9]:


[target['column'] for target in usecase_metadata["column_info"]["semantics"]['targets']]


# In[10]:


#DATA_CLEANING_OUTPUT_BASEDIR = f'az://{data_area}/lgm_components/data_cleaning/{use_case}/{tenant_id}'
DATA_CLEANING_OUTPUT_BASEDIR = f'az://{data_area}/lgm_components/data_cleaning_single_target_NEW/{use_case}/{tenant_id}'



# test_path = DATA_CLEANING_OUTPUT_BASEDIR
# da_test = DataArea.get_data_area(f"{data_area}_test")  # where we can read/write files
# fs_gssl = RemotePathAndFilesytem.from_data_area(
#     path=test_path,
#     data_area=data_area,
#     instance_type=da_test.instance
#)


# In[11]:


print(sorted(DataArea.data_areas.keys()))


# In[12]:


# os.listdir( os.path.join(fs_gssl.resolved_path))


# In[13]:


os.listdir(resolve_url(f'az://{data_area}/lgm_components/data_cleaning_single_target_NEW/{use_case}/{tenant_id}'))
 


# In[14]:


#GSSL automatic parquet reader and converting them to dataframe name and injecting them into the namespace


# Regex that matches any non-word char OR the start of a string if first char is a digit.
# This lets us turn arbitrary names into safe Python identifiers.
_VAR_SAFE = re.compile(r'\W|^(?=\d)')

def _sanitize_var(name: str) -> str:
    """
    Convert an arbitrary table/folder name into a safe Python identifier.
    Examples:
      "ISD.SALES-DOCITEM" -> "ISD_SALES_DOCITEM"
      "123ABC"            -> "_123ABC"  (prefix underscore if leading digit)
    """
    return _VAR_SAFE.sub('_', name)





def auto_load_chunked_parquets(
    tenant_base_dir,               # az:// path up to tenant folder (no table name)
    data_area: str = "gssl",            # which data area to mount (e.g., "gssl")
    inject_globals: bool = True,        # if True: create df_<table> and table_<TABLE>_chunks variables
    to_pandas: bool = True,             # if True: convert pyarrow.Table -> pandas.DataFrame
    ext: str = ".parquet",              # look for files ending with ".parquet" (case-insensitive)
) -> Dict[str, Dict[str, object]]:
    """
    Walk a GSSL-style directory layout:

      az://{data_area}/lgm_components/data_cleaning/{use_case}/{tenant_id}/<TABLE>/*.parquet

    For each <TABLE> folder:
      • list chunk files (*.parquet)
      • combine chunks via your `load_parquet_chunks(...)`
      • (optionally) convert to pandas
      • expose in namespace as:
           df_<table_lower>               -> combined DataFrame (or pa.Table if to_pandas=False)
           table_<TABLE>_chunks           -> list of az:// chunk paths

    Returns a summary dict with chunk paths, dataframes, and discovered table names.
    """

    # Create a mounted filesystem object for the GIVEN cloud path.
    # NOTE: make sure instance_type is taken from your data-area handle, e.g.:
    #   da_test = DataArea.get_data_area(f"{area}_test")
    #   instance_type = da_test.instance
    # fs = RemotePathAndFilesytem.from_data_area(
    #     path=tenant_base_dir,           # e.g. 'az://gssl/.../{use_case}/{tenant_id}'
    #     data_area=data_area,            # e.g. 'gssl'
    #     instance_type=DataArea.get_data_area(f"{data_area}_test").instance    # e.g. da_test.instance  <<< IMPORTANT
    # )

    # Translate the az:// path into a local, mounted directory on this machine.
    # Example:
    #   tenant_base_dir = 'az://gssl/.../740257989'
    #   fs.resolved_path = '/tmp/mnt/gssl/.../740257989'
    tenant_local_root = resolve_url(tenant_base_dir)

    # List entries directly under the mounted tenant folder.
    # In GSSL, these entries are TABLE FOLDERS like:
    #   ['ICALENDARDATE','ISDDOCITEMPART','ISDDOCPARTNER', ...]
    entries: List[str] = os.listdir(tenant_local_root)
    #entries: List[str] = os.listdir(resolve_url(f'az://{data_area}/lgm_components/data_cleaning/{use_case}/{tenant_id}'))

    # Keep only directories (each is a table folder).
    # Example: 'ISDSALESDOCITEM' is a directory containing many *.parquet chunks.
    table_dirs: List[str] = sorted(
        name for name in entries
        if os.path.isdir(os.path.join(tenant_local_root, name))
    )

    # Prepare dict that will hold:
    #   - all az:// chunk file paths per table
    #   - the created DataFrames (or pyarrow Tables) keyed by df_<table>
    all_chunk_paths: Dict[str, List[str]] = {}
    dataframes: Dict[str, Union[pd.DataFrame, pa.Table]] = {}

    # Iterate each discovered table folder
    for table_name in table_dirs:
        # Compose the az:// URL to that specific table folder
        # Example: 'az://gssl/.../740257989/ISDSALESDOCITEM'
        table_dir_az = f"{tenant_base_dir}/{table_name}"

        # Compose the LOCAL (mounted) path to that folder, for local listing
        # Example: '/tmp/mnt/gssl/.../740257989/ISDSALESDOCITEM'
        table_dir_local = os.path.join(tenant_local_root, table_name) #same as fs.resolved_path


        # Gather all *.parquet chunk files inside this local folder.
        # Example: ['740257989_ISDSALESDOCITEM_20250918-..._0_0_10000.parquet', ...]
        chunk_files: List[str] = sorted(
            f for f in os.listdir(table_dir_local) if f.lower().endswith(ext.lower())
        )

        # Convert those local basenames into their az:// counterparts.
        # We pass az:// paths to `load_parquet_chunks`, which knows how to resolve/mount them.
        # Example result:
        #   ['az://gssl/.../ISDSALESDOCITEM/740257989_ISDSA..._0_0_10000.parquet', ...]
        chunk_paths_az: List[str] = [f"{table_dir_az}/{fname}" for fname in chunk_files] #assigning the whole folder

        
        # --- NEW FIX: Create the local paths for the PyArrow fallback ---
        # PyArrow needs the actual mounted OS paths, not the cloud URIs
        chunk_paths_local: List[str] = [os.path.join(table_dir_local, fname) for fname in chunk_files]
        
        
        
        
        # Save these in the return dict under the TABLE name key.
        all_chunk_paths[table_name] = chunk_files

        # If a table folder is empty (no chunk files), just skip it.
        if not chunk_paths_az:
            continue

        
        try:
            # Use your helper function to read and concatenate all chunk files.
            # This returns a single pyarrow.Table for the entire table.
            table_pa: pa.Table = load_parquet_chunks(
                chunk_paths_az, #list of chunk paths
                data_area=data_area,         # 'gssl'
                instance_type="test"  
            )
            
            print(f"[LOAD METHOD] Successfully used the 'load_parquet_chunks' function.")
            
        except Exception as e:

            print(f"[LOAD WARNING] 'load_parquet_chunks' failed with error: {e}")

            # --- THE ULTIMATE FIX: Pandas-Level Bypass ---
            # PyArrow is too strict about merging 'null' types with 'date' types.
            # We bypass this by reading chunks individually into Pandas DataFrames,
            # which natively resolves mixed null/date schemas perfectly during concatenation.
            
            # 1. Use a list comprehension to read each local chunk file into a separate Pandas DataFrame.
            dfs = [pd.read_parquet(chunk_path) for chunk_path in chunk_paths_local]
            
            # 2. Concatenate all the individual DataFrames into one massive DataFrame.
            # ignore_index=True ensures the final DataFrame has a clean 0 to N index.
            combined_df = pd.concat(dfs, ignore_index=True)
            
            # 3. Convert the forgiving Pandas DataFrame back into the strict PyArrow Table format 
            # that the rest of your script (and the 'to_pandas' toggle) expects.
            table_pa: pa.Table = pa.Table.from_pandas(combined_df)
            
            print(f"[LOAD METHOD] Successfully used the Pandas bypass methodology (pd.read_parquet + concat).")
            


        # Optionally convert to pandas; many downstream users expect DataFrames like df_<table>.
        df= table_pa.to_pandas() if to_pandas else table_pa

        # Build variable names:
        #   - df_<lowercased_sanitized_table>   e.g., 'df_isdsalesdocitem'
        #   - table_<OriginalTABLE>_chunks      e.g., 'table_ISDSALESDOCITEM_chunks'
        chunks_var = f"table_{_sanitize_var(table_name)}_chunks"
        df_var     = f"df_{_sanitize_var(table_name).lower()}"

        # Put into the dict we return to the caller
        dataframes[df_var] = df

        # Optionally expose them in globals() so you can immediately use df_isdsalesdocitem, etc.
        if inject_globals:
            globals()[chunks_var] = chunk_paths_az     # list of az:// chunk files for this table
            globals()[df_var]     = df           # combined DataFrame (or pyarrow.Table)

    # Return a compact summary object
    return {
        "chunk_paths": all_chunk_paths,   # { "ISDSALESDOCITEM": [az://..., az://...], ... }
        "dataframes": dataframes,         # { "df_isdsalesdocitem": <DataFrame>, ... }
        "tables": table_dirs,             # [ "ICALENDARDATE", "ISDSALESDOCITEM", ... ]
    }









result = auto_load_chunked_parquets(
    tenant_base_dir=DATA_CLEANING_OUTPUT_BASEDIR,
    data_area=data_area,  
    inject_globals=True,
    to_pandas=True
)


# In[15]:


# # --- GMUL AUTOMATED PARQUET LOADER (no hardcoding of table/df vars) ---


# # expects your `load_parquet_input` to already be defined (uses resolve_url inside)
# # from above in your notebook/script:
# # def load_parquet_input(path: str, **kwargs) -> pd.DataFrame: ...

# _VAR_SAFE = re.compile(r'\W|^(?=\d)')  # replace non-word chars & leading digit with "_"




# # _VAR_SAFE = re.compile(r'\W|^(?=\d)')
# # The pattern matches either:

# # \W → any non-word character (anything not [A-Za-z0-9_]), or

# # ^(?=\d) → the start of the string if the very first character is a digit (zero-width match).

# # _sanitize_var(name)
# # Replaces every match of that pattern with an underscore: _VAR_SAFE.sub('_', name).

# # So it:

# # turns spaces, dots, hyphens, etc. into _

# # prefixes an underscore if the name would start with a digit

# # leaves letters/digits/underscores intact

# # Examples:

# # "BKPF" → "BKPF"

# # "VBRK-2024" → "VBRK_2024"

# # "abc.def" → "abc_def"

# # "9lives" → "_9lives"




# def _sanitize_var(name: str) -> str:
#     """Make a safe python identifier from a filename stem."""
#     return _VAR_SAFE.sub('_', name)






# def auto_load_cleaned_parquets(
#     file_system_list_dir_path: str,
#     data_cleaning_base_dir: str,
#     inject_globals: bool = True,
#     loader_fn: Callable[[str], "pd.DataFrame"] = load_parquet_input,
#     ext: str = ".parquet",
# ) -> Dict[str, Dict[str, object]]:
#     """
#     1) Lists all *.parquet files in `list_dir_path`.
#     2) For each file NAME.parquet:
#        - builds a path f"{data_base_dir}/{NAME}.parquet" → assigns to variable: table_NAME  (case preserved)
#        - loads the parquet → assigns to variable: df_{name.lower()} (e.g., df_bkpf)
#     3) Returns dicts with all paths and DataFrames too.

#     Parameters
#     ----------
#     list_dir_path : str
#         Local/mounted path whose contents you list (e.g., fs_gssl.resolved_path).
#     data_base_dir : str
#         The az:// base used to actually read the files (e.g., DATA_CLEANING_OUTPUT_BASEDIR).
#     inject_globals : bool
#         If True, creates variables `table_NAME` and `df_name` in the caller/global scope.
#     loader_fn : callable
#         Function to load a parquet path → DataFrame. Defaults to `load_parquet_input`.
#     ext : str
#         File extension to match (default ".parquet").

#     Returns
#     -------
#     {
#       "table_paths": { "NAME": "az://.../NAME.parquet", ... },
#       "dataframes":  { "df_name": <DataFrame>, ... },
#       "names":       [ "NAME", ... ]  # basenames as found
#     }
#     """
#     # discover parquet files
#     files: List[str] = sorted(
#         f for f in os.listdir(os.path.join(file_system_list_dir_path)) if f.lower().endswith(ext.lower()) 
#     )   # os.listdir(list_dir_path) lists all entries (files & dirs) in the folder, f.lower().endswith(ext.lower()) filters to entries whose names end with the desired extension (case-insensitive), e.g. “.parquet”.
    
    
    
#     names: List[str] = [Path(f).stem for f in files]  # e.g., ["BKPF", "KNB5", "VBRK", ...],  Path(f).stem strips the extension from each filename, leaving just the base name, e.g. "BKPF.parquet" → "BKPF".

#     table_paths: Dict[str, str] = {
#         name: f"{data_cleaning_base_dir}/{name}{ext}" for name in names
#     }

#     dataframes: Dict[str, object] = {}

#     for name in names:
#         # variable names
#         table_var = f"table_{_sanitize_var(name)}"        # keep original case visible; sanitize if needed
#         df_var    = f"df_{_sanitize_var(name).lower()}"   # df_bkpf, df_knb5, ...

#         # build final cloud path & load
#         path = table_paths[name]
#         df = loader_fn(path) #the function is load_parquet_input

#         # stash in dicts we return
#         dataframes[df_var] = df

#         # optionally inject into global scope (so you can use df_bkpf, table_BKPF, etc.)
#         if inject_globals:
#             globals()[table_var] = path
#             globals()[df_var] = df #this line puts all the df_dataframe names in the global scope

#     return {"table_paths": table_paths, "dataframes": dataframes, "names": names}









# result = auto_load_cleaned_parquets(
#     file_system_list_dir_path=fs_gssl.resolved_path,
#     data_cleaning_base_dir=DATA_CLEANING_OUTPUT_BASEDIR,
#     inject_globals=True,   # creates table_BKPF, df_bkpf, etc.
# )



# # After this, you can directly use variables like:
# #   table_BKPF  -> "az://.../BKPF.parquet"
# #   df_bkpf     -> pandas DataFrame loaded from that path
# # And similarly for every other *.parquet discovered in the directory.


# In[16]:


usecase_metadata['fkey_pkey_metadata']


# **This function safely deletes the capitalized keys and updates the foreign key references to point to the correct df_ prefixed parent tables**

# In[17]:


def clean_schema_tables(schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Filters the schema to only keep table keys that start with 'df_'.
    It also proactively updates foreign key mappings to ensure they 
    point to the 'df_' tables instead of the deleted capitalized tables.
    """
    cleaned_schema = {}
    
    # STEP 1: Keep only the keys that start with 'df_'
    for table_key, table_info in schema.items():
        if table_key.startswith('df_'):
            # Use deepcopy to avoid accidentally modifying the original schema
            cleaned_schema[table_key] = copy.deepcopy(table_info)
            
    # STEP 2: Update the foreign key mappings to match the new 'df_' names
    for table_name, table_info in cleaned_schema.items():
        fkey_list = table_info.get('fkey_col_to_pkey_table', [])
        
        for fkey_dict in fkey_list:
            for fk_columns, parent_table in fkey_dict.items():
                
                # If the parent table is a capital key (e.g., 'QMEL'), 
                # convert it to the lowercase 'df_' format (e.g., 'df_qmel')
                if not parent_table.startswith('df_'):
                    fkey_dict[fk_columns] = f"df_{parent_table.lower()}"
                    
    return cleaned_schema


# In[18]:


# Assuming your original dictionary is named `database_schema`
usecase_metadata['fkey_pkey_metadata'] = clean_schema_tables(usecase_metadata['fkey_pkey_metadata'])


usecase_metadata['fkey_pkey_metadata']


# **Assign targets as target columns and target tasks**

# In[19]:


usecase_metadata


# In[20]:


usecase_metadata["column_info"]["semantics"]['targets']


# In[21]:


[target['column'] for target in usecase_metadata["column_info"]["semantics"]['targets']]


# In[22]:


[target['task'] for target in usecase_metadata["column_info"]["semantics"]['targets']]


# In[23]:


def resolve_target_columns_from_dataframes(
    usecase_metadata: Dict[str, Any], 
    namespace: Dict[str, Any] = None
) -> List[str]:
    
    namespace = namespace or globals()
    schema = usecase_metadata.get('fkey_pkey_metadata', {})
    raw_targets = usecase_metadata.get("column_info", {}).get("semantics", {}).get("targets", [])
    
    # 1. Pre-fetch only valid DataFrames to save time and code later
    valid_dfs = {
        name: namespace[name] 
        for name in schema.keys() 
        if isinstance(namespace.get(name), pd.DataFrame)
    }

    resolved_targets = []

    for target in raw_targets:
        orig_name = target.get("original_column")
        fallback_name = target.get("column")
        
        # 2. Check which tables contain the original name (if it exists)
        orig_found_in = [name for name, df in valid_dfs.items() if orig_name and orig_name in df.columns]
        
        if orig_found_in:
            resolved_targets.append(orig_name)
            for tbl in orig_found_in:
                print(f"[TARGET FOUND] '{orig_name}' (Original Name) exists in DataFrame: '{tbl}'.")
            continue  # Move to the next target immediately

        # 3. If original wasn't found, check the fallback name
        fallback_found_in = [name for name, df in valid_dfs.items() if fallback_name and fallback_name in df.columns]
        
        resolved_targets.append(fallback_name) # Append fallback even if missing (matches original logic)
        
        if fallback_found_in:
            for tbl in fallback_found_in:
                print(f"[TARGET FOUND] '{fallback_name}' (Fallback Name) exists in DataFrame: '{tbl}'.")
        else:
            print(f"[WARNING] Target not found under '{orig_name}' OR '{fallback_name}' in any DataFrame.")

    return resolved_targets



target_columns = resolve_target_columns_from_dataframes(
    usecase_metadata=usecase_metadata, 
    namespace=globals()
)
 

target_columns


# In[24]:


target_columns


# In[25]:


target_tasks=[target['task'] for target in usecase_metadata["column_info"]["semantics"]['targets']]

target_tasks


# **This function creates a synthetic time column aligned with row order only for dataframes that originally have a time column, leaves others unchanged, and updates the use case metadata with the new time column name.**

# In[26]:


# def inject_synthetic_time_column(
#     usecase_metadata: dict, 
#     namespace: dict = None, 
#     new_time_col_name: str = "new_time_col",
#     target_columns: list = None  # <-- NEW: Added as a parameter
# ) -> dict:
#     """
#     Scans the provided metadata and checks the global namespace for the corresponding DataFrames.
#     If a DataFrame has a defined `time_col` in the schema (not None) OR contains the primary 
#     target column, and it possesses a `__row_order__` column, this function creates a new time 
#     column with synthetic dates.
    
#     The synthetic dates are generated by finding the maximum `__row_order__` value and assigning 
#     it today's date. All other rows are pushed back 1 unit of time based on their specific 
#     distance from the maximum row order.
    
#     Parameters:
#     - usecase_metadata: The dictionary containing 'fkey_pkey_metadata' (your schema).
#     - namespace: The namespace where the DataFrames reside (defaults to globals()).
#     - new_time_col_name: The name of the new column to inject into the DataFrame.
#     - target_columns: A list of target column names.
    
#     Returns:
#     - The updated usecase_metadata dictionary.
#     """
    
#     # If no namespace is provided, default to the global variables dictionary
#     if namespace is None:
#         namespace = globals()

#     # Create a reference to the specific metadata schema for easier access
#     schema = usecase_metadata.get('fkey_pkey_metadata', {})
    
#     # Get today's date (with time set to 00:00:00) using pandas
#     today = pd.Timestamp.today().normalize()

#     # Iterate over every table defined in the schema
#     for df_name, table_info in schema.items():
        
#         # Look up the actual DataFrame object in the namespace using the key (e.g., 'df_qmel')
#         df = namespace.get(df_name)
        
#         # If the DataFrame isn't found in memory, or isn't a pandas DataFrame, skip to the next table
#         if df is None or not isinstance(df, pd.DataFrame):
#             print(f"[WARN] DataFrame '{df_name}' not found in namespace. Skipping.")
#             continue
            
#         # Extract the declared time column from the metadata
#         existing_time_col = table_info.get('time_col')
        
#         # --- NEW FIX: Safely check if the target column exists in this DataFrame ---
#         # We ensure target_columns is passed, is a list, and has at least one item to avoid IndexErrors.
#         has_target_col = False
#         if target_columns and len(target_columns) > 0:
#             has_target_col = target_columns[0] in df.columns
        
#         # Only proceed if the schema declares a time column OR the target column is in this df
#         if existing_time_col is not None or has_target_col:
            
#             # Check if the mandatory '__row_order__' column exists in the DataFrame
#             if '__row_order__' in df.columns:
                
#                 print(f"[PROCESS] Generating '{new_time_col_name}' for '{df_name}'.")
                
#                 # Delete the original time column before proceeding
#                 if existing_time_col and existing_time_col in df.columns and existing_time_col != new_time_col_name:
#                     df.drop(columns=[existing_time_col], inplace=True)
#                     print(f"  -> Dropped original time column '{existing_time_col}'.")
                
#                 # --- ROBUST DATE ASSIGNMENT ---
#                 # 1. Find the highest __row_order__ value in the entire table
#                 max_order = df['__row_order__'].max()
                
#                 # 2. Calculate the distance from "today" for every single row.
#                 offsets = max_order - df['__row_order__']
                
#                 # 3. Convert those offsets into time deltas 
#                 time_offsets = pd.to_timedelta(offsets, unit='D')
                
#                 # 4. Subtract the offsets from today to get the actual dates
#                 synthetic_dates = today - time_offsets
                
#                 # 5. Assign the newly generated sequence of dates to the DataFrame.
#                 df[new_time_col_name] = synthetic_dates
                
#                 # Finally, update the schema metadata to point to this newly created time column.
#                 table_info['time_col'] = new_time_col_name
                
#             else:
#                 print(f"[SKIP] '{df_name}' requires a time_col but is missing '__row_order__'. Cannot generate dates.")
#         else:
#             print(f"[SKIP] '{df_name}' time_col is None in schema and contains no target column.")

#     return usecase_metadata





# usecase_metadata = inject_synthetic_time_column(
#     usecase_metadata=usecase_metadata, 
#     namespace=globals(), 
#     new_time_col_name="NEW_TIME_COL",
#     target_columns= target_columns # Explicitly passing the list here
# )


# **update the time_col in the usecase_metadata schema so that every table's time_col attribute points to '__row_order__'**

# In[27]:


usecase_metadata['fkey_pkey_metadata']


# In[28]:


def update_time_col_to_row_order(
    usecase_metadata: Dict[str, Any], 
    namespace: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Scans the provided metadata and checks the global namespace for the corresponding DataFrames.
    If a DataFrame contains a '__row_order__' column, this function updates the metadata's
    'time_col' attribute for that table to point to '__row_order__'.
    
    Parameters:
    - usecase_metadata: The dictionary containing 'fkey_pkey_metadata' (your schema).
    - namespace: The namespace where the DataFrames reside (defaults to globals()).
    
    Returns:
    - The updated usecase_metadata dictionary.
    """
    
    # ---------------------------------------------------------
    # 1. Resolve Namespace
    # ---------------------------------------------------------
    # If the user didn't pass a specific namespace (like locals()), we default to 
    # globals() so we can find the DataFrame variables currently loaded in memory.
    if namespace is None:
        namespace = globals()

    # ---------------------------------------------------------
    # 2. Extract Target Schema
    # ---------------------------------------------------------
    # Safely extract the inner dictionary containing the table definitions.
    # We default to an empty dict {} if the key is missing to prevent crashes.
    schema = usecase_metadata.get('fkey_pkey_metadata', {})
    
    # ---------------------------------------------------------
    # 3. Iterate and Update
    # ---------------------------------------------------------
    # Loop through every table name (e.g., 'df_isdcrenhanced') and its associated 
    # metadata properties in the schema.
    for df_name, table_info in schema.items():
        
        # Attempt to fetch the actual Pandas DataFrame object from memory 
        # using the string name (e.g., globals()['df_isdcrenhanced']).
        df_obj = namespace.get(df_name)
        
        # ---------------------------------------------------------
        # 4. Safety Checks
        # ---------------------------------------------------------
        # Check if the fetched object is missing (None) or isn't a valid Pandas DataFrame.
        # If it's invalid, we skip updating this table's metadata and move to the next one.
        if df_obj is None or not isinstance(df_obj, pd.DataFrame):
            print(f"[WARN] DataFrame '{df_name}' not found in namespace or is invalid. Skipping.")
            continue
            
        # Check if the specific column '__row_order__' physically exists in this DataFrame.
        if '__row_order__' in df_obj.columns:
            
            # If the column exists, log the action for visibility.
            print(f"[UPDATE] Setting time_col for '{df_name}' to '__row_order__'.")
            
            # Update the 'time_col' key inside the metadata dictionary for this specific table.
            # This overwrites whatever was there previously (e.g., 'CREATIONDATE' or None).
            table_info['time_col'] = '__row_order__'
            
        else:
            # If the column does not exist in the DataFrame, log it and leave the metadata alone.
            print(f"[SKIP] '{df_name}' does not contain a '__row_order__' column. time_col unchanged.")

    # ---------------------------------------------------------
    # 5. Return Updated Object
    # ---------------------------------------------------------
    # Return the entire modified usecase_metadata dictionary back to the caller.
    return usecase_metadata


# --- Execution Example ---
# Run the function, passing in your metadata and the global namespace.
usecase_metadata = update_time_col_to_row_order(
    usecase_metadata=usecase_metadata, 
    namespace=globals()
)


usecase_metadata['fkey_pkey_metadata']


# In[29]:


usecase_metadata['fkey_pkey_metadata']


# **Data semantic alignment and data ensurity at dataframe level**

# In[30]:


#this creates the semantic dataframes in the namespace (the key of the schema) if the dataframe is represented by its origin_table name.
# df_isdsales -> df_salesdocitem 



def ensure_semantic_df_names(
    schema: Dict[str, Any],
    namespace: Mapping[str, Any] | None = None,
) -> Dict[str, Dict[str, str] | list[str]]:
    """
    Ensure every schema key (semantic DF name like 'df_salesdocument') exists as a
    pandas DataFrame in the provided namespace. If it's missing but the origin-table
    variant exists (e.g., 'df_isdsalesdoc'), create an alias:
        globals()['df_salesdocument'] = globals()['df_isdsalesdoc']

    Parameters
    ----------
    schema : dict
        Your schema dict. Keys are semantic dataframe names (e.g., 'df_salesdocument').
        Each value is a dict that must contain 'origin_table' (e.g., 'ISDSALESDOC').
    namespace : mapping, default None
        Where to look up/set variables. Defaults to globals().

    Returns
    -------
    dict
        {
          "created": { "<semantic_df>": "<origin_df>", ... },  # aliases made
          "missing": [ "<semantic_df>", ... ]                  # still missing
        }
    """

    # Use the provided namespace for variable lookup/injection; fall back to global scope.
    scope = globals() if namespace is None else namespace

    # Keep track of what we created and what is still missing.
    created_aliases: Dict[str, str] = {}
    missing_semantic_names: list[str] = []

    # Go through each semantic dataframe name defined in the schema.
    for semantic_df_name, table_info in schema.items():
        # If the semantic name already exists and is a DataFrame, nothing to do.
        if semantic_df_name in scope and isinstance(scope[semantic_df_name], pd.DataFrame):
            continue

        # Pull the origin table name (e.g., 'ISDSALESDOC'). If missing, we cannot build the fallback.
        origin_table_name = table_info.get("origin_table")
        if not origin_table_name:
            # We can't guess a source; record as missing and move on.
            missing_semantic_names.append(semantic_df_name)
            continue

        # Build the expected origin-table variable name: df_<origin_table.lower()>
        # e.g., origin_table_name == 'ISDSALESDOC' -> origin_df_var == 'df_isdsalesdoc'
        origin_df_var = f"df_{origin_table_name.lower()}"

        # Try to fetch that origin-table DataFrame from the namespace.
        origin_obj = scope.get(origin_df_var)

        # If the origin object exists and is a DataFrame, alias it under the semantic name.
        if isinstance(origin_obj, pd.DataFrame):
            scope[semantic_df_name] = origin_obj  # alias/assign into the namespace
            created_aliases[semantic_df_name] = origin_df_var
        else:
            # Neither the semantic name nor the origin-table variant is available → record missing.
            missing_semantic_names.append(semantic_df_name)

    # Return a summary so the caller knows what happened.
    return {"created": created_aliases, "missing": missing_semantic_names}




result = ensure_semantic_df_names(usecase_metadata['fkey_pkey_metadata'])


# In[31]:


print("Aliases created:", result["created"])
print("Still missing:", result["missing"])


# In[32]:


usecase_metadata


# In[33]:


usecase_metadata["column_info"]["semantics"]['targets']


# **Data semantic alignment and data ensurity at column_metadata level**

# In[34]:


#develop column metdata for kg4hana_type to stype


def convert_kg4hana_type_to_new_format(old_data):
    """
    Expects a structure like:
      {
        "kg_info": {
          "BSEG": KGTableInfo(...),
          "KNA1": KGTableInfo(...),
          ...
        }
      }
    Returns a dict in the new format:
      {
        "BSEG": [
          { "ABPER": "ACCP", "ABSBT": "CURR", ... }
        ],
        "KNA1": [
          { "ANRED": "CHAR", ... }
        ],
        ...
      }
    """

    # Extract the dictionary of tables from old_data.
    # If your old_data has exactly one key "kg_info" that holds all tables,
    # we do:
    kg_info = old_data["kg_info"]

    new_format = {}

    # Loop over each table key and KGTableInfo object
    for table_alias, table_obj in kg_info.items():
        # table_obj is an instance of KGTableInfo,
        # which has an attribute 'table_fields' with a list of KGFieldInfo.

        # Build a dictionary of { field_name: field_data_type }
        field_dict = {}
        for field_info in table_obj['table_fields']:
            # 'field_name' is a string, e.g. "ABPER"
            # 'field_data_type' is a string, e.g. "ACCP", "CHAR", "NUMC", etc.
            field_dict[field_info['field_name']] = field_info['field_data_type']

        # The new format wants a list containing this one dictionary
        new_format[f"df_{table_alias.lower()}"] = [field_dict]

    return new_format


use_case_column_metadata=convert_kg4hana_type_to_new_format(usecase_metadata["column_info"])


use_case_column_metadata


# In[35]:


#to remap the column_metadata key names which are origin_table name to semantically correct table names
#df_KNA1 to df_customer



def remap_column_metadata_keys(
    schema: Dict[str, Dict[str, Any]],
    column_metadata: Dict[str, Any],
    *,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Rename the keys of `column_metadata` from origin-table names to the
    schema's semantic table names (i.e., keys of `schema`).

    Parameters
    ----------
    schema : dict
        Example: {
          "df_salesdocument": {"origin_table": "ISDSALESDOC", ...},
          "df_customer":      {"origin_table": "KNA1", ...},
          ...
        }
        • Keys are the semantic names you want (e.g., "df_salesdocument").
        • Each value must contain "origin_table" with the raw/original name.

    column_metadata : dict
        Example: {
          "df_isdsalesdoc": [{...}],
          "df_kna1":        [{...}],
          ...
        }
        • Keys currently reflect *origin* table names (sometimes prefixed with "df_").
        • Values are arbitrary metadata payloads that should be preserved as-is.

    verbose : bool
        If True, prints a small before/after mapping summary.
        (Useful to confirm what was remapped and what wasn't.)

    Returns
    -------
    Dict[str, Any]
        A new dict where keys have been replaced with the semantic names
        whenever a match could be found. Any keys that couldn't be matched
        are kept unchanged.

    Notes
    -----
    • Matching is done with a normalization that:
        - lowercases,
        - strips non-alphanumeric characters,
        - optionally strips a leading "df_" prefix on the metadata side.
      This makes "ISDSALESDOC", "df_isdsalesdoc", and "isdsalesdoc" all match.
    """

    # ------------------------------
    # Helper: normalize a table name so different spellings still match.
    # Example:
    #   normalize_name("ISDSALESDOC")            -> "isdsalesdoc"
    #   normalize_name("df_isdsalesdoc", True)   -> "isdsalesdoc"
    #   normalize_name("I_ADDR ORG-Name", True)  -> "iaddrorgname"
    # ------------------------------
    def normalize_name(raw_name: str, strip_df_prefix: bool = False) -> str:
        name = (raw_name or "").strip()
        # If requested, drop a leading "df_" (many metadata keys use this)
        #   e.g., "df_kna1" -> "kna1"
        if strip_df_prefix and name.lower().startswith("df_"):
            name = name[3:]
        # Lowercase and remove all non-alphanumeric characters
        #   e.g., "I_ADDRORGNAMEPOSTALADDRESS" -> "iaddrorgnamepostaladdress"
        #         "KNA1" -> "kna1"
        return re.sub(r"[^a-z0-9]", "", name.lower())

    # ------------------------------------------------------------------
    # 1) Build a mapping: normalized origin-table name  -> semantic name
    #    Example row:
    #      "isdsalesdoc" -> "df_salesdocument"
    # ------------------------------------------------------------------
    origin_to_semantic: Dict[str, str] = {}
    for semantic_table_name, schema_entry in schema.items():
        # Grab the origin table name from the schema entry
        # Example: schema_entry["origin_table"] == "ISDSALESDOC"
        origin_table_name = schema_entry.get("origin_table")
        if origin_table_name is None:
            continue  # skip if no origin table is provided

        # Normalize the origin name so it will match different spellings
        normalized_origin = normalize_name(origin_table_name, strip_df_prefix=False)

        # Map normalized origin -> the semantic key we want to use
        origin_to_semantic[normalized_origin] = semantic_table_name

    # ------------------------------------------------------------------
    # 2) Walk through column_metadata and rename keys when a match exists
    #    We normalize the metadata key (usually "df_<origin>") in a way
    #    that strips "df_" so "df_kna1" matches "KNA1".
    # ------------------------------------------------------------------
    remapped: Dict[str, Any] = {}
    report_renamed: Dict[str, str] = {}   # for optional logging
    report_unmatched: Dict[str, str] = {} # for optional logging

    for metadata_key, metadata_payload in column_metadata.items():
        # Normalize the metadata key, stripping "df_" because many inputs use it
        # Example: "df_isdsalesdoc" -> "isdsalesdoc"
        normalized_metadata_key = normalize_name(metadata_key, strip_df_prefix=True)

        # Look up the semantic table name using the normalized origin
        semantic_table_name = origin_to_semantic.get(normalized_metadata_key)

        if semantic_table_name:
            # We found a match — rename the key to the semantic name
            remapped[semantic_table_name] = metadata_payload
            report_renamed[metadata_key] = semantic_table_name
        else:
            # No match — keep the original key to avoid data loss
            remapped[metadata_key] = metadata_payload
            report_unmatched[metadata_key] = normalized_metadata_key

    # ------------------------------------------------------------------
    # 3) Optional: print a small summary to help validate the result
    # ------------------------------------------------------------------
    if verbose:
        if report_renamed:
            print("Renamed column_metadata keys:")
            for old_key, new_key in report_renamed.items():
                print(f"  {old_key}  ->  {new_key}")
        else:
            print("No keys were renamed (no matches found).")

        if report_unmatched:
            print("\nUnmatched metadata keys (kept as-is):")
            for k, norm in report_unmatched.items():
                print(f"  {k}  (normalized: '{norm}')")

    return remapped


# In[36]:


updated_column_metadata = remap_column_metadata_keys(usecase_metadata['fkey_pkey_metadata'], use_case_column_metadata, verbose=True)

use_case_column_metadata=updated_column_metadata


use_case_column_metadata


# In[37]:


#to infer the abap types

def infer_abap_type(values: Iterable[Any]) -> str:
    """
    Heuristically infer an ABAP type for a column from its observed values.

    Notes
    -----
    • Uses "majority" voting with a 0.9 threshold so a few bad rows won't flip the type.
    • Distinguishes NUMC vs INT by looking for leading zeros / string nature / fixed width.
    • Picks INT2/INT4/INT8 based on observed numeric range.
    • Detects common SAP textual codes: CUKY (currency keys), UNIT, LANG.
    • Detects DATS (several formats), TIMS, ACCP (YYYYMM posting period).
    • Falls back to "CHAR" when unsure.

    Returns one of: {"CHAR","NUMC","INT2","INT4","INT8","DEC","DATS","TIMS","ACCP",
                     "CUKY","UNIT","LANG"}
    (If you only want INT, replace INT2/4/8 with INT in the return mapping.)
    """

    # ---------------------------------------------------------
    # HELPER FUNCTIONS
    # ---------------------------------------------------------
    
    def strip_nones(seq: Iterable[Any]) -> List[Any]:
        # Removes physical `None` and floating point `NaN` values from the dataset
        return [x for x in seq if x is not None and (not (isinstance(x, float) and math.isnan(x)))]

    def share(condition_mask: List[bool]) -> float:
        # Calculates the percentage (from 0.0 to 1.0) of True values in a boolean list.
        # This is used to check if a condition meets the 90% (0.90) threshold.
        return (sum(1 for x in condition_mask if x) / len(condition_mask)) if condition_mask else 0.0

    def looks_numeric_str(s: str) -> bool:
        # Checks if a string contains ONLY digits (with an optional +/- sign at the front)
        return bool(re.fullmatch(r"[+-]?\d+", s))

    def looks_floatish(s: str) -> bool:
        # Checks if a string looks like a decimal/float (e.g., "3.14", "-0.5", "1e-4")
        return bool(re.fullmatch(r"[+-]?\d+([.]\d+)?([eE][+-]?\d+)?", s))

    def parse_int_or_none(x: Any) -> Optional[int]:
        # Safely attempts to convert a value to an integer. Returns None if it fails.
        try:
            if isinstance(x, str) and looks_numeric_str(x):
                return int(x)
            if isinstance(x, (int,)):
                return int(x)
        except Exception:
            return None
        return None

    # ---------------------------------------------------------
    # REGEX PATTERNS FOR SAP TYPES
    # ---------------------------------------------------------
    
    # DATS patterns: ISO, compact, dotted, slashed (YYYY-MM-DD / YYYYMMDD / DD.MM.YYYY / YYYY/MM/DD)
    # We added an optional time component group ( \d{2}:\d{2}:\d{2})?$ to catch string datetimes!
    dats_regexes = [
        re.compile(r"\d{4}-\d{2}-\d{2}( \d{2}:\d{2}:\d{2}(\.\d+)?)?$"),  # e.g., 2024-08-30 OR 2024-08-30 00:00:00
        re.compile(r"\d{8}$"),                                           # e.g., 20240830
        re.compile(r"\d{2}[.]\d{2}[.]\d{4}$"),                           # e.g., 30.08.2024
        re.compile(r"\d{4}/\d{2}/\d{2}$"),                               # e.g., 2024/08/30
    ]

    # TIMS patterns (HH:MM:SS or HHMMSS)
    tims_regexes = [
        re.compile(r"\d{2}:\d{2}:\d{2}$"),        # e.g., 13:45:59
        re.compile(r"\d{6}$"),                    # e.g., 134559
    ]

    # ACCP (accounting period): YYYYMM (MM can be 01..16 in SAP)
    accp_regex = re.compile(r"^(?P<y>\d{4})(?P<m>\d{2})$")

    # ---------------------------------------------------------
    # CHECKER FUNCTIONS
    # ---------------------------------------------------------
    # These return True if a given string matches any of the compiled regex patterns above.

    def is_dats(s: str) -> bool:
        return any(rx.fullmatch(s) for rx in dats_regexes)

    def is_tims(s: str) -> bool:
        return any(rx.fullmatch(s) for rx in tims_regexes)

    def is_accp(s: str) -> bool:
        m = accp_regex.fullmatch(s)
        if not m:
            return False
        mm = int(m.group("m"))
        return 1 <= mm <= 16  # SAP accounting months can go up to 16 (includes special periods)

    def is_cuky(s: str) -> bool:
        # Currency code check: Exactly 3 uppercase letters (e.g., USD, EUR, JPY)
        return bool(re.fullmatch(r"[A-Z]{3}", s))

    def is_lang(s: str) -> bool:
        # Language code check: 1 to 2 uppercase letters (e.g., 'E', 'EN', 'DE')
        return bool(re.fullmatch(r"[A-Z]{1,2}", s))

    def is_unit(s: str) -> bool:
        # Unit of Measure check: 1 to 5 uppercase letters (e.g., KG, EA, PC, M2)
        return bool(re.fullmatch(r"[A-Z]{1,5}", s))

    # ---------------------------------------------------------
    # MAIN INFERENCE LOGIC
    # ---------------------------------------------------------
    
    # Clean the raw input by dropping nulls/NaNs
    vals = strip_nones(values)
    
    # If the column is completely empty, default to "CHAR" as a safe fallback
    if not vals:
        return "CHAR"  

    # --- NATIVE DATETIME CHECK ---
    # Fast path: If the column already natively contains pandas Timestamps or Python datetimes,
    # immediately return "DATS" without running slow string conversions or regex loops.
    # We only check the first 10 values for performance.
    if any(isinstance(v, (pd.Timestamp, datetime)) for v in vals[:10]): 
        return "DATS"

    # Convert all clean values to strings for the regex and pattern detectors
    as_str = [str(v) for v in vals]

    # --- Strong pattern candidates (dates/times/periods) ---
    # Check if >= 90% of the column matches date formats
    if share([is_dats(s) for s in as_str]) >= 0.90:
        return "DATS"

    # Check if >= 90% of the column matches time formats
    if share([is_tims(s) for s in as_str]) >= 0.90:
        return "TIMS"

    # Check if >= 90% of the column matches SAP accounting periods
    if share([is_accp(s) for s in as_str]) >= 0.90:
        return "ACCP"

    # --- Numeric vs numeric-text (NUMC) ---
    # Check if >= 90% of the column consists entirely of digits
    is_digit = [s.isdigit() for s in as_str]
    if share(is_digit) >= 0.90:
        
        # Calculate the string length of all the items that are pure digits
        lengths = [len(s) for s, d in zip(as_str, is_digit) if d]
        
        # Determine if any numbers have a leading zero (e.g., "00123")
        has_leading_zero = any(s.startswith("0") and len(s) > 1 for s, d in zip(as_str, is_digit) if d)
        
        # Determine if every single number in the column has the exact same character length
        fixed_width = (max(lengths) == min(lengths)) if lengths else False

        # Special SAP case: If every number is exactly 3 digits long, it's highly likely a Client ID (MANDT)
        if all(l == 3 for l in lengths):
            return "CLNT"  

        # If numbers have leading zeros or are strictly fixed-width, they are identifiers, not math numbers (NUMC)
        if has_leading_zero or fixed_width:
            return "NUMC"

        # If it's pure math integers, figure out how much memory it needs by finding the min/max values
        ints = [parse_int_or_none(v) for v in vals]
        ints = [x for x in ints if x is not None]
        if ints:
            min_v, max_v = min(ints), max(ints)
            if -32768 <= min_v <= max_v <= 32767:
                return "INT2" # Fits in 16-bit integer
            if -2147483648 <= min_v <= max_v <= 2147483647:
                return "INT4" # Fits in 32-bit integer
            return "INT8"     # Huge numbers, needs 64-bit integer
            
        return "INT4" # Default fallback for integers

    # --- Float / Decimal amounts (DEC) ---
    # Determine if >= 90% of the column looks like a float or decimal value
    is_floatish = []
    for v in vals:
        if isinstance(v, (float,)):
            is_floatish.append(True)
        elif isinstance(v, str) and looks_floatish(v):
            is_floatish.append(True)
        else:
            is_floatish.append(False)
            
    if share(is_floatish) >= 0.90:
        return "DEC"   

    # --- Small, code-like uppercase tokens ---
    # Check if >= 90% of the column matches 3-letter currency codes
    if share([is_cuky(s) for s in as_str]) >= 0.90:
        return "CUKY"

    # Check if >= 90% of the column matches standard unit of measure abbreviations
    if share([is_unit(s) for s in as_str]) >= 0.90:
        return "UNIT"

    # Check if >= 90% of the column matches short language codes
    if share([is_lang(s) for s in as_str]) >= 0.90:
        return "LANG"

    # --- Default Fallback ---
    # If it failed the 90% threshold for all the specific checks above, it's just general text.
    return "CHAR"


# In[38]:


#this is to match column_metadata column names exactly with dataframe, it should align, and adds any missing DataFrame columns to the metadata with ABAP type USING THE INFER_ABAP_TYPE function.


def align_use_case_column_metadata_with_dataframe(
    use_case_column_metadata: Dict[str, Any],
    namespace: Optional[Dict[str, Any]] = None,
    default_abap_type: str = "CHAR",
) -> Dict[str, Any]:
    """
    Make the per-table column metadata exactly match the columns that exist
    in the corresponding DataFrame objects in the (given or global) namespace.

    What it does for each df_* key:
    - Keeps only columns that actually exist in the DataFrame (drops extras).
    - Adds any missing DataFrame columns to the metadata:
        • ABAP type is inferred from real values via `infer_abap_type`.
        • Falls back to `default_abap_type` only if inference fails/empty.
    - Preserves the order of columns as they appear in the DataFrame.
    - Returns the metadata in the SAME structure you provided:
        { "df_xxx": [ { "COL_A": "CHAR", ... } ], ... }

    Parameters
    ----------
    use_case_column_metadata : dict
        Mapping of df-name -> [ { column_name: abap_type, ... } ].
    namespace : dict, optional
        A mapping in which to look up DataFrames by name (e.g., globals()).
        If not provided, the function will use globals().
    default_abap_type : str, optional
        ABAP type to assign only if inference cannot determine a type
        (e.g., column is entirely empty). Defaults to "CHAR".

    Returns
    -------
    dict
        A NEW metadata dict in the same shape, reconciled with the DataFrames.

    Example
    -------
    # Suppose df_salesdocument has columns ["A","B"] and metadata has only {"A":"CHAR"}.
    # Then:
    # - "A" keeps "CHAR" (existing type).
    # - "B" is added with type = infer_abap_type(df_salesdocument["B"].dropna().tolist()).
    """

    # Decide where to read DataFrame variables from.
    # Example: if your notebook defines df_customer, it likely lives in globals().
    active_namespace: Dict[str, Any] = namespace if namespace is not None else globals()

    # New dict to accumulate reconciled (fixed) metadata.
    reconciled_metadata: Dict[str, Any] = {}

    # Walk over each "df_*" entry in your metadata bundle.
    for dataframe_name, table_metadata_list in use_case_column_metadata.items():
        # Look up the actual DataFrame by variable name (e.g., "df_customer").
        dataframe_obj = active_namespace.get(dataframe_name, None)

        # If we can't find the DataFrame, keep this metadata entry as-is and continue.
        # This avoids accidental data loss in cases where a DF isn't in scope.
        if dataframe_obj is None:
            reconciled_metadata[dataframe_name] = table_metadata_list
            continue

        # Extract the inner {column_name: abap_type, ...} mapping
        # Your structure is: "df_xxx": [ { ... } ]
        if isinstance(table_metadata_list, list) and table_metadata_list:
            existing_col_to_type: Dict[str, str] = dict(table_metadata_list[0])
        elif isinstance(table_metadata_list, dict):
            existing_col_to_type = dict(table_metadata_list)
        else:
            existing_col_to_type = {}

        # Preserve the DataFrame's column order exactly as shown in the DF.
        dataframe_columns_in_order = list(getattr(dataframe_obj, "columns", []))

        # Build a fresh mapping in that exact order.
        new_col_to_type: Dict[str, str] = {}

        # Iterate columns in DF order so the output metadata mirrors the DF.
        for column_name in dataframe_columns_in_order:
            if column_name in existing_col_to_type:
                # CASE 1: Column already present in metadata → keep its declared type.
                # Example: metadata says "KOSTL": "CHAR" → we keep "CHAR".
                new_col_to_type[column_name] = existing_col_to_type[column_name]
            else:
                # CASE 2: Column is missing in metadata → infer its ABAP type from values.

                # Pull the real values from the DataFrame column and drop NaN/None.
                # Example: df_customer["KOSTL"].dropna().tolist() -> ["001234", "009876", ...]
                try:
                    # If pandas is available, drop nulls first to avoid misleading inference.
                    series = getattr(dataframe_obj, column_name)
                    # Convert to a plain Python list for the inference helper.
                    non_null_values = series.dropna().tolist() if hasattr(series, "dropna") else list(series)
                except Exception:
                    # If anything goes wrong (unexpected object), treat as empty.
                    non_null_values = []

                # Use the provided helper to infer the best-fitting ABAP type
                # from the observed values. Your helper returns one of
                # {"CHAR","NUMC","INT","DEC","DATS"} (per your logic).
                # If the column has no usable values, we fall back to default_abap_type.
                try:
                    inferred_type = infer_abap_type(non_null_values)
                except Exception:
                    inferred_type = default_abap_type  # safety fallback

                # If inference returns a falsey value (e.g., empty), use the default.
                if not inferred_type:
                    inferred_type = default_abap_type

                # Record the inferred (or fallback) ABAP type for this newly added column.
                new_col_to_type[column_name] = inferred_type

        # Write back using the exact same outer shape: a single-element list.
        reconciled_metadata[dataframe_name] = [new_col_to_type]

    return reconciled_metadata







# Build the updated metadata using your live DataFrames in the notebook
new_use_case_column_metadata = align_use_case_column_metadata_with_dataframe(
    use_case_column_metadata,   # your dict shown in the prompt
    namespace=globals()         # look up df_* variables from the current notebook
)

# If you want to replace the original reference:
use_case_column_metadata = new_use_case_column_metadata

# Inspect the result
use_case_column_metadata


# In[39]:


usecase_metadata['fkey_pkey_metadata']


# In[40]:


#add missing df to column metadata and assign ABAP type using the infer_abap_type function


def ensure_metadata_for_schema_dfs(
    schema: Dict[str, Any],
    metadata: Dict[str, Any],
    default_abap_type: str = "CHAR",            # kept for API compatibility; no longer used when inferring
    namespace: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Ensure every dataframe listed in `schema` exists in `metadata`.

    What stays the same as your original:
    - Only dataframe *keys* missing in `metadata` are added.
    - Existing entries in `metadata` are left untouched.
    - Returned dict keeps the same shape: { df_name: [ { col: ABAP_TYPE, ... } ], ... }

    What's changed (per your request):
    - For every *newly added* dataframe key, each column's ABAP type is
      **inferred** from its values with `infer_abap_type(values)` instead of
      using the `default_abap_type`.

    Example
    -------
    Suppose:
        schema = {"df_customer": {...}}   # df_customer is present in schema
        metadata = {}                     # but missing in metadata
        And in your notebook you have a pandas DataFrame named df_customer, e.g.:
            df_customer.columns = ["KUNNR", "LAND1", "UMSAT", "ERDAT"]
            df_customer["KUNNR"] = ["000123", "000124"]         -> NUMC
            df_customer["LAND1"] = ["DE", "US"]                 -> LANG (or CHAR, depending on your heuristic)
            df_customer["UMSAT"] = [12345.67, 890.0]            -> DEC
            df_customer["ERDAT"] = ["2024-08-30", "2024-09-01"] -> DATS

    Then the added metadata entry becomes:
        {"df_customer": [{
            "KUNNR": "NUMC",
            "LAND1": "LANG",
            "UMSAT": "DEC",
            "ERDAT": "DATS",
        }]}

    Parameters
    ----------
    schema : dict
        Dictionary whose **keys** are dataframe names (e.g., 'df_bseg', 'df_bkpf', ...).
        Only the keys are used here; the rest of the schema contents are ignored.
    metadata : dict
        Existing metadata mapping: df_name -> [ { column_name: ABAP_type, ... } ].
        If a df_name is missing, a new entry will be added by inferring types.
    default_abap_type : str, default "CHAR"
        Kept for backward compatibility; not used for newly added columns,
        since we infer types with `infer_abap_type`. (Left in the signature on purpose.)
    namespace : mapping, default None
        Where to look up DataFrames by variable name (e.g., globals()).
        If None, defaults to globals().

    Returns
    -------
    dict
        A deep-copied, updated metadata dict that includes any missing
        dataframe keys with column types inferred from their values.

    Raises
    ------
    ValueError
        If a dataframe from `schema` is missing in `namespace`, so its columns
        cannot be discovered.
    """

    # Use the provided namespace for DataFrame lookup; fall back to global scope.
    # Example: if your DataFrame variable is named `df_customer` in your notebook,
    # this code will fetch it via lookup_space["df_customer"].
    lookup_space = globals() if namespace is None else namespace

    # Work on a copy so we don't mutate the original metadata dict passed in.
    updated_metadata = deepcopy(metadata)

    # Iterate over every dataframe name defined by the schema (keys only).
    for dataframe_name in schema.keys():

        # If the dataframe already exists in metadata, leave it as-is (unchanged).
        if dataframe_name in updated_metadata:
            continue

        # Fetch the actual pandas DataFrame object from the namespace by its variable name.
        df_obj = lookup_space.get(dataframe_name, None)

        # If it's not found, we cannot infer columns; instruct the caller to define it.
        if df_obj is None:
            raise ValueError(
                f"DataFrame {dataframe_name!r} not found in the provided namespace; "
                "cannot construct metadata for missing entry."
            )

        # Validate it's a pandas DataFrame (so we can read .columns and values).
        if not isinstance(df_obj, pd.DataFrame):
            raise ValueError(
                f"Object named {dataframe_name!r} is not a pandas DataFrame (got {type(df_obj)})."
            )

        # Build {column_name: inferred_abap_type} by **inspecting values** with `infer_abap_type`.
        # We pass the actual column values (with NaNs included; the helper handles them).
        inferred_col_to_type: Dict[str, str] = {}
        for column_name in df_obj.columns:
            column_values = df_obj[column_name].tolist()     # Example: ["000123","000124",...]
            inferred_type = infer_abap_type(column_values)   # Example: "NUMC"
            inferred_col_to_type[column_name] = inferred_type

        # The metadata format uses a list containing a single dict: [ {col: type, ...} ]
        updated_metadata[dataframe_name] = [inferred_col_to_type]

    # Return the updated copy to the caller.
    return updated_metadata







use_case_column_metadata = ensure_metadata_for_schema_dfs(schema=usecase_metadata['fkey_pkey_metadata'], metadata=use_case_column_metadata)


use_case_column_metadata


# In[41]:


usecase_metadata['fkey_pkey_metadata']


# **Build new schema - canon pkey and fkey**

# In[42]:


# # update the schema by renaming the fkey and pkey and triming the unused pkey  ←# handy aliases for type annotations
# #it also updates the dataframes and column metadata


# # ────────────────────────────────────────────────────────────────────
# # helper: canon                                                             #
# # --------------------------------------------------------------------------#
# def canon(col: str, tbl: str) -> str:            # ← build canonical column name from raw column + table name
#     return f"{col}_df_{tbl.removeprefix('df_')}" #     (strip a leading “df_” from the table name once)



# # ────────────────────────────────────────────────────────────────────
# # main transformation routine                                           #
# # --------------------------------------------------------------------#
# def normalise_keys_and_trim_pk(              # ← top-level API users call
#     schema:   Dict[str, Any],                    # • original FK/PK schema (mutable copy created internally)
#     col_meta: Dict[str, Any],                    # • original column-metadata dictionary
#     namespace: Dict[str, Any] | None = None,     # • optional alt namespace; defaults to real globals()
#     drop_raw_pk: bool  = True,                   # • whether to drop *raw* (non-canonical) PK columns
#     drop_raw_fk: bool  = True                    # • whether to drop *raw* FK columns
# ) -> Tuple[Dict[str, Any], Dict[str, Any]]:      # → returns (new_schema , new_metadata) only

#     ns = namespace or globals()                  # ← resolve where DataFrames live (Jupyter globals by default)

#     new_schema = deepcopy(schema)                # ← work on a deep copy so we never mutate caller originals
#     new_meta   = deepcopy(col_meta)              # ← same for the metadata dict

#     to_drop_raw = {t: set() for t in new_schema} # ← per-DataFrame set of raw columns that will be deleted later



#     # ───────────────── PASS 1 – parent PK canonicalisation ─────────────────
#     for p_tbl, spec in new_schema.items():                       # loop over every *parent* table
#         pk_raw = spec["pkey_col"]                                # grab raw PK (string *or* list)
#         pk_raw = [pk_raw] if isinstance(pk_raw, str) else list(pk_raw)  # normalise to list
#         pk_can = [canon(c, p_tbl) for c in pk_raw]               # build canonical names
#         spec["pkey_col"] = pk_can                                # patch schema in-place

#         meta_map = new_meta[p_tbl][0]                            # inner dict: {col: dtype}
#         for old, new in zip(pk_raw, pk_can):                     # move dtype entries over
#             meta_map[new] = meta_map.get(old, "CHAR")            # default dtype “CHAR” if unknown

#         df_p = ns.get(p_tbl)                                     # try to fetch the real DataFrame
#         if df_p is not None:                                     # -- only if loaded in memory
#             for old, new in zip(pk_raw, pk_can):                 #   iterate matching pairs
#                 df_p[new] = df_p[old] if old in df_p.columns else pd.NA  # copy values or fill <NA>
#             if drop_raw_pk:                                      #   shall we zap originals later?
#                 to_drop_raw[p_tbl].update(pk_raw)                #   remember which ones



#     # storage structure: which canonical parent-PK columns are *actually* referenced by children
#     truly_used: Dict[str, set[str]] = {t: set() for t in new_schema}



#     # ───────────────── PASS 2 – child FK canonicalisation ──────────────────
#     for c_tbl, spec in new_schema.items():                       # loop over each *child* table
#         df_c      = ns.get(c_tbl)                                # DataFrame for child (or None)
#         new_fkeys = []                                           # rebuilt FK-mapping list for this child

#         for mapping in spec["fkey_col_to_pkey_table"]:           # mapping is always single-item dict
#             (fk_raw_str, p_tbl), = mapping.items()               # unpack the only KV pair

#             # split raw FK column string on commas, strip spaces, ditch blanks / “_” placeholders
#             #fk_raw_parts = [tok for tok in re.split(r"\s*,\s*", fk_raw_str.strip()) if tok and tok != "_"]
#             fk_raw_parts = [tok.strip() for tok in re.split(r"\s*,\s*", fk_raw_str.strip())]

#             p_pk_can = new_schema[p_tbl]["pkey_col"]             # canonical parent PK list/str
#             p_pk_can = p_pk_can if isinstance(p_pk_can, list) else [p_pk_can]

#             mapped_parts = []                                    # track which FK parts survive

#             if df_c is not None:                                 # DataFrame present → copy values
#                 for raw, can in zip(fk_raw_parts, p_pk_can):     # iterate corresponding pairs
#                     df_c[can] = df_c[raw] if raw in df_c.columns else pd.NA  # copy / NA
#                     mapped_parts.append(can)                     # remember canonical part
                
#                 if drop_raw_fk:                                  # queue raw FK cols for later drop
#                     time_col = spec.get("time_col") 
#                     safe_raws = [r for r in fk_raw_parts if r != time_col] # dont not delete if its the time column of that dataframe 
#                     to_drop_raw[c_tbl].update(safe_raws)
#             else:                                                # DataFrame isn't loaded
#                 mapped_parts = p_pk_can[:len(fk_raw_parts)]      # still mark them as referenced

#             meta_c = new_meta[c_tbl][0]                          # column-metadata for child
#             for raw, can in zip(fk_raw_parts, mapped_parts):     # shift dtype info
#                 meta_c[can] = meta_c.get(raw, "CHAR")

#             truly_used[p_tbl].update(mapped_parts)               # flag parent parts as referenced
#             new_fkeys.append({", ".join(mapped_parts): p_tbl})   # save canonical FK mapping

#         spec["fkey_col_to_pkey_table"] = new_fkeys               # overwrite with canonical list



#     # ───────────────── PASS 3 – actually drop raw columns ──────────────────
#     for tbl, cols in to_drop_raw.items():                        # iterate queued deletions
#         df = ns.get(tbl)                                         # DataFrame object
#         if df is not None and cols:                              # only if DF present and something to drop
#             df.drop(columns=[c for c in cols if c in df.columns], inplace=True, errors="ignore")



#     # ───────────────── PASS 4 – trim unused parent-PK parts ────────────────
#     for p_tbl, spec in new_schema.items():                       # reconsider parent PKs
#         pk_can = spec["pkey_col"]                                # list or str
#         pk_can = pk_can if isinstance(pk_can, list) else [pk_can]

#         used_parts = truly_used.get(p_tbl, set())                # what children actually referenced
#         # keep every part if *nothing* referenced (single-column parents) else filter
#         keep = [c for c in pk_can if (not used_parts) or (c in used_parts)]

#         spec["pkey_col"] = keep if len(keep) != 1 else keep[0]   # shrink list→str when length==1




#         # meta_map = new_meta[p_tbl][0]                            # sync metadata removals
#         # for col in pk_can:
#         #     if col not in keep:
#         #         meta_map.pop(col, None)

#         # df_p = ns.get(p_tbl)                                     # also drop from DataFrame
#         # if df_p is not None:
#         #     df_p.drop(columns=[c for c in pk_can if c not in keep],
#         #               inplace=True, errors="ignore")





#     # ───────────────── PASS 5 – rebuild FK lists post-trim ────────────────
    

#     #just a checker and nothing is as such changing here as fkey lists are already rebuilt using the mapped parts previously
    
    
#     for c_tbl, spec in new_schema.items():                       # each child again
#         rebuilt = []                                             # fresh FK list
#         for mapping in spec["fkey_col_to_pkey_table"]:           # walk old list
#             _, p_tbl = next(iter(mapping.items()))               # extract parent table
#             pk_cols = new_schema[p_tbl]["pkey_col"]              # get *final* PK parts
#             pk_cols = pk_cols if isinstance(pk_cols, list) else [pk_cols]
#             fk_cols = [c for c in pk_cols if c in truly_used[p_tbl]]  # only parts that were referenced
#             rebuilt.append({", ".join(fk_cols): p_tbl})          # add cleaned mapping
#         spec["fkey_col_to_pkey_table"] = rebuilt                 # replace old FK list





#     # ───────────────── PASS 6 – drop columns that are all-NA ───────────────
#     for tbl in new_schema:                                       # for every DataFrame
#         df = ns.get(tbl)
#         if df is not None:
#             all_na = [c for c in df.columns if df[c].isna().all()]  # detect 100 % NA columns
#             if all_na:
#                 df.drop(columns=all_na, inplace=True, errors="ignore")  # delete from DF

#                 meta_map = new_meta[tbl][0]                      # delete from metadata
#                 for c in all_na:
#                     meta_map.pop(c, None)

#                 pk = new_schema[tbl]["pkey_col"]                 # adjust PK lists if needed
#                 pk = pk if isinstance(pk, list) else [pk]
#                 pk = [c for c in pk if c not in all_na]          # keep pkey only if not in all_NA
#                 new_schema[tbl]["pkey_col"] = pk if len(pk) != 1 else pk[0]

#                 for spec in new_schema.values():                 # scrub from *every* FK list
#                     cleaned = []
#                     for mapping in spec["fkey_col_to_pkey_table"]: #traverse the list of dict one by one
#                         fk_cols, p_tbl = next(iter(mapping.items())) #first dict of the list
#                         cols = [c for c in fk_cols.split(", ") if c not in all_na]
#                         if cols:
#                             cleaned.append({", ".join(cols): p_tbl})
#                     spec["fkey_col_to_pkey_table"] = cleaned



#     return new_schema, new_meta                                  # ← output canonicalised artefacts


# In[43]:


def canon(col: str, tbl: str) -> str:
    """
    Build canonical column name from raw column + table name.

    Example:
        canon("EBELN", "df_ekko") -> "EBELN_df_ekko"
    """
    return f"{col}_df_{tbl.removeprefix('df_')}"  # comment: strip "df_" once and attach as suffix


def normalise_keys_and_trim_pk_flexible(
    schema: Dict[str, Any],
    col_meta: Dict[str, Any],
    namespace: Optional[Dict[str, Any]] = None,
    drop_raw_pk: bool = True,
    drop_raw_fk: bool = True,
    overlap_threshold: float = 0.10,
    overlap_sample_size: int = 300000,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Canonicalize PK/FK column names (same convention as your existing function),
    but make FK->PK alignment ROBUST to:
      - FK order mismatches (child FK tokens not aligned to parent PK order)
      - FK column names differing from parent PK column names
      - FK being a subset of parent PK columns (only keep the FK parts that can be matched)

    How matching works (core change):
      - For each child FK column (raw), we try to find which parent PK column it refers to by
        comparing values overlap between:
            child_fk_values  vs  each parent_pk_values
      - If overlap >= overlap_threshold (default 50%), we treat it as a match and canonicalize the
        child FK to the matched parent PK canonical name.
      - If a child FK column does not match ANY parent PK column (by values), we drop it from the FK mapping.

    IMPORTANT design decision (per your request):
      - We DO NOT trim parent PK columns even if "unused".
        (So we do NOT delete parent PK parts just because no child references them.)

    Inputs:
      schema:
        Your original schema dict in the "old format" (pkey_col can be str or list;
        fkey_col_to_pkey_table is a list of 1-item dicts).
      col_meta:
        Column metadata dict (the one you already pass) that contains per-table dtype mappings.
      namespace:
        Dict containing live DataFrames (e.g., {**globals(), **locals()}).
        If None, defaults to globals().
      drop_raw_pk:
        If True, drop raw PK columns after creating canonical PK columns.
      drop_raw_fk:
        If True, drop raw FK columns after creating canonical FK columns (except time_col).
      overlap_threshold:
        Minimum overlap fraction to consider FK column matches a parent PK column.
      overlap_sample_size:
        Max number of rows used for overlap checks (sampling for speed).

    Outputs:
      (new_schema, new_meta)
        new_schema:
          Same structure but PKs and FKs are canonicalized and FK mappings are cleaned to only include
          FK columns that truly match parent PK columns by values overlap.
        new_meta:
          Updated metadata with canonical columns added and raw columns optionally removed.
    """

    # ------------------------------ #
    # Resolve namespace for DataFrames
    # ------------------------------ #

    ns = namespace or globals()  # comment: where df_* live (defaults to module globals)

    # ------------------------------ #
    # Deep copy inputs (never mutate caller objects)
    # ------------------------------ #

    new_schema = deepcopy(schema)  # comment: safe copy schema
    new_meta = deepcopy(col_meta)  # comment: safe copy metadata

    # ------------------------------ #
    # Track raw columns to drop at the end (per table)
    # ------------------------------ #

    to_drop_raw: Dict[str, set] = {t: set() for t in new_schema}  # comment: raw cols queued for deletion per df

    # ------------------------------ #
    # Helper: compute overlap fraction between two columns
    # ------------------------------ #
    # ------------------------------ #
    # Helper: compute overlap fraction between two columns
    # ------------------------------ #
    def compute_overlap_fraction(child_series: pd.Series, parent_series: pd.Series) -> float:
        """
        Compute overlap fraction = |intersection| / |child_set|.
        We MUST evaluate against the full parent set to avoid the sampling paradox
        on high-cardinality columns like Document Numbers.
        """
        # Drop NAs
        child_clean = child_series.dropna()
        parent_clean = parent_series.dropna()

        if len(child_clean) == 0 or len(parent_clean) == 0:
            return 0.0

        # 1. Grab the FULL set of unique parent values (very fast in pandas)
        parent_set = set(parent_clean.astype(str).unique())

        # 2. We can still sample the child to save time if it is massive
        if len(child_clean) > overlap_sample_size:
            child_clean = child_clean.sample(overlap_sample_size, random_state=0)

        # 3. Grab the unique child values
        child_set = set(child_clean.astype(str).unique())

        if len(child_set) == 0:
            return 0.0

        # 4. Overlap is the fraction of unique child values that exist ANYWHERE in the parent
        return len(child_set.intersection(parent_set)) / float(len(child_set))

    # ------------------------------ #
    # PASS 1: Canonicalize parent PK columns (same as your existing function)
    # ------------------------------ #
    for parent_table_name, table_spec in new_schema.items():  # comment: iterate all tables
        # comment: extract raw pk list (normalize to list)
        pk_raw = table_spec.get("pkey_col")  # comment: raw pk spec
        pk_raw_list = [pk_raw] if isinstance(pk_raw, str) else list(pk_raw or [])  # comment: normalize to list

        # comment: build canonical pk list
        pk_can_list = [canon(raw_col, parent_table_name) for raw_col in pk_raw_list]  # comment: canonical names

        # comment: write canonical pk list back into schema
        table_spec["pkey_col"] = pk_can_list  # comment: keep as list for now

        # comment: update metadata (dtype map assumed at new_meta[table][0])
        meta_map = new_meta[parent_table_name][0]  # comment: inner dtype map
        for old_name, new_name in zip(pk_raw_list, pk_can_list):  # comment: move dtype from raw->canonical
            meta_map[new_name] = meta_map.get(old_name, "CHAR")  # comment: default dtype if missing

        # comment: copy pk values in the actual dataframe if available
        df_parent = ns.get(parent_table_name)  # comment: actual df object
        if df_parent is not None:
            for old_name, new_name in zip(pk_raw_list, pk_can_list):  # comment: copy each pk column
                df_parent[new_name] = df_parent[old_name] if old_name in df_parent.columns else pd.NA  # comment: copy or NA
            if drop_raw_pk:
                to_drop_raw[parent_table_name].update(pk_raw_list)  # comment: queue raw pk cols for deletion

    # ------------------------------ #
    # PASS 2: Canonicalize child FKs (NEW: match by value overlap, not order)
    # ------------------------------ #
    for child_table_name, child_spec in new_schema.items():  # comment: iterate all child tables
        # comment: grab child df if present
        df_child = ns.get(child_table_name)  # comment: child df
        # comment: prepare rebuilt list of FK mappings for this child table
        rebuilt_fk_list: List[Dict[str, str]] = []  # comment: will replace old list

        # comment: read the original fk mapping list (list of 1-item dicts)
        original_fk_list = child_spec.get("fkey_col_to_pkey_table", [])  # comment: raw fk mappings

        # comment: loop each relationship mapping in schema
        for mapping in original_fk_list:
            # comment: unpack the single mapping pair: {fk_raw_str: parent_table_name}
            (fk_raw_str, parent_table_name), = mapping.items()  # comment: fk string + parent table

            # comment: parse raw fk tokens preserving order and underscores
            fk_tokens = [tok.strip() for tok in re.split(r"\s*,\s*", str(fk_raw_str).strip())]  # comment: keep "_" positions

            # comment: fetch parent pk canonical list (we wrote it in PASS 1)
            parent_pk_can = new_schema[parent_table_name]["pkey_col"]  # comment: can be list
            parent_pk_can_list = parent_pk_can if isinstance(parent_pk_can, list) else [parent_pk_can]  # comment: normalize

            # comment: fetch parent dataframe if available (needed for overlap matching)
            df_parent = ns.get(parent_table_name)  # comment: parent df

            # comment: track which canonical parent pk columns we decide to use for this relationship
            matched_parent_pk_cols: List[str] = []  # comment: final fk columns (canonical) for this mapping

            # comment: build a set of parent PK columns already claimed within this mapping to avoid double-assigning one pk
            used_parent_pk_cols: set = set()  # comment: prevent two child cols mapping to same parent pk col

            # comment: iterate each child FK token, skipping placeholders
            for raw_fk in fk_tokens:  # comment: process each FK token
                if raw_fk == "_" or raw_fk == "":  # comment: skip placeholder slots
                    continue  # comment: no actual fk column here

                # comment: if child doesn't have this raw FK column, we can't use it
                if raw_fk not in df_child.columns:
                    continue  # comment: skip missing column

                # comment: take the child fk series
                child_fk_series = df_child[raw_fk]  # comment: raw fk values

                # comment: search for the best matching parent pk column by overlap
                best_parent_pk_col = None  # comment: will store canonical parent pk col
                best_overlap = 0.0  # comment: best overlap so far

                # comment: evaluate overlap against each parent pk canonical column
                for parent_pk_col in parent_pk_can_list:  # comment: loop over all parent pk columns
                    if parent_pk_col in used_parent_pk_cols:
                        continue  # comment: avoid reusing same parent pk within this mapping
                    if parent_pk_col not in df_parent.columns:
                        continue  # comment: parent df missing this pk col (should not happen, but safe)

                    # comment: compute overlap fraction for this candidate
                    overlap = compute_overlap_fraction(child_fk_series, df_parent[parent_pk_col])  # comment: overlap score

                    # comment: keep the best scoring candidate
                    if overlap > best_overlap:
                        best_overlap = overlap  # comment: update best score
                        best_parent_pk_col = parent_pk_col  # comment: update best pk col

                # comment: if best overlap passes threshold, accept the match
                if best_parent_pk_col is not None and best_overlap >= overlap_threshold:
                    # comment: create/overwrite the child canonical fk column name to match parent pk canonical col
                    df_child[best_parent_pk_col] = df_child[raw_fk]  # comment: copy raw FK values into canonical col
                    matched_parent_pk_cols.append(best_parent_pk_col)  # comment: remember used parent pk col
                    used_parent_pk_cols.add(best_parent_pk_col)  # comment: mark this pk col as used

                    # comment: update metadata so canonical fk col has a dtype entry
                    meta_child_map = new_meta[child_table_name][0]  # comment: dtype map for child table
                    meta_child_map[best_parent_pk_col] = meta_child_map.get(raw_fk, "CHAR")  # comment: copy dtype or default

                    # comment: queue raw fk column for deletion if enabled (but never delete if it's the child time column)
                    if drop_raw_fk:
                        time_col = child_spec.get("time_col")  # comment: child's time col in schema
                        if raw_fk != time_col:  # comment: don't drop time col
                            to_drop_raw[child_table_name].add(raw_fk)  # comment: queue raw FK col for deletion

                # comment: if overlap is below threshold, we drop this fk token (it is likely not a true FK to parent PK)

            # comment: if we matched no parent pk columns, drop this relationship entirely
            if len(matched_parent_pk_cols) == 0:
                continue  # comment: nothing to connect child->parent

            # comment: add the rebuilt relationship using canonical parent pk column names
            rebuilt_fk_list.append({", ".join(matched_parent_pk_cols): parent_table_name})  # comment: store relationship

        # comment: write rebuilt fk list into schema for this child table
        child_spec["fkey_col_to_pkey_table"] = rebuilt_fk_list  # comment: replace original mappings

    # ------------------------------ #
    # PASS 3: Drop raw columns (optional), after we've created canonical columns
    # ------------------------------ #
    for table_name, cols_to_drop in to_drop_raw.items():  # comment: iterate queued deletions
        df_obj = ns.get(table_name)  # comment: fetch df
        if df_obj is not None and len(cols_to_drop) > 0:
            df_obj.drop(columns=[c for c in cols_to_drop if c in df_obj.columns], inplace=True, errors="ignore")  # comment: drop safely

    # ------------------------------ #
    # PASS 4: DO NOT TRIM parent PK parts (per your request)
    # ------------------------------ #
    # comment: intentionally removed trimming logic; parent PK stays as originally canonicalized

    # ------------------------------ #
    # PASS 5: Rebuild FK lists post-drop-all-NA (we keep it simple: no-op here)
    # ------------------------------ #
    # comment: we rely on PASS 6 to scrub FK mappings if canonical columns become all-NA

    # ------------------------------ #
    # PASS 6: Drop columns that are all-NA and scrub schema/meta accordingly
    # ------------------------------ #
    for table_name in new_schema:  # comment: iterate all tables
        df_obj = ns.get(table_name)  # comment: get df
        if df_obj is None:
            continue  # comment: skip if df not loaded

        # comment: identify columns that are entirely NA
        all_na_cols = [c for c in df_obj.columns if df_obj[c].isna().all()]  # comment: fully missing columns
        if len(all_na_cols) == 0:
            continue  # comment: nothing to clean

        # comment: drop all-NA columns from df
        df_obj.drop(columns=all_na_cols, inplace=True, errors="ignore")  # comment: safe drop

        # comment: drop all-NA columns from metadata
        meta_map = new_meta[table_name][0]  # comment: dtype map
        for c in all_na_cols:
            meta_map.pop(c, None)  # comment: remove dtype entry

        # comment: scrub FK mappings in every table to remove all-NA FK columns
        for spec in new_schema.values():  # comment: iterate each table spec
            cleaned_fk_list: List[Dict[str, str]] = []  # comment: rebuilt list
            for mapping in spec.get("fkey_col_to_pkey_table", []):  # comment: iterate fk mappings
                fk_cols_str, parent_tbl = next(iter(mapping.items()))  # comment: unpack mapping
                fk_cols = [x.strip() for x in fk_cols_str.split(",") if x.strip()]  # comment: split fk cols
                fk_cols = [x for x in fk_cols if x not in all_na_cols]  # comment: remove all-NA cols
                if len(fk_cols) > 0:
                    cleaned_fk_list.append({", ".join(fk_cols): parent_tbl})  # comment: keep mapping if still valid
            spec["fkey_col_to_pkey_table"] = cleaned_fk_list  # comment: write back cleaned list

    # ------------------------------ #
    # Final: normalize PK storage (list->str when length==1), for convenience
    # ------------------------------ #
    for table_name, spec in new_schema.items():  # comment: loop all tables
        pk = spec.get("pkey_col")  # comment: pk field
        if isinstance(pk, list) and len(pk) == 1:
            spec["pkey_col"] = pk[0]  # comment: shrink list to string

    # comment: return the transformed schema + metadata
    return new_schema, new_meta


# In[44]:


new_schema, new_meta = normalise_keys_and_trim_pk_flexible(usecase_metadata['fkey_pkey_metadata'], use_case_column_metadata)


new_schema


# In[45]:


use_case_column_metadata = new_meta

use_case_column_metadata


# In[46]:


new_schema


# **Remove the subset fkey to pkey in the child table and also remove from df and col_metadata**

# In[47]:


def enforce_exact_matches_and_clean_data(
    schema: Dict[str, Any], # Parameter: The full database schema dictionary.
    column_metadata: Dict[str, Any], # Parameter: The metadata dictionary mapping table columns to their ABAP types.
    namespace: Dict[str, Any] = None # Parameter: The environment where the DataFrames live (defaults to None so we can set it to globals later).
) -> Tuple[Dict[str, Any], Dict[str, Any]]: # Defines the return type: A tuple containing two dictionaries (updated schema, updated metadata).
    """
    1. Removes subset FK relationships from the schema.
    2. Drops the orphaned FK columns from the corresponding DataFrames (in-place).
    3. Drops the orphaned FK columns from the use_case_column_metadata.
    """
    
    # Default to global namespace if none provided
    if namespace is None: # Checks if the user explicitly provided a namespace dict.
        namespace = globals() # If not, defaults to the global variables of the environment where the function is running.

    # Create deep copies of the schemas so we don't mutate the originals mid-loop
    clean_schema = copy.deepcopy(schema) # Creates a completely independent copy of the schema so original data isn't altered by mistake.
    clean_metadata = copy.deepcopy(column_metadata) # Creates a completely independent copy of the column metadata.
    
    # 1. Iterate through every child table in the schema
    for child_table, table_info in clean_schema.items(): # Loops through each table name (key) and its schema properties (value) in the copied schema.
        
        valid_exact_matches = [] # Initializes an empty list to store the foreign key relationships that are confirmed 1-to-1 exact matches.
        cols_to_potentially_drop = set() # Initializes an empty set to collect column names that belong to subset relationships (candidates for deletion).
        
        fkey_list = table_info.get('fkey_col_to_pkey_table', []) # Safely retrieves the list of foreign key dictionaries for the current child table.
        
        # 2. Evaluate every foreign key relationship this child has
        for fkey_dict in fkey_list: # Loops through each specific relationship mapping (e.g., {'FK1, FK2': 'parent_table'}).
            for fk_string, parent_table in fkey_dict.items(): # Unpacks the dictionary to get the comma-separated FK string and the target parent table name.
                
                # A. Count child FKs
                child_fks = [k.strip() for k in fk_string.split(',') if k.strip()] # Splits the "FK1, FK2" string by commas, strips spaces, and creates a clean list.
                num_child_fks = len(child_fks) # Counts exactly how many foreign keys the child is offering for this relationship.
                
                # B. Count parent PKs
                parent_info = clean_schema.get(parent_table, {}) # Safely fetches the parent table's schema definition.
                parent_pk_spec = parent_info.get('pkey_col', []) # Safely fetches the parent table's primary key definition.
                
                if isinstance(parent_pk_spec, str): # Checks if the parent's PK is just a single string.
                    parent_pks = [parent_pk_spec] # If so, wraps it in a list so we can count it uniformly.
                elif isinstance(parent_pk_spec, list): # Checks if the parent's PK is already a list.
                    parent_pks = parent_pk_spec # Keeps the list as is.
                else: # Fallback if the parent PK is missing or malformed.
                    parent_pks = [] # Sets it to an empty list.
                    
                num_parent_pks = len(parent_pks) # Counts exactly how many primary keys the parent requires for a full match.
                
                # C. The Decision: Exact Match vs. Subset
                if num_child_fks == num_parent_pks: # Compares the number of keys offered by the child to the number demanded by the parent.
                    # EXACT MATCH: Keep the relationship
                    valid_exact_matches.append(fkey_dict) # Because counts match, saves this relationship to the "keep" list.
                else:
                    # SUBSET: Mark the columns for deletion
                    cols_to_potentially_drop.update(child_fks) # Adds the child's subset FK columns to the deletion set.
                    print(f"[EDGE REMOVED] {child_table} -> {parent_table} (Subset: {num_child_fks} keys to {num_parent_pks} keys)") # Logs that an edge was severed due to being a subset.
        
        # Update the schema to ONLY contain the exact matches
        clean_schema[child_table]['fkey_col_to_pkey_table'] = valid_exact_matches # Overwrites the child's FK list in the schema with ONLY the valid, full matches.
        
        # ---------------------------------------------------------
        # 3. SAFETY CHECK: Ensure we don't drop crucial columns
        # ---------------------------------------------------------
        if cols_to_potentially_drop: # Checks if we flagged any columns for deletion in this table.
            
            
      
                
                # A. Drop from the actual DataFrame in the namespace
                df = namespace.get(child_table) # Looks up the actual Pandas DataFrame object using the table's string name as the variable name.
                if df is not None and isinstance(df, pd.DataFrame): # Verifies that the variable exists and is truly a DataFrame.
                    # Only drop columns that actually exist in the DataFrame
                    existing_cols_to_drop = [c for c in cols_to_potentially_drop if c in df.columns] # Filters the drop list to ensure we don't try to drop a column that's already missing from the DataFrame.
                    if existing_cols_to_drop: # Checks if there is anything actionable to drop.
                        df.drop(columns=existing_cols_to_drop, inplace=True) # Physically removes the columns from the DataFrame memory.
                        print(f"  -> [DF CLEANED] Dropped {existing_cols_to_drop} from DataFrame '{child_table}'.") # Logs the successful DataFrame mutation.
                

                # B. Drop from the Use Case Column Metadata
                if child_table in clean_metadata and len(clean_metadata[child_table]) > 0: # Verifies this table exists in the metadata dictionary and isn't empty.
                    meta_dict = clean_metadata[child_table][0] # Extracts the inner dictionary {column_name: ABAP_type} from the metadata list.
                    for col in cols_to_potentially_drop: # Loops through the columns we are trying to delete.
                        if col in meta_dict: # Checks if the column currently exists in the metadata dictionary.
                            del meta_dict[col] # Removes the column and its ABAP type from the metadata dictionary.
                            print(f"  -> [META CLEANED] Dropped '{col}' from column metadata of '{child_table}'.") # Logs the successful metadata cleanup.

    return clean_schema, clean_metadata # Returns the fully processed, clean schema and the updated metadata dictionaries to the caller.


# In[48]:


# Pass in your schema, metadata, and the local namespace
updated_database_schema, updated_use_case_column_metadata = enforce_exact_matches_and_clean_data(
    schema= new_schema, 
    column_metadata=use_case_column_metadata, 
    namespace= globals(),
)

# Optional: Reassign them back to your main variables if desired
database_schema = updated_database_schema
use_case_column_metadata = updated_use_case_column_metadata


# In[49]:


database_schema


# In[50]:


use_case_column_metadata


# **CONVERT EVERYTHING TO SOAC FORMAT - Metadata level**

# In[51]:


#metadata schema to soac database schema format

def convert_old_format_to_new_database_schema(old_dict,use_case):
    """
    Transforms the old-format dictionary into the new format.
    Drops 'origin_table', keeps 'pkey_col' and 'time_col',
    and flattens the list-of-dicts in 'fkey_col_to_pkey_table'
    into a single dictionary of { column_name: table }.
    """
    new_dict = {}

    for df_name, info in old_dict.items():
        # info is something like:
        # {
        #   'origin_table': 'BSEG',
        #   'pkey_col': [...],
        #   'fkey_col_to_pkey_table': [ { 'COL1,COL2': 'some_other_df'}, ...],
        #   'time_col': ...
        # }

        pkey_col = info.get('pkey_col')               # copy pkey_col as is
        time_col = info.get('time_col', None)         # copy time_col, defaulting to None

        # Build the new fkey dict
        new_fkey_dict = {}
        # "fkey_col_to_pkey_table" is a list of dicts. Each dict might look like:
        # { 'BUKRS,BELNR,GJAHR': 'df_bkpf' }
        # We want to flatten that.
        for dct in info.get('fkey_col_to_pkey_table', []):
            # For each dictionary in the list, it might have one key or multiple, e.g.:
            #   { "COL1,COL2": "df_xyz" }
            # We'll parse each key -> value
            for joined_cols, target_table in dct.items():
                # If columns are joined by comma, let's split them
                col_list = joined_cols.split(',')
                # For each column in col_list, map it to target_table
                for col in col_list:
                    # It's possible to strip whitespace if needed:
                    col = col.strip()
                    new_fkey_dict[col] = target_table

        # Construct the final dictionary for this df
        new_dict[df_name] = {
            'pkey_col': pkey_col,
            'fkey_col_to_pkey_table': new_fkey_dict,
            'time_col': time_col
        }

    database_schema_dict={}
    
    database_schema_dict[use_case]=new_dict
    
    return database_schema_dict


database_schema=convert_old_format_to_new_database_schema(database_schema,use_case)




database_schema


# In[52]:


#updating the database_schema by checking if the time_col exist in the dataframe or not, if not then it assigns the time_col None


def update_time_cols_with_namespace(  # define a function to validate/normalize time_col entries using namespace lookups
    database_schema: Mapping[str, Mapping[str, Dict[str, Any]]],  # full schema: {use_case: {table_name: {..}}}
    use_case: str,  # which use_case key in the schema to operate on
    namespace: Optional[Dict[str, Any]] = None,  # if None, use caller's globals/locals
    inplace: bool = True,  # whether to mutate the input schema or work on a copy
    verbose: bool = True,  # whether to print progress/info messages
):
    """
    For each table in database_schema[use_case]:
      • Look up a variable with the same name as the table (e.g., "df_pslsdocitemptd")
        in the provided `namespace` (or in the caller's globals/locals if None).
      • If that variable is a pandas DataFrame, validate/normalize its `time_col`:
          - If time_col is a string and exists in df.columns → keep it.
          - If time_col is a list/tuple → use the first name that exists; else set to None.
          - If time_col is invalid or not found in df → set to None.
      • If the DataFrame variable isnt found, leave time_col as-is.

    Returns the updated schema (or the same object if `inplace=True`).
    """

    # Work on a copy unless the caller explicitly wants in-place edits
    schema = database_schema if inplace else deepcopy(database_schema)  # choose original or a deep copy based on `inplace`

    if use_case not in schema:  # ensure the requested use_case exists in the schema
        raise KeyError(f"use_case {use_case!r} not found in database_schema")  # fail early with a helpful message

    # Resolve the namespace to search for DataFrames.
    # If none supplied, use the CALLER's globals + locals.
    if namespace is None:  # if caller didn't pass a namespace dict
        frame = sys._getframe(1)  # caller's frame (1 level up from this function)
        ns = dict(frame.f_globals)  # start with caller's global variables
        ns.update(frame.f_locals)  # overlay caller's local variables
    else:
        ns = namespace  # otherwise use the provided namespace directly

    case_dict = schema[use_case]  # get the dict of tables for the specified use_case

    for table_name, table_info in case_dict.items():  # iterate over each table and its metadata
        # Try to find a variable with this exact name that is a DataFrame
        obj = ns.get(table_name)  # look up a variable whose name matches the table_name
        if not isinstance(obj, pd.DataFrame):  # if no DataFrame variable is found
            if verbose:
                print(f"[{table_name}] DataFrame variable not found in namespace → skipping (time_col stays {table_info.get('time_col')!r}).")
            continue  # skip to next table without modifying time_col
        df = obj  # alias for clarity

        tcol = table_info.get("time_col", None)  # read the declared time_col (could be str, list, None, etc.)

        # Nothing to validate if None/empty
        if not tcol:  # if time_col is None, empty string, or falsy
            if verbose:
                print(f"[{table_name}] time_col is None/empty → leaving unchanged.")
            continue  # nothing to do for this table

        # Normalize to a list of candidate names
        if isinstance(tcol, str):  # a single column name
            candidates = [tcol]  # wrap into a 1-element list
        elif isinstance(tcol, (list, tuple)):  # multiple possible names provided
            candidates = [c for c in tcol if isinstance(c, str) and c]  # keep only non-empty strings
        else:
            if verbose:
                print(f"[{table_name}] time_col has invalid type {type(tcol).__name__} → setting to None.")
            table_info["time_col"] = None  # invalid type: normalize to None
            continue  # move on to the next table

        # Keep candidates that actually exist in the DataFrame
        existing = [c for c in candidates if c in df.columns]  # check which candidate names are real columns

        if existing:  # if at least one candidate exists in the DataFrame
            chosen = existing[0]  # pick the first existing candidate
            if verbose and chosen != tcol:  # if we normalized from list or changed from the original string
                print(f"[{table_name}] normalized time_col from {tcol!r} to {chosen!r}.")
            table_info["time_col"] = chosen  # write back the validated/normalized time_col
        else:
            if verbose:
                print(f"[{table_name}] declared time_col {tcol!r} not found in DataFrame → setting to None.")
            table_info["time_col"] = None  # none of the candidates are valid → set to None

    return schema  # return the updated (or original, if inplace=True) schema object


# In[53]:


updated_schema = update_time_cols_with_namespace(
    database_schema=database_schema,
    use_case=use_case,  # or "PDP", etc.
    # namespace=None  # optional; uses caller's globals/locals automatically
    inplace=True,   # set True if you want to mutate database_schema directly
    verbose=True
)

updated_schema


# In[54]:


database_schema = updated_schema

database_schema


# In[55]:


###############################################################################
# ensure_pk_fk_are_strings.py                                                 #
# --------------------------------------------------------------------------- #
# A single helper that **mutates in‑memory DataFrames** so that every         #
#  • primary‑key column                                                      │
#  • foreign‑key column                                                      │
# is stored with pandas’ **nullable string dtype**.                           #
#                                                                             #
# Why? → Later joins (`merge`, `map`, …) will work even when the original     #
#        sources used mixed dtypes (object, Int64, category, …).              #
###############################################################################




def ensure_pk_fk_are_strings(
    db_schema : Mapping[str, Dict[str, Any]],
    use_case  : str,
    *,
    namespace : Mapping[str, Any] | None = None,
) -> None:
    """
    Walk through every table listed in `db_schema[use_case]` **in‑place**
    and convert its PK/FK columns to the *nullable string* dtype.

    Parameters
    ----------
    db_schema   : the big `database_schema` dict you posted above.
    use_case    : top level key inside `db_schema` (e.g. "SaDP…").
    namespace   : where the `df_*` DataFrames live.
                  •  None (default)  →  callers `globals()`
                  •  `locals()` inside another function
                  •  any mapping of {variablename: object}

    Returns
    -------
    None  all mutations happen directly on the DataFrames.

    Example
    -------
    >>> ensure_pk_fk_are_strings(database_schema, "SaDP…")
    >>> df_isdsalesdocitem.dtypes                     # check result
    SALESDOCUMENT_df_isdsalesdoc      string
    CALENDARDATE_df_icalendardate     string
    ...
    """
    # ----------------------------------------------------------------------
    # 1. decide where to look for the df_* variables
    # ----------------------------------------------------------------------
    if namespace is None:
        # caller’s global namespace (not this helper’s own globals!)
        namespace = inspect.currentframe().f_back.f_globals

    # ----------------------------------------------------------------------
    # 2. fetch the sub‑dict for the chosen use‑case
    # ----------------------------------------------------------------------
    try:
        tables_meta = db_schema[use_case]            # { "df_x": {...}, ... }
    except KeyError as err:
        raise KeyError(f"use_case='{use_case}' not found in database_schema") from err

    # ----------------------------------------------------------------------
    # 3. iterate over every table description
    # ----------------------------------------------------------------------
    for table_name, meta in tables_meta.items():
        # ---- 3 a) locate the actual DataFrame object --------------------
        try:
            df = namespace[table_name]               # e.g. df_isdsalesdocitem
        except KeyError:
            print(f"[warn] DataFrame variable '{table_name}' not found – skipped")
            continue                                # move on to next table

        # ---- 3 b) collect *all* PK + FK column names --------------------
        # • pkey_col value can be  str    or  List[str]
        pk_raw = meta["pkey_col"]
        pk_cols = pk_raw if isinstance(pk_raw, list) else [pk_raw]

        fk_cols = list(meta["fkey_col_to_pkey_table"].keys())  # keys are the FK columns
        cols_to_cast = set(pk_cols) | set(fk_cols)             # remove duplicates

        # ---- 3 c) cast each column to pandas' *nullable* string ---------
        for col in cols_to_cast:
            if col not in df.columns:
                print(f"[warn] '{col}' not found in '{table_name}' – skipped")
                continue

            # use StringDtype so <NA> stays as missing, not the literal "NA"
            df[col] = df[col].astype("string[python]")

       
       
ensure_pk_fk_are_strings(database_schema, use_case)


# In[56]:


tenant_metadata


# In[57]:


#convert the GSSL tenant_metadata to GMUL tenant_metadata format 


def normalize_tenant_metadata(
    obj: Dict[str, Any]
) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Convert old timesplit format -> new format.
    - If `obj` is already in the new format, return it unchanged.
    - If `obj` is in the old format and has one tenant id, return a single new-format dict.
    - If `obj` is in the old format and has multiple tenant ids, return a list of new-format dicts.

    Old format (example):
    {
      "740257989": {
        "ISDSALESDOCITEM": {"train": [...], "validation": [...], "test": [...]},
        "ISDSALESDOC":     {"train": [...], "validation": [...], "test": [...]}
      }
    }

    New format (example):
    {
      "tenant_id": "740257989",
      "timesplits": {
        "ISDSALESDOCITEM": {"train": [...], "validation": [...], "test": [...]},
        "ISDSALESDOC":     {"train": [...], "validation": [...], "test": [...]}
      }
    }
    """
    if not isinstance(obj, dict):
        raise ValueError("Expected a dict.")

    # Already new format? (has both keys)
    if "tenant_id" in obj and "timesplits" in obj:
        return obj

    # Otherwise assume old format: {tenant_id: {table: {train/validation/test}} , ...}
    converted: List[Dict[str, Any]] = []
    for tenant_id, timesplits in obj.items():
        if not isinstance(timesplits, dict):
            raise ValueError(f"Invalid old-format structure for tenant {tenant_id!r}: expected a dict of tables.")
        converted.append({
            "tenant_id": str(tenant_id),
            "timesplits": timesplits
        })

    if not converted:
        raise ValueError("Could not detect old or new format (empty or invalid dict).")

    # If a single tenant was present, return the single dict; else return list.
    return converted[0] if len(converted) == 1 else converted




tenant_metadata = normalize_tenant_metadata(tenant_metadata)



tenant_metadata





# In[58]:


# we had moved this cell after the auto entity table selection, as we need the enity table as the parameter to select between the actual splits or row_splits




# #convert tenant metadata to soac metadata format



# #   Dynamic Behavior:
# #     - Scans the namespace for DataFrames associated with the tables in the metadata (e.g., 'df_qmel').
# #     - If any of these DataFrames contain a '__row_order__' column, the function extracts the 
# #       integer 'row_splits' to slice the data.
# #     - If the '__row_order__' column is completely absent, it safely falls back to the conventional
# #       datetime strings ('train', 'validation', 'test').



# # # ------------------------- EXAMPLE USAGE -------------------------
# # old_data = {
# #   'tenant_id': '740401466',
# #   'timesplits': {'QMEL': {'train': ['2017-10-06', '2024-06-10'],
# #   'validation': ['2023-11-17', '2024-11-25'],
# #   'test': ['2024-07-15', '2026-01-30'],
# #   'row_splits': {'train': [0, 1153],
# #    'validation': [1154, 1318],
# #    'test': [1319, 1648]}},
# # }

# # result = convert_old_to_new(old_data)
# # print(result)
# # # Expected:
# # # {
# # #   "date_train_test_split": {
# # #       "743367704": {
# # #           "train": ["2023-11-01", "2024-06-01"],
# # #           "validation": ["2024-06-01", "2024-07-31"],
# # #           "test": ["2024-07-31", "2024-10-15"]
# # #       }
# # #   }
# # # }




# def convert_old_to_new_tenant_metadata(old_data: dict, namespace: dict = None) -> dict:
#     """
#     Converts the old-format tenant metadata dictionary into the new format required by the pipeline.
    
#     Dynamic Behavior:
#     - Scans the namespace for DataFrames associated with the tables in the metadata (e.g., 'df_qmel').
#     - If any of these DataFrames contain a '__row_order__' column, the function extracts the 
#       integer 'row_splits' to slice the data.
#     - If the '__row_order__' column is completely absent, it safely falls back to the conventional
#       datetime strings ('train', 'validation', 'test').
#     """
    
#     # Set the namespace to globals() if none is provided, allowing us to find the DataFrames in memory.
#     if namespace is None:
#         # Fallback to the global namespace of the caller.
#         namespace = globals()

#     # Extract the tenant_id from the original metadata dictionary.
#     tenant_id = old_data["tenant_id"]
    
#     # Extract the dictionary containing the time splits for all the tables.
#     timesplit_dict = old_data["timesplits"]

#     # Initialize a boolean flag to track whether we should use row_splits (integer format) instead of datetimes.
#     use_row_splits = False
    
#     # Initialize a variable to store the name of the table we ultimately pull the splits from.
#     selected_table = None

#     # Loop through every table name defined in the timesplits dictionary (e.g., 'QMEL', 'QMFE').
#     for table_name in timesplit_dict.keys():
        
#         # Construct the expected DataFrame variable name by lowercasing the table name and prepending 'df_'.
#         df_name = f"df_{table_name.lower()}"
        
#         # Attempt to fetch the actual DataFrame object from the namespace using the constructed name.
#         df = namespace.get(df_name)

#         # Check if the fetched object actually exists and is a valid pandas DataFrame.
#         if df is not None and isinstance(df, pd.DataFrame):
            
#             # Check if the special '__row_order__' column exists within this DataFrame's columns.
#             if '__row_order__' in df.columns:
                
#                 # If it does, set our flag to True so we know to use the integer row_splits.
#                 use_row_splits = True
                
#                 # Record this specific table as the one we will extract the split values from.
#                 selected_table = table_name
                
#                 # Print the requested confirmation message indicating we found the row order column.
#                 print(f"Row order '__row_order__' is present in DataFrame '{df_name}'. Using integer row splits format.")
                
#                 # Break out of the loop early since we found what we needed and all tables share the same splits.
#                 break

#     # If the loop finishes and we never found the '__row_order__' column in any DataFrame...
#     if not use_row_splits:
        
#         # Default the selected table to the very first table in the timesplit_dict.
#         selected_table = next(iter(timesplit_dict))
        
#         # Print the requested fallback message indicating we are using the conventional datetime format.
#         print("Row order '__row_order__' is not present in any associated DataFrame. Using conventional datetime format.")

#     # Extract the sub-dictionary of splits for the table we decided to use.
#     table_splits = timesplit_dict[selected_table]

#     # Check if our flag was set to True, meaning we should extract the integer row splits.
#     if use_row_splits:
        
#         # Extract the train list from the nested 'row_splits' dictionary.
#         train_vals = table_splits['row_splits']['train']
        
#         # Extract the validation list from the nested 'row_splits' dictionary.
#         val_vals = table_splits['row_splits']['validation']
        
#         # Extract the test list from the nested 'row_splits' dictionary.
#         test_vals = table_splits['row_splits']['test']
        
#     # Otherwise, fall back to the conventional datetime string lists.
#     else:
        
#         # Extract the train list directly from the table's dictionary.
#         train_vals = table_splits['train']
        
#         # Extract the validation list directly from the table's dictionary.
#         val_vals = table_splits['validation']
        
#         # Extract the test list directly from the table's dictionary.
#         test_vals = table_splits['test']

#     # Construct the final new-format dictionary mapping the tenant_id to its chosen splits.
#     new_data = {
#         # Create the top-level key required by the downstream pipeline.
#         "date_train_test_split": {
#             # Nest the extracted tenant_id.
#             tenant_id: {
#                 # Assign the chosen train values (either integers or datetime strings).
#                 "train": train_vals,
#                 # Assign the chosen validation values (either integers or datetime strings).
#                 "validation": val_vals,
#                 # Assign the chosen test values (either integers or datetime strings).
#                 "test": test_vals
#             }
#         }
#     }
    
#     # Return the newly constructed, perfectly formatted dictionary back to the caller.
#     return new_data


# tenant_metadata=convert_old_to_new_tenant_metadata(tenant_metadata)

# tenant_metadata


# In[59]:


#required by the soac_db class

usecase_metadata_semantics=usecase_metadata["column_info"]["semantics"]

usecase_metadata_semantics


# **Data Preprocessing (removing non-signaling columns)**

# In[60]:


#function to remove special columns from the dataframe "__row_idx__", "__row_url__"

# ────────────────────────────────────────────────────────────────────────
#  1. mini helper – drops the two “special” columns from *one* DataFrame
# ────────────────────────────────────────────────────────────────────────
def _strip_special_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Return *df* without __row_idx__ / __row_url__ (if present)."""
    return df.drop(columns=[c for c in ("__row_idx__", "__row_url__") if c in df.columns])


# ────────────────────────────────────────────────────────────────────────
#  2. main routine – nothing fancy, just straight-line logic
# ────────────────────────────────────────────────────────────────────────
def auto_drop_special_cols(
    schema: Dict[str, Any],
    use_case: str,
    env: Dict[str, Any] | None = None,      # where the df_* variables live
) -> None:
    """
    • Look only inside  schema[use_case]
    • For every key that starts with 'df_'  ➜  treat it as a DataFrame name
    • Grab that variable from *env* (defaults to globals())
    • Strip the special columns and write the cleaned DataFrame back
    """
    if env is None:                         # fall back to module-level globals()
        env = globals()

    # ── step-1: get the sub-dict that belongs to the chosen use-case ──
    try:
        tables = schema[use_case]           # dict with df_* keys
    except KeyError:
        raise KeyError(f"Use-case '{use_case}' not found in the supplied schema")

    # ── step-2: loop through those keys and clean their DataFrames ──
    for df_name in tables:                  # e.g. 'df_isdsalesdocitem'
        if not df_name.lower().startswith("df_"):
            continue                        # skip non-DataFrame entries

        df_obj = env.get(df_name)           # fetch variable by name
        if isinstance(df_obj, pd.DataFrame):
            env[df_name] = _strip_special_cols(df_obj)   # overwrite in place


auto_drop_special_cols(database_schema, use_case = use_case)


# In[61]:


def drop_highly_empty_columns(database_schema, use_case, threshold=0.95):
    """
    Drops columns from DataFrames (in the current namespace) if 95% or more values are missing.
    
    Args:
        database_schema (dict): The schema dictionary where keys are DataFrame variable names.
        threshold (float): The fraction of allowed missing values (default=0.85).
    """
    namespace = globals()
    
     # Extract the schema for the use case (assumes single root key)
    schema = database_schema[use_case]
    
    # Loop through every table in the schema
    for tbl_name, tbl_schema in schema.items():
        # Try to find the DataFrame with the given name in the namespace
        df = namespace.get(tbl_name)
        # Calculate fraction of missing for each column
        missing_fraction = df.isnull().mean()
        # Get columns where missing fraction is >= threshold
        cols_to_drop = missing_fraction[missing_fraction >= threshold].index.tolist()
        # Drop those columns in place
        df.drop(columns=cols_to_drop, inplace=True)
        print(f"Table '{tbl_name}': Dropped columns {cols_to_drop}")
   

drop_highly_empty_columns(database_schema, use_case)


# In[62]:


#stripping leading zeros from pkey and fkey for better reindexing 


def strip_leading_zeros_all_tables(database_schema, use_case,namespace=None, date_keywords=('DATE', 'TIME', "DAY", "MONTH", "YEAR", "TS", "STAMP")):
    """
    For each table in database_schema, strips leading zeros from all pkey/fkey columns,
    except for those whose names look like date/time columns.
    Modifies DataFrames in-place in the given namespace.
    """

    # Use the provided namespace or fallback to globals()
    if namespace is None:
        namespace = globals()
    
    # Extract the schema for the use case (assumes single root key)
    schema = database_schema[use_case]
    
    # Loop through every table in the schema
    for tbl_name, tbl_schema in schema.items():

        # Try to find the DataFrame with the given name in the namespace
        df = namespace.get(tbl_name)

        # If DataFrame does not exist, skip and warn
        if df is None:
            print(f"[WARN] DataFrame {tbl_name} not found in namespace, skipping.")
            continue

        # ---- Collect PK and FK column names ----
        pkey_cols = tbl_schema['pkey_col']
        # If pkey_cols is not a list, make it a list for uniformity
        if not isinstance(pkey_cols, list): 
            pkey_cols = [pkey_cols]
        
        # fkey_col_to_pkey_table is always a dict; just get the keys (column names)
        fkey_cols = list(tbl_schema['fkey_col_to_pkey_table'].keys())

        # Combine all key columns (primary + foreign)
        key_cols = pkey_cols + fkey_cols

        # ---- Filter out date/time columns by name ----
        # Only keep columns NOT containing any of the date_keywords
        # Now filter out columns that contain any date/time keywords
        filtered_key_cols = []
        for col in key_cols:
            col_upper = col.upper()
            has_date_keyword = False
            for keyword in date_keywords:
                if keyword in col_upper:
                    has_date_keyword = True
                    break
            if not has_date_keyword:
                filtered_key_cols.append(col)


        # ---- Strip leading zeros for each key column ----
        for col in filtered_key_cols:
            if col in df.columns:
                # Convert column to string, then strip leading zeros
                df[col] = df[col].astype(str).str.lstrip('0')

        # Print which table and columns were processed
        print(f"[OK] Stripped leading zeros in {tbl_name}: {filtered_key_cols}")

# Example usage in your notebook:
# strip_leading_zeros_all_tables(database_schema)


strip_leading_zeros_all_tables(database_schema, use_case)


# **Target value_help leaker replacement  (target = target_TEXT) in the dataframe**

# In[63]:


#this is to replace target columns with target_text as to avoid the target leakers
#this needs to be taken care before the merge is being done

def replace_target_with_text_in_namespace(
    use_case: str,                                           # The top-level key inside `database_schema`
    database_schema: Dict[str, Dict[str, Dict[str, Any]]],  # The full database schema dictionary
    target_columns: List[str],                               # List of target column names (without the "_TEXT" suffix)
    namespace: Optional[Mapping[str, Any]] = None            # Where DataFrames live; defaults to globals()
) -> Dict[str, pd.DataFrame]:
    """
    For each DataFrame referenced by `database_schema[use_case]`, if a target column `C`
    and its text twin `C_TEXT` both exist, drop `C` and rename `C_TEXT` -> `C` (keeping text values).
    Prints a per-table summary of which target columns were replaced.
    Returns a dict {table_name: dataframe} for all tables found in the namespace.
    """

    # Use the provided namespace (e.g., locals()) or default to the global namespace
    data_object_namespace = namespace if namespace is not None else globals()

    # Pull the schema for the given use case; this should contain keys like 'df_isdsalesdocitem', etc.
    schema_for_use_case = database_schema.get(use_case)

    # If the use case is not found in the schema, raise a clear error
    if schema_for_use_case is None:
        raise ValueError(f"Use case '{use_case}' not found in database_schema.")

    # Prepare a container to return the DataFrames we actually found in the namespace
    tables_found: Dict[str, pd.DataFrame] = {}

    # Iterate over each table name (keys like 'df_isdsalesdocitem') under the use case
    for table_name in schema_for_use_case.keys():
        # Fetch the object with the same name from the namespace
        table_object = data_object_namespace.get(table_name)

        # If the object does not exist or is not a pandas DataFrame, skip it
        if table_object is None or not isinstance(table_object, pd.DataFrame):
            continue  # Nothing to do for this table name

        # Keep a reference to the DataFrame (so we can also return it later)
        dataframe: pd.DataFrame = table_object
        tables_found[table_name] = dataframe  # Record that we found this table

        # Track which target columns we actually replaced for this table (for printing)
        replaced_targets_for_table: List[str] = []

        # For each requested target column, check if both the base and its "_TEXT" twin exist
        for target_column_name in target_columns:
            # Build the exact text-variant column name (always uppercase '_TEXT' per your spec)
            text_variant_name = f"{target_column_name}_TEXT"

            # Check presence of both columns in this DataFrame
            has_base = target_column_name in dataframe.columns
            has_text = text_variant_name in dataframe.columns

            # Only proceed if BOTH base and text columns exist
            if has_base and has_text:
                # Drop the original base column to avoid name collision during rename
                dataframe.drop(columns=[target_column_name], inplace=True, errors="ignore")

                # Rename '<target>_TEXT' to '<target>' so text values become the canonical target column
                dataframe.rename(columns={text_variant_name: target_column_name}, inplace=True)

                # Record that we replaced this target for this table
                replaced_targets_for_table.append(target_column_name)

            # If either is missing, we do nothing for this target in this table (leave columns as-is)

        # If we replaced any targets in this table, print a concise summary for visibility
        if replaced_targets_for_table:
            print(
                f"Table '{table_name}': replaced target columns "
                f"{replaced_targets_for_table} using their *_TEXT values."
            )


    # check after the for-loop over tables
    if not any(
        (f"{col}_TEXT" in df.columns) and (col in df.columns)
        for df in tables_found.values()
        for col in target_columns
    ):
      print("No replacements performed: no table had both a target column and its _TEXT twin.")


    # Return the dict of DataFrames we found (modified or not) for convenience
    return tables_found


# In[64]:


updated_tables = replace_target_with_text_in_namespace(
    use_case=use_case,
    database_schema=database_schema,
    target_columns=target_columns,
    # namespace=locals(),  # uncomment if your DataFrames are in local scope
)


# **Merges = Creating target columns for the entity table**

# In[65]:


database_schema


# In[66]:


if use_case == 'SaDP-PSLSDOCITEMPTD' or use_case == f'SaDP-PSLSDOCITEMPTD___{target_columns[0]}':

    df_pslsdocitemptd_subset=df_pslsdocitemptd[database_schema[use_case]['df_isdsalesdocitem']['pkey_col'] + [target['column'] for target in usecase_metadata["column_info"]["semantics"]['targets']]]

    df_merge=pd.merge(
    df_isdsalesdocitem,
    df_pslsdocitemptd_subset,         # only the time column (plus keys)
    on = database_schema[use_case]['df_isdsalesdocitem']['pkey_col'],       # join keys
    how='left'
    )
    
    df_isdsalesdocitem = df_merge

    entity_table='df_isdsalesdocitem'



elif use_case == 'SaDP-PDELIVPROCGPTD' or use_case == f'SaDP-PDELIVPROCGPTD___{target_columns[0]}':

    df_pdelivprocgptd1_subset=df_pdelivprocgptd1[database_schema[use_case]['df_isdsalesdocitem']['pkey_col'] + [target['column'] for target in usecase_metadata["column_info"]["semantics"]['targets']]]

    
    df_merge=pd.merge(
    df_isdsalesdocitem,
    df_pdelivprocgptd1_subset,         # only the time column (plus keys)
    on = database_schema[use_case]['df_isdsalesdocitem']['pkey_col'],       # join keys
    how='left'
    )

    df_isdsalesdocitem = df_merge

    entity_table='df_isdsalesdocitem'



elif use_case == 'IOR' or use_case == 'IOR_ATS' or use_case == f'IOR___{target_columns[0]}' or use_case == f'IOR_ATS___{target_columns[0]}':


    df_bseg_subset=df_bseg[database_schema[use_case]['df_bkpf']['pkey_col'] + target_columns]


    df_merge=pd.merge(
    df_bkpf,
    df_bseg_subset,         # only the time column (plus keys)
    on = database_schema[use_case]['df_bkpf']['pkey_col'],       # join keys
    how='left'
    )

    df_bkpf = df_merge

    entity_table='df_bkpf'



elif (use_case == 'SOAC' or use_case == f'SOAC___{target_columns[0]}') and data_area=='gssl':

    df_salesdocument_subset=df_salesdocument[[database_schema[use_case]['df_salesdocument']['pkey_col']] + [tar for tar in target_columns if tar in df_salesdocument.columns]]

    df_merge=pd.merge(
    df_salesdocumentitem,
    df_salesdocument_subset,         # only the time column (plus keys)
    on = database_schema[use_case]['df_salesdocument']['pkey_col'],       # join keys
    how='left'
    )

    df_salesdocumentitem = df_merge

    entity_table='df_salesdocumentitem'



elif (use_case == 'SOAC' or use_case == f'SOAC___{target_columns[0]}') and data_area == 'gmul':

    
    df_salesdocumentitem = df_salesdocumentitem.drop(columns=['SALESGROUP','SHIPPINGCONDITION','SALESOFFICE','CUSTOMERPAYMENTTERMS'])
    
    
    df_salesdocument_subset=df_salesdocument[[database_schema[use_case]['df_salesdocument']['pkey_col']] + [tar for tar in target_columns if tar in df_salesdocument.columns]+['ISDSALESDOC_CREATIONDATE']]



    df_merge=pd.merge(
    df_salesdocumentitem,
    df_salesdocument_subset,         # only the time column (plus keys)
    on = database_schema[use_case]['df_salesdocument']['pkey_col'],       # join keys
    how='left'
    )

    df_salesdocumentitem = df_merge

    entity_table = 'df_salesdocumentitem'


    database_schema={'SOAC': {'df_salesdocument': {'pkey_col': 'SALESDOCUMENT_df_salesdocument',
            'fkey_col_to_pkey_table': {},
            'time_col': None},
        'df_salesdocumentitem': {'pkey_col': ['SALESDOCUMENTITEM_df_salesdocumentitem',
            'SALESDOCUMENT_df_salesdocumentitem'],
            'fkey_col_to_pkey_table': {'SALESDOCUMENT_df_salesdocument': 'df_salesdocument',
                'KUNNR_df_customer': 'df_customer'},
            'time_col':'ISDSALESDOC_CREATIONDATE'},
        'df_customer': {'pkey_col': 'KUNNR_df_customer',
            'fkey_col_to_pkey_table': {'ADDRESSID_df_address': 'df_address'},
            'time_col': None},
        'df_address': {'pkey_col': 'ADDRESSID_df_address',
            'fkey_col_to_pkey_table': {},
            'time_col': None}}}



elif use_case == 'PDP' or use_case == f'PDP___{target_columns[0]}':

    entity_table = 'df_ifiopacctgdocit'



#for the automatic entity table selection
else:
   
# this function is used for single targets in different tables 

    # def choose_entity_table_for_single_target(  # comment: main function name
    #     database_schema: Dict[str, Any],  # comment: full schema dict {use_case: {table: meta}}
    #     use_case: str,  # comment: which use case key to use, e.g. "SOAC"
    #     target_col: str,  # comment: SINGLE target column (this is what you want)
    #     namespace: Dict[str, Any],  # comment: where df_* live (globals(), locals(), or merged dict)
    #     time_col_override: Optional[str] = None,  # comment: user-provided time column name if schema time_col is None
    # ) -> Dict[str, Any]:  # comment: returns entity_table, time_col, and updated schema
    #     """
    #     Purpose (Single Target)
    #     ----------------------
    #     1) Find entity_table = the df_* DataFrame in `namespace` that contains `target_col`.
    #     2) Get entity_time_col = database_schema[use_case][entity_table]["time_col"].
    #     3) If entity_time_col is None:
    #         - require time_col_override
    #         - update schema for that entity_table time_col = time_col_override

    #     Returns
    #     -------
    #     {
    #     "use_case": ...,
    #     "target_col": ...,
    #     "target_task": ...,
    #     "entity_table": ...,
    #     "entity_time_col": ...,
    #     "updated_schema": ...,
    #     "schema_was_updated": ...
    #     }
    #     """

    #     # ----------------------------
    #     # 1) Validate inputs
    #     # ----------------------------

        
    #     if not isinstance(namespace, dict):  # comment: namespace must be dict-like (globals(), locals(), etc.)
    #         raise ValueError("namespace must be a dict (example: globals(), locals(), or merged)")  # comment: fail clearly

    #     use_case_schema = database_schema[use_case]  # comment: grab the schema for this use case

      

    #     # ----------------------------
    #     # 2) Find entity table for this target_col
    #     # ----------------------------

    #     candidate_tables = []  # comment: tables where df.columns contains target_col

    #     for table_name in use_case_schema.keys():  # comment: scan tables in schema order
    #         if table_name not in namespace:  # comment: df_* not present in namespace -> skip
    #             continue  # comment: skip safely

    #         df_obj = namespace[table_name]  # comment: fetch object by name

    #         if not hasattr(df_obj, "columns"):  # comment: must look like a DataFrame
    #             continue  # comment: skip non-dataframe objects

    #         if target_col in list(df_obj.columns):  # comment: target column exists in this df
    #             candidate_tables.append(table_name)  # comment: record candidate entity table

    #     if len(candidate_tables) == 0:  # comment: no table had the target
    #         raise ValueError(  # comment: error message
    #             f"Target column '{target_col}' not found in any DataFrame among: {list(use_case_schema.keys())}"
    #         )

    #     # comment: If multiple tables contain the target, prefer one whose schema has time_col != None
    #     entity_table = candidate_tables[0]  # comment: default to first found
       
    #     # ----------------------------
    #     # 3) Resolve entity time column (schema or override)
    #     # ----------------------------

    #     entity_meta = use_case_schema.get(entity_table, {})  # comment: metadata for chosen entity table
    #     entity_time_col = entity_meta.get("time_col", None)  # comment: time_col can be None
    #     schema_was_updated = False  # comment: track whether we update schema

    #     # comment: If schema time_col is None, user must supply override
    #     if entity_time_col is None:  # comment: scenario 2
    #         if time_col_override is None or len(str(time_col_override).strip()) == 0:  # comment: must have a valid override
    #             raise ValueError(  # comment: fail clearly
    #                 f"Schema time_col is None for entity_table='{entity_table}'. "
    #                 f"Provide time_col_override to update schema."
    #             )

    #         # ----------------------------
    #         # 4) Update schema (copy-on-write so we don't mutate original accidentally)
    #         # ----------------------------

    #         updated_schema = dict(database_schema)  # comment: shallow copy top-level
    #         updated_schema[use_case] = dict(updated_schema[use_case])  # comment: copy use_case dict
    #         updated_schema[use_case][entity_table] = dict(updated_schema[use_case][entity_table])  # comment: copy table meta
    #         updated_schema[use_case][entity_table]["time_col"] = time_col_override  # comment: write override into schema

    #         entity_time_col = time_col_override  # comment: resolved time col is now the override
    #         schema_was_updated = True  # comment: record schema update

    #     else:
    #         # comment: scenario 1 (schema already had time_col)
    #         updated_schema = database_schema  # comment: no need to change schema

    #     # ----------------------------
    #     # 5) Return outputs
    #     # ----------------------------

    #     return {  # comment: return a structured output
    #         "use_case": use_case,  # comment: same use case
    #         "target_col": target_col,  # comment: the target you asked for
    #         "target_task": target_task,  # comment: optional (passed through)
    #         "entity_table": entity_table,  # comment: df_* that contains the target_col
    #         "entity_time_col": entity_time_col,  # comment: schema time_col or user override
    #         "updated_schema": updated_schema,  # comment: updated if needed, else original schema
    #         "schema_was_updated": schema_was_updated,  # comment: True only when schema time_col was None and we used override
    # }
    
    # merged_namespace = {**globals(), **locals()}  # typical notebook pattern

    # out = choose_entity_table_for_single_target(
    #     database_schema=database_schema,
    #     use_case=use_case,
    #     target_col=target_col, #fetch this as well
    #     namespace=merged_namespace,
    #     time_col_override=None,  # not needed if schema already has time_col , else give the fetched time_col from the user
    # )

    # entity_table = out["entity_table"]
    

    def auto_select_entity_table(
    target_columns,
    database_schema,
    use_case,
    local_namespace=None,
    global_namespace=None,
    ):
        """
        Automatically select the entity table (table name string) for a task based on target columns.

        Purpose:
            This function finds which dataframe table (e.g., 'df_salesdocument', 'df_salesdocumentitem', etc.)
            contains ALL the given target columns in a single table, and also has a valid time column defined
            in the schema. This is useful for automatic entity table selection in task setup.

        Input:
            target_columns (list[str]):
                List of target column names that must all exist in one single dataframe.
                Example:
                    ['SALESGROUP', 'PLANT', 'SHIPPINGPOINT', ...]

            database_schema (dict):
                Full schema dictionary that contains use-case-level schema information.
                Example format:
                    {
                        'SOAC': {
                            'df_salesdocument': {...},
                            'df_salesdocumentitem': {...},
                            ...
                        }
                    }

            use_case (str):
                The use-case key used to enter the schema dictionary.
                Example:
                    'SOAC'

            local_namespace (dict, optional):
                Usually pass `locals()` from the calling scope.
                Used to find actual dataframe variables by name (e.g., 'df_salesdocumentitem').

            global_namespace (dict, optional):
                Usually pass `globals()` from the calling scope.
                Used as a fallback if the dataframe is not found in local_namespace.

        Output:
            str or None:
                - Returns the entity table name string (e.g., 'df_salesdocumentitem') if exactly one valid table
                is found that contains all target columns and has a non-null time column in schema.
                - Returns None if:
                    * use_case is not present in database_schema
                    * dataframe variables are missing in namespace
                    * no table contains all target columns
                    * multiple tables satisfy the condition (ambiguous selection)
                    * the matching table has no valid time column
        """

        # If caller did not pass locals(), initialize an empty dictionary for safety.
        if local_namespace is None:
            local_namespace = {}

        # If caller did not pass globals(), initialize an empty dictionary for safety.
        if global_namespace is None:
            global_namespace = {}

        # Validate that the requested use case exists inside the schema dictionary.
        if use_case not in database_schema:
            # If the use case is missing, we cannot proceed, so return None.
            return None

        # Enter the schema for the provided use case (e.g., database_schema['SOAC']).
        use_case_schema = database_schema[use_case]

        # Initialize a list to store candidate table names that satisfy all conditions.
        valid_entity_table_candidates = []

        # Loop through each table defined under the selected use case schema.
        for table_name, table_metadata in use_case_schema.items():

            # Skip anything that does not look like a dataframe namespace key (defensive check).
            if not isinstance(table_name, str) or not table_name.startswith("df_"):
                # Continue to the next schema entry if this is not a dataframe-like table key.
                continue

            # Try to find the actual dataframe object in local namespace first.
            dataframe_object = local_namespace.get(table_name)

            # If not found in locals, try to find it in globals.
            if dataframe_object is None:
                dataframe_object = global_namespace.get(table_name)

            # If the dataframe object is still not found, we cannot inspect its columns, so skip it.
            if dataframe_object is None:
                continue

            # Ensure the object looks like a pandas DataFrame by checking for a "columns" attribute.
            if not hasattr(dataframe_object, "columns"):
                # Skip non-DataFrame objects even if the variable name matches.
                continue

            # Read the time column name from schema metadata for this table.
            time_column_name = table_metadata.get("time_col")

            # If the schema does not define a valid time column for this table, skip it.
            if time_column_name is None or time_column_name == "":
                continue

            
            if '__row_order__' not in dataframe_object.columns:
                continue
            
            # Convert dataframe columns to a set for fast subset checking.
            dataframe_columns_set = set(dataframe_object.columns)

            # Convert target columns to a set for subset checking.
            target_columns_set = set(target_columns)

            # Check whether ALL target columns are present in this dataframe.
            all_targets_exist_in_this_table = target_columns_set.issubset(dataframe_columns_set)

            # If all targets are present, this table is a valid candidate.
            if all_targets_exist_in_this_table:
                valid_entity_table_candidates.append(table_name)

        # If exactly one valid candidate exists, return that table name as the entity table.
        if len(valid_entity_table_candidates) == 1:
            return valid_entity_table_candidates[0]

        # If zero candidates or more than one candidate exist, return None to avoid wrong selection.
        return None


    
    entity_table = auto_select_entity_table(
    target_columns=target_columns,
    database_schema=database_schema,
    use_case=use_case,
    local_namespace=locals(),
    global_namespace=globals(),
    )




# In[67]:


use_case


# In[68]:


print(entity_table)


# In[73]:


target_columns


# In[74]:


database_schema


# In[75]:


for table_name, table_metadata in database_schema[use_case].items():
    if not table_name.startswith("df_"):
        continue

    df_obj = locals().get(table_name)
    if df_obj is None:
        df_obj = globals().get(table_name)

    print("\nTable:", table_name)
    print("  found_df:", df_obj is not None)
    print("  time_col:", table_metadata.get("time_col"))

    if df_obj is not None and hasattr(df_obj, "columns"):
        missing_targets = set(target_columns) - set(df_obj.columns)
        print("  missing_targets_count:", len(missing_targets))
        if missing_targets:
            print("  missing_targets_sample:", list(sorted(missing_targets))[:10])


# In[76]:


target_columns


# In[77]:


tenant_metadata


# In[78]:


entity_table


# **This function converts tenant_metadata to SOAC format and also auto selects the time_col or row_splits as the split range column based on the entity table**

# In[79]:


tenant_metadata


# In[80]:


#convert tenant metadata to soac metadata format



#   Dynamic Behavior:
#     - Scans the namespace for DataFrames associated with the tables in the metadata (e.g., 'df_qmel').
#     - If any of these DataFrames contain a '__row_order__' column, the function extracts the 
#       integer 'row_splits' to slice the data.
#     - If the '__row_order__' column is completely absent, it safely falls back to the conventional
#       datetime strings ('train', 'validation', 'test').



# # ------------------------- EXAMPLE USAGE -------------------------
# old_data = {
#   'tenant_id': '740401466',
#   'timesplits': {'QMEL': {'train': ['2017-10-06', '2024-06-10'],
#   'validation': ['2023-11-17', '2024-11-25'],
#   'test': ['2024-07-15', '2026-01-30'],
#   'row_splits': {'train': [0, 1153],
#    'validation': [1154, 1318],
#    'test': [1319, 1648]}},
# }

# result = convert_old_to_new(old_data)
# print(result)
# # Expected:
# # {
# #   "date_train_test_split": {
# #       "743367704": {
# #           "train": ["2023-11-01", "2024-06-01"],
# #           "validation": ["2024-06-01", "2024-07-31"],
# #           "test": ["2024-07-31", "2024-10-15"]
# #       }
# #   }
# # }






def convert_old_to_new_tenant_metadata(old_data: dict, entity_table_name: str, namespace: dict = None) -> dict:
    """
    Converts old-format metadata to new-format.
    CRITICAL FIX: It only checks the selected 'entity_table_name' to decide whether to use 
    integer row_splits or conventional datetimes, preventing global type-forcing collisions.
    """
    if namespace is None:
        namespace = globals()

    tenant_id = old_data["tenant_id"]
    timesplit_dict = old_data["timesplits"]

    # Match the entity_table (e.g., 'df_qmel') to the JSON key (e.g., 'QMEL')
    table_key = entity_table_name[3:].upper() if entity_table_name.startswith('df_') else entity_table_name.upper()
    if table_key not in timesplit_dict:
        table_key = next(iter(timesplit_dict)) # Safe fallback

    table_splits = timesplit_dict[table_key]
    df = namespace.get(entity_table_name)
    
    # Check if the TARGET table specifically has row order
    use_row_splits = False
    if df is not None and isinstance(df, pd.DataFrame) and '__row_order__' in df.columns:
        use_row_splits = True
        print(f"[METADATA] '__row_order__' found in entity table '{entity_table_name}'. Using INTEGER row_splits.")
    else:
        print(f"[METADATA] '__row_order__' NOT found in entity table '{entity_table_name}'. Using DATETIME format.")

    if use_row_splits and 'row_splits' in table_splits:
        train_vals, val_vals, test_vals = table_splits['row_splits']['train'], table_splits['row_splits']['validation'], table_splits['row_splits']['test']
    else:
        train_vals, val_vals, test_vals = table_splits['train'], table_splits['validation'], table_splits['test']

    return {
        "date_train_test_split": {
            tenant_id: {"train": train_vals, "validation": val_vals, "test": test_vals}
        }
    }

# CALL IT HERE, after entity_table is known!
tenant_metadata = convert_old_to_new_tenant_metadata(tenant_metadata, entity_table, namespace=globals())

tenant_metadata


# **Removing targets from non-entity table**
# 

# In[81]:


#removing targets from non-entity table

def delete_target_columns_from_non_entity_tables_in_namespace(
    use_case: str,                                  # Top-level key under database_schema (e.g., "SaDP-PDELIVPROCGPTD")
    database_schema: Dict[str, Dict],               # Full database schema dict you provided
    entity_table: str,                               # Name of the entity table (we do NOT remove from this one)
    target_columns: List[str],                       # Columns to remove from non-entity tables
    namespace: Optional[Mapping[str, object]] = None # Where to look up DataFrames by name; defaults to globals()
) -> Dict[str, List[str]]:
    """
    Remove `target_columns` from all non-entity tables that are listed under `database_schema[use_case]`.
    - Resolves each table's DataFrame directly from the provided `namespace` (or `globals()` if None).
    - Skips the entity table entirely.
    - Prints which tables were modified and which columns were removed.
    - Returns {table_name: [removed_columns]} for programmatic inspection.
    """

    # Use the provided namespace if given; otherwise fall back to the global namespace.
    lookup_namespace = namespace if namespace is not None else globals()

    # Grab the schema slice for this use case (e.g., database_schema["SaDP-PDELIVPROCGPTD"]).
    schema_for_use_case = database_schema.get(use_case, {})

    # If the use case is missing from the schema, fail fast with a clear message.
    if not schema_for_use_case:
        raise ValueError(f"Use case '{use_case}' not found in database_schema.")

 

    # Iterate over each table name defined under this use case in the schema.
    for table_name in schema_for_use_case.keys():
        # Skip the entity table completely (never remove target columns from it).
        if table_name == entity_table:
            continue

        # Pull the DataFrame object from the namespace using the table name as the variable name.
        dataframe = lookup_namespace.get(table_name)

        # If the DataFrame isn't found in the namespace, skip quietly (nothing to remove).
        if dataframe is None:
            continue

        # Determine which target columns actually exist in this DataFrame.
        columns_present = [col for col in target_columns if col in dataframe.columns]

        # If none of the target columns are present, there's nothing to do for this table.
        if not columns_present:
            continue

        # Drop the present target columns in place. `errors="ignore"` avoids exceptions if a race condition occurs.
        dataframe.drop(columns=columns_present, inplace=True, errors="ignore")

        # Record and report the change.
        
        print(f"Removed {columns_present} from non-entity table '{table_name}'.")

    # Return the summary mapping of tables to removed columns.
    return None


# In[82]:


removed_summary = delete_target_columns_from_non_entity_tables_in_namespace(
    use_case=use_case,                 # e.g., "SaDP-PDELIVPROCGPTD"
    database_schema=database_schema,   # your schema dict
    entity_table=entity_table,         # e.g., "df_isdsalesdocitem"
    target_columns=target_columns      # e.g., ["DELIVPROCGDELAYINDAYS", "DELIVPROCGDELAYCLASS"]
    # namespace defaults to globals(); pass a custom dict if needed
)


# **Value_help Leakers handling at dataframe level**

# In[83]:


# this is just an extra checker, target_text wont be present at all as it has been taken care and replaed with its value-help column values before the merge was done

# target_text leaker column removal if any from all the dataframes



def delete_target_text_columns_in_namespace(
    use_case: str,                                  # Top-level key in database_schema (e.g., "SaDP-PDELIVPROCGPTD")
    database_schema: Dict[str, Dict],               # Full schema dict you provided
    target_columns: List[str],                      # Base target columns (e.g., ["DELIVPROCGDELAYINDAYS", "DELIVPROCGDELAYCLASS"])
    namespace: Optional[Mapping[str, object]] = None# Where DataFrames are stored; defaults to globals()
) -> List[Tuple[str, List[str]]]:
    """
    For each table under `database_schema[use_case]`, drop columns named exactly `<target_column>_TEXT`.
    Prints which tables were modified and which columns were removed.
    Returns a list of (table_name, removed_columns). Returns [] if nothing was removed.
    """

    lookup_namespace = namespace if namespace is not None else globals()  # Use provided namespace or globals
    schema_for_use_case = database_schema.get(use_case, {})               # Get the sub-schema for the use case
    if not schema_for_use_case:                                           # If the use case is missing, fail clearly
        raise ValueError(f"Use case '{use_case}' not found in database_schema.")

    # Build the exact *_TEXT column names to search for (case-sensitive, as requested)
    exact_text_columns = [f"{col.strip()}_TEXT" for col in target_columns]# Create exact names like "FOO_TEXT"

    tables_with_removed_columns: List[Tuple[str, List[str]]] = []         # Collect (table, removed_cols) results

    for table_name in schema_for_use_case.keys():                         # Iterate each declared table in the use case
        dataframe = lookup_namespace.get(table_name)                      # Fetch the DataFrame by variable name
        if dataframe is None:                                             # If not found in the namespace, skip quietly
            continue

        columns_to_drop = [c for c in exact_text_columns if c in dataframe.columns]  # Check which *_TEXT columns exist
        if not columns_to_drop:                                           # If none exist for this table, move on
            continue

        dataframe.drop(columns=columns_to_drop, inplace=True, errors="ignore")       # Drop them in place
        tables_with_removed_columns.append((table_name, columns_to_drop))            # Record what we removed
        print(f"Removed columns {columns_to_drop} from table '{table_name}'.")       # Visibility for the user

    return tables_with_removed_columns                                     # Return summary (empty list if nothing removed)



# In[84]:


removed = delete_target_text_columns_in_namespace(
    use_case=use_case,
    database_schema=database_schema,
    target_columns= target_columns  # removes exactly "..._TEXT"
    
)



# In[85]:


#removing col_text columns and replacing original col with col_text values (value help leaker removal)

# not removing pkey and fkey for pkey_text and fkey_text and even same holds for the time_text


def replace_base_columns_with_text_versions_except_keys(
    use_case: str,
    database_schema: Dict[str, Dict[str, dict]],
    namespace: Optional[dict] = None,
) -> Dict[str, pd.DataFrame]:
    """
    For each DataFrame referenced by `database_schema[use_case]`:
      - If a column <col> and its paired <col>_TEXT both exist, replace <col> with <col>_TEXT
        by dropping <col> and renaming <col>_TEXT → <col>.
      - Do NOT perform this replacement for:
          * primary key columns (from `pkey_col`),
          * foreign key columns (keys of `fkey_col_to_pkey_table`),
          * the time column (from `time_col`).
      - Print each replacement as it happens.
    Returns a dict {dataframe_name: DataFrame} for the DataFrames found in the namespace.
    """

    # Use the provided namespace (e.g., locals()) or fallback to the global namespace.
    active_namespace = namespace if namespace is not None else globals()

    # Pull the per-table metadata for the requested use case; if missing, nothing to do.
    tables_metadata_for_use_case = database_schema.get(use_case, {})
    if not tables_metadata_for_use_case:
        print(f"No tables found for use_case='{use_case}' in the provided database_schema.")
        return {}

    # This will hold the DataFrame objects we successfully looked up (by variable name).
    dataframes_found: Dict[str, pd.DataFrame] = {}

    # Track whether anything was changed anywhere (optional, but nice for diagnostics).
    any_replacement_made = False

    # Walk every declared table (its key is the DataFrame variable name in your namespace).
    for dataframe_name, table_metadata in tables_metadata_for_use_case.items():
        # Only consider keys that represent DataFrames (you said they start with 'df_').
        if not isinstance(dataframe_name, str) or not dataframe_name.startswith("df_"):
            continue

        # Fetch the actual DataFrame object from the active namespace; skip if it's not there.
        dataframe_obj = active_namespace.get(dataframe_name)
        if dataframe_obj is None or not isinstance(dataframe_obj, pd.DataFrame):
            # If the variable is missing or not a DataFrame, move on silently.
            continue

        # Keep a reference so we can return it later.
        dataframes_found[dataframe_name] = dataframe_obj

        # ----- Build the "protected" column set we must NOT touch (pkeys, fkeys, time_col). -----

        # Normalize primary key(s) to a set (metadata can be a string or a list).
        primary_key_cols = table_metadata.get("pkey_col")
        if primary_key_cols is None:
            primary_key_set: Set[str] = set()
        elif isinstance(primary_key_cols, (list, tuple)):
            primary_key_set = set(primary_key_cols)
        else:
            primary_key_set = {str(primary_key_cols)}

        # Foreign key columns are the KEYS of the mapping {fkey_column_name: referenced_table}.
        foreign_key_map = table_metadata.get("fkey_col_to_pkey_table", {}) or {}
        foreign_key_set: Set[str] = set(foreign_key_map.keys())

        # Optional time column (single string or None).
        time_col = table_metadata.get("time_col")
        time_col_set: Set[str] = {time_col} if isinstance(time_col, str) else set()

        # Union of all protected columns we must never replace.
        protected_columns: Set[str] = primary_key_set | foreign_key_set | time_col_set

        # ----- Perform replacements for columns that have a paired *_TEXT, except protected. -----

        # Take a snapshot of current columns so we can safely modify the DataFrame while iterating.
        original_column_names = list(dataframe_obj.columns)

        # Track if we changed this particular DataFrame (optional, used for neat printing).
        replacement_made_in_this_df = False

        # Examine each base column name once (ignore columns that are already *_TEXT).
        for base_column in original_column_names:
            # Skip columns that are *_TEXT themselves; we evaluate pairs from the base name.
            if str(base_column).endswith("_TEXT"):
                continue

            # Skip if this base column is protected (primary key, foreign key, or time column).
            if base_column in protected_columns:
                continue

            # Compute the paired *_TEXT column name exactly as specified (uppercase TEXT).
            text_column = f"{base_column}_TEXT"

            # Only act if BOTH the base column and its *_TEXT twin exist right now.
            if base_column in dataframe_obj.columns and text_column in dataframe_obj.columns:
                # Drop the base column to prevent duplication/leakage.
                dataframe_obj.drop(columns=[base_column], inplace=True)
                # Rename the *_TEXT column to the original base column name.
                dataframe_obj.rename(columns={text_column: base_column}, inplace=True)

                # Print a precise audit line and mark that we changed something.
                print(
                    f"[{dataframe_name}] Promoted '{text_column}' → '{base_column}' "
                    f"(dropped original '{base_column}')."
                )
                replacement_made_in_this_df = True
                any_replacement_made = True

        # Optional: if you want a per-DF summary when nothing changed, uncomment the next two lines.
        # if not replacement_made_in_this_df:
        #     print(f"[{dataframe_name}] No eligible *_TEXT promotions (or only protected columns).")

    # Optional: global summary when absolutely nothing changed anywhere.
    if not any_replacement_made:
        print("No replacements performed in any table for this use case.")

    # Return the DataFrames we found (changed or not), keyed by their names.
    return dataframes_found





# In[86]:


remove_text_features = replace_base_columns_with_text_versions_except_keys(
    use_case=use_case,
    database_schema=database_schema,
    # namespace=locals(),  # optional; if omitted, it uses globals()
)


# **remove value help leakers in use_case_column_metdata and align with dataframe**

# In[87]:


use_case_column_metadata


# In[88]:


def infer_abap_type(values: Iterable[Any]) -> str:
    """
    Heuristically infer an ABAP type for a column from its observed values.

    Notes
    -----
    • Uses "majority" voting with a 0.9 threshold so a few bad rows won't flip the type.
    • Distinguishes NUMC vs INT by looking for leading zeros / string nature / fixed width.
    • Picks INT2/INT4/INT8 based on observed numeric range.
    • Detects common SAP textual codes: CUKY (currency keys), UNIT, LANG.
    • Detects DATS (several formats), TIMS, ACCP (YYYYMM posting period).
    • Falls back to "CHAR" when unsure.

    Returns one of: {"CHAR","NUMC","INT2","INT4","INT8","DEC","DATS","TIMS","ACCP",
                     "CUKY","UNIT","LANG"}
    (If you only want INT, replace INT2/4/8 with INT in the return mapping.)
    """

    # ---------------------------------------------------------
    # HELPER FUNCTIONS
    # ---------------------------------------------------------
    
    def strip_nones(seq: Iterable[Any]) -> List[Any]:
        # Removes physical `None` and floating point `NaN` values from the dataset
        return [x for x in seq if x is not None and (not (isinstance(x, float) and math.isnan(x)))]

    def share(condition_mask: List[bool]) -> float:
        # Calculates the percentage (from 0.0 to 1.0) of True values in a boolean list.
        # This is used to check if a condition meets the 90% (0.90) threshold.
        return (sum(1 for x in condition_mask if x) / len(condition_mask)) if condition_mask else 0.0

    def looks_numeric_str(s: str) -> bool:
        # Checks if a string contains ONLY digits (with an optional +/- sign at the front)
        return bool(re.fullmatch(r"[+-]?\d+", s))

    def looks_floatish(s: str) -> bool:
        # Checks if a string looks like a decimal/float (e.g., "3.14", "-0.5", "1e-4")
        return bool(re.fullmatch(r"[+-]?\d+([.]\d+)?([eE][+-]?\d+)?", s))

    def parse_int_or_none(x: Any) -> Optional[int]:
        # Safely attempts to convert a value to an integer. Returns None if it fails.
        try:
            if isinstance(x, str) and looks_numeric_str(x):
                return int(x)
            if isinstance(x, (int,)):
                return int(x)
        except Exception:
            return None
        return None

    # ---------------------------------------------------------
    # REGEX PATTERNS FOR SAP TYPES
    # ---------------------------------------------------------
    
    # DATS patterns: ISO, compact, dotted, slashed (YYYY-MM-DD / YYYYMMDD / DD.MM.YYYY / YYYY/MM/DD)
    # We added an optional time component group ( \d{2}:\d{2}:\d{2})?$ to catch string datetimes!
    dats_regexes = [
        re.compile(r"\d{4}-\d{2}-\d{2}( \d{2}:\d{2}:\d{2}(\.\d+)?)?$"),  # e.g., 2024-08-30 OR 2024-08-30 00:00:00
        re.compile(r"\d{8}$"),                                           # e.g., 20240830
        re.compile(r"\d{2}[.]\d{2}[.]\d{4}$"),                           # e.g., 30.08.2024
        re.compile(r"\d{4}/\d{2}/\d{2}$"),                               # e.g., 2024/08/30
    ]

    # TIMS patterns (HH:MM:SS or HHMMSS)
    tims_regexes = [
        re.compile(r"\d{2}:\d{2}:\d{2}$"),        # e.g., 13:45:59
        re.compile(r"\d{6}$"),                    # e.g., 134559
    ]

    # ACCP (accounting period): YYYYMM (MM can be 01..16 in SAP)
    accp_regex = re.compile(r"^(?P<y>\d{4})(?P<m>\d{2})$")

    # ---------------------------------------------------------
    # CHECKER FUNCTIONS
    # ---------------------------------------------------------
    # These return True if a given string matches any of the compiled regex patterns above.

    def is_dats(s: str) -> bool:
        return any(rx.fullmatch(s) for rx in dats_regexes)

    def is_tims(s: str) -> bool:
        return any(rx.fullmatch(s) for rx in tims_regexes)

    def is_accp(s: str) -> bool:
        m = accp_regex.fullmatch(s)
        if not m:
            return False
        mm = int(m.group("m"))
        return 1 <= mm <= 16  # SAP accounting months can go up to 16 (includes special periods)

    def is_cuky(s: str) -> bool:
        # Currency code check: Exactly 3 uppercase letters (e.g., USD, EUR, JPY)
        return bool(re.fullmatch(r"[A-Z]{3}", s))

    def is_lang(s: str) -> bool:
        # Language code check: 1 to 2 uppercase letters (e.g., 'E', 'EN', 'DE')
        return bool(re.fullmatch(r"[A-Z]{1,2}", s))

    def is_unit(s: str) -> bool:
        # Unit of Measure check: 1 to 5 uppercase letters (e.g., KG, EA, PC, M2)
        return bool(re.fullmatch(r"[A-Z]{1,5}", s))

    # ---------------------------------------------------------
    # MAIN INFERENCE LOGIC
    # ---------------------------------------------------------
    
    # Clean the raw input by dropping nulls/NaNs
    vals = strip_nones(values)
    
    # If the column is completely empty, default to "CHAR" as a safe fallback
    if not vals:
        return "CHAR"  

    # --- NATIVE DATETIME CHECK ---
    # Fast path: If the column already natively contains pandas Timestamps or Python datetimes,
    # immediately return "DATS" without running slow string conversions or regex loops.
    # We only check the first 10 values for performance.
    if any(isinstance(v, (pd.Timestamp, datetime)) for v in vals[:10]): 
        return "DATS"

    # Convert all clean values to strings for the regex and pattern detectors
    as_str = [str(v) for v in vals]

    # --- Strong pattern candidates (dates/times/periods) ---
    # Check if >= 90% of the column matches date formats
    if share([is_dats(s) for s in as_str]) >= 0.90:
        return "DATS"

    # Check if >= 90% of the column matches time formats
    if share([is_tims(s) for s in as_str]) >= 0.90:
        return "TIMS"

    # Check if >= 90% of the column matches SAP accounting periods
    if share([is_accp(s) for s in as_str]) >= 0.90:
        return "ACCP"

    # --- Numeric vs numeric-text (NUMC) ---
    # Check if >= 90% of the column consists entirely of digits
    is_digit = [s.isdigit() for s in as_str]
    if share(is_digit) >= 0.90:
        
        # Calculate the string length of all the items that are pure digits
        lengths = [len(s) for s, d in zip(as_str, is_digit) if d]
        
        # Determine if any numbers have a leading zero (e.g., "00123")
        has_leading_zero = any(s.startswith("0") and len(s) > 1 for s, d in zip(as_str, is_digit) if d)
        
        # Determine if every single number in the column has the exact same character length
        fixed_width = (max(lengths) == min(lengths)) if lengths else False

        # Special SAP case: If every number is exactly 3 digits long, it's highly likely a Client ID (MANDT)
        if all(l == 3 for l in lengths):
            return "CLNT"  

        # If numbers have leading zeros or are strictly fixed-width, they are identifiers, not math numbers (NUMC)
        if has_leading_zero or fixed_width:
            return "NUMC"

        # If it's pure math integers, figure out how much memory it needs by finding the min/max values
        ints = [parse_int_or_none(v) for v in vals]
        ints = [x for x in ints if x is not None]
        if ints:
            min_v, max_v = min(ints), max(ints)
            if -32768 <= min_v <= max_v <= 32767:
                return "INT2" # Fits in 16-bit integer
            if -2147483648 <= min_v <= max_v <= 2147483647:
                return "INT4" # Fits in 32-bit integer
            return "INT8"     # Huge numbers, needs 64-bit integer
            
        return "INT4" # Default fallback for integers

    # --- Float / Decimal amounts (DEC) ---
    # Determine if >= 90% of the column looks like a float or decimal value
    is_floatish = []
    for v in vals:
        if isinstance(v, (float,)):
            is_floatish.append(True)
        elif isinstance(v, str) and looks_floatish(v):
            is_floatish.append(True)
        else:
            is_floatish.append(False)
            
    if share(is_floatish) >= 0.90:
        return "DEC"   

    # --- Small, code-like uppercase tokens ---
    # Check if >= 90% of the column matches 3-letter currency codes
    if share([is_cuky(s) for s in as_str]) >= 0.90:
        return "CUKY"

    # Check if >= 90% of the column matches standard unit of measure abbreviations
    if share([is_unit(s) for s in as_str]) >= 0.90:
        return "UNIT"

    # Check if >= 90% of the column matches short language codes
    if share([is_lang(s) for s in as_str]) >= 0.90:
        return "LANG"

    # --- Default Fallback ---
    # If it failed the 90% threshold for all the specific checks above, it's just general text.
    return "CHAR"


# In[89]:


#replace the original column and its abap type if its _text version is present and after replacement remove the _text suffix

#also align the dataframe columns exactly with the use_case_column metadata and infer the abap type if there is something missing


def remove_value_help_leaker_from_use_case_column_metadata_only(
    use_case_column_metadata: Dict[str, Any],
    namespace: Optional[Dict[str, Any]] = None,
    text_suffix: str = "_TEXT",
    default_abap_type: str = "CHAR",
) -> Dict[str, Any]:
    """
    Update the *metadata only* (do NOT touch DataFrames):
      1) For each df_* entry, if both <col> and <col>_TEXT exist in the metadata,
         prefer the TEXT entry: drop <col>, keep <col>_TEXT's type under the key <col>.
      2) Re-align the metadata to the actual DataFrame columns in the namespace:
           - Keep only columns present in the DataFrame.
           - Add any DataFrame columns missing from metadata with ABAP type default_abap_type.
           - Preserve DataFrame column order.
      3) Return a new metadata dict in the exact same shape you provided.

    Parameters
    ----------
    use_case_column_metadata : dict
        Mapping like: { "df_name": [ { "COL_A": "CHAR", ... } ], ... }.
    namespace : dict, optional
        Namespace to fetch DataFrames by variable name (e.g., globals()).
        If None, globals() is used.
    text_suffix : str, optional
        Exact suffix for value-help columns; default "_TEXT" (always uppercase per your spec).
    default_abap_type : str, optional
        ABAP type to use when a column is missing in metadata; default "CHAR".

    Returns
    -------
    dict
        New metadata dict with the same outer shape and per-table inner shape.
    """

    # Use the provided namespace if given; otherwise fall back to globals().
    active_namespace: Dict[str, Any] = namespace if namespace is not None else globals()

    # Prepare a new dict to collect the reconciled (final) metadata.
    reconciled_metadata: Dict[str, Any] = {}

    # Loop over each table listed in the incoming metadata.
    for dataframe_name, table_metadata_list in use_case_column_metadata.items():

        # Extract the inner {column_name: abap_type, ...} from your shape: [ { ... } ].
        # Support a direct dict as well, but your format uses a one-item list.
        if isinstance(table_metadata_list, list) and table_metadata_list:
            original_col_to_type: Dict[str, str] = dict(table_metadata_list[0])
        elif isinstance(table_metadata_list, dict):
            original_col_to_type = dict(table_metadata_list)
        else:
            # If unexpected/empty, treat as no existing metadata for this table.
            original_col_to_type = {}

        # -------- Stage 1: resolve TEXT pairs INSIDE METADATA ONLY --------
        #
        # Build a working copy we can mutate without touching the input object.
        working_col_to_type: Dict[str, str] = dict(original_col_to_type)

        # Collect all base columns (keys that do NOT end with the TEXT suffix).
        base_columns_in_metadata = [
            col_name for col_name in working_col_to_type.keys()
            if not col_name.endswith(text_suffix)
        ]

        # For each base column, check if its TEXT partner exists *in metadata*.
        for base_column_name in base_columns_in_metadata:
            text_column_name = f"{base_column_name}{text_suffix}"

            # If both base and TEXT exist in metadata, prefer the TEXT entry:
            #   - Drop the base column's entry,
            #   - Keep the TEXT column's type but map it to the base column name,
            #   - Remove the TEXT key (since we "renamed" it logically to the base).
            if text_column_name in working_col_to_type:
                # Fetch the ABAP type from the TEXT entry.
                text_abap_type = working_col_to_type[text_column_name]

                # Remove the base entry (we prefer the TEXT semantics).
                if base_column_name in working_col_to_type:
                    del working_col_to_type[base_column_name]

                # Remove the TEXT entry, we'll replace it under the base name.
                del working_col_to_type[text_column_name]

                # Insert back under the base column name with the TEXT ABAP type.
                working_col_to_type[base_column_name] = text_abap_type

        # -------- Stage 2: re-align metadata to the actual DataFrame --------
        #
        # Try to fetch the real DataFrame from the namespace.
        dataframe_obj = active_namespace.get(dataframe_name, None)

        # If no DataFrame is found, we cannot align; keep the metadata we just computed.
        if dataframe_obj is None:
            reconciled_metadata[dataframe_name] = [working_col_to_type]
            continue

        # Build a new column->type dict that matches the DataFrame's column order exactly.
        final_col_to_type: Dict[str, str] = {}

        # Keep only columns present in the DataFrame, and add any missing with default type.
        for dataframe_column in list(dataframe_obj.columns):
            
            # If our working metadata already has an entry for this column, use it.
            if dataframe_column in working_col_to_type:
                final_col_to_type[dataframe_column] = working_col_to_type[dataframe_column]
            
            else:
                # Otherwise, add it with the default ABAP type. to add targets in entity table after the merge
                series = dataframe_obj[dataframe_column]
                sample_vals = series.dropna().astype(str).tolist()
                inferred = infer_abap_type(sample_vals) if sample_vals else default_abap_type

                final_col_to_type[dataframe_column] = inferred
        
        
        # Save back using the SAME shape you provided: a single-item list holding the dict.
        reconciled_metadata[dataframe_name] = [final_col_to_type]

    # Return the fully reconciled metadata dictionary (DataFrames untouched).
    return reconciled_metadata


# In[90]:


# just pass your current metadata; DataFrames remain untouched
new_column_metadata = remove_value_help_leaker_from_use_case_column_metadata_only(
    use_case_column_metadata
    # , namespace=globals()  # optional; defaults to globals()
)

use_case_column_metadata=new_column_metadata

use_case_column_metadata


# **Target column masking**
# 
#     If task == "regression":
#           # - Treat native nulls, empty/whitespace strings, and common null-like tokens as missing.
#           # - Write real missing values (np.nan for numeric columns; pd.NA for others).
# 
#     If task == "classification":
#           # - Keep empty/whitespace strings as-is (don't treat them as missing).
#           # - Treat native nulls and common null-like tokens (excluding empty/whitespace) as missing.
#           # - Replace those with a sentinel -100 (dtype-aware: numeric -> -100; categorical -> -100 or "-100" based on categories; others -> -100)

# In[91]:


def sanitize_targets_by_task_missing_policy(
    use_case: str,                                       # use-case key inside `database_schema`
    database_schema: Dict[str, Dict[str, Any]],          # schema mapping use_case -> table specs
    target_columns: List[str],                           # list of target column names
    target_tasks: List[str],                             # parallel list of tasks ("regression"/"classification")
    namespace: Optional[Dict[str, Any]] = None,          # where DataFrames live (e.g., globals())
) -> Dict[str, Dict[str, int]]:                          # {table_name: {column_name: num_changes}}
    """
    For each table in `database_schema[use_case]`, if a target column is present:

      • If task == "regression":
          - Treat native nulls, empty/whitespace strings, and common null-like tokens as missing.
          - Write real missing values (np.nan for numeric columns; pd.NA for others).

      • If task == "classification":
          - Keep empty/whitespace strings as-is (don't treat them as missing).
          - Treat native nulls and common null-like tokens (excluding empty/whitespace) as missing.
          - Replace those with a sentinel -100 (dtype-aware: numeric -> -100; categorical -> -100 or "-100" based on categories; others -> -100).

    DataFrames are modified in-place in `namespace` (defaults to globals()).
    A summary of replacements (counts per table/column) is returned.
    """

    # Use the provided namespace (e.g., globals() / locals()); default to globals() if none is passed.
    active_namespace: Dict[str, Any] = namespace if namespace is not None else globals()

    # Map each target column to its corresponding task (assumes parallel ordering).
    # If lengths differ, extra items are ignored.
    task_by_column: Dict[str, str] = {
        col: task for col, task in zip(target_columns, target_tasks)
    }

    # Common "null-like" tokens we will detect in strings (EXACT tokens; no case folding).
    # NOTE: We intentionally DO NOT include "" or " " here, because classification must keep empties.
    null_like_tokens = {
        "NA", "Na", "na",
        "N/A", "n/a", "Na/", "NA/", "na/",
        "NAN", "NaN", "nan",
        "NONE", "None", "none",
        "NULL", "Null", "null",
        "NIL", "Nil", "nil",
        "NILL", "Nill", "nill",
        "NAT", "Nat", "nat", "NaT",
        "-", "--",
    }

    # This will record how many values were changed per table and per column.
    changes_summary: Dict[str, Dict[str, int]] = {}

    # If the use_case key is missing, nothing to do—return an empty summary.
    if use_case not in database_schema:
        return changes_summary

    # Iterate over each declared table under this use_case.
    for dataframe_name, _table_spec in database_schema[use_case].items():

        # Only process names that are supposed to be DataFrames (by convention, they start with "df_").
        if not dataframe_name.startswith("df_"):
            continue

        # Fetch the actual DataFrame object from the active namespace.
        dataframe_obj = active_namespace.get(dataframe_name, None)

        # Skip safely if the variable is not a pandas DataFrame.
        if not isinstance(dataframe_obj, pd.DataFrame):
            continue

        # Track replacements made in this specific DataFrame by column.
        table_changes: Dict[str, int] = {}

        # Check each target column we know about.
        for target_col in task_by_column.keys():

            # Skip if this DataFrame does not contain the current target column.
            if target_col not in dataframe_obj.columns:
                continue

            # Identify the task for this target ("regression" or "classification").
            task_type = task_by_column[target_col]

            # Short alias to the Series for easier reading.
            col_series = dataframe_obj[target_col]

            # ---------- Build the per-row "should change" mask based on task rules ----------

            # 1) Native nulls (NaN/None/pd.NA/NaT) – always considered missing for BOTH tasks.
            native_missing = col_series.isna()

            # 2) String-based checks – evaluated only for string-like / object / categorical columns.
            #    We do no case folding; we test raw and stripped forms.
            if pdt.is_object_dtype(col_series) or pdt.is_string_dtype(col_series) or pdt.is_categorical_dtype(col_series):
                # For regression:
                #   - Treat empty/whitespace strings as missing.
                #   - Treat tokens in `null_like_tokens` as missing (raw or stripped).
                if task_type == "regression":
                    to_change = col_series.apply(
                        lambda x: (
                            pd.isna(x)  # native missing (NaN/None/pd.NA/NaT)
                            or (isinstance(x, str) and (x.strip() == ""))  # empty or whitespace-only
                            or (isinstance(x, str) and (x in null_like_tokens or x.strip() in null_like_tokens))  # null-like tokens
                        )
                    )



                # For classification:
                #   - KEEP empty/whitespace strings as-is (not missing).
                #   - Treat tokens in `null_like_tokens` as missing (raw or stripped).
                else:  # task_type == "classification"
                    to_change = col_series.apply(
                        lambda x: (
                            pd.isna(x)  # native missing (NaN/None/pd.NA/NaT)
                            or (
                                isinstance(x, str)
                                and (x in null_like_tokens or x.strip() in null_like_tokens)  # null-like tokens
                                # IMPORTANT: we intentionally DO NOT include `x.strip()==""` here
                            )
                        )
                    )
            else:
                # Non-string dtypes: the only missing values are the native ones.
                to_change = native_missing

            # Count how many rows will be changed for this column.
            num_to_change = int(to_change.sum())

            # If nothing is marked for change, skip to the next target column.
            if num_to_change == 0:
                continue

            # ---------- Apply the replacement based on task and dtype ----------
            if task_type == "regression":
                # For regression, write *real* missing values.
                # if pdt.is_numeric_dtype(col_series):
                    # Numeric columns: use np.nan (keeps numeric dtype).
                dataframe_obj.loc[to_change, target_col] = np.nan
                # elif pdt.is_categorical_dtype(col_series):
                #     # Categoricals: assign pd.NA (valid missing for categoricals).
                #     dataframe_obj.loc[to_change, target_col] = pd.NA
                # else:
                #     # Other dtypes (e.g., object/string): use pd.NA.
                #     dataframe_obj.loc[to_change, target_col] = pd.NA

            else:  # task_type == "classification"
                # For classification, replace with -100 sentinel, respecting dtype.
                # if pdt.is_categorical_dtype(col_series):
                #     # Choose sentinel type based on the categories' dtype.
                #     categories_dtype = col_series.dtype.categories.dtype
                #     sentinel_value: Any = -100 if pdt.is_numeric_dtype(categories_dtype) else "-100"
                #     # Ensure the sentinel is a valid category before assignment.
                #     if sentinel_value not in list(col_series.dtype.categories):
                #         dataframe_obj[target_col] = col_series.cat.add_categories([sentinel_value])
                #         col_series = dataframe_obj[target_col]  # refresh reference after category update
                #     dataframe_obj.loc[to_change, target_col] = sentinel_value
                # elif pdt.is_numeric_dtype(col_series):
                #     # Numeric columns: use numeric -100 directly.
                #     dataframe_obj.loc[to_change, target_col] = -100
                # else:
                    # Other dtypes (e.g., object/string): write numeric -100 (dtype may remain 'object').
                    dataframe_obj.loc[to_change, target_col] = -100
                
                # NEW: keep as true nulls instead
                #dataframe_obj.loc[to_change, target_col] = pd.NA #for classification no more -100
            
            # Record how many updates we performed for this column.
            table_changes[target_col] = num_to_change

        # If anything changed in this table, add it to the overall summary.
        if table_changes:
            changes_summary[dataframe_name] = table_changes

    # Return a concise audit of changes made across all tables/targets.
    return changes_summary


# In[92]:


summary = sanitize_targets_by_task_missing_policy(
    use_case,                                       # use-case key inside `database_schema`
    database_schema,          # schema mapping use_case -> table specs
    target_columns,                           # list of target column names
    target_tasks,                             # parallel list of tasks ("regression"/"classification")
    namespace = None,          # where DataFrames live (e.g., globals())
)



print(summary)


# **Delete the target column if everything is null and also update the target column list and target task list and in step-2 align the col_metadata with dataframes**

# In[93]:


# step 1 = Delete the target column if everything is null and also update the target column list and target task list



def prune_targets_if_all_null(
    database_schema: Dict[str, Dict[str, dict]],   # outer: {<use_case>: {... df_* ...}}
    target_columns: List[str],                      # e.g., ["DELIVERYDELAYINDAYS", "DELIVERYDELAYCLASS"]
    target_tasks: List[str],                       # e.g., ["regression", "classification"], same length/order
) -> Tuple[List[str], List[str]]:
    """
    For each (target_column, target_task):
      • For every dataframe that contains the column:
          - If the column is entirely null (NaN/None), drop it from that dataframe.
      • After that pass:
          - If the column still exists in at least one dataframe with any non-null value → keep the pair.
          - If it existed but was all-null in every dataframe → remove the pair from outputs.
          - If it never existed in any dataframe → keep the pair (conservative).

    Returns updated (target_columns, target_tasks) with the same order, minus any pairs fully pruned.
    """

    # --- 1) Resolve the inner schema (the one that lists df_* keys) -------------------------------
    # Pick the first top-level key (typical case: there is exactly one use-case block).
    if not database_schema:
        # Nothing to inspect → return inputs unchanged.
        return target_columns, target_tasks
    use_case_key = next(iter(database_schema))                 # get the first (and usually only) use-case key
    inner_schema = database_schema[use_case_key]               # this has 'df_*' entries

    # --- 2) Collect the dataframe variable names we should look for in globals() ------------------
    df_var_names = [k for k in inner_schema.keys()             # scan the inner schema keys
                    if isinstance(k, str) and k.startswith("df_")]  # keep only df_* names

    # --- 3) Build a map of df name -> actual pandas DataFrame objects present in globals() --------
    # We only include valid pandas DataFrames that exist; missing ones are silently skipped.
    available_dfs: Dict[str, pd.DataFrame] = {}
    for df_name in df_var_names:                                # iterate candidate dataframe names
        maybe_df = globals().get(df_name)                       # try to fetch variable from global namespace
        if isinstance(maybe_df, pd.DataFrame):                  # ensure it is indeed a pandas DataFrame
            available_dfs[df_name] = maybe_df                   # keep it if present and valid

    # --- 4) Prepare the output containers ---------------------------------------------------------
    updated_columns: List[str] = []                             # columns that survive pruning
    updated_tasks: List[str] = []                               # aligned tasks for surviving columns

    # --- 5) Process each (target_column, target_task) pair in order -------------------------------
    for col, task in zip(target_columns, target_tasks):
        # Track whether we saw the column in any dataframe at all.
        seen_anywhere = False                                    # becomes True if any df has this column
        # Track whether after dropping all-null columns, the column still exists somewhere with any non-null.
        remains_with_values_somewhere = False                    # becomes True if any df has non-null values

        # Check every available dataframe listed in the schema.
        for df_name, df in available_dfs.items():                # iterate df_name -> DataFrame
            if col in df.columns:                                # only act if this df contains the target column
                seen_anywhere = True                             # mark that we saw this column in at least one df
              
                print(f" {df_name}: column '{col}' present")
                # Determine if the entire column is null (NaN/None). Empty strings "" count as values.
                # pandas treats "" as a value, so isna() will NOT mark it as null (this matches your requirement).
                all_null = df[col].isna().all()                  # True only if every entry is NaN/None

                if all_null:
                    # Column is entirely null in this dataframe → drop it here.
                    df.drop(columns=[col], inplace=True, errors="ignore")
                    print(f"      → Action: DROPPED '{col}' from {df_name} (all values were null).")
                
                else:
                    # Column has at least one real (non-null) value here → it remains meaningful.
                    remains_with_values_somewhere = True
                    print(f"      → Action: KEPT '{col}' in {df_name} (has non-null values).")

        # Decide whether to keep or remove the (column, task) pair in the outputs:
        if seen_anywhere:
            # If column was present somewhere:
            if remains_with_values_somewhere:
                # At least one dataframe kept non-null values → keep the pair.
                updated_columns.append(col)
                updated_tasks.append(task)
            else:
                # Column existed but was all-null everywhere (thus dropped) → remove the pair.
                # (Do nothing: we simply do not append to outputs.)
                pass
        else:
            # Column not found in any dataframe → conservative choice: keep the pair.
            updated_columns.append(col)
            updated_tasks.append(task)

    # --- 6) Return the pruned lists ---------------------------------------------------------------
    return updated_columns, updated_tasks


# In[94]:


target_columns, target_tasks = prune_targets_if_all_null(
    database_schema = database_schema,
    target_columns = target_columns,
    target_tasks = target_tasks,
    )


# In[95]:


target_tasks


# In[96]:


# step 2 = align use_case_column_metadata with dataframes after removal of target with all NaN values in previous step

def align_use_case_column_metadata_with_dataframes(
    use_case_column_metadata: Dict[str, List[Dict[str, str]]],
    verbose: bool = True,
) -> Dict[str, List[Dict[str, str]]]:
    """
    Remove columns from use_case_column_metadata that are NOT present
    in the actual DataFrames (df_* variables in globals()).

    The shape expected is: { "df_xxx": [ { "COL_A": "ABAP_TYPE", ... } ], ... }

    Returns the updated metadata dict (in-place modification is also performed).
    """

    # Iterate over each df_* entry described in the metadata
    for df_var_name, metadata_list in use_case_column_metadata.items():
       

        # Fetch the dataframe object from the global namespace
        df_obj = globals().get(df_var_name)

        

        # If the list is empty, create a single empty dict to operate on
        if len(metadata_list) == 0:
            metadata_list.append({})

        # The first (and only) dict should contain column→ABAP_type mappings
        columns_meta = metadata_list[0]

      

        # Build a set of actual DataFrame columns (ground truth)
        df_columns = set(df_obj.columns)

        # Build a set of columns currently in the metadata for this DataFrame
        meta_columns = set(columns_meta.keys())

        # Compute the "extra" columns—present in metadata but NOT in the actual DataFrame
        extra_columns = sorted(meta_columns - df_columns)

        # If there are no extras, report and move on
        if not extra_columns:
            if verbose:
                print(f"  [OK] No extra columns to remove for '{df_var_name}'. Metadata matches DataFrame columns.")
            continue

        # Otherwise, remove each extra column from the metadata dict
        for extra_col in extra_columns:
            # Print what exactly we’re deleting for traceability
            if verbose:
                print(f"  [REMOVE] '{extra_col}' → removed from metadata of '{df_var_name}' (not in DataFrame).")
            # Delete the key safely (exists by construction)
            del columns_meta[extra_col]

        # After deletions, confirm completion for this DataFrame
        if verbose:
            print(f"  [DONE] Removed {len(extra_columns)} extra column(s) from '{df_var_name}'.")

    
    return use_case_column_metadata


# In[97]:


use_case_column_metadata = align_use_case_column_metadata_with_dataframes(
    use_case_column_metadata = use_case_column_metadata,
    verbose = True,
)


# **Building clean_dict for the soac_db**

# In[98]:


def build_clean_dict(
        schema: Dict[str, Dict[str, dict]],
        use_case: str = "PDP",
        namespace: Mapping[str, object] | None = None,
) -> Dict[str, pd.DataFrame]:
    """
    Auto-assemble {table_name: dataframe} for every table listed in
    `schema[use_case]`.

    Parameters
    ----------
    schema     : your `database_schema` dict
    use_case   : which top-level key to read inside `schema`
    namespace  : where to look for the DataFrame variables
                 • None  → caller’s *globals()*
                 • locals() inside a function
                 • any mapping that holds the DFs

    Returns
    -------
    clean_dict : dict mapping table names to DataFrames

    Raises
    ------
    KeyError if a table listed in the schema has no matching DataFrame variable.
    """
    # default: globals() of the *caller*, not this helper
    if namespace is None:
        namespace = inspect.currentframe().f_back.f_globals

    clean_dict: Dict[str, pd.DataFrame] = {}
    for tbl_name in schema[use_case]:
        try:
            clean_dict[tbl_name] = namespace[tbl_name]
        except KeyError as err:
            raise KeyError(
                f"Expected a DataFrame variable named '{tbl_name}' "
                "but it was not found in the supplied namespace."
            ) from err
    return clean_dict



clean_dict = build_clean_dict(database_schema, use_case)

clean_dict


# In[99]:


clean_dict.keys()


# In[100]:


entity_table


# In[101]:


def is_missing_or_none(val):
    if pd.isna(val):
        return True
    val_str = str(val).strip().lower()
    return val_str in ["", "none", "nan"]

mask = clean_dict[entity_table][target_columns].applymap(is_missing_or_none)
filtered =clean_dict[entity_table][target_columns][mask.any(axis=1)]

print(filtered)


# In[102]:


print(filtered)


# In[103]:


clean_dict[entity_table][target_columns]


# In[104]:


clean_dict[entity_table]


# In[105]:


def count_minus_100(df, target_cols):
    for col in target_cols:
        count = (df[col] == -100).sum()
        print(f"Column: {col} | Count of -100: {count}")
    print(df[target_columns].dtypes)

count_minus_100(clean_dict[entity_table], target_columns)


# **update database_schema by checking if pkey and fkey does exist in the dataframes or not, if it does not exist in dataframes then you update the schema accordingly**

# In[106]:


database_schema


# In[107]:


# Define the function, accepting the original schema and an optional namespace (defaults to None).
def clean_schema_against_namespace(database_schema: Dict[str, Any], namespace: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Validates a database schema against the actual Pandas DataFrames in memory.
    Removes missing tables, and prunes missing primary keys, foreign keys, and time columns.
    """
    
    # Check if the user provided a specific namespace (like locals()).
    if namespace is None:
        # If no namespace was provided, default to the global variables of the script.
        namespace = globals()

    # Create a deep copy of the input schema to serve as our working draft.
    # This prevents the function from mutating the user's original data during execution.
    updated_schema = deepcopy(database_schema)

    # Create a static list of the table names (keys) currently in the schema.
    # We must use list() because Python forbids deleting keys from a dictionary while actively iterating over its keys.
    for table_name in list(updated_schema[use_case].keys()):
        
        # Try to retrieve the actual object from the namespace using the table's name (e.g., 'df_salesdocument').
        actual_dataframe = namespace.get(table_name)

        # Verify whether the retrieved object is missing (None) or is not a valid pandas DataFrame.
        if actual_dataframe is None or not isinstance(actual_dataframe, pd.DataFrame):
            # If the DataFrame doesn't exist in memory, print an informative log message.
            print(f"[TABLE DROPPED] '{table_name}' was not found in the namespace. Removing from schema.")
            # Delete the entire table entry from our updated schema.
            del updated_schema[use_case][table_name]
            # Skip the rest of the loop for this table, as there are no columns to check.
            continue

        # Extract the metadata sub-dictionary for this specific table to update its properties.
        table_metadata = updated_schema[use_case][table_name]
        
        # Extract the columns of the actual DataFrame and convert them to a Python Set.
        # Sets provide ultra-fast O(1) lookups, making our column checks highly efficient.
        dataframe_columns = set(actual_dataframe.columns)

        # ---------------------------------------------------------
        # 1. Validate Primary Keys (pkey_col)
        # ---------------------------------------------------------
        
        # Retrieve the primary key definition from the metadata (it could be a string, a list, or None).
        primary_key = table_metadata.get('pkey_col')
        
        # Check if the primary key is defined as a single string.
        if isinstance(primary_key, str):
            # Check if this primary key column is missing from the actual DataFrame.
            if primary_key not in dataframe_columns:
                # Log that the primary key column is missing.
                print(f"[PKEY DROPPED] '{primary_key}' is missing from DataFrame '{table_name}'.")
                # Set the primary key in the schema to None to reflect reality.
                table_metadata['pkey_col'] = None
                
        # Check if the primary key is defined as a list (e.g., a composite key).
        elif isinstance(primary_key, list):
            # Use a list comprehension to filter and keep ONLY the keys that actually exist in the DataFrame columns.
            valid_pkeys = [pk for pk in primary_key if pk in dataframe_columns]
            
            # Identify which keys were dropped by subtracting the valid keys from the original keys.
            dropped_pkeys = set(primary_key) - set(valid_pkeys)
            
            # If any keys were dropped, log them for visibility.
            if dropped_pkeys:
                print(f"[PKEY DROPPED] Missing composite keys {list(dropped_pkeys)} removed from '{table_name}'.")
            
            # If the filtered list of valid keys is completely empty...
            if not valid_pkeys:
                # ...set the primary key property to None.
                table_metadata['pkey_col'] = None
            # If at least one key survived the filter...
            else:
                # ...overwrite the schema's primary key list with the surviving valid keys.
                table_metadata['pkey_col'] = valid_pkeys

        # ---------------------------------------------------------
        # 2. Validate Foreign Keys (fkey_col_to_pkey_table)
        # ---------------------------------------------------------
        
        # Retrieve the dictionary of foreign keys (e.g., {'KUNNR_df_customer': 'df_customer'}).
        # Default to an empty dictionary {} if it doesn't exist.
        foreign_keys_dict = table_metadata.get('fkey_col_to_pkey_table', {})
        
        # Ensure that the retrieved value is actually a dictionary before iterating.
        if isinstance(foreign_keys_dict, dict):
            
            # Create a static list of the foreign key column names so we can safely delete from the dict mid-loop.
            for fk_column_name in list(foreign_keys_dict.keys()):
                
                # Check if this foreign key column is missing from the actual DataFrame.
                if fk_column_name not in dataframe_columns:
                    # Log that the foreign key column is missing and its relationship is being severed.
                    print(f"[FKEY DROPPED] '{fk_column_name}' is missing from DataFrame '{table_name}'.")
                    # Delete the foreign key entry from the schema's relationship dictionary.
                    del foreign_keys_dict[fk_column_name]

        # ---------------------------------------------------------
        # 3. Validate Time Column (time_col)
        # ---------------------------------------------------------
        
        # Retrieve the designated time column from the metadata.
        time_column = table_metadata.get('time_col')
        
        # Check if a time column is defined (is a string) AND if it is missing from the actual DataFrame.
        if isinstance(time_column, str) and time_column not in dataframe_columns:
            # Log that the time column was not found.
            print(f"[TIME DROPPED] Time column '{time_column}' is missing from DataFrame '{table_name}'.")
            # Set the time column to None in the schema.
            table_metadata['time_col'] = None

    # Finally, return the fully cleaned and synchronized schema dictionary.
    return updated_schema


# In[108]:


use_case


# In[109]:


print("--- Starting Schema Cleanup ---")
database_schema = clean_schema_against_namespace(database_schema=database_schema, namespace=globals())
print("--- Schema Cleanup Complete ---")

database_schema


# **soac_db class**

# In[110]:


# # → compile patterns once for speed
# eight_digit_date_pattern = re.compile(r"\d{8}$")               # e.g. "20240803"
# ymd_separators_pattern   = re.compile(r"\d{4}[-/]\d{2}[-/]\d{2}$")  # e.g. "2024-08-03"

# def appears_to_be_date(column: pd.Series, min_share: float = 0.20) -> bool:
#     """
#     Decide whether `column` looks like dates.
#     • at least 30% (default) of values must match one of two regexes.

#     Example
#     -------
#     "2024-01-01", "foo", "20240102"  → 2/3 = 0.67 ≥ 0.30 → True
#     """
#     as_text = column.astype(str)                              # → ensure str ops work
#     match_mask = (as_text.str.match(eight_digit_date_pattern) |
#                   as_text.str.match(ymd_separators_pattern))
#     return match_mask.mean() >= min_share




# def normalise_to_yyyymmdd(column: pd.Series) -> pd.Series:
#     """
#     Convert many date notations to an `Int64` YYYYMMDD.
#     Non date rows become `<NA>`.

#     Example
#     -------
#     ["2024/08/03", "foo"] → [20240803, <NA>]
#     """
#     digits_only = column.astype(str).str.replace(r"\D", "", regex=True)
#     is_eight_digits = digits_only.str.len() == 8
#     output = pd.Series(pd.NA, index=column.index, dtype="Int64")
#     output[is_eight_digits] = (
#         pd.to_numeric(digits_only[is_eight_digits], errors="coerce")
#         .astype("Int64")
#     )
#     return output





# def share_of_non_na(array_like) -> float:
#     """Return fraction of elements that are not NA."""
#     return (~pd.isna(array_like)).mean()    







# class Database:
#     r"""A database is a collection of named tables linked by foreign key - primary key
#     connections."""

#     def __init__(self, table_dict: Dict[str, Table]) -> None:
#         r"""Creates a database from a dictionary of tables."""
#         self.table_dict = table_dict

#     def __repr__(self) -> str:
#         return f"{self.__class__.__name__}()"

#     def save(self, path: Union[str, os.PathLike]) -> None:
#         r"""Save the database to a directory."""
#         for name, table in self.table_dict.items():
#             table.save(f"{path}/{name}.parquet")

#     @classmethod
#     def load(cls, path: Union[str, os.PathLike]) -> Self:
#         r"""Load a database from a directory of tables in parquet files."""
#         table_dict = {}
#         for table_path in Path(path).glob("*.parquet"):
#             table = Table.load(table_path)
#             table_dict[table_path.stem] = table
#         return cls(table_dict)

#     @property
#     @lru_cache(maxsize=None)
#     def min_timestamp(self) -> pd.Timestamp:
#         r"""Return the earliest timestamp in the database."""
#         return min(
#             table.min_timestamp
#             for table in self.table_dict.values()
#             if table.time_col is not None
#         )

#     @property
#     @lru_cache(maxsize=None)
#     def max_timestamp(self) -> pd.Timestamp:
#         r"""Return the latest timestamp in the database."""
#         return max(
#             table.max_timestamp
#             for table in self.table_dict.values()
#             if table.time_col is not None
#         )

#     def upto(self, timestamp: pd.Timestamp) -> Self:
#         r"""Return a database with all rows upto timestamp."""
#         return Database(
#             table_dict={
#                 name: table.upto(timestamp) for name, table in self.table_dict.items()
#             }
#         )

#     def from_(self, timestamp: pd.Timestamp) -> Self:
#         r"""Return a database with all rows from timestamp."""
#         return Database(
#             table_dict={
#                 name: table.from_(timestamp) for name, table in self.table_dict.items()
#             }
#         )

    
#     def reindex_pkeys_and_fkeys(self) -> None:
#         """
#         Replace every primary key / foreign key value with contiguous Int64
#         indices, so RelBench can build a graph.

#         Strategy for each FK column
#         ---------------------------
#         1. direct merge                             (fast path)
#         2. if hit rate <5%   & looks numeric  → strip leading zeros
#         3. if hit rate still <5% & looks date  → normalise YYYYMMDD
#         """
#         min_hit_rate = 0.05                                      # → 5 % threshold
#         parent_pk_to_new_index = {}                              # → {parent: Series}

#         # ─── 1) construct mapping Series for every parent table ───
#         for parent_table_name, parent_table in self.table_dict.items():

#             if parent_table.pkey_col is None:                    # → skip if no PK
#                 continue

#             parent_old_pk_series = parent_table.df[parent_table.pkey_col]
#             #   e.g. ["000123", "000124", …]

#             parent_new_index_values = pd.RangeIndex(
#                 len(parent_old_pk_series)
#             ).astype("Int64")                                    # → [0,1,2,…]

#             parent_pk_to_new_index[parent_table_name] = pd.Series(
#                 parent_new_index_values, index=parent_old_pk_series, name="index"
#             )
#             #   index   "000123"  →  value 0
#             #   index   "000124"  →  value 1

#             parent_table.df[parent_table.pkey_col] = parent_new_index_values #dictionary of parent table original values to index numeric values
#             #   PK column now holds 0,1,2,…

#         # ─── 2) map every foreign key in every child table ───
#         for child_table in self.table_dict.values():

#             # build {parent_table_name: [fk_col1, fk_col2, …]}
#             parent_to_fk_list = defaultdict(list)
#             for fk_column, parent_name in child_table.fkey_col_to_pkey_table.items():
#                 parent_to_fk_list[parent_name].append(fk_column)

#             for parent_name, child_fk_columns in parent_to_fk_list.items():
#                 parent_table     = self.table_dict[parent_name]
#                 parent_pk_column = parent_table.pkey_col             # str or composite
#                 parent_lookup_series = parent_pk_to_new_index[parent_name]

#                 # ───── A) parent has *composite* string PK (contains '-') 
#                 if isinstance(parent_pk_column, str) and "-" in parent_pk_column:
#                     pk_parts = parent_pk_column.split("-")           # e.g. ["SALESDOC","ITEM"]

#                     # 1 raw join: concat the FK parts exactly like the parent PK
#                     fk_concat_raw = (
#                         child_table.df[pk_parts].astype(str).agg("-".join, axis=1)
#                     )                                                # "000123-10"
#                     mapped_indices = (
#                         parent_lookup_series.reindex(fk_concat_raw)  # lookup
#                         .astype("Int64")                             # keep NA support
#                         .values
#                     )

#                     # 2️ leading‑zero fix if few matches
#                     if share_of_non_na(mapped_indices) < min_hit_rate:
#                         fk_concat_nozeros = (
#                             child_table.df[pk_parts]
#                             .astype(str)
#                             .apply(lambda col: col.str.lstrip("0"))  # col‑wise "000123"→"123"
#                             .agg("-".join, axis=1)                   # "123-10"
#                         )
#                         lookup_nozeros = parent_lookup_series.copy()
#                         lookup_nozeros.index = (                     # strip zeros in *index* too
#                             lookup_nozeros.index.str.split("-")
#                             .map(lambda tokens: "-".join(t.lstrip("0") for t in tokens))
#                         )
#                         zero_fix_indices = (
#                             lookup_nozeros.reindex(fk_concat_nozeros)
#                             .astype("Int64")
#                             .values
#                         )
#                         if share_of_non_na(zero_fix_indices) > share_of_non_na(mapped_indices):
#                             mapped_indices = zero_fix_indices        # keep better result

#                     # 3️ date fix if still poor & parts look like dates
#                     if share_of_non_na(mapped_indices) < min_hit_rate and any(
#                             appears_to_be_date(child_table.df[col]) for col in pk_parts):

#                         fk_concat_dates = (
#                             child_table.df[pk_parts]
#                             .applymap(lambda cell: re.sub(r"\D", "", str(cell)))
#                             .agg("-".join, axis=1)                   # "20240803-99"
#                         )
#                         lookup_dates = parent_lookup_series.copy()
#                         lookup_dates.index = (
#                             lookup_dates.index.str.split("-")
#                             .map(lambda tokens: "-".join(re.sub(r"\D", "", t) for t in tokens))
#                         )
#                         date_fix_indices = (
#                             lookup_dates.reindex(fk_concat_dates)
#                             .astype("Int64")
#                             .values
#                         )
#                         if share_of_non_na(date_fix_indices) > share_of_non_na(mapped_indices):
#                             mapped_indices = date_fix_indices

#                     # write final Int64 array into *each* FK part
#                     for part_name in pk_parts:
#                         child_table.df[part_name] = mapped_indices
#                     continue                                          # next parent/child pair

#                 # B) normal single‑column foreign keys ─────
#                 for fk_col in child_fk_columns:
#                     foreign_key_series = child_table.df[fk_col]

#                     # 1️ direct merge if dtypes already align
#                     if foreign_key_series.dtype == parent_lookup_series.index.dtype:
#                         merged_result = (
#                             pd.merge(child_table.df[[fk_col]],
#                                     parent_lookup_series,
#                                     left_on=fk_col, right_index=True,
#                                     how="left")["index"]
#                             .astype("Int64")
#                         )
#                         child_table.df[fk_col] = merged_result
#                         continue

#                     # 2️ generic string mapping
#                     lookup_as_str = parent_lookup_series.copy()
#                     lookup_as_str.index = lookup_as_str.index.astype(str)
#                     mapped_indices = (
#                         foreign_key_series.astype(str)
#                         .map(lookup_as_str)
#                         .astype("Int64")
#                     )

#                     # 3️ date fix if still poor & looks like dates
#                     if share_of_non_na(mapped_indices) < min_hit_rate and (
#                             appears_to_be_date(foreign_key_series) or
#                             appears_to_be_date(parent_lookup_series.index.to_series())):

#                         foreign_key_dates = normalise_to_yyyymmdd(foreign_key_series)
#                         parent_pk_dates   = normalise_to_yyyymmdd(
#                             parent_lookup_series.index.to_series()
#                         )
#                         date_lookup_series = pd.Series(
#                             parent_lookup_series.values,              # 0,1,2,…
#                             index=parent_pk_dates                     # 20240803, …
#                         )
#                         date_fix_indices = (
#                             foreign_key_dates.map(date_lookup_series)
#                             .astype("Int64")
#                         )
#                         if share_of_non_na(date_fix_indices) > share_of_non_na(mapped_indices):
#                             mapped_indices = date_fix_indices

#                     child_table.df[fk_col] = mapped_indices          # final Int64 indices
            


# In[111]:


# # → compile patterns once for speed
# eight_digit_date_pattern = re.compile(r"\d{8}$")               # e.g. "20240803"
# ymd_separators_pattern   = re.compile(r"\d{4}[-/]\d{2}[-/]\d{2}$")  # e.g. "2024-08-03"

# def appears_to_be_date(column: pd.Series, min_share: float = 0.20) -> bool:
#     """
#     Decide whether `column` looks like dates.
#     • at least 30% (default) of values must match one of two regexes.

#     Example
#     -------
#     "2024-01-01", "foo", "20240102"  → 2/3 = 0.67 ≥ 0.30 → True
#     """
#     as_text = column.astype(str)                              # → ensure str ops work
#     match_mask = (as_text.str.match(eight_digit_date_pattern) |
#                   as_text.str.match(ymd_separators_pattern))
#     return match_mask.mean() >= min_share




# def normalise_to_yyyymmdd(column: pd.Series) -> pd.Series:
#     """
#     Convert many date notations to an `Int64` YYYYMMDD.
#     Non date rows become `<NA>`.

#     Example
#     -------
#     ["2024/08/03", "foo"] → [20240803, <NA>]
#     """
#     digits_only = column.astype(str).str.replace(r"\D", "", regex=True)
#     is_eight_digits = digits_only.str.len() == 8
#     output = pd.Series(pd.NA, index=column.index, dtype="Int64")
#     output[is_eight_digits] = (
#         pd.to_numeric(digits_only[is_eight_digits], errors="coerce")
#         .astype("Int64")
#     )
#     return output





# def share_of_non_na(array_like) -> float:
#     """Return fraction of elements that are not NA."""
#     return (~pd.isna(array_like)).mean()    







# class Database:
#     r"""A database is a collection of named tables linked by foreign key - primary key
#     connections."""

#     def __init__(self, table_dict: Dict[str, Table]) -> None:
#         r"""Creates a database from a dictionary of tables."""
#         self.table_dict = table_dict

#     def __repr__(self) -> str:
#         return f"{self.__class__.__name__}()"

#     def save(self, path: Union[str, os.PathLike]) -> None:
#         r"""Save the database to a directory."""
#         for name, table in self.table_dict.items():
#             table.save(f"{path}/{name}.parquet")

#     @classmethod
#     def load(cls, path: Union[str, os.PathLike]) -> Self:
#         r"""Load a database from a directory of tables in parquet files."""
#         table_dict = {}
#         for table_path in Path(path).glob("*.parquet"):
#             table = Table.load(table_path)
#             table_dict[table_path.stem] = table
#         return cls(table_dict)

#     @property
#     @lru_cache(maxsize=None)
#     def min_timestamp(self) -> pd.Timestamp:
#         r"""Return the earliest timestamp in the database."""
#         return min(
#             table.min_timestamp
#             for table in self.table_dict.values()
#             if table.time_col is not None
#         )

#     @property
#     @lru_cache(maxsize=None)
#     def max_timestamp(self) -> pd.Timestamp:
#         r"""Return the latest timestamp in the database."""
#         return max(
#             table.max_timestamp
#             for table in self.table_dict.values()
#             if table.time_col is not None
#         )

#     def upto(self, timestamp: pd.Timestamp) -> Self:
#         r"""Return a database with all rows upto timestamp."""
#         return Database(
#             table_dict={
#                 name: table.upto(timestamp) for name, table in self.table_dict.items()
#             }
#         )

#     def from_(self, timestamp: pd.Timestamp) -> Self:
#         r"""Return a database with all rows from timestamp."""
#         return Database(
#             table_dict={
#                 name: table.from_(timestamp) for name, table in self.table_dict.items()
#             }
#         )

    
#     def reindex_pkeys_and_fkeys(self) -> None:
#         """
#         Replace every primary key / foreign key value with contiguous Int64 indices, so RelBench can build a graph.

#         Added functionality (NEW):
#         --------------------------
#         - Supports SUBSET foreign keys to a composite parent primary key.
#         - If the subset uniquely identifies a parent row -> map to that single index.
#         - If ambiguous (subset matches multiple parent rows) -> set FK to <NA> (no row explosion).

#         Existing functionality retained:
#         -------------------------------
#         - Exact composite PK matching by concatenated "A-B-C" strings + optional leading-zero and date fixes.
#         - Single-column FK mapping by exact match + optional date fix.
#         """

#         # ----------------------------
#         # threshold: if mapping hits <5% we try additional cleanup strategies
#         # ----------------------------
#         min_hit_rate = 0.05  # comment: minimum acceptable fraction of mapped rows

#         # ----------------------------
#         # mapping storage: parent_table_name -> Series(index=old_pk_value, value=new_int_id)
#         # ----------------------------
#         parent_pk_to_new_index = {}  # comment: lookup series for each parent table

#         # ----------------------------
#         # extra storage: for composite PK parents, store split parts (from the composite string) for subset matching
#         # ----------------------------
#         parent_composite_parts_cache = {}  # comment: caches split PK parts for subset matching

#         # ──────────────────────────────────────────────────────────────────────
#         # STEP 1) Build reindex mapping for every parent table
#         # ──────────────────────────────────────────────────────────────────────
#         for parent_table_name, parent_table in self.table_dict.items():  # comment: loop over all tables

#             # comment: skip tables without primary key
#             if parent_table.pkey_col is None:
#                 continue

#             # comment: grab PK column name (possibly composite like "A-B-C")
#             parent_pk_col_name = parent_table.pkey_col

#             # comment: store old PK values (mapping FROM these)
#             parent_old_pk_series = parent_table.df[parent_pk_col_name]

#             # comment: create new contiguous Int64 IDs [0..N-1]
#             parent_new_index_values = pd.RangeIndex(len(parent_old_pk_series)).astype("Int64")

#             # comment: store lookup: old_pk_value -> new_int_id
#             parent_pk_to_new_index[parent_table_name] = pd.Series(
#                 parent_new_index_values,
#                 index=parent_old_pk_series,
#                 name="index",
#             )

#             # comment: overwrite parent PK column with new contiguous IDs
#             parent_table.df[parent_pk_col_name] = parent_new_index_values

#             # comment: cache split parts if composite PK column
#             if isinstance(parent_pk_col_name, str) and "-" in parent_pk_col_name:
#                 pk_parts = parent_pk_col_name.split("-")  # comment: PK part column names

#                 # comment: split old PK values into columns aligned with pk_parts
#                 parts_df = parent_old_pk_series.astype(str).str.split("-", n=len(pk_parts) - 1, expand=True)

#                 # comment: pad if split produced fewer columns than expected
#                 if parts_df.shape[1] < len(pk_parts):
#                     for _ in range(len(pk_parts) - parts_df.shape[1]):
#                         parts_df[parts_df.shape[1]] = pd.NA

#                 # comment: name the columns by pk_parts
#                 parts_df.columns = pk_parts

#                 # comment: store parent IDs aligned with parts_df rows
#                 id_series = pd.Series(parent_new_index_values, index=parts_df.index)

#                 # comment: cache these for subset mapping
#                 parent_composite_parts_cache[parent_table_name] = {
#                     "pk_parts": pk_parts,
#                     "parts_df": parts_df,
#                     "id_series": id_series,
#                 }

#         # ──────────────────────────────────────────────────────────────────────
#         # helper: fraction of non-NA entries
#         # ──────────────────────────────────────────────────────────────────────
#         def _share_of_non_na(array_like) -> float:
#             return (~pd.isna(array_like)).mean()

#         # ──────────────────────────────────────────────────────────────────────
#         # helper: build composite key string from df[cols] joined by '-'
#         # ──────────────────────────────────────────────────────────────────────
#         def _concat_cols_as_key(df: pd.DataFrame, cols: list[str]) -> pd.Series:
#             return df[cols].astype(str).agg("-".join, axis=1)

#         # ──────────────────────────────────────────────────────────────────────
#         # helper: build dict subset_key -> list[parent_ids]
#         # ──────────────────────────────────────────────────────────────────────
#         def _build_subset_to_ids_lookup(parent_name: str, subset_cols_in_order: list[str]) -> dict:
#             cache = parent_composite_parts_cache[parent_name]  # comment: get cached split PK info
#             parent_parts_df = cache["parts_df"]                # comment: split PK values dataframe
#             parent_ids = cache["id_series"]                    # comment: parent IDs aligned by row

#             parent_subset_keys = parent_parts_df[subset_cols_in_order].astype(str).agg("-".join, axis=1)

#             tmp = pd.DataFrame({"subset_key": parent_subset_keys, "pid": parent_ids.values})
#             grouped = tmp.groupby("subset_key")["pid"].apply(list)

#             return grouped.to_dict()

#         # ──────────────────────────────────────────────────────────────────────
#         # STEP 2) Map every FK in every child table
#         # ──────────────────────────────────────────────────────────────────────
#         for child_table in self.table_dict.values():  # comment: each table is a potential child

#             parent_to_fk_list = defaultdict(list)  # comment: parent -> list of fk columns in this child
#             for fk_column, parent_name in child_table.fkey_col_to_pkey_table.items():
#                 parent_to_fk_list[parent_name].append(fk_column)

#             for parent_name, child_fk_columns in parent_to_fk_list.items():

#                 parent_table = self.table_dict[parent_name]
#                 parent_pk_column = parent_table.pkey_col
#                 parent_lookup_series = parent_pk_to_new_index[parent_name]

#                 # ──────────────────────────────────────────────────────────────
#                 # CASE A) Parent has composite PK column (contains '-')
#                 # ──────────────────────────────────────────────────────────────
#                 if isinstance(parent_pk_column, str) and "-" in parent_pk_column:

#                     pk_parts = parent_pk_column.split("-")
#                     child_has_parts = [c for c in pk_parts if c in child_table.df.columns]
#                     has_all_parts = len(child_has_parts) == len(pk_parts)

#                     # ─────────────────────────────────────────────────────────
#                     # A1) Exact composite mapping (UNCHANGED)
#                     # ─────────────────────────────────────────────────────────
#                     if has_all_parts:

#                         fk_concat_raw = _concat_cols_as_key(child_table.df, pk_parts)

#                         mapped_indices = (
#                             parent_lookup_series.reindex(fk_concat_raw)
#                             .astype("Int64")
#                             .values
#                         )

#                         if _share_of_non_na(mapped_indices) < min_hit_rate:
#                             fk_concat_nozeros = (
#                                 child_table.df[pk_parts]
#                                 .astype(str)
#                                 .apply(lambda col: col.str.lstrip("0"))
#                                 .agg("-".join, axis=1)
#                             )
#                             lookup_nozeros = parent_lookup_series.copy()
#                             lookup_nozeros.index = (
#                                 lookup_nozeros.index.astype(str)
#                                 .str.split("-")
#                                 .map(lambda tokens: "-".join(t.lstrip("0") for t in tokens))
#                             )
#                             zero_fix_indices = (
#                                 lookup_nozeros.reindex(fk_concat_nozeros)
#                                 .astype("Int64")
#                                 .values
#                             )
#                             if _share_of_non_na(zero_fix_indices) > _share_of_non_na(mapped_indices):
#                                 mapped_indices = zero_fix_indices

#                         if _share_of_non_na(mapped_indices) < min_hit_rate and any(
#                             appears_to_be_date(child_table.df[col]) for col in pk_parts
#                         ):
#                             fk_concat_dates = (
#                                 child_table.df[pk_parts]
#                                 .applymap(lambda cell: re.sub(r"\D", "", str(cell)))
#                                 .agg("-".join, axis=1)
#                             )
#                             lookup_dates = parent_lookup_series.copy()
#                             lookup_dates.index = (
#                                 lookup_dates.index.astype(str)
#                                 .str.split("-")
#                                 .map(lambda tokens: "-".join(re.sub(r"\D", "", t) for t in tokens))
#                             )
#                             date_fix_indices = (
#                                 lookup_dates.reindex(fk_concat_dates)
#                                 .astype("Int64")
#                                 .values
#                             )
#                             if _share_of_non_na(date_fix_indices) > _share_of_non_na(mapped_indices):
#                                 mapped_indices = date_fix_indices

#                         for part_name in pk_parts:
#                             child_table.df[part_name] = mapped_indices

#                         continue

#                     # ─────────────────────────────────────────────────────────
#                     # A2) SUBSET composite mapping (CHANGED: unique-only else NA)
#                     # ─────────────────────────────────────────────────────────
#                     # comment: pick subset columns that exist in the child and are part of the parent's pk_parts
#                     subset_cols = [c for c in pk_parts if c in child_fk_columns and c in child_table.df.columns]

#                     # comment: if child has no usable subset columns, we cannot map -> set NA
#                     if len(subset_cols) == 0:
#                         for fk_col in child_fk_columns:
#                             if fk_col in child_table.df.columns:
#                                 child_table.df[fk_col] = pd.Series(pd.NA, index=child_table.df.index, dtype="Int64")
#                         continue

#                     # comment: build subset_key -> [parent_ids] lookup from the parent
#                     subset_to_ids = _build_subset_to_ids_lookup(parent_name, subset_cols)

#                     # comment: build child subset keys in the same subset_cols order
#                     child_subset_keys = _concat_cols_as_key(child_table.df, subset_cols)

#                     # comment: map each key to list of parent ids; missing -> []
#                     mapped_lists = child_subset_keys.map(subset_to_ids)
#                     mapped_lists = mapped_lists.apply(lambda x: x if isinstance(x, list) else [])

#                     # comment: keep ONLY unique matches; ambiguous (len!=1) becomes NA
#                     unique_or_na = mapped_lists.apply(lambda ids: ids[0] if len(ids) == 1 else pd.NA)

#                     # comment: convert to Int64 nullable
#                     unique_or_na = pd.to_numeric(unique_or_na, errors="coerce").astype("Int64")

#                     # comment: write this result into each FK column for this relationship (no explode!)
#                     for fk_col in child_fk_columns:
#                         if fk_col in child_table.df.columns:
#                             child_table.df[fk_col] = unique_or_na

#                     continue

#                 # ──────────────────────────────────────────────────────────────
#                 # CASE B) Parent has single-column PK (UNCHANGED)
#                 # ──────────────────────────────────────────────────────────────
#                 for fk_col in child_fk_columns:

#                     if fk_col not in child_table.df.columns:
#                         continue

#                     foreign_key_series = child_table.df[fk_col]

#                     if foreign_key_series.dtype == parent_lookup_series.index.dtype:
#                         merged_result = (
#                             pd.merge(
#                                 child_table.df[[fk_col]],
#                                 parent_lookup_series,
#                                 left_on=fk_col,
#                                 right_index=True,
#                                 how="left",
#                             )["index"]
#                             .astype("Int64")
#                         )
#                         child_table.df[fk_col] = merged_result
#                         continue

#                     lookup_as_str = parent_lookup_series.copy()
#                     lookup_as_str.index = lookup_as_str.index.astype(str)

#                     mapped_indices = (
#                         foreign_key_series.astype(str)
#                         .map(lookup_as_str)
#                         .astype("Int64")
#                     )

#                     if _share_of_non_na(mapped_indices) < min_hit_rate and (
#                         appears_to_be_date(foreign_key_series)
#                         or appears_to_be_date(parent_lookup_series.index.to_series())
#                     ):
#                         foreign_key_dates = normalise_to_yyyymmdd(foreign_key_series)
#                         parent_pk_dates = normalise_to_yyyymmdd(parent_lookup_series.index.to_series())

#                         date_lookup_series = pd.Series(
#                             parent_lookup_series.values,
#                             index=parent_pk_dates,
#                         )

#                         date_fix_indices = (
#                             foreign_key_dates.map(date_lookup_series)
#                             .astype("Int64")
#                         )

#                         if _share_of_non_na(date_fix_indices) > _share_of_non_na(mapped_indices):
#                             mapped_indices = date_fix_indices

#                     child_table.df[fk_col] = mapped_indices


# In[112]:


# Reindexing functions

#→ compile patterns once for speed
eight_digit_date_pattern = re.compile(r"\d{8}$")               # e.g. "20240803"
ymd_separators_pattern   = re.compile(r"\d{4}[-/]\d{2}[-/]\d{2}$")  # e.g. "2024-08-03"


#checks the column if it is a date or not

def appears_to_be_date(column: pd.Series, min_share: float = 0.20) -> bool:
    """
    Decide whether `column` looks like dates.
    • at least 30% (default) of values must match one of two regexes.

    Example
    -------
    "2024-01-01", "foo", "20240102"  → 2/3 = 0.67 ≥ 0.30 → True
    """
    as_text = column.astype(str)                              # → ensure str ops work
    match_mask = (as_text.str.match(eight_digit_date_pattern) |
                  as_text.str.match(ymd_separators_pattern))
    return match_mask.mean() >= min_share



#normalizes the date column

def normalise_to_yyyymmdd(column: pd.Series) -> pd.Series:
    """
    Convert many date notations to an `Int64` YYYYMMDD.
    Non date rows become `<NA>`.

    Example
    -------
    ["2024/08/03", "foo"] → [20240803, <NA>]
    """
    digits_only = column.astype(str).str.replace(r"\D", "", regex=True)
    is_eight_digits = digits_only.str.len() == 8
    output = pd.Series(pd.NA, index=column.index, dtype="Int64")
    output[is_eight_digits] = (
        pd.to_numeric(digits_only[is_eight_digits], errors="coerce")
        .astype("Int64")
    )
    return output



#returns the fraction 

def share_of_non_na(array_like) -> float:
    """Return fraction of elements that are not NA."""
    return (~pd.isna(array_like)).mean()    







class Database:
    r"""A database is a collection of named tables linked by foreign key - primary key
    connections."""

    def __init__(self, table_dict: Dict[str, Table]) -> None:
        r"""Creates a database from a dictionary of tables."""
        self.table_dict = table_dict

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    def save(self, path: Union[str, os.PathLike]) -> None:
        r"""Save the database to a directory."""
        for name, table in self.table_dict.items():
            table.save(f"{path}/{name}.parquet")

    @classmethod
    def load(cls, path: Union[str, os.PathLike]) -> Self:
        r"""Load a database from a directory of tables in parquet files."""
        table_dict = {}
        for table_path in Path(path).glob("*.parquet"):
            table = Table.load(table_path)
            table_dict[table_path.stem] = table
        return cls(table_dict)

    @property
    @lru_cache(maxsize=None)
    def min_timestamp(self) -> pd.Timestamp:
        r"""Return the earliest timestamp in the database."""
        return min(
            table.min_timestamp
            for table in self.table_dict.values()
            if table.time_col is not None
        )

    @property
    @lru_cache(maxsize=None)
    def max_timestamp(self) -> pd.Timestamp:
        r"""Return the latest timestamp in the database."""
        return max(
            table.max_timestamp
            for table in self.table_dict.values()
            if table.time_col is not None
        )

    def upto(self, timestamp: pd.Timestamp) -> Self:
        r"""Return a database with all rows upto timestamp."""
        return Database(
            table_dict={
                name: table.upto(timestamp) for name, table in self.table_dict.items()
            }
        )

    def from_(self, timestamp: pd.Timestamp) -> Self:
        r"""Return a database with all rows from timestamp."""
        return Database(
            table_dict={
                name: table.from_(timestamp) for name, table in self.table_dict.items()
            }
        )

    
    def reindex_pkeys_and_fkeys(self) -> None:
        """
        Replaces every primary key (PK) and foreign key (FK) value across all tables 
        with contiguous integer indices (Int64).
        
        Custom Architecture Updates:
        ----------------------------
        - Creates TRUE GROUP IDs for subset combinations. If "123-10" appears multiple 
          times in the parent, it is assigned the same integer Group ID.
        - Automatically matches a child table's available FK subset to the exact corresponding 
          parent combo column, completely agnostic to the order of columns in the child.
        """

        # Set the minimum acceptable match rate before triggering fallback cleanup logic (5%)
        min_hit_rate = 0.05
        # Initialize dictionary to store standard 1-to-1 mappings for single-column PKs
        parent_pk_to_new_index = {}  
        # Initialize dictionary to store the Group ID mappings for each combination subset
        parent_subset_lookups = {}  

        # Define a helper function to calculate the percentage of non-null values in an array
        def _share_of_non_na(array_like) -> float:
            # Create a boolean mask of non-nulls and compute the mean (percentage)
            return (~pd.isna(array_like)).mean()

        # Define a helper function to concatenate multiple columns into a single hyphenated string
        def _concat_cols_as_key(df: pd.DataFrame, cols: list[str]) -> pd.Series:
            # Convert specified columns to strings and join them row-wise with a "-"
            return df[cols].astype(str).agg("-".join, axis=1)

        # ──────────────────────────────────────────────────────────────────────
        # STEP 1) Build reindex mapping for every parent table
        # ──────────────────────────────────────────────────────────────────────
        # Loop through every table object stored in the database
        for parent_table_name, parent_table in self.table_dict.items():

            # If the current table does not have a defined primary key, skip it
            if parent_table.pkey_col is None:
                continue

            # CASE A: List of Primary Keys (TRUE GROUP ID LOGIC)
            # Check if the primary key attribute is a list (indicating combinatorial subsets)
            if isinstance(parent_table.pkey_col, list):
                
                # Initialize a nested dictionary for this specific parent table's lookups
                parent_subset_lookups[parent_table_name] = {}

                # Loop through each generated combination column name (e.g., 'P1', 'P1-P2')
                for combo_col in parent_table.pkey_col:
                    # Verify that this combination column actually exists in the dataframe
                    if combo_col in parent_table.df.columns:
                        # Extract the column values and convert them to raw strings
                        raw_strings = parent_table.df[combo_col].astype(str)
                        
                        # 1. Extract an array of all unique string values in this combination column
                        unique_strings = raw_strings.unique()
                        
                        # 2. Create a contiguous integer index (0, 1, 2...) equal to the number of unique strings
                        group_ids = pd.RangeIndex(len(unique_strings)).astype("Int64")
                        
                        # 3. Zip the unique strings and integers together into a Pandas Series lookup table
                        group_mapping = pd.Series(group_ids, index=unique_strings, name="index")
                        
                        # 4. Save this newly created lookup table into the main dictionary for later use
                        parent_subset_lookups[parent_table_name][combo_col] = group_mapping

                        # 5. Overwrite the original string column in the dataframe with the mapped integer Group IDs
                        parent_table.df[combo_col] = raw_strings.map(group_mapping).astype("Int64")

            # CASE B: Standard Single Column Primary Key
            # If the primary key is just a single string name (legacy logic)
            else:
                # Get the name of the single primary key column
                parent_pk_col_name = parent_table.pkey_col
                # Extract the actual data series for this primary key
                parent_old_pk_series = parent_table.df[parent_pk_col_name]
                # Create a contiguous integer sequence from 0 to the total number of rows
                parent_new_index_values = pd.RangeIndex(len(parent_old_pk_series)).astype("Int64")

                # Create a 1-to-1 mapping Series mapping the old string IDs to the new integer IDs
                parent_pk_to_new_index[parent_table_name] = pd.Series(
                    parent_new_index_values,
                    index=parent_old_pk_series,
                    name="index",
                )
                # Overwrite the parent table's primary key column with the new integer IDs
                parent_table.df[parent_pk_col_name] = parent_new_index_values



        # ──────────────────────────────────────────────────────────────────────
        # STEP 2) Map every FK in every child table to the new Group IDs
        # ──────────────────────────────────────────────────────────────────────
        # Loop through every table in the database again, treating them as potential child tables
        for child_table in self.table_dict.values():

            # Create a dictionary to group the foreign keys in this child table by their target parent table
            parent_to_fk_list = defaultdict(list)
            # Iterate through the child's schema mapping of FK -> Parent Table
            for fk_column, parent_name in child_table.fkey_col_to_pkey_table.items():
                # Group the FK columns belonging to the same parent
                parent_to_fk_list[parent_name].append(fk_column)

            # Process the FK mappings grouped by parent table
            for parent_name, child_fk_columns in parent_to_fk_list.items():
                # Retrieve the parent table object
                parent_table = self.table_dict[parent_name]
                # Retrieve the parent table's primary key definition
                parent_pk_column = parent_table.pkey_col

                # CASE A: Parent has a list of combo primary keys
                # Check if the parent utilizes the combinatorial subset logic
                if isinstance(parent_pk_column, list):
                    
                    # Filter the expected child FK columns to only those that actually exist in the dataframe
                    existing_child_fks = [c for c in child_fk_columns if c in child_table.df.columns]
                    # If the child contains none of the expected FK columns, skip to the next relationship
                    if not existing_child_fks:
                        continue

                    # Initialize a variable to store the matching parent combination column name
                    target_combo_col = None
                    # Convert the list of existing child FK columns to a set to ignore ordering
                    set_existing = set(existing_child_fks)
                    
                    # Loop through all the combination columns generated in the parent table
                    for combo_col in parent_pk_column:
                        # Split the parent combination string, convert to a set, and compare with the child's set
                        if set(combo_col.split("-")) == set_existing:
                            # If the sets match perfectly, we found the right parent lookup column
                            target_combo_col = combo_col #we find the exact single combo to that of a child
                            # Break the loop since we found the match
                            break

                    # Check if we successfully identified a target column AND its lookup table exists
                    if target_combo_col and target_combo_col in parent_subset_lookups[parent_name]:
                        
                        # Split the parent's combination string to extract the exact ordering it used
                        ordered_child_cols = target_combo_col.split("-")
                        
                        # Concatenate the child's columns using the exact order dictated by the parent string
                        fk_concat_raw = _concat_cols_as_key(child_table.df, ordered_child_cols)

                        # Retrieve the pre-computed Group ID lookup Series for this specific combination
                        lookup_series = parent_subset_lookups[parent_name][target_combo_col]

                        # Map the child's concatenated strings to the lookup Series to fetch the integer Group IDs
                        mapped_indices = fk_concat_raw.map(lookup_series).astype("Int64")

                        # --- Fallback logic (Zero stripping) ---
                        # If the percentage of successfully mapped IDs is below the 5% threshold
                        if _share_of_non_na(mapped_indices) < min_hit_rate:
                            # Strip leading zeros from the child columns and re-concatenate them
                            fk_concat_nozeros = (
                                child_table.df[ordered_child_cols]
                                .astype(str)
                                .apply(lambda col: col.str.lstrip("0"))
                                .agg("-".join, axis=1)
                            )
                            # Create a copy of the lookup Series to safely modify
                            lookup_nozeros = lookup_series.copy()
                            # Strip leading zeros from the keys (index) in the parent lookup Series
                            lookup_nozeros.index = (
                                lookup_nozeros.index.astype(str)
                                .str.split("-")
                                .map(lambda tokens: "-".join(t.lstrip("0") for t in tokens))
                            )
                            # Attempt to map the zero-stripped child keys to the zero-stripped parent lookup
                            zero_fix_indices = (
                                lookup_nozeros.reindex(fk_concat_nozeros)
                                .astype("Int64")
                                .values
                            )
                            # If zero-stripping improved the match rate, adopt these new indices
                            if _share_of_non_na(zero_fix_indices) > _share_of_non_na(mapped_indices):
                                mapped_indices = zero_fix_indices

                        # --- Fallback logic (Date fixing) ---
                        # If the match rate is still below 5% and any of the columns appear to contain dates
                        if _share_of_non_na(mapped_indices) < min_hit_rate and any(
                            appears_to_be_date(child_table.df[col]) for col in ordered_child_cols
                        ):
                            # Remove all non-numeric characters from the child strings (e.g. format to YYYYMMDD)
                            fk_concat_dates = (
                                child_table.df[ordered_child_cols]
                                .applymap(lambda cell: re.sub(r"\D", "", str(cell)))
                                .agg("-".join, axis=1)
                            )
                            # Create another copy of the original lookup Series
                            lookup_dates = lookup_series.copy()
                            # Remove all non-numeric characters from the keys in the parent lookup Series
                            lookup_dates.index = (
                                lookup_dates.index.astype(str)
                                .str.split("-")
                                .map(lambda tokens: "-".join(re.sub(r"\D", "", t) for t in tokens))
                            )
                            # Attempt to map using the cleaned numeric date strings
                            date_fix_indices = (
                                lookup_dates.reindex(fk_concat_dates)
                                .astype("Int64")
                                .values
                            )
                            # If date formatting improved the match rate, adopt these new indices
                            if _share_of_non_na(date_fix_indices) > _share_of_non_na(mapped_indices):
                                mapped_indices = date_fix_indices

                        # Finally, overwrite the child table's original FK columns with the matched Group IDs
                        for fk_col in ordered_child_cols:
                            child_table.df[fk_col] = mapped_indices

                    # If no valid subset combination was found in the parent
                    else:
                        # Fill the child's existing FK columns with Pandas NA (null) values
                        for fk_col in existing_child_fks:
                            child_table.df[fk_col] = pd.Series(pd.NA, index=child_table.df.index, dtype="Int64")

                # CASE B: Single-column PK logic
                # If the parent utilizes a traditional single-column primary key
                else:
                    # Retrieve the single-column 1-to-1 lookup Series for the parent
                    parent_lookup_series = parent_pk_to_new_index[parent_name]
                    
                    # Iterate over the specified FK columns in the child
                    
                    for fk_col in child_fk_columns:   #However, in relational databases, a child table can actually have multiple different Foreign Keys that all point to the exact same Parent Table.
                        # If the FK column does not exist in the dataframe, skip it
                        if fk_col not in child_table.df.columns:
                            continue

                        # Extract the data series for the current FK column
                        foreign_key_series = child_table.df[fk_col]

                        # FAST PATH: If the child FK and parent PK data types match exactly
                        if foreign_key_series.dtype == parent_lookup_series.index.dtype:
                            # Perform a fast Pandas left merge to map the indices
                            merged_result = (
                                pd.merge(
                                    child_table.df[[fk_col]],
                                    parent_lookup_series,
                                    left_on=fk_col,
                                    right_index=True, #find the match here for the right table
                                    how="left",
                                )["index"] #this is the column name from parent look which has ids.
                                .astype("Int64")
                            )
                            # Overwrite the child FK column with the merged integer indices
                            child_table.df[fk_col] = merged_result
                            # Move to the next FK column
                            continue

                        # SLOW PATH: If data types differ, convert the lookup keys to strings
                        lookup_as_str = parent_lookup_series.copy()
                        lookup_as_str.index = lookup_as_str.index.astype(str)

                        # Convert the child FK to string and map it against the string-based lookup
                        mapped_indices = (
                            foreign_key_series.astype(str)
                            .map(lookup_as_str)
                            .astype("Int64")
                        )

                        # --- Single Column Date Fixing Fallback ---
                        # If the match rate is below 5% and either the child or parent column looks like a date
                        if _share_of_non_na(mapped_indices) < min_hit_rate and (
                            appears_to_be_date(foreign_key_series)
                            or appears_to_be_date(parent_lookup_series.index.to_series())
                        ):
                            # Normalize the child FK column to a clean YYYYMMDD integer format
                            foreign_key_dates = normalise_to_yyyymmdd(foreign_key_series)
                            # Normalize the parent PK index to a clean YYYYMMDD integer format
                            parent_pk_dates = normalise_to_yyyymmdd(parent_lookup_series.index.to_series())

                            # Rebuild the lookup Series using the normalized date integers as the index
                            date_lookup_series = pd.Series(
                                parent_lookup_series.values,
                                index=parent_pk_dates,
                            )

                            # Attempt to map the normalized child dates to the normalized parent dates
                            date_fix_indices = (
                                foreign_key_dates.map(date_lookup_series)
                                .astype("Int64")
                            )

                            # If date normalization improved the match rate, adopt the new indices
                            if _share_of_non_na(date_fix_indices) > _share_of_non_na(mapped_indices):
                                mapped_indices = date_fix_indices

                        # Overwrite the child FK column with the final mapped integer indices
                        child_table.df[fk_col] = mapped_indices


# In[113]:


#function goes into rel_db class

def find_target_table(table_to_col_mapping, target_cols):
    # Ensure target_cols is a list, even if a single target is provided
    if not isinstance(target_cols, list):
        target_cols = [target_cols]
    
    tables = []
    # Find which tables contain any of the target columns
    for table_name, columns in table_to_col_mapping.items():
        # Ensure columns is a list or set and handle comparisons correctly
        if not isinstance(columns, list):
            raise TypeError(f"Expected list for columns in table {table_name}, but got {type(columns).__name__}")
        for target_col in target_cols:
            if target_col in columns:
                tables.append(table_name)
    
    return list(set(tables))  # Remove duplicates if the same table appears multiple times


# In[114]:


#SOAC_db class custom_dataset

class relbench_dataset(Dataset):
  ################################################################################
  # Choose the val_timestamp and test_timestamp carefully
  ################################################################################

  def __init__(self,
               tenant_id, cleaned_tables, use_case: str, multi_task, tenant_metadata, database_schema, metadata_semantics, target_columns, cache_dir):
    



    
    self.use_case = use_case
    self.tenant_id = tenant_id
    self.cache_dir = cache_dir
    
    self.multi_task = multi_task
    


    self.cleaned_tables = cleaned_tables
    
   
    self.metadata = tenant_metadata
    
    
    self.database_schema = database_schema
   
    self.metadata_semantics = metadata_semantics
    

    

   
   
   
    # load in val and test timestamp
    self.load_train_test_timestamp()
    # metadata related to SOAC task (targets)
    self.load_task_metadata()
    if self.multi_task:
        # Overwrite self.target with all possible targets
        targs = self.task_metadata['available_targets']
        self.target_col = list(set(targs))
    
  
   
  # def validate_and_correct_db(self, db):
  #       r"""Validate and correct input db in-place.

  #       Removing rows after test_timestamp can result in dangling foreign keys.
  #       """
  #       # Validate that all primary keys are consecutively index.

  #       for table_name, table in db.table_dict.items():
  #           if table.pkey_col is not None:
  #               ser = table.df[table.pkey_col]
  #               if not (ser.values == np.arange(len(ser))).all():
  #                   raise RuntimeError(
  #                       f"The primary key column {table.pkey_col} of table "
  #                       f"{table_name} is not consecutively index."
  #                   )

  #       # Discard any foreign keys that are larger than primary key table as
  #       # dangling foreign keys (represented as None).
  #       for table_name, table in db.table_dict.items():
  #           for fkey_col, pkey_table_name in table.fkey_col_to_pkey_table.items():
  #               num_pkeys = len(db.table_dict[pkey_table_name])
  #               mask = table.df[fkey_col] >= num_pkeys
  #               if mask.any():
  #                   table.df.loc[mask, fkey_col] = None
  

  def validate_and_correct_db(self, db):
        r"""Validate and correct input db in-place.
        Removing rows after test_timestamp can result in dangling foreign keys.
        """
        for table_name, table in db.table_dict.items():
            if table.pkey_col is not None:
                
                # --- CUSTOM GRAPH BUILDER LOGIC ---
                # Since combination subsets utilize Group IDs, they will naturally have 
                # duplicate integers. We just validate that the mapping didn't fail (no NAs).
                if isinstance(table.pkey_col, list):
                    for col in table.pkey_col:
                        if table.df[col].isna().any():
                            raise RuntimeError(
                                f"The generated Group ID column {col} in table "
                                f"{table_name} contains null values."
                            )
                            
                # --- ORIGINAL STRICT LOGIC ---
                # Single string primary keys should still be perfectly unique 0 to N-1.
                else:
                    ser = table.df[table.pkey_col]
                    if not (ser.values == np.arange(len(ser))).all():
                        raise RuntimeError(
                            f"The primary key column {table.pkey_col} of table "
                            f"{table_name} is not consecutively indexed."
                        )

        # Discard any foreign keys that are larger than primary key table as
        # dangling foreign keys.
        for table_name, table in db.table_dict.items():
            for fkey_col, pkey_table_name in table.fkey_col_to_pkey_table.items():
                num_pkeys = len(db.table_dict[pkey_table_name])
                mask = table.df[fkey_col] >= num_pkeys
                if mask.any():
                    table.df.loc[mask, fkey_col] = pd.NA
  
  
  
  
    
 
  
  
  
  # @staticmethod
  # def _force_safe_strings(self, df: pd.DataFrame, cols: list[str]) -> None:
  #     """
  #     In place conversion of the listed columns to Pandas 'string[python]'
  #     (work around for Arrow/NA coercion bug).
  #     """
  #     for c in cols:
  #         if c in df.columns:
  #             df[c] = df[c].astype(str)
  #     return df



# load timestamp information
  def load_train_test_timestamp(self):
    """
    Logic for loading in time split information
    #FIX ME: Add logic for loading in soac_meta from s3
    """
    # FIX ME: Verify: taking first timestamp from validation and test split
    train_start = self.metadata['date_train_test_split'][self.tenant_id]['train'][0]
    val_time = self.metadata['date_train_test_split'][self.tenant_id]['validation'][0]
    test_start_time = self.metadata['date_train_test_split'][self.tenant_id]['test'][0]
    test_end_time = self.metadata['date_train_test_split'][self.tenant_id]['test'][1]

    # --- NEW LOGIC: Dynamic Type Parsing (Supports IDs) ---
    def parse_split_value(val):
        # 1. Try Numeric ID first (handles 1, "1000", 2.5)
        try:
            f_val = float(val)
            # If it's a whole number, keep it as an int, else float
            return int(f_val) if f_val.is_integer() else f_val
        except (ValueError, TypeError):
            pass
            
        # 2. Try Datetime next (handles "2024-01-01")
        try:
            return pd.Timestamp(val)
        except (ValueError, TypeError):
            pass
            
        # 3. Fallback to Alphanumeric String (handles "A100")
        return str(val)


    # Assign the safely parsed boundaries
    self.train_timestamp = parse_split_value(train_start)
    self.val_timestamp = parse_split_value(val_time)
    self.test_start_time = parse_split_value(test_start_time)
    self.test_timestamp = self.test_start_time
    self.test_end_time = parse_split_value(test_end_time)

    print(f'Accessing time splits for tenant: {self.tenant_id}. \n'
          f'Train start timestamp: {self.train_timestamp}. \n'
          f'Validation start timestamp: {self.val_timestamp}. \n'
          f'Test start timestamp: {self.test_start_time}. \n'
          f'Test end timestamp: {self.test_end_time}')





  # Custom function to handle parsing of time
  @staticmethod
  def parse_date(date_str, date_formats = ['%Y-%m-%d', '%d-%m-%y', '%Y/%m/%d', '%d/%m/%y']):
    if pd.isna(date_str):
        return pd.NaT
    for fmt in date_formats:
        try:
            # Attempt to parse the date string with the current format
            parsed_date = datetime.strptime(date_str, fmt)
            # Check if the year is within bounds for pandas to_datetime
            if 1678 <= parsed_date.year <= 2262:
                return dateutil.parser.parse(str(parsed_date), fuzzy=True)
        except (ValueError, pd.errors.OutOfBoundsDatetime):
            continue
    # If all formats fail or the date is out of bounds, return NaT
    return pd.NaT
  
  
  def clean_table(self, data, pkey, fkey, time_col = None, 
                  drop_dups = True, test_sample = False, 
                  sample_size = 1000, num_columns_to_sample = 5):
    """
    Basic table cleaning with safe column checks to prevent KeyErrors.
    """
    print(f'table before cleaning: {data.head()}')

    # ---------------------------------------------------------
    # THE FIX: Safe Primary Key Handling
    # All PK logic (nulls and duplicates) is safely tucked inside this check.
    # ---------------------------------------------------------
    if pkey is not None:
        # Ensure pkey_subset is a flat list whether pkey is a string or already a list
        pkey_subset = pkey if isinstance(pkey, list) else [pkey]
        
        # Filter to ONLY the columns that actually exist in the DataFrame
        existing_pkeys = [col for col in pkey_subset if col in data.columns]
        
        if existing_pkeys:
            print('Counting nulls for primary key column')
            print(data[existing_pkeys].isnull().sum())
            
            # Drop nulls using only the existing columns
            data.dropna(subset=existing_pkeys, inplace=True)

            # Identify any duplicate primary keys using the existing columns
            duplicate_pkey = data.duplicated(subset=existing_pkeys).sum()
            print(f'Number of duplicate primary keys: {duplicate_pkey}')
            
            if drop_dups:
                data.drop_duplicates(subset=existing_pkeys, inplace=True)
        else:
            # If the PK doesn't exist at all, print a warning instead of crashing
            print(f"[WARNING] Primary key(s) {pkey_subset} not found in the DataFrame! Skipping null/duplicate checks.")
    else:
        print("[INFO] pkey is None for this table. Skipping primary key cleanup.")

    # ---------------------------------------------------------
    # Remove other possible targets / invalid lifecycle cols
    # ---------------------------------------------------------
    original_num_columns = len(data.columns)
    available_targets = self.task_metadata['available_targets']
    print(f'Available targets are: {available_targets}')
    
    # Drop all available targets except the one present in the data
    if not self.multi_task:
      targets_to_drop = [col for col in available_targets if col != self.target_col]
      targets_to_drop_filtered = data.columns.intersection(targets_to_drop)
      # Drop the columns in-place
      data.drop(columns=targets_to_drop_filtered, inplace=True)
      print(f"Number of columns before: {original_num_columns}, after: {len(data.columns)}. Removed the possible targets: {targets_to_drop_filtered}")
      for targ in targets_to_drop:
        assert targ not in data.columns.tolist()

    # remove lifecycle columns == 1
    original_num_columns = len(data.columns)
    print(f'Retaining lifecyle colummns with 0 or 2. Removing invalid lifecyle columns ==1. \nThe invalid columns are: {self.task_metadata["invalid_lifecycle_cols"]}')
    # Get the list of columns that exist in both the DataFrame and the invalid columns list
    existing_invalid_columns = [col for col in self.task_metadata['invalid_lifecycle_cols'] if col in data.columns]
    data.drop(columns = existing_invalid_columns, inplace = True)
    new_num_columns = len(data.columns)
    print(f'Original number of columns: {original_num_columns}. After removing invalid lifecycle columns ({existing_invalid_columns}): {new_num_columns} columns.')

    if test_sample:
      # Sampling just for testing
      if sample_size > len(data):
        sample_size = len(data)
      # Sample rows
      data.sample(sample_size).reset_index(drop = True, inplace = True)
      # Sample columns. Don't remove targets or pkey, fkeys, and time_col
      if not isinstance(time_col, list):
        time_col = [time_col]
      
      # Safe list construction ensuring we don't add None to the list
      safe_pkey = [pkey] if isinstance(pkey, str) else (pkey if isinstance(pkey, list) else [])
      targets_and_pkey_fkey = list(fkey.keys()) + self.task_metadata['available_targets'] + safe_pkey + time_col
      
      columns_to_sample_from = [col for col in data.columns if col not in targets_and_pkey_fkey]
      sampled_columns = random.sample(columns_to_sample_from, min(num_columns_to_sample, len(columns_to_sample_from)))
      # retain sampled cols and targets_and_pkey_fkey
      retained_columns = sampled_columns + targets_and_pkey_fkey
      retained_columns = [col for col in retained_columns if col in data.columns]
      print(f'Sampling the following columns for testing: {retained_columns}')
      data = data[retained_columns]

    print(f'table after cleaning: {data.head()}')

    return data

  
  
  
  
  def load_task_metadata(self):
      """
      load metadata related to task
      """
      task_metadata = self.metadata_semantics 
      # possible targets for SOAC
      #available_targets = [target.get("original_column", target.get("column")) for target in task_metadata["targets"]]
      
      available_targets = target_columns
      
      # if self.use_case == 'SOAC':
      #   available_targets = available_targets + ['HEADERINCOTERMSCLASSIFICATION', 'ITEMINCOTERMSCLASSIFICATION']
      
      
      # feature lifecycles
      # 0 - no
      # 1 - leakage
      # 2 - label
      invalid_lifecycle_cols = [feature['column'] for feature in task_metadata['features'] if feature['lifecycle'] == 1]
      task_info = {'task_metadata': task_metadata, 'available_targets': available_targets, 'invalid_lifecycle_cols': invalid_lifecycle_cols}
      self.task_metadata = task_info 



  # def convert_to_rel_table(self, database_schema, table_name, use_case, table_dict, drop_dups = True):

  #   """
  #   Parse fkey: table, pkey, time_col based on meta data
  #   We may update how to read in database_schema, I just import the dictionary for now
  #   If we detect multiple pkeys, a new pkey column is created as the concatenation of the two pkey cols
    
  #   Usage Example:
  #   table = convert_to_rel_table(database_schemas, 'df_salesdocument', 'SOAC', table_dict)
  #   """
    
   
    
  #   # gather meta data
  #   fkey = self.database_schema[self.use_case][table_name]['fkey_col_to_pkey_table']
  #   pkey = self.database_schema[self.use_case][table_name]['pkey_col']
  #   time = self.database_schema[self.use_case][table_name]['time_col']
  #   print(f'metadata for: {table_name}\nuse case: {use_case}\nfkey:{fkey}, pkey:{pkey}, time:{time}')
  #   data = table_dict[table_name]


  #   #  # >>>>>>>  add these three lines  <<<<<<<
  #   key_cols = list(fkey) + (pkey if isinstance(pkey, list) else [pkey])
  #   # data = self._force_safe_strings(data, key_cols)
  #   # print(f'data after force safe string: {data.head()}')
  #   # # >>>>>>>  end of added lines  <<<<<<<<
 



  #   # table may have multiple pkeys
  #   # use list of pkey columns for multiple pkeys
  #   if isinstance(pkey, list):
  #     # create a composite pkey column
  #     data[pkey] = data[pkey].astype(str)
  #     pkey_composite = '-'.join(key for key in pkey)
  #     data[pkey_composite] = data[pkey].fillna('').agg('-'.join, axis=1)
  #     print(f'Warning, multiple primary keys detected for table: {table_name}. Be sure this is the case. Concatenating columns: {pkey}. New primary key is {pkey_composite}')
      
  #     # Overwrite with new concatenated pkey
  #     pkey = pkey_composite
    
    
  #   data = self.clean_table(data, pkey, fkey, time_col = time)
  #   # Convert to relbench Table
  #   # FIXME: We may want to provide time columns. They convert them to unix time
  #   table = Table(data, fkey_col_to_pkey_table = fkey, time_col = time, pkey_col=pkey)
    
  #   #table.original_pk_cols = data.attrs.get("original_pk_cols")
    
    
  #   print(f'rel table: {table.df}')
  #   return table



  def convert_to_rel_table(self, database_schema, table_name, use_case, table_dict, drop_dups=True):
        
        # ─── SETUP: Fetching initial data from your dictionaries ───
        fkey = self.database_schema[self.use_case][table_name].get('fkey_col_to_pkey_table', {}) # -> {}
        pkey = self.database_schema[self.use_case][table_name].get('pkey_col')                   # -> ['P1', 'P2', 'P3']
        time = self.database_schema[self.use_case][table_name].get('time_col')                   # -> 'DATE_COLUMN'
        data = table_dict[table_name]                                                            # -> The Pandas DataFrame

        # If we have multiple primary keys (which we do: ['P1', 'P2', 'P3'])
        if isinstance(pkey, list):
            
            # Cast all original PK columns to strings for safe joining later
            data[pkey] = data[pkey].astype(str)                                                  
            
            # --- NEW FIX: TARGETED SUBSET GENERATION ---
            
            # Create an empty set to hold our unique combinations
            required_subsets = set()                                                             # -> set()
            
            # 1. Always include the full primary key combination.
            # We convert the list ['P1', 'P2', 'P3'] to a tuple so the set accepts it.
            required_subsets.add(tuple(pkey))                                                    # -> { ('P1', 'P2', 'P3') }
            
            
            
            # 3. Scan the schema to see exactly which composite subsets child tables request.
            for c_tbl, c_info in self.database_schema[self.use_case].items():                    
                
                c_fkeys = c_info.get('fkey_col_to_pkey_table', {})                               # -> {'P1': 'df_parent', 'P2': 'df_parent'}
                
                # Find all FKs in the child that point to THIS specific parent table
                child_subset = [fk for fk, p_tbl in c_fkeys.items() if p_tbl == table_name and fk in pkey] 
                # -> ['P1', 'P2']
                
                # We only need to generate a new composite column if the child needs >1 key.
                # (Single keys like 'P1' were already added in Step 2 above).
                if len(child_subset) > 1:                                                        # -> len is 2, so this runs!
                    
                    # Ensure the subset is in the exact order the parent defined it
                    ordered_subset = tuple([p for p in pkey if p in child_subset])               # -> ('P1', 'P2')
                    
                    # Add this specific combination to our required set
                    required_subsets.add(ordered_subset)                                         
                    # -> { ('P1', 'P2', 'P3'), ('P1',), ('P2',), ('P3',), ('P1', 'P2') }
            
            # Initialize a list to hold the final string names of our columns
            all_pk_combinations = []                                                             # -> []
            
            # 4. Only build the combination columns that are strictly required
            # The set currently contains 5 tuples. We loop through each one.
            for subset in required_subsets:                                                      
                
                # Let's trace the loop for subset = ('P1', 'P2')
                combo_name = '-'.join(subset)                                                    # -> 'P1-P2'
                
                # Only run the heavy string concatenation if it's a true composite (length > 1)
                # If subset is just ('P1',), length is 1, so we skip the heavy concatenation!
                if len(subset) > 1:                                                              # -> len is 2, so this runs!
                    # Glues the row values together with hyphens and saves it as a new column
                    data[combo_name] = data[list(subset)].fillna('').agg('-'.join, axis=1)       # -> Creates DataFrame column 'P1-P2'
                
                # Add the string name to our final list of primary keys
                all_pk_combinations.append(combo_name)                                           # -> ['P1-P2']
                
                # ... Loop repeats for the other 4 subsets in the set ...
                # Final output of loop -> ['P1-P2', 'P1-P2-P3', 'P1', 'P2', 'P3']

            print(f"Targeted Generation for {table_name}: Created {len(all_pk_combinations)} combos instead of {2**len(pkey) - 1}.")
            # -> "Targeted Generation for df_parent: Created 5 combos instead of 7."
            
            # Overwrite pkey with the targeted list of combination columns
            pkey = all_pk_combinations                                                           # -> ['P1-P2', 'P1-P2-P3', 'P1', 'P2', 'P3']
        
        # Clean the table
        data = self.clean_table(data, pkey, fkey, time_col=time)



        # --- THE FINAL SAFETY NET (UPDATED) ---
        # Identify the active split column for this specific table before finalizing it
        is_row_split = '__row_order__' in data.columns
        active_col = '__row_order__' if is_row_split else time
        
        # Guarantee the active column's data type perfectly matches our split boundaries
        if active_col and active_col in data.columns:
            if isinstance(self.test_timestamp, (int, float)):
                # Boundary is numeric, force column to numeric
                data[active_col] = pd.to_numeric(data[active_col], errors='coerce')
            elif isinstance(self.test_timestamp, pd.Timestamp):
                # Boundary is datetime, force column to datetime
                data[active_col] = pd.to_datetime(data[active_col], errors='coerce')
            else:
                # Boundary is string, force column to string
                data[active_col] = data[active_col].astype(str)
                
        

        
        # Pass the list of targeted combinations as the pkey_col to RelBench
        table = Table(data, fkey_col_to_pkey_table=fkey, time_col=time, pkey_col=pkey)
        
        return table



  # def fill_na_fk_with_self_index(self, db: "Database") -> None:
      

  #     for tbl_name, tbl in db.table_dict.items():

  #         # skip tables without PK or without FKs
  #         if tbl.pkey_col is None or not tbl.fkey_col_to_pkey_table:
  #             continue

          

  #         # -----------------------------------------------------------------
  #         # 2) patch every FK column that still has missing values
  #         # -----------------------------------------------------------------
  #         for fk_col in tbl.fkey_col_to_pkey_table.keys():
  #             na_mask = tbl.df[fk_col].isna()
  #             if na_mask.any():
  #                 tbl.df.loc[na_mask, fk_col] = tbl.df[tbl.pkey_col][na_mask]
                

  

  def make_db(self, minimal_version = False) -> Database:
    r"""Process the raw files into a database."""
    ################################################################################
    # Load the data. You can use any URL or even local files. We load from azure here. This may be updated with GPT4HANA logic
    ################################################################################
    print(f'Loading tenant id: {self.tenant_id}')
    #table_dict = load_soac(self.tenant_id)
    table_dict = self.cleaned_tables
    print(f'Table Dictionary Keys: {table_dict.keys()}')
    length_at_load = {}
    for key, value in table_dict.items():
      length_at_load[key] = f'Table Name: {key}. Num Rows at load: {len(value)}, Num Cols at load: {len(value.columns)}'

    ################################################################################
    # It is important to understand the data.
    # This can point to columns which should be removed.
    # The most important of these is temporal leakage columns, which we will
    # discuss in detail later.
    ################################################################################

    # Remove columns that are irrelevant, leak time,
    # or have too many missing values

    ################################################################################
    # Make sure that the missing data has been parsed properly.
    # Following Pandas, we represent missing values with NaNs in the dataframe.
    ################################################################################

    ################################################################################
    # Here, we collect all tables in the database as relbench.base.Table objects.
    ################################################################################

    # Create a dictionary of relbench tables
    tables = {}
    for table_name, table in table_dict.items():
      rel_table = self.convert_to_rel_table(self.database_schema, table_name, self.use_case, table_dict)
      tables[table_name] = rel_table
    
    # Convert to a relbench Database
    db = Database(tables)
    

    # if self.use_case == 'SOAC':
    #   # Rename a few target columns
    #     db.table_dict['df_salesdocument'].df.rename(columns = {'INCOTERMSCLASSIFICATION': 'HEADERINCOTERMSCLASSIFICATION'}, inplace = True)
    #     db.table_dict['df_salesdocumentitem'].df.rename(columns = {'INCOTERMSCLASSIFICATION': 'ITEMINCOTERMSCLASSIFICATION'}, inplace = True)

    # Create minimal version
    # if minimal_version and self.use_case == 'SOAC':
    #   minimal_sales_doc = ['CREATIONDATE', 'SALESDOCUMENT', 'SOLDTOPARTY', 'BILLINGCOMPANYCODE', 'CREATIONDATE', 'CREATIONTIME', 'CUSTOMERPAYMENTTERMS', 'DISTRIBUTIONCHANNEL', 'HEADERINCOTERMSCLASSIFICATION', 'ORGANIZATIONDIVISION', 'SALESDOCUMENTTYPE','SALESGROUP', 'SALESOFFICE', 'SALESORGANIZATION', 'SDDOCUMENTREASON', 'SHIPPINGCONDITION', 'TRANSACTIONCURRENCY']
    #   minimal_sales_doc_item = ['CREATIONDATE', 'SALESDOCUMENT', 'SALESDOCUMENTITEM', 'BILLTOPARTY', 'PAYERPARTY','SHIPTOPARTY', 'SOLDTOPARTY', 'PLANT', 'PRODUCT', 'RETURNREASON', 'RETURNSREFUNDEXTENT', 'SALESDOCUMENTITEMCATEGORY', 'SHIPPINGPOINT','ITEMINCOTERMSCLASSIFICATION']
      
    #   # add pkey and fkey
    #   minimal_sales_doc_item.append(db.table_dict['df_salesdocumentitem'].pkey_col)

    #   # Filter down column list to only columns existing in the dataframe
    #   minimal_sales_doc_filtered = db.table_dict['df_salesdocument'].df.columns.intersection(minimal_sales_doc)
    #   minimal_sales_doc_item_filtered = db.table_dict['df_salesdocumentitem'].df.columns.intersection(minimal_sales_doc_item)

    #   db.table_dict['df_salesdocument'].df = db.table_dict['df_salesdocument'].df[minimal_sales_doc_filtered]
    #   db.table_dict['df_salesdocumentitem'].df = db.table_dict['df_salesdocumentitem'].df[minimal_sales_doc_item_filtered]
      
    #   print(f"Minimal Cols Sales Doc: {db.table_dict['df_salesdocument'].df.columns}")
    #   print(f"Minimal Cols Sales Doc Item: {db.table_dict['df_salesdocumentitem'].df.columns}")

    # # Ugly workaround to filter out targets
    # if not self.multi_task and self.use_case == 'SOAC':
    #   if self.target_col != 'ITEMINCOTERMSCLASSIFICATION':
    #     if 'ITEMINCOTERMSCLASSIFICATION' in db.table_dict['df_salesdocumentitem'].df.columns:
    #       db.table_dict['df_salesdocumentitem'].df.drop(columns = ['ITEMINCOTERMSCLASSIFICATION'], inplace = True)
    #   if self.target_col != 'HEADERINCOTERMSCLASSIFICATION':
    #     if 'HEADERINCOTERMSCLASSIFICATION' in db.table_dict['df_salesdocument'].df.columns:
    #       db.table_dict['df_salesdocument'].df.drop(columns = ['HEADERINCOTERMSCLASSIFICATION'], inplace = True)
    
    # if not self.multi_task:
    # # Check if target table was found
    #   if len(target_table) == 0:
    #       print(f"Assertion failed: Did not find target column '{self.target_col}' in any table.")
    #       print(f"Table to Column Mapping: {table_to_col_mapping}")
    #       assert False, "Did not find target column in any table"

    # Add timestamp attributes to the Database object
    db.train_timestamp = self.train_timestamp
    db.val_timestamp = self.val_timestamp
    db.test_start_time = self.test_start_time
    db.test_end_time = self.test_end_time
    # Add target column attribute to Database object
    db.target_col = self.target_col
    db.task_metadata = self.task_metadata

    return db





#removing unused old pkey_col compare schema to current active pkey from the soac_db.table_dict.df because we dont want them for predictions and graph building.
 
 
  # Define the method as part of your class, taking 'db' (the RelBench Database object) as an argument.
  def _drop_only_multiple_primary_key_columns(self, db):
      """
      Drops original PK columns (e.g., P2, P3) ONLY IF they are no longer 
      being actively used as a standalone key by the graph builder.
      """
      
      # Initialize an empty dictionary to keep track of exactly which columns we delete for each table.
      dropped_by_table = {}

      # Safety check: ensure the current use case actually exists in the provided schema.
      if self.use_case not in self.database_schema:
          return dropped_by_table

      # Isolate the portion of the schema that belongs to our specific use case (e.g., SOAC).
      schema_for_case = self.database_schema[self.use_case]

      # Loop through every table name and its metadata defined in the schema.
      for table_name, table_info in schema_for_case.items():
          
          # Check if this table actually made it into our built RelBench Database.
          if table_name not in db.table_dict:
              dropped_by_table[table_name] = []
              continue

          # Grab the RelBench Table object and its internal Pandas DataFrame.
          table = db.table_dict[table_name]
          df = table.df

          # ---------------------------------------------------------
          # Step 1: Find the RAW INGREDIENTS (The Original Schema Keys)
          # ---------------------------------------------------------
          # Get the original primary keys exactly as they were written in your JSON schema.
          original_pkeys = table_info.get("pkey_col")
          
          # --- THE FIX: Safely handle None values! ---
          if original_pkeys is None:
              original_pkeys = []
          # Normalize strings into a list
          elif isinstance(original_pkeys, str):
              original_pkeys = [original_pkeys]
          # Normalize lists/tuples into a flat list of strings
          elif isinstance(original_pkeys, (list, tuple)):
              original_pkeys = [str(c) for c in original_pkeys]

          # ---------------------------------------------------------
          # Step 2: Find the ACTIVE KEYS (What the Graph is actually using)
          # ---------------------------------------------------------
          active_pkeys = table.pkey_col
          
          if isinstance(active_pkeys, str):
              active_pkeys = [active_pkeys]
          elif active_pkeys is None:
              active_pkeys = []

          # ---------------------------------------------------------
          # Step 3: THE SMART DROP LOGIC
          # ---------------------------------------------------------
          # We only want to execute this cleanup if the table originally had more than 1 primary key.
          if len(original_pkeys) > 1:
              
              # Mark a column for deletion if it physically exists but is NOT in the active_pkeys list
              cols_to_drop = [c for c in original_pkeys if c in df.columns and c not in active_pkeys]
              
              # Physically delete useless raw ingredients from the Pandas DataFrame
              if cols_to_drop:
                  df.drop(columns=cols_to_drop, inplace=True)
              
              dropped_by_table[table_name] = cols_to_drop
              
          else:
              dropped_by_table[table_name] = []

      return dropped_by_table






  def get_db(self, upto_test_timestamp=True) -> Database:
        r"""Return the database object.

        The returned database object is cached in memory.

        Args:
            upto_test_timestamp: If True, only return rows upto test_timestamp.

        Returns:
            Database: The database object.

        `upto_test_timestamp` is True by default to prevent test leakage.
        """

        db_path = f"{self.cache_dir}/db"
        if self.cache_dir and Path(db_path).exists() and any(Path(db_path).iterdir()):
            print(f"Loading Database object from {db_path}...")
            tic = time.time()
            db = Database.load(db_path)
            toc = time.time()
            print(f"Done in {toc - tic:.2f} seconds.")

        else:
            print("Making Database object from scratch...")
            print(
                "(You can also use `get_dataset(..., download=True)` "
                "for datasets prepared by the RelBench team.)"
            )
            tic = time.time()
            db = self.make_db() #returns the object
            db.reindex_pkeys_and_fkeys()
            #self.fill_na_fk_with_self_index(db)
            
            toc = time.time()
            print(f"Done in {toc - tic:.2f} seconds.")

            if self.cache_dir:
                print(f"Caching Database object to {db_path}...")
                tic = time.time()
                db.save(db_path)
                toc = time.time()
                print(f"Done in {toc - tic:.2f} seconds.")

        if upto_test_timestamp:
            
            # --- NEW FIX: Custom Data Leakage Prevention ---
            # RelBench's built-in db.upto() forcibly slices on time_col. We must bypass it 
            # and manually slice our tables so we can route the logic to __row_order__ if it exists.
            print(f"[SAFETY] Slicing database to prevent leakage. Boundary: {self.test_timestamp}")
            
            # Loop through every table in the compiled database
            for table_name, table in db.table_dict.items():
                
                # 1. Determine which column this specific table uses for leakage slicing
                if '__row_order__' in table.df.columns:
                    target_split_col = '__row_order__'
                elif table.time_col and table.time_col in table.df.columns:
                    target_split_col = table.time_col
                else:
                    # If neither exists, we can't slice future data safely, so warn and skip
                    print(f"[WARN] No split column found for '{table_name}'. Cannot slice future data.")
                    continue
                
                # 2. Align the column's data type to perfectly match self.test_timestamp
                if isinstance(self.test_timestamp, (int, float)):
                    table.df[target_split_col] = pd.to_numeric(table.df[target_split_col], errors='coerce')
                elif isinstance(self.test_timestamp, pd.Timestamp):
                    table.df[target_split_col] = pd.to_datetime(table.df[target_split_col], errors='coerce')
                else:
                    table.df[target_split_col] = table.df[target_split_col].astype(str)
                
                # 3. Apply the slice: keep only rows <= the test set boundary
                table.df = table.df[table.df[target_split_col] <= self.test_timestamp]




        self.validate_and_correct_db(db)
        #self.fill_na_fk_with_self_index(db)

       
        dropped_summary = self._drop_only_multiple_primary_key_columns(db)

        non_empty = {t: cols for t, cols in dropped_summary.items() if cols}
        
        if non_empty:
             print("Dropped composite PK source columns:", non_empty)


        # --- NEW FINAL CLEANUP ---
        # After all processing is fully complete, unpack 1-item PK lists into strings.
        for table_name, table in db.table_dict.items():
            if isinstance(table.pkey_col, list) and len(table.pkey_col) == 1:
                table.pkey_col = table.pkey_col[0]


        return db


# In[115]:


SOAC_dataset = relbench_dataset(tenant_id = tenant_id, cleaned_tables = clean_dict, use_case = use_case, multi_task = True, tenant_metadata = tenant_metadata, database_schema = database_schema, metadata_semantics = usecase_metadata_semantics, target_columns = target_columns, cache_dir = None)
SOAC_db = SOAC_dataset.get_db(upto_test_timestamp=False)


# In[116]:


SOAC_db.target_col


# In[117]:


database_schema


# In[118]:


SOAC_db.table_dict


# In[119]:


database_schema


# In[120]:


SOAC_db.table_dict[entity_table].df


# In[121]:


SOAC_db.table_dict[entity_table].df.columns


# In[123]:


SOAC_db.table_dict[entity_table]


# In[124]:


def count_minus_100(df, target_cols):
    for col in target_cols:
        count = (df[col] == -100).sum()
        print(f"Column: {col} | Count of -100: {count}")
    print(df[target_columns].dtypes)

count_minus_100(SOAC_db.table_dict[entity_table].df, target_columns)


# In[125]:


def is_missing_or_none(val):
    if pd.isna(val):
        return True
    val_str = str(val).strip().lower()
    return val_str in ["", "none", "nan"]

mask = SOAC_db.table_dict[entity_table].df[target_columns].applymap(is_missing_or_none)
filtered =SOAC_db.table_dict[entity_table].df[target_columns][mask.any(axis=1)]

print(filtered)


# In[126]:


print(filtered)


# In[127]:


SOAC_db.table_dict[entity_table].df[target_columns].dtypes


# In[128]:


SOAC_db.table_dict[entity_table].df[target_columns].isna().sum()


# In[129]:


SOAC_db.table_dict[entity_table].df[target_columns].dtypes


# **updating database_schema according to soac_db.table_dict so that it can have have updated composite pkey**

# In[130]:


database_schema


# In[131]:


# Define the function, accepting the old database schema and the rel_db.table_dict as inputs, and returning an updated dictionary.


def update_schema_from_reldb(old_schema: Dict[str, Any], rel_db_table_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Synchronizes the database_schema with the finalized metadata stored inside rel_db.table_dict.
    It updates the primary key, foreign key, and time columns to their finalized composite/string forms
    while preserving other keys like 'origin_table'.
    """

    # Create a deep copy of the old schema dictionary.
    # We do this so we are working on a fresh dictionary. If something fails mid-execution, your original schema remains perfectly intact.
    updated_schema = deepcopy(old_schema) #this is with the use_case

    # Start a loop over every table name and its corresponding metadata inside the newly copied schema.
    # We iterate over the schema to ensure we retain any table-level attributes (like 'origin_table').
    for table_name, table_metadata in updated_schema[use_case].items():

        # Attempt to retrieve the finalized Table object from the rel_db's table dictionary using the table_name.
        # We use .get() instead of direct indexing [table_name] so the code doesn't crash if a table was dropped during preprocessing.
        rel_table_obj = rel_db_table_dict.get(table_name)

        # Check if the Table object was successfully found in the rel_db.
        # If the table doesn't exist in the rel_db, we skip updating it because we have no new metadata to pull from.
        if rel_table_obj is not None:

            # Extract the finalized primary key from the RelBench Table object.
            # We use getattr with a default of None to safely retrieve the attribute without throwing an AttributeError.
            new_pkey = getattr(rel_table_obj, "pkey_col", None)

            # Update the 'pkey_col' value in our schema dictionary to match the finalized primary key.
            # We do this because the original schema might have had a list of columns, but rel_db now holds the actual composite string (e.g., 'A-B').
            table_metadata["pkey_col"] = new_pkey

            # Extract the finalized foreign key dictionary from the RelBench Table object.
            # We default to an empty dictionary {} if it doesn't exist to maintain consistent data types.
            new_fkey_dict = getattr(rel_table_obj, "fkey_col_to_pkey_table", {})

            # Update the 'fkey_col_to_pkey_table' value in our schema to match the rel_db mapping.
            # We do this to ensure any severed edges or normalized foreign keys are accurately reflected in the final schema.
            table_metadata["fkey_col_to_pkey_table"] = new_fkey_dict

            # Extract the finalized time column from the RelBench Table object.
            # We use getattr to safely fetch it, defaulting to None if the table doesn't use temporal splitting.
            new_time_col = getattr(rel_table_obj, "time_col", None)

            # Update the 'time_col' value in our schema.
            # We do this because the original schema might have had a list of potential time columns, but rel_db has settled on the final, single string.
            table_metadata["time_col"] = new_time_col

            # Print a confirmation message to the console.
            # We do this so the developer can visually audit the logs and confirm that the synchronization succeeded for this specific table.
            print(f"[SYNC OK] Updated schema for '{table_name}' using rel_db metadata.")



        # If the rel_table_obj was None (meaning it wasn't found in rel_db_table_dict)...
        else:
            # ...print a warning message to the console.
            # We do this to alert the developer that a table present in the original schema is missing from the finalized graph database.
            print(f"[SYNC WARN] Table '{table_name}' not found in rel_db_table_dict. Skipping update.")



    # Return the fully synchronized schema dictionary back to the caller.
    # We do this so the downstream SDV pipeline can ingest the perfectly updated JSON schema.
    return updated_schema


# In[132]:


database_schema = update_schema_from_reldb(
    old_schema=database_schema, 
    rel_db_table_dict=SOAC_db.table_dict
)



database_schema


# In[133]:


database_schema


# **kg4hana_type_to_stype_mapping**

# In[134]:


use_case_column_metadata


# In[135]:


# ─────────────────────────────────────────────────────────────────────────────
# fix_none_dtypes  – replace dtype==None by "CHAR" everywhere in the metadata
# ─────────────────────────────────────────────────────────────────────────────
         
from copy import deepcopy                   # | we’ll work on a copy, not in-place
from typing import Dict, Any                # | for readable type annotations


def fix_none_dtypes(
    col_meta: Dict[str, list[Dict[str, str | None]]],   # original metadata
    *,                                                   # force keyword args ↓
    default_dtype: str = 'CHAR'                          # replacement for None
) -> Dict[str, list[Dict[str, str]]]:
    """
    Return a **new** copy of `col_meta` where every dtype that was `None`
    is replaced by `default_dtype` (default: "CHAR").
    """

    new_meta = deepcopy(col_meta)    #  deep-copy so caller’s data stay intact

    # ──iterate over all table-level entries in the metadata dict
    for tbl_name, col_list in new_meta.items():

        # col_list is the *single-element* list holding one big {col: dtype} dict
        if not col_list:             # safeguard: empty list means nothing to fix
            continue

        col_dtype_map = col_list[0]  # the inner {column → dtype} mapping

        # replace every None by the desired default
        for col, dtype in col_dtype_map.items():
            if dtype is None:        # found a missing dtype …
                col_dtype_map[col] = default_dtype  # … overwrite with "CHAR"

    return new_meta                  # hand back the fully patched copy



use_case_column_metadata = fix_none_dtypes(use_case_column_metadata)

use_case_column_metadata 


# In[136]:


SOAC_db.table_dict[entity_table].df[target_columns].isna().sum()


# In[137]:


SOAC_db.table_dict[entity_table].df.columns


# In[138]:


from torch_frame.utils.infer_stype import infer_series_stype


# load data types from kg4hana meta data
def load_column_type_metadata(use_case_column_metadata):

  """
  Data types according to kg4hana data
  We may update this to load it from an s3 bucket or azure, for now load Tom's json
  column data type is first entry for each table in the dictionary, column description is second entry

  """
  metadata_old = use_case_column_metadata
   
  metadata={}
  
  for table_name, data in metadata_old.items():
    # column data types
    metadata[table_name] = data[0]


  # show unique data types
  unique_values = set()
  
  for outer_key, inner_dict in metadata.items():
      for key, value in inner_dict.items():
          unique_values.add(value)
  print(f'Unique data types from kg4hana: {unique_values}')

  return metadata
  
  


# In[139]:


# TO DO: Save these
from torch_frame import stype

stype_to_python_type_mapping = {
    stype.text_embedded: 'str',           # Text embedded
    stype.numerical: 'float',             # Numerical
    stype.timestamp: 'datetime.date',     # Date (and time)
    stype.text_tokenized: 'str',          # Text tokenized
    stype.multicategorical: 'str',        # Multi-categorical
    stype.sequence_numerical: 'float',    # Sequence numerical
    stype.image_embedded: 'str',          # Image embedded
    stype.embedding: 'str',               # Embedding
    stype.categorical: 'str',
    None: 'str' # Default
}



kg4hana_type_to_stype_python_mapping = {
    '': {'pythontype': 'str', 'torch_frame_stype_': stype.text_embedded},              # Defaulting to str for unspecified type
    'CHAR': {'pythontype': 'str', 'torch_frame_stype_': stype.text_embedded},          # Character string
    'abap.char': {'pythontype': 'str', 'torch_frame_stype_': stype.text_embedded},
    'RAW': {'pythontype': 'str', 'torch_frame_stype_': stype.text_embedded},
    'CLNT': {'pythontype': 'str', 'torch_frame_stype_': stype.text_embedded},          # Client number
    'CUKY': {'pythontype': 'str', 'torch_frame_stype_': stype.text_embedded},          # Currency key
    'CURR': {'pythontype': 'float', 'torch_frame_stype_': stype.numerical},            # Currency value
    'DATS': {'pythontype': 'datetime', 'torch_frame_stype_': stype.timestamp},    # Date
    'DEC': {'pythontype': 'float', 'torch_frame_stype_': stype.numerical},             # Decimal
    'FLTP': {'pythontype': 'float', 'torch_frame_stype_': stype.numerical},            # Floating point number
    'INT2': {'pythontype': 'int', 'torch_frame_stype_': stype.numerical},              # 2-byte integer
    'INT8': {'pythontype': 'int', 'torch_frame_stype_': stype.numerical},              # 8-byte integer
    'LANG': {'pythontype': 'str', 'torch_frame_stype_': stype.text_embedded},          # Language key
    'NUMC': {'pythontype': 'str', 'torch_frame_stype_': stype.text_embedded},          # Numeric text (keeping as str to retain leading zeros)
    'QUAN': {'pythontype': 'float', 'torch_frame_stype_': stype.numerical},            # Quantity
    'TIMS': {'pythontype': 'datetime', 'torch_frame_stype_': stype.timestamp},    # Time
    'UNIT': {'pythontype': 'str', 'torch_frame_stype_': stype.text_embedded},          # Unit of measure
    'abap.char': {'pythontype': 'str', 'torch_frame_stype_': stype.text_embedded},     # ABAP character string
    'D34R': {'pythontype': 'str', 'torch_frame_stype_': stype.text_embedded},
    'ACCP': {'pythontype': 'str', 'torch_frame_stype_': stype.text_embedded},
    'INT4': {'pythontype': 'int', 'torch_frame_stype_': stype.numerical}
}



# In[140]:


def convert_kg4hana_datatype_to_stype_python_type(kg4hana_data):
  """
  Convert kg4hana data type to torch_frame stype
  
  Options:
  numerical = 'numerical'
  categorical = 'categorical'
  text_embedded = 'text_embedded'
  text_tokenized = 'text_tokenized'
  multicategorical = 'multicategorical'
  sequence_numerical = 'sequence_numerical'
  timestamp = 'timestamp'
  image_embedded = 'image_embedded'
  embedding = 'embedding'
  """
  
  # Traverse the dictionary and update the types
  updated_data = {}
  
  for df_name, columns in kg4hana_data.items():
      updated_columns = {}
      for col_name, col_type in columns.items():
          updated_columns[col_name] = kg4hana_type_to_stype_python_mapping.get(col_type,{'pythontype': 'str', 'torch_frame_stype_': stype.text_embedded})
      updated_data[df_name] = updated_columns
  
  return updated_data


# In[141]:


def find_non_overlapping_columns(cols1, cols2):
    
    """
    To find if we are missing any columns
    Find unique elements in two lists
    """

    # Find non-overlapping items
    only_in_list1 = cols1 - cols2
    only_in_list2 = cols2 - cols1
    
    # Combine the non-overlapping items
    non_overlapping_items = only_in_list1.union(only_in_list2)
    
    # Check if there are no non-overlapping items
    no_non_overlapping = len(non_overlapping_items) == 0
    assert no_non_overlapping
    return no_non_overlapping, non_overlapping_items

def infer_kg4hana_col_to_stype(db: Database, use_case_column_metadata, default_to_text_embed = False):
    """
    Infer column type based on kg4hana metadata. You can also use relbench like:
    
    col_to_stype_dict = get_stype_proposal(SOAC_db)

    import json

    # save inferred dtype
    with open('/Workspace/Repos/joseph.meyer@sap.com/gpt4hana-us-private/data_pipeline/inferred_use_case_data_type_metadata.json', 'w') as json_file:
        json.dump(col_to_stype_dict, json_file, indent=4, default=str)
    
    Args:
        db: Database object containing table information.
        
    Returns:
        dict: A dictionary with table names as keys and column stypes as values.
    """
    print('Using k4hana metadata to map columns to data types')
    col_to_stype_dict = {}
    # load kg4hana meta data for column types
    kg4hana_column_types = load_column_type_metadata(use_case_column_metadata)
    # Convert meta data to torch frame stype and python type
    kg_col_to_stype_dict = convert_kg4hana_datatype_to_stype_python_type(kg4hana_column_types)
    
    # Now map the columns we have to the type
    for table_name, table in db.table_dict.items():
        table_cols = {}
        # Available columns in the table
        cols = table.df.columns.tolist()
        for col in cols:
            # Find the stype mapping
            if col in kg_col_to_stype_dict.get(table_name, {}):
                table_cols[col] = kg_col_to_stype_dict[table_name][col]
            # column not found in kg4hana metadata
            else:
                if default_to_text_embed:
                    print(f'Warning, column type not found for {col} in table: {table_name}. Defaulting to text_embedded..')
                    # Default to text_embedded if not found
                    table_cols[col] ={'pythontype': 'str', 'torch_frame_stype_': stype.text_embedded}
                else:
                    # If we dont' find the column type in our metadata, try to infer it using torch frame
                    print(f'Warning, column type not found for {col} in table: {table_name}. Inferring stype..')
                    # use torch frame infer_dtype
                    series = table.df[col]
                    inferred_stype = infer_series_stype(series)

                    # If inference fails, force safe defaults
                    if inferred_stype is None:
                        inferred_type = {'pythontype': 'str', 'torch_frame_stype_': stype.text_embedded}
                    else:
                        inferred_type = {
                            'pythontype': stype_to_python_type_mapping.get(inferred_stype, 'str'),
                            'torch_frame_stype_': inferred_stype
                        }

                    table_cols[col] = inferred_type
                    print(f'For column: {col} in table: {table_name}. Inferred stype is..{inferred_type}')
      
        col_to_stype_dict[table_name] = table_cols

    # Verify we have covered all columns in the databases
    for table_name, table in db.table_dict.items():
      type_columns = set(col_to_stype_dict[table_name])
      table_columns = set(table.df.columns.tolist())

    no_non_overlapping, non_overlapping_items = find_non_overlapping_columns(type_columns, table_columns)
    if non_overlapping_items:
        print(f'Columns not covered in stype are: {non_overlapping_items}')
    
    return col_to_stype_dict


# In[142]:


# def extract_type(kg_col_to_stype_python_dict, key):
#     """
#     Extract just stype or python type for graph creation
#     """
#     result = {}
#     for table_name, columns in kg_col_to_stype_python_dict.items():
#         result[table_name] = {col_name: col_info[key] for col_name, col_info in columns.items()}
#     return result



#extracts one type like pythontype or stype.

def extract_type(kg_col_to_stype_python_dict, key):
    result = {}
    for table_name, columns in kg_col_to_stype_python_dict.items():
        table_result = {}
        if columns is None:
            result[table_name] = {}
            continue
        for col_name, col_info in columns.items():
            if isinstance(col_info, dict) and key in col_info:
                table_result[col_name] = col_info[key]
        result[table_name] = table_result
    return result


# In[143]:


# def coerce_fillna_for_column_type(df, kg, pkey_col, fkey_cols: list, 
#                                   table_name, target_col, fill_nas = False,
#                                   drop_na_thresh = 0,
#                                   time_split_col = 'CREATIONDATE'):
#     """
#     Convert each column in a pandas DataFrame to the specified type based on a nested dictionary,
#     and fill NaN values based on the column type with the mean, median, or mode as appropriate.
    
#     column_type_dict (dict): Dictionary from kg4hana data with column types for each table
#     """

#     # Show max number of columns
#     pd.set_option('display.max_columns', None)
#     # Get a sense of per tenant missingness
#     print(f'Null Counts before Imputation:\n{df.isnull().sum()}')

#     # Extract the python types
#     kg_col_to_python_dict = extract_type(kg, 'pythontype')
    
#     # Kg data types for each column
#     columns = kg_col_to_python_dict.get(table_name, {})

    
#     # --- Build the skip set (targets + keys + time split) ---
#     targets_to_skip = set(target_columns or [])
    
#     skip_cols = targets_to_skip | {pkey_col, time_split_col} | set(fkey_cols)
    
    
    
#     if drop_na_thresh > 0:
        
#         print(f'Dropping sparse columns with na % > {drop_na_thresh}')

#         # Calculate the percentage of missing values for each column
#         missing_percentage = df.isnull().mean()

#         # Set the percentage threshold for dropping columns
#         min_count = int((1 - drop_na_thresh) * len(df))

#         # Identify columns to drop based on the threshold
#         columns_to_drop = missing_percentage[missing_percentage > drop_na_thresh].index

        
        
#         # Protect any targets from being dropped
#         protected = columns_to_drop.intersection(targets_to_skip)
#         if len(protected) > 0:
#             print(f"[INFO] Not dropping protected target columns: {list(protected)}")
#             columns_to_drop = columns_to_drop.difference(protected)
    
        
        
#         # # Retain target column, sometimes it is very sparse
#         # if target_col in columns_to_drop:
#         #     columns_to_drop.remove(target_col)



#         # Show the columns that were removed and their % of missing values
#         removed_columns = missing_percentage[columns_to_drop]

#         # Drop columns with more than 'drop_na_thresh' % null values (inplace)
#         df.drop(columns=columns_to_drop, inplace=True)

#         # Prepare the removed columns DataFrame and show % missing (inplace operation)
#         removed_columns_df = removed_columns * 100  # Convert to percentage
#         removed_columns_df.name = 'Percent Missing'
#         removed_columns_df = removed_columns_df.reset_index().rename(columns={'index': 'Column'})

#         # Print or log the removed columns and their percentages
#         print("Removed Columns and their % Missing:")
#         print(removed_columns_df)
        
#     # Iterate over columns and apply conversions and fillna
#     for col_name in df.columns:
        
#         if col_name in skip_cols:
#             continue
        
        
#         # Get the type from KG4hana
#         col_type = columns.get(col_name)
        
    

#         # # Skip target, primary key, time split and foreign key columns
#         # if col_name == target_col or col_name == pkey_col or col_name in fkey_cols:
#         #     continue
        
#         # Remove nas for time column we split on
#         if col_name == time_split_col:
#             df.dropna(subset = [col_name], inplace = True, axis = 0)
        
#         if col_type:
#             try:
#                 if col_type == 'str':
#                     # This will convert nas to 'None'
#                     df[col_name] = df[col_name].astype(str)
#                     if fill_nas:
#                         if df[col_name].nunique() > 0:
#                             mode_value = df[col_name].mode()[0]
#                             df[col_name].fillna(mode_value, inplace=True)
#                 elif col_type == 'float':
#                     # Convert to float and fill NaN with the mean
#                     df[col_name] = pd.to_numeric(df[col_name], errors='coerce')
#                     if fill_nas:
#                         mean_value = df[col_name].mean()
#                         df[col_name].fillna(mean_value, inplace=True)
#                 elif col_type == 'int':
#                     # Convert to Int32 (more memory efficient than Int64) and fill NaN with the median
#                     df[col_name] = pd.to_numeric(df[col_name], errors='coerce').astype('Int32')
#                     if fill_nas:
#                         median_value = df[col_name].median()
#                         df[col_name].fillna(median_value, inplace=True)
#                 elif col_type in ['datetime', 'datetime.date', 'datetime.time']:
#                     # Convert to datetime and fill NaT with the median date
#                     df[col_name] = pd.to_datetime(df[col_name], errors='coerce')
#                     if fill_nas:
#                         if not df[col_name].isna().all():
#                             median_date = df[col_name].dropna().median()
#                             df[col_name].fillna(median_date, inplace=True)
#                 else:
#                     print(f'Unknown type {col_type} for column {col_name} in table {table_name}')
#             except Exception as e:
#                 # If there's an error, drop the column and collect garbage
#                 print(f'Error converting column {col_name} in table {table_name} to type {col_type}: {e}. Removing column.')
#                 df.drop(columns=col_name, inplace=True)
#                 gc.collect()
    
#     # Ensure primary key and foreign keys are int and convert them in place
#     try:
#         df[pkey_col] = pd.to_numeric(df[pkey_col], errors='coerce').astype('Int32')
#     except Exception as e:
#         print(f'Error converting primary key column {pkey_col} to int: {e}')
    
#     for fkey_col in fkey_cols:
#         try:
#             df[fkey_col] = pd.to_numeric(df[fkey_col], errors='coerce').astype('Int32')
#         except Exception as e:
#             print(f'Error converting foreign key column {fkey_col} to int: {e}')
    
#     # Drop columns with all NaN values and collect garbage
#     initial_columns = df.shape[1]
#     #df.dropna(axis=1, how='all', inplace=True)
#     final_columns = df.shape[1]
#     gc.collect()

#     # Calculate null counts and proportions
#     null_counts = df.isnull().sum()
#     total_length = len(df)

#     # Print nulls as a portion of the total length
#     print(f'Proportion of nulls: {null_counts / total_length}')
#     return df







def coerce_fillna_for_column_type(df, kg, pkey_col, fkey_cols: list, 
                                  table_name, target_col, fill_nas = False,
                                  drop_na_thresh = 0,
                                  time_split_col = 'CREATIONDATE'):
    """
    Convert each column in a pandas DataFrame to the specified type based on a nested dictionary,
    and fill NaN values based on the column type with the mean, median, or mode as appropriate.
    
    column_type_dict (dict): Dictionary from kg4hana data with column types for each table
    """

    # Show max number of columns
    pd.set_option('display.max_columns', None)
    # Get a sense of per tenant missingness
    print(f'Null Counts before Imputation:\n{df.isnull().sum()}')

    # Extract the python types
    kg_col_to_python_dict = extract_type(kg, 'pythontype')
    
    # Kg data types for each column
    columns = kg_col_to_python_dict.get(table_name, {})

    # --- NEW FIX: Ensure pkey_col is safely flattened into a list for set math ---
    pkey_list = pkey_col if isinstance(pkey_col, list) else ([pkey_col] if pkey_col else [])
    
    # --- Build the skip set safely ---
    # target_columns is assumed to be defined globally in your script based on your traceback
    targets_to_skip = set(target_columns or [])
    
    skip_cols = targets_to_skip | set(pkey_list) | {time_split_col} | set(fkey_cols)
    
    if drop_na_thresh > 0:
        
        print(f'Dropping sparse columns with na % > {drop_na_thresh}')

        # Calculate the percentage of missing values for each column
        missing_percentage = df.isnull().mean()

        # Set the percentage threshold for dropping columns
        min_count = int((1 - drop_na_thresh) * len(df))

        # Identify columns to drop based on the threshold
        columns_to_drop = missing_percentage[missing_percentage > drop_na_thresh].index

        # Protect any targets from being dropped
        protected = columns_to_drop.intersection(targets_to_skip)
        if len(protected) > 0:
            print(f"[INFO] Not dropping protected target columns: {list(protected)}")
            columns_to_drop = columns_to_drop.difference(protected)
    
        # Show the columns that were removed and their % of missing values
        removed_columns = missing_percentage[columns_to_drop]

        # Drop columns with more than 'drop_na_thresh' % null values (inplace)
        df.drop(columns=columns_to_drop, inplace=True)

        # Prepare the removed columns DataFrame and show % missing (inplace operation)
        removed_columns_df = removed_columns * 100  # Convert to percentage
        removed_columns_df.name = 'Percent Missing'
        removed_columns_df = removed_columns_df.reset_index().rename(columns={'index': 'Column'})

        # Print or log the removed columns and their percentages
        print("Removed Columns and their % Missing:")
        print(removed_columns_df)
        
    # Iterate over columns and apply conversions and fillna
    for col_name in df.columns:
        
        if col_name in skip_cols:
            continue
        
        # Get the type from KG4hana
        col_type = columns.get(col_name)
        
        # Remove nas for time column we split on
        if col_name == time_split_col:
            df.dropna(subset = [col_name], inplace = True, axis = 0)
        
        if col_type:
            try:
                if col_type == 'str':
                    # This will convert nas to 'None'
                    df[col_name] = df[col_name].astype(str)
                    if fill_nas:
                        if df[col_name].nunique() > 0:
                            mode_value = df[col_name].mode()[0]
                            df[col_name].fillna(mode_value, inplace=True)
                elif col_type == 'float':
                    # Convert to float and fill NaN with the mean
                    df[col_name] = pd.to_numeric(df[col_name], errors='coerce')
                    if fill_nas:
                        mean_value = df[col_name].mean()
                        df[col_name].fillna(mean_value, inplace=True)
                elif col_type == 'int':
                    # Convert to Int32 (more memory efficient than Int64) and fill NaN with the median
                    df[col_name] = pd.to_numeric(df[col_name], errors='coerce').astype('Int32')
                    if fill_nas:
                        median_value = df[col_name].median()
                        df[col_name].fillna(median_value, inplace=True)
                elif col_type in ['datetime', 'datetime.date', 'datetime.time']:
                    # Convert to datetime and fill NaT with the median date
                    df[col_name] = pd.to_datetime(df[col_name], errors='coerce')
                    if fill_nas:
                        if not df[col_name].isna().all():
                            median_date = df[col_name].dropna().median()
                            df[col_name].fillna(median_date, inplace=True)
                else:
                    print(f'Unknown type {col_type} for column {col_name} in table {table_name}')
            except Exception as e:
                # If there's an error, drop the column and collect garbage
                print(f'Error converting column {col_name} in table {table_name} to type {col_type}: {e}. Removing column.')
                df.drop(columns=col_name, inplace=True)
                gc.collect()
    
    # --- NEW FIX: Iterate over primary keys since it might be a list of combinations ---
    for p_col in pkey_list:
        if p_col in df.columns:
            try:
                df[p_col] = pd.to_numeric(df[p_col], errors='coerce').astype('Int32')
            except Exception as e:
                print(f'Error converting primary key column {p_col} to int: {e}')
    
    for fkey_col in fkey_cols:
        if fkey_col in df.columns:
            try:
                df[fkey_col] = pd.to_numeric(df[fkey_col], errors='coerce').astype('Int32')
            except Exception as e:
                print(f'Error converting foreign key column {fkey_col} to int: {e}')
    
    # Drop columns with all NaN values and collect garbage
    initial_columns = df.shape[1]
    #df.dropna(axis=1, how='all', inplace=True)
    final_columns = df.shape[1]
    gc.collect()

    # Calculate null counts and proportions
    null_counts = df.isnull().sum()
    total_length = len(df)

    # Print nulls as a portion of the total length
    print(f'Proportion of nulls: {null_counts / total_length}')
    return df


# In[144]:


def fill_na_for_column_type_database(db: Database, column_type_dict, target_col):

  for table_name, table in db.table_dict.items():
    # infer pkey and fkey col from db object
    pkey_col = table.pkey_col
    fkey_data = table.fkey_col_to_pkey_table
    fkey_col = list(fkey_data.keys())
    
    # --- FIX: Grab the actual time column ---
    time_col = table.time_col 
    
    df = table.df
    
    # --- FIX: Pass time_split_col=time_col so it gets protected! ---
    df = coerce_fillna_for_column_type(
        df, column_type_dict, pkey_col, fkey_col, 
        table_name, target_col, time_split_col=time_col
    )
    
    db.table_dict[table_name].df = df
  
  return db


# In[145]:


def count_minus_100(df, target_cols):
    for col in target_cols:
        count = (df[col] == -100).sum()
        print(f"Column: {col} | Count of -100: {count}")
    print(df[target_columns].dtypes)

count_minus_100(SOAC_db.table_dict[entity_table].df, target_columns)


# In[146]:


SOAC_db.table_dict[entity_table].df[target_columns].isna().sum()


# In[147]:


kg_col_to_stype_python_dict = infer_kg4hana_col_to_stype(SOAC_db,use_case_column_metadata)
kg_col_to_stype_python_dict


# In[148]:


def count_minus_100(df, target_cols):
    for col in target_cols:
        count = (df[col] == -100).sum()
        print(f"Column: {col} | Count of -100: {count}")
    print(df[target_columns].dtypes)

count_minus_100(SOAC_db.table_dict[entity_table].df, target_columns)


# In[149]:


SOAC_db.table_dict[entity_table].df[target_columns].isna().sum()


# In[150]:


SOAC_db.target_col


# In[151]:


SOAC_db = fill_na_for_column_type_database(SOAC_db, kg_col_to_stype_python_dict, SOAC_db.target_col)

# redo after removing columns for now
kg_col_to_stype_python_dict = infer_kg4hana_col_to_stype(SOAC_db, use_case_column_metadata)
kg_col_to_stype_python_dict


# In[152]:


def count_minus_100(df, target_cols):
    for col in target_cols:
        count = (df[col] == -100).sum()
        print(f"Column: {col} | Count of -100: {count}")
    print(df[target_columns].dtypes)

count_minus_100(SOAC_db.table_dict[entity_table].df, target_columns)


# In[153]:


SOAC_db.table_dict[entity_table].df[target_columns].isna().sum()


# In[154]:


kg_col_to_stype_dict = extract_type(kg_col_to_stype_python_dict, 'torch_frame_stype_')

kg_col_to_stype_dict


# In[155]:


def count_minus_100(df, target_cols):
    for col in target_cols:
        count = (df[col] == -100).sum()
        print(f"Column: {col} | Count of -100: {count}")
    print(df[target_columns].dtypes)

count_minus_100(SOAC_db.table_dict[entity_table].df, target_columns)


# In[156]:


SOAC_db.table_dict[entity_table].df[target_columns].isna().sum()


# In[157]:


def remove_target_from_col_to_stype(kg_col_to_stype_dict, db: Database):
    """
    Prevent target in training set leakage
    This removes it from the col:stype dict, which then does not include it in constructing the graph
    """
    # Ensure db.target_col is a list, even if a single target is provided
    target_cols = db.target_col if isinstance(db.target_col, list) else [db.target_col]
    print(target_cols)
    
    for table, types in kg_col_to_stype_dict.items():
        for target_col in target_cols:
            # Perform the pop operation for each target column
            if target_col in types:
                print(f'Removing target column {target_col} from the col_to_stype_dict for table: {table}')
                kg_col_to_stype_dict[table].pop(target_col, None)
                # Ensure the target column is not in the dictionary anymore
                assert target_col not in kg_col_to_stype_dict[table]

    return kg_col_to_stype_dict


# In[158]:


kg_col_to_stype_dict = remove_target_from_col_to_stype(kg_col_to_stype_dict, SOAC_db)
kg_col_to_stype_dict


# In[159]:


def count_minus_100(df, target_cols):
    for col in target_cols:
        count = (df[col] == -100).sum()
        print(f"Column: {col} | Count of -100: {count}")
    print(df[target_columns].dtypes)

count_minus_100(SOAC_db.table_dict[entity_table].df, target_columns)


# In[160]:


SOAC_db.table_dict[entity_table].df[target_columns].isna().sum()


# In[161]:


def is_missing_or_none(val):
    if pd.isna(val):
        return True
    val_str = str(val).strip().lower()
    return val_str in ["", "none", "nan"]

mask = SOAC_db.table_dict[entity_table].df[target_columns].applymap(is_missing_or_none)
filtered =SOAC_db.table_dict[entity_table].df[target_columns][mask.any(axis=1)]

print(filtered)


# In[162]:


print(filtered)


# **Drop stype.timestamp columns with more than 60 percent null values and that does not appear in time_col and it syncs with kg_col_stype metadata to keep it upto date**

# In[163]:


def drop_sparse_timestamps(
    db_obj,  # A RelBench-style DB object whose `.table_dict` holds Table objects
    kg_col_to_stype_dict: Mapping[str, Mapping[str, str]],  # table → (column → stype) map
    *,  # Forces everything after this to be passed by keyword
    threshold: float = 0.60,  # Minimum fraction (0‑1) of missing values to trigger deletion
) -> None:  # The function mutates in place and returns nothing
    """
    Remove timestamp columns that
      1. are NOT the designated `time_col` for the table, and
      2. are >= `threshold` (default 60 %) missing (None, NaN, <NA>, 'null', 'none').

    The function mutates:
      • the DataFrames inside `db_obj.table_dict[...].df`
      • `kg_col_to_stype_dict` (off-board metadata)

    Parameters
    ----------
    db_obj : object
        Any object whose `.table_dict` maps table names → Table objects with
        attributes `.df` (pandas.DataFrame) and `.time_col` (str | None).
    kg_col_to_stype_dict : dict[str, dict[str, stype]]
        Column-to-stype mapping produced by RelBench (`stype.timestamp`, …).
        It is updated in place so it stays in sync with the pruned DataFrames.
    threshold : float, default 0.60
        Minimum fraction of missing values required to drop a column.
    """

    # A set of lower‑cased textual tokens that should be treated as NULLs as well
    _NULL_STRS = {"none", "null", "<na>", "nat","nan", "n/a"}  # Hash‑set for O(1) membership checks

    # ---------------------------------------------------------------------
    # Walk through every table registered in the DB object
    # ---------------------------------------------------------------------
    for tbl_name, table in db_obj.table_dict.items():  # tbl_name: str, table: Table
        df = table.df  # The raw pandas DataFrame for this table
        time_col = table.time_col  # Canonical timestamp column to *keep* (may be None)

        # ---------------- identify timestamp columns --------------------
        stype_map = kg_col_to_stype_dict.get(tbl_name, {})  # Column → stype mapping
        ts_cols = [
            c for c, st in stype_map.items()
            # `st` is sometimes an Enum with a `.value`; otherwise it's already a str.
            if getattr(st, "value", str(st)) == "timestamp"
        ]  # All columns tagged as timestamp in the KG metadata

        # ---------------- evaluate each candidate -----------------------
        to_drop = []  # Collect columns that pass the two drop rules
        for col in ts_cols:
            # Rule 1: never drop the designated time column
            if col == time_col:
                continue  # Skip deletion

            pk_cols = [table.pkey_col] if isinstance(table.pkey_col, str) else table.pkey_col or []
            
            # do not remove columns that are still listed as a foreign key
            if (col in table.fkey_col_to_pkey_table) or (col in pk_cols):
                continue
            
            
            # Extra guard: ensure the column actually exists in the DataFrame
            if col not in df.columns:
                continue  # Metadata/DataFrame mismatch; ignore

            # Rule 2: compute proportion of missing values (multi‑modal NULLs)
            ser = df[col]  # Shorthand Series reference
            null_mask = ser.isna() | ser.astype(str).str.lower().isin(_NULL_STRS)  # True where value is null‑like
            if null_mask.mean() >= threshold:  # Fraction of nulls meets/exceeds threshold?
                to_drop.append(col)  # Mark this column for removal

        # ---------------- apply the pruning -----------------------------
        if to_drop:  # Only touch structures if something qualified for dropping
            df.drop(columns=to_drop, inplace=True)  # Mutate DataFrame in place
            for col in to_drop:
                stype_map.pop(col, None)

    # Function intentionally returns None — callers already see mutated structures as they
    # kept the original object references.


    return db_obj, kg_col_to_stype_dict


    # db_obj and kg_col_to_stype_dict are now mutated


# In[164]:


#both are updated and mutated now


SOAC_db, kg_col_to_stype_dict = drop_sparse_timestamps(SOAC_db, kg_col_to_stype_dict)


# In[165]:


SOAC_db.table_dict


# In[166]:


SOAC_db.table_dict[entity_table].df[target_columns].isna().sum()


# In[167]:


def count_minus_100(df, target_cols):
    for col in target_cols:
        count = (df[col] == -100).sum()
        print(f"Column: {col} | Count of -100: {count}")
    print(df[target_columns].dtypes)

count_minus_100(SOAC_db.table_dict[entity_table].df, target_columns)


# **Turning the regression targets to float64 before it enters the task class as we want all the regression targets across all the tenants to be of same data type, so we deal with lot of merges and NaN values so float64 is safer**
# 
# 
# **for classification columns**
# 
# **1. Original None/NaN values are kept as missing (null) values.**
# 
# **2. Empty strings are retained as empty strings (not converted to nulls).**
# 
# **3. Any float-like integers or string representations of integers in classification columns are converted to proper integers.**

# In[168]:


SOAC_db.table_dict[entity_table].df[target_columns].isna().sum()


# In[169]:


SOAC_db.table_dict[entity_table].df


# In[170]:


# Filter rows where any target column is an empty string
filtered_rows = SOAC_db.table_dict[entity_table].df[target_columns][SOAC_db.table_dict[entity_table].df[target_columns].eq("").any(axis=1)]

print(filtered_rows)


# In[171]:


def is_missing_or_none(val):
    if pd.isna(val):
        return True
    val_str = str(val).strip().lower()
    return val_str in ["", "none", "nan"]

mask = SOAC_db.table_dict[entity_table].df[target_columns].applymap(is_missing_or_none)
filtered = SOAC_db.table_dict[entity_table].df[target_columns][mask.any(axis=1)]

filtered


# In[172]:


SOAC_db.table_dict[entity_table].df[target_columns].dtypes


# In[173]:


SOAC_db.table_dict[entity_table].df


# In[174]:


# Turning the regression targets to float64 before it enters the task class as we want all the regression targets across all the tenants to be of same data type, so we deal with lot of merges and NaN values so float64 is safer


# for classification columns

# 1. Original None/NaN values are kept as missing (null) values.

# 2. Empty strings are retained as empty strings (not converted to nulls).

# 3. Any float-like integers or string representations of integers in classification columns are converted to proper integers.




def set_targets_to_float64_in_table_dict(
    table_dict: Dict[str, Any],          # mapping: table name -> object exposing `.df` (pandas DataFrame)
    target_columns: List[str],           # list of target column names to process
    target_tasks: List[str],             # parallel list of tasks ("regression" / "classification")
) -> Dict[str, List[str]]:
    """
    For each table in `table_dict`:
      • If target task == "regression": force that column to float64 (coerce non-parsable → NaN).
      • If target task == "classification":
          - Preserve NaN/None exactly as-is
          - Preserve empty/whitespace strings exactly as-is
          - Convert only whole-number values to Python int
              (int → int; float with .is_integer() → int; "2", " 2 ", "2.0" → int)
          - Leave non-integer floats (e.g., 2.3 / "2.3") and other strings unchanged.

    Returns a summary: {table_name: [converted_target_cols_in_that_table, ...]}.
    """

    # --- Build a per-target mapping from column -> task to avoid comparing the whole list later ---
    #     (This fixes the earlier bug where we compared `target_tasks == "regression"`.)
    column_to_task: Dict[str, str] = {col: task for col, task in zip(target_columns, target_tasks)}

    # --- Helper: classification-safe coercion (preserve missing/empties; coerce whole numbers to int) ---
    def _coerce_classification_value_keep_missing(v):
        """Return a value for classification targets with the following rules:
           - Keep NaN/None as-is
           - Keep empty strings (and whitespace-only) as-is
           - If value is whole number (numeric int; float with no fractional part; or a numeric-looking string like '2'/'2.0'):
             return Python int
           - Otherwise return value unchanged
        """
        # Keep native missing as-is (pd.NA, NaN, None, NaT)
        if pd.isna(v):
            return v

        # If it's a string, examine trimmed form to decide
        if isinstance(v, str):
            trimmed = v.strip()
            # Keep empty/whitespace-only strings as-is
            if trimmed == "":
                return v
            # Try to interpret as integer directly (e.g., "2")
            if trimmed.isdigit() or (trimmed.startswith(('+', '-')) and trimmed[1:].isdigit()):
                try:
                    return int(trimmed)
                except Exception:
                    return v  # if somehow fails, leave unchanged
            # Try to parse as float and reduce to int if it is an exact integer (e.g., "2.0")
            try:
                f = float(trimmed)
                if float.is_integer(f):
                    return int(f)
                return v  # non-integer float like "2.3" → leave unchanged
            except Exception:
                return v  # non-numeric strings → leave unchanged

        # If it's already a number, normalize only if it's a whole number
        if isinstance(v, (int, np.integer)):
            return int(v)  # ensure pure Python int
        if isinstance(v, (float, np.floating)):
            return int(v) if float(v).is_integer() else v  # 2.0→2, 2.3→2.3 (unchanged)

        # Any other type → leave unchanged
        return v

    # --- Summary we will return: which columns were converted per table ---
    summary: Dict[str, List[str]] = {}

    # --- Iterate all tables in the dictionary ---
    for table_name, table_obj in table_dict.items():
        # Pull the DataFrame from the table-like object
        df = getattr(table_obj, "df", None)

        # If no DataFrame → skip
        if df is None or not isinstance(df, pd.DataFrame):
            print(f"[warn] '{table_name}' has no valid .df DataFrame — skipped")
            continue

        # If DataFrame is empty → nothing to do
        if df.empty:
            print(f"[info] DataFrame for '{table_name}' is empty — nothing to convert")
            continue

        # Track columns we actually changed in this table
        converted_here: List[str] = []

        # Process each requested target column
        for target_col in target_columns:
            # Only touch columns that exist in the current DataFrame
            if target_col not in df.columns:
                print(f"[info] '{target_col}' not found in '{table_name}' — skipped")
                continue

            # Determine the task for this target (default to classification if unspecified)
            task = column_to_task.get(target_col, "classification")

            # Remember dtype before changes (for logging)
            before_dtype = df[target_col].dtype

            if task == "regression":
                # --- Regression: force to float64 with numeric coercion ---
                # Convert anything non-parsable to NaN; then set dtype to float64
                df[target_col] = pd.to_numeric(df[target_col], errors="coerce").astype("float64", copy=False)

            else:
                # --- Classification: preserve NaN and empty strings; coerce ONLY whole numbers to int ---
                # Apply value-wise coercion; dtype will likely stay 'object', which is desired
                df[target_col] = df[target_col].apply(_coerce_classification_value_keep_missing)

            # Capture dtype after changes (for logging)
            after_dtype = df[target_col].dtype

            # If dtype changed or we ran a classification normalization, record this target in summary
            if (before_dtype != after_dtype) or (task == "classification"):
                converted_here.append(target_col)

            # Log what happened for this column
            print(
                f"[ok] {table_name}.{target_col}: task={task} | dtype {before_dtype} -> {after_dtype}"
            )

        # If we changed anything for this table, record it
        if converted_here:
            summary[table_name] = converted_here

    # Return the per-table summary for visibility/validation
    return summary





# Example:
summary = set_targets_to_float64_in_table_dict(
    table_dict=SOAC_db.table_dict,
    target_columns=target_columns,
    target_tasks=target_tasks,
)

print("Summary:", summary)


# In[175]:


SOAC_db.table_dict[entity_table].df[target_columns].dtypes


# In[176]:


# Filter rows where any target column is an empty string
filtered_rows = SOAC_db.table_dict[entity_table].df[target_columns][SOAC_db.table_dict[entity_table].df[target_columns].eq("").any(axis=1)]

print(filtered_rows)


# In[177]:


def count_minus_100(df, target_cols):
    for col in target_cols:
        count = (df[col] == -100).sum()
        print(f"Column: {col} | Count of -100: {count}")
    print(df[target_columns].dtypes)

count_minus_100(SOAC_db.table_dict[entity_table].df, target_columns)


# In[178]:


SOAC_db.table_dict[entity_table].df[target_columns].isna().sum()


# In[179]:


def is_missing_or_none(val):
    if pd.isna(val):
        return True
    val_str = str(val).strip().lower()
    return val_str in ["", "none", "nan"]

mask = SOAC_db.table_dict[entity_table].df[target_columns].applymap(is_missing_or_none)
filtered =SOAC_db.table_dict[entity_table].df[target_columns][mask.any(axis=1)]

filtered


# **SOAC_Task**

# In[180]:


def f1(true, pred, thresh = 0.5) -> float:
    assert pred.ndim == 1 or pred.shape[1] == 1
    label = pred >= thresh
    return skm.f1_score(true, label, average="macro", zero_division = 1)

def macro_precision(true, pred) -> float:
    assert pred.ndim > 1
    label = pred.argmax(axis=1)
    return skm.precision_score(true, label, average="macro", zero_division = 1)


# In[181]:


target_columns


# In[182]:


SOAC_db.target_col


# In[210]:


def filter_table_for_merge(table):
    table_name = next((name for name, t in SOAC_db.table_dict.items() if t == table), None)
    if table_name is None:
        raise ValueError("Table not found in SOAC_db.table_dict")
    
    # Get all target columns present in the table
    present_targets = [col for col in SOAC_db.target_col if col in table.df.columns]

    # Get primary key (handle both string and list formats)
    pkey = table.pkey_col
    if not isinstance(pkey, list):
        pkey = [pkey]
        
    # Get foreign keys
    fkeys = list(table.fkey_col_to_pkey_table.keys())
    
    # Safely handle the time_col in case it is None
    time_col = [table.time_col] if table.time_col else []
    
    # add the __idx__ column to map LGM data to the STM data
    entity_name = table_name.replace("df_", "").upper()
    idx_col = f"__idx_{entity_name}__"
    idx_col=[idx_col]
    
    
    # Combine the core columns to keep
    cols = pkey + present_targets + fkeys + time_col + idx_col
    
    # --- NEW FIX: Preserve the __row_order__ column ---
    # If the original table has __row_order__, we MUST keep it so make_table() 
    # can use it for the train/test/val splits later.
    
    
    # if '__row_order__' in table.df.columns:
    #     cols.append('__row_order__')

    
        
    # Remove any potential duplicates from the list (just to be safe)
    cols = list(dict.fromkeys(cols))
    
    # Slice the dataframe to keep only the requested columns
    df = table.df[cols]
    
    return df


# In[205]:


SOAC_db.table_dict


# In[206]:


SOAC_db.table_dict[entity_table].df.dtypes


# In[211]:


full_train_table=filter_table_for_merge(SOAC_db.table_dict[entity_table])

full_train_table


# In[212]:


full_train_table.isna().sum()


# In[213]:


full_train_table


# In[214]:


def count_minus_100(df, target_cols):
    for col in target_cols:
        count = (df[col] == -100).sum()
        print(f"Column: {col} | Count of -100: {count}")
    print(df[target_columns].dtypes)

count_minus_100(full_train_table, target_columns)


# **Convert NaN and mask = -100 to float values**

# In[215]:


# # Convert NaN and mask = -100 to float values


# def coerce_mask_and_nan_to_float(
#     df: pd.DataFrame,               # df: the DataFrame to modify (mutated in place and also returned)
#     columns,                        # columns: a column name or iterable of column names to process
#     minus100_value: float = -100.0, # minus100_value: the float value to use when we detect "-100" variants
#     null_tokens = ("NA", "Na", "na",
#         "N/A", "n/a", "Na/", "NA/", "na/",
#         "NAN", "NaN", "nan",
#         "NONE", "None", "none",
#         "NULL", "Null", "null",
#         "NIL", "Nil", "nil",
#         "NILL", "Nill", "nill",
#         "NAT", "Nat", "nat", "NaT",
#         "-", "--",)  # strings that should be treated as null (case-insensitive)
# ) -> pd.DataFrame:
#     """
#     For each column in `columns`, convert ONLY:
#       • -100 (numeric or string like "-100" / "-100.0") → float -100.0
#       • null-like tokens (nan/none/null/na/n/a/<na>, any case) → np.nan
#     Leaves empty strings "" (and other values) untouched.
#     """                                  # Docstring describing the purpose and behavior of the function

#     if isinstance(columns, (str, int)):  # If a single column label (string/int) was passed instead of a list/tuple...
#         columns = [columns]              # ...wrap it in a list so we can iterate uniformly below.

#     def fix_cell(v):                     # Inner helper that converts a single cell value according to the rules
#         # real missing → np.nan
#         if pd.isna(v):                   # If pandas already considers this value missing (NaN, None, <NA>, etc.)
#             return np.nan                # ...normalize to a consistent np.nan

#         # numeric -100 → -100.0
#         if isinstance(v, (int, float, np.integer, np.floating)) and v == -100:
#                                          # If it's a numeric value equal to -100 (covers Python and NumPy scalars)...
#             return float(minus100_value) # ...return it as a float (default -100.0, but user can override)

#         # strings: check empties, null tokens, and "-100"
#         if isinstance(v, str):           # If the value is a string, we may need to treat some strings specially
#             t = v.strip()                # Trim surrounding whitespace for robust comparisons
#             # keep empty strings as-is (important for your label encoding)
#             if t == "":                  # If the string is empty after trimming...
#                 return v                 # ...leave it unchanged (you want to keep "" for label encoding)

#             # null-like strings → np.nan
#             if t in null_tokens: # If the lowercase text matches any of the declared null tokens...
#                 return np.nan            # ...treat it as missing

#             # catch string "-100", "-100.0", also unicode minus "−100"
#             t_numeric = t.replace("-", "-")  # Replace unicode minus (U+2212) with ASCII hyphen-minus for parsing
#             try:
#                 f = float(t_numeric)     # Try to parse the (possibly normalized) string as a float
#                 if f == -100.0:          # If it parses and equals -100.0...
#                     return float(minus100_value)  # ...emit the chosen sentinel float (default -100.0)
#             except ValueError:
#                 pass                     # Not a number (e.g., "abc"): do nothing and fall through

#         # everything else unchanged
#         return v                         # If no rule matched, return the original value unchanged

#     for col in columns:                  # Process each requested column
#         # ensure mixed types are allowed in the column
#         if df[col].dtype != "object":    # If the column has a strict dtype (e.g., int/float) that would block strings...
#             df[col] = df[col].astype("object")  # ...cast to "object" so we can store mixed types during conversion
#         df[col] = df[col].apply(fix_cell)  # Apply the cell fixer to every value in the column

#     return df                            # Return the same DataFrame (also already modified in place)


# In[216]:


# convert = coerce_mask_and_nan_to_float(full_train_table, target_columns)


# In[217]:


full_train_table


# In[218]:


full_train_table.isna().sum()


# In[219]:


import numpy as np

# This creates a boolean mask: True if any target column is -100 or -100.0 in a row
mask = (full_train_table[target_columns] == -100) | (full_train_table[target_columns] == -100.0)

# .any(axis=1) finds rows where any target column matches
rows_with_minus_100 = full_train_table[mask.any(axis=1)]

print(rows_with_minus_100)


# In[220]:


full_train_table


# In[221]:


target_columns


# In[222]:


for col in target_columns:
    print(f"Column: {col}")
    print(full_train_table[col].value_counts(dropna=False))
    print("="*40)


# In[223]:


# # to make string NaN to actual pandas NaN

# full_train_table[target_columns] = full_train_table[target_columns].astype(float)

# full_train_table


# In[224]:


from relbench.base import EntityTask, TaskType
from relbench.metrics import accuracy, average_precision, roc_auc, mae, mse, r2, rmse, macro_f1

from collections import OrderedDict

class SOAC_Task(EntityTask):
    ################################################################################
    # Use docstrings to describe the task
    ################################################################################
    r"""SOAC"""

    ################################################################################
    # Fill out the task attributes
    ################################################################################
    
    def __init__(self, database: Database, entity_table, use_case, full_train_table,
                 target_col = 'SHIPPINGCONDITION', cache_dir = None, multi_task = True):
        
        """
        We may also init with entity_col = 'SALESDOCUMENT', 
        entity_table = 'df_salesdocument', time_col = 'CREATIONDATE', 
        """

        
        self.use_case=use_case
        self.entity_table = entity_table
        
        self.database = database

        self.multi_task = multi_task
        if self.multi_task:
            self.training_table = full_train_table
        self.dataset = relbench_dataset(tenant_id = tenant_id, cleaned_tables = clean_dict, use_case = use_case, multi_task = True, tenant_metadata = tenant_metadata, database_schema = database_schema, metadata_semantics = usecase_metadata_semantics, target_columns = target_columns, cache_dir = None)
        # Set target variable to all targets
        # In the multi-task setting don't include targets that aren't in the training table so we may not have 8 heads
        targs = target_col if not self.multi_task else [targ for targ in self.dataset.task_metadata['available_targets'] if targ in self.training_table.columns.tolist()]
        self.target_col = list(OrderedDict.fromkeys(targs))

        # infer task table information
        self.infer_entity_info()
        # infer task type given target column
        self.infer_task_type()
        # infer performance metrics
        self.infer_metrics()
        self.cache_dir = cache_dir


            
    def infer_entity_info(self, auto_infer = False):

        """
        Given target column, find entity column, entity table, time_col

        SalesOffice, SalesGroup, CustomerPaymentTerms and ShippingCondition are always predicted on header (higher granularity) level, no matter of they are present on the item level. Plant and Shipping point are always predicted on item level, even if they are present on header level. 

        """
        target = self.target_col

        if auto_infer:
            # For the SOAC task, we should predict at certain granularities
            entity_item_header_granularity = {'df_salesdocument': ['SALESOFFICE', 'SALESGROUP', 'CUSTOMERPAYMENTTERMS','SHIPPINGCONDITION', 'HEADERINCOTERMSCLASSIFICATION'],
                                'df_salesdocumentitem': ['PLANT', 'SHIPPINGPOINT', 'ITEMINCOTERMSCLASSIFICATION']}
            
            # For the targets with a specified granularity, we find the entity correct entity table
            entity_table = None
            for key, value in entity_item_header_granularity.items():
                if target in value:
                    entity_table = key
                    break
            
            assert entity_table is not None

        elif self.multi_task:
            
            # Verify this makes sense
            entity_table = self.entity_table
    
        # Else, find the target table and meta info        
        else:
            raise NotImplementedError("This method needs to be implemented.")
        
        # Get entity col, time col
        target_rel_table = self.database.table_dict[entity_table]

        # entity_col = target_rel_table.fkey_col_to_pkey_table
        # Relbench recommends to use fkey as entity col. I believe this informs neighbor sampling and connection between tables
        # However, our model has not had issues learning
        entity_col = target_rel_table.pkey_col

        time_col = target_rel_table.time_col

        print(f'For the given target column: {target}, we have inferred \n the task_table as: {entity_table}, entity_col: {entity_col}, time_col: {time_col}')

        # store information
        self.entity_col = entity_col
        self.entity_table = entity_table
        self.time_col = time_col


    def infer_metrics(self):

        """
        Infer metric after task_type is inferred
        """
        # Single task
        if not self.multi_task:
            if self.task_type == TaskType.BINARY_CLASSIFICATION:
                self.metrics = [average_precision, accuracy, f1]
            elif self.task_type == TaskType.MULTICLASS_CLASSIFICATION:
                self.metrics = [accuracy, macro_f1, macro_precision]  # Adjust metrics for multi-class classification if needed
            elif self.task_type == TaskType.REGRESSION:
                self.metrics = [mse, mae]
        
        # multi task
        if self.multi_task:
            # performance metrics for each head
            metrics = {}
            for task, task_type in self.task_type.items():
                if task_type['task_type'] == TaskType.BINARY_CLASSIFICATION:
                    metrics[task] = [average_precision, accuracy, f1]
                elif task_type['task_type'] == TaskType.MULTICLASS_CLASSIFICATION:
                    metrics[task] = [accuracy, macro_f1, macro_precision]
                elif task_type['task_type'] == TaskType.REGRESSION:
                    metrics[task] = [mse, mae]
            self.metrics = metrics

    def evaluate(self, pred, target_table, metrics = None):

        print('Evaluating performance....')
        if metrics is None:
            metrics = self.metrics

        if self.multi_task:
            # Compute metrics for each head's task
            results = {}

            # Loop over each task and compute the metrics
            for task_name, task_metrics in metrics.items():
                task_pred = pred[task_name]  # Get the predictions for the specific task
                task_target = target_table.df[task_name].to_numpy()  # Get the true labels for the task
                if len(task_pred) != len(task_target):
                    raise ValueError(
                        f"The length of pred and target must be the same for task {task_name} "
                        f"(got {len(task_pred)} and {len(task_target)}, respectively)."
                    )

                if self.task_type[task_name]['task_type'] == TaskType.BINARY_CLASSIFICATION:
                    print(task_pred)
                    # Take the first class logit
                    if task_pred.dim() == 2 and task_pred.size(-1) == 2:
                        # Take the first class logit
                        task_pred = task_pred[:, 0]
                    # binarize
                    task_pred = task_pred >= 0.5
                    class_report = classification_report(task_target, task_pred, zero_division=1, output_dict=True)
                elif self.task_type[task_name]['task_type'] == TaskType.MULTICLASS_CLASSIFICATION:
                    # argmax multiclass predictions
                    argmax_pred = np.argmax(task_pred, axis=1)
                    class_report = classification_report(task_target, argmax_pred, zero_division=1, output_dict=True)
                else:
                    pass
                
                # Evaluate the metrics for the task
                task_results = {fn.__name__: fn(task_target, task_pred) for fn in task_metrics}
                # Add classification report for each head
                task_results["classification_report"] = class_report
                results[task_name] = task_results
            return results

        else:
            # Singel task
            target = target_table.df[self.target_col].to_numpy()
            if len(pred) != len(target):
                raise ValueError(
                    f"The length of pred and target must be the same (got "
                    f"{len(pred)} and {len(target)}, respectively)."
                )

            return {fn.__name__: fn(target, pred) for fn in metrics}


    def infer_task_type(self):
        """
        Given the target column, find the number of unique values
        """
            
        if self.multi_task:
            muti_class_task_type = {}
            for i, target in enumerate(self.target_col):
                # find the num unique values in the target, geared for classification as of now
                targets = self.training_table[target]
                print(targets)
                
                # # Make sure '' is None for mapping to integer
                # targets.fillna(value = 'None', inplace = True)
                # targets.replace('None', None, inplace = True)



                # NEW (uses a copy; original df stays untouched)
                targets = self.training_table[target].copy()
                tmp = targets.fillna('None').replace('None', None)
                print(f'Value counts for inferring target type: {tmp.value_counts(dropna=True)}')
                num_classes = tmp.nunique(dropna=True)
                print(f'Found {num_classes} classes for {target} column')
                


                if num_classes ==2 and [target['task'] for target in self.dataset.metadata_semantics['targets']][i] == 'classification':
                    task_type = TaskType.BINARY_CLASSIFICATION
                elif num_classes > 2 and [target['task'] for target in self.dataset.metadata_semantics['targets']][i] == 'classification':
                    task_type = TaskType.MULTICLASS_CLASSIFICATION
                elif num_classes == 1 and [target['task'] for target in self.dataset.metadata_semantics['targets']][i] == 'classification':
                    task_type = TaskType.BINARY_CLASSIFICATION
                elif [target['task'] for target in self.dataset.metadata_semantics['targets']][i] == 'regression':
                    task_type = TaskType.REGRESSION
                
                
                
                # still set num_classes to 2 for single value targets for now
                num_classes = 2 if num_classes == 1 else num_classes
                
                if task_type == TaskType.REGRESSION:
                    num_classes = None
                
                
                muti_class_task_type[target] = {'task_type': task_type, 'num_classes': num_classes}
            
            self.task_type = muti_class_task_type

        else:
            # {table: list(columns)}
            table_to_col_mapping = {table_name: data.df.columns.tolist() for table_name, data in self.database.table_dict.items()}

            # find target table
            target_table = find_target_table(table_to_col_mapping, self.target_col)
            # There may be more than one table with our target column
            print(f'For this task, the target table is: {target_table}')
            
            # FIX ME: We may want to automatically find the entity table like above
            # However I see that targets may be present in multiple tables
            target_table = self.entity_table

            # find the num unique values in the target, geared for classification as of now
            targets = self.database.table_dict[target_table].df[self.target_col]
            print(f'Value counts for inferring target type: {targets.value_counts(dropna=False)}')
            num_classes = targets.nunique(dropna=False)
            # For now, output channels will still be 2 even if it is a single value col
            num_classes = 2 if num_classes == 1 else num_classes
            print(f'Found {num_classes} classes for {self.target_col} column in the target_table: {target_table}')
            print(f'Target distribution: {self.database.table_dict[target_table].df[self.target_col].value_counts(dropna=False)}')
            
            self.num_classes = num_classes
            if self.num_classes ==2:
                self.task_type = TaskType.BINARY_CLASSIFICATION
            elif self.num_classes > 2:
                self.task_type = TaskType.MULTICLASS_CLASSIFICATION
            elif self.num_classes == 1:
                self.task_type = TaskType.BINARY_CLASSIFICATION
                # TO DO: self.task_type = TaskType.REGRESSION
            else:
                raise ValueError("Invalid number of classes: num_classes should be 1 or greater.")

            print(f'Inferred task type is: {self.task_type}. Found {num_classes} classes for target: {self.target_col}')



    def _get_table(self, split: str, mask_input_cols:bool = None) -> 'Table':
        """
        Overwrite _get table function for time range splits, rather than timedelta
        Helper function to get a table for a split.
        """

        if mask_input_cols is None:
            mask_input_cols = split == "test"
                
        if split not in ['train', 'validation', 'test']:
            raise ValueError(f"Unknown split: {split}. Expected one of: 'train', 'validation', 'test'.")

        if split == 'train':
            start_date = self.database.train_timestamp
            # start of val is the same as end of train
            end_date = self.database.val_timestamp
        elif split == 'validation':
            start_date = self.database.val_timestamp
            # end of val is the same as start of test
            end_date = self.database.test_start_time
        elif split == 'test':
            start_date = self.database.test_start_time
            end_date = self.database.test_end_time

        print(f'For split: {split} extracting datespan: {start_date} --> {end_date}')
        
        table = self.make_table(self.database, start_date, end_date, split)
        table = self.filter_dangling_entities(table)

        # Prevent leakage
        if mask_input_cols:
            table = self._mask_input_cols(table)
        return table

    # def filter_dangling_entities(self, table: Table) -> Table:

    #     """
    #     Overwrite so database is not loaded from scratch again, we init this class with database

    #     """
    #     db = self.database
    #     num_entities = len(db.table_dict[self.entity_table])
    #     filter_mask = table.df[self.entity_col] >= num_entities
    #     if filter_mask.any():
    #         table.df = table.df[~filter_mask]
    #     return table



    def filter_dangling_entities(self, table: Table) -> Table:
        """
        Overwrite so database is not loaded from scratch again, we init this class with database
        """
        db = self.database
        num_entities = len(db.table_dict[self.entity_table])
        
        # --- NEW FIX: Handle list of entity combinations ---
        if isinstance(self.entity_col, list):
            # Create an all-False mask to start
            filter_mask = pd.Series(False, index=table.df.index)
            
            # Check every combination column for dangling IDs
            for col in self.entity_col:
                if col in table.df.columns:
                    filter_mask = filter_mask | (table.df[col] >= num_entities)
        else:
            # Original logic for single columns
            filter_mask = table.df[self.entity_col] >= num_entities
            
        if filter_mask.any():
            table.df = table.df[~filter_mask]
            
        return table



        
    
    def load_task_metadata(self):
        """
        load metadata related to task
        """
        task_metadata = load_metadata(catalog.get_metadata('cdh_soac_az')).semantics
        # possible targets for SOAC
        available_targets = [target.column for target in task_metadata.targets]
        # add in additional target names
        
        # if use_case =='SOAC':
        #   available_targets = available_targets + ['HEADERINCOTERMSCLASSIFICATION', 'ITEMINCOTERMSCLASSIFICATION']
        
        
        assert self.target_col in available_targets
        return {'task_metadata': task_metadata, 'available_targets': available_targets}
    


    def map_target_column(self, df, split):
            # Determine which target columns to process (multi-task vs single-task)
            target_cols = self.target_col if isinstance(self.target_col, list) and self.multi_task else [self.target_col]

            # # Helper to identify sentinel -100 (both numeric and string)
            # def _is_sentinel_100(x):
            #     return x == -100 or x == "-100"

            if split == 'train':
                self.class_mapping = {}
                self.int_to_class = {}

                for target_col in target_cols:

                    # --- Skip regression targets completely (unchanged behavior) ---
                    if self.multi_task and self.task_type[target_col]['task_type'] == TaskType.REGRESSION:
                        print(f"[map_target_column] Skipping label encoding for regression target: {target_col}")
                        continue


                    # --- Normalize empties just like your original code ---
                    #df[target_col] = df[target_col].fillna("None")
                    # df[target_col] = df[target_col].replace('', None)
                    # df[target_col] = df[target_col].replace(' ', None)
                    #df[target_col] = df[target_col].where(pd.notna(df[target_col]), None)

                    # --- Identify sentinel rows we must NOT encode ---
                    #sentinel_mask = df[target_col].apply(_is_sentinel_100)

                    # sentinel_mask = df[target_col].isna()


                    #df[target_col] = df[target_col].where(~pd.isna(df[target_col]), pd.NA)

                    #sentinel_mask = df[target_col].isna() | df[target_col].isin([-100, "-100"])
                    
                    
                    sentinel_mask = df[target_col].isin([-100, "-100"])
                    
                    
                    # --- Build mapping ONLY from non-sentinel values ---
                    unique_classes = df.loc[~sentinel_mask, target_col].unique()
                    class_to_int = {
                        cls: i for i, cls in enumerate(sorted(unique_classes, key=lambda x: (x is None, str(x))))
                    }

                    # Safe alias so both "None" (string) and None (actual None) share the same id
                    # if "None" in class_to_int:
                    #     class_to_int[None] = class_to_int["None"]
                    # elif None in class_to_int:
                    #     class_to_int["None"] = class_to_int[None]

                    print(f'Class mapping for classification in {target_col} (train, excluding NaN): {class_to_int}')

                    # --- Apply mapping only to non-sentinel rows; keep sentinel as -100 ---
                    df.loc[~sentinel_mask, target_col] = df.loc[~sentinel_mask, target_col].map(class_to_int)
                    

                    # --- Keep labels as integers while preserving missing as <NA> ---
                    # Convert any floats/objects produced by mapping to pandas' nullable Int64
                    try:
                        df[target_col] = pd.to_numeric(df[target_col], errors="coerce").astype("Int64")
                    except Exception:
                        # If something odd slips through, at least don't crash; leave as-is.
                        pass
                    
                    
                    # df.loc[sentinel_mask, target_col] = -100

                    # --- Store mappings ---
                    self.class_mapping[target_col] = class_to_int
                    self.int_to_class[target_col] = {v: k for k, v in class_to_int.items()}
                    
                    # Ensure the ignore marker exists in reverse mapping for consistency.
                    # This is useful when decoding or debugging rows labeled as -100.
                    self.int_to_class[target_col][-100] = "UNSEEN_CLASS"



            #Any unseen class (present in validation/test but not in train) is mapped to -100.

            elif split in ['validation', 'test']:
                if not hasattr(self, 'class_mapping'):
                    raise ValueError('Class mapping should have been initialized during the training split')

                for target_col in target_cols:

                    # --- Skip regression targets completely (unchanged behavior) ---
                    if self.multi_task and self.task_type[target_col]['task_type'] == TaskType.REGRESSION:
                        print(f"[map_target_column] Skipping label encoding for regression target: {target_col}")
                        continue

                    # Print the current target column being processed.
                    print(f"[{split}] Mapping target column: {target_col}")

                    # --- Normalize empties just like your original code ---
                    #df[target_col] = df[target_col].fillna("None")
                    # df[target_col] = df[target_col].replace('', None)
                    # df[target_col] = df[target_col].replace(' ', None)
                    #df[target_col] = df[target_col].where(pd.notna(df[target_col]), None)

                    # --- Identify sentinel rows to keep as -100 ---
                    # sentinel_mask = df[target_col].apply(_is_sentinel_100)

                    
                    df[target_col] = df[target_col].where(~pd.isna(df[target_col]), pd.NA)
                    
                    #sentinel_mask = df[target_col].isna() | df[target_col].isin([-100, "-100"])

                    sentinel_mask = df[target_col].isin([-100, "-100"])


                            # Collect unique values from the current split excluding sentinel rows.
                    # Drop NA because missing values are not label classes to encode.
                    current_split_unique_values = [
                        raw_class_value
                        for raw_class_value in df.loc[~sentinel_mask, target_col].dropna().unique()
                    ]

                    # Find values present in validation/test but absent in the train mapping.
                    unseen_class_values = [
                        raw_class_value
                        for raw_class_value in current_split_unique_values
                        if raw_class_value not in self.class_mapping[target_col]
                    ]

                    # If unseen classes are found, map each one to -100 in the forward mapping.
                    if len(unseen_class_values) > 0:
                        # Sort unseen values for deterministic debug print order.
                        sorted_unseen_class_values = sorted(unseen_class_values, key=lambda value: (value is None, str(value)))

                        # Loop through unseen values and assign them to -100 in the forward mapping.
                        for unseen_raw_class_value in sorted_unseen_class_values:
                            # Add unseen raw class to the mapping with the ignore label -100.
                            self.class_mapping[target_col][unseen_raw_class_value] = -100

                        # Ensure reverse mapping includes -100 marker.
                        # Multiple unseen classes can map to -100, so reverse mapping should use a marker string.
                        self.int_to_class[target_col][-100] = "__UNSEEN_OR_IGNORED__"

                        # Print which unseen values were found and marked as -100.
                        print(
                            f"[{split}] Unseen classes for {target_col} mapped to -100: "
                            f"{sorted_unseen_class_values}"
                        )

                    # Apply mapping to non-sentinel rows using the now-extended train mapping.
                    # Unseen classes will map to -100 because we inserted them above.
                    df.loc[~sentinel_mask, target_col] = (
                        df.loc[~sentinel_mask, target_col].map(self.class_mapping[target_col])
                    )

                    # Convert the encoded column to pandas nullable integer type.
                    # This keeps integer labels and allows NA if any unexpected values sli
                     
                     
                     
                     
                     
                     # --- Keep labels as integers while preserving missing as <NA> ---
                    # Convert any floats/objects produced by mapping to pandas' nullable Int64
                    try:
                        df[target_col] = pd.to_numeric(df[target_col], errors="coerce").astype("Int64")
                    except Exception:
                        # If something odd slips through, at least don't crash; leave as-is.
                        pass
                    
                    
                    
                    # --- Unknowns (non-sentinel) become NaN; fill them to -100. Sentinels remain -100. ---
                    # na_class = -100
                    # df[target_col] = df[target_col].fillna(na_class) #the non mapped missing class turn NaN after it does not gets mapped

            else:
                raise ValueError(f"Unknown split: {split}. Must be one of ['train', 'validation', 'test'].")

            return df

    
    

    def make_table(self, db: Database, start_date, end_date, split: str) -> Table:

        """
        https://relbench.stanford.edu/paper.pdf
        This is a task table. It contains:
        "The training table holds the ground truth labels, optional
        timestamps, and keys indicating which (combination of) entities each label is associated with."

        Examples:
        1. Custom task: https://github.com/snap-stanford/relbench/blob/main/tutorials/custom_task.ipynb
        2. Relbench's tasks: https://github.com/snap-stanford/relbench/tree/c4c6628469ee38ab9f7ba91e777b27fb81a79fb8/relbench/tasks

        You can extract each dataframe like:

        # Extract necessary DataFrames
        salesdocuments = db.table_dict["df_salesdocument"].df
        salesdocumentitems = db.table_dict["df_salesdocumentitem"].df
        customers = db.table_dict["df_customer"].df
        addresses = db.table_dict["df_address"].df

        """
        assert split in ['train', 'validation', 'test'], 'Split is not in [train, validation, test]'


        if self.multi_task:
            # load training table
            df = self.training_table
            print('Training table nulls:')
            print(df.isnull().sum())
        
        else:
            
            table_dict = {table_name: table.df for table_name, table in db.table_dict.items()}

            print(table_dict.keys())
            
            # Filter in entity col (pkey), target, and time
            df = table_dict[self.entity_table][[self.entity_col, self.target_col, self.time_col]]

        # # Make sure target data type is correct
        # Ensure regression targets are truly numeric so missing values show as NaN
        if isinstance(self.target_col, list):
            for target in self.target_col:
                if self.multi_task and self.task_type[target]['task_type'] == TaskType.REGRESSION:
                    df[target] = pd.to_numeric(df[target], errors='coerce')
        else:
            if self.task_type == TaskType.REGRESSION:
                df[self.target_col] = pd.to_numeric(df[self.target_col], errors='coerce')
        

        
        # --- Make sure '' is treated as None ONLY for classification targets ---
        if isinstance(self.target_col, list):
            
            for target in self.target_col:
                
                # If this is a regression target, do nothing to its NaNs
                if self.multi_task and self.task_type[target]['task_type'] == TaskType.REGRESSION:
                    continue

                # Classification target: normalize empties the usual way
                #df[target] = df[target].fillna("None")
                # df[target] = df[target].replace('', None)
                # df[target] = df[target].replace(' ', None)
                #df[target] = df[target].where(pd.notna(df[target]), None)
        # else:
        #     # Single-task: normalize only if the single target is NOT regression
        #     if self.task_type != TaskType.REGRESSION:
        #         #df[self.target_col] = df[self.target_col].fillna("None")
        #         # df[self.target_col] = df[self.target_col].replace('', "empty")
        #         # df[self.target_col] = df[self.target_col].replace(' ', None)
        #         #df[self.target_col] = df[self.target_col].where(pd.notna(df[self.target_col]), None)




        # --- NEW LOGIC: Determine which column to split on based purely on presence ---
        # If the dataframe has the '__row_order__' column, we MUST use it for splitting.
        is_row_split = '__row_order__' in df.columns

        if is_row_split:
            split_col = '__row_order__'
            print(f"[SPLIT] Using '{split_col}' for slicing. Time column '{self.time_col}' remains untouched.")
        else:
            split_col = self.time_col
            print(f"[SPLIT] Using conventional time column '{split_col}' for slicing.")

        # --- DYNAMIC TYPE ALIGNMENT ---
        # Whatever column we selected, we must force its data type to match our parsed boundary.
        if isinstance(start_date, (int, float)):
            # Boundary is numeric (e.g., 1319), so cast the column to numeric.
            df[split_col] = pd.to_numeric(df[split_col], errors='coerce')
        elif isinstance(start_date, pd.Timestamp):
            # Boundary is a Datetime (e.g., 2024-01-01), so cast the column to datetime.
            df[split_col] = pd.to_datetime(df[split_col], errors='coerce')
        else:
            # Boundary is an Alphanumeric String (e.g., "Batch_A"), so cast the column to string.
            df[split_col] = df[split_col].astype(str)

        # --- APPLY THE SLICE ---
        # Slicing logic remains mathematically identical, just using our dynamically chosen column!
        if split != 'test':
            # Train/Val: Include start bound, strictly less than end bound
            filtered_df = df[(df[split_col] >= start_date) & (df[split_col] < end_date)]
        else:
            # Test: Include both start and end bounds
            filtered_df = df[(df[split_col] >= start_date) & (df[split_col] <= end_date)]


        
        
        if self.multi_task:
            filtered_df = self.map_target_column(filtered_df, split)
        else:
            if self.task_type != TaskType.REGRESSION:
                filtered_df = self.map_target_column(filtered_df, split)

        # Print the time span for the table that was extracted for debugging
        print(f'Task Table for split {split}:\n{filtered_df.head()}')

        
        
        
        # --- NEW FIX: Safely handle list of entity columns ---
        if isinstance(self.entity_col, list):
            fkey_mapping = {col: self.entity_table for col in self.entity_col if col in filtered_df.columns}
        else:
            fkey_mapping = {self.entity_col: self.entity_table}



        return Table(
            df=filtered_df,
            fkey_col_to_pkey_table=fkey_mapping,
            pkey_col=None,
            time_col=self.time_col
        )


# In[225]:


soac_task = SOAC_Task(SOAC_db, entity_table = entity_table, use_case = use_case, full_train_table = full_train_table, multi_task = True)


# In[226]:


task_type=soac_task.task_type
task_type


# In[227]:


train_table = soac_task._get_table('train')
val_table = soac_task._get_table('validation')
test_table_with_targets = soac_task._get_table("test", mask_input_cols=False)
test_table = soac_task._get_table('test')



# **Remove _row_order_ from all the tables**

# In[228]:


# def remove_row_order_from_tables(*tables) -> tuple:
#     """
#     Takes any number of RelBench Table objects (e.g., train, val, test).
#     For each Table, checks if the '__row_order__' column exists in its underlying DataFrame.
#     If it exists, the column is dropped in-place.
#     Returns the cleaned Table objects in the exact same order they were passed.
#     """
    
#     # Iterate through all the Table objects passed into the function
#     for table in tables:
        
#         # Ensure the object passed actually has a '.df' attribute (a pandas DataFrame)
#         if hasattr(table, 'df') and isinstance(table.df, pd.DataFrame):
            
#             # Check if the '__row_order__' column exists in this specific DataFrame
#             if '__row_order__' in table.df.columns:
                
#                 # Drop the '__row_order__' column in-place. 
#                 # errors='ignore' acts as a safety net in case of a race condition.
#                 table.df.drop(columns=['__row_order__'], inplace=True, errors='ignore')
                
#                 # Optional: Print a confirmation message for visibility during execution
#                 print(f"[CLEANUP] Removed '__row_order__' from table with {len(table.df)} rows.")
#             else:
#                 # If the column isn't there, we just quietly move on
#                 pass
                
#     # Return the modified Table objects back to the caller.
#     # If a single table was passed, we return just the table (not a 1-item tuple) for convenience.
#     if len(tables) == 1:
#         return tables[0]
    
#     # If multiple tables were passed, return them as a tuple to unpack
#     return tables



# # Clean them all in one go
# train_table, val_table, test_table_with_targets, test_table = remove_row_order_from_tables(
#     train_table, 
#     val_table, 
#     test_table_with_targets, 
#     test_table
# )


# In[229]:


train_table


# In[230]:


for col in target_columns:
    print(f"Column: {col}")
    print(train_table.df[col].value_counts(dropna=False))
    print("="*40)


# In[231]:


train_table.df.isna().sum()


# In[232]:


train_table.df.dtypes


# In[233]:


val_table


# In[234]:


val_table.df.dtypes


# In[235]:


for col in target_columns:
    print(f"Column: {col}")
    print(val_table.df[col].value_counts(dropna=False))
    print("="*40)


# In[236]:


val_table.df.isna().sum()


# In[237]:


test_table


# In[238]:


test_table.df.dtypes


# In[239]:


test_table_with_targets


# In[240]:


for col in target_columns:
    print(f"Column: {col}")
    print(test_table_with_targets.df[col].value_counts(dropna=False))
    print("="*40)


# In[241]:


test_table.df.isna().sum()


# **updating the test_table by adding the idx column from the test_table_with_targets table**

# In[254]:


def restore_entity_idx_column(
    test_table, 
    test_table_with_targets, 
    entity_table_name: str
):
    """
    Derives the internal surrogate key column name from the entity table string 
    (e.g., 'df_qmel' -> '__idx_QMEL__') and copies that column from the 
    target-inclusive test table back into the standard test table.

    Parameters
    ----------
    test_table : Table
        The RelBench Table object for the test split (destination).
    test_table_with_targets : Table
        The RelBench Table object containing the unmasked test targets (source).
    entity_table_name : str
        The string variable name of the entity table (e.g., 'df_qmel' or 'df_bkpf').

    Returns
    -------
    Table
        The modified test_table object containing the restored index column.
    """
    
    # 1. Strip the 'df_' prefix (if present) and convert to uppercase to get the pure entity name
    # Example: 'df_qmel' -> 'QMEL'
    entity_name = entity_table_name.replace("df_", "").upper()
    
    # 2. Construct the exact internal index column string
    # Example: '__idx_QMEL__'
    idx_col = f"__idx_{entity_name}__"
    
    # 3. Safely verify the source DataFrame actually contains this column before copying
    if idx_col in test_table_with_targets.df.columns:
        
        # Copy the column directly over to the destination DataFrame's memory
        test_table.df[idx_col] = test_table_with_targets.df[idx_col]
        print(f"[SUCCESS] Restored '{idx_col}' to test_table.")
        
    else:
        # Log a warning if the column is missing rather than crashing the script
        print(f"[WARN] Source column '{idx_col}' not found in test_table_with_targets.df. Skipping copy.")
        
    # Return the updated Table object back to the caller
    return test_table



# In[255]:


test_table = restore_entity_idx_column(
    test_table=test_table,
    test_table_with_targets=test_table_with_targets,
    entity_table_name=entity_table  # Passing the string variable directly
)

test_table


# In[242]:


# #GMUL data area paths for data_init

# use_case = 'DCP_FAILURE_KNOWN_FECOD___FECOD'
# tenant_id= "740401466"
# data_area="gssl"


# # This path is only valid for the pre-computed "frozen" outputs. We will not access the real data-init outputs
# # using a hardcoded path like this in the future
# DATA_INIT_OUTPUT_BASEDIR = f'az://{data_area}/lgm_components/TaskDB/single_target_data/{use_case}_Value_Help/{tenant_id}'

# RAW_DATA_PATH = f'{DATA_INIT_OUTPUT_BASEDIR}/train_table.pkl'



# #"az://gmul/lgm_components/frozen_data_DO_NOT_DELETE/filtered/SaDP-PDELIVPROCGPTD/740343471/ISDSALESDOCITEM/740343471_ISDSALESDOCITEM_20250415-184814_0_1_10000.parquet"


# In[243]:


# data=load_pickle_input(RAW_DATA_PATH)

# data


# **Save the data artifacts only if train/test/val splits are non-empty**

# In[244]:


# def create_file_in_folder(folder: str, filename: str, obj_to_save):
#     """
#     Pickles *obj_to_save* under *filename* inside *folder* and
# returns the Path to the file that was written.
# """
#     local_file = Path(folder).joinpath(filename)
#     with open(local_file, "wb") as f:
#         pickle.dump(obj_to_save, f)
#     return local_file


# In[245]:


# test_path = f"az://{data_area}/lgm_components/TaskDB/{use_case}_Value_Help/{tenant_id}"
# da_test = DataArea.get_data_area(f"{data_area}_test") # where we can read/write files

# fs_gmul = RemotePathAndFilesytem.from_data_area(
#     path=test_path, data_area=data_area, instance_type=da_test.instance
# ) #creates the remote path 

# files_to_save = {
#     "rel_db.pkl": SOAC_db,        # whatever your object is called
#     "kg_col_to_stype_dict.pkl": kg_col_to_stype_dict,
#     "rel_task.pkl": soac_task,
#     "rel_task_type.pkl": soac_task.task_type,
#     "entity_table.pkl": entity_table,
#     "train_table.pkl": train_table,
#     "test_table.pkl": test_table,
#     "val_table.pkl": val_table,
#     "test_table_with_targets.pkl": test_table_with_targets,
#     "updated_clean_tables_dict.pkl": clean_dict,
# }

# remote_output_az_dir = f"az://{data_area}/lgm_components/TaskDB/{use_case}_Value_Help/{tenant_id}"

# with tempfile.TemporaryDirectory() as tmpdir:  #creates local temp directory
#     for fname, obj_to_save in files_to_save.items():
#         local_path = create_file_in_folder(tmpdir, fname, obj_to_save) #creates all teh files
#         fs_gmul.fs_write.put(str(local_path), remote_output_az_dir) #put all files to remote path


# In[246]:


# def save_data_artifacts() -> None:
 
 
 
 
#     def create_file_in_folder(folder: str, filename: str, obj_to_save):
#             """Pickles *obj_to_save* under *filename* inside *folder* and
#             returns the Path to the file that was written.
#             """    
#             local_file = Path(folder).joinpath(filename)
#             with open(local_file, "wb") as f:
#                 pickle.dump(obj_to_save, f)
#             return local_file



#     test_path = f"az://gssl/lgm_components/TaskDB/{use_case}_Value_Help/{tenant_id}"
    
#     da_test = DataArea.get_data_area("gssl_test") # where we can read/write files

#     fs_gmul = RemotePathAndFilesytem.from_data_area(
#         path=test_path, data_area="gssl", instance_type=da_test.instance
#     ) #creates the remote path 

#     files_to_save = {
#     "rel_db.pkl": SOAC_db,        # whatever your object is called
#     "kg_col_to_stype_dict.pkl": kg_col_to_stype_dict,
#     "rel_task.pkl": soac_task,
#     "rel_task_type.pkl": soac_task.task_type,
#     "entity_table.pkl": entity_table,
#     "train_table.pkl": train_table,
#     "test_table.pkl": test_table,
#     "val_table.pkl": val_table,
#     "test_table_with_targets.pkl": test_table_with_targets,
#     "usecase_schema": database_schema,
#     "usecase_column_metadata": use_case_column_metadata,
#     "updated_tenant_metadata":tenant_metadata,
#     }

#     remote_output_az_dir = f"az://gssl/lgm_components/TaskDB/{use_case}_Value_Help/{tenant_id}"

#     with tempfile.TemporaryDirectory() as tmpdir:  #creates local temp directory
#         for fname, obj_to_save in files_to_save.items():
#             local_path = create_file_in_folder(tmpdir, fname, obj_to_save) #creates all teh files
#             fs_gmul.fs_write.put(str(local_path), remote_output_az_dir) #put all files to remote path


# In[247]:


def save_data_artifacts() -> None:
 
 
 
 
    def create_file_in_folder(folder: str, filename: str, obj_to_save):
            """Pickles *obj_to_save* under *filename* inside *folder* and
            returns the Path to the file that was written.
            """    
            local_file = Path(folder).joinpath(filename)
            with open(local_file, "wb") as f:
                pickle.dump(obj_to_save, f)
            return local_file



    test_path = f"az://{data_area}/lgm_components/TaskDB/single_target_data/{use_case}_Value_Help/{tenant_id}"
    
    da_test = DataArea.get_data_area(f"{data_area}_test") # where we can read/write files

    fs_gmul = RemotePathAndFilesytem.from_data_area(
        path=test_path, data_area=data_area, instance_type=da_test.instance
    ) #creates the remote path 

    files_to_save = {
    "rel_db.pkl": SOAC_db,        # whatever your object is called
    "kg_col_to_stype_dict.pkl": kg_col_to_stype_dict,
    "rel_task.pkl": soac_task,
    "rel_task_type.pkl": soac_task.task_type,
    "entity_table.pkl": entity_table,
    "train_table.pkl": train_table,
    "test_table.pkl": test_table,
    "val_table.pkl": val_table,
    "test_table_with_targets.pkl": test_table_with_targets,
    "usecase_schema": database_schema,
    "usecase_column_metadata": use_case_column_metadata,
    "updated_tenant_metadata":tenant_metadata,
    }

    remote_output_az_dir = f"az://{data_area}/lgm_components/TaskDB/single_target_data/{use_case}_Value_Help/{tenant_id}"
    

    with tempfile.TemporaryDirectory() as tmpdir:  #creates local temp directory
        for fname, obj_to_save in files_to_save.items():
            local_path = create_file_in_folder(tmpdir, fname, obj_to_save) #creates all teh files
            fs_gmul.fs_write.put(str(local_path), remote_output_az_dir) #put all files to remote path


# In[248]:


# import os
# import pickle

# def save_data_artifacts() -> None:
    
#     # 1. Define the destination path and resolve it to the local mount
#     remote_output_az_dir = f"az://{data_area}/lgm_components/TaskDB/{use_case}_Value_Help/{tenant_id}"
#     local_output_dir = resolve_url(remote_output_az_dir)

#     # 2. Ensure the destination directory actually exists before writing
#     os.makedirs(local_output_dir, exist_ok=True)

#     # 3. Define the objects to save
#     files_to_save = {
#         "rel_db.pkl": SOAC_db,        
#         "kg_col_to_stype_dict.pkl": kg_col_to_stype_dict,
#         "rel_task.pkl": soac_task,
#         "rel_task_type.pkl": soac_task.task_type,
#         "entity_table.pkl": entity_table,
#         "train_table.pkl": train_table,
#         "test_table.pkl": test_table,
#         "val_table.pkl": val_table,
#         "test_table_with_targets.pkl": test_table_with_targets,
#         "usecase_schema": database_schema,
#         "usecase_column_metadata": use_case_column_metadata,
#         "updated_tenant_metadata": tenant_metadata,
#     }

#     # 4. Save the files directly into the resolved directory
#     for fname, obj_to_save in files_to_save.items():
#         file_path = os.path.join(local_output_dir, fname)
#         with open(file_path, "wb") as f:
#             pickle.dump(obj_to_save, f)
            
#     print(f"  - Successfully wrote {len(files_to_save)} files to {local_output_dir}")


# In[249]:


def _assert_table_has_dataframe(table_obj, table_name: str) -> pd.DataFrame:
    """Validate that `table_obj` exposes a pandas DataFrame at `.df`, print basic stats, and return it."""
    
    # Grab the underlying dataframe
    df = table_obj.df
    # Verify it is a pandas DataFrame
    
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"[ERROR] `{table_name}.df` is not a pandas DataFrame. Found: {type(df)}")
    # Print row/column counts for visibility
    print(f"[CHECK] {table_name}: rows={len(df)}, cols={len(df.columns)}")
    # Hand back the DataFrame for further checks
    return df





def save_artifacts_if_all_splits_nonempty(
    train_table,
    val_table,
    test_table,
    tenant_id,
   ) -> None:
    """
    Save artifacts only if train/val/test DataFrames are all non-empty.
    Prints a report and raises a ValueError if any split is empty.

    Args:
        train_table: object exposing `.df` (pandas DataFrame)
        val_table:   object exposing `.df` (pandas DataFrame)
        test_table:  object exposing `.df` (pandas DataFrame)
        files_to_save: dict[str, Any]  -> filename -> python object to pickle
        remote_output_az_dir: az://... destination directory
        fs_gmul: RemotePathAndFilesytem handle (already constructed)
        create_file_in_folder_fn: function(folder, filename, obj) -> Path (your helper)
    """
    print("\n[STEP] Validating split DataFrames before saving...")

    # Validate and fetch the DataFrames for each split, with prints
    df_train = _assert_table_has_dataframe(train_table, "train_table")
    df_val   = _assert_table_has_dataframe(val_table,   "val_table")
    df_test  = _assert_table_has_dataframe(test_table,  "test_table")

    # Track any empties for a single consolidated error
    empty_splits = []
    if len(df_train) == 0:
        empty_splits.append("train_table")
    if len(df_val) == 0:
        empty_splits.append("val_table")
    if len(df_test) == 0:
        empty_splits.append("test_table")

    # If any split is empty, print a clear report and fail fast
    if empty_splits:
        print(f"\n[FAIL] One or more splits are empty for the tenant {tenant_id}. Skipping save.")
        for split in empty_splits:
            print(f"  - {split} is EMPTY")
        raise ValueError(
            f"Cannot save artifacts for the tenant {tenant_id} because these splits are empty: {', '.join(empty_splits)}"
        )

    # Otherwise, proceed to save
    print(f"\n[OK] All splits are non-empty for the tenant {tenant_id}. Proceeding to save artifacts.")
    



   #save the data artifacts now

    save_data_artifacts()




    print(f"\n[SUCCESS] All artifacts saved successfully for the tenant {tenant_id}.")


# In[250]:


save_artifacts_if_all_splits_nonempty(
    train_table=train_table,
    val_table=val_table,
    test_table=test_table,
    tenant_id = tenant_id,
)

