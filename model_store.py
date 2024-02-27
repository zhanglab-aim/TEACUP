"""A SQLite3 database for storing and retrieval of NAS results from different tasks
"""

from typing import Optional
import sqlite3
import pandas as pd
import amber
import json
from model_space import deserilizer as get_model_space


class ModelStore:
    """Model store Class for storing model arch and performances
    SQL schema:
        performance: id text (primary key), arc text, data text, model_fp text   # meta
                     params_million float, flops_gb float,                       # model complexity
                     optimizer text, learning_rate float, batchsize int,         # optimizer
                     reward_func text, val_reward float,                         # valid eval
                     test_reward float                                           # test eval

        dataset: data_id, input_shape, output_shape, output_func, loss_func, reward_func
                     
        arch: model_space_id, arc

    TODO schema:
        training-free metrics, ntk, lrr, each is a table
    """
    def __init__(self, store_path: Optional[str]=None):
        if store_path is None:
            store_path = "./store.db"
        self.con = sqlite3.connect(store_path)
        self.cur = self.con.cursor()
        self.validate_database()

    def validate_database(self):
        if self.cur.execute("SELECT name FROM sqlite_master WHERE name='performance'").fetchone() is None:
            self.cur.execute("""
            CREATE TABLE performance (
            id TEXT PRIMARY KEY,
            arc TEXT,
            model_space TEXT,
            model_fp TEXT,
            data TEXT,
            reward_func TEXT,
            val_reward FLOAT,
            test_reward FLOAT,
            params_million FLOAT,
            flops_gb FLOAT,
            optimizer TEXT,
            learning_rate FLOAT,
            batchsize INT
            ); """)
        
        if self.cur.execute("SELECT name FROM sqlite_master WHERE name='dataset'").fetchone() is None:
            self.cur.execute("""
            CREATE TABLE dataset (
            id TEXT PRIMARY KEY,
            input_shape TEXT,
            output_shape TEXT,
            output_func TEXT,
            loss_func TEXT,
            reward_func TEXT
            );
            """)
        
        if self.cur.execute("SELECT name FROM sqlite_master WHERE name='arch'").fetchone() is None:
            self.cur.execute("""
            CREATE TABLE arch (
            model_space_id TEXT,
            arch TEXT
            );
            """)

    def insert_row(self, row: dict):
        self.cur.execute("INSERT INTO performance (id, arc, model_space, model_fp, data, reward_func, val_reward, test_reward, params_million, flops_gb, optimizer, learning_rate, batchsize) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", 
                         (row['id'], row['arc'], row['model_space'], row['model_fp'], row['data'], row['reward_func'], row['val_reward'], row['test_reward'], row['params_million'], row['flops_gb'], row['optimizer'], row['learning_rate'], row['batchsize']))
        self.con.commit()

    def delete_row(self, row_id: int):
        self.cur.execute("DELETE FROM performance WHERE id=?", (row_id,))
        self.con.commit()

    def to_pandas(self, table_name='performance'):
        df = pd.read_sql_query(f"SELECT * FROM {table_name}", self.con)
        return df

    def close(self):
        self.con.close()
    
    def update_dataset_table(self, df: pd.DataFrame):
        """
        Example
        --------
            %run model_store.py
            ms = ModelStore()
            df = pd.read_table("datasets.tsv")
            ms.update_dataset_table(df)
            data_info = ms.get_dataset_info("SmallDeepSea")
        """
        df.to_sql('dataset', self.con, if_exists='replace', index=False)
        self.con.commit()

    def get_dataset_info(self, data_id):
        res = self.cur.execute("SELECT * FROM dataset WHERE id=?", (data_id,)).fetchone()
        assert res is not None, ValueError(f"cannot find dataset with id '{data_id}'")
        header = ('id', 'input_shape', 'output_shape', 'output_func', 'loss_func', 'reward_func')
        res_dict = {k: v for k,v in zip(header, res)}
        res_dict['input_shape'] = eval(res_dict['input_shape']) if type(res_dict['input_shape']) is str else res_dict['input_shape']
        res_dict['output_shape'] = eval(res_dict['output_shape']) if type(res_dict['output_shape']) is str else res_dict['output_shape']
        return res_dict

    def populate_arch_table_by_id(self, model_space_id: str, num_arc: int, skip_connection: bool=False):
        model_space = get_model_space(model_space_id)
        # NASbench201 doesn't specify skip connection
        controller = amber.architect.GeneralController(model_space=model_space, with_skip_connection=skip_connection)
        arcs = set([])
        for _ in range(10000):
            arc = ''.join([str(x) for x in controller.get_action()[0]])
            arcs.add(arc)
            if len(arcs) == num_arc: break
        df = pd.DataFrame({'model_space_id': model_space_id, 'arch': list(arcs)})
        df.to_sql('arch', self.con, if_exists='replace', index=False)
        self.con.commit()
    
    def populate_arch_table_by_json(self, model_space_id: str, json_fp: str):
        """
        Example
        --------
           ms.populate_arch_table_by_json(model_space_id="Conv1D_9_3_64", json_fp="results_R3.Arc150.BS128.TASKS132.json")
        """
        with open(json_fp) as json_file:
            arcs = json.load(json_file)["arcs"]
            arcs = [''.join([str(x) for x in arc]) for arc in arcs]
        df = pd.DataFrame({'model_space_id': model_space_id, 'arch': list(arcs)})
        df.to_sql('arch', self.con, if_exists='replace', index=False)
        self.con.commit()
    
    def get_arch_by_model_space(self, model_space_id):
        """
        Example
        --------
            res = ms.get_arch_by_model_space("Conv1D_9_3_64")
            pd.DataFrame({'arc':res}).to_csv("Conv1D_9_3_64.archs.txt", index=False)
        """
        res = self.cur.execute("SELECT arch FROM arch WHERE model_space_id=?", (model_space_id,)).fetchall()
        res = [_[0] for _ in res]
        return res