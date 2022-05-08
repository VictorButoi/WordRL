# claims imports
import wordrl as wdl

# random imports
import submitit
import yaml
import itertools
import copy
import os


def run_submitit_job_array(config_dicts, timeout=60*3, mem=16, num_gpus=0):
    jobs = []
    executor = submitit.AutoExecutor(folder=os.path.join(wdl.filepaths.FILE_PATHS["ROOT_PATH"],"bash/submitit"))
    executor.update_parameters(timeout_min=timeout, mem_gb=mem, gpus_per_node=num_gpus, slurm_partition="sablab", slurm_wckey="")
    for config in config_dicts:
        if config["training"]["algorithm"] == "a2c":
            job = executor.submit(wdl.a2c.a2c_train.train_func, config)
        elif config["training"]["algorithm"] == "dqn":
            job = executor.submit(wdl.dqn.dqn_train.train_func, config)
        else:
            raise ValueError("RL Algorithm not implemented yet!")
        jobs.append(job)
    return jobs


def return_empty_dict_copy(original_dict):
    new_dict = {}
    for key in original_dict.keys():
        new_dict[key] = original_dict[key]
        for sub_key in new_dict[key].keys():
            new_dict[key][sub_key] = 0
    return new_dict


def get_num_options(original_dict):
    num_options = 0
    for key in original_dict.keys():
        for sub_key in original_dict[key].keys():
            num_options = len(original_dict[key][sub_key])
    return num_options


def gen_training_name(run_dict, params):

    names = []
    for key in params.keys():
        if key != "experiment":
            subkeys = params[key].keys()
            for sk in subkeys:
                names.append(f"{key}-{sk}")
    print(run_dict)
    return_name = f'{run_dict["training"]["algorithm"]}'
    for field in names:
        poi = field.split("-")
        return_name += f"/{poi[1]}:{run_dict[poi[0]][poi[1]]}"
    return return_name


def create_gridsearch(params, default, merge_default=False, root_dir="/home/vib9/src/WordRL"):
    with open(f"{root_dir}/wordrl/configs/{default}.yaml", 'r') as stream:
        default = yaml.safe_load(stream)

    #new_dicts = [return_empty_dict_copy(default) for _ in range(get_num_options(params))]
    new_dicts = []
    first_dicts = True

    # go through all options you want to set
    for key in params.keys():
        # lower levels like num_feat, model_type, num levels, etc.
        for sub_key in params[key].keys():
            prepared_new_dicts = []
            for option in params[key][sub_key]:
                nd = {
                    key: {
                        sub_key: option
                    }
                }
                prepared_new_dicts.append(nd)
            if first_dicts:
                for nd in prepared_new_dicts:
                    new_dicts.append(nd)
            else:
                old_dicts = new_dicts
                merged_dicts = []
                for od in old_dicts:
                    for nd in prepared_new_dicts:
                        merged_dicts.append(merge_dicts(od, nd))
                new_dicts = merged_dicts
            first_dicts = False

    if merge_default:
        for n in range(len(new_dicts)):
            new_dicts[n] = merge_dicts(default, new_dicts[n])
            new_dicts[n]["experiment"]["name"] = gen_training_name(new_dicts[n], params)

    return new_dicts


def merge_dicts(od, nd):
    merged_dict = copy.deepcopy(od)
    original_dict = copy.deepcopy(nd)

    for key in list(original_dict.keys()):
        if key in merged_dict and isinstance(merged_dict[key], dict):
            for subkey in original_dict[key]:
                merged_dict[key][subkey] = original_dict[key][subkey]
        else:
            merged_dict[key] = original_dict[key]
    return merged_dict
