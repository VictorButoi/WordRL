# claims imports
import wordrl as wdl

# random imports
import submitit
import yaml
import itertools
import copy


def run_submitit_job_array(config_dicts, timeout=20, mem=64, num_gpus=1):
    jobs = []
    executor = submitit.AutoExecutor(
        folder="/home/vib9/src/UniverSandbox/bash/submitit")
    executor.update_parameters(timeout_min=timeout, mem_gb=mem,
                               gpus_per_node=num_gpus, slurm_partition="sablab", slurm_wckey="")
    for config in config_dicts:
        job = executor.submit(uvs.training_funcs.train_net, config)
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
        if key != "misc":
            subkeys = params[key].keys()
            for sk in subkeys:
                if key == "model" and sk == "type":
                    names.append(f"{key}-{sk}")

    return_name = f'model_type:{run_dict["model"]["type"]}'
    for field in names:
        poi = field.split("-")
        return_name += f"/{poi[1]}:{run_dict[poi[0]][poi[1]]}"
    return return_name


def create_gridsearch(params, merge_default=False, default=None):
    if not default:
        with open("/home/vib9/src/UniverSeg/universeg/torch/configs/DEFAULT.yaml", 'r') as stream:
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
            new_dicts[n]["training"]["name"] = gen_training_name(
                new_dicts[n], params)

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