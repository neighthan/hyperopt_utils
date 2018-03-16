import os
from subprocess import Popen
import inspect
from jupyter_utils.jupyter_utils import get_nb_imports, get_notebook_name
from tensorflow.python.keras import layers as k_layers
from tensorflow.python.keras.models import Sequential
from hyperopt import hp
from random import random
from typing import List, Callable, Optional, Dict, Any


def flatten_dict(d: dict) -> dict:
    new_dict = d.copy()
    for val in d.values():
        if type(val) == dict:
            val = flatten_dict(val)
            new_dict.update(val)
    return new_dict


def convert_ints(d: dict):
    for key, val in d.items():
        try:
            if int(val) == val:
                d[key] = int(val)
        except (ValueError, TypeError):
            if type(val) in [list, tuple]:
                for e in val:
                    if type(e) == dict:
                        convert_ints(e)
    return d


def write_funcs_to_file(fname: str, funcs: List[Callable], local_vars: Optional[Dict[str, Any]]) -> None:
    """
    Write the source for `funcs` in a file at `fname`, including imports.

    A best-effort attempt is made at including any imports needed for your functions to run; there
    are several caveats to this (see `get_nb_imports` for more information).

    :param fname: path to the file to write the functions in
    :param funcs: a list of functions whose source code should be written in the file at fname
    :param local_vars: just set local_vars=locals() if using this.
      If provided, a check is done for each local function (unless it's imported)
      to determine if one of the functions in `funcs` calls it; if so, an AssertionError is raised.
      You can add these functions to `funcs` or set `local_vars` to `None` to ignore this.
      Note that this is a best-effort check and may not catch all cases or may raise errors when
      there are no problems.
    """

    imports_needed = set()
    source = []
    imports = get_nb_imports(get_notebook_name())

    for func in funcs:
        source_lines = inspect.getsourcelines(func)[0]

        for import_names in imports:
            for import_name in import_names:
                for line in source_lines:
                    if import_name not in line:
                        continue

                    # we have to be a little careful here; some short import
                    # names like np or pd may show up by chance in another word,
                    # so we don't just want to check that the name occurs _anywhere_
                    # these checks should help, though there will still be
                    # errors possible

                    function_call = f"{import_name}("
                    module_use = f"{import_name}."
                    if function_call in line or module_use in line or len(import_name) > 3:
                        imports_needed.add(imports[import_names] + '\n')

        source.extend(source_lines)
        source.extend(['\n'] * 2)

    if local_vars:
        local_funcs_needed = set()
        local_funcs = {name: var for name, var in local_vars.items() if type(var) == type(lambda x: x)}

        for local_func in local_funcs:

            func_imported = False

            for import_names in imports:
                if local_func in import_names:
                    func_imported = True

            if func_imported or local_func in [func.__name__ for func in funcs]:
                continue

            for line in source:
                if f"{local_func}(" in line:
                    local_funcs_needed.add(local_func)

        if local_funcs_needed:
            assert False, f"Add the following local functions to `funcs` or set `local_vars` to `None`: {local_funcs_needed}"

    with open(fname, 'w') as f:
        f.writelines(sorted(imports_needed))
        f.write('\n')
        f.writelines(source)


def layer_from_base_space(base_space: dict, i: int) -> dict:
    layer = {}
    for key, val in base_space.items():
        if type(val) == tuple and type(val[0]) == type(lambda x: x):
            layer[key + f"_{i}"] = val[0](key + f"_{i}", *val[1:])
        else:
            layer[key] = val
    return layer


def replicate_layer(base_space: dict, n_layers: int, start_i: int = 0) -> dict:
    i = start_i + n_layers - 1
    layer_ip1 = layer_from_base_space(base_space, i)
    i -= 1

    while i >= start_i:
        layer_i = layer_from_base_space(base_space, i)
        layer_i[f'add_layer_{i + 1}'] = hp.choice(f'add_layer_{i + 1}', [False, layer_ip1])

        layer_ip1 = layer_i
        i -= 1
    return layer_i


def add_nested_layers(layer, flat_layers):
    flat_layers.append(layer)
    for key in layer:
        if key.startswith('add_layer') and layer[key]:
            add_nested_layers(layer[key], flat_layers)


def unpack_layers(space_sample):
    flat_layers = []
    for layer in space_sample['layers']:
        if type(layer) == tuple:
            flat_layers.extend(layer)
        else:
            flat_layers.append(layer)

    flatter_layers = []
    for layer in flat_layers:
        add_nested_layers(layer, flatter_layers)

    for layer in flatter_layers:
        for key in layer:
            if key.startswith('add_layer'):
                del layer[key]
                break

    space_sample['layers'] = flatter_layers
    return space_sample


def space_from_layers_spec(layers_spec: list, start_i=1):
    i = [start_i] if type(start_i) == int else start_i
    space = []
    for layer_spec in layers_spec:
        if type(layer_spec) == list:  # list of multiple options; only choose one
            spaces = [space_from_layers_spec(layer_spec[j] if type(layer_spec[j]) == list else [layer_spec[j]], i)['layers']
                      for j in range(len(layer_spec))]
            space.append(hp.choice(f'choice_{i[0]}', spaces))
            i[0] += 1
        else:
            min_count, max_count = layer_spec[:2]

            assert 0 <= min_count <= max_count

            base_space = {'layer_type': layer_spec[2]}
            base_space.update(layer_spec[3])

            while min_count > 0:
                space.append(layer_from_base_space(base_space, i[0]))
                i[0] += 1
                min_count -= 1
                max_count -= 1

            if min_count < max_count:
                space.append(replicate_layer(base_space, max_count - min_count, i[0]))
                i[0] += max_count - min_count
    return {'layers': space}


def model_from_space_sample(space_sample: dict):
    """

    Only works for Sequential models right now, though it shouldn't be
    hard to support more general models using some of the code/ideas from
    tf_layers.layers and the functional Keras API.
    """

    model = Sequential()
    # need either an input layer or input shape...
    for i, layer in enumerate(space_sample['layers']):
        layer = layer.copy()
        layer_type = layer.pop('layer_type')
        layer_func = getattr(k_layers, layer_type)
        layer_params = {}
        for key, val in layer.items():
            try:
                int(key[key.rindex('_') + 1:])
                layer_params[key[:key.rindex('_')]] = val
            except ValueError:  # either no _ or not followed by a number
                layer_params[key] = val

        # if using multiple RNN layers, make sure to return 3D input
        rnns = ['GRU', 'LSTM', 'RNN', 'SimpleRNN', 'ConvLSTM2D', 'CuDNNGRU', 'CuDNNLSTM']
        if layer_type in rnns and space_sample['layers'][i + 1]['layer_type'] in rnns:
            layer_params['return_sequences'] = True

        model.add(layer_func(**layer_params))

    model.compile(space_sample.get('optimizer', 'adam'), space_sample.get('loss_func', 'mse'))
    return model


def hyper_optimize(funcs, exp_key: str, max_evals: int, n_workers: int=1,
                   local_vars=None, space_args='', keep_files: bool=False):
    tmp_dir = '/cluster/nhunt/.tmp/'

    dir_path = f"{tmp_dir}{round(random() * 1e10)}/"

    os.mkdir(dir_path)

    write_funcs_to_file(f'{dir_path}/hyper_module.py', funcs, local_vars)

    with open(os.path.dirname(os.path.abspath(__file__)) + '/hyper_runner_template.py') as f:
        hyper_runner_source = f.read()

    templates = [
        ('model_fn', funcs[0].__name__),
        ('space_fn', funcs[1].__name__),
        ('exp_key', f"'{exp_key}'"),
        ('max_evals', max_evals),
        ('space_args', space_args)
    ]

    for template, val in templates:
        hyper_runner_source = hyper_runner_source.replace(f"{{{{{template}}}}}", str(val))

    with open(f"{dir_path}/hyper_runner.py", 'w') as f:
        f.write(hyper_runner_source)

    env = os.environ.copy()

    env['PYTHONPATH'] += f":{dir_path}:/cluster/nhunt/github/antibody_design/methods/nhunt/scripts"

    with open('/cluster/nhunt/logs/hyper.log', 'w+') as f:
        # start all of the workers and then continue executing;
        # they'll exit automatically once they haven't found a job in a bit
        for _ in range(n_workers):
            Popen('hyperopt-mongo-worker --mongo=localhost:9876/hyperopt --poll-interval=10'.split(' '),
                  stderr=f, stdout=f, env=env)

        Popen(f'python {dir_path}/hyper_runner.py && rm -rf {dir_path}'.split(' '),
              stderr=f, stdout=f, env=env)

        if keep_files:
            Popen(f'cp {dir_path}/* {tmp_dir}/'.split(' '))
