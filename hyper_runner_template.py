from hyperopt import tpe, fmin
from hyperopt.mongoexp import MongoTrials
from hyper_module import {{model_fn}}, {{space_fn}}

trials = MongoTrials('mongo://localhost:9876/hyperopt/jobs', exp_key={{exp_key}})

best = fmin({{model_fn}}, {{space_fn}}({{space_args}}), tpe.suggest,
            max_evals={{max_evals}}, trials=trials)
