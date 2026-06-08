from joint_tune_worker import run_combination

result = run_combination({
    "combo_id": 0,
    "decay_lambda": 0.20,
    "regularization": 0.0010,
    "elite": 1.00,
    "caf": 1.10,
    "concacaf": 1.05,
    "afc": 0.95,
    "ofc": 0.90,
})

print(result)