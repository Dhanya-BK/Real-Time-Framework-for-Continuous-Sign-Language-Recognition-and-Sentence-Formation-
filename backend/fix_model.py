import h5py

MODEL_PATH = "gesture_model.h5"

with h5py.File(MODEL_PATH, "r+") as f:
    model_config = f.attrs.get("model_config")
    if model_config is None:
        print("No model_config found — this .h5 may not store architecture inline.")
    else:
        config_str = model_config if isinstance(model_config, str) else model_config.decode("utf-8")
        config_str = config_str.replace('"batch_shape"', '"batch_input_shape"')
        f.attrs.modify("model_config", config_str)
        print("Patched batch_shape -> batch_input_shape successfully.")