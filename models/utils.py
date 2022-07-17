import logging
import numpy as np
import os

LOG_LEVEL=int(os.environ['LOG_LEVEL']) if 'LOG_LEVEL' in os.environ else logging.INFO
LOG_FORMAT="[%(filename)s:%(lineno)d %(levelname)s] %(message)s"
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger("RNNTLogger")

labels = [" ", "a", "b", "c", "d", "e", "f", "g", "h", "i", \
          "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", \
          "t", "u", "v", "w", "x", "y", "z", "'"]

def seq_to_sen(seq):
    sen = "".join([labels[idx] for idx in seq])
    return sen

def migrate_state_dict(model, split_fc1=False):
    state_dict = model["state_dict"] if "state_dict" in model else model
    migrated_state_dict = {}
    for key, value in state_dict.items():
        if key == "joint_net.0.weight" and split_fc1:
            migrated_state_dict["joint.linear1_trans.weight"] = value[:, : 1024]
            migrated_state_dict["joint.linear1_pred.weight"] = value[:, 1024 : ]
            continue
        key = key.replace("encoder.pre_rnn.lstm", "transcription.pre_rnn")
        key = key.replace("encoder.post_rnn.lstm", "transcription.post_rnn")
        key = key.replace("dec_rnn.lstm", "pred_rnn")
        key = key.replace("joint_net.0", "joint.linear1")
        key = key.replace("joint_net.3", "joint.linear2")
        migrated_state_dict[key] = value
    if "audio_preprocessor.featurizer.fb" in migrated_state_dict.keys():
        del migrated_state_dict["audio_preprocessor.featurizer.fb"]
    if "audio_preprocessor.featurizer.window" in migrated_state_dict.keys():
        del migrated_state_dict["audio_preprocessor.featurizer.window"]
    return migrated_state_dict

def jit_module(module):
    jmodule = torch.jit.script(module)
    fmodule = torch.jit._recursive.wrap_cpp_module(
        torch._C._freeze_module(jmodule._c))
    # fmodule = torch.jit.freeze(jmodule)
    torch._C._jit_pass_constant_propagation(fmodule.graph)
    return fmodule

def jit_model(model, model_path):
    model.transcription = jit_module(model.transcription)
    model.prediction = jit_module(model.prediction)
    model.joint = jit_module(model.joint)
    model = torch.jit.script(model)
    torch.jit.save(model, model_path)
    return model

def parse_calib(calib_path):
    if not os.path.exists(calib_path):
        return

    with open(calib_path, "rb") as calib_file:
        lines = calib_file.read().decode('ascii').splitlines()

    calib_dict = {}
    for line in lines:
        split = line.split(':')
        if len(split) != 2:
            continue
        tensor = split[0]
        calib_dict[tensor] = np.uint32(int(split[1], 16)).view(np.dtype('float32')).item()
    return calib_dict

def save_calib(calib_path, model):
    calib_dict = {}
    for name, layer in model.rnnt.named_children():
        for sub_name, sub_layer in layer.named_children():
            if isinstance(sub_layer, torch.nn.LSTM):
                for layer in range(sub_layer.num_layers):
                    calib_dict[f"{sub_name}_{layer}"] = sub_layer._input_quantizers[layer]._calibrator.get_amax()
            if hasattr(sub_layer, "_input_quantizer"):
                calib_dict[sub_name] = sub_layer._input_quantizer.scale.item()
    with open(calib_path, 'w') as calib_file:
        json.dump(calib_dict, calib_file, indent=4)
    return calib_dict

def init_scales(model, calib_path):
    calib_dict = parse_calib(calib_path)
    for layer in range(2):
        model.transcription.pre_rnn._input_quantizers[layer]._scale = 1 / calib_dict["input"]

    for layer in range(3):
        model.transcription.post_rnn._input_quantizers[layer]._scale = 1 / calib_dict["encoder_reshape"]
    
    # model.transcription.pre_rnn._input_quantizers[0]._scale = 1 / calib_dict["input"]
    # model.transcription.pre_rnn._input_quantizers[1]._scale = 127
    # model.transcription.post_rnn._input_quantizers[0]._scale = 127
    # model.transcription.post_rnn._input_quantizers[1]._scale = 127
    # model.transcription.post_rnn._input_quantizers[2]._scale = 127
    return model

