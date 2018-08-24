from keras.models import load_model
import tensorflow as tf
from keras import backend as K
from data_process_before_train import learning_rate_type
from tensorflow.python.framework import graph_io

def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a prunned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    prunned so subgraphs that are not neccesary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    from tensorflow.python.framework.graph_util import convert_variables_to_constants
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = convert_variables_to_constants(session, input_graph_def,
                                                      output_names, freeze_var_names)
        return frozen_graph


"""----------------------------------配置路径-----------------------------------"""
# 载入训练好的模型
model_name = 'hongqiao'
load_data_type = 5
load_learning_rate = 0.00001
load_lr_m, load_lr_index = learning_rate_type(load_learning_rate)
load_batch_size = 1
load_epoch = 50

h5_model_path='saved_models/{}_dt{}_lr{}e{}_bs{}_ep{}.h5'.format(
            model_name, load_data_type, load_lr_m, load_lr_index,load_batch_size, load_epoch)
output_path='./saved_models'
pb_model_name='{}_dt{}_lr{}e{}_bs{}_ep{}.pb'.format(
            model_name, load_data_type, load_lr_m, load_lr_index,load_batch_size, load_epoch)


"""----------------------------------导入keras模型------------------------------"""
K.set_learning_phase(0)
net_model = load_model(h5_model_path)

print('input is :', net_model.input.name)
print ('output is:', net_model.output.name)

"""----------------------------------保存为.pb格式------------------------------"""
sess = K.get_session()
frozen_graph = freeze_session(K.get_session(), output_names=[net_model.output.op.name])
graph_io.write_graph(frozen_graph, output_path, pb_model_name, as_text=False)

